#include "Memory.h"

#include <algorithm>
#include <iterator>
#include <stdexcept>

namespace {

std::uint64_t ceilDiv(std::size_t numerator, std::size_t denominator) {
    return static_cast<std::uint64_t>((numerator + denominator - 1) / denominator);
}

}  // namespace

MemoryLevel::MemoryLevel(std::string name, int latency_cycles, std::size_t bandwidth_bytes_per_cycle)
    : name_(std::move(name)),
      latency_cycles_(latency_cycles),
      bandwidth_bytes_per_cycle_(bandwidth_bytes_per_cycle) {
    if (latency_cycles_ < 0) {
        throw std::invalid_argument("Memory latency cannot be negative.");
    }
    if (bandwidth_bytes_per_cycle_ == 0) {
        throw std::invalid_argument("Memory bandwidth must be positive.");
    }
}

std::uint64_t MemoryLevel::request(std::size_t bytes, std::uint64_t current_cycle) {
    if (bytes == 0) {
        return current_cycle;
    }

    const std::uint64_t transfer_cycles = ceilDiv(bytes, bandwidth_bytes_per_cycle_);
    const std::uint64_t start_cycle = std::max(current_cycle, next_transfer_cycle_);
    const std::uint64_t transfer_done_cycle = start_cycle + transfer_cycles;
    const std::uint64_t complete_cycle =
        transfer_done_cycle + static_cast<std::uint64_t>(latency_cycles_);

    next_transfer_cycle_ = transfer_done_cycle;
    ++access_count_;
    bytes_moved_ += bytes;

    return complete_cycle;
}

void MemoryLevel::reset() {
    next_transfer_cycle_ = 0;
    access_count_ = 0;
    bytes_moved_ = 0;
}

std::uint64_t MemoryLevel::accessCount() const {
    return access_count_;
}

std::uint64_t MemoryLevel::bytesMoved() const {
    return bytes_moved_;
}

DRAM::DRAM(int latency_cycles, std::size_t bandwidth_bytes_per_cycle)
    : MemoryLevel("DRAM", latency_cycles, bandwidth_bytes_per_cycle) {}

Buffer::Buffer(const std::string& name, int latency_cycles, std::size_t bandwidth_bytes_per_cycle)
    : MemoryLevel(name, latency_cycles, bandwidth_bytes_per_cycle) {}

PresenceCache::PresenceCache(std::size_t capacity_entries) : capacity_entries_(capacity_entries) {}

bool PresenceCache::contains(std::uint64_t key) const {
    return table_.find(key) != table_.end();
}

void PresenceCache::insert(std::uint64_t key) {
    if (capacity_entries_ == 0) {
        return;
    }

    auto it = table_.find(key);
    if (it != table_.end()) {
        touch(it);
        return;
    }

    lru_.push_front(key);
    table_[key] = lru_.begin();

    while (table_.size() > capacity_entries_) {
        const std::uint64_t victim = lru_.back();
        lru_.pop_back();
        table_.erase(victim);
    }
}

void PresenceCache::clear() {
    lru_.clear();
    table_.clear();
}

void PresenceCache::touch(
    std::unordered_map<std::uint64_t, std::list<std::uint64_t>::iterator>::iterator it) {
    const std::uint64_t key = it->first;
    lru_.erase(it->second);
    lru_.push_front(key);
    it->second = lru_.begin();
}

MemoryHierarchy::MemoryHierarchy(const MemoryConfig& config, int num_groups)
    : dram_(config.dram_latency_cycles, config.dram_bandwidth_bytes_per_cycle),
      global_buffer_(
          "GlobalBuffer", config.global_buffer_latency_cycles, config.global_buffer_bandwidth_bytes_per_cycle),
      local_enabled_(config.enable_local_buffer),
      num_groups_(num_groups),
      activation_reuse_enabled_(config.enable_activation_reuse),
      weight_reuse_enabled_(config.enable_weight_reuse),
      global_activation_cache_(config.global_buffer_capacity_entries),
      global_weight_cache_(config.global_buffer_capacity_entries) {
    if (num_groups_ <= 0) {
        throw std::invalid_argument("Memory hierarchy requires at least one group.");
    }

    if (local_enabled_) {
        local_buffers_.reserve(static_cast<std::size_t>(num_groups_));
        local_activation_caches_.reserve(static_cast<std::size_t>(num_groups_));
        local_weight_caches_.reserve(static_cast<std::size_t>(num_groups_));
        for (int group_id = 0; group_id < num_groups_; ++group_id) {
            local_buffers_.emplace_back("LocalBuffer[" + std::to_string(group_id) + "]",
                                        config.local_buffer_latency_cycles,
                                        config.local_buffer_bandwidth_bytes_per_cycle);
            local_activation_caches_.emplace_back(config.local_buffer_capacity_entries);
            local_weight_caches_.emplace_back(config.local_buffer_capacity_entries);
        }
    }
}

std::uint64_t MemoryHierarchy::requestActivation(int group_id,
                                                 std::uint64_t key,
                                                 std::uint64_t current_cycle) {
    return requestWithReuse(sizeof(std::int8_t),
                            group_id,
                            key,
                            activation_reuse_enabled_,
                            global_activation_cache_,
                            local_activation_caches_,
                            activation_reuse_hits_,
                            activation_reuse_misses_,
                            current_cycle);
}

std::uint64_t MemoryHierarchy::requestWeight(int group_id, std::uint64_t key, std::uint64_t current_cycle) {
    return requestWithReuse(sizeof(std::int8_t),
                            group_id,
                            key,
                            weight_reuse_enabled_,
                            global_weight_cache_,
                            local_weight_caches_,
                            weight_reuse_hits_,
                            weight_reuse_misses_,
                            current_cycle);
}

std::uint64_t MemoryHierarchy::requestOutputStore(int group_id, std::uint64_t current_cycle) {
    return moveThroughHierarchy(sizeof(std::int32_t), group_id, current_cycle);
}

void MemoryHierarchy::reset() {
    dram_.reset();
    global_buffer_.reset();
    for (Buffer& local : local_buffers_) {
        local.reset();
    }

    global_activation_cache_.clear();
    global_weight_cache_.clear();
    for (PresenceCache& cache : local_activation_caches_) {
        cache.clear();
    }
    for (PresenceCache& cache : local_weight_caches_) {
        cache.clear();
    }

    activation_reuse_hits_ = 0;
    activation_reuse_misses_ = 0;
    weight_reuse_hits_ = 0;
    weight_reuse_misses_ = 0;
    memory_requests_avoided_ = 0;
    bytes_saved_due_to_reuse_ = 0;
}

std::uint64_t MemoryHierarchy::dramAccesses() const {
    return dram_.accessCount();
}

std::uint64_t MemoryHierarchy::onChipBufferAccesses() const {
    std::uint64_t accesses = global_buffer_.accessCount();
    for (const Buffer& local : local_buffers_) {
        accesses += local.accessCount();
    }
    return accesses;
}

std::uint64_t MemoryHierarchy::dramBytesMoved() const {
    return dram_.bytesMoved();
}

std::uint64_t MemoryHierarchy::onChipBufferBytesMoved() const {
    std::uint64_t bytes = global_buffer_.bytesMoved();
    for (const Buffer& local : local_buffers_) {
        bytes += local.bytesMoved();
    }
    return bytes;
}

std::uint64_t MemoryHierarchy::activationReuseHits() const {
    return activation_reuse_hits_;
}

std::uint64_t MemoryHierarchy::activationReuseMisses() const {
    return activation_reuse_misses_;
}

std::uint64_t MemoryHierarchy::weightReuseHits() const {
    return weight_reuse_hits_;
}

std::uint64_t MemoryHierarchy::weightReuseMisses() const {
    return weight_reuse_misses_;
}

std::uint64_t MemoryHierarchy::memoryRequestsAvoided() const {
    return memory_requests_avoided_;
}

std::uint64_t MemoryHierarchy::bytesSavedDueToReuse() const {
    return bytes_saved_due_to_reuse_;
}

std::uint64_t MemoryHierarchy::moveThroughHierarchy(std::size_t bytes,
                                                    int group_id,
                                                    std::uint64_t current_cycle) {
    validateGroupId(group_id);

    std::uint64_t ready_cycle = dram_.request(bytes, current_cycle);
    ready_cycle = global_buffer_.request(bytes, ready_cycle);
    if (local_enabled_) {
        ready_cycle = local_buffers_[static_cast<std::size_t>(group_id)].request(bytes, ready_cycle);
    }
    return ready_cycle;
}

std::uint64_t MemoryHierarchy::requestWithReuse(std::size_t bytes,
                                                int group_id,
                                                std::uint64_t key,
                                                bool enable_reuse,
                                                PresenceCache& global_cache,
                                                std::vector<PresenceCache>& local_caches,
                                                std::uint64_t& hit_counter,
                                                std::uint64_t& miss_counter,
                                                std::uint64_t current_cycle) {
    validateGroupId(group_id);

    const std::size_t group_index = static_cast<std::size_t>(group_id);
    const bool local_cache_exists = local_enabled_ && !local_caches.empty();
    const bool local_hit =
        enable_reuse && local_cache_exists && local_caches[group_index].contains(key);
    const bool global_hit = enable_reuse && !local_hit && global_cache.contains(key);

    if (local_hit || global_hit) {
        ++hit_counter;
    } else {
        ++miss_counter;
    }

    if (!enable_reuse) {
        return moveThroughHierarchy(bytes, group_id, current_cycle);
    }

    if (local_hit) {
        memory_requests_avoided_ += 2;
        bytes_saved_due_to_reuse_ += static_cast<std::uint64_t>(bytes * 2);
        return local_buffers_[group_index].request(bytes, current_cycle);
    }

    if (global_hit) {
        ++memory_requests_avoided_;
        bytes_saved_due_to_reuse_ += static_cast<std::uint64_t>(bytes);
        std::uint64_t ready_cycle = global_buffer_.request(bytes, current_cycle);
        if (local_enabled_) {
            ready_cycle = local_buffers_[group_index].request(bytes, ready_cycle);
            local_caches[group_index].insert(key);
        }
        return ready_cycle;
    }

    std::uint64_t ready_cycle = dram_.request(bytes, current_cycle);
    ready_cycle = global_buffer_.request(bytes, ready_cycle);
    global_cache.insert(key);
    if (local_enabled_) {
        ready_cycle = local_buffers_[group_index].request(bytes, ready_cycle);
        local_caches[group_index].insert(key);
    }
    return ready_cycle;
}

void MemoryHierarchy::validateGroupId(int group_id) const {
    if (group_id < 0 || group_id >= num_groups_) {
        throw std::out_of_range("Memory hierarchy group id out of range.");
    }
}
