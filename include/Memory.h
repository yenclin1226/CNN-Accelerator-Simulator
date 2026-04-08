#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <unordered_map>
#include <string>
#include <vector>

struct MemoryConfig {
    int dram_latency_cycles{50};
    std::size_t dram_bandwidth_bytes_per_cycle{32};

    int global_buffer_latency_cycles{8};
    std::size_t global_buffer_bandwidth_bytes_per_cycle{128};

    bool enable_local_buffer{true};
    int local_buffer_latency_cycles{2};
    std::size_t local_buffer_bandwidth_bytes_per_cycle{64};

    bool enable_activation_reuse{true};
    bool enable_weight_reuse{true};
    std::size_t local_buffer_capacity_entries{512};
    std::size_t global_buffer_capacity_entries{4096};
};

class MemoryLevel {
public:
    MemoryLevel(std::string name, int latency_cycles, std::size_t bandwidth_bytes_per_cycle);
    virtual ~MemoryLevel() = default;

    std::uint64_t request(std::size_t bytes, std::uint64_t current_cycle);
    void reset();

    std::uint64_t accessCount() const;
    std::uint64_t bytesMoved() const;

private:
    std::string name_;
    int latency_cycles_{0};
    std::size_t bandwidth_bytes_per_cycle_{1};
    std::uint64_t next_transfer_cycle_{0};
    std::uint64_t access_count_{0};
    std::uint64_t bytes_moved_{0};
};

class DRAM : public MemoryLevel {
public:
    DRAM(int latency_cycles, std::size_t bandwidth_bytes_per_cycle);
};

class Buffer : public MemoryLevel {
public:
    Buffer(const std::string& name, int latency_cycles, std::size_t bandwidth_bytes_per_cycle);
};

class PresenceCache {
public:
    explicit PresenceCache(std::size_t capacity_entries = 0);

    bool contains(std::uint64_t key) const;
    void insert(std::uint64_t key);
    void clear();

private:
    void touch(std::unordered_map<std::uint64_t, std::list<std::uint64_t>::iterator>::iterator it);

    std::size_t capacity_entries_{0};
    std::list<std::uint64_t> lru_;
    std::unordered_map<std::uint64_t, std::list<std::uint64_t>::iterator> table_;
};

class MemoryHierarchy {
public:
    MemoryHierarchy(const MemoryConfig& config, int num_groups);

    std::uint64_t requestActivation(int group_id, std::uint64_t key, std::uint64_t current_cycle);
    std::uint64_t requestWeight(int group_id, std::uint64_t key, std::uint64_t current_cycle);
    std::uint64_t requestOutputStore(int group_id, std::uint64_t current_cycle);

    void reset();

    std::uint64_t dramAccesses() const;
    std::uint64_t onChipBufferAccesses() const;
    std::uint64_t dramBytesMoved() const;
    std::uint64_t onChipBufferBytesMoved() const;

    std::uint64_t activationReuseHits() const;
    std::uint64_t activationReuseMisses() const;
    std::uint64_t weightReuseHits() const;
    std::uint64_t weightReuseMisses() const;
    std::uint64_t memoryRequestsAvoided() const;
    std::uint64_t bytesSavedDueToReuse() const;

private:
    std::uint64_t moveThroughHierarchy(std::size_t bytes, int group_id, std::uint64_t current_cycle);
    std::uint64_t requestWithReuse(std::size_t bytes,
                                   int group_id,
                                   std::uint64_t key,
                                   bool enable_reuse,
                                   PresenceCache& global_cache,
                                   std::vector<PresenceCache>& local_caches,
                                   std::uint64_t& hit_counter,
                                   std::uint64_t& miss_counter,
                                   std::uint64_t current_cycle);
    void validateGroupId(int group_id) const;

    DRAM dram_;
    Buffer global_buffer_;
    std::vector<Buffer> local_buffers_;
    bool local_enabled_{false};
    int num_groups_{0};

    bool activation_reuse_enabled_{false};
    bool weight_reuse_enabled_{false};

    PresenceCache global_activation_cache_;
    PresenceCache global_weight_cache_;
    std::vector<PresenceCache> local_activation_caches_;
    std::vector<PresenceCache> local_weight_caches_;

    std::uint64_t activation_reuse_hits_{0};
    std::uint64_t activation_reuse_misses_{0};
    std::uint64_t weight_reuse_hits_{0};
    std::uint64_t weight_reuse_misses_{0};
    std::uint64_t memory_requests_avoided_{0};
    std::uint64_t bytes_saved_due_to_reuse_{0};
};
