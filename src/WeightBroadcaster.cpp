#include "WeightBroadcaster.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <unordered_map>

namespace {

std::uint64_t composeVoteKey(std::uint64_t weight_key, int bit_index) {
    const std::uint64_t bit_part =
        static_cast<std::uint64_t>(static_cast<std::uint32_t>(bit_index + 1) & 0xFFU);
    return (weight_key << 8U) | bit_part;
}

int circularBitDistance(int bit_a, int bit_b, int modulo) {
    if (modulo <= 0) {
        return 0;
    }
    const int raw = std::abs(bit_a - bit_b);
    return std::min(raw, modulo - raw);
}

}  // namespace

WeightBroadcaster::WeightBroadcaster(int weight_precision_bits, int phase_offset)
    : weight_precision_bits_(weight_precision_bits), phase_offset_(phase_offset) {
    if (weight_precision_bits_ <= 0) {
        throw std::invalid_argument("Weight broadcaster precision must be positive.");
    }
}

void WeightBroadcaster::setFixedSchedule(std::vector<FixedBroadcastScheduleEntry> schedule) {
    fixed_schedule_ = std::move(schedule);
}

void WeightBroadcaster::clearFixedSchedule() {
    fixed_schedule_.clear();
}

void WeightBroadcaster::prepareCycle(std::uint64_t cycle,
                                     const std::vector<BroadcastDemand>& demands,
                                     ExecutionMode execution_mode,
                                     BroadcastMode broadcast_mode) {
    if (has_prepared_cycle_ && prepared_cycle_ == cycle) {
        return;
    }
    has_prepared_cycle_ = true;
    prepared_cycle_ = cycle;
    current_payload_ = BroadcastPayload{};

    if (broadcast_mode == BroadcastMode::SnapeaFixedSchedule) {
        prepareFixedScheduleCycle(cycle);
        return;
    }
    prepareDemandDrivenCycle(cycle, demands, execution_mode);
}

void WeightBroadcaster::prepareDemandDrivenCycle(std::uint64_t cycle,
                                                 const std::vector<BroadcastDemand>& demands,
                                                 ExecutionMode execution_mode) {
    current_payload_ = BroadcastPayload{};

    const std::vector<DemandVote> votes = aggregateDemands(demands);
    if (votes.empty()) {
        return;
    }

    if (execution_mode == ExecutionMode::Int8BitParallel) {
        const std::size_t best_index = pickBestWordVote(votes);
        current_payload_.valid = true;
        current_payload_.weight_key = votes[best_index].weight_key;
        current_payload_.bit_index = -1;
        current_payload_.fanout = votes[best_index].count;
        return;
    }

    const int target_bit = static_cast<int>(
        (cycle + static_cast<std::uint64_t>(phase_offset_)) %
        static_cast<std::uint64_t>(weight_precision_bits_));
    const std::size_t best_index = pickBestBitVote(votes, target_bit);
    current_payload_.valid = true;
    current_payload_.weight_key = votes[best_index].weight_key;
    current_payload_.bit_index = votes[best_index].bit_index;
    current_payload_.fanout = votes[best_index].count;
}

void WeightBroadcaster::prepareFixedScheduleCycle(std::uint64_t cycle) {
    if (fixed_schedule_.empty()) {
        return;
    }

    const std::size_t phase_offset =
        static_cast<std::size_t>(phase_offset_ % static_cast<int>(fixed_schedule_.size()));
    const std::size_t schedule_index =
        static_cast<std::size_t>((cycle + phase_offset) % fixed_schedule_.size());
    const FixedBroadcastScheduleEntry& entry = fixed_schedule_[schedule_index];

    current_payload_.valid = true;
    current_payload_.weight_key = entry.weight_key;
    current_payload_.bit_index = entry.bit_index;
    current_payload_.fanout = 0;
    current_payload_.schedule_driven = true;
    current_payload_.schedule_index = schedule_index;
}

bool WeightBroadcaster::isDemandInPayload(const BroadcastDemand& demand) const {
    if (!current_payload_.valid) {
        return false;
    }
    if (demand.weight_key != current_payload_.weight_key) {
        return false;
    }

    // Full-word payload satisfies both word and bit demands for the selected key.
    if (current_payload_.bit_index < 0) {
        return true;
    }
    return demand.required_bit == current_payload_.bit_index;
}

const BroadcastPayload& WeightBroadcaster::currentPayload() const {
    return current_payload_;
}

std::optional<std::uint64_t> WeightBroadcaster::currentPayloadWeightKey() const {
    if (!current_payload_.valid) {
        return std::nullopt;
    }
    return current_payload_.weight_key;
}

int WeightBroadcaster::weightPrecisionBits() const {
    return weight_precision_bits_;
}

int WeightBroadcaster::phaseOffset() const {
    return phase_offset_;
}

std::vector<WeightBroadcaster::DemandVote> WeightBroadcaster::aggregateDemands(
    const std::vector<BroadcastDemand>& demands) {
    std::unordered_map<std::uint64_t, DemandVote> vote_map;
    vote_map.reserve(demands.size());

    for (const BroadcastDemand& demand : demands) {
        const std::uint64_t key = composeVoteKey(demand.weight_key, demand.required_bit);
        auto it = vote_map.find(key);
        if (it == vote_map.end()) {
            DemandVote vote;
            vote.weight_key = demand.weight_key;
            vote.bit_index = demand.required_bit;
            vote.count = 1;
            vote_map.emplace(key, vote);
        } else {
            ++it->second.count;
        }
    }

    std::vector<DemandVote> votes;
    votes.reserve(vote_map.size());
    for (const auto& entry : vote_map) {
        votes.push_back(entry.second);
    }
    return votes;
}

std::size_t WeightBroadcaster::pickBestWordVote(const std::vector<DemandVote>& votes) const {
    std::size_t best_index = 0;
    for (std::size_t i = 1; i < votes.size(); ++i) {
        const DemandVote& cur = votes[i];
        const DemandVote& best = votes[best_index];
        if (cur.count != best.count) {
            if (cur.count > best.count) {
                best_index = i;
            }
            continue;
        }
        if (cur.weight_key < best.weight_key) {
            best_index = i;
        }
    }
    return best_index;
}

std::size_t WeightBroadcaster::pickBestBitVote(const std::vector<DemandVote>& votes, int target_bit) const {
    std::size_t best_index = 0;
    for (std::size_t i = 1; i < votes.size(); ++i) {
        const DemandVote& cur = votes[i];
        const DemandVote& best = votes[best_index];
        if (cur.count != best.count) {
            if (cur.count > best.count) {
                best_index = i;
            }
            continue;
        }

        const int cur_dist = circularBitDistance(cur.bit_index, target_bit, weight_precision_bits_);
        const int best_dist = circularBitDistance(best.bit_index, target_bit, weight_precision_bits_);
        if (cur_dist != best_dist) {
            if (cur_dist < best_dist) {
                best_index = i;
            }
            continue;
        }

        if (cur.weight_key < best.weight_key) {
            best_index = i;
        }
    }
    return best_index;
}
