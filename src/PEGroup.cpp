#include "PEGroup.h"

#include <stdexcept>

PEGroup::PEGroup(int id, int num_lanes, int weight_precision_bits, int phase_offset)
    : id_(id), broadcaster_(weight_precision_bits, phase_offset) {
    if (id_ < 0) {
        throw std::invalid_argument("PEGroup id must be non-negative.");
    }
    if (num_lanes <= 0) {
        throw std::invalid_argument("PEGroup must contain at least one lane.");
    }

    lanes_.reserve(static_cast<std::size_t>(num_lanes));
    for (int lane_id = 0; lane_id < num_lanes; ++lane_id) {
        lanes_.emplace_back(lane_id);
    }
}

int PEGroup::id() const {
    return id_;
}

WeightBroadcaster& PEGroup::broadcaster() {
    return broadcaster_;
}

const WeightBroadcaster& PEGroup::broadcaster() const {
    return broadcaster_;
}

std::vector<Lane>& PEGroup::lanes() {
    return lanes_;
}

const std::vector<Lane>& PEGroup::lanes() const {
    return lanes_;
}
