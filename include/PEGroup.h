#pragma once

#include "Lane.h"
#include "WeightBroadcaster.h"

#include <vector>

class PEGroup {
public:
    PEGroup(int id, int num_lanes, int weight_precision_bits, int phase_offset);

    int id() const;
    WeightBroadcaster& broadcaster();
    const WeightBroadcaster& broadcaster() const;

    std::vector<Lane>& lanes();
    const std::vector<Lane>& lanes() const;

private:
    int id_{0};
    WeightBroadcaster broadcaster_;
    std::vector<Lane> lanes_;
};
