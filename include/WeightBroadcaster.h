#pragma once

#include "ExecutionMode.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

struct BroadcastDemand {
    std::uint64_t weight_key{0};
    int required_bit{-1};  // -1 means full weight-word demand.
};

struct BroadcastPayload {
    bool valid{false};
    std::uint64_t weight_key{0};
    int bit_index{-1};  // -1 means full weight-word payload.
    std::size_t fanout{0};
    bool schedule_driven{false};
    std::size_t schedule_index{0};
};

struct FixedBroadcastScheduleEntry {
    std::uint64_t weight_key{0};
    int bit_index{-1};
};

class WeightBroadcaster {
public:
    WeightBroadcaster(int weight_precision_bits, int phase_offset);

    void setFixedSchedule(std::vector<FixedBroadcastScheduleEntry> schedule);
    void clearFixedSchedule();

    void prepareCycle(std::uint64_t cycle,
                      const std::vector<BroadcastDemand>& demands,
                      ExecutionMode execution_mode,
                      BroadcastMode broadcast_mode);
    bool isDemandInPayload(const BroadcastDemand& demand) const;
    const BroadcastPayload& currentPayload() const;
    std::optional<std::uint64_t> currentPayloadWeightKey() const;

    int weightPrecisionBits() const;
    int phaseOffset() const;

private:
    struct DemandVote {
        std::uint64_t weight_key{0};
        int bit_index{-1};
        std::size_t count{0};
    };

    static std::vector<DemandVote> aggregateDemands(const std::vector<BroadcastDemand>& demands);
    std::size_t pickBestWordVote(const std::vector<DemandVote>& votes) const;
    std::size_t pickBestBitVote(const std::vector<DemandVote>& votes, int target_bit) const;
    void prepareDemandDrivenCycle(std::uint64_t cycle,
                                  const std::vector<BroadcastDemand>& demands,
                                  ExecutionMode execution_mode);
    void prepareFixedScheduleCycle(std::uint64_t cycle);

    int weight_precision_bits_{8};
    int phase_offset_{0};
    std::uint64_t prepared_cycle_{0};
    bool has_prepared_cycle_{false};
    BroadcastPayload current_payload_{};
    std::vector<FixedBroadcastScheduleEntry> fixed_schedule_;
};
