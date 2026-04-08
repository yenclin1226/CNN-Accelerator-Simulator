#pragma once

#include "ExecutionMode.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

enum class TaskStatus {
    Queued,
    Issued,
    WaitingForData,
    Executing,
    Writeback,
    Completed
};

class ConvLayer;

struct MacOp {
    int cin{0};
    int ky{0};
    int kx{0};
    int input_y{0};
    int input_x{0};

    std::int8_t activation{0};
    std::int8_t weight{0};

    std::uint64_t activation_key{0};
    std::uint64_t weight_key{0};

    std::int32_t full_product{0};
    std::array<std::int32_t, 8> bit_contribution{};

    int abs_weight{0};
    int leading_one{0};
    std::int64_t abs_product{0};
};

class Task {
public:
    Task() = default;
    Task(int id, int output_channel, int output_y, int output_x);

    int id() const;
    int outputChannel() const;
    int outputY() const;
    int outputX() const;

    std::int32_t accumulator() const;
    std::int32_t preReluOutput() const;
    std::int32_t finalOutput() const;

    TaskStatus status() const;

    void setAccumulator(std::int32_t value);
    void setPreReluOutput(std::int32_t value);
    void setFinalOutput(std::int32_t value);
    void setStatus(TaskStatus status);

    void setEarlyTerminated(bool value);
    bool earlyTerminated() const;

    std::size_t processedMacs() const;
    std::size_t skippedMacs() const;
    std::size_t processedBitSteps() const;
    std::size_t skippedBitSteps() const;
    std::size_t totalMacs() const;
    std::size_t totalBitSteps() const;

    // Exact ET state in accumulator units. After each processed term, accumulator_ already
    // includes that term and this bound covers only future positive rescue work. Bias is
    // excluded because initializeContext folds it into accumulator_ before any MAC work.
    std::int64_t remainingPositiveContributionUpperBound() const;
    double predictedCost() const;
    int costBucketId() const;
    int phaseAffinityHint() const;
    std::uint64_t assignedCycle() const;

    void initializeSchedulingMetadata(const ConvLayer& layer, int weight_precision_bits);
    void setCostBucketId(int bucket_id);
    void setAssignedCycle(std::uint64_t cycle);
    void initializeContext(const ConvLayer& layer,
                           ExecutionMode execution_mode,
                           bool enable_importance_ordering,
                           MacOrderingPolicy mac_ordering_policy,
                           BroadcastMode broadcast_mode);
    const MacOp* currentOp() const;
    bool hasMoreOps() const;

    void markMacStarted();
    void markBitProcessed(std::int32_t signed_contribution);
    void markParallelMacProcessed(std::int32_t signed_product);
    void advanceToNextOp();
    void finalizeSkippedWorkOnEarlyTermination();

    static constexpr std::size_t activationBytes() {
        return sizeof(std::int8_t);
    }

    static constexpr std::size_t weightBytes() {
        return sizeof(std::int8_t);
    }

    static constexpr std::size_t outputBytes() {
        return sizeof(std::int32_t);
    }

private:
    static std::uint64_t packActivationKey(int cin, int y, int x);
    static std::uint64_t packWeightKey(int oc, int cin, int ky, int kx);
    static int leadingOnePosition(int value);
    void decrementRemainingPositiveContributionUpperBound(std::int32_t signed_delta);

    int id_{-1};
    int output_channel_{0};
    int output_y_{0};
    int output_x_{0};

    std::vector<MacOp> worklist_;
    std::size_t op_index_{0};

    std::int32_t accumulator_{0};
    std::int32_t pre_relu_output_{0};
    std::int32_t final_output_{0};

    bool early_terminated_{false};
    double predicted_cost_{0.0};
    int cost_bucket_id_{0};
    int phase_affinity_hint_{0};
    std::uint64_t assigned_cycle_{0};

    std::size_t processed_macs_{0};
    std::size_t processed_bit_steps_{0};

    // Exact ET state in accumulator units. With the after-processing invariant, this tracks
    // only future positive rescue work that could still make the pre-ReLU result positive.
    std::int64_t remaining_positive_contribution_upper_bound_{0};

    TaskStatus status_{TaskStatus::Queued};
};
