#include "Task.h"

#include "ConvLayer.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace {

bool importanceOrderLess(const MacOp& a, const MacOp& b) {
    if (a.weight != b.weight) {
        return a.weight > b.weight;
    }
    if (a.full_product != b.full_product) {
        return a.full_product > b.full_product;
    }
    if (a.leading_one != b.leading_one) {
        return a.leading_one > b.leading_one;
    }
    if (a.cin != b.cin) {
        return a.cin < b.cin;
    }
    if (a.ky != b.ky) {
        return a.ky < b.ky;
    }
    return a.kx < b.kx;
}

int etAwareBucket(const MacOp& op) {
    if (op.full_product < 0) {
        return 0;
    }
    if (op.full_product == 0) {
        return 1;
    }
    return 2;
}

bool etAwareNegativeFirstLess(const MacOp& a, const MacOp& b) {
    const int a_bucket = etAwareBucket(a);
    const int b_bucket = etAwareBucket(b);
    if (a_bucket != b_bucket) {
        return a_bucket < b_bucket;
    }

    if (a_bucket == 0) {
        if (a.full_product != b.full_product) {
            return a.full_product < b.full_product;
        }
        if (a.abs_product != b.abs_product) {
            return a.abs_product > b.abs_product;
        }
        if (a.leading_one != b.leading_one) {
            return a.leading_one > b.leading_one;
        }
        if (a.cin != b.cin) {
            return a.cin < b.cin;
        }
        if (a.ky != b.ky) {
            return a.ky < b.ky;
        }
        return a.kx < b.kx;
    }

    return importanceOrderLess(a, b);
}

}  // namespace

Task::Task(int id, int output_channel, int output_y, int output_x)
    : id_(id), output_channel_(output_channel), output_y_(output_y), output_x_(output_x) {
    if (id_ < 0) {
        throw std::invalid_argument("Task id must be non-negative.");
    }
}

int Task::id() const {
    return id_;
}

int Task::outputChannel() const {
    return output_channel_;
}

int Task::outputY() const {
    return output_y_;
}

int Task::outputX() const {
    return output_x_;
}

std::int32_t Task::accumulator() const {
    return accumulator_;
}

std::int32_t Task::preReluOutput() const {
    return pre_relu_output_;
}

std::int32_t Task::finalOutput() const {
    return final_output_;
}

TaskStatus Task::status() const {
    return status_;
}

void Task::setAccumulator(std::int32_t value) {
    accumulator_ = value;
}

void Task::setPreReluOutput(std::int32_t value) {
    pre_relu_output_ = value;
}

void Task::setFinalOutput(std::int32_t value) {
    final_output_ = value;
}

void Task::setStatus(TaskStatus status) {
    status_ = status;
}

void Task::setEarlyTerminated(bool value) {
    early_terminated_ = value;
}

bool Task::earlyTerminated() const {
    return early_terminated_;
}

std::size_t Task::processedMacs() const {
    return processed_macs_;
}

std::size_t Task::skippedMacs() const {
    return totalMacs() - processed_macs_;
}

std::size_t Task::processedBitSteps() const {
    return processed_bit_steps_;
}

std::size_t Task::skippedBitSteps() const {
    return totalBitSteps() - processed_bit_steps_;
}

std::size_t Task::totalMacs() const {
    return worklist_.size();
}

std::size_t Task::totalBitSteps() const {
    return worklist_.size() * 8U;
}

std::int64_t Task::remainingPositiveContributionUpperBound() const {
    return remaining_positive_contribution_upper_bound_;
}

double Task::predictedCost() const {
    return predicted_cost_;
}

int Task::costBucketId() const {
    return cost_bucket_id_;
}

int Task::phaseAffinityHint() const {
    return phase_affinity_hint_;
}

std::uint64_t Task::assignedCycle() const {
    return assigned_cycle_;
}

void Task::initializeSchedulingMetadata(const ConvLayer& layer, int weight_precision_bits) {
    constexpr double kSerialBitStepsPerMac = 8.0;
    constexpr double kMarginWeight = 0.45;
    constexpr double kNegativeRatioWeight = 0.25;
    constexpr double kZeroProductWeight = 0.20;
    constexpr double kPositiveBiasWeight = 0.20;
    constexpr double kPositiveTrendWeight = 0.10;
    constexpr double kExpectedFractionBase = 0.90;
    constexpr double kExpectedFractionEtWeight = 0.45;
    constexpr double kExpectedFractionZeroWeight = 0.20;
    constexpr double kExpectedFractionPositiveBiasWeight = 0.12;
    constexpr double kExpectedFractionPositiveTrendWeight = 0.08;
    constexpr double kMinExpectedProcessedFraction = 0.15;
    constexpr double kMagnitudeDensityScale = 128.0;
    constexpr double kDensityPenaltyScale = 0.03;
    constexpr double kMaxDensityPenaltyFraction = 0.20;

    const int kernel_size = layer.kernelSize();
    const int input_channels = layer.inputChannels();
    const std::size_t total_macs =
        static_cast<std::size_t>(input_channels * kernel_size * kernel_size);

    std::size_t zero_activation_count = 0;
    std::size_t zero_weight_count = 0;
    std::size_t zero_product_count = 0;
    std::size_t positive_product_count = 0;
    std::size_t negative_product_count = 0;
    std::int64_t abs_product_sum = 0;
    std::int64_t signed_product_sum = 0;

    for (int cin = 0; cin < input_channels; ++cin) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                const int input_y = output_y_ * layer.stride() + ky - layer.padding();
                const int input_x = output_x_ * layer.stride() + kx - layer.padding();
                const std::int8_t activation = layer.readActivation(cin, input_y, input_x);
                const std::int8_t weight = layer.readWeight(output_channel_, cin, ky, kx);
                const std::int32_t product =
                    static_cast<std::int32_t>(activation) * static_cast<std::int32_t>(weight);

                if (activation == 0) {
                    ++zero_activation_count;
                }
                if (weight == 0) {
                    ++zero_weight_count;
                }
                if (product == 0) {
                    ++zero_product_count;
                } else if (product > 0) {
                    ++positive_product_count;
                } else {
                    ++negative_product_count;
                }

                abs_product_sum += std::llabs(static_cast<long long>(product));
                signed_product_sum += static_cast<std::int64_t>(product);
            }
        }
    }

    const std::int32_t bias = layer.hasBias() ? layer.readBias(output_channel_) : 0;
    const double total_macs_d = std::max(1.0, static_cast<double>(total_macs));
    const double serial_base = total_macs_d * kSerialBitStepsPerMac;
    const double zero_activation_ratio = static_cast<double>(zero_activation_count) / total_macs_d;
    const double zero_weight_ratio = static_cast<double>(zero_weight_count) / total_macs_d;
    const double zero_product_ratio = static_cast<double>(zero_product_count) / total_macs_d;
    const double signed_nonzero_count = static_cast<double>(
        std::max<std::size_t>(1U, positive_product_count + negative_product_count));
    const double negative_ratio =
        static_cast<double>(negative_product_count) / signed_nonzero_count;
    const double signed_margin =
        static_cast<double>(bias) + static_cast<double>(signed_product_sum);
    const double abs_product_sum_d = std::max(1.0, static_cast<double>(abs_product_sum));
    const double normalized_margin = signed_margin / abs_product_sum_d;
    const double magnitude_density =
        static_cast<double>(abs_product_sum) / total_macs_d;
    const double positive_bias_term =
        std::max(0.0, static_cast<double>(bias)) / abs_product_sum_d;

    // Sparsity and positive-trend hints separate "easy ET" tasks from dense, positive-leaning ones.
    const double sparsity_tendency =
        std::clamp(0.5 * (zero_activation_ratio + zero_weight_ratio), 0.0, 1.0);
    const double positive_trend_term =
        std::max(0.0, normalized_margin) *
        std::min(1.0, magnitude_density / kMagnitudeDensityScale);

    // Rough ET likelihood increases for negative margins, negative-leaning products, and zeros.
    const double et_likelihood = std::clamp(
        0.50 + kMarginWeight * (-normalized_margin) +
            kNegativeRatioWeight * (negative_ratio - 0.50) +
            kZeroProductWeight * zero_product_ratio +
            0.05 * sparsity_tendency - kPositiveBiasWeight * positive_bias_term -
            kPositiveTrendWeight * positive_trend_term,
        0.0,
        1.0);

    // Expected processed fraction approximates how much serial work survives exact ET.
    const double expected_processed_fraction = std::clamp(
        kExpectedFractionBase - kExpectedFractionEtWeight * et_likelihood -
            kExpectedFractionZeroWeight * zero_product_ratio +
            kExpectedFractionPositiveBiasWeight * positive_bias_term +
            kExpectedFractionPositiveTrendWeight * positive_trend_term,
        kMinExpectedProcessedFraction,
        1.0);

    // Dense, high-magnitude tasks tend to spend more time even when ET does not trigger.
    const double density_tendency = std::clamp(1.0 - sparsity_tendency, 0.0, 1.0);
    const double cost_density_penalty = std::min(
        serial_base * kMaxDensityPenaltyFraction,
        magnitude_density * density_tendency * kDensityPenaltyScale);

    predicted_cost_ = std::max(
        1.0,
        serial_base * expected_processed_fraction + cost_density_penalty);
    cost_bucket_id_ = 0;
    phase_affinity_hint_ =
        (weight_precision_bits > 0)
            ? static_cast<int>((positive_product_count + static_cast<std::size_t>(output_channel_)) %
                               static_cast<std::size_t>(weight_precision_bits))
            : 0;
}

void Task::setCostBucketId(int bucket_id) {
    cost_bucket_id_ = std::max(0, bucket_id);
}

void Task::setAssignedCycle(std::uint64_t cycle) {
    assigned_cycle_ = cycle;
}

void Task::initializeContext(const ConvLayer& layer,
                             ExecutionMode execution_mode,
                             bool enable_importance_ordering,
                             MacOrderingPolicy mac_ordering_policy,
                             BroadcastMode broadcast_mode) {
    worklist_.clear();
    op_index_ = 0;

    accumulator_ = layer.hasBias() ? layer.readBias(output_channel_) : 0;
    pre_relu_output_ = 0;
    final_output_ = 0;
    early_terminated_ = false;
    processed_macs_ = 0;
    processed_bit_steps_ = 0;
    remaining_positive_contribution_upper_bound_ = 0;
    status_ = TaskStatus::Issued;

    const int kernel_size = layer.kernelSize();
    const int input_channels = layer.inputChannels();
    worklist_.reserve(static_cast<std::size_t>(input_channels * kernel_size * kernel_size));

    for (int cin = 0; cin < input_channels; ++cin) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                MacOp op;
                op.cin = cin;
                op.ky = ky;
                op.kx = kx;
                op.input_y = output_y_ * layer.stride() + ky - layer.padding();
                op.input_x = output_x_ * layer.stride() + kx - layer.padding();
                op.activation = layer.readActivation(cin, op.input_y, op.input_x);
                op.weight = layer.readWeight(output_channel_, cin, ky, kx);
                op.activation_key = packActivationKey(cin, op.input_y, op.input_x);
                op.weight_key = packWeightKey(output_channel_, cin, ky, kx);
                op.full_product =
                    static_cast<std::int32_t>(op.activation) * static_cast<std::int32_t>(op.weight);
                op.abs_product = std::llabs(static_cast<long long>(op.full_product));
                op.abs_weight = std::abs(static_cast<int>(op.weight));
                op.leading_one = leadingOnePosition(op.abs_weight);

                const int abs_activation = std::abs(static_cast<int>(op.activation));
                const int abs_weight = std::abs(static_cast<int>(op.weight));
                const int sign = ((op.activation < 0) ^ (op.weight < 0)) ? -1 : 1;
                for (int bit = 0; bit < 8; ++bit) {
                    if (((abs_weight >> bit) & 0x1) != 0) {
                        op.bit_contribution[static_cast<std::size_t>(bit)] =
                            static_cast<std::int32_t>(sign * (abs_activation << bit));
                    } else {
                        op.bit_contribution[static_cast<std::size_t>(bit)] = 0;
                    }
                }

                if (execution_mode == ExecutionMode::Int8BitParallel) {
                    if (op.full_product > 0) {
                        remaining_positive_contribution_upper_bound_ += op.full_product;
                    }
                } else {
                    for (const std::int32_t contribution : op.bit_contribution) {
                        if (contribution > 0) {
                            remaining_positive_contribution_upper_bound_ += contribution;
                        }
                    }
                }
                worklist_.push_back(op);
            }
        }
    }

    const bool use_importance_ordering =
        enable_importance_ordering && broadcast_mode == BroadcastMode::DemandDriven;
    if (use_importance_ordering) {
        if (mac_ordering_policy == MacOrderingPolicy::EtAwareNegativeFirst) {
            std::stable_sort(worklist_.begin(), worklist_.end(), etAwareNegativeFirstLess);
        } else {
            std::stable_sort(worklist_.begin(), worklist_.end(), importanceOrderLess);
        }
    }
}

const MacOp* Task::currentOp() const {
    if (op_index_ >= worklist_.size()) {
        return nullptr;
    }
    return &worklist_[op_index_];
}

bool Task::hasMoreOps() const {
    return op_index_ < worklist_.size();
}

void Task::markMacStarted() {
    if (!hasMoreOps()) {
        return;
    }
    ++processed_macs_;
}

void Task::markBitProcessed(std::int32_t signed_contribution) {
    accumulator_ += signed_contribution;
    ++processed_bit_steps_;
    decrementRemainingPositiveContributionUpperBound(signed_contribution);
}

void Task::markParallelMacProcessed(std::int32_t signed_product) {
    accumulator_ += signed_product;
    ++processed_bit_steps_;
    processed_bit_steps_ += 7U;
    decrementRemainingPositiveContributionUpperBound(signed_product);
}

void Task::advanceToNextOp() {
    if (op_index_ < worklist_.size()) {
        ++op_index_;
    }
}

void Task::finalizeSkippedWorkOnEarlyTermination() {
    if (!early_terminated_) {
        return;
    }

    op_index_ = worklist_.size();
    remaining_positive_contribution_upper_bound_ = 0;
}

void Task::decrementRemainingPositiveContributionUpperBound(std::int32_t signed_delta) {
    if (signed_delta <= 0) {
        return;
    }

    const std::int64_t positive_delta = static_cast<std::int64_t>(signed_delta);
    if (positive_delta > remaining_positive_contribution_upper_bound_) {
        remaining_positive_contribution_upper_bound_ = 0;
    } else {
        remaining_positive_contribution_upper_bound_ -= positive_delta;
    }
}

std::uint64_t Task::packActivationKey(int cin, int y, int x) {
    const std::uint64_t cin_u = static_cast<std::uint64_t>(static_cast<std::uint32_t>(cin) & 0xFFFFU);
    const std::uint64_t y_u = static_cast<std::uint64_t>(static_cast<std::uint32_t>(y + 1024) & 0xFFFFU);
    const std::uint64_t x_u = static_cast<std::uint64_t>(static_cast<std::uint32_t>(x + 1024) & 0xFFFFU);
    return (cin_u << 32U) | (y_u << 16U) | x_u;
}

std::uint64_t Task::packWeightKey(int oc, int cin, int ky, int kx) {
    const std::uint64_t oc_u = static_cast<std::uint64_t>(static_cast<std::uint32_t>(oc) & 0xFFFFU);
    const std::uint64_t cin_u = static_cast<std::uint64_t>(static_cast<std::uint32_t>(cin) & 0xFFFFU);
    const std::uint64_t ky_u = static_cast<std::uint64_t>(static_cast<std::uint32_t>(ky) & 0xFFU);
    const std::uint64_t kx_u = static_cast<std::uint64_t>(static_cast<std::uint32_t>(kx) & 0xFFU);
    return (oc_u << 32U) | (cin_u << 16U) | (ky_u << 8U) | kx_u;
}

int Task::leadingOnePosition(int value) {
    if (value <= 0) {
        return -1;
    }

    int position = 0;
    while (value > 1) {
        value >>= 1;
        ++position;
    }
    return position;
}
