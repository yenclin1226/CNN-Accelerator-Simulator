#include "Accelerator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace {

MemoryConfig composeMemoryConfig(const AcceleratorConfig& config) {
    MemoryConfig memory_config = config.memory;
    memory_config.enable_activation_reuse = config.enable_activation_reuse;
    memory_config.enable_weight_reuse = config.enable_weight_reuse;
    memory_config.local_buffer_capacity_entries = config.local_buffer_capacity_entries;
    memory_config.global_buffer_capacity_entries = config.global_buffer_capacity_entries;
    return memory_config;
}

LaneExecutionConfig composeLaneConfig(const AcceleratorConfig& config) {
    LaneExecutionConfig lane_config;
    lane_config.execution_mode = config.execution_mode;
    lane_config.broadcast_mode = config.broadcast_mode;
    lane_config.pipeline_mode = config.pipeline_mode;
    lane_config.enable_msb_first = config.enable_msb_first;
    lane_config.enable_importance_ordering = config.enable_importance_ordering;
    lane_config.mac_ordering_policy = config.mac_ordering_policy;
    lane_config.enable_early_termination = config.enable_early_termination;
    lane_config.enable_reactive_zero_skip = config.enable_reactive_zero_skip;
    lane_config.enable_proactive_zero_run_skip = config.enable_proactive_zero_run_skip;
    lane_config.zero_run_order_mode = config.zero_run_order_mode;
    lane_config.enable_bit_column_skip = config.enable_bit_column_skip;
    return lane_config;
}

int decodeOutputChannelFromWeightKey(std::uint64_t weight_key) {
    return static_cast<int>((weight_key >> 32U) & 0xFFFFU);
}

constexpr int kEtaCostBuckets = 4;
constexpr double kObservedProcessedWorkWeight = 0.70;
constexpr double kObservedResidenceWeight = 0.30;
constexpr double kPerProcessedMacOverhead = 1.0;
constexpr double kTaskFixedOverhead = 4.0;

constexpr double kPlacementCostWeight = 2.80;
constexpr double kPlacementBucketWeight = 0.85;
constexpr double kPlacementLoadWeight = 1.35;
constexpr double kPlacementNaturalGroupPenalty = 0.80;
constexpr double kPlacementExistingFamilyRewardScale = 0.12;
constexpr double kPlacementExistingFamilyRewardCap = 0.45;
constexpr double kPlacementPhaseMismatchPenalty = 0.35;
// Memory-aware ECBG keeps ET/runtime similarity primary and adds bounded locality rewards
// so static grouping preserves more weight/activation reuse without turning into a
// locality-first policy.
constexpr double kPlacementWeightLocalityRewardScale = 0.22;
constexpr double kPlacementWeightLocalityRewardCap = 0.70;
constexpr double kPlacementSpatialLocalityRewardScale = 0.06;
constexpr double kPlacementSpatialLocalityRewardCap = 0.22;
constexpr double kPlacementLocalityTieEpsilon = 0.08;
constexpr double kPlacementLowCostLocalityShortlistEpsilon = 0.14;
constexpr double kBroadcastFanoutDominantFamilyReward = 2.0;
constexpr double kBroadcastFanoutNaturalGroupReward = 1.5;
constexpr double kBroadcastFanoutPhaseMatchReward = 1.5;
constexpr double kBroadcastFanoutExactChannelRewardScale = 0.35;
constexpr double kBroadcastFanoutExactChannelRewardCap = 1.25;
constexpr double kBroadcastFanoutAccumulatedCostPenalty = 0.8;
constexpr double kBroadcastFanoutDeviationPenalty = 0.4;
constexpr int kSpatialTileSize = 2;

std::uint64_t packWeightKey(int oc, int cin, int ky, int kx) {
    const std::uint64_t oc_u = static_cast<std::uint64_t>(static_cast<std::uint32_t>(oc) & 0xFFFFU);
    const std::uint64_t cin_u =
        static_cast<std::uint64_t>(static_cast<std::uint32_t>(cin) & 0xFFFFU);
    const std::uint64_t ky_u = static_cast<std::uint64_t>(static_cast<std::uint32_t>(ky) & 0xFFU);
    const std::uint64_t kx_u = static_cast<std::uint64_t>(static_cast<std::uint32_t>(kx) & 0xFFU);
    return (oc_u << 32U) | (cin_u << 16U) | (ky_u << 8U) | kx_u;
}

bool groupOwnsOutputChannel(int group_id,
                            int output_channel,
                            int num_groups,
                            int total_output_channels) {
    if (total_output_channels <= 0) {
        return false;
    }

    if (total_output_channels >= num_groups) {
        return (output_channel % num_groups) == group_id;
    }

    return output_channel == (group_id % total_output_channels);
}

double computeVariance(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }

    const double mean =
        std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
    double variance = 0.0;
    for (double value : values) {
        const double diff = value - mean;
        variance += diff * diff;
    }
    return variance / static_cast<double>(values.size());
}

int computeSpatialTileId(const Task& task) {
    const int tile_y = task.outputY() / kSpatialTileSize;
    const int tile_x = task.outputX() / kSpatialTileSize;
    return tile_y * 1024 + tile_x;
}

double computeJainFairness(const std::vector<double>& values) {
    if (values.empty()) {
        return 1.0;
    }

    double sum = 0.0;
    double sum_sq = 0.0;
    for (double value : values) {
        sum += value;
        sum_sq += value * value;
    }
    if (sum_sq == 0.0) {
        return 1.0;
    }
    const double n = static_cast<double>(values.size());
    return (sum * sum) / (n * sum_sq);
}

double observedTaskCostProxy(const Task& task, std::uint64_t completion_cycle) {
    // Feedback blends deterministic processed work with light residence-time attribution so
    // the scheduler can react to real occupancy/stall behavior without heavy instrumentation.
    const double processed_work =
        static_cast<double>(task.processedBitSteps()) +
        static_cast<double>(task.processedMacs()) * kPerProcessedMacOverhead +
        kTaskFixedOverhead;
    double residence_cycles = 0.0;
    if (task.assignedCycle() > 0U && completion_cycle + 1U >= task.assignedCycle()) {
        residence_cycles =
            static_cast<double>(completion_cycle + 1U - task.assignedCycle());
    }

    return std::max(
        1.0,
        kObservedProcessedWorkWeight * processed_work +
            kObservedResidenceWeight * residence_cycles);
}

}  // namespace

Accelerator::Accelerator(const AcceleratorConfig& config)
    : config_(config), scheduler_(config.num_groups), memory_(composeMemoryConfig(config), config.num_groups) {
    if (config_.num_groups <= 0) {
        throw std::invalid_argument("Accelerator must have at least one PE group.");
    }
    if (config_.lanes_per_group <= 0) {
        throw std::invalid_argument("Accelerator must have at least one lane per group.");
    }
    if (config_.weight_precision_bits <= 0) {
        throw std::invalid_argument("Weight precision bits must be positive.");
    }
    initializeGroups();
}

SimulationStats Accelerator::run(ConvLayer& layer) {
    initializeGroups();
    scheduler_.clear();
    scheduler_.setCostAwareStealingEnabled(
        config_.grouping_policy == GroupingPolicy::ETAwareCostBalanced ||
        config_.grouping_policy == GroupingPolicy::ETAwareCostBalancedMemoryAware ||
        config_.grouping_policy == GroupingPolicy::BroadcastPhaseAwareFanoutBalanced);
    memory_.reset();
    layer.zeroOutput();
    configureBroadcastersForLayer(layer);

    std::vector<Task> tasks = layer.generateTasks();
    const std::uint64_t total_tasks = static_cast<std::uint64_t>(tasks.size());
    enqueueLayerTasks(std::move(tasks), layer);

    const LaneExecutionConfig lane_config = composeLaneConfig(config_);

    std::vector<TaskReport> task_reports;
    task_reports.reserve(static_cast<std::size_t>(total_tasks));

    std::uint64_t completed_tasks = 0;
    std::uint64_t cycle = 0;

    while (completed_tasks < total_tasks) {
        if (cycle > config_.max_cycles) {
            throw std::runtime_error("Simulation exceeded max cycle budget.");
        }

        for (PEGroup& group : groups_) {
            std::vector<BroadcastDemand> demands;
            if (config_.broadcast_mode == BroadcastMode::DemandDriven) {
                demands.reserve(group.lanes().size());
                for (const Lane& lane : group.lanes()) {
                    if (std::optional<BroadcastDemand> demand = lane.broadcastDemand(cycle, lane_config)) {
                        demands.push_back(*demand);
                    }
                }
            }
            group.broadcaster().prepareCycle(
                cycle, demands, lane_config.execution_mode, config_.broadcast_mode);
        }

        for (PEGroup& group : groups_) {
            for (Lane& lane : group.lanes()) {
                std::optional<Task> done = lane.step(
                    cycle, group.id(), group.broadcaster(), memory_, layer, lane_config);
                if (!done) {
                    continue;
                }

                scheduler_.recordTaskCompletion(
                    group.id(), *done, observedTaskCostProxy(*done, cycle));
                ++completed_tasks;
                TaskReport report;
                report.task_id = done->id();
                report.output_channel = done->outputChannel();
                report.output_y = done->outputY();
                report.output_x = done->outputX();
                report.early_terminated = done->earlyTerminated();
                report.pre_relu_output = done->preReluOutput();
                report.post_relu_output = done->finalOutput();
                report.processed_macs = static_cast<std::uint64_t>(done->processedMacs());
                report.skipped_macs_total = static_cast<std::uint64_t>(done->skippedMacsTotal());
                report.skipped_macs_et_only = static_cast<std::uint64_t>(done->skippedMacsEtOnly());
                report.skipped_macs_reactive_only =
                    static_cast<std::uint64_t>(done->skippedMacsReactiveOnly());
                report.skipped_macs_proactive_only =
                    static_cast<std::uint64_t>(done->skippedMacsProactiveOnly());
                report.skipped_macs_zero_only =
                    static_cast<std::uint64_t>(done->skippedMacsZeroOnly());
                report.processed_bit_steps = static_cast<std::uint64_t>(done->processedBitSteps());
                report.skipped_bit_steps_total =
                    static_cast<std::uint64_t>(done->skippedBitStepsTotal());
                report.skipped_bit_steps_et_only =
                    static_cast<std::uint64_t>(done->skippedBitStepsEtOnly());
                report.skipped_bit_steps_bit_column_only =
                    static_cast<std::uint64_t>(done->skippedBitStepsBitColumnOnly());
                report.zero_run_events = static_cast<std::uint64_t>(done->zeroRunEvents());
                task_reports.push_back(report);
            }
        }

        for (PEGroup& group : groups_) {
            for (Lane& lane : group.lanes()) {
                if (!lane.isIdle()) {
                    continue;
                }

                std::optional<int> preferred_output_channel;
                if (std::optional<std::uint64_t> payload_weight_key =
                        group.broadcaster().currentPayloadWeightKey()) {
                    preferred_output_channel = decodeOutputChannelFromWeightKey(*payload_weight_key);
                }

                std::optional<Task> task = scheduler_.fetchTask(group.id(), preferred_output_channel);
                if (!task) {
                    continue;
                }

                task->setAssignedCycle(cycle + 1U);
                lane.assignTask(std::move(*task), cycle, group.id(), lane_config, memory_, layer);
            }
        }

        ++cycle;

        if (completed_tasks < total_tasks && !scheduler_.hasPendingTasks() && !anyLaneBusy()) {
            throw std::runtime_error("Simulation stalled before all tasks completed.");
        }
    }

    std::sort(task_reports.begin(), task_reports.end(), [](const TaskReport& a, const TaskReport& b) {
        return a.task_id < b.task_id;
    });

    SimulationStats stats;
    stats.total_cycles = cycle;
    stats.completed_tasks = completed_tasks;
    stats.task_reports = std::move(task_reports);
    stats.average_queued_predicted_cost_per_group.resize(static_cast<std::size_t>(config_.num_groups), 0.0);
    stats.queued_predicted_cost_variance_per_group.resize(static_cast<std::size_t>(config_.num_groups), 0.0);
    stats.completed_tasks_per_group.resize(static_cast<std::size_t>(config_.num_groups), 0);
    stats.active_lane_cycles_per_group.resize(static_cast<std::size_t>(config_.num_groups), 0);

    for (const PEGroup& group : groups_) {
        std::uint64_t group_active_cycles = 0;
        std::uint64_t group_completed_tasks = 0;
        for (const Lane& lane : group.lanes()) {
            stats.active_lane_cycles += lane.activeCycles();
            stats.idle_lane_cycles += lane.idleCycles();
            stats.idle_finished_cycles += lane.idleFinishedCycles();
            stats.stall_memory_cycles += lane.stallMemoryCycles();
            stats.stall_broadcast_cycles += lane.stallBroadcastCycles();
            stats.idle_broadcast_mismatch_cycles += lane.idleBroadcastMismatchCycles();
            stats.multiplier_active_cycles += lane.multiplierActiveCycles();
            stats.accumulator_active_cycles += lane.accumulatorActiveCycles();
            stats.writeback_cycles += lane.writebackCycles();
            stats.task_issue_events += lane.taskIssueEvents();
            stats.total_macs += lane.macsExecuted();
            stats.tasks_terminated_early += lane.earlyTerminatedTasks();
            stats.macs_skipped_total += lane.skippedMacsTotal();
            stats.macs_skipped_et_only += lane.skippedMacsEtOnly();
            stats.macs_skipped_reactive_only += lane.skippedMacsReactiveOnly();
            stats.macs_skipped_proactive_only += lane.skippedMacsProactiveOnly();
            stats.macs_skipped_zero_only += lane.skippedMacsZeroOnly();
            stats.bit_steps_skipped_total += lane.skippedBitStepsTotal();
            stats.bit_steps_skipped_et_only += lane.skippedBitStepsEtOnly();
            stats.bit_steps_skipped_bit_column_only += lane.skippedBitStepsBitColumnOnly();
            stats.zero_run_events += lane.zeroRunEvents();
            stats.estimated_cycles_saved_early_termination +=
                lane.estimatedCyclesSavedByEarlyTermination();
            group_active_cycles += lane.activeCycles();
            group_completed_tasks += lane.completedTasks();
        }
        stats.active_lane_cycles_per_group[static_cast<std::size_t>(group.id())] = group_active_cycles;
        stats.completed_tasks_per_group[static_cast<std::size_t>(group.id())] = group_completed_tasks;
    }

    const std::uint64_t total_lane_cycles =
        cycle * static_cast<std::uint64_t>(config_.num_groups) *
        static_cast<std::uint64_t>(config_.lanes_per_group);
    if (total_lane_cycles == 0) {
        stats.lane_occupancy = 0.0;
        stats.lane_utilization = 0.0;
        stats.multiplier_utilization = 0.0;
        stats.accumulator_utilization = 0.0;
        stats.memory_stall_ratio = 0.0;
        stats.broadcast_stall_ratio = 0.0;
        stats.broadcast_mismatch_idle_ratio = 0.0;
        stats.finished_lane_idle_ratio = 0.0;
        stats.idle_ratio = 0.0;
    } else {
        const double denom = static_cast<double>(total_lane_cycles);
        stats.lane_occupancy = static_cast<double>(stats.active_lane_cycles) / denom;
        stats.lane_utilization = stats.lane_occupancy;
        stats.multiplier_utilization = static_cast<double>(stats.multiplier_active_cycles) / denom;
        stats.accumulator_utilization = static_cast<double>(stats.accumulator_active_cycles) / denom;
        stats.memory_stall_ratio = static_cast<double>(stats.stall_memory_cycles) / denom;
        stats.broadcast_stall_ratio = static_cast<double>(stats.stall_broadcast_cycles) / denom;
        stats.broadcast_mismatch_idle_ratio =
            static_cast<double>(stats.idle_broadcast_mismatch_cycles) / denom;
        stats.finished_lane_idle_ratio = static_cast<double>(stats.idle_finished_cycles) / denom;
        stats.idle_ratio = static_cast<double>(stats.idle_lane_cycles) / denom;
    }
    stats.macs_skipped = stats.macs_skipped_total;
    stats.bit_steps_skipped = stats.bit_steps_skipped_total;

    stats.dram_accesses = memory_.dramAccesses();
    stats.onchip_buffer_accesses = memory_.onChipBufferAccesses();
    stats.dram_bytes = memory_.dramBytesMoved();
    stats.onchip_buffer_bytes = memory_.onChipBufferBytesMoved();
    stats.activation_reuse_hits = memory_.activationReuseHits();
    stats.activation_reuse_misses = memory_.activationReuseMisses();
    stats.weight_reuse_hits = memory_.weightReuseHits();
    stats.weight_reuse_misses = memory_.weightReuseMisses();
    stats.memory_requests_avoided = memory_.memoryRequestsAvoided();
    stats.bytes_saved_due_to_reuse = memory_.bytesSavedDueToReuse();

    stats.work_stealing_events = scheduler_.workStealingEvents();
    stats.throughput_macs_per_cycle =
        (cycle == 0) ? 0.0 : static_cast<double>(stats.total_macs) / static_cast<double>(cycle);

    std::vector<double> group_average_costs;
    std::vector<double> group_completed_task_values;
    std::vector<double> group_active_cycle_values;
    group_average_costs.reserve(initial_group_task_counts_.size());
    group_completed_task_values.reserve(stats.completed_tasks_per_group.size());
    group_active_cycle_values.reserve(stats.active_lane_cycles_per_group.size());
    for (std::size_t group = 0; group < initial_group_task_counts_.size(); ++group) {
        const double count = static_cast<double>(initial_group_task_counts_[group]);
        const double mean_cost =
            (count == 0.0) ? 0.0 : initial_group_predicted_cost_sums_[group] / count;
        const double mean_sq =
            (count == 0.0) ? 0.0 : initial_group_predicted_cost_square_sums_[group] / count;
        const double variance = std::max(0.0, mean_sq - mean_cost * mean_cost);
        stats.average_queued_predicted_cost_per_group[group] = mean_cost;
        stats.queued_predicted_cost_variance_per_group[group] = variance;
        group_average_costs.push_back(mean_cost);
        group_completed_task_values.push_back(
            static_cast<double>(stats.completed_tasks_per_group[group]));
        group_active_cycle_values.push_back(
            static_cast<double>(stats.active_lane_cycles_per_group[group]));
    }
    stats.variance_group_average_predicted_cost = computeVariance(group_average_costs);
    stats.variance_group_completed_tasks = computeVariance(group_completed_task_values);
    stats.variance_group_active_lane_cycles = computeVariance(group_active_cycle_values);

    std::vector<double> lane_workloads;
    lane_workloads.reserve(static_cast<std::size_t>(config_.num_groups * config_.lanes_per_group));
    for (const PEGroup& group : groups_) {
        for (const Lane& lane : group.lanes()) {
            lane_workloads.push_back(static_cast<double>(lane.activeCycles()));
        }
    }
    stats.lane_workload_jain_fairness = computeJainFairness(lane_workloads);

    std::uint64_t terminated_outputs = 0;
    double processed_fraction_sum = 0.0;
    for (const TaskReport& report : stats.task_reports) {
        if (report.early_terminated) {
            ++terminated_outputs;
        }
        const std::uint64_t total_ops = report.processed_macs + report.skipped_macs_total;
        if (total_ops == 0) {
            processed_fraction_sum += 1.0;
        } else {
            processed_fraction_sum +=
                static_cast<double>(report.processed_macs) / static_cast<double>(total_ops);
        }
    }
    stats.output_elements_terminated_early = terminated_outputs;
    stats.average_processed_fraction_per_task =
        stats.task_reports.empty()
            ? 1.0
            : processed_fraction_sum / static_cast<double>(stats.task_reports.size());

    return stats;
}

void Accelerator::initializeGroups() {
    groups_.clear();
    groups_.reserve(static_cast<std::size_t>(config_.num_groups));

    for (int group_id = 0; group_id < config_.num_groups; ++group_id) {
        int phase_offset = group_id % config_.weight_precision_bits;
        if (!config_.phase_offsets.empty()) {
            phase_offset = config_.phase_offsets[static_cast<std::size_t>(group_id) %
                                                 config_.phase_offsets.size()];
        }
        groups_.emplace_back(
            group_id, config_.lanes_per_group, config_.weight_precision_bits, phase_offset);
    }
}

void Accelerator::configureBroadcastersForLayer(const ConvLayer& layer) {
    if (config_.broadcast_mode != BroadcastMode::SnapeaFixedSchedule) {
        for (PEGroup& group : groups_) {
            group.broadcaster().clearFixedSchedule();
        }
        return;
    }

    for (PEGroup& group : groups_) {
        const std::vector<FixedBroadcastScheduleEntry> fixed_schedule =
            buildFixedBroadcastSchedule(layer, group.id());
        group.broadcaster().setFixedSchedule(fixed_schedule);
    }
}

void Accelerator::initializeTaskSchedulingMetadata(std::vector<Task>& tasks,
                                                   const ConvLayer& layer) const {
    for (Task& task : tasks) {
        task.initializeSchedulingMetadata(layer, config_.weight_precision_bits);
    }
}

void Accelerator::assignTaskCostBuckets(std::vector<Task>& tasks) {
    if (tasks.empty()) {
        initial_bucket_reference_costs_.assign(static_cast<std::size_t>(kEtaCostBuckets), 0.0);
        return;
    }

    // First-pass ECBG uses per-layer quantile buckets, then runtime feedback re-centers
    // bucket reference costs for later stealing without reshuffling the queued tasks.
    std::vector<double> sorted_costs;
    sorted_costs.reserve(tasks.size());
    for (const Task& task : tasks) {
        sorted_costs.push_back(task.predictedCost());
    }
    std::sort(sorted_costs.begin(), sorted_costs.end());

    std::vector<double> thresholds;
    thresholds.reserve(static_cast<std::size_t>(kEtaCostBuckets - 1));
    for (int bucket = 1; bucket < kEtaCostBuckets; ++bucket) {
        const std::size_t index = std::min(
            sorted_costs.size() - 1U,
            (sorted_costs.size() * static_cast<std::size_t>(bucket)) /
                static_cast<std::size_t>(kEtaCostBuckets));
        thresholds.push_back(sorted_costs[index]);
    }

    for (Task& task : tasks) {
        const int bucket_id = static_cast<int>(
            std::upper_bound(thresholds.begin(), thresholds.end(), task.predictedCost()) -
            thresholds.begin());
        task.setCostBucketId(bucket_id);
    }

    initial_bucket_reference_costs_.assign(static_cast<std::size_t>(kEtaCostBuckets), 0.0);
    std::vector<std::vector<double>> bucket_costs(static_cast<std::size_t>(kEtaCostBuckets));
    for (const Task& task : tasks) {
        bucket_costs[static_cast<std::size_t>(task.costBucketId())].push_back(task.predictedCost());
    }

    const double fallback_center =
        std::accumulate(sorted_costs.begin(), sorted_costs.end(), 0.0) /
        static_cast<double>(sorted_costs.size());
    for (int bucket = 0; bucket < kEtaCostBuckets; ++bucket) {
        const std::vector<double>& values = bucket_costs[static_cast<std::size_t>(bucket)];
        initial_bucket_reference_costs_[static_cast<std::size_t>(bucket)] =
            values.empty()
                ? fallback_center
                : std::accumulate(values.begin(), values.end(), 0.0) /
                      static_cast<double>(values.size());
    }
}

std::vector<FixedBroadcastScheduleEntry> Accelerator::buildFixedBroadcastSchedule(
    const ConvLayer& layer,
    int group_id) const {
    std::size_t scheduled_output_channels = 0;
    for (int oc = 0; oc < layer.outputChannels(); ++oc) {
        if (groupOwnsOutputChannel(group_id, oc, config_.num_groups, layer.outputChannels())) {
            ++scheduled_output_channels;
        }
    }

    const std::size_t weight_count = scheduled_output_channels *
                                     static_cast<std::size_t>(layer.inputChannels()) *
                                     static_cast<std::size_t>(layer.kernelSize()) *
                                     static_cast<std::size_t>(layer.kernelSize());
    const std::size_t stream_length =
        weight_count *
        static_cast<std::size_t>(config_.execution_mode == ExecutionMode::Int8BitParallel
                                     ? 1
                                     : config_.weight_precision_bits);

    std::vector<FixedBroadcastScheduleEntry> schedule;
    schedule.reserve(stream_length);

    for (int oc = 0; oc < layer.outputChannels(); ++oc) {
        if (!groupOwnsOutputChannel(group_id, oc, config_.num_groups, layer.outputChannels())) {
            continue;
        }
        for (int cin = 0; cin < layer.inputChannels(); ++cin) {
            for (int ky = 0; ky < layer.kernelSize(); ++ky) {
                for (int kx = 0; kx < layer.kernelSize(); ++kx) {
                    FixedBroadcastScheduleEntry entry;
                    entry.weight_key = packWeightKey(oc, cin, ky, kx);

                    if (config_.execution_mode == ExecutionMode::Int8BitParallel) {
                        entry.bit_index = -1;
                        schedule.push_back(entry);
                        continue;
                    }

                    for (int step = 0; step < config_.weight_precision_bits; ++step) {
                        entry.bit_index = config_.enable_msb_first
                                              ? (config_.weight_precision_bits - 1 - step)
                                              : step;
                        schedule.push_back(entry);
                    }
                }
            }
        }
    }

    return schedule;
}

std::vector<int> Accelerator::buildEtawareBalancedAssignments(const std::vector<Task>& tasks) const {
    struct GroupAssignmentState {
        double predicted_cost_sum{0.0};
        std::uint64_t task_count{0};
        std::vector<std::uint64_t> output_family_counts;
    };

    std::vector<int> task_indices(tasks.size(), 0);
    std::iota(task_indices.begin(), task_indices.end(), 0);
    std::stable_sort(task_indices.begin(), task_indices.end(), [&](int lhs, int rhs) {
        const Task& a = tasks[static_cast<std::size_t>(lhs)];
        const Task& b = tasks[static_cast<std::size_t>(rhs)];
        if (a.predictedCost() != b.predictedCost()) {
            return a.predictedCost() > b.predictedCost();
        }
        if (a.costBucketId() != b.costBucketId()) {
            return a.costBucketId() < b.costBucketId();
        }
        if (a.outputChannel() != b.outputChannel()) {
            return a.outputChannel() < b.outputChannel();
        }
        return a.id() < b.id();
    });

    std::vector<double> bucket_centers = initial_bucket_reference_costs_;
    if (bucket_centers.size() != static_cast<std::size_t>(kEtaCostBuckets)) {
        double fallback_center = 0.0;
        for (const Task& task : tasks) {
            fallback_center += task.predictedCost();
        }
        fallback_center /= static_cast<double>(tasks.size());
        bucket_centers.assign(static_cast<std::size_t>(kEtaCostBuckets), fallback_center);
    }

    std::vector<GroupAssignmentState> group_states(static_cast<std::size_t>(config_.num_groups));
    for (GroupAssignmentState& state : group_states) {
        state.output_family_counts.assign(static_cast<std::size_t>(config_.num_groups), 0U);
    }
    std::vector<int> assignments(tasks.size(), 0);
    double total_assigned_cost = 0.0;

    // Static greedy placement tries to minimize ET-runtime divergence without sacrificing too
    // much broadcast/locality efficiency. Cost remains primary, while output-family locality,
    // phase affinity, and overload avoidance act as meaningful secondary corrections.
    for (int task_index : task_indices) {
        const Task& task = tasks[static_cast<std::size_t>(task_index)];
        double best_score = std::numeric_limits<double>::max();
        int best_group = 0;
        const int output_family = task.outputChannel() % config_.num_groups;
        const int natural_group = initialGroupForTask(task);

        for (int group_id = 0; group_id < config_.num_groups; ++group_id) {
            const GroupAssignmentState& state = group_states[static_cast<std::size_t>(group_id)];
            const int preferred_bucket = preferredCostBucketForGroup(group_id);
            const double reference_cost =
                (state.task_count == 0U)
                    ? bucket_centers[static_cast<std::size_t>(preferred_bucket)]
                    : state.predicted_cost_sum / static_cast<double>(state.task_count);
            const double normalized_cost_penalty =
                std::abs(task.predictedCost() - reference_cost) /
                std::max(32.0, std::max(task.predictedCost(), reference_cost));
            const double bucket_penalty =
                static_cast<double>(std::abs(task.costBucketId() - preferred_bucket));
            const double projected_group_load = state.predicted_cost_sum + task.predictedCost();
            const double average_projected_load =
                (total_assigned_cost + task.predictedCost()) /
                static_cast<double>(config_.num_groups);
            const double load_penalty =
                std::max(0.0, projected_group_load - average_projected_load) /
                std::max(32.0, average_projected_load);
            const double existing_family_reward =
                std::min(kPlacementExistingFamilyRewardCap,
                         static_cast<double>(
                             state.output_family_counts[static_cast<std::size_t>(output_family)]) *
                             kPlacementExistingFamilyRewardScale);
            const double locality_penalty = std::max(
                0.0,
                ((group_id == natural_group) ? 0.0 : kPlacementNaturalGroupPenalty) -
                    existing_family_reward);
            const double phase_penalty =
                (config_.phase_offsets.empty())
                    ? 0.0
                    : (config_.phase_offsets[static_cast<std::size_t>(group_id) %
                                             config_.phase_offsets.size()] == task.phaseAffinityHint())
                          ? 0.0
                          : kPlacementPhaseMismatchPenalty;

            const double score =
                normalized_cost_penalty * kPlacementCostWeight +
                bucket_penalty * kPlacementBucketWeight +
                load_penalty * kPlacementLoadWeight + locality_penalty + phase_penalty;
            if (score < best_score ||
                (score == best_score && state.task_count <
                                           group_states[static_cast<std::size_t>(best_group)].task_count)) {
                best_score = score;
                best_group = group_id;
            }
        }

        assignments[static_cast<std::size_t>(task_index)] = best_group;
        GroupAssignmentState& state = group_states[static_cast<std::size_t>(best_group)];
        state.predicted_cost_sum += task.predictedCost();
        ++state.task_count;
        ++state.output_family_counts[static_cast<std::size_t>(output_family)];
        total_assigned_cost += task.predictedCost();
    }

    return assignments;
}

std::vector<int> Accelerator::buildEtawareMemoryAwareAssignments(
    const std::vector<Task>& tasks) const {
    struct GroupAssignmentState {
        double predicted_cost_sum{0.0};
        std::uint64_t task_count{0};
        std::vector<std::uint64_t> output_family_counts;
        std::unordered_map<int, std::uint64_t> output_channel_counts;
        std::unordered_map<int, std::uint64_t> spatial_tile_counts;
    };

    std::vector<int> task_indices(tasks.size(), 0);
    std::iota(task_indices.begin(), task_indices.end(), 0);
    std::stable_sort(task_indices.begin(), task_indices.end(), [&](int lhs, int rhs) {
        const Task& a = tasks[static_cast<std::size_t>(lhs)];
        const Task& b = tasks[static_cast<std::size_t>(rhs)];
        if (a.predictedCost() != b.predictedCost()) {
            return a.predictedCost() > b.predictedCost();
        }
        if (a.costBucketId() != b.costBucketId()) {
            return a.costBucketId() < b.costBucketId();
        }
        if (a.outputChannel() != b.outputChannel()) {
            return a.outputChannel() < b.outputChannel();
        }
        return a.id() < b.id();
    });

    std::vector<double> bucket_centers = initial_bucket_reference_costs_;
    if (bucket_centers.size() != static_cast<std::size_t>(kEtaCostBuckets)) {
        double fallback_center = 0.0;
        for (const Task& task : tasks) {
            fallback_center += task.predictedCost();
        }
        fallback_center /= static_cast<double>(tasks.size());
        bucket_centers.assign(static_cast<std::size_t>(kEtaCostBuckets), fallback_center);
    }

    std::vector<GroupAssignmentState> group_states(static_cast<std::size_t>(config_.num_groups));
    for (GroupAssignmentState& state : group_states) {
        state.output_family_counts.assign(static_cast<std::size_t>(config_.num_groups), 0U);
    }
    std::vector<int> assignments(tasks.size(), 0);
    double total_assigned_cost = 0.0;

    // This variant keeps ET-aware cost balancing primary, then uses same-output-channel and
    // nearby-spatial-tile rewards as lightweight corrections to preserve local memory reuse.
    for (int task_index : task_indices) {
        const Task& task = tasks[static_cast<std::size_t>(task_index)];
        const int output_family = task.outputChannel() % config_.num_groups;
        const int natural_group = initialGroupForTask(task);
        const int spatial_tile_id = computeSpatialTileId(task);
        const std::size_t high_cost_bucket_index =
            std::min<std::size_t>(bucket_centers.size() - 1U, static_cast<std::size_t>(kEtaCostBuckets / 2));
        const double high_cost_reference = bucket_centers[high_cost_bucket_index];
        const bool tighten_locality_window =
            task.costBucketId() >= (kEtaCostBuckets - 1) || task.predictedCost() >= high_cost_reference;
        const double locality_shortlist_epsilon =
            tighten_locality_window ? kPlacementLocalityTieEpsilon
                                    : kPlacementLowCostLocalityShortlistEpsilon;

        struct CandidateScore {
            int group_id{0};
            double core_score{0.0};
            double locality_score{0.0};
            std::uint64_t task_count{0U};
        };

        std::vector<CandidateScore> candidates;
        candidates.reserve(static_cast<std::size_t>(config_.num_groups));
        double best_core_score = std::numeric_limits<double>::max();
        std::uint64_t best_anchor_channel_count = 0U;
        std::uint64_t best_anchor_tile_count = 0U;
        int best_anchor_channel_group = -1;
        int best_anchor_tile_group = -1;

        for (int group_id = 0; group_id < config_.num_groups; ++group_id) {
            const GroupAssignmentState& state = group_states[static_cast<std::size_t>(group_id)];
            const int preferred_bucket = preferredCostBucketForGroup(group_id);
            const double reference_cost =
                (state.task_count == 0U)
                    ? bucket_centers[static_cast<std::size_t>(preferred_bucket)]
                    : state.predicted_cost_sum / static_cast<double>(state.task_count);
            const double normalized_cost_penalty =
                std::abs(task.predictedCost() - reference_cost) /
                std::max(32.0, std::max(task.predictedCost(), reference_cost));
            const double bucket_penalty =
                static_cast<double>(std::abs(task.costBucketId() - preferred_bucket));
            const double projected_group_load = state.predicted_cost_sum + task.predictedCost();
            const double average_projected_load =
                (total_assigned_cost + task.predictedCost()) /
                static_cast<double>(config_.num_groups);
            const double load_penalty =
                std::max(0.0, projected_group_load - average_projected_load) /
                std::max(32.0, average_projected_load);
            const double locality_reward_slack =
                (projected_group_load > average_projected_load) ? (1.0 / (1.0 + load_penalty)) : 1.0;
            const double existing_family_reward =
                std::min(kPlacementExistingFamilyRewardCap,
                         static_cast<double>(
                             state.output_family_counts[static_cast<std::size_t>(output_family)]) *
                             kPlacementExistingFamilyRewardScale);
            const double locality_penalty = std::max(
                0.0,
                ((group_id == natural_group) ? 0.0 : kPlacementNaturalGroupPenalty) -
                    existing_family_reward);
            const double phase_penalty =
                (config_.phase_offsets.empty())
                    ? 0.0
                    : (config_.phase_offsets[static_cast<std::size_t>(group_id) %
                                             config_.phase_offsets.size()] ==
                       task.phaseAffinityHint())
                          ? 0.0
                          : kPlacementPhaseMismatchPenalty;

            const auto output_channel_it = state.output_channel_counts.find(task.outputChannel());
            const std::uint64_t same_channel_count =
                (output_channel_it == state.output_channel_counts.end()) ? 0U : output_channel_it->second;
            const double same_channel_reward =
                std::min(kPlacementWeightLocalityRewardCap,
                         static_cast<double>(same_channel_count) *
                             kPlacementWeightLocalityRewardScale) *
                locality_reward_slack;

            const auto spatial_tile_it = state.spatial_tile_counts.find(spatial_tile_id);
            const std::uint64_t same_tile_count =
                (spatial_tile_it == state.spatial_tile_counts.end()) ? 0U : spatial_tile_it->second;
            const double same_tile_reward =
                std::min(kPlacementSpatialLocalityRewardCap,
                         static_cast<double>(same_tile_count) *
                             kPlacementSpatialLocalityRewardScale) *
                locality_reward_slack;

            const double core_score =
                normalized_cost_penalty * kPlacementCostWeight +
                bucket_penalty * kPlacementBucketWeight +
                load_penalty * kPlacementLoadWeight + locality_penalty + phase_penalty;
            const double locality_score = same_channel_reward + same_tile_reward;
            best_core_score = std::min(best_core_score, core_score);
            if (same_channel_count > best_anchor_channel_count) {
                best_anchor_channel_count = same_channel_count;
                best_anchor_channel_group = group_id;
            }
            if (same_tile_count > best_anchor_tile_count) {
                best_anchor_tile_count = same_tile_count;
                best_anchor_tile_group = group_id;
            }
            candidates.push_back(CandidateScore{
                group_id,
                core_score,
                locality_score,
                state.task_count});
        }

        double best_shortlist_core_score = std::numeric_limits<double>::max();
        double best_locality_score = -std::numeric_limits<double>::infinity();
        std::uint64_t best_task_count = std::numeric_limits<std::uint64_t>::max();
        int best_group = candidates.front().group_id;
        std::vector<bool> shortlist_members(static_cast<std::size_t>(config_.num_groups), false);
        for (const CandidateScore& candidate : candidates) {
            const bool core_in_shortlist =
                candidate.core_score <= best_core_score + locality_shortlist_epsilon;
            if (core_in_shortlist) {
                shortlist_members[static_cast<std::size_t>(candidate.group_id)] = true;
            }
        }
        if (best_anchor_channel_count > 0U) {
            shortlist_members[static_cast<std::size_t>(best_anchor_channel_group)] = true;
        }
        if (best_anchor_tile_count > 0U) {
            shortlist_members[static_cast<std::size_t>(best_anchor_tile_group)] = true;
        }

        for (const CandidateScore& candidate : candidates) {
            if (!shortlist_members[static_cast<std::size_t>(candidate.group_id)]) {
                continue;
            }
            if (candidate.locality_score > best_locality_score ||
                (candidate.locality_score == best_locality_score &&
                 (candidate.core_score + kPlacementLocalityTieEpsilon < best_shortlist_core_score ||
                  (std::abs(candidate.core_score - best_shortlist_core_score) <=
                       kPlacementLocalityTieEpsilon &&
                   candidate.task_count < best_task_count)))) {
                best_shortlist_core_score = candidate.core_score;
                best_locality_score = candidate.locality_score;
                best_task_count = candidate.task_count;
                best_group = candidate.group_id;
            }
        }

        if (best_locality_score == -std::numeric_limits<double>::infinity()) {
            for (const CandidateScore& candidate : candidates) {
                if (candidate.core_score + kPlacementLocalityTieEpsilon < best_shortlist_core_score ||
                    (std::abs(candidate.core_score - best_shortlist_core_score) <=
                         kPlacementLocalityTieEpsilon &&
                     candidate.task_count < best_task_count)) {
                    best_shortlist_core_score = candidate.core_score;
                    best_task_count = candidate.task_count;
                    best_group = candidate.group_id;
                }
            }
        }

        assignments[static_cast<std::size_t>(task_index)] = best_group;
        GroupAssignmentState& state = group_states[static_cast<std::size_t>(best_group)];
        state.predicted_cost_sum += task.predictedCost();
        ++state.task_count;
        ++state.output_family_counts[static_cast<std::size_t>(output_family)];
        ++state.output_channel_counts[task.outputChannel()];
        ++state.spatial_tile_counts[spatial_tile_id];
        total_assigned_cost += task.predictedCost();
    }

    return assignments;
}

std::vector<int> Accelerator::buildBroadcastPhaseAwareFanoutBalancedAssignments(
    const std::vector<Task>& tasks) const {
    struct GroupAssignmentState {
        double predicted_cost_sum{0.0};
        std::uint64_t task_count{0};
        std::vector<std::uint64_t> output_family_counts;
        std::unordered_map<int, std::uint64_t> output_channel_counts;
    };

    std::vector<int> task_indices(tasks.size(), 0);
    std::iota(task_indices.begin(), task_indices.end(), 0);
    std::stable_sort(task_indices.begin(), task_indices.end(), [&](int lhs, int rhs) {
        const Task& a = tasks[static_cast<std::size_t>(lhs)];
        const Task& b = tasks[static_cast<std::size_t>(rhs)];
        if (a.predictedCost() != b.predictedCost()) {
            return a.predictedCost() > b.predictedCost();
        }
        if (a.costBucketId() != b.costBucketId()) {
            return a.costBucketId() < b.costBucketId();
        }
        if (a.outputChannel() != b.outputChannel()) {
            return a.outputChannel() < b.outputChannel();
        }
        return a.id() < b.id();
    });

    std::vector<GroupAssignmentState> group_states(static_cast<std::size_t>(config_.num_groups));
    for (GroupAssignmentState& state : group_states) {
        state.output_family_counts.assign(static_cast<std::size_t>(config_.num_groups), 0U);
    }

    std::vector<int> assignments(tasks.size(), 0);
    double total_assigned_cost = 0.0;
    double max_group_cost = 0.0;

    // BroadcastPhaseAwareFanoutBalanced keeps groups aligned on output-weight fanout and phase
    // compatibility first, then uses soft predicted-cost penalties to avoid severe imbalance.
    for (int task_index : task_indices) {
        const Task& task = tasks[static_cast<std::size_t>(task_index)];
        const int output_family = task.outputChannel() % config_.num_groups;
        const bool phase_hints_available = !config_.phase_offsets.empty();
        const double predicted_cost = task.predictedCost();
        const double projected_total_cost = total_assigned_cost + predicted_cost;
        const double average_projected_cost =
            projected_total_cost / static_cast<double>(config_.num_groups);

        double best_score = -std::numeric_limits<double>::infinity();
        int best_group = 0;

        for (int group_id = 0; group_id < config_.num_groups; ++group_id) {
            const GroupAssignmentState& state = group_states[static_cast<std::size_t>(group_id)];

            std::uint64_t dominant_family_count = 0U;
            int dominant_family = group_id;
            for (int family = 0; family < config_.num_groups; ++family) {
                const std::uint64_t family_count =
                    state.output_family_counts[static_cast<std::size_t>(family)];
                if (family_count > dominant_family_count) {
                    dominant_family_count = family_count;
                    dominant_family = family;
                }
            }

            const double family_match_reward =
                (state.task_count == 0U || dominant_family == output_family)
                    ? kBroadcastFanoutDominantFamilyReward
                    : 0.0;
            const double natural_group_reward =
                (group_id == output_family) ? kBroadcastFanoutNaturalGroupReward : 0.0;

            const auto output_channel_it = state.output_channel_counts.find(task.outputChannel());
            const std::uint64_t same_channel_count =
                (output_channel_it == state.output_channel_counts.end()) ? 0U : output_channel_it->second;
            const double exact_channel_reward = std::min(
                kBroadcastFanoutExactChannelRewardCap,
                static_cast<double>(same_channel_count) * kBroadcastFanoutExactChannelRewardScale);

            double phase_match_reward = 0.0;
            if (phase_hints_available) {
                const int group_phase_hint = config_.phase_offsets[static_cast<std::size_t>(group_id) %
                                                                   config_.phase_offsets.size()];
                if (group_phase_hint == task.phaseAffinityHint()) {
                    phase_match_reward = kBroadcastFanoutPhaseMatchReward;
                }
            }

            const double cost_norm =
                (max_group_cost > 0.0) ? (state.predicted_cost_sum / max_group_cost) : 0.0;
            const double projected_group_cost = state.predicted_cost_sum + predicted_cost;
            const double projected_deviation_norm =
                (average_projected_cost > 0.0)
                    ? std::abs(projected_group_cost - average_projected_cost) / average_projected_cost
                    : 0.0;

            const double score =
                family_match_reward + natural_group_reward + exact_channel_reward +
                phase_match_reward -
                kBroadcastFanoutAccumulatedCostPenalty * cost_norm -
                kBroadcastFanoutDeviationPenalty * projected_deviation_norm;

            if (score > best_score ||
                (score == best_score && state.predicted_cost_sum <
                                            group_states[static_cast<std::size_t>(best_group)]
                                                .predicted_cost_sum) ||
                (score == best_score &&
                 state.predicted_cost_sum ==
                     group_states[static_cast<std::size_t>(best_group)].predicted_cost_sum &&
                 state.task_count <
                     group_states[static_cast<std::size_t>(best_group)].task_count) ||
                (score == best_score &&
                 state.predicted_cost_sum ==
                     group_states[static_cast<std::size_t>(best_group)].predicted_cost_sum &&
                 state.task_count ==
                     group_states[static_cast<std::size_t>(best_group)].task_count &&
                 group_id < best_group)) {
                best_score = score;
                best_group = group_id;
            }
        }

        assignments[static_cast<std::size_t>(task_index)] = best_group;
        GroupAssignmentState& state = group_states[static_cast<std::size_t>(best_group)];
        state.predicted_cost_sum += predicted_cost;
        ++state.task_count;
        ++state.output_family_counts[static_cast<std::size_t>(output_family)];
        ++state.output_channel_counts[task.outputChannel()];
        total_assigned_cost += predicted_cost;
        max_group_cost = std::max(max_group_cost, state.predicted_cost_sum);
    }

    return assignments;
}

int Accelerator::preferredCostBucketForGroup(int group_id) const {
    if (config_.num_groups <= 0) {
        return 0;
    }
    return (group_id * kEtaCostBuckets) / config_.num_groups;
}

int Accelerator::initialGroupForTask(const Task& task) const {
    switch (config_.grouping_policy) {
    case GroupingPolicy::TaskRoundRobin:
        return task.id() % config_.num_groups;
    case GroupingPolicy::OutputChannelModulo:
    case GroupingPolicy::ETAwareCostBalanced:
    case GroupingPolicy::ETAwareCostBalancedMemoryAware:
    case GroupingPolicy::BroadcastPhaseAwareFanoutBalanced:
        return task.outputChannel() % config_.num_groups;
    }
    throw std::invalid_argument("Unhandled grouping policy in initialGroupForTask.");
}

void Accelerator::enqueueLayerTasks(std::vector<Task> tasks, const ConvLayer& layer) {
    initializeTaskSchedulingMetadata(tasks, layer);
    assignTaskCostBuckets(tasks);
    scheduler_.setBucketReferenceCosts(initial_bucket_reference_costs_);
    std::vector<int> group_phase_hints(static_cast<std::size_t>(config_.num_groups), 0);
    for (int group_id = 0; group_id < config_.num_groups; ++group_id) {
        group_phase_hints[static_cast<std::size_t>(group_id)] =
            config_.phase_offsets.empty()
                ? (group_id % config_.weight_precision_bits)
                : config_.phase_offsets[static_cast<std::size_t>(group_id) %
                                        config_.phase_offsets.size()];
    }
    scheduler_.setGroupPhaseHints(std::move(group_phase_hints));

    initial_group_predicted_cost_sums_.assign(static_cast<std::size_t>(config_.num_groups), 0.0);
    initial_group_predicted_cost_square_sums_.assign(static_cast<std::size_t>(config_.num_groups),
                                                     0.0);
    initial_group_task_counts_.assign(static_cast<std::size_t>(config_.num_groups), 0);

    std::vector<int> policy_assignments;
    switch (config_.grouping_policy) {
    case GroupingPolicy::OutputChannelModulo:
    case GroupingPolicy::TaskRoundRobin:
        break;
    case GroupingPolicy::ETAwareCostBalanced:
        policy_assignments = buildEtawareBalancedAssignments(tasks);
        break;
    case GroupingPolicy::ETAwareCostBalancedMemoryAware:
        policy_assignments = buildEtawareMemoryAwareAssignments(tasks);
        break;
    case GroupingPolicy::BroadcastPhaseAwareFanoutBalanced:
        policy_assignments = buildBroadcastPhaseAwareFanoutBalancedAssignments(tasks);
        break;
    }

    for (std::size_t task_index = 0; task_index < tasks.size(); ++task_index) {
        Task& task = tasks[task_index];
        const int group_id =
            policy_assignments.empty() ? initialGroupForTask(task) : policy_assignments[task_index];
        const double predicted_cost = task.predictedCost();
        initial_group_predicted_cost_sums_[static_cast<std::size_t>(group_id)] += predicted_cost;
        initial_group_predicted_cost_square_sums_[static_cast<std::size_t>(group_id)] +=
            predicted_cost * predicted_cost;
        ++initial_group_task_counts_[static_cast<std::size_t>(group_id)];
        scheduler_.addTask(group_id, std::move(task));
    }

    std::vector<double> group_target_costs(static_cast<std::size_t>(config_.num_groups), 0.0);
    for (int group_id = 0; group_id < config_.num_groups; ++group_id) {
        const std::size_t group_index = static_cast<std::size_t>(group_id);
        const double task_count = static_cast<double>(initial_group_task_counts_[group_index]);
        group_target_costs[group_index] =
            (task_count == 0.0) ? 0.0 : initial_group_predicted_cost_sums_[group_index] / task_count;
    }
    scheduler_.setGroupTargetCosts(std::move(group_target_costs));
}

bool Accelerator::anyLaneBusy() const {
    for (const PEGroup& group : groups_) {
        for (const Lane& lane : group.lanes()) {
            if (lane.isBusy()) {
                return true;
            }
        }
    }
    return false;
}
