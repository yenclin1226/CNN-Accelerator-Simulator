#include "BenchmarkSuite.h"

#include "Accelerator.h"
#include "ConvLayer.h"
#include "MNISTReader.h"
#include "ReferenceConv.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace {

constexpr const char* kDefaultDatasetPath = "mnist_test.csv";
constexpr const char* kDefaultWeightsPath = "training/conv1_weight.txt";
constexpr const char* kDefaultBiasPath = "training/conv1_bias.txt";

struct PreparedScenario {
    BenchmarkScenario scenario;
    ConvLayer base_layer;
    Tensor3D<std::int32_t> reference_post_relu;
};

struct RecordKey {
    std::string scenario_id;
    GroupingPolicy grouping_policy{GroupingPolicy::OutputChannelModulo};
    bool early_termination_enabled{false};

    bool operator<(const RecordKey& other) const {
        return std::tie(scenario_id, grouping_policy, early_termination_enabled) <
               std::tie(other.scenario_id, other.grouping_policy, other.early_termination_enabled);
    }
};

std::string formatDouble(double value, int precision = 6) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}

std::string boolName(bool value) {
    return value ? "On" : "Off";
}

std::string zeroRunOrderModeName(ZeroRunOrderMode mode) {
    switch (mode) {
    case ZeroRunOrderMode::ExecutionOrder:
        return "execution";
    case ZeroRunOrderMode::KernelOrder:
        return "kernel";
    }
    throw std::invalid_argument("Unhandled ZeroRunOrderMode.");
}

std::vector<GroupingPolicy> allGroupingPolicies() {
    return {
        GroupingPolicy::OutputChannelModulo,
        GroupingPolicy::TaskRoundRobin,
        GroupingPolicy::ETAwareCostBalanced,
        GroupingPolicy::ETAwareCostBalancedMemoryAware,
        GroupingPolicy::BroadcastPhaseAwareFanoutBalanced,
    };
}

AcceleratorConfig makeDefaultAcceleratorConfig() {
    AcceleratorConfig config;
    config.num_groups = 6;
    config.lanes_per_group = 6;
    config.weight_precision_bits = 8;
    config.phase_offsets = {0, 1, 2, 3, 4, 5};
    config.execution_mode = ExecutionMode::Int8BitSerial;
    config.grouping_policy = GroupingPolicy::OutputChannelModulo;
    config.broadcast_mode = BroadcastMode::DemandDriven;
    config.enable_activation_reuse = true;
    config.enable_weight_reuse = true;
    config.local_buffer_capacity_entries = 128;
    config.global_buffer_capacity_entries = 1024;
    config.enable_msb_first = true;
    config.enable_importance_ordering = true;
    config.max_cycles = 50'000'000;
    config.memory.dram_latency_cycles = 16;
    config.memory.dram_bandwidth_bytes_per_cycle = 64;
    config.memory.global_buffer_latency_cycles = 4;
    config.memory.global_buffer_bandwidth_bytes_per_cycle = 128;
    config.memory.enable_local_buffer = true;
    config.memory.local_buffer_latency_cycles = 1;
    config.memory.local_buffer_bandwidth_bytes_per_cycle = 64;
    return config;
}

ConvLayerConfig makeMnistLayerConfig() {
    ConvLayerConfig config;
    config.input_height = 28;
    config.input_width = 28;
    config.kernel_size = 3;
    config.stride = 1;
    config.padding = 1;
    config.input_channels = 1;
    config.output_channels = 8;
    config.use_bias = true;
    return config;
}

ConvLayerConfig makeSyntheticLayerConfig(int size, int input_channels, int output_channels) {
    ConvLayerConfig config;
    config.input_height = size;
    config.input_width = size;
    config.kernel_size = 3;
    config.stride = 1;
    config.padding = 1;
    config.input_channels = input_channels;
    config.output_channels = output_channels;
    config.use_bias = true;
    return config;
}

AcceleratorConfig makeMemoryPressureAcceleratorConfig() {
    AcceleratorConfig config = makeDefaultAcceleratorConfig();
    config.local_buffer_capacity_entries = 32;
    config.global_buffer_capacity_entries = 256;
    config.memory.dram_latency_cycles = 32;
    config.memory.dram_bandwidth_bytes_per_cycle = 32;
    config.memory.global_buffer_latency_cycles = 8;
    config.memory.global_buffer_bandwidth_bytes_per_cycle = 64;
    config.memory.local_buffer_latency_cycles = 2;
    config.memory.local_buffer_bandwidth_bytes_per_cycle = 32;
    return config;
}

BenchmarkScenario makeMnistScenario(BenchmarkSuiteTier tier,
                                    int sample_index,
                                    const std::string& name,
                                    const BenchmarkRunOptions& options) {
    BenchmarkScenario scenario;
    scenario.id = benchmarkSuiteTierName(tier) + "-mnist-sample-" + std::to_string(sample_index);
    scenario.name = name;
    scenario.profile_name = "mnist";
    scenario.suite_tier = tier;
    scenario.source = WorkloadSource::MnistSample;
    scenario.layer_config = makeMnistLayerConfig();
    scenario.accelerator_config = makeDefaultAcceleratorConfig();
    scenario.dataset_path = options.dataset_path;
    scenario.weight_path = options.weights_path;
    scenario.bias_path = options.bias_path;
    scenario.mnist_sample_index = sample_index;
    return scenario;
}

BenchmarkScenario makeSyntheticScenario(BenchmarkSuiteTier tier,
                                        const std::string& id_suffix,
                                        const std::string& name,
                                        const ConvLayerConfig& layer_config,
                                        const AcceleratorConfig& accelerator_config,
                                        RandomDataMode random_mode,
                                        std::uint32_t seed) {
    BenchmarkScenario scenario;
    scenario.id = benchmarkSuiteTierName(tier) + "-synthetic-" + id_suffix;
    scenario.name = name;
    scenario.profile_name = id_suffix;
    scenario.suite_tier = tier;
    scenario.source = WorkloadSource::SyntheticRandom;
    scenario.layer_config = layer_config;
    scenario.accelerator_config = accelerator_config;
    scenario.random_mode = random_mode;
    scenario.seed = seed;
    return scenario;
}

AcceleratorConfig makeExperimentConfig(const AcceleratorConfig& base,
                                       GroupingPolicy grouping_policy,
                                       bool early_termination_enabled,
                                       const BenchmarkRunOptions& options) {
    AcceleratorConfig config = base;
    config.grouping_policy = grouping_policy;
    config.enable_early_termination = early_termination_enabled;
    config.pipeline_mode = early_termination_enabled
                               ? PipelineMode::FusedConvEarlyTerminationRelu
                               : PipelineMode::BaselineConvRelu;
    config.enable_reactive_zero_skip = options.enable_reactive_zero_skip;
    config.enable_proactive_zero_run_skip = options.enable_proactive_zero_run_skip;
    config.zero_run_order_mode = options.zero_run_order_mode;
    config.enable_bit_column_skip = options.enable_bit_column_skip;
    return config;
}

PreparedScenario prepareScenario(const BenchmarkScenario& scenario) {
    BenchmarkScenario resolved = scenario;
    ConvLayer layer(scenario.layer_config);

    if (scenario.source == WorkloadSource::MnistSample) {
        const MNISTSample sample = readMNISTSample(scenario.dataset_path, scenario.mnist_sample_index);
        resolved.mnist_label = sample.label;
        layer.loadInputFromMNISTRow(sample.pixels);
        layer.loadQuantizedWeights(
            readQuantizedConvWeights(scenario.weight_path,
                                     scenario.layer_config.output_channels,
                                     scenario.layer_config.input_channels,
                                     scenario.layer_config.kernel_size));
        layer.loadQuantizedBias(
            readQuantizedConvBias(scenario.bias_path, scenario.layer_config.output_channels));
    } else {
        layer.randomizeData(scenario.seed, scenario.random_mode);
    }

    PreparedScenario prepared{
        resolved,
        layer,
        applyRelu(runReferenceConvolution(layer)),
    };
    return prepared;
}

BenchmarkRecord runSingleRecord(const PreparedScenario& prepared,
                                const BenchmarkRunOptions& options,
                                GroupingPolicy grouping_policy,
                                bool early_termination_enabled,
                                std::size_t repetition_index) {
    ConvLayer working_layer = prepared.base_layer;
    const AcceleratorConfig config = makeExperimentConfig(
        prepared.scenario.accelerator_config, grouping_policy, early_termination_enabled, options);
    Accelerator accelerator(config);

    const auto start = std::chrono::steady_clock::now();
    SimulationStats stats = accelerator.run(working_layer);
    const auto end = std::chrono::steady_clock::now();

    BenchmarkRecord record;
    record.scenario = prepared.scenario;
    record.grouping_policy = grouping_policy;
    record.early_termination_enabled = early_termination_enabled;
    record.repetition_index = repetition_index;
    record.wall_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    record.stats = std::move(stats);
    record.error_metrics =
        computeErrorMetrics(prepared.reference_post_relu, working_layer.outputTensor());
    return record;
}

std::vector<bool> etStatesForMode(BenchmarkEtMode et_mode) {
    switch (et_mode) {
    case BenchmarkEtMode::Paired:
        return {false, true};
    case BenchmarkEtMode::Off:
        return {false};
    case BenchmarkEtMode::On:
        return {true};
    }
    throw std::invalid_argument("Unhandled BenchmarkEtMode.");
}

double meanOf(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    return std::accumulate(values.begin(), values.end(), 0.0) /
           static_cast<double>(values.size());
}

std::uint64_t roundedMeanMetric(
    const std::vector<const BenchmarkRecord*>& records,
    std::uint64_t (*accessor)(const BenchmarkRecord&)) {
    std::vector<double> values;
    values.reserve(records.size());
    for (const BenchmarkRecord* record : records) {
        values.push_back(static_cast<double>(accessor(*record)));
    }
    return static_cast<std::uint64_t>(std::llround(meanOf(values)));
}

BenchmarkAggregateRow aggregateGroup(const std::vector<const BenchmarkRecord*>& records) {
    const BenchmarkRecord& first = *records.front();
    BenchmarkAggregateRow row;
    row.scenario_id = first.scenario.id;
    row.scenario_name = first.scenario.name;
    row.suite_tier = first.scenario.suite_tier;
    row.source = first.scenario.source;
    row.mnist_label = first.scenario.mnist_label;
    row.grouping_policy = first.grouping_policy;
    row.early_termination_enabled = first.early_termination_enabled;
    row.repetitions = records.size();

    std::vector<double> wall_times;
    std::vector<double> cycles;
    std::vector<double> throughputs;
    std::vector<double> occupancies;
    std::vector<double> multiplier_utils;
    std::vector<double> accumulator_utils;
    std::vector<double> memory_stalls;
    std::vector<double> broadcast_stalls;
    std::vector<double> processed_fractions;
    std::vector<double> fairness_values;
    std::vector<double> completed_variances;
    std::vector<double> active_cycle_variances;
    std::vector<double> estimated_cycles_saved;
    std::vector<double> skipped_macs_per_terminated_task;

    for (const BenchmarkRecord* record : records) {
        row.exact_match = row.exact_match && (record->error_metrics.total_mismatches == 0);
        row.total_mismatches += record->error_metrics.total_mismatches;
        row.max_absolute_error = std::max(row.max_absolute_error,
                                          record->error_metrics.max_absolute_error);
        row.mean_absolute_error += record->error_metrics.mean_absolute_error;
        row.mean_relative_error += record->error_metrics.mean_relative_error;

        wall_times.push_back(record->wall_time_ms);
        cycles.push_back(static_cast<double>(record->stats.total_cycles));
        throughputs.push_back(record->stats.throughput_macs_per_cycle);
        occupancies.push_back(record->stats.lane_occupancy);
        multiplier_utils.push_back(record->stats.multiplier_utilization);
        accumulator_utils.push_back(record->stats.accumulator_utilization);
        memory_stalls.push_back(record->stats.memory_stall_ratio);
        broadcast_stalls.push_back(record->stats.broadcast_stall_ratio);
        processed_fractions.push_back(record->stats.average_processed_fraction_per_task);

        const BenchmarkSecondaryMetrics secondary = deriveSecondaryMetrics(record->stats);
        fairness_values.push_back(secondary.lane_workload_jain_fairness);
        completed_variances.push_back(secondary.variance_group_completed_tasks);
        active_cycle_variances.push_back(secondary.variance_group_active_lane_cycles);
        estimated_cycles_saved.push_back(
            static_cast<double>(secondary.estimated_cycles_saved_early_termination));
        skipped_macs_per_terminated_task.push_back(
            secondary.mean_skipped_macs_per_terminated_task);
    }

    row.wall_time_ms = computeSummaryStats(wall_times);
    row.total_cycles = computeSummaryStats(cycles);
    row.throughput_macs_per_cycle = computeSummaryStats(throughputs);
    row.lane_occupancy = meanOf(occupancies);
    row.multiplier_utilization = meanOf(multiplier_utils);
    row.accumulator_utilization = meanOf(accumulator_utils);
    row.memory_stall_ratio = meanOf(memory_stalls);
    row.broadcast_stall_ratio = meanOf(broadcast_stalls);
    row.average_processed_fraction_per_task = meanOf(processed_fractions);

    row.dram_bytes = roundedMeanMetric(records, [](const BenchmarkRecord& record) {
        return record.stats.dram_bytes;
    });
    row.onchip_buffer_bytes = roundedMeanMetric(records, [](const BenchmarkRecord& record) {
        return record.stats.onchip_buffer_bytes;
    });
    row.activation_reuse_hits = roundedMeanMetric(records, [](const BenchmarkRecord& record) {
        return record.stats.activation_reuse_hits;
    });
    row.activation_reuse_misses = roundedMeanMetric(records, [](const BenchmarkRecord& record) {
        return record.stats.activation_reuse_misses;
    });
    row.weight_reuse_hits = roundedMeanMetric(records, [](const BenchmarkRecord& record) {
        return record.stats.weight_reuse_hits;
    });
    row.weight_reuse_misses = roundedMeanMetric(records, [](const BenchmarkRecord& record) {
        return record.stats.weight_reuse_misses;
    });
    row.memory_requests_avoided = roundedMeanMetric(records, [](const BenchmarkRecord& record) {
        return record.stats.memory_requests_avoided;
    });
    row.work_stealing_events = roundedMeanMetric(records, [](const BenchmarkRecord& record) {
        return record.stats.work_stealing_events;
    });
    row.tasks_terminated_early = roundedMeanMetric(records, [](const BenchmarkRecord& record) {
        return record.stats.tasks_terminated_early;
    });
    row.output_elements_terminated_early =
        roundedMeanMetric(records, [](const BenchmarkRecord& record) {
            return record.stats.output_elements_terminated_early;
        });
    row.macs_skipped_total = roundedMeanMetric(records, [](const BenchmarkRecord& record) {
        return record.stats.macs_skipped_total;
    });
    row.macs_skipped_et_only = roundedMeanMetric(records, [](const BenchmarkRecord& record) {
        return record.stats.macs_skipped_et_only;
    });
    row.macs_skipped_reactive_only = roundedMeanMetric(records, [](const BenchmarkRecord& record) {
        return record.stats.macs_skipped_reactive_only;
    });
    row.macs_skipped_proactive_only = roundedMeanMetric(records, [](const BenchmarkRecord& record) {
        return record.stats.macs_skipped_proactive_only;
    });
    row.macs_skipped_zero_only = roundedMeanMetric(records, [](const BenchmarkRecord& record) {
        return record.stats.macs_skipped_zero_only;
    });
    row.bit_steps_skipped_total = roundedMeanMetric(records, [](const BenchmarkRecord& record) {
        return record.stats.bit_steps_skipped_total;
    });
    row.bit_steps_skipped_et_only = roundedMeanMetric(records, [](const BenchmarkRecord& record) {
        return record.stats.bit_steps_skipped_et_only;
    });
    row.bit_steps_skipped_bit_column_only =
        roundedMeanMetric(records, [](const BenchmarkRecord& record) {
            return record.stats.bit_steps_skipped_bit_column_only;
        });
    row.zero_run_events = roundedMeanMetric(records, [](const BenchmarkRecord& record) {
        return record.stats.zero_run_events;
    });
    row.macs_skipped = row.macs_skipped_total;
    row.bit_steps_skipped = row.bit_steps_skipped_total;

    row.secondary.lane_workload_jain_fairness = meanOf(fairness_values);
    row.secondary.variance_group_completed_tasks = meanOf(completed_variances);
    row.secondary.variance_group_active_lane_cycles = meanOf(active_cycle_variances);
    row.secondary.estimated_cycles_saved_early_termination =
        static_cast<std::uint64_t>(std::llround(meanOf(estimated_cycles_saved)));
    row.secondary.processed_fraction_percentiles = computePercentiles(processed_fractions);
    row.secondary.mean_skipped_macs_per_terminated_task = meanOf(skipped_macs_per_terminated_task);

    row.mean_absolute_error /= static_cast<double>(records.size());
    row.mean_relative_error /= static_cast<double>(records.size());
    return row;
}

std::vector<BenchmarkAggregateRow> buildAggregateRows(const std::vector<BenchmarkRecord>& records) {
    std::map<RecordKey, std::vector<const BenchmarkRecord*>> grouped;
    for (const BenchmarkRecord& record : records) {
        grouped[RecordKey{record.scenario.id, record.grouping_policy, record.early_termination_enabled}]
            .push_back(&record);
    }

    std::vector<BenchmarkAggregateRow> rows;
    rows.reserve(grouped.size());
    for (const auto& entry : grouped) {
        rows.push_back(aggregateGroup(entry.second));
    }

    std::sort(rows.begin(), rows.end(), [](const BenchmarkAggregateRow& lhs,
                                           const BenchmarkAggregateRow& rhs) {
        return std::tie(lhs.scenario_id, lhs.grouping_policy, lhs.early_termination_enabled) <
               std::tie(rhs.scenario_id, rhs.grouping_policy, rhs.early_termination_enabled);
    });
    return rows;
}

std::vector<BenchmarkComparisonRow> buildComparisonRows(
    const std::vector<BenchmarkAggregateRow>& aggregate_rows) {
    std::map<std::pair<std::string, GroupingPolicy>, const BenchmarkAggregateRow*> off_rows;
    std::map<std::pair<std::string, GroupingPolicy>, const BenchmarkAggregateRow*> on_rows;
    for (const BenchmarkAggregateRow& row : aggregate_rows) {
        const auto key = std::make_pair(row.scenario_id, row.grouping_policy);
        if (row.early_termination_enabled) {
            on_rows[key] = &row;
        } else {
            off_rows[key] = &row;
        }
    }

    std::vector<BenchmarkComparisonRow> rows;
    for (const auto& entry : off_rows) {
        const auto it = on_rows.find(entry.first);
        if (it == on_rows.end()) {
            continue;
        }

        const BenchmarkAggregateRow& off = *entry.second;
        const BenchmarkAggregateRow& on = *it->second;
        BenchmarkComparisonRow row;
        row.scenario_id = off.scenario_id;
        row.scenario_name = off.scenario_name;
        row.suite_tier = off.suite_tier;
        row.source = off.source;
        row.mnist_label = off.mnist_label;
        row.grouping_policy = off.grouping_policy;
        row.cycles_et_off = off.total_cycles.mean;
        row.cycles_et_on = on.total_cycles.mean;
        row.cycle_delta = off.total_cycles.mean - on.total_cycles.mean;
        row.cycle_speedup = on.total_cycles.mean > 0.0
                                ? off.total_cycles.mean / on.total_cycles.mean
                                : 0.0;
        row.wall_time_et_off_ms = off.wall_time_ms.median;
        row.wall_time_et_on_ms = on.wall_time_ms.median;
        row.wall_time_speedup = on.wall_time_ms.median > 0.0
                                    ? off.wall_time_ms.median / on.wall_time_ms.median
                                    : 0.0;
        row.tasks_terminated_early = on.tasks_terminated_early;
        row.output_elements_terminated_early = on.output_elements_terminated_early;
        row.macs_skipped_total = on.macs_skipped_total;
        row.macs_skipped_et_only = on.macs_skipped_et_only;
        row.macs_skipped_reactive_only = on.macs_skipped_reactive_only;
        row.macs_skipped_proactive_only = on.macs_skipped_proactive_only;
        row.macs_skipped_zero_only = on.macs_skipped_zero_only;
        row.bit_steps_skipped_total = on.bit_steps_skipped_total;
        row.bit_steps_skipped_et_only = on.bit_steps_skipped_et_only;
        row.bit_steps_skipped_bit_column_only = on.bit_steps_skipped_bit_column_only;
        row.zero_run_events = on.zero_run_events;
        row.macs_skipped = row.macs_skipped_total;
        row.bit_steps_skipped = row.bit_steps_skipped_total;
        row.exact_match = off.exact_match && on.exact_match;
        row.cycle_reduction_observed = row.cycles_et_on < row.cycles_et_off;
        row.math_consistent =
            (off.tasks_terminated_early == 0U) &&
            (off.output_elements_terminated_early == 0U) &&
            (off.macs_skipped_et_only == 0U) &&
            (off.bit_steps_skipped_et_only == 0U) &&
            (std::abs(row.cycle_delta - (row.cycles_et_off - row.cycles_et_on)) < 1e-9) &&
            (row.cycles_et_on <= 0.0 ||
             std::abs(row.cycle_speedup - (row.cycles_et_off / row.cycles_et_on)) < 1e-9);
        rows.push_back(row);
    }

    std::sort(rows.begin(), rows.end(), [](const BenchmarkComparisonRow& lhs,
                                           const BenchmarkComparisonRow& rhs) {
        return std::tie(lhs.scenario_id, lhs.cycles_et_on, lhs.grouping_policy) <
               std::tie(rhs.scenario_id, rhs.cycles_et_on, rhs.grouping_policy);
    });
    return rows;
}

bool hasSelectedScenarioFilter(const BenchmarkRunOptions& options) {
    return !options.selected_scenario_ids.empty();
}

std::vector<BenchmarkScenario> filterSelectedScenarios(const BenchmarkRunOptions& options,
                                                       std::vector<BenchmarkScenario> scenarios) {
    std::vector<BenchmarkScenario> filtered = std::move(scenarios);
    if (!options.synthetic_profile_filters.empty()) {
        std::vector<BenchmarkScenario> source_filtered;
        for (const BenchmarkScenario& scenario : filtered) {
            if (scenario.source != WorkloadSource::SyntheticRandom) {
                source_filtered.push_back(scenario);
                continue;
            }

            for (const std::string& filter : options.synthetic_profile_filters) {
                if (scenario.profile_name.find(filter) != std::string::npos) {
                    source_filtered.push_back(scenario);
                    break;
                }
            }
        }
        filtered = std::move(source_filtered);
    }

    if (hasSelectedScenarioFilter(options)) {
        const std::set<std::string> selected_ids(options.selected_scenario_ids.begin(),
                                                 options.selected_scenario_ids.end());
        std::vector<BenchmarkScenario> selected_filtered;
        for (const BenchmarkScenario& scenario : filtered) {
            if (selected_ids.count(scenario.id) != 0U) {
                selected_filtered.push_back(scenario);
            }
        }
        filtered = std::move(selected_filtered);
    }

    if (filtered.empty()) {
        throw std::invalid_argument("No benchmark scenarios matched the selected filters.");
    }
    return filtered;
}

bool tierRequested(const BenchmarkRunOptions& options, BenchmarkSuiteTier tier) {
    return std::find(options.suite_tiers.begin(), options.suite_tiers.end(), tier) !=
           options.suite_tiers.end();
}

void appendQuickScenarios(const BenchmarkRunOptions& options,
                          std::vector<BenchmarkScenario>& scenarios) {
    const BenchmarkSuiteTier tier = BenchmarkSuiteTier::Quick;

    scenarios.push_back(makeMnistScenario(tier, 0, "MNIST sample 0", options));
    scenarios.push_back(makeMnistScenario(tier, 2827, "MNIST sparse sample 2827", options));
    scenarios.push_back(makeMnistScenario(tier, 1038, "MNIST sparse sample 1038", options));
    scenarios.push_back(makeMnistScenario(tier, 2462, "MNIST dense sample 2462", options));
    scenarios.push_back(makeMnistScenario(tier, 3768, "MNIST dense sample 3768", options));

    const ConvLayerConfig synthetic_layer = makeSyntheticLayerConfig(16, 8, 8);
    const AcceleratorConfig synthetic_accelerator = makeDefaultAcceleratorConfig();
    scenarios.push_back(makeSyntheticScenario(
        tier,
        "uniform-seed-2026",
        "Synthetic uniform seed 2026",
        synthetic_layer,
        synthetic_accelerator,
        RandomDataMode::UniformSymmetric,
        2026));
    scenarios.push_back(makeSyntheticScenario(
        tier,
        "sparse-seed-2026",
        "Synthetic sparse seed 2026",
        synthetic_layer,
        synthetic_accelerator,
        RandomDataMode::SparseActivations,
        2026));
    scenarios.push_back(makeSyntheticScenario(
        tier,
        "negative-bias-seed-2026",
        "Synthetic negative-bias seed 2026",
        synthetic_layer,
        synthetic_accelerator,
        RandomDataMode::NegativeBias,
        2026));
}

void appendExtendedScenarios(const BenchmarkRunOptions& options,
                             std::vector<BenchmarkScenario>& scenarios) {
    const BenchmarkSuiteTier tier = BenchmarkSuiteTier::Extended;
    const std::vector<int> per_digit_indices = {3, 2, 1, 18, 4, 8, 11, 0, 61, 7};
    for (int sample_index : per_digit_indices) {
        scenarios.push_back(makeMnistScenario(
            tier,
            sample_index,
            "MNIST digit representative sample " + std::to_string(sample_index),
            options));
    }

    for (int sample_index : {2827, 1038, 2462, 3768}) {
        scenarios.push_back(makeMnistScenario(
            tier,
            sample_index,
            "MNIST density profile sample " + std::to_string(sample_index),
            options));
    }

    const ConvLayerConfig default_layer = makeSyntheticLayerConfig(16, 8, 8);
    const ConvLayerConfig large_layer = makeSyntheticLayerConfig(32, 16, 16);
    const AcceleratorConfig default_accelerator = makeDefaultAcceleratorConfig();
    const AcceleratorConfig pressure_accelerator = makeMemoryPressureAcceleratorConfig();

    for (std::uint32_t seed : {1U, 7U, 13U, 29U, 42U}) {
        scenarios.push_back(makeSyntheticScenario(
            tier,
            "default-uniform-seed-" + std::to_string(seed),
            "Extended synthetic default uniform seed " + std::to_string(seed),
            default_layer,
            default_accelerator,
            RandomDataMode::UniformSymmetric,
            seed));
    }

    for (std::uint32_t seed : {1U, 7U, 13U}) {
        scenarios.push_back(makeSyntheticScenario(
            tier,
            "large-uniform-seed-" + std::to_string(seed),
            "Extended synthetic large uniform seed " + std::to_string(seed),
            large_layer,
            default_accelerator,
            RandomDataMode::UniformSymmetric,
            seed));
        scenarios.push_back(makeSyntheticScenario(
            tier,
            "memory-pressure-uniform-seed-" + std::to_string(seed),
            "Extended synthetic memory-pressure seed " + std::to_string(seed),
            large_layer,
            pressure_accelerator,
            RandomDataMode::UniformSymmetric,
            seed));
    }
}

void validateOptions(const BenchmarkRunOptions& options,
                     const std::vector<BenchmarkScenario>& scenarios) {
    if (options.suite_tiers.empty()) {
        throw std::invalid_argument("At least one benchmark suite tier must be selected.");
    }
    if (options.timed_repetitions == 0U) {
        throw std::invalid_argument("Timed repetitions must be at least 1.");
    }
    if (options.mode == BenchmarkRunMode::Compare && options.et_mode != BenchmarkEtMode::Paired) {
        throw std::invalid_argument("Compare mode requires --et paired.");
    }
    if (scenarios.empty()) {
        throw std::invalid_argument("The selected benchmark configuration produced no scenarios.");
    }

    const bool has_mnist = std::any_of(scenarios.begin(), scenarios.end(), [](const BenchmarkScenario& s) {
        return s.source == WorkloadSource::MnistSample;
    });
    const bool has_synthetic = std::any_of(scenarios.begin(), scenarios.end(), [](const BenchmarkScenario& s) {
        return s.source == WorkloadSource::SyntheticRandom;
    });
    const bool custom_dataset_paths =
        options.dataset_path != kDefaultDatasetPath ||
        options.weights_path != kDefaultWeightsPath ||
        options.bias_path != kDefaultBiasPath;
    if (!has_mnist && custom_dataset_paths) {
        throw std::invalid_argument(
            "Dataset/weight/bias path overrides are only valid when MNIST-backed scenarios are selected.");
    }
    if (!has_synthetic && !options.synthetic_profile_filters.empty()) {
        throw std::invalid_argument(
            "Synthetic profile filters are only valid when synthetic scenarios are selected.");
    }
}

std::string scenarioDescriptor(const BenchmarkAggregateRow& row) {
    std::ostringstream stream;
    stream << row.scenario_id;
    if (row.mnist_label >= 0) {
        stream << " (label=" << row.mnist_label << ")";
    }
    return stream.str();
}

std::string scenarioDescriptor(const BenchmarkComparisonRow& row) {
    std::ostringstream stream;
    stream << row.scenario_id;
    if (row.mnist_label >= 0) {
        stream << " (label=" << row.mnist_label << ")";
    }
    return stream.str();
}

void writeSectionHeader(std::ostream& out, const std::string& title) {
    out << title << "\n";
    out << std::string(title.size(), '=') << "\n";
}

template <typename Row, typename Accessor>
std::vector<Row> filterRows(const std::vector<Row>& rows, Accessor accessor) {
    std::vector<Row> filtered;
    for (const Row& row : rows) {
        if (accessor(row)) {
            filtered.push_back(row);
        }
    }
    return filtered;
}

void writeConfigurationSection(std::ostream& out, const BenchmarkSuiteResult& result) {
    writeSectionHeader(out, "Section 1: Benchmark Configuration");
    out << "mode=" << benchmarkRunModeName(result.options.mode) << "\n";
    out << "suite_tiers=";
    for (std::size_t index = 0; index < result.options.suite_tiers.size(); ++index) {
        if (index != 0U) {
            out << ",";
        }
        out << benchmarkSuiteTierName(result.options.suite_tiers[index]);
    }
    out << "\n";
    out << "et_mode=" << benchmarkEtModeName(result.options.et_mode) << "\n";
    out << "reactive_zero_skip=" << boolName(result.options.enable_reactive_zero_skip) << "\n";
    out << "proactive_zero_run_skip=" << boolName(result.options.enable_proactive_zero_run_skip)
        << "\n";
    out << "zero_run_order_mode=" << zeroRunOrderModeName(result.options.zero_run_order_mode)
        << "\n";
    out << "bit_column_skip=" << boolName(result.options.enable_bit_column_skip) << "\n";
    out << "warmup_iterations=" << result.options.warmup_iterations << "\n";
    out << "timed_repetitions=" << result.options.timed_repetitions << "\n";
    out << "grouping_policies=";
    for (std::size_t index = 0; index < result.options.grouping_policies.size(); ++index) {
        if (index != 0U) {
            out << ",";
        }
        out << groupingPolicyName(result.options.grouping_policies[index]);
    }
    out << "\n";
    out << "scenario_count=" << result.scenarios.size() << "\n";
    out << "exactness_pass=" << (result.exactness_pass ? "PASS" : "FAIL") << "\n";
    out << "primary_invariants_pass=" << (result.primary_invariants_pass ? "PASS" : "FAIL")
        << "\n\n";
}

void writeSpeedSection(std::ostream& out, const BenchmarkSuiteResult& result) {
    writeSectionHeader(out, "Section 2: Speed Summary");
    if (!result.comparison_rows.empty()) {
        out << "scenario | grouping_policy | cycles_et_off | cycles_et_on | cycle_speedup | "
               "wall_median_et_off_ms | wall_median_et_on_ms | wall_speedup | et_activity | "
               "cycle_reduction_observed\n";
        for (const BenchmarkComparisonRow& row : result.comparison_rows) {
            out << scenarioDescriptor(row) << " | " << groupingPolicyName(row.grouping_policy)
                << " | " << formatDouble(row.cycles_et_off, 3)
                << " | " << formatDouble(row.cycles_et_on, 3)
                << " | " << formatDouble(row.cycle_speedup, 6)
                << " | " << formatDouble(row.wall_time_et_off_ms, 6)
                << " | " << formatDouble(row.wall_time_et_on_ms, 6)
                << " | " << formatDouble(row.wall_time_speedup, 6)
                << " | " << row.tasks_terminated_early
                << " | " << (row.cycle_reduction_observed ? "yes" : "no") << "\n";
        }
    } else {
        out << "scenario | grouping_policy | et_enabled | cycles_mean | cycles_median | "
               "wall_median_ms | wall_min_ms | wall_max_ms | throughput_macs_per_cycle\n";
        for (const BenchmarkAggregateRow& row : result.aggregate_rows) {
            out << scenarioDescriptor(row) << " | " << groupingPolicyName(row.grouping_policy)
                << " | " << boolName(row.early_termination_enabled)
                << " | " << formatDouble(row.total_cycles.mean, 3)
                << " | " << formatDouble(row.total_cycles.median, 3)
                << " | " << formatDouble(row.wall_time_ms.median, 6)
                << " | " << formatDouble(row.wall_time_ms.min, 6)
                << " | " << formatDouble(row.wall_time_ms.max, 6)
                << " | " << formatDouble(row.throughput_macs_per_cycle.mean, 6) << "\n";
        }
    }
    out << "\n";
}

void writeDetailedSection(std::ostream& out, const BenchmarkSuiteResult& result) {
    writeSectionHeader(out, "Section 2: Detailed Primary Metrics");
    out << "scenario | grouping_policy | et_enabled | cycles_mean | wall_median_ms | "
           "throughput_macs_per_cycle | lane_occupancy | multiplier_utilization | "
           "accumulator_utilization | memory_stall_ratio | broadcast_stall_ratio | dram_bytes | "
           "onchip_buffer_bytes | activation_reuse_hits | activation_reuse_misses | "
           "weight_reuse_hits | weight_reuse_misses | memory_requests_avoided | "
           "work_stealing_events | tasks_terminated_early | output_elements_terminated_early | "
           "macs_skipped_total | macs_skipped_et_only | macs_skipped_reactive_only | "
           "macs_skipped_proactive_only | macs_skipped_zero_only | "
           "bit_steps_skipped_total | bit_steps_skipped_et_only | "
           "bit_steps_skipped_bit_column_only | zero_run_events | "
           "average_processed_fraction_per_task | mismatches\n";
    for (const BenchmarkAggregateRow& row : result.aggregate_rows) {
        out << scenarioDescriptor(row) << " | " << groupingPolicyName(row.grouping_policy)
            << " | " << boolName(row.early_termination_enabled)
            << " | " << formatDouble(row.total_cycles.mean, 3)
            << " | " << formatDouble(row.wall_time_ms.median, 6)
            << " | " << formatDouble(row.throughput_macs_per_cycle.mean, 6)
            << " | " << formatDouble(row.lane_occupancy, 6)
            << " | " << formatDouble(row.multiplier_utilization, 6)
            << " | " << formatDouble(row.accumulator_utilization, 6)
            << " | " << formatDouble(row.memory_stall_ratio, 6)
            << " | " << formatDouble(row.broadcast_stall_ratio, 6)
            << " | " << row.dram_bytes
            << " | " << row.onchip_buffer_bytes
            << " | " << row.activation_reuse_hits
            << " | " << row.activation_reuse_misses
            << " | " << row.weight_reuse_hits
            << " | " << row.weight_reuse_misses
            << " | " << row.memory_requests_avoided
            << " | " << row.work_stealing_events
            << " | " << row.tasks_terminated_early
            << " | " << row.output_elements_terminated_early
            << " | " << row.macs_skipped_total
            << " | " << row.macs_skipped_et_only
            << " | " << row.macs_skipped_reactive_only
            << " | " << row.macs_skipped_proactive_only
            << " | " << row.macs_skipped_zero_only
            << " | " << row.bit_steps_skipped_total
            << " | " << row.bit_steps_skipped_et_only
            << " | " << row.bit_steps_skipped_bit_column_only
            << " | " << row.zero_run_events
            << " | " << formatDouble(row.average_processed_fraction_per_task, 6)
            << " | " << row.total_mismatches << "\n";
    }
    out << "\n";

    writeSectionHeader(out, "Section 3: Secondary Diagnostics");
    out << "scenario | grouping_policy | et_enabled | jain_fairness | "
           "variance_group_completed_tasks | variance_group_active_lane_cycles | "
           "estimated_cycles_saved_early_termination | processed_fraction_p10 | "
           "processed_fraction_p50 | processed_fraction_p90 | mean_skipped_macs_per_terminated_task\n";
    for (const BenchmarkAggregateRow& row : result.aggregate_rows) {
        out << scenarioDescriptor(row) << " | " << groupingPolicyName(row.grouping_policy)
            << " | " << boolName(row.early_termination_enabled)
            << " | " << formatDouble(row.secondary.lane_workload_jain_fairness, 6)
            << " | " << formatDouble(row.secondary.variance_group_completed_tasks, 6)
            << " | " << formatDouble(row.secondary.variance_group_active_lane_cycles, 6)
            << " | " << row.secondary.estimated_cycles_saved_early_termination
            << " | " << formatDouble(row.secondary.processed_fraction_percentiles.p10, 6)
            << " | " << formatDouble(row.secondary.processed_fraction_percentiles.p50, 6)
            << " | " << formatDouble(row.secondary.processed_fraction_percentiles.p90, 6)
            << " | " << formatDouble(row.secondary.mean_skipped_macs_per_terminated_task, 6)
            << "\n";
    }
    out << "\n";
}

void writeCompareSection(std::ostream& out, const BenchmarkSuiteResult& result) {
    writeSectionHeader(out, "Section 2: ET Comparisons");
    out << "scenario | grouping_policy | cycles_et_off | cycles_et_on | cycle_delta | "
           "cycle_speedup | wall_et_off_ms | wall_et_on_ms | wall_speedup | et_activity | "
           "macs_skipped_total | macs_skipped_et_only | macs_skipped_reactive_only | "
           "macs_skipped_proactive_only | macs_skipped_zero_only | "
           "bit_steps_skipped_total | bit_steps_skipped_et_only | "
           "bit_steps_skipped_bit_column_only | zero_run_events | math_consistent | "
           "cycle_reduction_observed\n";
    for (const BenchmarkComparisonRow& row : result.comparison_rows) {
        out << scenarioDescriptor(row) << " | " << groupingPolicyName(row.grouping_policy)
            << " | " << formatDouble(row.cycles_et_off, 3)
            << " | " << formatDouble(row.cycles_et_on, 3)
            << " | " << formatDouble(row.cycle_delta, 3)
            << " | " << formatDouble(row.cycle_speedup, 6)
            << " | " << formatDouble(row.wall_time_et_off_ms, 6)
            << " | " << formatDouble(row.wall_time_et_on_ms, 6)
            << " | " << formatDouble(row.wall_time_speedup, 6)
            << " | " << row.tasks_terminated_early
            << " | " << row.macs_skipped_total
            << " | " << row.macs_skipped_et_only
            << " | " << row.macs_skipped_reactive_only
            << " | " << row.macs_skipped_proactive_only
            << " | " << row.macs_skipped_zero_only
            << " | " << row.bit_steps_skipped_total
            << " | " << row.bit_steps_skipped_et_only
            << " | " << row.bit_steps_skipped_bit_column_only
            << " | " << row.zero_run_events
            << " | " << (row.math_consistent ? "yes" : "no")
            << " | " << (row.cycle_reduction_observed ? "yes" : "no") << "\n";
    }
    out << "\n";

    writeSectionHeader(out, "Section 3: Policy Rankings");
    std::map<std::string, std::vector<BenchmarkComparisonRow>> rows_by_scenario;
    for (const BenchmarkComparisonRow& row : result.comparison_rows) {
        rows_by_scenario[row.scenario_id].push_back(row);
    }

    for (auto& entry : rows_by_scenario) {
        std::sort(entry.second.begin(), entry.second.end(), [](const BenchmarkComparisonRow& lhs,
                                                               const BenchmarkComparisonRow& rhs) {
            return std::tie(lhs.cycles_et_on, lhs.grouping_policy) <
                   std::tie(rhs.cycles_et_on, rhs.grouping_policy);
        });
        out << entry.first << "\n";
        for (std::size_t index = 0; index < entry.second.size(); ++index) {
            const BenchmarkComparisonRow& row = entry.second[index];
            out << "  " << (index + 1U) << ". " << groupingPolicyName(row.grouping_policy)
                << " cycles_et_on=" << formatDouble(row.cycles_et_on, 3)
                << " speedup=" << formatDouble(row.cycle_speedup, 6) << "\n";
        }
    }
    out << "\n";

    writeSectionHeader(out, "Section 4: Soft Benchmark Expectations");
    const auto sample_zero_rows = filterRows(result.comparison_rows, [](const BenchmarkComparisonRow& row) {
        return row.scenario_id.find("sample-0") != std::string::npos;
    });
    if (sample_zero_rows.empty()) {
        out << "No sample-0 comparisons were included in this run.\n\n";
        return;
    }

    for (const BenchmarkComparisonRow& row : sample_zero_rows) {
        out << scenarioDescriptor(row) << " | " << groupingPolicyName(row.grouping_policy)
            << " | et_activity=" << row.tasks_terminated_early
            << " | cycle_reduction_observed=" << (row.cycle_reduction_observed ? "yes" : "no")
            << "\n";
    }
    out << "\n";
}

}  // namespace

std::string benchmarkSuiteTierName(BenchmarkSuiteTier tier) {
    switch (tier) {
    case BenchmarkSuiteTier::Quick:
        return "quick";
    case BenchmarkSuiteTier::Extended:
        return "extended";
    }
    throw std::invalid_argument("Unhandled BenchmarkSuiteTier.");
}

std::string benchmarkRunModeName(BenchmarkRunMode mode) {
    switch (mode) {
    case BenchmarkRunMode::Speed:
        return "speed";
    case BenchmarkRunMode::Detailed:
        return "detailed";
    case BenchmarkRunMode::Compare:
        return "compare";
    }
    throw std::invalid_argument("Unhandled BenchmarkRunMode.");
}

std::string workloadSourceName(WorkloadSource source) {
    switch (source) {
    case WorkloadSource::MnistSample:
        return "mnist";
    case WorkloadSource::SyntheticRandom:
        return "synthetic";
    }
    throw std::invalid_argument("Unhandled WorkloadSource.");
}

std::string benchmarkEtModeName(BenchmarkEtMode mode) {
    switch (mode) {
    case BenchmarkEtMode::Paired:
        return "paired";
    case BenchmarkEtMode::Off:
        return "off";
    case BenchmarkEtMode::On:
        return "on";
    }
    throw std::invalid_argument("Unhandled BenchmarkEtMode.");
}

std::string groupingPolicyName(GroupingPolicy policy) {
    switch (policy) {
    case GroupingPolicy::OutputChannelModulo:
        return "OutputChannelModulo";
    case GroupingPolicy::TaskRoundRobin:
        return "TaskRoundRobin";
    case GroupingPolicy::ETAwareCostBalanced:
        return "ETAwareCostBalanced";
    case GroupingPolicy::ETAwareCostBalancedMemoryAware:
        return "ETAwareCostBalancedMemoryAware";
    case GroupingPolicy::BroadcastPhaseAwareFanoutBalanced:
        return "BroadcastPhaseAwareFanoutBalanced";
    }
    throw std::invalid_argument("Unhandled GroupingPolicy.");
}

SummaryStats computeSummaryStats(const std::vector<double>& values) {
    SummaryStats summary;
    if (values.empty()) {
        return summary;
    }

    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
    summary.mean = sum / static_cast<double>(sorted.size());
    summary.min = sorted.front();
    summary.max = sorted.back();

    const std::size_t midpoint = sorted.size() / 2U;
    summary.median =
        (sorted.size() % 2U == 0U)
            ? (sorted[midpoint - 1U] + sorted[midpoint]) / 2.0
            : sorted[midpoint];

    double variance = 0.0;
    for (double value : sorted) {
        const double delta = value - summary.mean;
        variance += delta * delta;
    }
    variance /= static_cast<double>(sorted.size());
    summary.stddev = std::sqrt(variance);
    return summary;
}

double computePopulationVariance(const std::vector<double>& values) {
    const SummaryStats summary = computeSummaryStats(values);
    return summary.stddev * summary.stddev;
}

double computeJainFairness(const std::vector<double>& values) {
    if (values.empty()) {
        return 1.0;
    }

    double sum = 0.0;
    double sum_squares = 0.0;
    for (double value : values) {
        sum += value;
        sum_squares += value * value;
    }
    if (sum_squares == 0.0) {
        return 1.0;
    }
    const double count = static_cast<double>(values.size());
    return (sum * sum) / (count * sum_squares);
}

PercentileSummary computePercentiles(const std::vector<double>& values) {
    PercentileSummary summary;
    if (values.empty()) {
        return summary;
    }

    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const auto pick = [&sorted](double percentile) {
        if (sorted.size() == 1U) {
            return sorted.front();
        }
        const double scaled_index =
            percentile * static_cast<double>(sorted.size() - 1U);
        const std::size_t lower =
            static_cast<std::size_t>(std::floor(scaled_index));
        const std::size_t upper =
            static_cast<std::size_t>(std::ceil(scaled_index));
        if (lower == upper) {
            return sorted[lower];
        }
        const double fraction = scaled_index - static_cast<double>(lower);
        return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
    };

    summary.p10 = pick(0.10);
    summary.p50 = pick(0.50);
    summary.p90 = pick(0.90);
    return summary;
}

BenchmarkSecondaryMetrics deriveSecondaryMetrics(const SimulationStats& stats) {
    BenchmarkSecondaryMetrics metrics;
    metrics.lane_workload_jain_fairness = stats.lane_workload_jain_fairness;
    metrics.variance_group_completed_tasks = stats.variance_group_completed_tasks;
    metrics.variance_group_active_lane_cycles = stats.variance_group_active_lane_cycles;
    metrics.estimated_cycles_saved_early_termination =
        stats.estimated_cycles_saved_early_termination;

    std::vector<double> processed_fractions;
    std::uint64_t skipped_macs_sum = 0;
    std::size_t terminated_tasks = 0;
    processed_fractions.reserve(stats.task_reports.size());
    for (const TaskReport& report : stats.task_reports) {
        const std::uint64_t total_macs = report.processed_macs + report.skipped_macs_total;
        const double fraction = total_macs == 0U
                                    ? 1.0
                                    : static_cast<double>(report.processed_macs) /
                                          static_cast<double>(total_macs);
        processed_fractions.push_back(fraction);
        if (report.early_terminated) {
            skipped_macs_sum += report.skipped_macs_et_only;
            ++terminated_tasks;
        }
    }
    metrics.processed_fraction_percentiles = computePercentiles(processed_fractions);
    metrics.mean_skipped_macs_per_terminated_task =
        terminated_tasks == 0U
            ? 0.0
            : static_cast<double>(skipped_macs_sum) /
                  static_cast<double>(terminated_tasks);
    return metrics;
}

BenchmarkRunOptions makeLegacyMainRunOptions() {
    BenchmarkRunOptions options;
    options.mode = BenchmarkRunMode::Detailed;
    options.suite_tiers = {BenchmarkSuiteTier::Quick};
    options.et_mode = BenchmarkEtMode::Paired;
    options.timed_repetitions = 1;
    options.warmup_iterations = 0;
    options.text_output_path = "experiment_report.txt";
    options.csv_output_path = "experiment_report.csv";
    options.selected_scenario_ids = {"quick-mnist-sample-0"};
    options.grouping_policies = allGroupingPolicies();
    return options;
}

std::vector<BenchmarkScenario> buildDefaultBenchmarkScenarios(const BenchmarkRunOptions& options) {
    std::vector<BenchmarkScenario> scenarios;
    if (tierRequested(options, BenchmarkSuiteTier::Quick)) {
        appendQuickScenarios(options, scenarios);
    }
    if (tierRequested(options, BenchmarkSuiteTier::Extended)) {
        appendExtendedScenarios(options, scenarios);
    }
    return filterSelectedScenarios(options, std::move(scenarios));
}

BenchmarkSuiteResult runBenchmarkSuite(const BenchmarkRunOptions& options) {
    BenchmarkRunOptions normalized = options;
    if (normalized.grouping_policies.empty()) {
        normalized.grouping_policies = allGroupingPolicies();
    }

    std::vector<BenchmarkScenario> scenarios = buildDefaultBenchmarkScenarios(normalized);
    validateOptions(normalized, scenarios);

    BenchmarkSuiteResult result;
    result.options = normalized;
    result.scenarios = scenarios;

    const std::vector<bool> et_states = etStatesForMode(normalized.et_mode);
    for (const BenchmarkScenario& scenario : scenarios) {
        const PreparedScenario prepared = prepareScenario(scenario);
        for (GroupingPolicy grouping_policy : normalized.grouping_policies) {
            for (bool et_enabled : et_states) {
                for (std::size_t warmup_index = 0; warmup_index < normalized.warmup_iterations;
                     ++warmup_index) {
                    (void)runSingleRecord(
                        prepared, normalized, grouping_policy, et_enabled, warmup_index);
                }
                for (std::size_t repetition_index = 0;
                     repetition_index < normalized.timed_repetitions;
                     ++repetition_index) {
                    result.records.push_back(runSingleRecord(
                        prepared, normalized, grouping_policy, et_enabled, repetition_index));
                }
            }
        }
    }

    result.aggregate_rows = buildAggregateRows(result.records);
    result.comparison_rows = buildComparisonRows(result.aggregate_rows);
    result.exactness_pass = std::all_of(
        result.records.begin(),
        result.records.end(),
        [](const BenchmarkRecord& record) { return record.error_metrics.total_mismatches == 0U; });

    result.primary_invariants_pass = true;
    for (const BenchmarkAggregateRow& row : result.aggregate_rows) {
        if (!row.early_termination_enabled &&
            (row.tasks_terminated_early != 0U || row.output_elements_terminated_early != 0U ||
             row.macs_skipped_et_only != 0U || row.bit_steps_skipped_et_only != 0U)) {
            result.primary_invariants_pass = false;
        }
    }
    for (const BenchmarkComparisonRow& row : result.comparison_rows) {
        if (!row.math_consistent) {
            result.primary_invariants_pass = false;
        }
    }

    return result;
}

std::string renderBenchmarkTextReport(const BenchmarkSuiteResult& result) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(6);
    writeConfigurationSection(out, result);

    switch (result.options.mode) {
    case BenchmarkRunMode::Speed:
        writeSpeedSection(out, result);
        break;
    case BenchmarkRunMode::Detailed:
        writeDetailedSection(out, result);
        break;
    case BenchmarkRunMode::Compare:
        writeCompareSection(out, result);
        break;
    }

    return out.str();
}

std::string renderBenchmarkCsvReport(const BenchmarkSuiteResult& result) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(6);

    if (result.options.mode == BenchmarkRunMode::Detailed) {
        out << "scenario_id,scenario_name,suite_tier,source,mnist_label,grouping_policy,et_enabled,"
               "repetitions,cycles_mean,cycles_median,wall_median_ms,throughput_macs_per_cycle,"
               "lane_occupancy,multiplier_utilization,accumulator_utilization,memory_stall_ratio,"
               "broadcast_stall_ratio,dram_bytes,onchip_buffer_bytes,activation_reuse_hits,"
               "activation_reuse_misses,weight_reuse_hits,weight_reuse_misses,"
               "memory_requests_avoided,work_stealing_events,tasks_terminated_early,"
               "output_elements_terminated_early,macs_skipped_total,macs_skipped_et_only,"
               "macs_skipped_reactive_only,macs_skipped_proactive_only,macs_skipped_zero_only,"
               "bit_steps_skipped_total,bit_steps_skipped_et_only,"
               "bit_steps_skipped_bit_column_only,zero_run_events,"
               "average_processed_fraction_per_task,total_mismatches,max_absolute_error,"
               "mean_absolute_error,mean_relative_error,jain_fairness,"
               "variance_group_completed_tasks,variance_group_active_lane_cycles,"
               "estimated_cycles_saved_early_termination,processed_fraction_p10,"
               "processed_fraction_p50,processed_fraction_p90,"
               "mean_skipped_macs_per_terminated_task\n";
        for (const BenchmarkAggregateRow& row : result.aggregate_rows) {
            out << row.scenario_id << "," << row.scenario_name << ","
                << benchmarkSuiteTierName(row.suite_tier) << "," << workloadSourceName(row.source)
                << "," << row.mnist_label << "," << groupingPolicyName(row.grouping_policy) << ","
                << boolName(row.early_termination_enabled) << "," << row.repetitions << ","
                << row.total_cycles.mean << "," << row.total_cycles.median << ","
                << row.wall_time_ms.median << "," << row.throughput_macs_per_cycle.mean << ","
                << row.lane_occupancy << "," << row.multiplier_utilization << ","
                << row.accumulator_utilization << "," << row.memory_stall_ratio << ","
                << row.broadcast_stall_ratio << "," << row.dram_bytes << ","
                << row.onchip_buffer_bytes << "," << row.activation_reuse_hits << ","
                << row.activation_reuse_misses << "," << row.weight_reuse_hits << ","
                << row.weight_reuse_misses << "," << row.memory_requests_avoided << ","
                << row.work_stealing_events << "," << row.tasks_terminated_early << ","
                << row.output_elements_terminated_early << "," << row.macs_skipped_total << ","
                << row.macs_skipped_et_only << "," << row.macs_skipped_reactive_only << ","
                << row.macs_skipped_proactive_only << "," << row.macs_skipped_zero_only << ","
                << row.bit_steps_skipped_total << "," << row.bit_steps_skipped_et_only << ","
                << row.bit_steps_skipped_bit_column_only << "," << row.zero_run_events << ","
                << row.average_processed_fraction_per_task << ","
                << row.total_mismatches << "," << row.max_absolute_error << ","
                << row.mean_absolute_error << "," << row.mean_relative_error << ","
                << row.secondary.lane_workload_jain_fairness << ","
                << row.secondary.variance_group_completed_tasks << ","
                << row.secondary.variance_group_active_lane_cycles << ","
                << row.secondary.estimated_cycles_saved_early_termination << ","
                << row.secondary.processed_fraction_percentiles.p10 << ","
                << row.secondary.processed_fraction_percentiles.p50 << ","
                << row.secondary.processed_fraction_percentiles.p90 << ","
                << row.secondary.mean_skipped_macs_per_terminated_task << "\n";
        }
        return out.str();
    }

    if (!result.comparison_rows.empty()) {
        out << "scenario_id,scenario_name,suite_tier,source,mnist_label,grouping_policy,"
               "cycles_et_off,cycles_et_on,cycle_delta,cycle_speedup,wall_time_et_off_ms,"
               "wall_time_et_on_ms,wall_time_speedup,tasks_terminated_early,"
               "output_elements_terminated_early,macs_skipped_total,macs_skipped_et_only,"
               "macs_skipped_reactive_only,macs_skipped_proactive_only,macs_skipped_zero_only,"
               "bit_steps_skipped_total,bit_steps_skipped_et_only,"
               "bit_steps_skipped_bit_column_only,zero_run_events,exact_match,math_consistent,"
               "cycle_reduction_observed\n";
        for (const BenchmarkComparisonRow& row : result.comparison_rows) {
            out << row.scenario_id << "," << row.scenario_name << ","
                << benchmarkSuiteTierName(row.suite_tier) << "," << workloadSourceName(row.source)
                << "," << row.mnist_label << "," << groupingPolicyName(row.grouping_policy) << ","
                << row.cycles_et_off << "," << row.cycles_et_on << "," << row.cycle_delta << ","
                << row.cycle_speedup << "," << row.wall_time_et_off_ms << ","
                << row.wall_time_et_on_ms << "," << row.wall_time_speedup << ","
                << row.tasks_terminated_early << "," << row.output_elements_terminated_early << ","
                << row.macs_skipped_total << "," << row.macs_skipped_et_only << ","
                << row.macs_skipped_reactive_only << "," << row.macs_skipped_proactive_only << ","
                << row.macs_skipped_zero_only << "," << row.bit_steps_skipped_total << ","
                << row.bit_steps_skipped_et_only << ","
                << row.bit_steps_skipped_bit_column_only << "," << row.zero_run_events << ","
                << (row.exact_match ? "true" : "false") << ","
                << (row.math_consistent ? "true" : "false") << ","
                << (row.cycle_reduction_observed ? "true" : "false") << "\n";
        }
        return out.str();
    }

    out << "scenario_id,scenario_name,suite_tier,source,mnist_label,grouping_policy,et_enabled,"
           "cycles_mean,cycles_median,wall_median_ms,wall_min_ms,wall_max_ms,"
           "throughput_macs_per_cycle\n";
    for (const BenchmarkAggregateRow& row : result.aggregate_rows) {
        out << row.scenario_id << "," << row.scenario_name << ","
            << benchmarkSuiteTierName(row.suite_tier) << "," << workloadSourceName(row.source)
            << "," << row.mnist_label << "," << groupingPolicyName(row.grouping_policy) << ","
            << boolName(row.early_termination_enabled) << "," << row.total_cycles.mean << ","
            << row.total_cycles.median << "," << row.wall_time_ms.median << ","
            << row.wall_time_ms.min << "," << row.wall_time_ms.max << ","
            << row.throughput_macs_per_cycle.mean << "\n";
    }
    return out.str();
}

bool tryParseBenchmarkSuiteTier(const std::string& value, BenchmarkSuiteTier& tier) {
    if (value == "quick") {
        tier = BenchmarkSuiteTier::Quick;
        return true;
    }
    if (value == "extended") {
        tier = BenchmarkSuiteTier::Extended;
        return true;
    }
    return false;
}

bool tryParseBenchmarkRunMode(const std::string& value, BenchmarkRunMode& mode) {
    if (value == "speed") {
        mode = BenchmarkRunMode::Speed;
        return true;
    }
    if (value == "detailed") {
        mode = BenchmarkRunMode::Detailed;
        return true;
    }
    if (value == "compare") {
        mode = BenchmarkRunMode::Compare;
        return true;
    }
    return false;
}

bool tryParseBenchmarkEtMode(const std::string& value, BenchmarkEtMode& mode) {
    if (value == "paired") {
        mode = BenchmarkEtMode::Paired;
        return true;
    }
    if (value == "off") {
        mode = BenchmarkEtMode::Off;
        return true;
    }
    if (value == "on") {
        mode = BenchmarkEtMode::On;
        return true;
    }
    return false;
}

bool tryParseGroupingPolicy(const std::string& value, GroupingPolicy& policy) {
    for (GroupingPolicy candidate : allGroupingPolicies()) {
        if (value == groupingPolicyName(candidate)) {
            policy = candidate;
            return true;
        }
    }
    return false;
}

bool tryParseZeroRunOrderMode(const std::string& value, ZeroRunOrderMode& mode) {
    if (value == "execution") {
        mode = ZeroRunOrderMode::ExecutionOrder;
        return true;
    }
    if (value == "kernel") {
        mode = ZeroRunOrderMode::KernelOrder;
        return true;
    }
    return false;
}

void writeBenchmarkOutputs(const BenchmarkSuiteResult& result,
                           std::ostream& text_out,
                           std::ostream* csv_out) {
    text_out << renderBenchmarkTextReport(result);
    if (csv_out != nullptr) {
        *csv_out << renderBenchmarkCsvReport(result);
    }
}
