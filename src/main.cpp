#include "Accelerator.h"
#include "MNISTReader.h"
#include "ReferenceConv.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

    struct DatasetRunConfig {
        bool use_dataset_mode{false};
        std::string dataset_source;
        std::string weight_source;
        std::string bias_source;
        int dataset_sample_index{-1};
        int dataset_label{-1};
    };

    struct ExperimentRecord {
        RandomDataMode random_mode{RandomDataMode::UniformSymmetric};
        std::uint32_t seed{0};
        AcceleratorConfig config;
        SimulationStats stats;
        ErrorMetrics error_metrics;
    };

    struct RegressionSuiteResult {
        std::uint32_t seed{0};
        RandomDataMode random_mode{RandomDataMode::UniformSymmetric};
        std::vector<ExperimentRecord> records;
        bool pass{true};
    };

    struct PairedSpeedupRecord {
        RandomDataMode random_mode{RandomDataMode::UniformSymmetric};
        std::uint32_t seed{0};
        GroupingPolicy grouping_policy{GroupingPolicy::OutputChannelModulo};
        std::uint64_t cycles_without_et{0};
        std::uint64_t cycles_with_et{0};
        double speedup{0.0};
    };

    struct SummaryStats {
        double mean{0.0};
        double stddev{0.0};
        double min{0.0};
        double max{0.0};
    };

    struct GroupingBaseline {
        std::string label;
        GroupingPolicy policy{GroupingPolicy::OutputChannelModulo};
    };

    std::string executionModeName(ExecutionMode mode) {
        return mode == ExecutionMode::Int8BitParallel ? "INT8 Bit-Parallel" : "INT8 Bit-Serial";
    }

    std::string broadcastModeName(BroadcastMode mode) {
        return mode == BroadcastMode::DemandDriven ? "Demand-Driven Vote-Based"
                                                   : "SnaPEA-like Fixed Schedule";
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
        throw std::invalid_argument("Unhandled grouping policy in groupingPolicyName.");
    }

    std::string pipelineModeName(PipelineMode mode) {
        return mode == PipelineMode::BaselineConvRelu ? "Baseline Conv + ReLU"
                                                      : "Fused Conv + ET + ReLU";
    }

    std::string boolName(bool value) {
        return value ? "On" : "Off";
    }

    std::string etStatusName(bool enabled) {
        return enabled ? "Enabled (Exact)" : "Disabled";
    }

    std::string formatDouble(double value, int precision = 6) {
        std::ostringstream stream;
        stream << std::fixed << std::setprecision(precision) << value;
        return stream.str();
    }

    std::string formatUint32List(const std::vector<std::uint32_t> &values) {
        std::ostringstream stream;
        stream << "[";
        for (std::size_t index = 0; index < values.size(); ++index) {
            if (index != 0U) {
                stream << ", ";
            }
            stream << values[index];
        }
        stream << "]";
        return stream.str();
    }

    std::string formatIntList(const std::vector<int> &values) {
        std::ostringstream stream;
        stream << "[";
        for (std::size_t index = 0; index < values.size(); ++index) {
            if (index != 0U) {
                stream << ", ";
            }
            stream << values[index];
        }
        stream << "]";
        return stream.str();
    }

    std::string formatModeList(const std::vector<RandomDataMode> &modes) {
        std::ostringstream stream;
        stream << "[";
        for (std::size_t index = 0; index < modes.size(); ++index) {
            if (index != 0U) {
                stream << ", ";
            }
            stream << toString(modes[index]);
        }
        stream << "]";
        return stream.str();
    }

    std::string experimentModeName(const DatasetRunConfig &dataset_config,
                                   RandomDataMode random_mode) {
        return dataset_config.use_dataset_mode ? "MNISTDataset" : toString(random_mode);
    }

    std::string experimentCaseId(const DatasetRunConfig &dataset_config, std::uint32_t seed) {
        if (dataset_config.use_dataset_mode) {
            return "sample=" + std::to_string(dataset_config.dataset_sample_index);
        }
        return "seed=" + std::to_string(seed);
    }

    void writeSectionHeader(std::ostream &out, const std::string &title) {
        out << title << "\n";
        out << std::string(title.size(), '=') << "\n";
    }

    SummaryStats computeMeanStdMinMax(const std::vector<double> &values) {
        SummaryStats summary;
        if (values.empty()) {
            return summary;
        }

        const double sum = std::accumulate(values.begin(), values.end(), 0.0);
        summary.mean = sum / static_cast<double>(values.size());
        summary.min = *std::min_element(values.begin(), values.end());
        summary.max = *std::max_element(values.begin(), values.end());

        double variance = 0.0;
        for (double value : values) {
            const double delta = value - summary.mean;
            variance += delta * delta;
        }
        variance /= static_cast<double>(values.size());
        summary.stddev = std::sqrt(variance);
        return summary;
    }

    AcceleratorConfig makeExperimentConfig(const AcceleratorConfig &common_config,
                                           GroupingPolicy grouping_policy,
                                           bool enable_exact_early_termination) {
        AcceleratorConfig config = common_config;
        config.grouping_policy = grouping_policy;
        config.pipeline_mode = enable_exact_early_termination
                                   ? PipelineMode::FusedConvEarlyTerminationRelu
                                   : PipelineMode::BaselineConvRelu;
        config.enable_early_termination = enable_exact_early_termination;
        return config;
    }

    ExperimentRecord collectExperimentRecord(RandomDataMode random_mode,
                                             std::uint32_t seed,
                                             GroupingPolicy grouping_policy,
                                             bool enable_exact_early_termination,
                                             const AcceleratorConfig &common_config,
                                             const ConvLayer &base_layer,
                                             const Tensor3D<std::int32_t> &reference_post_relu) {
        const AcceleratorConfig config =
            makeExperimentConfig(common_config, grouping_policy, enable_exact_early_termination);

        ConvLayer working_layer = base_layer;
        Accelerator accelerator(config);
        SimulationStats stats = accelerator.run(working_layer);

        ExperimentRecord record;
        record.random_mode = random_mode;
        record.seed = seed;
        record.config = config;
        record.stats = std::move(stats);
        record.error_metrics = computeErrorMetrics(reference_post_relu, working_layer.outputTensor());
        return record;
    }

    const ExperimentRecord *findRecord(const std::vector<ExperimentRecord> &records,
                                       RandomDataMode random_mode,
                                       std::uint32_t seed,
                                       GroupingPolicy grouping_policy,
                                       bool et_enabled) {
        for (const ExperimentRecord &record : records) {
            if (record.random_mode == random_mode && record.seed == seed &&
                record.config.grouping_policy == grouping_policy &&
                record.config.enable_early_termination == et_enabled) {
                return &record;
            }
        }
        return nullptr;
    }

    std::vector<PairedSpeedupRecord> buildPairedSpeedupRecords(
        const std::vector<ExperimentRecord> &records,
        const std::vector<RandomDataMode> &random_modes,
        const std::vector<std::uint32_t> &seeds,
        const std::vector<GroupingBaseline> &grouping_baselines) {
        std::vector<PairedSpeedupRecord> speedups;
        speedups.reserve(random_modes.size() * seeds.size() * grouping_baselines.size());

        for (RandomDataMode random_mode : random_modes) {
            for (std::uint32_t seed : seeds) {
                for (const GroupingBaseline &grouping_baseline : grouping_baselines) {
                    const ExperimentRecord *without_et =
                        findRecord(records, random_mode, seed, grouping_baseline.policy, false);
                    const ExperimentRecord *with_et =
                        findRecord(records, random_mode, seed, grouping_baseline.policy, true);
                    if (without_et == nullptr || with_et == nullptr || with_et->stats.total_cycles == 0) {
                        continue;
                    }

                    PairedSpeedupRecord speedup_record;
                    speedup_record.random_mode = random_mode;
                    speedup_record.seed = seed;
                    speedup_record.grouping_policy = grouping_baseline.policy;
                    speedup_record.cycles_without_et = without_et->stats.total_cycles;
                    speedup_record.cycles_with_et = with_et->stats.total_cycles;
                    speedup_record.speedup =
                        static_cast<double>(speedup_record.cycles_without_et) /
                        static_cast<double>(speedup_record.cycles_with_et);
                    speedups.push_back(speedup_record);
                }
            }
        }

        return speedups;
    }

    template <typename Accessor>
    SummaryStats summarizeRecords(const std::vector<ExperimentRecord> &records,
                                  RandomDataMode random_mode,
                                  GroupingPolicy grouping_policy,
                                  bool et_enabled,
                                  Accessor accessor) {
        std::vector<double> values;
        for (const ExperimentRecord &record : records) {
            if (record.random_mode == random_mode &&
                record.config.grouping_policy == grouping_policy &&
                record.config.enable_early_termination == et_enabled) {
                values.push_back(accessor(record));
            }
        }
        return computeMeanStdMinMax(values);
    }

    SummaryStats summarizeSpeedups(const std::vector<PairedSpeedupRecord> &speedup_records,
                                   RandomDataMode random_mode,
                                   GroupingPolicy grouping_policy) {
        std::vector<double> values;
        for (const PairedSpeedupRecord &record : speedup_records) {
            if (record.random_mode == random_mode && record.grouping_policy == grouping_policy) {
                values.push_back(record.speedup);
            }
        }
        return computeMeanStdMinMax(values);
    }

    void writeSummaryLine(std::ostream &out, const std::string &metric_name, const SummaryStats &summary) {
        out << "  " << metric_name << ": mean=" << formatDouble(summary.mean, 6)
            << ", stddev=" << formatDouble(summary.stddev, 6)
            << ", min=" << formatDouble(summary.min, 6)
            << ", max=" << formatDouble(summary.max, 6) << "\n";
    }

    void writeExperimentConfiguration(std::ostream &out,
                                      const DatasetRunConfig &dataset_config,
                                      const ConvLayerConfig &layer_config,
                                      const AcceleratorConfig &common_config,
                                      const std::vector<std::uint32_t> &seeds,
                                      const std::vector<RandomDataMode> &random_modes,
                                      const std::vector<GroupingBaseline> &grouping_baselines) {
        writeSectionHeader(out, "Section 1: Experiment Configuration");
        out << "Layer config\n";
        out << "  input_height=" << layer_config.input_height << "\n";
        out << "  input_width=" << layer_config.input_width << "\n";
        out << "  kernel_size=" << layer_config.kernel_size << "\n";
        out << "  stride=" << layer_config.stride << "\n";
        out << "  padding=" << layer_config.padding << "\n";
        out << "  input_channels=" << layer_config.input_channels << "\n";
        out << "  output_channels=" << layer_config.output_channels << "\n";
        out << "  use_bias=" << (layer_config.use_bias ? "true" : "false") << "\n";
        out << "\nInput/parameter mode\n";
        out << "  dataset_mode=" << (dataset_config.use_dataset_mode ? "true" : "false") << "\n";
        if (dataset_config.use_dataset_mode) {
            out << "  dataset_source=" << dataset_config.dataset_source << "\n";
            out << "  dataset_sample_index=" << dataset_config.dataset_sample_index << "\n";
            out << "  dataset_label=" << dataset_config.dataset_label << "\n";
            out << "  weight_source=" << dataset_config.weight_source << "\n";
            out << "  bias_source=" << dataset_config.bias_source << "\n";
        }
        out << "\nAccelerator common config\n";
        out << "  num_groups=" << common_config.num_groups << "\n";
        out << "  lanes_per_group=" << common_config.lanes_per_group << "\n";
        out << "  weight_precision_bits=" << common_config.weight_precision_bits << "\n";
        out << "  execution_mode=" << executionModeName(common_config.execution_mode) << "\n";
        out << "  broadcast_mode=" << broadcastModeName(common_config.broadcast_mode) << "\n";
        out << "  phase_offsets=" << formatIntList(common_config.phase_offsets) << "\n";
        out << "  enable_activation_reuse=" << (common_config.enable_activation_reuse ? "true" : "false")
            << "\n";
        out << "  enable_weight_reuse=" << (common_config.enable_weight_reuse ? "true" : "false")
            << "\n";
        out << "  local_buffer_capacity_entries=" << common_config.local_buffer_capacity_entries << "\n";
        out << "  global_buffer_capacity_entries=" << common_config.global_buffer_capacity_entries
            << "\n";
        out << "  enable_msb_first=" << (common_config.enable_msb_first ? "true" : "false") << "\n";
        out << "  enable_importance_ordering="
            << (common_config.enable_importance_ordering ? "true" : "false") << "\n";
        out << "  max_cycles=" << common_config.max_cycles << "\n";
        out << "  memory.dram_latency_cycles=" << common_config.memory.dram_latency_cycles << "\n";
        out << "  memory.dram_bandwidth_bytes_per_cycle="
            << common_config.memory.dram_bandwidth_bytes_per_cycle << "\n";
        out << "  memory.global_buffer_latency_cycles="
            << common_config.memory.global_buffer_latency_cycles << "\n";
        out << "  memory.global_buffer_bandwidth_bytes_per_cycle="
            << common_config.memory.global_buffer_bandwidth_bytes_per_cycle << "\n";
        out << "  memory.enable_local_buffer="
            << (common_config.memory.enable_local_buffer ? "true" : "false") << "\n";
        out << "  memory.local_buffer_latency_cycles="
            << common_config.memory.local_buffer_latency_cycles << "\n";
        out << "  memory.local_buffer_bandwidth_bytes_per_cycle="
            << common_config.memory.local_buffer_bandwidth_bytes_per_cycle << "\n";
        if (dataset_config.use_dataset_mode) {
            out << "\nSeed list\n";
            out << "  not used in dataset mode\n";
            out << "Random modes\n";
            out << "  not used in dataset mode\n";
        } else {
            out << "\nSeed list\n";
            out << "  " << formatUint32List(seeds) << "\n";
            out << "Random modes\n";
            out << "  " << formatModeList(random_modes) << "\n";
        }
        out << "Grouping policies\n";
        for (const GroupingBaseline &grouping_baseline : grouping_baselines) {
            out << "  " << groupingPolicyName(grouping_baseline.policy) << "\n";
        }
        out << "ET settings\n";
        out << "  Off, On (Exact)\n\n";
    }

    void writeRegressionSection(std::ostream &out,
                                const DatasetRunConfig &dataset_config,
                                const RegressionSuiteResult &regression_result) {
        writeSectionHeader(out, "Section 2: Regression Test");
        out << "Regression dataset\n";
        out << "  dataset_mode=" << (dataset_config.use_dataset_mode ? "true" : "false") << "\n";
        if (dataset_config.use_dataset_mode) {
            out << "  dataset_source=" << dataset_config.dataset_source << "\n";
            out << "  dataset_sample_index=" << dataset_config.dataset_sample_index << "\n";
            out << "  dataset_label=" << dataset_config.dataset_label << "\n";
            out << "  weight_source=" << dataset_config.weight_source << "\n";
            out << "  bias_source=" << dataset_config.bias_source << "\n";
        } else {
            out << "  seed=" << regression_result.seed << "\n";
            out << "  random_mode=" << toString(regression_result.random_mode) << "\n";
        }
        out << "  overall_result=" << (regression_result.pass ? "PASS" : "FAIL") << "\n\n";
        out << "grouping_policy | et_enabled | total_cycles | mismatches | max_abs_error | mean_abs_error | mean_relative_error | result\n";
        for (const ExperimentRecord &record : regression_result.records) {
            out << groupingPolicyName(record.config.grouping_policy) << " | "
                << boolName(record.config.enable_early_termination) << " | "
                << record.stats.total_cycles << " | " << record.error_metrics.total_mismatches << " | "
                << record.error_metrics.max_absolute_error << " | "
                << formatDouble(record.error_metrics.mean_absolute_error, 6) << " | "
                << formatDouble(record.error_metrics.mean_relative_error, 6) << " | "
                << (record.error_metrics.total_mismatches == 0 ? "PASS" : "FAIL") << "\n";
        }
        out << "\n";
    }

    void writeDetailedResultsSection(std::ostream &out,
                                     const DatasetRunConfig &dataset_config,
                                     const std::vector<ExperimentRecord> &records) {
        writeSectionHeader(out, "Section 3: Per-Seed Detailed Results");
        out << "mode | case_id | grouping_policy | et_enabled | total_cycles | completed_tasks | lane_occupancy | multiplier_utilization | accumulator_utilization | memory_stall_ratio | broadcast_stall_ratio | work_stealing_events | tasks_terminated_early | output_elements_terminated_early | macs_skipped | bit_steps_skipped | average_processed_fraction_per_task | variance_group_completed_tasks | variance_group_active_lane_cycles | lane_workload_jain_fairness | total_mismatches | max_abs_error | mean_abs_error | mean_relative_error\n";
        for (const ExperimentRecord &record : records) {
            out << experimentModeName(dataset_config, record.random_mode) << " | "
                << experimentCaseId(dataset_config, record.seed) << " | "
                << groupingPolicyName(record.config.grouping_policy) << " | "
                << boolName(record.config.enable_early_termination) << " | "
                << record.stats.total_cycles << " | " << record.stats.completed_tasks << " | "
                << formatDouble(record.stats.lane_occupancy, 6) << " | "
                << formatDouble(record.stats.multiplier_utilization, 6) << " | "
                << formatDouble(record.stats.accumulator_utilization, 6) << " | "
                << formatDouble(record.stats.memory_stall_ratio, 6) << " | "
                << formatDouble(record.stats.broadcast_stall_ratio, 6) << " | "
                << record.stats.work_stealing_events << " | "
                << record.stats.tasks_terminated_early << " | "
                << record.stats.output_elements_terminated_early << " | "
                << record.stats.macs_skipped << " | "
                << record.stats.bit_steps_skipped << " | "
                << formatDouble(record.stats.average_processed_fraction_per_task, 6) << " | "
                << formatDouble(record.stats.variance_group_completed_tasks, 6) << " | "
                << formatDouble(record.stats.variance_group_active_lane_cycles, 6) << " | "
                << formatDouble(record.stats.lane_workload_jain_fairness, 6) << " | "
                << record.error_metrics.total_mismatches << " | "
                << record.error_metrics.max_absolute_error << " | "
                << formatDouble(record.error_metrics.mean_absolute_error, 6) << " | "
                << formatDouble(record.error_metrics.mean_relative_error, 6) << "\n";
        }
        out << "\n";
    }

    void writePairedSpeedupSection(std::ostream &out,
                                   const DatasetRunConfig &dataset_config,
                                   const std::vector<PairedSpeedupRecord> &speedup_records) {
        writeSectionHeader(out, "Section 4: Paired ET Speedup");
        out << "mode | case_id | grouping_policy | cycles_et_off | cycles_et_on | speedup\n";
        for (const PairedSpeedupRecord &record : speedup_records) {
            out << experimentModeName(dataset_config, record.random_mode) << " | "
                << experimentCaseId(dataset_config, record.seed) << " | "
                << groupingPolicyName(record.grouping_policy) << " | "
                << record.cycles_without_et << " | " << record.cycles_with_et << " | "
                << formatDouble(record.speedup, 6) << "\n";
        }
        out << "\n";
    }

    void writeSummaryStatisticsSection(std::ostream &out,
                                       const DatasetRunConfig &dataset_config,
                                       const std::vector<ExperimentRecord> &records,
                                       const std::vector<RandomDataMode> &random_modes,
                                       const std::vector<GroupingBaseline> &grouping_baselines) {
        writeSectionHeader(out, "Section 5: Summary Statistics");
        for (RandomDataMode random_mode : random_modes) {
            for (const GroupingBaseline &grouping_baseline : grouping_baselines) {
                for (bool et_enabled : {false, true}) {
                    out << "mode=" << experimentModeName(dataset_config, random_mode)
                        << ", grouping_policy=" << groupingPolicyName(grouping_baseline.policy);
                    if (dataset_config.use_dataset_mode) {
                        out << ", case_id="
                            << experimentCaseId(dataset_config, 0);
                    }
                    out << ", et_enabled=" << boolName(et_enabled) << "\n";
                    writeSummaryLine(
                        out,
                        "total_cycles",
                        summarizeRecords(records, random_mode, grouping_baseline.policy, et_enabled,
                                         [](const ExperimentRecord &record) {
                                             return static_cast<double>(record.stats.total_cycles);
                                         }));
                    writeSummaryLine(
                        out,
                        "memory_stall_ratio",
                        summarizeRecords(records, random_mode, grouping_baseline.policy, et_enabled,
                                         [](const ExperimentRecord &record) {
                                             return record.stats.memory_stall_ratio;
                                         }));
                    writeSummaryLine(
                        out,
                        "broadcast_stall_ratio",
                        summarizeRecords(records, random_mode, grouping_baseline.policy, et_enabled,
                                         [](const ExperimentRecord &record) {
                                             return record.stats.broadcast_stall_ratio;
                                         }));
                    writeSummaryLine(
                        out,
                        "tasks_terminated_early",
                        summarizeRecords(records, random_mode, grouping_baseline.policy, et_enabled,
                                         [](const ExperimentRecord &record) {
                                             return static_cast<double>(
                                                 record.stats.tasks_terminated_early);
                                         }));
                    writeSummaryLine(
                        out,
                        "macs_skipped",
                        summarizeRecords(records, random_mode, grouping_baseline.policy, et_enabled,
                                         [](const ExperimentRecord &record) {
                                             return static_cast<double>(record.stats.macs_skipped);
                                         }));
                    writeSummaryLine(
                        out,
                        "bit_steps_skipped",
                        summarizeRecords(records, random_mode, grouping_baseline.policy, et_enabled,
                                         [](const ExperimentRecord &record) {
                                             return static_cast<double>(record.stats.bit_steps_skipped);
                                         }));
                    writeSummaryLine(
                        out,
                        "average_processed_fraction_per_task",
                        summarizeRecords(records, random_mode, grouping_baseline.policy, et_enabled,
                                         [](const ExperimentRecord &record) {
                                             return record.stats.average_processed_fraction_per_task;
                                         }));
                    writeSummaryLine(
                        out,
                        "work_stealing_events",
                        summarizeRecords(records, random_mode, grouping_baseline.policy, et_enabled,
                                         [](const ExperimentRecord &record) {
                                             return static_cast<double>(
                                                 record.stats.work_stealing_events);
                                         }));
                    writeSummaryLine(
                        out,
                        "lane_workload_jain_fairness",
                        summarizeRecords(records, random_mode, grouping_baseline.policy, et_enabled,
                                         [](const ExperimentRecord &record) {
                                             return record.stats.lane_workload_jain_fairness;
                                         }));
                    writeSummaryLine(
                        out,
                        "mean_absolute_error",
                        summarizeRecords(records, random_mode, grouping_baseline.policy, et_enabled,
                                         [](const ExperimentRecord &record) {
                                             return record.error_metrics.mean_absolute_error;
                                         }));
                    out << "\n";
                }
            }
        }
    }

    void writeSpeedupSummarySection(std::ostream &out,
                                    const DatasetRunConfig &dataset_config,
                                    const std::vector<PairedSpeedupRecord> &speedup_records,
                                    const std::vector<RandomDataMode> &random_modes,
                                    const std::vector<GroupingBaseline> &grouping_baselines) {
        writeSectionHeader(out, "Section 6: Summary of Paired Speedup");
        for (RandomDataMode random_mode : random_modes) {
            for (const GroupingBaseline &grouping_baseline : grouping_baselines) {
                out << "mode=" << experimentModeName(dataset_config, random_mode)
                    << ", grouping_policy=" << groupingPolicyName(grouping_baseline.policy);
                if (dataset_config.use_dataset_mode) {
                    out << ", case_id=" << experimentCaseId(dataset_config, 0);
                }
                out << "\n";
                writeSummaryLine(out,
                                 "paired_et_speedup",
                                 summarizeSpeedups(
                                     speedup_records, random_mode, grouping_baseline.policy));
                out << "\n";
            }
        }
    }

    void writeReport(const std::string &report_path,
                     const DatasetRunConfig &dataset_config,
                     const ConvLayerConfig &layer_config,
                     const AcceleratorConfig &common_config,
                     const std::vector<std::uint32_t> &seeds,
                     const std::vector<RandomDataMode> &random_modes,
                     const std::vector<GroupingBaseline> &grouping_baselines,
                     const RegressionSuiteResult &regression_result,
                     const std::vector<ExperimentRecord> &records,
                     const std::vector<PairedSpeedupRecord> &speedup_records) {
        std::ofstream report(report_path);
        if (!report.is_open()) {
            throw std::runtime_error("Failed to open experiment report for writing.");
        }

        report << std::fixed << std::setprecision(6);
        writeExperimentConfiguration(report, dataset_config, layer_config, common_config, seeds, random_modes,
                                     grouping_baselines);
        writeRegressionSection(report, dataset_config, regression_result);
        writeDetailedResultsSection(report, dataset_config, records);
        writePairedSpeedupSection(report, dataset_config, speedup_records);
        writeSummaryStatisticsSection(report, dataset_config, records, random_modes, grouping_baselines);
        writeSpeedupSummarySection(report, dataset_config, speedup_records, random_modes, grouping_baselines);
    }

    bool allRecordsExact(const std::vector<ExperimentRecord> &records) {
        return std::all_of(records.begin(), records.end(), [](const ExperimentRecord &record) {
            return record.error_metrics.total_mismatches == 0;
        });
    }

} // namespace

int main() {
    const std::string report_path = "experiment_report.txt";
    const bool use_dataset_mode = true;
    const int dataset_sample_index = 0;
    const std::string dataset_input_path = "mnist_test.csv";
    const std::string dataset_weight_path = "training/conv1_weight.txt";
    const std::string dataset_bias_path = "training/conv1_bias.txt";
    const std::vector<std::uint32_t> seeds = {1, 7, 13, 29, 42, 87, 123, 256, 512, 2026};
    const std::vector<RandomDataMode> random_modes = {
        RandomDataMode::UniformSymmetric,
        RandomDataMode::SparseActivations,
        RandomDataMode::NegativeBias,
    };
    const std::vector<GroupingBaseline> grouping_baselines = {
        {"Baseline-A", GroupingPolicy::OutputChannelModulo},
        {"Baseline-B", GroupingPolicy::TaskRoundRobin},
        {"ECBG", GroupingPolicy::ETAwareCostBalanced},
        {"ECBG+Mem", GroupingPolicy::ETAwareCostBalancedMemoryAware},
        {"BPFB", GroupingPolicy::BroadcastPhaseAwareFanoutBalanced},
    };

    DatasetRunConfig dataset_config;
    dataset_config.use_dataset_mode = use_dataset_mode;
    dataset_config.dataset_source = dataset_input_path;
    dataset_config.weight_source = dataset_weight_path;
    dataset_config.bias_source = dataset_bias_path;
    dataset_config.dataset_sample_index = dataset_sample_index;

    ConvLayerConfig layer_config;
    layer_config.input_height = use_dataset_mode ? 28 : 16;
    layer_config.input_width = use_dataset_mode ? 28 : 16;
    layer_config.kernel_size = 3;
    layer_config.stride = 1;
    layer_config.padding = 1;
    layer_config.input_channels = use_dataset_mode ? 1 : 8;
    layer_config.output_channels = 8;
    layer_config.use_bias = true;

    AcceleratorConfig common_config;
    common_config.num_groups = 6;
    common_config.lanes_per_group = 6;
    common_config.weight_precision_bits = 8;
    common_config.phase_offsets = {0, 1, 2, 3, 4, 5};
    common_config.execution_mode = ExecutionMode::Int8BitSerial;
    common_config.grouping_policy = GroupingPolicy::OutputChannelModulo;
    common_config.broadcast_mode = BroadcastMode::DemandDriven;
    common_config.enable_activation_reuse = true;
    common_config.enable_weight_reuse = true;
    common_config.local_buffer_capacity_entries = 128;
    common_config.global_buffer_capacity_entries = 1024;
    common_config.enable_msb_first = true;
    common_config.enable_importance_ordering = true;
    common_config.max_cycles = 50'000'000;
    common_config.memory.dram_latency_cycles = 16;
    common_config.memory.dram_bandwidth_bytes_per_cycle = 64;
    common_config.memory.global_buffer_latency_cycles = 4;
    common_config.memory.global_buffer_bandwidth_bytes_per_cycle = 128;
    common_config.memory.enable_local_buffer = true;
    common_config.memory.local_buffer_latency_cycles = 1;
    common_config.memory.local_buffer_bandwidth_bytes_per_cycle = 64;

    const std::uint32_t regression_seed = 2026;
    const RandomDataMode regression_mode = RandomDataMode::UniformSymmetric;
    const std::vector<std::uint32_t> active_seeds =
        use_dataset_mode ? std::vector<std::uint32_t>{0U} : seeds;
    const std::vector<RandomDataMode> active_random_modes =
        use_dataset_mode ? std::vector<RandomDataMode>{regression_mode} : random_modes;

    std::vector<int> mnist_pixels;
    std::vector<std::int8_t> quantized_weights;
    std::vector<std::int32_t> quantized_bias;
    if (use_dataset_mode) {
        const MNISTSample sample = readMNISTSample(dataset_input_path, dataset_sample_index);
        dataset_config.dataset_label = sample.label;
        mnist_pixels = sample.pixels;
        quantized_weights = readQuantizedConvWeights(dataset_weight_path,
                                                     layer_config.output_channels,
                                                     layer_config.input_channels,
                                                     layer_config.kernel_size);
        quantized_bias = readQuantizedConvBias(dataset_bias_path, layer_config.output_channels);
    }

    if (use_dataset_mode) {
        std::cout << "Running regression test: dataset sample=" << dataset_sample_index
                  << ", label=" << dataset_config.dataset_label << "\n";
    } else {
        std::cout << "Running regression test: mode=" << toString(regression_mode)
                  << ", seed=" << regression_seed << "\n";
    }
    ConvLayer regression_base_layer(layer_config);
    if (use_dataset_mode) {
        regression_base_layer.loadInputFromMNISTRow(mnist_pixels);
        regression_base_layer.loadQuantizedWeights(quantized_weights);
        regression_base_layer.loadQuantizedBias(quantized_bias);
    } else {
        regression_base_layer.randomizeData(regression_seed, regression_mode);
    }
    const Tensor3D<std::int32_t> regression_reference_post_relu =
        applyRelu(runReferenceConvolution(regression_base_layer));

    RegressionSuiteResult regression_result;
    regression_result.seed = regression_seed;
    regression_result.random_mode = regression_mode;
    regression_result.pass = true;
    for (const GroupingBaseline &grouping_baseline : grouping_baselines) {
        for (bool et_enabled : {false, true}) {
            ExperimentRecord record = collectExperimentRecord(
                regression_mode,
                regression_seed,
                grouping_baseline.policy,
                et_enabled,
                common_config,
                regression_base_layer,
                regression_reference_post_relu);
            regression_result.pass =
                regression_result.pass && (record.error_metrics.total_mismatches == 0);
            regression_result.records.push_back(std::move(record));
        }
    }

    std::vector<ExperimentRecord> records;
    records.reserve(active_random_modes.size() * active_seeds.size() * grouping_baselines.size() * 2U);

    for (RandomDataMode random_mode : active_random_modes) {
        for (std::uint32_t seed : active_seeds) {
            if (use_dataset_mode) {
                std::cout << "Running statistical experiments: dataset sample="
                          << dataset_sample_index << ", label=" << dataset_config.dataset_label
                          << "\n";
            } else {
                std::cout << "Running statistical experiments: mode=" << toString(random_mode)
                          << ", seed=" << seed << "\n";
            }

            ConvLayer base_layer(layer_config);
            if (use_dataset_mode) {
                base_layer.loadInputFromMNISTRow(mnist_pixels);
                base_layer.loadQuantizedWeights(quantized_weights);
                base_layer.loadQuantizedBias(quantized_bias);
            } else {
                base_layer.randomizeData(seed, random_mode);
            }
            const Tensor3D<std::int32_t> reference_post_relu =
                applyRelu(runReferenceConvolution(base_layer));

            for (const GroupingBaseline &grouping_baseline : grouping_baselines) {
                for (bool et_enabled : {false, true}) {
                    records.push_back(collectExperimentRecord(random_mode,
                                                              seed,
                                                              grouping_baseline.policy,
                                                              et_enabled,
                                                              common_config,
                                                              base_layer,
                                                              reference_post_relu));
                }
            }
        }
    }

    const std::vector<PairedSpeedupRecord> speedup_records =
        buildPairedSpeedupRecords(records, active_random_modes, active_seeds, grouping_baselines);

    writeReport(report_path,
                dataset_config,
                layer_config,
                common_config,
                active_seeds,
                active_random_modes,
                grouping_baselines,
                regression_result,
                records,
                speedup_records);

    const bool statistical_exact = allRecordsExact(records);
    std::cout << "Regression result: " << (regression_result.pass ? "PASS" : "FAIL") << "\n";
    std::cout << "Statistical exactness: " << (statistical_exact ? "PASS" : "FAIL") << "\n";
    std::cout << "Report written to: " << report_path << "\n";

    if (!regression_result.pass || !statistical_exact) {
        return 1;
    }
    return 0;
}
