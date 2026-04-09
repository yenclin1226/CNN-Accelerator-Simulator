#pragma once

#include "Accelerator.h"
#include "ReferenceConv.h"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

enum class BenchmarkSuiteTier {
    Quick,
    Extended
};

enum class BenchmarkRunMode {
    Speed,
    Detailed,
    Compare
};

enum class WorkloadSource {
    MnistSample,
    SyntheticRandom
};

enum class BenchmarkEtMode {
    Paired,
    Off,
    On
};

struct BenchmarkScenario {
    std::string id;
    std::string name;
    std::string profile_name;
    BenchmarkSuiteTier suite_tier{BenchmarkSuiteTier::Quick};
    WorkloadSource source{WorkloadSource::SyntheticRandom};
    ConvLayerConfig layer_config;
    AcceleratorConfig accelerator_config;
    RandomDataMode random_mode{RandomDataMode::UniformSymmetric};
    std::uint32_t seed{0};
    int mnist_sample_index{-1};
    int mnist_label{-1};
    std::string dataset_path;
    std::string weight_path;
    std::string bias_path;
};

struct BenchmarkRunOptions {
    std::vector<BenchmarkSuiteTier> suite_tiers{BenchmarkSuiteTier::Quick};
    BenchmarkRunMode mode{BenchmarkRunMode::Compare};
    std::size_t warmup_iterations{1};
    std::size_t timed_repetitions{1};
    std::vector<GroupingPolicy> grouping_policies;
    BenchmarkEtMode et_mode{BenchmarkEtMode::Paired};
    std::string dataset_path{"mnist_test.csv"};
    std::string weights_path{"training/conv1_weight.txt"};
    std::string bias_path{"training/conv1_bias.txt"};
    std::string text_output_path;
    std::string csv_output_path;
    std::vector<std::string> selected_scenario_ids;
    std::vector<std::string> synthetic_profile_filters;
    bool enable_reactive_zero_skip{false};
    bool enable_proactive_zero_run_skip{false};
    ZeroRunOrderMode zero_run_order_mode{ZeroRunOrderMode::ExecutionOrder};
    bool enable_bit_column_skip{false};
};

struct BenchmarkRecord {
    BenchmarkScenario scenario;
    GroupingPolicy grouping_policy{GroupingPolicy::OutputChannelModulo};
    bool early_termination_enabled{false};
    std::size_t repetition_index{0};
    double wall_time_ms{0.0};
    SimulationStats stats;
    ErrorMetrics error_metrics;
};

struct SummaryStats {
    double mean{0.0};
    double stddev{0.0};
    double min{0.0};
    double max{0.0};
    double median{0.0};
};

struct PercentileSummary {
    double p10{0.0};
    double p50{0.0};
    double p90{0.0};
};

struct BenchmarkSecondaryMetrics {
    double lane_workload_jain_fairness{1.0};
    double variance_group_completed_tasks{0.0};
    double variance_group_active_lane_cycles{0.0};
    std::uint64_t estimated_cycles_saved_early_termination{0};
    PercentileSummary processed_fraction_percentiles;
    double mean_skipped_macs_per_terminated_task{0.0};
};

struct BenchmarkAggregateRow {
    std::string scenario_id;
    std::string scenario_name;
    BenchmarkSuiteTier suite_tier{BenchmarkSuiteTier::Quick};
    WorkloadSource source{WorkloadSource::SyntheticRandom};
    int mnist_label{-1};
    GroupingPolicy grouping_policy{GroupingPolicy::OutputChannelModulo};
    bool early_termination_enabled{false};
    std::size_t repetitions{0};
    bool exact_match{true};
    SummaryStats wall_time_ms;
    SummaryStats total_cycles;
    SummaryStats throughput_macs_per_cycle;
    double lane_occupancy{0.0};
    double multiplier_utilization{0.0};
    double accumulator_utilization{0.0};
    double memory_stall_ratio{0.0};
    double broadcast_stall_ratio{0.0};
    std::uint64_t dram_bytes{0};
    std::uint64_t onchip_buffer_bytes{0};
    std::uint64_t activation_reuse_hits{0};
    std::uint64_t activation_reuse_misses{0};
    std::uint64_t weight_reuse_hits{0};
    std::uint64_t weight_reuse_misses{0};
    std::uint64_t memory_requests_avoided{0};
    std::uint64_t work_stealing_events{0};
    std::uint64_t tasks_terminated_early{0};
    std::uint64_t output_elements_terminated_early{0};
    std::uint64_t macs_skipped{0};
    std::uint64_t macs_skipped_total{0};
    std::uint64_t macs_skipped_et_only{0};
    std::uint64_t macs_skipped_reactive_only{0};
    std::uint64_t macs_skipped_proactive_only{0};
    std::uint64_t macs_skipped_zero_only{0};
    std::uint64_t bit_steps_skipped{0};
    std::uint64_t bit_steps_skipped_total{0};
    std::uint64_t bit_steps_skipped_et_only{0};
    std::uint64_t bit_steps_skipped_bit_column_only{0};
    std::uint64_t zero_run_events{0};
    double average_processed_fraction_per_task{1.0};
    std::size_t total_mismatches{0};
    std::int32_t max_absolute_error{0};
    double mean_absolute_error{0.0};
    double mean_relative_error{0.0};
    BenchmarkSecondaryMetrics secondary;
};

struct BenchmarkComparisonRow {
    std::string scenario_id;
    std::string scenario_name;
    BenchmarkSuiteTier suite_tier{BenchmarkSuiteTier::Quick};
    WorkloadSource source{WorkloadSource::SyntheticRandom};
    int mnist_label{-1};
    GroupingPolicy grouping_policy{GroupingPolicy::OutputChannelModulo};
    double cycles_et_off{0.0};
    double cycles_et_on{0.0};
    double cycle_delta{0.0};
    double cycle_speedup{0.0};
    double wall_time_et_off_ms{0.0};
    double wall_time_et_on_ms{0.0};
    double wall_time_speedup{0.0};
    std::uint64_t tasks_terminated_early{0};
    std::uint64_t output_elements_terminated_early{0};
    std::uint64_t macs_skipped{0};
    std::uint64_t macs_skipped_total{0};
    std::uint64_t macs_skipped_et_only{0};
    std::uint64_t macs_skipped_reactive_only{0};
    std::uint64_t macs_skipped_proactive_only{0};
    std::uint64_t macs_skipped_zero_only{0};
    std::uint64_t bit_steps_skipped{0};
    std::uint64_t bit_steps_skipped_total{0};
    std::uint64_t bit_steps_skipped_et_only{0};
    std::uint64_t bit_steps_skipped_bit_column_only{0};
    std::uint64_t zero_run_events{0};
    bool exact_match{true};
    bool math_consistent{true};
    bool cycle_reduction_observed{false};
};

struct BenchmarkSuiteResult {
    BenchmarkRunOptions options;
    std::vector<BenchmarkScenario> scenarios;
    std::vector<BenchmarkRecord> records;
    std::vector<BenchmarkAggregateRow> aggregate_rows;
    std::vector<BenchmarkComparisonRow> comparison_rows;
    bool exactness_pass{true};
    bool primary_invariants_pass{true};
};

std::string benchmarkSuiteTierName(BenchmarkSuiteTier tier);
std::string benchmarkRunModeName(BenchmarkRunMode mode);
std::string workloadSourceName(WorkloadSource source);
std::string benchmarkEtModeName(BenchmarkEtMode mode);
std::string groupingPolicyName(GroupingPolicy policy);

SummaryStats computeSummaryStats(const std::vector<double>& values);
double computePopulationVariance(const std::vector<double>& values);
double computeJainFairness(const std::vector<double>& values);
PercentileSummary computePercentiles(const std::vector<double>& values);
BenchmarkSecondaryMetrics deriveSecondaryMetrics(const SimulationStats& stats);

BenchmarkRunOptions makeLegacyMainRunOptions();
std::vector<BenchmarkScenario> buildDefaultBenchmarkScenarios(const BenchmarkRunOptions& options);
BenchmarkSuiteResult runBenchmarkSuite(const BenchmarkRunOptions& options);

std::string renderBenchmarkTextReport(const BenchmarkSuiteResult& result);
std::string renderBenchmarkCsvReport(const BenchmarkSuiteResult& result);

bool tryParseBenchmarkSuiteTier(const std::string& value, BenchmarkSuiteTier& tier);
bool tryParseBenchmarkRunMode(const std::string& value, BenchmarkRunMode& mode);
bool tryParseBenchmarkEtMode(const std::string& value, BenchmarkEtMode& mode);
bool tryParseGroupingPolicy(const std::string& value, GroupingPolicy& policy);
bool tryParseZeroRunOrderMode(const std::string& value, ZeroRunOrderMode& mode);

void writeBenchmarkOutputs(const BenchmarkSuiteResult& result,
                           std::ostream& text_out,
                           std::ostream* csv_out);
