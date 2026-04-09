#include "BenchmarkSuite.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool containsScenario(const BenchmarkSuiteResult& result, const std::string& scenario_id) {
    for (const BenchmarkScenario& scenario : result.scenarios) {
        if (scenario.id == scenario_id) {
            return true;
        }
    }
    return false;
}

bool hasComparisonForScenario(const BenchmarkSuiteResult& result,
                              const std::string& scenario_id,
                              GroupingPolicy grouping_policy) {
    for (const BenchmarkComparisonRow& row : result.comparison_rows) {
        if (row.scenario_id == scenario_id && row.grouping_policy == grouping_policy) {
            return true;
        }
    }
    return false;
}

}  // namespace

int main() {
    {
        BenchmarkRunOptions options;
        options.suite_tiers = {BenchmarkSuiteTier::Quick};
        options.mode = BenchmarkRunMode::Compare;
        options.et_mode = BenchmarkEtMode::Paired;
        options.warmup_iterations = 0;
        options.timed_repetitions = 1;
        options.grouping_policies = {GroupingPolicy::OutputChannelModulo};
        options.selected_scenario_ids = {
            "quick-mnist-sample-2827",
            "quick-mnist-sample-2462",
        };

        const BenchmarkSuiteResult result = runBenchmarkSuite(options);
        expect(result.exactness_pass, "compare real-world suite lost exactness");
        expect(result.primary_invariants_pass, "compare real-world suite violated invariants");
        expect(result.scenarios.size() == 2U, "compare real-world suite did not select 2 scenarios");
        expect(containsScenario(result, "quick-mnist-sample-2827"),
               "missing sparse MNIST scenario in compare suite");
        expect(containsScenario(result, "quick-mnist-sample-2462"),
               "missing dense MNIST scenario in compare suite");
        expect(result.comparison_rows.size() == 2U, "expected one comparison row per scenario");

        bool saw_et_activity = false;
        for (const BenchmarkComparisonRow& row : result.comparison_rows) {
            expect(row.math_consistent, "compare row math inconsistency");
            expect(row.exact_match, "compare row exactness failure");
            expect(row.cycles_et_off > 0.0, "compare row ET-off cycles must be positive");
            expect(row.cycles_et_on > 0.0, "compare row ET-on cycles must be positive");
            if (row.tasks_terminated_early > 0U) {
                saw_et_activity = true;
            }
        }
        expect(saw_et_activity, "expected ET activity across sparse/dense MNIST compare workloads");

        const std::string text_report = renderBenchmarkTextReport(result);
        expect(text_report.find("Section 3: Policy Rankings") != std::string::npos,
               "compare report missing policy rankings");
        expect(text_report.find("quick-mnist-sample-2827") != std::string::npos,
               "compare report missing sparse scenario");
        expect(text_report.find("quick-mnist-sample-2462") != std::string::npos,
               "compare report missing dense scenario");
    }

    {
        BenchmarkRunOptions options;
        options.suite_tiers = {BenchmarkSuiteTier::Quick};
        options.mode = BenchmarkRunMode::Speed;
        options.et_mode = BenchmarkEtMode::Paired;
        options.warmup_iterations = 0;
        options.timed_repetitions = 2;
        options.grouping_policies = {GroupingPolicy::TaskRoundRobin};
        options.selected_scenario_ids = {"quick-mnist-sample-0"};

        const BenchmarkSuiteResult result = runBenchmarkSuite(options);
        expect(result.exactness_pass, "speed benchmark exactness failed");
        expect(result.aggregate_rows.size() == 2U, "speed mode should aggregate ET-off and ET-on rows");
        expect(result.comparison_rows.size() == 1U, "speed mode paired run should still produce one comparison row");

        for (const BenchmarkAggregateRow& row : result.aggregate_rows) {
            expect(row.repetitions == 2U, "speed mode should preserve timed repetition count");
            expect(row.wall_time_ms.min > 0.0, "speed mode wall min must be positive");
            expect(row.wall_time_ms.max >= row.wall_time_ms.min,
                   "speed mode wall max/min ordering invalid");
            expect(row.wall_time_ms.median >= row.wall_time_ms.min &&
                       row.wall_time_ms.median <= row.wall_time_ms.max,
                   "speed mode wall median not bracketed by min/max");
            expect(row.total_cycles.mean > 0.0, "speed mode cycle summary must be positive");
            expect(row.throughput_macs_per_cycle.mean > 0.0,
                   "speed mode throughput must be positive");
        }

        const std::string csv_report = renderBenchmarkCsvReport(result);
        expect(csv_report.find("cycle_speedup") != std::string::npos,
               "speed mode CSV should contain comparison header");
    }

    {
        BenchmarkRunOptions options;
        options.suite_tiers = {BenchmarkSuiteTier::Extended};
        options.mode = BenchmarkRunMode::Detailed;
        options.et_mode = BenchmarkEtMode::On;
        options.warmup_iterations = 0;
        options.timed_repetitions = 1;
        options.grouping_policies = {GroupingPolicy::ETAwareCostBalancedMemoryAware};
        options.synthetic_profile_filters = {"memory-pressure"};
        options.selected_scenario_ids = {"extended-synthetic-memory-pressure-uniform-seed-1"};

        const BenchmarkSuiteResult result = runBenchmarkSuite(options);
        expect(result.exactness_pass, "detailed memory-pressure benchmark lost exactness");
        expect(result.primary_invariants_pass, "detailed memory-pressure benchmark violated invariants");
        expect(result.aggregate_rows.size() == 1U,
               "detailed mode with ET-on should aggregate one row for the selected synthetic case");

        bool saw_nonzero_memory_traffic = false;
        bool saw_nonzero_et_skips = false;
        for (const BenchmarkAggregateRow& row : result.aggregate_rows) {
            expect(row.early_termination_enabled,
                   "detailed memory-pressure ET-on run produced an ET-off aggregate row");
            expect(row.dram_bytes > 0U, "detailed mode DRAM bytes must be positive");
            expect(row.onchip_buffer_bytes > 0U, "detailed mode on-chip bytes must be positive");
            expect(row.memory_stall_ratio >= 0.0, "detailed mode memory stall ratio invalid");
            expect(row.broadcast_stall_ratio >= 0.0, "detailed mode broadcast stall ratio invalid");
            expect(row.secondary.processed_fraction_percentiles.p10 >= 0.0 &&
                       row.secondary.processed_fraction_percentiles.p90 <= 1.0,
                   "detailed mode ET processed fraction percentiles out of range");
            expect(row.macs_skipped_total >= row.macs_skipped_et_only,
                   "detailed mode MAC skip totals must dominate ET-only counts");
            expect(row.bit_steps_skipped_total >= row.bit_steps_skipped_et_only,
                   "detailed mode bit-step totals must dominate ET-only counts");
            if (row.dram_bytes > 0U || row.onchip_buffer_bytes > 0U) {
                saw_nonzero_memory_traffic = true;
            }
            if (row.early_termination_enabled && row.macs_skipped_et_only > 0U) {
                saw_nonzero_et_skips = true;
            }
        }
        expect(saw_nonzero_memory_traffic, "memory-pressure profile did not exercise memory traffic");
        expect(saw_nonzero_et_skips, "memory-pressure profile did not record ET skips");

        const std::string text_report = renderBenchmarkTextReport(result);
        const std::string csv_report = renderBenchmarkCsvReport(result);
        expect(text_report.find("Section 2: Detailed Primary Metrics") != std::string::npos,
               "detailed report missing primary metrics section");
        expect(text_report.find("Section 3: Secondary Diagnostics") != std::string::npos,
               "detailed report missing secondary diagnostics section");
        expect(csv_report.find("jain_fairness") != std::string::npos,
               "detailed CSV missing secondary metric columns");
    }

    return 0;
}
