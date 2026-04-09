#include "BenchmarkSuite.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    {
        const SummaryStats stats = computeSummaryStats({1.0, 2.0, 3.0, 4.0});
        expect(std::abs(stats.mean - 2.5) < 1e-9, "summary mean mismatch");
        expect(std::abs(stats.median - 2.5) < 1e-9, "summary median mismatch");
        expect(std::abs(computePopulationVariance({1.0, 2.0, 3.0, 4.0}) - 1.25) < 1e-9,
               "population variance mismatch");
        expect(std::abs(computeJainFairness({1.0, 1.0, 1.0, 1.0}) - 1.0) < 1e-9,
               "Jain fairness mismatch");
    }

    {
        const PercentileSummary percentiles = computePercentiles({0.2, 0.4, 0.6, 0.8, 1.0});
        expect(std::abs(percentiles.p10 - 0.28) < 1e-9, "p10 percentile mismatch");
        expect(std::abs(percentiles.p50 - 0.6) < 1e-9, "p50 percentile mismatch");
        expect(std::abs(percentiles.p90 - 0.92) < 1e-9, "p90 percentile mismatch");
    }

    {
        SimulationStats stats;
        stats.lane_workload_jain_fairness = 0.95;
        stats.variance_group_completed_tasks = 12.0;
        stats.variance_group_active_lane_cycles = 34.0;
        stats.estimated_cycles_saved_early_termination = 56;
        stats.task_reports = {
            TaskReport{0, 0, 0, 0, false, 0, 0, 10, 0, 80, 0},
            TaskReport{1, 0, 0, 1, true, 0, 0, 4, 6, 32, 48},
            TaskReport{2, 0, 0, 2, true, 0, 0, 2, 8, 16, 64},
        };

        const BenchmarkSecondaryMetrics secondary = deriveSecondaryMetrics(stats);
        expect(std::abs(secondary.lane_workload_jain_fairness - 0.95) < 1e-9,
               "secondary fairness mismatch");
        expect(secondary.estimated_cycles_saved_early_termination == 56,
               "secondary estimated cycles mismatch");
        expect(std::abs(secondary.mean_skipped_macs_per_terminated_task - 7.0) < 1e-9,
               "secondary skipped MAC average mismatch");
        expect(secondary.processed_fraction_percentiles.p50 < 1.0,
               "secondary percentiles did not capture ET effects");
    }

    {
        BenchmarkSuiteResult result;
        result.options.mode = BenchmarkRunMode::Compare;

        BenchmarkComparisonRow slower;
        slower.scenario_id = "scenario-a";
        slower.scenario_name = "Scenario A";
        slower.grouping_policy = GroupingPolicy::TaskRoundRobin;
        slower.cycles_et_off = 100.0;
        slower.cycles_et_on = 70.0;
        slower.cycle_speedup = 100.0 / 70.0;
        slower.tasks_terminated_early = 5;
        slower.math_consistent = true;
        slower.cycle_reduction_observed = true;

        BenchmarkComparisonRow faster;
        faster.scenario_id = "scenario-a";
        faster.scenario_name = "Scenario A";
        faster.grouping_policy = GroupingPolicy::OutputChannelModulo;
        faster.cycles_et_off = 100.0;
        faster.cycles_et_on = 60.0;
        faster.cycle_speedup = 100.0 / 60.0;
        faster.tasks_terminated_early = 6;
        faster.math_consistent = true;
        faster.cycle_reduction_observed = true;

        result.comparison_rows = {slower, faster};

        const std::string text_report = renderBenchmarkTextReport(result);
        const std::string csv_report = renderBenchmarkCsvReport(result);

        expect(text_report.find("1. OutputChannelModulo") != std::string::npos,
               "ranking section missing sorted best policy");
        expect(text_report.find("2. TaskRoundRobin") != std::string::npos,
               "ranking section missing sorted second policy");
        expect(csv_report.find("scenario_id,scenario_name") != std::string::npos,
               "CSV header missing");
        expect(csv_report.find("scenario-a,Scenario A") != std::string::npos,
               "CSV body missing comparison row");
    }

    return 0;
}
