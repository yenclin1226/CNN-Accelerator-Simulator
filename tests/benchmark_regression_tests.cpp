#include "BenchmarkSuite.h"

#include <stdexcept>
#include <string>

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    BenchmarkRunOptions options;
    options.suite_tiers = {BenchmarkSuiteTier::Quick};
    options.mode = BenchmarkRunMode::Compare;
    options.et_mode = BenchmarkEtMode::Paired;
    options.warmup_iterations = 0;
    options.timed_repetitions = 1;
    options.selected_scenario_ids = {
        "quick-mnist-sample-0",
    };

    const BenchmarkSuiteResult result = runBenchmarkSuite(options);
    expect(result.exactness_pass, "benchmark exactness failed");
    expect(result.primary_invariants_pass, "benchmark primary invariants failed");

    bool saw_mnist_et_activity = false;
    for (const BenchmarkAggregateRow& row : result.aggregate_rows) {
        if (!row.early_termination_enabled) {
            expect(row.tasks_terminated_early == 0U, "ET-off row reported terminated tasks");
            expect(row.output_elements_terminated_early == 0U,
                   "ET-off row reported terminated outputs");
            expect(row.macs_skipped == 0U, "ET-off row reported skipped MACs");
            expect(row.bit_steps_skipped == 0U, "ET-off row reported skipped bit steps");
        }
        expect(row.total_mismatches == 0U, "aggregate row reported output mismatches");
    }

    for (const BenchmarkComparisonRow& row : result.comparison_rows) {
        expect(row.math_consistent, "comparison row math inconsistency detected");
        expect(row.exact_match, "comparison row lost exactness");
        if (row.scenario_id == "quick-mnist-sample-0" && row.tasks_terminated_early > 0U) {
            saw_mnist_et_activity = true;
        }
    }

    expect(saw_mnist_et_activity,
           "expected non-zero ET activity for quick-mnist-sample-0 compare runs");
    return 0;
}
