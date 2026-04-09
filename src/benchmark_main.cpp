#include "BenchmarkSuite.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string benchmarkHelpText() {
    return
        "Usage: bitserial_bench [options]\n"
        "\n"
        "Options:\n"
        "  --suite quick|extended|all          Benchmark tier selection. Default: quick.\n"
        "  --mode speed|detailed|compare       Output mode. Default: compare.\n"
        "  --repetitions N                     Timed repetitions per configuration. Default: 1.\n"
        "  --warmup N                          Warmup repetitions per configuration. Default: 1.\n"
        "  --csv-out PATH                      Optional CSV output path.\n"
        "  --text-out PATH                     Optional text output path.\n"
        "  --dataset-path PATH                 Override MNIST CSV path for MNIST-backed scenarios.\n"
        "  --weights-path PATH                 Override quantized weight file for MNIST-backed scenarios.\n"
        "  --bias-path PATH                    Override quantized bias file for MNIST-backed scenarios.\n"
        "  --grouping all|POLICY               Grouping policy filter. Repeatable. Default: all.\n"
        "  --et paired|off|on                  Early termination selection. Default: paired.\n"
        "  --scenario ID                       Optional scenario id filter. Repeatable.\n"
        "  --synthetic-profile NAME            Synthetic-only filter: default|large|memory-pressure|all.\n"
        "  --help                              Show this message.\n"
        "\n"
        "Defaults are conservative: quick suite, compare mode, one warmup, one timed repetition,\n"
        "all grouping policies, and paired ET. The combination of --suite all, --grouping all,\n"
        "--et paired, and --repetitions > 1 is intentionally large.\n"
        "\n"
        "Precedence rules:\n"
        "  Dataset/weight/bias overrides apply only to MNIST-backed scenarios.\n"
        "  Synthetic profile filters apply only to synthetic scenarios.\n"
        "  Mixed-source runs may use both; each override applies only to matching scenarios.\n";
}

std::string requireValue(int& index, int argc, char** argv, const std::string& flag) {
    if (index + 1 >= argc) {
        throw std::invalid_argument("Missing value for " + flag + ".");
    }
    ++index;
    return argv[index];
}

BenchmarkRunOptions parseBenchmarkRunOptions(int argc, char** argv) {
    BenchmarkRunOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help") {
            std::cout << benchmarkHelpText();
            std::exit(0);
        }

        if (argument == "--suite") {
            const std::string value = requireValue(index, argc, argv, argument);
            if (value == "all") {
                options.suite_tiers = {BenchmarkSuiteTier::Quick, BenchmarkSuiteTier::Extended};
                continue;
            }
            BenchmarkSuiteTier tier;
            if (!tryParseBenchmarkSuiteTier(value, tier)) {
                throw std::invalid_argument("Invalid suite tier: " + value);
            }
            options.suite_tiers = {tier};
            continue;
        }

        if (argument == "--mode") {
            const std::string value = requireValue(index, argc, argv, argument);
            if (!tryParseBenchmarkRunMode(value, options.mode)) {
                throw std::invalid_argument("Invalid benchmark mode: " + value);
            }
            continue;
        }

        if (argument == "--repetitions") {
            options.timed_repetitions = static_cast<std::size_t>(
                std::stoul(requireValue(index, argc, argv, argument)));
            continue;
        }

        if (argument == "--warmup") {
            options.warmup_iterations = static_cast<std::size_t>(
                std::stoul(requireValue(index, argc, argv, argument)));
            continue;
        }

        if (argument == "--csv-out") {
            options.csv_output_path = requireValue(index, argc, argv, argument);
            continue;
        }

        if (argument == "--text-out") {
            options.text_output_path = requireValue(index, argc, argv, argument);
            continue;
        }

        if (argument == "--dataset-path") {
            options.dataset_path = requireValue(index, argc, argv, argument);
            continue;
        }

        if (argument == "--weights-path") {
            options.weights_path = requireValue(index, argc, argv, argument);
            continue;
        }

        if (argument == "--bias-path") {
            options.bias_path = requireValue(index, argc, argv, argument);
            continue;
        }

        if (argument == "--grouping") {
            const std::string value = requireValue(index, argc, argv, argument);
            if (value == "all") {
                options.grouping_policies.clear();
                continue;
            }
            GroupingPolicy policy;
            if (!tryParseGroupingPolicy(value, policy)) {
                throw std::invalid_argument("Invalid grouping policy: " + value);
            }
            if (options.grouping_policies.empty()) {
                options.grouping_policies.push_back(policy);
            } else if (std::find(options.grouping_policies.begin(),
                                 options.grouping_policies.end(),
                                 policy) == options.grouping_policies.end()) {
                options.grouping_policies.push_back(policy);
            }
            continue;
        }

        if (argument == "--et") {
            const std::string value = requireValue(index, argc, argv, argument);
            if (!tryParseBenchmarkEtMode(value, options.et_mode)) {
                throw std::invalid_argument("Invalid ET mode: " + value);
            }
            continue;
        }

        if (argument == "--scenario") {
            options.selected_scenario_ids.push_back(requireValue(index, argc, argv, argument));
            continue;
        }

        if (argument == "--synthetic-profile") {
            const std::string value = requireValue(index, argc, argv, argument);
            if (value == "all") {
                options.synthetic_profile_filters.clear();
            } else {
                options.synthetic_profile_filters.push_back(value);
            }
            continue;
        }

        throw std::invalid_argument("Unknown argument: " + argument);
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const BenchmarkRunOptions options = parseBenchmarkRunOptions(argc, argv);
        const BenchmarkSuiteResult result = runBenchmarkSuite(options);

        const std::string text_report = renderBenchmarkTextReport(result);
        const std::string csv_report = renderBenchmarkCsvReport(result);

        if (!options.text_output_path.empty()) {
            std::ofstream text_file(options.text_output_path);
            if (!text_file.is_open()) {
                throw std::runtime_error("Failed to open text output path: " + options.text_output_path);
            }
            text_file << text_report;
        } else {
            std::cout << text_report;
        }

        if (!options.csv_output_path.empty()) {
            std::ofstream csv_file(options.csv_output_path);
            if (!csv_file.is_open()) {
                throw std::runtime_error("Failed to open CSV output path: " + options.csv_output_path);
            }
            csv_file << csv_report;
        }

        if (!result.exactness_pass || !result.primary_invariants_pass) {
            return 1;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "bitserial_bench error: " << error.what() << "\n";
        return 1;
    }
}
