#include "BenchmarkSuite.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

int main() {
    try {
        const BenchmarkRunOptions options = makeLegacyMainRunOptions();
        const BenchmarkSuiteResult result = runBenchmarkSuite(options);

        std::ofstream text_file(options.text_output_path);
        if (!text_file.is_open()) {
            throw std::runtime_error("Failed to open legacy text report path.");
        }
        text_file << renderBenchmarkTextReport(result);

        std::ofstream csv_file(options.csv_output_path);
        if (!csv_file.is_open()) {
            throw std::runtime_error("Failed to open legacy CSV report path.");
        }
        csv_file << renderBenchmarkCsvReport(result);

        std::cout << "Report written to: " << options.text_output_path << "\n";
        std::cout << "CSV written to: " << options.csv_output_path << "\n";
        std::cout << "Exactness: " << (result.exactness_pass ? "PASS" : "FAIL") << "\n";
        std::cout << "Primary invariants: "
                  << (result.primary_invariants_pass ? "PASS" : "FAIL") << "\n";

        if (!result.exactness_pass || !result.primary_invariants_pass) {
            return 1;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "bitserial_sim error: " << error.what() << "\n";
        return 1;
    }
}
