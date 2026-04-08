#pragma once

#include "ConvLayer.h"
#include "Tensor.h"

#include <cstddef>
#include <cstdint>
#include <vector>

struct OutputMismatch {
    int output_channel{0};
    int output_y{0};
    int output_x{0};
    std::int32_t expected{0};
    std::int32_t actual{0};
};

struct OutputComparison {
    bool pass{true};
    std::size_t total_mismatches{0};
    std::vector<OutputMismatch> mismatches;
};

struct ErrorMetrics {
    std::size_t total_elements{0};
    std::size_t total_mismatches{0};
    std::int32_t max_absolute_error{0};
    double mean_absolute_error{0.0};
    double mean_relative_error{0.0};
};

Tensor3D<std::int32_t> runReferenceConvolution(const ConvLayer& layer);
Tensor3D<std::int32_t> applyRelu(const Tensor3D<std::int32_t>& input);
OutputComparison compareOutputs(const Tensor3D<std::int32_t>& expected,
                               const Tensor3D<std::int32_t>& actual,
                               std::size_t max_mismatches_to_report);
ErrorMetrics computeErrorMetrics(const Tensor3D<std::int32_t>& expected,
                                 const Tensor3D<std::int32_t>& actual);
