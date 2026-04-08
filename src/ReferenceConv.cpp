#include "ReferenceConv.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

Tensor3D<std::int32_t> runReferenceConvolution(const ConvLayer& layer) {
    Tensor3D<std::int32_t> output(layer.outputChannels(), layer.outputHeight(), layer.outputWidth());

    for (int oc = 0; oc < layer.outputChannels(); ++oc) {
        for (int oy = 0; oy < layer.outputHeight(); ++oy) {
            for (int ox = 0; ox < layer.outputWidth(); ++ox) {
                std::int32_t acc = 0;

                for (int cin = 0; cin < layer.inputChannels(); ++cin) {
                    for (int ky = 0; ky < layer.kernelSize(); ++ky) {
                        for (int kx = 0; kx < layer.kernelSize(); ++kx) {
                            const int in_y = oy * layer.stride() + ky - layer.padding();
                            const int in_x = ox * layer.stride() + kx - layer.padding();

                            const std::int8_t act = layer.readActivation(cin, in_y, in_x);
                            const std::int8_t weight = layer.readWeight(oc, cin, ky, kx);
                            acc += static_cast<std::int32_t>(act) * static_cast<std::int32_t>(weight);
                        }
                    }
                }

                if (layer.hasBias()) {
                    acc += layer.readBias(oc);
                }
                output(oc, oy, ox) = acc;
            }
        }
    }

    return output;
}

Tensor3D<std::int32_t> applyRelu(const Tensor3D<std::int32_t>& input) {
    Tensor3D<std::int32_t> output(input.dim0(), input.dim1(), input.dim2());
    for (int oc = 0; oc < input.dim0(); ++oc) {
        for (int oy = 0; oy < input.dim1(); ++oy) {
            for (int ox = 0; ox < input.dim2(); ++ox) {
                output(oc, oy, ox) = std::max<std::int32_t>(0, input(oc, oy, ox));
            }
        }
    }
    return output;
}

OutputComparison compareOutputs(const Tensor3D<std::int32_t>& expected,
                               const Tensor3D<std::int32_t>& actual,
                               std::size_t max_mismatches_to_report) {
    if (expected.dim0() != actual.dim0() || expected.dim1() != actual.dim1() ||
        expected.dim2() != actual.dim2()) {
        throw std::invalid_argument("Expected and actual output tensor shapes do not match.");
    }

    OutputComparison result;
    result.pass = true;

    for (int oc = 0; oc < expected.dim0(); ++oc) {
        for (int oy = 0; oy < expected.dim1(); ++oy) {
            for (int ox = 0; ox < expected.dim2(); ++ox) {
                const std::int32_t exp = expected(oc, oy, ox);
                const std::int32_t act = actual(oc, oy, ox);

                if (exp == act) {
                    continue;
                }

                result.pass = false;
                ++result.total_mismatches;
                if (result.mismatches.size() < max_mismatches_to_report) {
                    result.mismatches.push_back(OutputMismatch{oc, oy, ox, exp, act});
                }
            }
        }
    }

    return result;
}

ErrorMetrics computeErrorMetrics(const Tensor3D<std::int32_t>& expected,
                                 const Tensor3D<std::int32_t>& actual) {
    if (expected.dim0() != actual.dim0() || expected.dim1() != actual.dim1() ||
        expected.dim2() != actual.dim2()) {
        throw std::invalid_argument("Expected and actual output tensor shapes do not match.");
    }

    ErrorMetrics metrics;
    metrics.total_elements = static_cast<std::size_t>(expected.dim0() * expected.dim1() * expected.dim2());

    std::int64_t abs_error_sum = 0;
    double rel_error_sum = 0.0;

    for (int oc = 0; oc < expected.dim0(); ++oc) {
        for (int oy = 0; oy < expected.dim1(); ++oy) {
            for (int ox = 0; ox < expected.dim2(); ++ox) {
                const std::int32_t exp = expected(oc, oy, ox);
                const std::int32_t act = actual(oc, oy, ox);
                const std::int32_t abs_error = std::abs(exp - act);
                abs_error_sum += abs_error;
                metrics.max_absolute_error = std::max(metrics.max_absolute_error, abs_error);
                if (abs_error != 0) {
                    ++metrics.total_mismatches;
                }

                const double denom = std::max(1.0, std::abs(static_cast<double>(exp)));
                rel_error_sum += static_cast<double>(abs_error) / denom;
            }
        }
    }

    if (metrics.total_elements > 0) {
        metrics.mean_absolute_error = static_cast<double>(abs_error_sum) /
                                      static_cast<double>(metrics.total_elements);
        metrics.mean_relative_error = rel_error_sum / static_cast<double>(metrics.total_elements);
    }
    return metrics;
}
