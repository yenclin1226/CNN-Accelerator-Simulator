#pragma once

#include "Task.h"
#include "Tensor.h"

#include <cstdint>
#include <string>
#include <vector>

struct ConvLayerConfig {
    int input_height{8};
    int input_width{8};
    int kernel_size{3};
    int stride{1};
    int padding{1};
    int input_channels{8};
    int output_channels{8};
    bool use_bias{true};
};

enum class RandomDataMode {
    UniformSymmetric,
    SparseActivations,
    NegativeBias
};

std::string toString(RandomDataMode mode);

class ConvLayer {
public:
    explicit ConvLayer(const ConvLayerConfig& config);

    int inputHeight() const;
    int inputWidth() const;
    int kernelSize() const;
    int stride() const;
    int padding() const;
    int inputChannels() const;
    int outputChannels() const;
    bool hasBias() const;

    int outputHeight() const;
    int outputWidth() const;

    void randomizeData(std::uint32_t seed, RandomDataMode mode);
    void loadQuantizedInput(const std::vector<std::int8_t>& activations);
    void loadInputFromMNISTRow(const std::vector<int>& pixels);
    void loadQuantizedWeights(const std::vector<std::int8_t>& weights);
    void loadQuantizedBias(const std::vector<std::int32_t>& bias);
    void zeroOutput();

    std::vector<Task> generateTasks() const;

    std::int8_t readActivation(int cin, int y, int x) const;
    std::int8_t readWeight(int oc, int cin, int ky, int kx) const;
    std::int32_t readBias(int oc) const;

    void writeOutput(int oc, int oy, int ox, std::int32_t value);

    const Tensor3D<std::int8_t>& inputTensor() const;
    const Tensor4D<std::int8_t>& weightTensor() const;
    const Tensor1D<std::int32_t>& biasTensor() const;
    const Tensor3D<std::int32_t>& outputTensor() const;

private:
    void validateConfig() const;

    ConvLayerConfig config_;
    Tensor3D<std::int8_t> input_;
    Tensor4D<std::int8_t> weights_;
    Tensor1D<std::int32_t> bias_;
    Tensor3D<std::int32_t> output_;
};
