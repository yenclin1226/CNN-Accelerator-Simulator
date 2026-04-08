#include "ConvLayer.h"

#include <algorithm>
#include <random>
#include <stdexcept>

namespace {

std::int8_t sampleUniformSymmetric(std::mt19937& rng) {
    static std::uniform_int_distribution<int> value_dist(-4, 4);
    return static_cast<std::int8_t>(value_dist(rng));
}

std::int8_t sampleUniformSymmetricNonZero(std::mt19937& rng) {
    static const std::int8_t kValues[] = {-4, -3, -2, -1, 1, 2, 3, 4};
    static std::uniform_int_distribution<int> index_dist(0, 7);
    return kValues[index_dist(rng)];
}

}  // namespace

std::string toString(RandomDataMode mode) {
    switch (mode) {
    case RandomDataMode::UniformSymmetric:
        return "UniformSymmetric";
    case RandomDataMode::SparseActivations:
        return "SparseActivations";
    case RandomDataMode::NegativeBias:
        return "NegativeBias";
    }
    return "UnknownRandomDataMode";
}

ConvLayer::ConvLayer(const ConvLayerConfig& config)
    : config_(config),
      input_(config.input_channels, config.input_height, config.input_width),
      weights_(config.output_channels, config.input_channels, config.kernel_size, config.kernel_size),
      bias_(config.output_channels),
      output_(config.output_channels,
              ((config.input_height + 2 * config.padding - config.kernel_size) / config.stride) + 1,
              ((config.input_width + 2 * config.padding - config.kernel_size) / config.stride) + 1) {
    validateConfig();
    zeroOutput();
}

int ConvLayer::inputHeight() const {
    return config_.input_height;
}

int ConvLayer::inputWidth() const {
    return config_.input_width;
}

int ConvLayer::kernelSize() const {
    return config_.kernel_size;
}

int ConvLayer::stride() const {
    return config_.stride;
}

int ConvLayer::padding() const {
    return config_.padding;
}

int ConvLayer::inputChannels() const {
    return config_.input_channels;
}

int ConvLayer::outputChannels() const {
    return config_.output_channels;
}

bool ConvLayer::hasBias() const {
    return config_.use_bias;
}

int ConvLayer::outputHeight() const {
    return ((config_.input_height + 2 * config_.padding - config_.kernel_size) / config_.stride) + 1;
}

int ConvLayer::outputWidth() const {
    return ((config_.input_width + 2 * config_.padding - config_.kernel_size) / config_.stride) + 1;
}

void ConvLayer::randomizeData(std::uint32_t seed, RandomDataMode mode) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> symmetric_bias_dist(-32, 32);
    std::uniform_int_distribution<int> negative_bias_dist(-64, 0);
    std::bernoulli_distribution sparse_zero_dist(0.70);

    for (int cin = 0; cin < inputChannels(); ++cin) {
        for (int y = 0; y < inputHeight(); ++y) {
            for (int x = 0; x < inputWidth(); ++x) {
                if (mode == RandomDataMode::SparseActivations && sparse_zero_dist(rng)) {
                    input_(cin, y, x) = 0;
                } else if (mode == RandomDataMode::SparseActivations) {
                    input_(cin, y, x) = sampleUniformSymmetricNonZero(rng);
                } else {
                    input_(cin, y, x) = sampleUniformSymmetric(rng);
                }
            }
        }
    }

    for (int oc = 0; oc < outputChannels(); ++oc) {
        for (int cin = 0; cin < inputChannels(); ++cin) {
            for (int ky = 0; ky < kernelSize(); ++ky) {
                for (int kx = 0; kx < kernelSize(); ++kx) {
                    weights_(oc, cin, ky, kx) = sampleUniformSymmetric(rng);
                }
            }
        }
    }

    for (int oc = 0; oc < outputChannels(); ++oc) {
        if (!hasBias()) {
            bias_(oc) = 0;
            continue;
        }

        bias_(oc) = (mode == RandomDataMode::NegativeBias)
                        ? static_cast<std::int32_t>(negative_bias_dist(rng))
                        : static_cast<std::int32_t>(symmetric_bias_dist(rng));
    }
}

void ConvLayer::loadQuantizedInput(const std::vector<std::int8_t>& activations) {
    const std::size_t expected_size =
        static_cast<std::size_t>(inputChannels()) * inputHeight() * inputWidth();
    if (activations.size() != expected_size) {
        throw std::invalid_argument("Quantized input tensor size does not match layer shape.");
    }

    input_.raw() = activations;
}

void ConvLayer::loadInputFromMNISTRow(const std::vector<int>& pixels) {
    if (pixels.size() != 784U) {
        throw std::invalid_argument("MNIST row must contain exactly 784 pixels.");
    }
    if (inputChannels() != 1 || inputHeight() != 28 || inputWidth() != 28) {
        throw std::invalid_argument("MNIST input requires layer shape 1x28x28.");
    }

    std::vector<std::int8_t> quantized_input;
    quantized_input.reserve(pixels.size());
    for (int y = 0; y < inputHeight(); ++y) {
        for (int x = 0; x < inputWidth(); ++x) {
            const std::size_t pixel_index = static_cast<std::size_t>(y * inputWidth() + x);
            const int pixel_value = pixels[pixel_index];
            if (pixel_value < 0 || pixel_value > 255) {
                throw std::invalid_argument("MNIST pixel values must be in the range [0, 255].");
            }

            quantized_input.push_back(static_cast<std::int8_t>(pixel_value / 2));
        }
    }

    loadQuantizedInput(quantized_input);
}

void ConvLayer::loadQuantizedWeights(const std::vector<std::int8_t>& weights) {
    const std::size_t expected_size = static_cast<std::size_t>(outputChannels()) * inputChannels() *
                                      kernelSize() * kernelSize();
    if (weights.size() != expected_size) {
        throw std::invalid_argument("Quantized weight tensor size does not match layer shape.");
    }

    weights_.raw() = weights;
}

void ConvLayer::loadQuantizedBias(const std::vector<std::int32_t>& bias) {
    const std::size_t expected_size = static_cast<std::size_t>(outputChannels());
    if (bias.size() != expected_size) {
        throw std::invalid_argument("Quantized bias tensor size does not match layer shape.");
    }

    bias_.raw() = bias;
}

void ConvLayer::zeroOutput() {
    std::fill(output_.raw().begin(), output_.raw().end(), 0);
}

std::vector<Task> ConvLayer::generateTasks() const {
    std::vector<Task> tasks;
    tasks.reserve(static_cast<std::size_t>(outputChannels()) * outputHeight() * outputWidth());

    int task_id = 0;
    for (int oc = 0; oc < outputChannels(); ++oc) {
        for (int oy = 0; oy < outputHeight(); ++oy) {
            for (int ox = 0; ox < outputWidth(); ++ox) {
                tasks.emplace_back(task_id, oc, oy, ox);
                ++task_id;
            }
        }
    }
    return tasks;
}

std::int8_t ConvLayer::readActivation(int cin, int y, int x) const {
    if (cin < 0 || cin >= inputChannels()) {
        throw std::out_of_range("Input channel out of range.");
    }
    if (y < 0 || y >= inputHeight() || x < 0 || x >= inputWidth()) {
        return 0;
    }
    return input_(cin, y, x);
}

std::int8_t ConvLayer::readWeight(int oc, int cin, int ky, int kx) const {
    if (oc < 0 || oc >= outputChannels() || cin < 0 || cin >= inputChannels() || ky < 0 ||
        ky >= kernelSize() || kx < 0 || kx >= kernelSize()) {
        throw std::out_of_range("Weight index out of range.");
    }
    return weights_(oc, cin, ky, kx);
}

std::int32_t ConvLayer::readBias(int oc) const {
    if (oc < 0 || oc >= outputChannels()) {
        throw std::out_of_range("Bias index out of range.");
    }
    return hasBias() ? bias_(oc) : 0;
}

void ConvLayer::writeOutput(int oc, int oy, int ox, std::int32_t value) {
    if (oc < 0 || oc >= outputChannels() || oy < 0 || oy >= outputHeight() || ox < 0 ||
        ox >= outputWidth()) {
        throw std::out_of_range("Output index out of range.");
    }
    output_(oc, oy, ox) = value;
}

const Tensor3D<std::int8_t>& ConvLayer::inputTensor() const {
    return input_;
}

const Tensor4D<std::int8_t>& ConvLayer::weightTensor() const {
    return weights_;
}

const Tensor1D<std::int32_t>& ConvLayer::biasTensor() const {
    return bias_;
}

const Tensor3D<std::int32_t>& ConvLayer::outputTensor() const {
    return output_;
}

void ConvLayer::validateConfig() const {
    if (config_.input_height <= 0 || config_.input_width <= 0) {
        throw std::invalid_argument("Input dimensions must be positive.");
    }
    if (config_.kernel_size <= 0 || config_.stride <= 0) {
        throw std::invalid_argument("Kernel size and stride must be positive.");
    }
    if (config_.input_channels <= 0 || config_.output_channels <= 0) {
        throw std::invalid_argument("Channel counts must be positive.");
    }
    if (config_.padding < 0) {
        throw std::invalid_argument("Padding cannot be negative.");
    }
    if (outputHeight() <= 0 || outputWidth() <= 0) {
        throw std::invalid_argument("Convolution parameters produce non-positive output shape.");
    }
}
