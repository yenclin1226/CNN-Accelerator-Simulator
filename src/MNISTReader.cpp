#include "MNISTReader.h"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float kWeightQuantizationScale = 127.0F;
constexpr float kActivationQuantizationScale = 127.5F;

std::string trimAsciiWhitespace(const std::string& value) {
    std::size_t begin = 0;
    while (begin < value.size() &&
           std::isspace(static_cast<unsigned char>(value[begin])) != 0) {
        ++begin;
    }

    std::size_t end = value.size();
    while (end > begin &&
           std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
        --end;
    }

    return value.substr(begin, end - begin);
}

bool isIntegerToken(const std::string& token) {
    const std::string trimmed = trimAsciiWhitespace(token);
    if (trimmed.empty()) {
        return false;
    }

    std::size_t index = 0;
    if (trimmed[index] == '+' || trimmed[index] == '-') {
        ++index;
    }
    if (index == trimmed.size()) {
        return false;
    }

    for (; index < trimmed.size(); ++index) {
        if (!std::isdigit(static_cast<unsigned char>(trimmed[index]))) {
            return false;
        }
    }

    return true;
}

std::vector<int> parseCsvIntegers(const std::string& line) {
    std::vector<int> values;
    std::stringstream line_stream(line);
    std::string token;

    while (std::getline(line_stream, token, ',')) {
        const std::string trimmed = trimAsciiWhitespace(token);
        if (!isIntegerToken(trimmed)) {
            throw std::runtime_error("Encountered non-integer token in MNIST CSV row.");
        }
        values.push_back(std::stoi(trimmed));
    }

    return values;
}

std::int8_t quantizeWeight(float value) {
    const long quantized = std::lround(static_cast<double>(value) * kWeightQuantizationScale);
    const long clamped = std::max<long>(std::numeric_limits<std::int8_t>::min(),
                                        std::min<long>(std::numeric_limits<std::int8_t>::max(),
                                                       quantized));
    return static_cast<std::int8_t>(clamped);
}

std::int32_t quantizeBias(float value) {
    const double scaled =
        static_cast<double>(value) * static_cast<double>(kWeightQuantizationScale) *
        static_cast<double>(kActivationQuantizationScale);
    const long long quantized = std::llround(scaled);
    const long long clamped =
        std::max<long long>(std::numeric_limits<std::int32_t>::min(),
                            std::min<long long>(std::numeric_limits<std::int32_t>::max(),
                                                quantized));
    return static_cast<std::int32_t>(clamped);
}

float parseFloatToken(const std::string& token) {
    const std::string trimmed = trimAsciiWhitespace(token);
    if (trimmed.empty()) {
        throw std::runtime_error("Encountered empty floating-point token.");
    }
    return std::stof(trimmed);
}

}  // namespace

MNISTSample readMNISTSample(const std::string& path, int row_index) {
    if (row_index < 0) {
        throw std::invalid_argument("MNIST row index must be non-negative.");
    }

    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open MNIST CSV file: " + path);
    }

    std::string line;
    int current_row = 0;
    bool header_checked = false;

    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }

        if (!header_checked) {
            header_checked = true;
            const std::size_t first_delimiter = line.find(',');
            const std::string first_token =
                (first_delimiter == std::string::npos) ? line : line.substr(0, first_delimiter);
            if (!isIntegerToken(first_token)) {
                continue;
            }
        }

        if (current_row != row_index) {
            ++current_row;
            continue;
        }

        const std::vector<int> values = parseCsvIntegers(line);
        if (values.size() != 785U) {
            throw std::runtime_error("MNIST CSV row must contain 785 integers including the label.");
        }

        MNISTSample sample;
        sample.label = values.front();
        sample.pixels.assign(values.begin() + 1, values.end());
        return sample;
    }

    throw std::out_of_range("Requested MNIST row index is out of range.");
}

std::vector<int> readMNISTRow(const std::string& path, int row_index) {
    return readMNISTSample(path, row_index).pixels;
}

std::vector<std::int8_t> readQuantizedConvWeights(const std::string& path,
                                                  int output_channels,
                                                  int input_channels,
                                                  int kernel_size) {
    if (output_channels <= 0 || input_channels <= 0 || kernel_size <= 0) {
        throw std::invalid_argument("Convolution weight dimensions must be positive.");
    }

    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open conv weight file: " + path);
    }

    const std::size_t expected_size = static_cast<std::size_t>(output_channels) * input_channels *
                                      kernel_size * kernel_size;
    std::vector<std::int8_t> weights(expected_size, 0);
    std::vector<bool> visited(expected_size, false);
    std::string line;
    std::size_t line_count = 0;

    while (std::getline(input, line)) {
        const std::string trimmed = trimAsciiWhitespace(line);
        if (trimmed.empty()) {
            continue;
        }

        std::istringstream line_stream(trimmed);
        int oc = 0;
        int cin = 0;
        int ky = 0;
        int kx = 0;
        std::string value_token;
        if (!(line_stream >> oc >> cin >> ky >> kx >> value_token)) {
            throw std::runtime_error("Malformed conv weight line: " + line);
        }

        if (oc < 0 || oc >= output_channels || cin < 0 || cin >= input_channels || ky < 0 ||
            ky >= kernel_size || kx < 0 || kx >= kernel_size) {
            throw std::runtime_error("Conv weight index out of range in file: " + path);
        }

        const std::size_t index =
            ((static_cast<std::size_t>(oc) * input_channels + cin) * kernel_size + ky) *
                kernel_size +
            kx;
        if (visited[index]) {
            throw std::runtime_error("Duplicate conv weight entry in file: " + path);
        }

        weights[index] = quantizeWeight(parseFloatToken(value_token));
        visited[index] = true;
        ++line_count;
    }

    if (line_count != expected_size ||
        std::find(visited.begin(), visited.end(), false) != visited.end()) {
        throw std::runtime_error("Conv weight file does not cover the expected tensor shape.");
    }

    return weights;
}

std::vector<std::int32_t> readQuantizedConvBias(const std::string& path, int output_channels) {
    if (output_channels <= 0) {
        throw std::invalid_argument("Output channel count must be positive.");
    }

    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open conv bias file: " + path);
    }

    std::vector<std::int32_t> bias(static_cast<std::size_t>(output_channels), 0);
    std::vector<bool> visited(static_cast<std::size_t>(output_channels), false);
    std::string line;
    std::size_t line_count = 0;

    while (std::getline(input, line)) {
        const std::string trimmed = trimAsciiWhitespace(line);
        if (trimmed.empty()) {
            continue;
        }

        std::istringstream line_stream(trimmed);
        int oc = 0;
        std::string value_token;
        if (!(line_stream >> oc >> value_token)) {
            throw std::runtime_error("Malformed conv bias line: " + line);
        }
        if (oc < 0 || oc >= output_channels) {
            throw std::runtime_error("Conv bias index out of range in file: " + path);
        }
        if (visited[static_cast<std::size_t>(oc)]) {
            throw std::runtime_error("Duplicate conv bias entry in file: " + path);
        }

        bias[static_cast<std::size_t>(oc)] = quantizeBias(parseFloatToken(value_token));
        visited[static_cast<std::size_t>(oc)] = true;
        ++line_count;
    }

    if (line_count != static_cast<std::size_t>(output_channels) ||
        std::find(visited.begin(), visited.end(), false) != visited.end()) {
        throw std::runtime_error("Conv bias file does not cover the expected tensor shape.");
    }

    return bias;
}
