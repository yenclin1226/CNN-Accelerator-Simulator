#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct MNISTSample {
    int label{0};
    std::vector<int> pixels;
};

MNISTSample readMNISTSample(const std::string& path, int row_index);
std::vector<int> readMNISTRow(const std::string& path, int row_index);
std::vector<std::int8_t> readQuantizedConvWeights(const std::string& path,
                                                  int output_channels,
                                                  int input_channels,
                                                  int kernel_size);
std::vector<std::int32_t> readQuantizedConvBias(const std::string& path, int output_channels);
