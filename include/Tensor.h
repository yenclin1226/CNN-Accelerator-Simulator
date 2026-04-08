#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

template <typename T>
class Tensor1D {
public:
    Tensor1D() = default;
    explicit Tensor1D(int d0) : d0_(d0), data_(static_cast<std::size_t>(d0)) {
        if (d0_ <= 0) {
            throw std::invalid_argument("Tensor1D dimension must be positive.");
        }
    }

    int dim0() const {
        return d0_;
    }

    T& operator()(int i0) {
        return data_.at(static_cast<std::size_t>(i0));
    }

    const T& operator()(int i0) const {
        return data_.at(static_cast<std::size_t>(i0));
    }

    std::vector<T>& raw() {
        return data_;
    }

    const std::vector<T>& raw() const {
        return data_;
    }

private:
    int d0_{0};
    std::vector<T> data_;
};

template <typename T>
class Tensor3D {
public:
    Tensor3D() = default;
    Tensor3D(int d0, int d1, int d2)
        : d0_(d0), d1_(d1), d2_(d2), data_(static_cast<std::size_t>(d0) * d1 * d2) {
        if (d0_ <= 0 || d1_ <= 0 || d2_ <= 0) {
            throw std::invalid_argument("Tensor3D dimensions must be positive.");
        }
    }

    int dim0() const {
        return d0_;
    }
    int dim1() const {
        return d1_;
    }
    int dim2() const {
        return d2_;
    }

    T& operator()(int i0, int i1, int i2) {
        return data_.at(index(i0, i1, i2));
    }

    const T& operator()(int i0, int i1, int i2) const {
        return data_.at(index(i0, i1, i2));
    }

    std::vector<T>& raw() {
        return data_;
    }

    const std::vector<T>& raw() const {
        return data_;
    }

private:
    std::size_t index(int i0, int i1, int i2) const {
        return (static_cast<std::size_t>(i0) * d1_ + i1) * d2_ + i2;
    }

    int d0_{0};
    int d1_{0};
    int d2_{0};
    std::vector<T> data_;
};

template <typename T>
class Tensor4D {
public:
    Tensor4D() = default;
    Tensor4D(int d0, int d1, int d2, int d3)
        : d0_(d0),
          d1_(d1),
          d2_(d2),
          d3_(d3),
          data_(static_cast<std::size_t>(d0) * d1 * d2 * d3) {
        if (d0_ <= 0 || d1_ <= 0 || d2_ <= 0 || d3_ <= 0) {
            throw std::invalid_argument("Tensor4D dimensions must be positive.");
        }
    }

    int dim0() const {
        return d0_;
    }
    int dim1() const {
        return d1_;
    }
    int dim2() const {
        return d2_;
    }
    int dim3() const {
        return d3_;
    }

    T& operator()(int i0, int i1, int i2, int i3) {
        return data_.at(index(i0, i1, i2, i3));
    }

    const T& operator()(int i0, int i1, int i2, int i3) const {
        return data_.at(index(i0, i1, i2, i3));
    }

    std::vector<T>& raw() {
        return data_;
    }

    const std::vector<T>& raw() const {
        return data_;
    }

private:
    std::size_t index(int i0, int i1, int i2, int i3) const {
        return ((static_cast<std::size_t>(i0) * d1_ + i1) * d2_ + i2) * d3_ + i3;
    }

    int d0_{0};
    int d1_{0};
    int d2_{0};
    int d3_{0};
    std::vector<T> data_;
};
