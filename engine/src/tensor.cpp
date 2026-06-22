#include "crucible/tensor.hpp"

#include <numeric>
#include <stdexcept>
#include <string>

namespace crucible {

namespace {

// -----------------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------------

// Return the product of all dimensions in a shape. Returns 1 for empty shape
// (so that Tensor() — rank 0 — has size 0, not 1, by special-casing at the
// call site rather than here). For non-empty shapes, throws if any dim <= 0.
int64_t shape_product_or_throw(const std::vector<int64_t>& shape) {
    if (shape.empty()) return 0;            // rank-0 tensor
    int64_t product = 1;
    for (int64_t d : shape) {
        if (d <= 0) {
            throw std::invalid_argument(
                "Tensor: shape dimensions must be positive, got " +
                std::to_string(d));
        }
        product *= d;
    }
    return product;
}

} // namespace

// -----------------------------------------------------------------------------
// Constructors
// -----------------------------------------------------------------------------

Tensor::Tensor(std::vector<int64_t> shape, float fill)
    : shape_(std::move(shape))
{
    const int64_t n = shape_product_or_throw(shape_);
    if (n > 0) {
        data_.assign(static_cast<size_t>(n), fill);
    }
    // For rank-0 tensors we deliberately leave data_ empty: size() returns 0
    // and data() returns nullptr. This is the convention used by NumPy for
    // 0-d arrays.
}

Tensor::Tensor(std::vector<int64_t> shape, std::vector<float> data)
    : shape_(std::move(shape))
{
    const int64_t expected = shape_product_or_throw(shape_);
    if (static_cast<int64_t>(data.size()) != expected) {
        throw std::invalid_argument(
            "Tensor: data size " + std::to_string(data.size()) +
            " does not match shape product " + std::to_string(expected));
    }
    data_ = std::move(data);
}

// -----------------------------------------------------------------------------
// Shape queries
// -----------------------------------------------------------------------------

int64_t Tensor::rank() const noexcept {
    return static_cast<int64_t>(shape_.size());
}

int64_t Tensor::size() const noexcept {
    return static_cast<int64_t>(data_.size());
}

// -----------------------------------------------------------------------------
// Element access
// -----------------------------------------------------------------------------

float& Tensor::at(const std::vector<int64_t>& indices) {
    // Discard the const-correct offset result; we re-do it without const to
    // get a writable reference. The bounds checks live in compute_offset so
    // they can't drift between the two overloads.
    const int64_t offset = compute_offset(indices);
    return data_[static_cast<size_t>(offset)];
}

const float& Tensor::at(const std::vector<int64_t>& indices) const {
    const int64_t offset = compute_offset(indices);
    return data_[static_cast<size_t>(offset)];
}

int64_t Tensor::compute_offset(const std::vector<int64_t>& indices) const {
    if (static_cast<int64_t>(indices.size()) != rank()) {
        throw std::out_of_range(
            "Tensor::at: rank mismatch (got " +
            std::to_string(indices.size()) + ", expected " +
            std::to_string(rank()) + ")");
    }

    int64_t offset = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        const int64_t idx = indices[i];
        const int64_t dim = shape_[i];
        if (idx < 0 || idx >= dim) {
            throw std::out_of_range(
                "Tensor::at: index " + std::to_string(idx) +
                " out of range for dim " + std::to_string(dim) +
                " at axis " + std::to_string(i));
        }
        // Stride for axis i is the product of all dimensions after i.
        int64_t stride = 1;
        for (size_t j = i + 1; j < shape_.size(); ++j) {
            stride *= shape_[j];
        }
        offset += idx * stride;
    }
    return offset;
}

} // namespace crucible