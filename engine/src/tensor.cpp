#include "crucible/tensor.hpp"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace crucible {

namespace {

// -----------------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------------

// Return the product of all dimensions in a shape. Returns 0 for an empty
// shape (so that Tensor() — rank 0 — has size 0, by convention). For
// non-empty shapes, throws if any dim <= 0.
int64_t shape_product_or_throw(const std::vector<int64_t>& shape) {
    if (shape.empty()) return 0;            // rank-0 tensor
    // A dimension of 0 is legal (matches NumPy `np.zeros((3, 0, 5))` — empty
    // along that axis). The product of any shape containing a 0 is 0, which
    // is the desired behaviour for size(). We only reject NEGATIVE dims.
    int64_t product = 1;
    for (int64_t d : shape) {
        if (d < 0) {
            throw std::invalid_argument(
                "Tensor: shape dimensions must be non-negative, got " +
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

// -----------------------------------------------------------------------------
// Shape operations (Issue #3)
// -----------------------------------------------------------------------------

Tensor Tensor::reshape(std::vector<int64_t> new_shape) const {
    // Validate the new shape first — this catches negative / zero dimensions
    // before we compare sizes. shape_product_or_throw throws std::invalid_argument.
    const int64_t new_total = shape_product_or_throw(new_shape);

    // If the new total differs from the current one, reject. Special case:
    // both rank-0 (new_total = old_total = 0) is allowed.
    if (new_total != size()) {
        throw std::invalid_argument(
            "Tensor::reshape: cannot reshape size " + std::to_string(size()) +
            " into shape with product " + std::to_string(new_total));
    }

    // Build the result. The data buffer is copied (no aliasing); the plan
    // promises Tensor is its own copy on every operation.
    Tensor out(std::move(new_shape));
    out.data_ = data_;   // copy
    return out;
}

Tensor Tensor::flatten() const {
    // For a non-empty tensor, new shape is {size()}.
    // For an empty tensor, new shape is also {0} (size() == 0).
    //   That keeps the result type-consistent: rank-1, length 0.
    return reshape({size()});
}

void Tensor::print(int max_elements) const {
    print_to(std::cout, max_elements);
}

void Tensor::print_to(std::ostream& os, int max_elements) const {
    if (max_elements < 0) {
        // Negative limit is a programming error — report it but don't crash.
        // Negative effectively means "print nothing"; we keep that as a no-op
        // (caller can pass 0 explicitly if that's what they want).
        max_elements = 0;
    }

    // Header line: shape and total element count.
    os << "Tensor shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i > 0) os << ", ";
        os << shape_[i];
    }
    os << "]  size=" << size() << "\n";

    // Data line: print up to max_elements values, then "..." if truncated.
    os << "  data = [";
    const int64_t n = std::min<int64_t>(size(), max_elements);
    for (int64_t i = 0; i < n; ++i) {
        if (i > 0) os << ", ";
        os << data_[static_cast<size_t>(i)];
    }
    if (size() > max_elements) {
        if (n > 0) os << ", ";
        os << "...";
    }
    os << "]\n";
}

} // namespace crucible