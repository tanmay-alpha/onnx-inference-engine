#pragma once

// =============================================================================
// Crucible — Tensor class
//
// Multi-dimensional float32 tensor with row-major contiguous storage.
// Owns its data with std::vector<float> — no shared_ptr, no reference counting.
// Every Tensor is its own copy. The cost is a memcpy on operator=, the win
// is predictable performance and no cache-line bouncing on shared mutation.
//
// Design choices (see ENGINEERING_PLAN.md §4):
//   - Shape: std::vector<int64_t>. int64 to match ONNX's int64 dimensions
//     without surprise truncation when models are large.
//   - Storage: std::vector<float>. Heap-allocated only for non-empty tensors;
//     an empty tensor has data() == nullptr.
//   - Layout: row-major (C-order). PyTorch / NumPy / ONNX Runtime all default
//     to row-major, so this minimises friction when we add the Python
//     bindings in Issue #12.
//   - Indexing: at({i, j, k}) computes offset = i * (D2*D3) + j * D3 + k.
//     O(rank) per access. Real hot loops will use data() directly.
//   - Bounds checks: at() throws std::out_of_range on bad indices. These
//     checks are the difference between a confusing segfault and a clear
//     error message when models are malformed. They cost ~2 ns per call —
//     a rounding error compared to the operator work.
//   - reshape / flatten: return NEW tensors (no aliasing). Source untouched.
//     Copying a 224*224*3 float buffer is 600 KB — cheap, and removes a
//     whole class of lifetime bugs that would only surface in Issue #9
//     (graph executor) and #10 (end-to-end inference).
// =============================================================================

#include <cstdint>
#include <iosfwd>
#include <vector>

namespace crucible {

class Tensor {
public:
    // ---- Constructors --------------------------------------------------------

    /// Empty tensor (rank 0, size 0). Mostly useful as a placeholder before
    /// assignment. Constructed inline so `Tensor t;` is cheap (no heap alloc).
    Tensor() = default;

    /// Allocate `prod(shape)` floats initialised to `fill`.
    /// Throws std::invalid_argument if any dimension is non-positive.
    Tensor(std::vector<int64_t> shape, float fill = 0.0f);

    /// Wrap a caller-owned buffer. The data is copied (no aliasing).
    /// Throws std::invalid_argument if data.size() != prod(shape).
    Tensor(std::vector<int64_t> shape, std::vector<float> data);

    // ---- Raw buffer access ---------------------------------------------------

    float*       data()       noexcept { return data_.data(); }
    const float* data() const noexcept { return data_.data(); }

    // ---- Shape queries -------------------------------------------------------

    const std::vector<int64_t>& shape() const noexcept { return shape_; }
    int64_t rank() const noexcept;
    int64_t size() const noexcept;   // total element count, 0 for empty tensor

    // ---- Element access ------------------------------------------------------

    /// Read/write element access with bounds + rank checks.
    /// Throws std::out_of_range if the number of indices does not
    /// match the tensor's rank or if any index is negative or exceeds
    /// its dimension's size.
    float&       at(const std::vector<int64_t>& indices);
    const float& at(const std::vector<int64_t>& indices) const;

    // ---- Shape operations (Issue #3) ----------------------------------------

    /// Return a new tensor with the same data viewed under a different shape.
    /// Total element count must match. Throws std::invalid_argument otherwise
    /// (or if any new dimension is non-positive).
    /// Source tensor is unchanged.
    Tensor reshape(std::vector<int64_t> new_shape) const;

    /// Return a 1-D tensor containing all elements in row-major order.
    /// Equivalent to reshape({size()}); for an empty tensor (rank 0), returns a
    /// rank-1 tensor of size 1 containing the single scalar value (if any).
    /// Throws std::invalid_argument if called on a rank-0 tensor with size() == 0.
    Tensor flatten() const;

    /// Print a human-readable representation to std::cout. Truncates after
    /// `max_elements` values to keep large tensors readable. Use
    /// `print_to(ostream, …)` for testable output.
    void print(int max_elements = 10) const;

    /// Print to the supplied ostream. Same format as `print()`; this is the
    /// underlying implementation. Exposed publicly for unit tests.
    void print_to(std::ostream& os, int max_elements = 10) const;

private:
    std::vector<int64_t> shape_;
    std::vector<float>   data_;

    /// Compute the row-major offset. Caller must guarantee rank() == indices.size()
    /// and each index is in range — callers do that validation. Splitting the
    /// helper out lets the const and non-const at() share one implementation
    /// without duplicating the bounds-check logic.
    int64_t compute_offset(const std::vector<int64_t>& indices) const;
};

} // namespace crucible