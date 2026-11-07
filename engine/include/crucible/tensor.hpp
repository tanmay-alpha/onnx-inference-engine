#pragma once

// =============================================================================
// Crucible — Tensor class
//
// Multi-dimensional float32 tensor with row-major contiguous storage.
// Owns its data with std::vector<float> — no shared_ptr, no reference counting.
// Every Tensor is its own copy. The cost is a memcpy on operator=, the win
// is predictable performance and no cache-line bouncing on shared mutation.
//
// Design choices:
//   - Shape: std::vector<int64_t>. int64 to match ONNX's int64 dimensions
//     without surprise truncation when models are large.
//   - Storage: std::vector<float>. Heap-allocated only for non-empty tensors;
//     an empty tensor has data() == nullptr.
//   - Layout: row-major (C-order). PyTorch / NumPy / ONNX Runtime all default
//     to row-major, so the Python bindings can hand buffers to NumPy
//     without transposing.
//   - Indexing: at({i, j, k}) computes offset = i * (D2*D3) + j * D3 + k.
//     O(rank) per access. Real hot loops will use data() directly.
//   - Bounds checks: at() throws std::out_of_range on bad indices. These
//     checks are the difference between a confusing segfault and a clear
//     error message when models are malformed. They cost ~2 ns per call —
//     a rounding error compared to the operator work.
//   - reshape / flatten: return NEW tensors (no aliasing). Source untouched.
//     Copying a 224*224*3 float buffer is 600 KB — cheap, and removes a
//     whole class of lifetime bugs in the graph executor.
// =============================================================================

#include <cstdint>
#include <iosfwd>
#include <stdexcept>
#include <vector>

namespace crucible {

// ---- Per-thread allocation budget ------------------------------------------
// Models are untrusted input: a few bytes of ONNX can describe a tensor of
// billions of elements (e.g. Conv with huge pads). While a ScopedElementLimit
// is alive on a thread, every Tensor allocation and every large engine
// temporary on that thread is checked against `max_elements` and fails with
// ResourceLimitError instead of exhausting memory. 0 = unlimited.
class ResourceLimitError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class ScopedElementLimit {
public:
    explicit ScopedElementLimit(int64_t max_elements);
    ~ScopedElementLimit();
    ScopedElementLimit(const ScopedElementLimit&) = delete;
    ScopedElementLimit& operator=(const ScopedElementLimit&) = delete;
private:
    int64_t previous_;
};

// Throws ResourceLimitError if `elements` exceeds the active limit.
void check_element_budget(int64_t elements, const char* what);

// Wall-clock deadline for the current thread's run. While a ScopedDeadline
// is alive, check_deadline() throws ResourceLimitError once it has passed.
// The executor calls it between nodes and the heavy kernels call it in
// their outer loops, so a run stops shortly after its deadline instead of
// holding a worker indefinitely. timeout_ms <= 0 = no deadline.
class ScopedDeadline {
public:
    explicit ScopedDeadline(int64_t timeout_ms);
    ~ScopedDeadline();
    ScopedDeadline(const ScopedDeadline&) = delete;
    ScopedDeadline& operator=(const ScopedDeadline&) = delete;
private:
    int64_t previous_;
};

void check_deadline(const char* where);

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

    // ---- Shape operations ---------------------------------------------------

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