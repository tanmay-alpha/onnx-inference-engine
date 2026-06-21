#pragma once

// =============================================================================
// Crucible — Tensor class (scaffold)
//
// Issue #1: this is the minimal interface required for `crucible_core` to
//           compile and link. The real implementation lands in Issue #2
//           (constructor + at()) and Issue #3 (reshape / flatten / print).
//
// Design note: we own the data with std::vector<float> — no shared_ptr, no
// reference counting. Every Tensor is its own copy. This is the same
// approach used by tinygrad and MLX. The cost is a memcpy on operator=,
// the win is predictable performance and no cache-line bouncing on
// shared mutation.
// =============================================================================

#include <cstdint>
#include <vector>

namespace crucible {

class Tensor {
public:
    // ---- Constructors (real impl in Issue #2) ------------------------------
    Tensor();
    explicit Tensor(std::vector<int64_t> shape, float fill = 0.0f);

    // ---- Accessors (real impl in Issue #2) ---------------------------------
    float*       data()        noexcept       { return data_.data(); }
    const float* data()  const noexcept       { return data_.data(); }

    const std::vector<int64_t>& shape() const noexcept { return shape_; }
    int64_t rank() const noexcept;
    int64_t size() const noexcept;   // total element count

private:
    std::vector<int64_t> shape_;
    std::vector<float>   data_;
};

} // namespace crucible
