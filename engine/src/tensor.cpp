#include "crucible/tensor.hpp"

#include <numeric>
#include <stdexcept>

namespace crucible {

// -----------------------------------------------------------------------------
// Tensor — scaffold implementation
//
// Issue #1: only what CMake needs to link a real .o file. Real logic (at(),
// reshape, flatten, print) lands in Issues #2 and #3.
// -----------------------------------------------------------------------------

Tensor::Tensor() = default;

Tensor::Tensor(std::vector<int64_t> shape, float fill)
    : shape_(std::move(shape))
{
    const int64_t n = size();
    if (n < 0) {
        throw std::invalid_argument("Tensor: negative dimension in shape");
    }
    data_.assign(static_cast<size_t>(n), fill);
}

int64_t Tensor::rank() const noexcept {
    return static_cast<int64_t>(shape_.size());
}

int64_t Tensor::size() const noexcept {
    if (shape_.empty()) return 0;
    int64_t n = 1;
    for (int64_t d : shape_) {
        if (d <= 0) return -1;       // signal: not yet a known total
        n *= d;
    }
    return n;
}

} // namespace crucible
