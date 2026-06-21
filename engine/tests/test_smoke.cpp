// =============================================================================
// Crucible — smoke test (Issue #1)
//
// Verifies the scaffold compiles, links, and that the Tensor constructor
// from Issue #1's scaffold is reachable. Real Tensor unit tests (at, shape,
// rank, size) land in Issue #2.
// =============================================================================

#include "crucible/tensor.hpp"

#include <gtest/gtest.h>

#include <vector>

TEST(CrucibleSmoke, EngineCompiles) {
    // If we got here, crucible_core linked successfully.
    SUCCEED();
}

TEST(CrucibleSmoke, TensorConstructorStoresShape) {
    crucible::Tensor t({2, 3}, 0.0f);
    EXPECT_EQ(t.shape(), (std::vector<int64_t>{2, 3}));
}

TEST(CrucibleSmoke, TensorRankAndSize) {
    crucible::Tensor t({2, 3, 4}, 1.0f);
    EXPECT_EQ(t.rank(), 3);
    EXPECT_EQ(t.size(), 24);
}

TEST(CrucibleSmoke, TensorDefaultsToEmpty) {
    crucible::Tensor t;
    EXPECT_EQ(t.rank(), 0);
    EXPECT_EQ(t.size(), 0);
    EXPECT_EQ(t.data(), nullptr);  // empty vector
}
