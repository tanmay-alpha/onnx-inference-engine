// =============================================================================
// Crucible — Tensor unit tests (Issue #2)
//
// Contract from ENGINEERING_PLAN.md §4:
//   - Tensor(std::vector<int64_t> shape, float fill = 0.0f)
//   - Tensor(std::vector<int64_t> shape, std::vector<float> data)
//   - float& at(std::vector<int64_t> indices);
//   - const float& at(std::vector<int64_t> indices) const;
//   - float* data();
//   - const float* data() const;
//   - std::vector<int64_t> shape() const;
//   - int64_t size() const;
//   - int64_t rank() const;
//
// Acceptance: Tensor({2,3}).at({1,2}) returns the correct element.
// =============================================================================

#include "crucible/tensor.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

using crucible::Tensor;

// -----------------------------------------------------------------------------
// Construction & shape
// -----------------------------------------------------------------------------

TEST(TensorTest, ConstructorSetsShape) {
    Tensor t({2, 3});
    EXPECT_EQ(t.shape(), (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(t.size(), 6);
    EXPECT_EQ(t.rank(), 2);
}

TEST(TensorTest, FillConstructorZeroByDefault) {
    Tensor t({2, 3});
    for (int64_t i = 0; i < 2; ++i) {
        for (int64_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(t.at({i, j}), 0.0f);
        }
    }
}

TEST(TensorTest, FillConstructorAppliesValue) {
    Tensor t({2, 3}, -1.5f);
    for (int64_t i = 0; i < 2; ++i) {
        for (int64_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(t.at({i, j}), -1.5f);
        }
    }
}

TEST(TensorTest, DataConstructorStoresElements) {
    // Row-major 2x3: [1,2,3, 4,5,6]
    Tensor t({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    EXPECT_FLOAT_EQ(t.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(t.at({0, 2}), 3.0f);
    EXPECT_FLOAT_EQ(t.at({1, 0}), 4.0f);
    EXPECT_FLOAT_EQ(t.at({1, 2}), 6.0f);
}

TEST(TensorTest, DataConstructorThrowsOnSizeMismatch) {
    EXPECT_THROW(Tensor({2, 3}, {1.0f, 2.0f}), std::invalid_argument);
}

TEST(TensorTest, RankFor3D) {
    Tensor t({2, 3, 4});
    EXPECT_EQ(t.rank(), 3);
    EXPECT_EQ(t.size(), 24);
}

TEST(TensorTest, RankFor1D) {
    Tensor t({5});
    EXPECT_EQ(t.rank(), 1);
    EXPECT_EQ(t.size(), 5);
    EXPECT_EQ(t.shape(), (std::vector<int64_t>{5}));
}

// -----------------------------------------------------------------------------
// at() — the headline feature
// -----------------------------------------------------------------------------

TEST(TensorTest, AtReturnsCorrectElement) {
    // This is the AC from the engineering plan, verbatim.
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    EXPECT_FLOAT_EQ(t.at({1, 2}), 6.0f);
}

TEST(TensorTest, AtIsRowMajor) {
    // For shape {2,3} the offset for {i,j} is i*3 + j.
    Tensor t({2, 3}, {0, 1, 2, 3, 4, 5});
    EXPECT_FLOAT_EQ(t.at({0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(t.at({0, 1}), 1.0f);
    EXPECT_FLOAT_EQ(t.at({0, 2}), 2.0f);
    EXPECT_FLOAT_EQ(t.at({1, 0}), 3.0f);
    EXPECT_FLOAT_EQ(t.at({1, 1}), 4.0f);
    EXPECT_FLOAT_EQ(t.at({1, 2}), 5.0f);
}

TEST(TensorTest, AtWorksForHigherRank) {
    // shape {2,2,2} → 8 elements, row-major.
    // Linear index = i*4 + j*2 + k
    Tensor t({2, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7});
    EXPECT_FLOAT_EQ(t.at({0, 0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(t.at({0, 0, 1}), 1.0f);
    EXPECT_FLOAT_EQ(t.at({0, 1, 0}), 2.0f);
    EXPECT_FLOAT_EQ(t.at({1, 0, 0}), 4.0f);
    EXPECT_FLOAT_EQ(t.at({1, 1, 1}), 7.0f);
}

TEST(TensorTest, AtIsWritable) {
    Tensor t({2, 2}, 0.0f);
    t.at({0, 0}) = 1.0f;
    t.at({0, 1}) = 2.0f;
    t.at({1, 0}) = 3.0f;
    t.at({1, 1}) = 4.0f;
    EXPECT_FLOAT_EQ(t.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(t.at({1, 1}), 4.0f);
}

// -----------------------------------------------------------------------------
// data() — raw pointer access
// -----------------------------------------------------------------------------

TEST(TensorTest, DataPointerMatchesRowMajor) {
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    const float* p = t.data();
    ASSERT_NE(p, nullptr);
    EXPECT_FLOAT_EQ(p[0], 1.0f);
    EXPECT_FLOAT_EQ(p[1], 2.0f);
    EXPECT_FLOAT_EQ(p[2], 3.0f);
    EXPECT_FLOAT_EQ(p[3], 4.0f);
    EXPECT_FLOAT_EQ(p[4], 5.0f);
    EXPECT_FLOAT_EQ(p[5], 6.0f);
}

TEST(TensorTest, DataPointerMutableThroughAt) {
    Tensor t({3}, 0.0f);
    t.at({0}) = 10.0f;
    t.at({1}) = 20.0f;
    t.at({2}) = 30.0f;
    EXPECT_FLOAT_EQ(t.data()[0], 10.0f);
    EXPECT_FLOAT_EQ(t.data()[2], 30.0f);
}

// -----------------------------------------------------------------------------
// Bounds checking (defensive — Tensor will be hot-pathed later)
// -----------------------------------------------------------------------------

TEST(TensorTest, AtThrowsOnOutOfBoundsIndex) {
    Tensor t({2, 3});
    EXPECT_THROW(t.at({2, 0}),  std::out_of_range);   // i >= 2
    EXPECT_THROW(t.at({0, 3}),  std::out_of_range);   // j >= 3
    EXPECT_THROW(t.at({-1, 0}), std::out_of_range);
}

TEST(TensorTest, AtThrowsOnRankMismatch) {
    Tensor t({2, 3});
    EXPECT_THROW(t.at({0}),          std::out_of_range);   // too few
    EXPECT_THROW(t.at({0, 0, 0}),    std::out_of_range);   // too many
}
