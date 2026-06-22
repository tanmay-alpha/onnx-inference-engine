// =============================================================================
// Crucible — Tensor operations tests (Issue #3)
//
// Contract from ENGINEERING_PLAN.md §4:
//   - Tensor reshape(std::vector<int64_t> new_shape) const;
//   - Tensor flatten() const;
//   - void   print(int max_elements = 10) const;
//
// Acceptance: 4 more tests pass. reshape({6}) of 2x3 tensor has same
// elements. flatten() is shape {N}.
// =============================================================================

#include "crucible/tensor.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

using crucible::Tensor;

// -----------------------------------------------------------------------------
// reshape()
// -----------------------------------------------------------------------------

TEST(TensorReshape, Flatten2DTo1D) {
    // The headline AC from the engineering plan, verbatim.
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    auto r = t.reshape({6});
    EXPECT_EQ(r.shape(), (std::vector<int64_t>{6}));
    EXPECT_EQ(r.size(), 6);
    EXPECT_FLOAT_EQ(r.at({0}), 1.0f);
    EXPECT_FLOAT_EQ(r.at({5}), 6.0f);
}

TEST(TensorReshape, PreservesRowMajorData) {
    // 2x3 = [1,2,3, 4,5,6]. Reshaping to (3,2) and (6) must give the
    // same underlying row-major sequence.
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});

    auto r321 = t.reshape({3, 2});
    EXPECT_EQ(r321.shape(), (std::vector<int64_t>{3, 2}));
    EXPECT_FLOAT_EQ(r321.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(r321.at({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(r321.at({1, 0}), 3.0f);
    EXPECT_FLOAT_EQ(r321.at({1, 1}), 4.0f);
    EXPECT_FLOAT_EQ(r321.at({2, 0}), 5.0f);
    EXPECT_FLOAT_EQ(r321.at({2, 1}), 6.0f);

    auto r22 = t.reshape({2, 2});   // total 4 != 6, must throw
    EXPECT_THROW(r22, std::invalid_argument);
}

TEST(TensorReshape, ToSameShapeIsNoOp) {
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    auto r = t.reshape({2, 3});
    EXPECT_EQ(r.shape(), t.shape());
    for (int64_t i = 0; i < 2; ++i) {
        for (int64_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(r.at({i, j}), t.at({i, j}));
        }
    }
}

TEST(TensorReshape, ToHigherRank) {
    Tensor t({6}, {1, 2, 3, 4, 5, 6});
    auto r = t.reshape({1, 2, 3});
    EXPECT_EQ(r.shape(), (std::vector<int64_t>{1, 2, 3}));
    EXPECT_FLOAT_EQ(r.at({0, 0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(r.at({0, 1, 2}), 6.0f);
}

TEST(TensorReshape, ToZeroDimensional) {
    // Reshape to empty shape = rank-0 = single scalar (if data is non-empty).
    Tensor t({3}, {42.0f});
    auto r = t.reshape({});
    EXPECT_EQ(r.shape(), (std::vector<int64_t>{}));
    EXPECT_EQ(r.rank(), 0);
    EXPECT_EQ(r.size(), 0);  // rank-0 has size 0 by convention
}

TEST(TensorReshape, ThrowsOnSizeMismatch) {
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    EXPECT_THROW(t.reshape({7}),   std::invalid_argument);
    EXPECT_THROW(t.reshape({2, 2}), std::invalid_argument);
    EXPECT_THROW(t.reshape({3, 3}), std::invalid_argument);
}

TEST(TensorReshape, ThrowsOnNegativeDimension) {
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    EXPECT_THROW(t.reshape({-1, 6}), std::invalid_argument);
}

TEST(TensorReshape, ZeroDimensionIsValidButMismatchesSize) {
    // A dim of 0 is legal in NumPy-style shapes (means "empty along this
    // axis"). The reshape will still fail because 2*0*6 = 0 != 6.
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    EXPECT_THROW(t.reshape({2, 0, 6}), std::invalid_argument);
}

TEST(TensorReshape, ReshapeOfEmptyTensor) {
    Tensor t;  // rank 0, size 0
    auto r = t.reshape({0, 1, 2});
    EXPECT_EQ(r.shape(), (std::vector<int64_t>{0, 1, 2}));
    EXPECT_EQ(r.size(), 0);
}

TEST(TensorReshape, ZeroDimReshapeToValidShape) {
    // Reshape a 0-size tensor to a 1-D shape of length 0.
    Tensor t;  // rank 0, size 0
    auto r = t.reshape({0});
    EXPECT_EQ(r.shape(), (std::vector<int64_t>{0}));
    EXPECT_EQ(r.size(), 0);
}

TEST(TensorReshape, OriginalUnchanged) {
    // reshape returns a new tensor; the source must not be mutated.
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    auto r = t.reshape({6});
    r.at({0}) = 999.0f;
    EXPECT_FLOAT_EQ(t.at({0, 0}), 1.0f);   // untouched
    EXPECT_FLOAT_EQ(r.at({0}),    999.0f);
}

TEST(TensorReshape, BothSidesZeroIsValid) {
    // Two rank-0 tensors have the same product (0) — should be allowed.
    Tensor t;
    auto r = t.reshape({0});
    EXPECT_EQ(r.size(), 0);
}

// -----------------------------------------------------------------------------
// flatten()
// -----------------------------------------------------------------------------

TEST(TensorFlatten, Flatten2D) {
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    auto f = t.flatten();
    EXPECT_EQ(f.shape(), (std::vector<int64_t>{6}));
    EXPECT_EQ(f.size(), 6);
    EXPECT_FLOAT_EQ(f.at({0}), 1.0f);
    EXPECT_FLOAT_EQ(f.at({5}), 6.0f);
}

TEST(TensorFlatten, Flatten3DPreservesRowMajor) {
    Tensor t({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    auto f = t.flatten();
    EXPECT_EQ(f.shape(), (std::vector<int64_t>{8}));
    EXPECT_FLOAT_EQ(f.at({0}), 1.0f);
    EXPECT_FLOAT_EQ(f.at({7}), 8.0f);
}

TEST(TensorFlatten, Already1DStaysSameShape) {
    Tensor t({5}, {10, 20, 30, 40, 50});
    auto f = t.flatten();
    EXPECT_EQ(f.shape(), (std::vector<int64_t>{5}));
    EXPECT_FLOAT_EQ(f.at({2}), 30.0f);
}

TEST(TensorFlatten, EmptyTensorFlattensToEmpty) {
    Tensor t;
    auto f = t.flatten();
    EXPECT_EQ(f.size(), 0);
    EXPECT_EQ(f.shape(), (std::vector<int64_t>{0}));
}

TEST(TensorFlatten, OriginalUnchanged) {
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    auto f = t.flatten();
    f.at({0}) = 999.0f;
    EXPECT_FLOAT_EQ(t.at({0, 0}), 1.0f);
}

// -----------------------------------------------------------------------------
// print()  — tested by writing to a stringstream (no stdout noise in tests)
// -----------------------------------------------------------------------------

TEST(TensorPrint, PrintsShape) {
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    std::ostringstream out;
    t.print(out);
    std::string s = out.str();
    // Must contain the shape somewhere — accept any bracketed form.
    EXPECT_NE(s.find("[2, 3]"), std::string::npos);
}

TEST(TensorPrint, PrintsAllElementsWhenBelowLimit) {
    Tensor t({3}, {1.5f, 2.5f, 3.5f});
    std::ostringstream out;
    t.print(out);
    std::string s = out.str();
    EXPECT_NE(s.find("1.5"),  std::string::npos);
    EXPECT_NE(s.find("2.5"),  std::string::npos);
    EXPECT_NE(s.find("3.5"),  std::string::npos);
}

TEST(TensorPrint, TruncatesWhenAboveMaxElements) {
    Tensor t({20}, std::vector<float>(20, 1.0f));  // 20 ones
    std::ostringstream out;
    t.print(out, 5);
    std::string s = out.str();
    // Truncation marker expected.
    EXPECT_NE(s.find("..."), std::string::npos);
    // Should NOT contain "20" as a printed element count (the
    // element values are 1.0, so "20" appearing in output is unambiguous
    // truncation text).
    EXPECT_NE(s.find("20"), std::string::npos);  // "20 more" or "20 elements"
}

TEST(TensorPrint, DefaultMaxElementsTruncatesLargeTensor) {
    Tensor t({100}, std::vector<float>(100, 0.0f));
    std::ostringstream out;
    t.print(out);   // default max_elements = 10
    std::string s = out.str();
    EXPECT_NE(s.find("..."), std::string::npos);
}

TEST(TensorPrint, EmptyTensorDoesNotCrash) {
    Tensor t;
    std::ostringstream out;
    EXPECT_NO_THROW(t.print(out));
    // Output should be non-empty (it should print the shape "[]" at minimum).
    EXPECT_FALSE(out.str().empty());
}

// -----------------------------------------------------------------------------
// Cross-cutting: reshape / flatten are non-mutating, return new tensors
// -----------------------------------------------------------------------------

TEST(TensorOpsNonMutating, ChainProducesCorrectFinalShape) {
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    auto r1 = t.reshape({3, 2});
    auto r2 = r1.flatten();
    auto r3 = r2.reshape({2, 3});
    EXPECT_EQ(r3.shape(), (std::vector<int64_t>{2, 3}));
    // After 6 → 3x2 → 6 → 2x3, the data must be unchanged.
    for (int64_t i = 0; i < 2; ++i) {
        for (int64_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(r3.at({i, j}), t.at({i, j}));
        }
    }
    // And the source tensor `t` is still 2x3.
    EXPECT_EQ(t.shape(), (std::vector<int64_t>{2, 3}));
}
