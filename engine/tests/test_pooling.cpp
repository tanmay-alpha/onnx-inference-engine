// GoogleTest cases for crucible::ops::maxpool_forward and
// crucible::ops::avgpool_forward (Issue #8: pooling operators).
//
// The AC test required by the plan:
//   * MaxPoolHalvesDims — MaxPool(2,2,stride=2) on (1,32,222,222)
//                          → (1,32,111,111).
//
// We also cover average pooling, multi-channel, padding, and source
// immutability. Expected values are hand-derived (no PyTorch
// dependency at build time) and asserted to float equality — pooling
// is exact integer arithmetic for these inputs.

#include "crucible/ops/pooling.hpp"
#include "crucible/tensor.hpp"

#include <gtest/gtest.h>

#include <vector>

using crucible::Tensor;
using crucible::ops::maxpool_forward;
using crucible::ops::avgpool_forward;

namespace {

template <typename T>
Tensor make(const std::vector<int64_t>& shape, const std::vector<T>& data) {
    return Tensor(shape, data);
}

}  // namespace

// -----------------------------------------------------------------------
// MaxPool
// -----------------------------------------------------------------------

TEST(MaxPool, MaxPoolHalvesDims) {
    // AC: MaxPool(2,2) on (1, 32, 222, 222) → (1, 32, 111, 111).
    // Values don't matter for the shape assertion; we just need a
    // valid tensor. Filling with a single repeated value keeps the
    // expected output the same value at every position.
    Tensor X = make<float>({1, 32, 222, 222},
                           std::vector<float>(1 * 32 * 222 * 222, 1.0f));
    Tensor Y = maxpool_forward(X, 2, 2, 2, 2, 0, 0);
    EXPECT_EQ(Y.shape(), (std::vector<int64_t>{1, 32, 111, 111}));
}

TEST(MaxPool, Basic2x2Stride2) {
    // Hand-derived 4x4 input:
    //   1  2  3  4
    //   5  6  7  8
    //   9 10 11 12
    //  13 14 15 16
    // MaxPool 2x2 stride=2: each 2x2 quadrant's max.
    Tensor X = make<float>({1, 1, 4, 4}, {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12,
       13, 14, 15, 16
    });
    Tensor Y = maxpool_forward(X, 2, 2, 2, 2, 0, 0);
    EXPECT_EQ(Y.shape(), (std::vector<int64_t>{1, 1, 2, 2}));
    EXPECT_FLOAT_EQ(Y.at({0, 0, 0, 0}),  6.0f);
    EXPECT_FLOAT_EQ(Y.at({0, 0, 0, 1}),  8.0f);
    EXPECT_FLOAT_EQ(Y.at({0, 0, 1, 0}), 14.0f);
    EXPECT_FLOAT_EQ(Y.at({0, 0, 1, 1}), 16.0f);
}

TEST(MaxPool, ThreeByThreeStrideOne) {
    // 5x5 input values 0..24. Max of each 3x3 window is the
    // bottom-right cell of that window.
    Tensor X = make<float>({1, 1, 5, 5}, {
         0,  1,  2,  3,  4,
         5,  6,  7,  8,  9,
        10, 11, 12, 13, 14,
        15, 16, 17, 18, 19,
        20, 21, 22, 23, 24
    });
    Tensor Y = maxpool_forward(X, 3, 3, 1, 1, 0, 0);
    EXPECT_EQ(Y.shape(), (std::vector<int64_t>{1, 1, 3, 3}));
    const std::vector<float> expected = {
        12, 13, 14,
        17, 18, 19,
        22, 23, 24
    };
    for (int i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(Y.data()[i], expected[i]) << "i=" << i;
    }
}

TEST(MaxPool, ZeroPadIntroducesZeros) {
    // All-ones 3x3 input, MaxPool 3x3 stride=1 pad=1. The output
    // 3x3 has all-ones EXCEPT the four corners which only see 4
    // valid input pixels (the centre is 1, but the pad contributes
    // -inf and never wins). So corner max = 1.0.
    Tensor X = make<float>({1, 1, 3, 3}, {
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
    });
    Tensor Y = maxpool_forward(X, 3, 3, 1, 1, 1, 1);
    EXPECT_EQ(Y.shape(), (std::vector<int64_t>{1, 1, 3, 3}));
    for (int i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(Y.data()[i], 1.0f) << "i=" << i;
    }
}

TEST(MaxPool, NegativeValuesTakeLargest) {
    // Ensure we sort by value, not by absolute or anything funny.
    Tensor X = make<float>({1, 1, 2, 2}, {
        -5.0f, -2.0f,
        -3.0f, -1.0f
    });
    Tensor Y = maxpool_forward(X, 2, 2, 1, 1, 0, 0);
    EXPECT_FLOAT_EQ(Y.at({0, 0, 0, 0}), -1.0f);
}

TEST(MaxPool, MultiChannelIndependent) {
    // Two channels pooled independently. The values for channel 1
    // are 100+channel-0-values so the max of each window is
    // 100 + (max of channel 0's same window).
    Tensor X = make<float>({1, 2, 4, 4}, {
        // channel 0
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16,
        // channel 1
       101,102,103,104,
       105,106,107,108,
       109,110,111,112,
       113,114,115,116
    });
    Tensor Y = maxpool_forward(X, 2, 2, 2, 2, 0, 0);
    EXPECT_EQ(Y.shape(), (std::vector<int64_t>{1, 2, 2, 2}));
    // channel 0
    EXPECT_FLOAT_EQ(Y.at({0, 0, 0, 0}),   6.0f);
    EXPECT_FLOAT_EQ(Y.at({0, 0, 0, 1}),   8.0f);
    EXPECT_FLOAT_EQ(Y.at({0, 0, 1, 0}),  14.0f);
    EXPECT_FLOAT_EQ(Y.at({0, 0, 1, 1}),  16.0f);
    // channel 1
    EXPECT_FLOAT_EQ(Y.at({0, 1, 0, 0}), 106.0f);
    EXPECT_FLOAT_EQ(Y.at({0, 1, 0, 1}), 108.0f);
    EXPECT_FLOAT_EQ(Y.at({0, 1, 1, 0}), 114.0f);
    EXPECT_FLOAT_EQ(Y.at({0, 1, 1, 1}), 116.0f);
}

TEST(MaxPool, SourceUnchanged) {
    // Run maxpool and verify the input tensor is untouched.
    Tensor X = make<float>({1, 1, 3, 3}, {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    });
    const Tensor X_before = X;
    (void)maxpool_forward(X, 2, 2, 1, 1, 0, 0);
    EXPECT_EQ(X.data(), X_before.data());
    for (int64_t i = 0; i < X.size(); ++i) {
        EXPECT_FLOAT_EQ(X.data()[i], X_before.data()[i]);
    }
}

TEST(MaxPool, RankMismatchThrows) {
    Tensor X = make<float>({1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});  // rank 3
    EXPECT_THROW(maxpool_forward(X, 2, 2, 1, 1, 0, 0), std::invalid_argument);
}

TEST(MaxPool, ZeroStrideThrows) {
    Tensor X = make<float>({1, 1, 4, 4}, std::vector<float>(16, 0.0f));
    EXPECT_THROW(maxpool_forward(X, 2, 2, 0, 0, 0, 0), std::invalid_argument);
}

// -----------------------------------------------------------------------
// AveragePool
// -----------------------------------------------------------------------

TEST(AvgPool, Basic2x2Stride2) {
    // Same 4x4 input as MaxPool test; avg of each quadrant.
    //   top-left:  mean(1,2,5,6)  = 14/4 = 3.5
    //   top-right: mean(3,4,7,8)  = 22/4 = 5.5
    //   bot-left:  mean(9,10,13,14) = 46/4 = 11.5
    //   bot-right: mean(11,12,15,16) = 54/4 = 13.5
    Tensor X = make<float>({1, 1, 4, 4}, {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12,
       13, 14, 15, 16
    });
    Tensor Y = avgpool_forward(X, 2, 2, 2, 2, 0, 0);
    EXPECT_EQ(Y.shape(), (std::vector<int64_t>{1, 1, 2, 2}));
    EXPECT_FLOAT_EQ(Y.at({0, 0, 0, 0}),  3.5f);
    EXPECT_FLOAT_EQ(Y.at({0, 0, 0, 1}),  5.5f);
    EXPECT_FLOAT_EQ(Y.at({0, 0, 1, 0}), 11.5f);
    EXPECT_FLOAT_EQ(Y.at({0, 0, 1, 1}), 13.5f);
}

TEST(AvgPool, ThreeByThreeStrideOne) {
    // Hand-derived: output[oh,ow] = (oh*5 + ow*5 + ...) / 9, with the
    // pattern from 0..24 / 9.
    Tensor X = make<float>({1, 1, 5, 5}, {
         0,  1,  2,  3,  4,
         5,  6,  7,  8,  9,
        10, 11, 12, 13, 14,
        15, 16, 17, 18, 19,
        20, 21, 22, 23, 24
    });
    Tensor Y = avgpool_forward(X, 3, 3, 1, 1, 0, 0);
    EXPECT_EQ(Y.shape(), (std::vector<int64_t>{1, 1, 3, 3}));
    const std::vector<float> expected = {
         6.0f,  7.0f,  8.0f,
        11.0f, 12.0f, 13.0f,
        16.0f, 17.0f, 18.0f
    };
    for (int i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(Y.data()[i], expected[i]) << "i=" << i;
    }
}

TEST(AvgPool, CountIncludePadIsOne) {
    // 3x3 all-ones input, 3x3 pool, stride=1, pad=1 → 3x3 output.
    // Even at the corners, the denominator is 3*3=9, so output = 4/9
    // (only 4 valid input pixels visible through the corner window).
    Tensor X = make<float>({1, 1, 3, 3}, {
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
    });
    Tensor Y = avgpool_forward(X, 3, 3, 1, 1, 1, 1);
    EXPECT_EQ(Y.shape(), (std::vector<int64_t>{1, 1, 3, 3}));
    const float expected_corner = 4.0f / 9.0f;
    const float expected_edge   = 6.0f / 9.0f;
    const float expected_center = 9.0f / 9.0f;
    EXPECT_FLOAT_EQ(Y.at({0, 0, 0, 0}), expected_corner);
    EXPECT_FLOAT_EQ(Y.at({0, 0, 0, 1}), expected_edge);
    EXPECT_FLOAT_EQ(Y.at({0, 0, 1, 1}), expected_center);
}

TEST(AvgPool, MultiChannelIndependent) {
    // Two channels. Verify each is reduced independently.
    Tensor X = make<float>({1, 2, 2, 2}, {
        // channel 0
        1, 2,
        3, 4,
        // channel 1
        10, 20,
        30, 40
    });
    Tensor Y = avgpool_forward(X, 2, 2, 1, 1, 0, 0);
    EXPECT_EQ(Y.shape(), (std::vector<int64_t>{1, 2, 1, 1}));
    EXPECT_FLOAT_EQ(Y.at({0, 0, 0, 0}),  2.5f);  // (1+2+3+4)/4
    EXPECT_FLOAT_EQ(Y.at({0, 1, 0, 0}), 25.0f);  // (10+20+30+40)/4
}

TEST(AvgPool, MaxPoolAndAvgPoolDiffer) {
    // Sanity: with mixed positive/negative inputs, max and avg give
    // different answers for the same window. This catches any
    // accidental copy-paste between the two implementations.
    Tensor X = make<float>({1, 1, 2, 2}, {
        1,  3,
        -2, 4
    });
    Tensor M = maxpool_forward(X, 2, 2, 1, 1, 0, 0);
    Tensor A = avgpool_forward(X, 2, 2, 1, 1, 0, 0);
    EXPECT_FLOAT_EQ(M.at({0, 0, 0, 0}),  4.0f);
    EXPECT_FLOAT_EQ(A.at({0, 0, 0, 0}),  1.5f);  // (1+3-2+4)/4
}

TEST(AvgPool, SourceUnchanged) {
    Tensor X = make<float>({1, 1, 3, 3}, {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    });
    const Tensor X_before = X;
    (void)avgpool_forward(X, 2, 2, 1, 1, 0, 0);
    EXPECT_EQ(X.data(), X_before.data());
    for (int64_t i = 0; i < X.size(); ++i) {
        EXPECT_FLOAT_EQ(X.data()[i], X_before.data()[i]);
    }
}
