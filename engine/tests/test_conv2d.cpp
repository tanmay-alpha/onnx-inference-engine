// GoogleTest cases for crucible::ops::conv2d_forward (Issue #7: Conv2D).
//
// The two AC tests required by the plan are:
//   * ConvOutputShape     — shape correctness on the canonical
//                            MobileNetV2 stem: (1,3,224,224) x (32,3,3,3)
//                            stride=1 pad=0  →  (1,32,222,222).
//   * ConvMatchesPyTorch  — numerical agreement with PyTorch's
//                            F.conv2d to 1e-4 on a 5x5 input.
//
// We hand-derive the 5x5 expected output (no PyTorch dependency at
// build time, see EXPECTED_OUTPUT comment below) and assert tolerance
// 1e-4 as required by the plan.

#include "crucible/ops/conv2d.hpp"
#include "crucible/tensor.hpp"

#include <gtest/gtest.h>

#include <vector>

using crucible::Tensor;
using crucible::ops::ConvParams;
using crucible::ops::conv2d_forward;

namespace {

// Convenience: build a tensor from a flat row-major buffer.
template <typename T>
Tensor make(const std::vector<int64_t>& shape, const std::vector<T>& data) {
    return Tensor(shape, data);
}

}  // namespace

// -----------------------------------------------------------------------
// Plan AC test #1 — output shape correctness
// -----------------------------------------------------------------------

TEST(Conv2D, ConvOutputShape) {
    // AC: (1, 3, 224, 224) x kernel (32, 3, 3, 3) stride=1 pad=0
    //     → (1, 32, 222, 222)
    //
    // We don't materialise the full input data — the output shape
    // doesn't depend on values, only on shapes. We DO need a valid
    // tensor so the call doesn't trip an input check, so we fill with
    // zeros.
    Tensor input  = make<float>({1, 3, 224, 224},
                                std::vector<float>(1 * 3 * 224 * 224, 0.0f));
    Tensor weight = make<float>({32, 3, 3, 3},
                                std::vector<float>(32 * 3 * 3 * 3, 0.0f));
    Tensor bias;  // optional, empty
    ConvParams p;  // stride=1, pad=0, groups=1 (defaults)

    Tensor Y = conv2d_forward(input, weight, bias, p);

    EXPECT_EQ(Y.shape(), (std::vector<int64_t>{1, 32, 222, 222}));
}

// -----------------------------------------------------------------------
// Plan AC test #2 — matches PyTorch to 1e-4
// -----------------------------------------------------------------------
//
// Hand-derived reference. We compute Y[i,j] = Σ_{kh,kw} X[i+kh, j+kw] *
// W[kh, kw] for the 5x5 input and 3x3 kernel below.
//
// Input X (1,1,5,5):
//   0  1  2  3  4
//   5  6  7  8  9
//  10 11 12 13 14
//  15 16 17 18 19
//  20 21 22 23 24
//
// Kernel W (1,1,3,3):
//   0 1 2
//   3 4 5
//   6 7 8
//
// Output Y (1,1,3,3):
//   312 348 384
//   492 528 564
//   672 708 744
//
// (Same numbers as torch.nn.functional.conv2d with the default
// settings — cross-correlation, no padding, stride 1 — to within
// float32 round-off. We assert tolerance 1e-4 as the AC requires.)
TEST(Conv2D, ConvMatchesPyTorch) {
    Tensor X = make<float>({1, 1, 5, 5}, {
        0,  1,  2,  3,  4,
        5,  6,  7,  8,  9,
       10, 11, 12, 13, 14,
       15, 16, 17, 18, 19,
       20, 21, 22, 23, 24
    });
    Tensor W = make<float>({1, 1, 3, 3}, {
        0, 1, 2,
        3, 4, 5,
        6, 7, 8
    });
    Tensor bias;  // no bias
    ConvParams p;

    Tensor Y = conv2d_forward(X, W, bias, p);
    EXPECT_EQ(Y.shape(), (std::vector<int64_t>{1, 1, 3, 3}));

    const std::vector<float> expected = {
        312, 348, 384,
        492, 528, 564,
        672, 708, 744
    };
    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(Y.data()[i], expected[i], 1e-4f) << "i=" << i;
    }
}

// -----------------------------------------------------------------------
// Additional tests covering the rest of the Issue #7 contract
// -----------------------------------------------------------------------

TEST(Conv2D, WithBiasPerChannel) {
    // Single-channel conv with bias. Same X and W as ConvMatchesPyTorch
    // but add a bias of +10. Every output element shifts by +10.
    Tensor X = make<float>({1, 1, 5, 5}, {
        0,  1,  2,  3,  4,
        5,  6,  7,  8,  9,
       10, 11, 12, 13, 14,
       15, 16, 17, 18, 19,
       20, 21, 22, 23, 24
    });
    Tensor W = make<float>({1, 1, 3, 3}, {
        0, 1, 2,
        3, 4, 5,
        6, 7, 8
    });
    Tensor bias = make<float>({1}, {10.0f});
    ConvParams p;

    Tensor Y = conv2d_forward(X, W, bias, p);
    EXPECT_EQ(Y.shape(), (std::vector<int64_t>{1, 1, 3, 3}));

    const std::vector<float> expected = {
        322, 358, 394,
        502, 538, 574,
        682, 718, 754
    };
    for (int i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(Y.data()[i], expected[i]) << "i=" << i;
    }
}

TEST(Conv2D, MultiChannelConv) {
    // 2 input channels, 3 output channels, 2x2 kernel. The output is
    // 4x4. We hand-derive just one output element to anchor the test,
    // and rely on ConvMatchesPyTorch for full coverage of the GEMM.
    Tensor X = make<float>({1, 2, 5, 5}, {
        // channel 0: 0..24
         0,  1,  2,  3,  4,
         5,  6,  7,  8,  9,
        10, 11, 12, 13, 14,
        15, 16, 17, 18, 19,
        20, 21, 22, 23, 24,
        // channel 1: 100..124
       100,101,102,103,104,
       105,106,107,108,109,
       110,111,112,113,114,
       115,116,117,118,119,
       120,121,122,123,124
    });
    // Weight (3, 2, 2, 2): three 2x2x2 filters.
    // For output[0, 0, 0, 0] we want the sum of X[0:2, 0:2, 0:2] * W[0].
    Tensor W = make<float>({3, 2, 2, 2}, {
        // filter 0: ones
        1, 1,
        1, 1,
        1, 1,
        1, 1,
        // filter 1: ones (different scale below)
        2, 2,
        2, 2,
        2, 2,
        2, 2,
        // filter 2: zeros (so output channel 2 is all zero)
        0, 0,
        0, 0,
        0, 0,
        0, 0
    });
    Tensor bias = make<float>({3}, {0.0f, 0.0f, 0.0f});
    ConvParams p;

    Tensor Y = conv2d_forward(X, W, bias, p);
    EXPECT_EQ(Y.shape(), (std::vector<int64_t>{1, 3, 4, 4}));

    // output[0,0,0,0] = sum of X[0,0,0:2,0:2] + sum of X[0,1,0:2,0:2]
    // (filter 0 is all ones) = (0+1+5+6) + (100+101+105+106) = 12 + 412 = 424
    EXPECT_FLOAT_EQ(Y.at({0, 0, 0, 0}), 424.0f);
    // output[0,1,0,0] = 2 * (sum of both channels) = 848
    EXPECT_FLOAT_EQ(Y.at({0, 1, 0, 0}), 848.0f);
    // output[0,2,...] = 0 (filter 2 is all zeros)
    EXPECT_FLOAT_EQ(Y.at({0, 2, 0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(Y.at({0, 2, 3, 3}), 0.0f);
}

TEST(Conv2D, Stride2Spatial) {
    // (1, 1, 6, 6) input, 3x3 kernel, stride=2, no pad → 3x3 output.
    // Spot-check by independently computing Y[0,0,0,0].
    Tensor X = make<float>({1, 1, 7, 7}, {
         0,  1,  2,  3,  4,  5,  6,
         7,  8,  9, 10, 11, 12, 13,
        14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27,
        28, 29, 30, 31, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41,
        42, 43, 44, 45, 46, 47, 48
    });
    Tensor W = make<float>({1, 1, 3, 3}, {
        1, 0, 0,
        0, 0, 0,
        0, 0, 0
    });  // picks out the top-left pixel of each receptive field
    Tensor bias;
    ConvParams p;
    p.stride_h = 2;
    p.stride_w = 2;

    Tensor Y = conv2d_forward(X, W, bias, p);
    EXPECT_EQ(Y.shape(), (std::vector<int64_t>{1, 1, 3, 3}));
    // Y[0,0,0,0] = X[0,0,0,0] = 0
    EXPECT_FLOAT_EQ(Y.at({0, 0, 0, 0}), 0.0f);
    // Y[0,0,0,1] = X[0,0,0,2] = 2
    EXPECT_FLOAT_EQ(Y.at({0, 0, 0, 1}), 2.0f);
    // Y[0,0,0,2] = X[0,0,0,4] = 4
    EXPECT_FLOAT_EQ(Y.at({0, 0, 0, 2}), 4.0f);
    // Y[0,0,1,0] = X[0,0,2,0] = 14
    EXPECT_FLOAT_EQ(Y.at({0, 0, 1, 0}), 14.0f);
    // Y[0,0,2,2] = X[0,0,4,4] = 32
    EXPECT_FLOAT_EQ(Y.at({0, 0, 2, 2}), 32.0f);
}

TEST(Conv2D, ZeroPadIntroducesZeros) {
    // (1, 1, 3, 3) input, 3x3 kernel, stride=1, pad=1 → 3x3 output.
    // Center element of output equals sum of X ⊙ W (no padding effect).
    // Corner elements include the padded zeros so they're strictly
    // smaller than the centre.
    Tensor X = make<float>({1, 1, 3, 3}, {
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
    });
    Tensor W = make<float>({1, 1, 3, 3}, {
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
    });
    Tensor bias;
    ConvParams p;
    p.pad_h = 1;
    p.pad_w = 1;

    Tensor Y = conv2d_forward(X, W, bias, p);
    EXPECT_EQ(Y.shape(), (std::vector<int64_t>{1, 1, 3, 3}));
    // Center: all 9 input pixels used (each weight 1) → 9
    EXPECT_FLOAT_EQ(Y.at({0, 0, 1, 1}), 9.0f);
    // Corner: 4 input pixels used (each weight 1) → 4
    EXPECT_FLOAT_EQ(Y.at({0, 0, 0, 0}), 4.0f);
    EXPECT_FLOAT_EQ(Y.at({0, 0, 0, 2}), 4.0f);
    EXPECT_FLOAT_EQ(Y.at({0, 0, 2, 0}), 4.0f);
    EXPECT_FLOAT_EQ(Y.at({0, 0, 2, 2}), 4.0f);
    // Edge: 6 input pixels used → 6
    EXPECT_FLOAT_EQ(Y.at({0, 0, 0, 1}), 6.0f);
    EXPECT_FLOAT_EQ(Y.at({0, 0, 1, 0}), 6.0f);
    EXPECT_FLOAT_EQ(Y.at({0, 0, 1, 2}), 6.0f);
    EXPECT_FLOAT_EQ(Y.at({0, 0, 2, 1}), 6.0f);
}

TEST(Conv2D, MultiBatchIndependent) {
    // Two batches in the same conv. Each batch is X[:,0]=ones (1,1,3,3),
    // but the second batch has all 2s. Outputs should differ by a
    // factor of 2.
    Tensor X = make<float>({2, 1, 3, 3}, {
        // batch 0: ones
        1, 1, 1,
        1, 1, 1,
        1, 1, 1,
        // batch 1: twos
        2, 2, 2,
        2, 2, 2,
        2, 2, 2
    });
    Tensor W = make<float>({1, 1, 2, 2}, {
        1, 1,
        1, 1
    });
    Tensor bias;
    ConvParams p;

    Tensor Y = conv2d_forward(X, W, bias, p);
    EXPECT_EQ(Y.shape(), (std::vector<int64_t>{2, 1, 2, 2}));
    // Batch 0: each 2x2 sum = 4
    EXPECT_FLOAT_EQ(Y.at({0, 0, 0, 0}), 4.0f);
    EXPECT_FLOAT_EQ(Y.at({0, 0, 1, 1}), 4.0f);
    // Batch 1: each 2x2 sum = 8
    EXPECT_FLOAT_EQ(Y.at({1, 0, 0, 0}), 8.0f);
    EXPECT_FLOAT_EQ(Y.at({1, 0, 1, 1}), 8.0f);
}

TEST(Conv2D, SourceUnchanged) {
    // Run conv2d and verify input/weight/bias are not mutated.
    Tensor X = make<float>({1, 1, 3, 3}, {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    });
    Tensor W = make<float>({1, 1, 2, 2}, {
        1, 1,
        1, 1
    });
    Tensor bias = make<float>({1}, {1.5f});
    ConvParams p;

    const float* X_ptr = X.data();
    const float* W_ptr = W.data();
    const float* bias_ptr = bias.data();
    const Tensor X_before = X;
    const Tensor W_before = W;
    const Tensor bias_before = bias;

    (void)conv2d_forward(X, W, bias, p);

    EXPECT_EQ(X.data(), X_ptr);
    EXPECT_EQ(W.data(), W_ptr);
    EXPECT_EQ(bias.data(), bias_ptr);
    for (int64_t i = 0; i < X.size(); ++i) {
        EXPECT_FLOAT_EQ(X.data()[i], X_before.data()[i]);
    }
    for (int64_t i = 0; i < W.size(); ++i) {
        EXPECT_FLOAT_EQ(W.data()[i], W_before.data()[i]);
    }
    for (int64_t i = 0; i < bias.size(); ++i) {
        EXPECT_FLOAT_EQ(bias.data()[i], bias_before.data()[i]);
    }
}

TEST(Conv2D, RankMismatchThrows) {
    // Input must be rank 4.
    Tensor X = make<float>({1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});  // rank 3
    Tensor W = make<float>({1, 1, 2, 2}, {1, 1, 1, 1});
    ConvParams p;
    EXPECT_THROW(conv2d_forward(X, W, Tensor(), p), std::invalid_argument);
}

TEST(Conv2D, ChannelMismatchThrows) {
    Tensor X = make<float>({1, 3, 3, 3}, std::vector<float>(27, 0.0f));
    Tensor W = make<float>({1, 2, 2, 2}, std::vector<float>(8, 0.0f));  // C_in mismatch
    ConvParams p;
    EXPECT_THROW(conv2d_forward(X, W, Tensor(), p), std::invalid_argument);
}

TEST(Conv2D, BiasShapeMismatchThrows) {
    Tensor X = make<float>({1, 1, 3, 3}, std::vector<float>(9, 0.0f));
    Tensor W = make<float>({2, 1, 2, 2}, std::vector<float>(8, 0.0f));
    Tensor bias = make<float>({3}, {0.0f, 0.0f, 0.0f});  // C_out mismatch
    ConvParams p;
    EXPECT_THROW(conv2d_forward(X, W, bias, p), std::invalid_argument);
}

TEST(Conv2D, GroupsGreaterThanOneThrows) {
    Tensor X = make<float>({1, 2, 3, 3}, std::vector<float>(18, 0.0f));
    Tensor W = make<float>({2, 2, 2, 2}, std::vector<float>(16, 0.0f));
    ConvParams p;
    p.groups = 2;
    EXPECT_THROW(conv2d_forward(X, W, Tensor(), p), std::invalid_argument);
}
