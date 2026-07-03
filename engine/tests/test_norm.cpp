// GoogleTest cases for crucible::ops::batchnorm_forward
// (Issue #8: BatchNormalization inference mode).
//
// The AC test required by the plan:
//   * BatchNormNormalizes — when running_mean=0 and running_var=1,
//                           the output should be close to the input
//                           (only off by the scale/bias affine and
//                           the epsilon in the denominator).
//
// We also cover per-channel affine transforms, the variance-shift
// case (the most common usage after training), and the epsilon
// floor. Expected values are hand-derived (no PyTorch dependency at
// build time).

#include "crucible/ops/norm.hpp"
#include "crucible/tensor.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using crucible::Tensor;
using crucible::ops::batchnorm_forward;

namespace {

template <typename T>
Tensor make(const std::vector<int64_t>& shape, const std::vector<T>& data) {
    return Tensor(shape, data);
}

}  // namespace

// -----------------------------------------------------------------------
// Plan AC test
// -----------------------------------------------------------------------

TEST(BatchNorm, BatchNormNormalizes) {
    // AC: BN with running_mean=0, running_var=1 → output ≈ input.
    // We also set scale=1, bias=0 so the only residual error is from
    // epsilon in the denominator. With the default eps=1e-5 the
    // relative error is ~1e-5, well inside EXPECT_NEAR's 1e-4.
    //
    // Input X = 1x2x2x2 with values 0..7.
    Tensor X = make<float>({1, 2, 2, 2}, {
        0, 1, 2, 3,
        4, 5, 6, 7
    });
    Tensor scale        = make<float>({2}, {1.0f, 1.0f});
    Tensor bias         = make<float>({2}, {0.0f, 0.0f});
    Tensor running_mean = make<float>({2}, {0.0f, 0.0f});
    Tensor running_var  = make<float>({2}, {1.0f, 1.0f});

    Tensor Y = batchnorm_forward(X, scale, bias, running_mean, running_var);
    EXPECT_EQ(Y.shape(), X.shape());
    for (int64_t i = 0; i < X.size(); ++i) {
        EXPECT_NEAR(Y.data()[i], X.data()[i], 1e-4f) << "i=" << i;
    }
}

// -----------------------------------------------------------------------
// Additional coverage
// -----------------------------------------------------------------------

TEST(BatchNorm, SingleElementIdentityCase) {
    // (1, 1, 1, 1) tensor with a single value 2.0. With scale=0.5,
    // bias=1.0, mean=0, var=1, epsilon=1e-5:
    //   a = 0.5 / sqrt(1.00001) ≈ 0.4999975
    //   b = 1.0 - 0 * a        = 1.0
    //   y = 0.4999975 * 2 + 1  = 1.999995
    Tensor X           = make<float>({1, 1, 1, 1}, {2.0f});
    Tensor scale       = make<float>({1}, {0.5f});
    Tensor bias        = make<float>({1}, {1.0f});
    Tensor mean        = make<float>({1}, {0.0f});
    Tensor var         = make<float>({1}, {1.0f});
    Tensor Y = batchnorm_forward(X, scale, bias, mean, var);
    EXPECT_EQ(Y.shape(), (std::vector<int64_t>{1, 1, 1, 1}));
    const float expected = 0.5f / std::sqrt(1.0f + 1e-5f) * 2.0f + 1.0f;
    EXPECT_NEAR(Y.at({0, 0, 0, 0}), expected, 1e-5f);
}

TEST(BatchNorm, AffineRescalesAndShifts) {
    // Hand-derived: mean=5, var=4, scale=2, bias=3, eps=1e-5.
    //   a = 2 / sqrt(4 + 1e-5) ≈ 1.0 (within 1e-5)
    //   b = 3 - 5 * a          ≈ -2.0
    //   y = a * x + b
    //   x=10 → y ≈ 8.0
    //   x=5  → y ≈ 3.0
    //   x=1  → y ≈ -1.0
    Tensor X = make<float>({1, 1, 3, 1}, {10.0f, 5.0f, 1.0f});
    Tensor scale = make<float>({1}, {2.0f});
    Tensor bias  = make<float>({1}, {3.0f});
    Tensor mean  = make<float>({1}, {5.0f});
    Tensor var   = make<float>({1}, {4.0f});
    Tensor Y = batchnorm_forward(X, scale, bias, mean, var);
    const float a = 2.0f / std::sqrt(4.0f + 1e-5f);
    const float b = 3.0f - 5.0f * a;
    EXPECT_NEAR(Y.at({0, 0, 0, 0}), a * 10.0f + b, 1e-5f);
    EXPECT_NEAR(Y.at({0, 0, 1, 0}), a *  5.0f + b, 1e-5f);
    EXPECT_NEAR(Y.at({0, 0, 2, 0}), a *  1.0f + b, 1e-5f);
}

TEST(BatchNorm, MultiChannelPerChannel) {
    // Two channels with different running stats and affine params.
    //   channel 0: mean=0, var=1, scale=1, bias=0 → y = x / sqrt(1+eps) ≈ x
    //   channel 1: mean=10, var=4, scale=1, bias=0 → a=0.5, b=-5
    //                y = 0.5*x - 5
    Tensor X = make<float>({1, 2, 2, 2}, {
        // channel 0
        1.0f, 2.0f,
        3.0f, 4.0f,
        // channel 1
        10.0f, 12.0f,
        14.0f, 16.0f
    });
    Tensor scale = make<float>({2}, {1.0f, 1.0f});
    Tensor bias  = make<float>({2}, {0.0f, 0.0f});
    Tensor mean  = make<float>({2}, { 0.0f, 10.0f});
    Tensor var   = make<float>({2}, { 1.0f,  4.0f});
    Tensor Y = batchnorm_forward(X, scale, bias, mean, var);
    EXPECT_EQ(Y.shape(), (std::vector<int64_t>{1, 2, 2, 2}));

    // Channel 0: y ≈ x.
    EXPECT_NEAR(Y.at({0, 0, 0, 0}), 1.0f, 1e-4f);
    EXPECT_NEAR(Y.at({0, 0, 1, 1}), 4.0f, 1e-4f);

    // Channel 1: y = 0.5*x - 5.
    EXPECT_NEAR(Y.at({0, 1, 0, 0}), 0.0f,  1e-4f);  // 0.5*10 - 5
    EXPECT_NEAR(Y.at({0, 1, 0, 1}), 1.0f,  1e-4f);  // 0.5*12 - 5
    EXPECT_NEAR(Y.at({0, 1, 1, 0}), 2.0f,  1e-4f);  // 0.5*14 - 5
    EXPECT_NEAR(Y.at({0, 1, 1, 1}), 3.0f,  1e-4f);  // 0.5*16 - 5
}

TEST(BatchNorm, EpsilonFloorsVariance) {
    // When running_var = 0, the only thing keeping the division
    // finite is epsilon. Use a custom eps=0.01 to make the effect
    // large enough to test without being huge.
    //   a = 1 / sqrt(0 + 0.01) = 1 / 0.1 = 10
    //   b = 0 - 0 * 10 = 0
    //   y = 10 * x
    Tensor X     = make<float>({1, 1, 1, 1}, {1.0f});
    Tensor scale = make<float>({1}, {1.0f});
    Tensor bias  = make<float>({1}, {0.0f});
    Tensor mean  = make<float>({1}, {0.0f});
    Tensor var   = make<float>({1}, {0.0f});
    Tensor Y = batchnorm_forward(X, scale, bias, mean, var, /*epsilon=*/0.01f);
    EXPECT_NEAR(Y.at({0, 0, 0, 0}), 10.0f, 1e-5f);
}

TEST(BatchNorm, MultiBatchIndependentAffine) {
    // Two batches with the same channel stats but different x.
    // The affine is per-channel, so both batches are transformed
    // the same way.
    Tensor X = make<float>({2, 1, 2, 1}, {
        // batch 0
        0.0f, 2.0f,
        // batch 1
        4.0f, 6.0f
    });
    Tensor scale = make<float>({1}, {1.0f});
    Tensor bias  = make<float>({1}, {0.0f});
    Tensor mean  = make<float>({1}, {0.0f});
    Tensor var   = make<float>({1}, {1.0f});
    Tensor Y = batchnorm_forward(X, scale, bias, mean, var);
    EXPECT_EQ(Y.shape(), (std::vector<int64_t>{2, 1, 2, 1}));
    EXPECT_NEAR(Y.at({0, 0, 0, 0}), 0.0f, 1e-4f);
    EXPECT_NEAR(Y.at({0, 0, 1, 0}), 2.0f, 1e-4f);
    EXPECT_NEAR(Y.at({1, 0, 0, 0}), 4.0f, 1e-4f);
    EXPECT_NEAR(Y.at({1, 0, 1, 0}), 6.0f, 1e-4f);
}

TEST(BatchNorm, SourceUnchanged) {
    // Run BN and verify all inputs are unmodified.
    Tensor X     = make<float>({1, 1, 2, 2}, {1, 2, 3, 4});
    Tensor scale = make<float>({1}, {1.0f});
    Tensor bias  = make<float>({1}, {0.5f});
    Tensor mean  = make<float>({1}, {0.5f});
    Tensor var   = make<float>({1}, {1.0f});
    const Tensor X_b     = X, s_b = scale, b_b = bias, m_b = mean, v_b = var;
    (void)batchnorm_forward(X, scale, bias, mean, var);
    for (int64_t i = 0; i < X.size();     ++i) EXPECT_FLOAT_EQ(X.data()[i],     X_b.data()[i]);
    for (int64_t i = 0; i < scale.size(); ++i) EXPECT_FLOAT_EQ(scale.data()[i], s_b.data()[i]);
    for (int64_t i = 0; i < bias.size();  ++i) EXPECT_FLOAT_EQ(bias.data()[i],  b_b.data()[i]);
    for (int64_t i = 0; i < mean.size();  ++i) EXPECT_FLOAT_EQ(mean.data()[i],  m_b.data()[i]);
    for (int64_t i = 0; i < var.size();   ++i) EXPECT_FLOAT_EQ(var.data()[i],   v_b.data()[i]);
}

TEST(BatchNorm, RankMismatchThrows) {
    Tensor X = make<float>({1, 4, 4}, std::vector<float>(16, 0.0f));  // rank 3
    Tensor s = make<float>({1}, {1.0f});
    Tensor b = make<float>({1}, {0.0f});
    Tensor m = make<float>({1}, {0.0f});
    Tensor v = make<float>({1}, {1.0f});
    EXPECT_THROW(batchnorm_forward(X, s, b, m, v), std::invalid_argument);
}

TEST(BatchNorm, ChannelMismatchThrows) {
    Tensor X = make<float>({1, 3, 2, 2}, std::vector<float>(12, 0.0f));
    Tensor s = make<float>({2}, {1.0f, 1.0f});  // wrong C
    Tensor b = make<float>({3}, {0.0f, 0.0f, 0.0f});
    Tensor m = make<float>({3}, {0.0f, 0.0f, 0.0f});
    Tensor v = make<float>({3}, {1.0f, 1.0f, 1.0f});
    EXPECT_THROW(batchnorm_forward(X, s, b, m, v), std::invalid_argument);
}

TEST(BatchNorm, MeanRankMismatchThrows) {
    // running_mean is rank-2 instead of rank-1.
    Tensor X = make<float>({1, 2, 2, 2}, std::vector<float>(8, 0.0f));
    Tensor s = make<float>({2}, {1.0f, 1.0f});
    Tensor b = make<float>({2}, {0.0f, 0.0f});
    Tensor m = make<float>({1, 2}, {0.0f, 0.0f});  // rank 2
    Tensor v = make<float>({2}, {1.0f, 1.0f});
    EXPECT_THROW(batchnorm_forward(X, s, b, m, v), std::invalid_argument);
}

TEST(BatchNorm, NegativeEpsilonThrows) {
    Tensor X = make<float>({1, 1, 1, 1}, {0.0f});
    Tensor s = make<float>({1}, {1.0f});
    Tensor b = make<float>({1}, {0.0f});
    Tensor m = make<float>({1}, {0.0f});
    Tensor v = make<float>({1}, {1.0f});
    EXPECT_THROW(batchnorm_forward(X, s, b, m, v, /*epsilon=*/-1.0f),
                 std::invalid_argument);
}
