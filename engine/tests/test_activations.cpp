// GoogleTest cases for crucible::ops activation functions
// (Issue #6: Activation functions — ReLU, Sigmoid, Softmax, GELU).
//
// Each op takes a Tensor and an optional std::unordered_map<std::string,
// float> of named attributes, and returns a fresh Tensor. Inputs are
// not modified.

#include "crucible/ops/activations.hpp"
#include "crucible/tensor.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <unordered_map>
#include <vector>

using crucible::Tensor;
using crucible::ops::relu_forward;
using crucible::ops::sigmoid_forward;
using crucible::ops::softmax_forward;
using crucible::ops::gelu_forward;

namespace {

// Empty attribute map shorthand.
const std::unordered_map<std::string, float> kNoAttrs;

template <typename T>
Tensor make(const std::vector<int64_t>& shape, std::initializer_list<T> d) {
    return Tensor(shape, std::vector<T>(d));
}

}  // namespace

// -----------------------------------------------------------------------
// ReLU
// -----------------------------------------------------------------------

TEST(ActivationsTest, ReluZerosNegatives) {
    // AC: ReLU zeros negatives.
    Tensor input = make<float>({4}, {-2.0f, -1.0f, 0.0f, 3.0f});
    Tensor output = relu_forward(input, kNoAttrs);
    EXPECT_EQ(output.shape(), (std::vector<int64_t>{4}));
    EXPECT_FLOAT_EQ(output.at({0}), 0.0f);
    EXPECT_FLOAT_EQ(output.at({1}), 0.0f);
    EXPECT_FLOAT_EQ(output.at({2}), 0.0f);
    EXPECT_FLOAT_EQ(output.at({3}), 3.0f);
}

TEST(ActivationsTest, ReluPreservesShape) {
    // 2D input, all-negative.
    Tensor input = make<float>({2, 3}, {-1.0f, -2.0f, -3.0f,
                                          -4.0f, -5.0f, -6.0f});
    Tensor output = relu_forward(input, kNoAttrs);
    EXPECT_EQ(output.shape(), input.shape());
    for (int64_t i = 0; i < input.size(); ++i) {
        EXPECT_FLOAT_EQ(output.data()[i], 0.0f);
    }
}

TEST(ActivationsTest, ReluSourceUnchanged) {
    Tensor input = make<float>({4}, {-2.0f, -1.0f, 0.0f, 3.0f});
    Tensor snapshot = input;
    (void)relu_forward(input, kNoAttrs);
    for (int64_t i = 0; i < input.size(); ++i) {
        EXPECT_FLOAT_EQ(input.data()[i], snapshot.data()[i]);
    }
}

// -----------------------------------------------------------------------
// Sigmoid
// -----------------------------------------------------------------------

TEST(ActivationsTest, SigmoidAtZeroIsHalf) {
    // AC: Sigmoid(0) = 0.5 (exact).
    Tensor input = make<float>({1}, {0.0f});
    Tensor output = sigmoid_forward(input, kNoAttrs);
    EXPECT_NEAR(output.at({0}), 0.5f, 1e-6f);
}

TEST(ActivationsTest, SigmoidSymmetric) {
    // Sigmoid(-x) = 1 - Sigmoid(x). AC: ±1 → 0.2689 and 0.7311.
    Tensor input = make<float>({2}, {-1.0f, 1.0f});
    Tensor output = sigmoid_forward(input, kNoAttrs);
    EXPECT_NEAR(output.at({0}), 0.26894142f, 1e-6f);
    EXPECT_NEAR(output.at({1}), 0.73105858f, 1e-6f);
    EXPECT_NEAR(output.at({0}) + output.at({1}), 1.0f, 1e-6f);
}

TEST(ActivationsTest, SigmoidAtLargePositiveIsOne) {
    // Sigmoid(10) ~ 0.99995
    Tensor input = make<float>({1}, {10.0f});
    Tensor output = sigmoid_forward(input, kNoAttrs);
    EXPECT_NEAR(output.at({0}), 1.0f, 1e-4f);
}

TEST(ActivationsTest, SigmoidAtLargeNegativeIsZero) {
    // Sigmoid(-10) ~ 0.0000454
    Tensor input = make<float>({1}, {-10.0f});
    Tensor output = sigmoid_forward(input, kNoAttrs);
    EXPECT_NEAR(output.at({0}), 0.0f, 1e-4f);
}

// -----------------------------------------------------------------------
// Softmax
// -----------------------------------------------------------------------

TEST(ActivationsTest, SoftmaxSumsToOne) {
    // AC: sum(Softmax({1,2,3})) = 1.0 ± 1e-6.
    Tensor input = make<float>({3}, {1.0f, 2.0f, 3.0f});
    Tensor output = softmax_forward(input, kNoAttrs);
    float sum = 0.0f;
    for (int64_t i = 0; i < 3; ++i) sum += output.at({i});
    EXPECT_NEAR(sum, 1.0f, 1e-6f);
}

TEST(ActivationsTest, SoftmaxValuesCorrect) {
    // Softmax([1,2,3]) = [0.0900306, 0.2447285, 0.6652409]
    Tensor input = make<float>({3}, {1.0f, 2.0f, 3.0f});
    Tensor output = softmax_forward(input, kNoAttrs);
    EXPECT_NEAR(output.at({0}), 0.0900306f, 1e-5f);
    EXPECT_NEAR(output.at({1}), 0.2447285f, 1e-5f);
    EXPECT_NEAR(output.at({2}), 0.6652409f, 1e-5f);
}

TEST(ActivationsTest, SoftmaxNumericalStabilityLargeInput) {
    // AC: large inputs (e.g. 10000) should not produce NaN/Inf —
    // the max-shift trick must keep everything finite.
    Tensor input = make<float>({4}, {10000.0f, 10001.0f, 10002.0f, 10003.0f});
    Tensor output = softmax_forward(input, kNoAttrs);
    float sum = 0.0f;
    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(std::isfinite(output.at({i})));
        sum += output.at({i});
    }
    EXPECT_NEAR(sum, 1.0f, 1e-6f);
    // Shifting by a constant does not change softmax; these inputs
    // differ by 1 per position, so the result matches softmax([0,1,2,3])
    // = [0.0321, 0.0871, 0.2369, 0.6439] (ONNX test_softmax_large_number).
    EXPECT_NEAR(output.at({0}), 0.0320586f, 1e-5f);
    EXPECT_NEAR(output.at({1}), 0.0871443f, 1e-5f);
    EXPECT_NEAR(output.at({2}), 0.2368828f, 1e-5f);
    EXPECT_NEAR(output.at({3}), 0.6439142f, 1e-5f);
}

TEST(ActivationsTest, Softmax2DRowWise) {
    // 2D input, softmax along the last (default) axis.
    // Rows: [1, 2] -> [0.2689, 0.7311]
    // Rows: [0, 0] -> [0.5, 0.5]
    Tensor input = make<float>({2, 2}, {1.0f, 2.0f,
                                        0.0f, 0.0f});
    Tensor output = softmax_forward(input, kNoAttrs);
    EXPECT_EQ(output.shape(), input.shape());
    EXPECT_NEAR(output.at({0, 0}), 0.26894142f, 1e-5f);
    EXPECT_NEAR(output.at({0, 1}), 0.73105858f, 1e-5f);
    EXPECT_NEAR(output.at({1, 0}), 0.5f, 1e-6f);
    EXPECT_NEAR(output.at({1, 1}), 0.5f, 1e-6f);
}

TEST(ActivationsTest, SoftmaxDefaultAxisIsLast) {
    // softmax(x) with no attribute and softmax(x, axis=-1) should be
    // the same on a 2D input.
    Tensor input = make<float>({2, 3}, {1.0f, 2.0f, 3.0f,
                                         4.0f, 5.0f, 6.0f});
    Tensor a = softmax_forward(input, kNoAttrs);
    std::unordered_map<std::string, float> attrs{{"axis", -1.0f}};
    Tensor b = softmax_forward(input, attrs);
    for (int64_t i = 0; i < input.size(); ++i) {
        EXPECT_NEAR(a.data()[i], b.data()[i], 1e-7f);
    }
}

// -----------------------------------------------------------------------
// GELU
// -----------------------------------------------------------------------

TEST(ActivationsTest, GeluAtZeroIsZero) {
    // AC: GELU(0) = 0 exactly.
    Tensor input = make<float>({1}, {0.0f});
    Tensor output = gelu_forward(input, kNoAttrs);
    EXPECT_NEAR(output.at({0}), 0.0f, 1e-7f);
}

TEST(ActivationsTest, GeluMatchesFormula) {
    // AC: GELU(1) ≈ 0.841 (tanh approximation).
    // tanh-approx GELU(1) = 0.5 * 1 * (1 + tanh(0.7978846 * (1 + 0.044715)))
    //                   = 0.5 * (1 + tanh(0.8335580)) = 0.5 * 1.6824
    //                   ≈ 0.8412
    Tensor input = make<float>({1}, {1.0f});
    Tensor output = gelu_forward(input, kNoAttrs);
    EXPECT_NEAR(output.at({0}), 0.8411920f, 1e-3f);
}

TEST(ActivationsTest, GeluNegative) {
    // GELU(-1) ≈ -0.1588 (tanh approximation).
    // tanh-approx GELU(-1) = 0.5 * -1 * (1 + tanh(0.7978846 * (-1 + -0.044715)))
    //                     = -0.5 * (1 + tanh(-0.8335580)) = -0.5 * (1 - 0.7028)
    //                     ≈ -0.5 * 0.2972 ≈ -0.1486 (note: with high precision
    //     this resolves to about -0.158655 for erf-form and -0.158808 for
    //     tanh-form — well within the 1e-3 AC tolerance).
    Tensor input = make<float>({1}, {-1.0f});
    Tensor output = gelu_forward(input, kNoAttrs);
    EXPECT_NEAR(output.at({0}), -0.158808f, 1e-3f);
}

TEST(ActivationsTest, GeluLinearForLargeX) {
    // GELU(x) ≈ x for large positive x (the tanh → 1).
    Tensor input = make<float>({1}, {10.0f});
    Tensor output = gelu_forward(input, kNoAttrs);
    EXPECT_NEAR(output.at({0}), 10.0f, 1e-2f);
}

TEST(ActivationsTest, GeluExactVsApproxAreClose) {
    // The two GELU forms (erf and tanh) should agree to ~1e-3 for
    // moderate inputs. Verify both for a vector of values.
    Tensor input = make<float>({5}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f});
    std::unordered_map<std::string, float> approx_attrs{{"approximate", 1.0f}};
    std::unordered_map<std::string, float> exact_attrs{{"approximate", 0.0f}};
    Tensor a = gelu_forward(input, approx_attrs);
    Tensor b = gelu_forward(input, exact_attrs);
    for (int64_t i = 0; i < input.size(); ++i) {
        EXPECT_NEAR(a.data()[i], b.data()[i], 1e-3f)
            << "GELU forms disagree at index " << i;
    }
}

TEST(ActivationsTest, GeluSourceUnchanged) {
    Tensor input = make<float>({4}, {-1.0f, 0.0f, 0.5f, 1.0f});
    Tensor snapshot = input;
    (void)gelu_forward(input, kNoAttrs);
    for (int64_t i = 0; i < input.size(); ++i) {
        EXPECT_FLOAT_EQ(input.data()[i], snapshot.data()[i]);
    }
}