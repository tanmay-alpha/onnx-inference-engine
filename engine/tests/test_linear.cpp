// GoogleTest cases for crucible::ops::matmul and crucible::ops::gemm
// (Issue #5: Linear operator).
//
// Both ops take crucibe::Tensor arguments and return a Tensor. Inputs and
// outputs are float32 and stored row-major.

#include "crucible/ops/linear.hpp"
#include "crucible/tensor.hpp"

#include <gtest/gtest.h>

#include <vector>

using crucible::Tensor;
using crucible::ops::matmul;
using crucible::ops::gemm;

namespace {

// Convenience: build a tensor from a flat row-major buffer.
template <typename T>
Tensor make(const std::vector<int64_t>& shape, const std::vector<T>& data) {
    return Tensor(shape, data);
}

// Hand-checked expected result for matmul of (3x4) @ (4x5) = (3x5).
// Hand-coded so test does not depend on any external library.
std::vector<float> expected_3x4_4x5() {
    // A =
    //   1  2  3  4
    //   5  6  7  8
    //   9 10 11 12
    // B =
    //   1 0 0 0 1
    //   0 1 0 1 0
    //   0 0 1 0 0
    //   1 0 0 0 1
    // A*B =
    //   row 0: 1+4=5, 2, 3, 2, 1+4=5
    //   row 1: 5+8=13, 6, 7, 6, 5+8=13
    //   row 2: 9+12=21, 10, 11, 10, 9+12=21
    return {
        5,  2,  3,  2,  5,
        13, 6,  7,  6,  13,
        21, 10, 11, 10, 21
    };
}

}  // namespace

// -----------------------------------------------------------------------
// matmul
// -----------------------------------------------------------------------

TEST(MatMul, AcMatrixMultiply) {
    // AC: MatMul of (3x4) @ (4x5) = (3x5), matches numpy to 1e-5.
    Tensor A = make<float>({3, 4}, {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12
    });
    Tensor B = make<float>({4, 5}, {
        1, 0, 0, 0, 1,
        0, 1, 0, 1, 0,
        0, 0, 1, 0, 0,
        1, 0, 0, 0, 1
    });
    Tensor Y = matmul(A, B);
    EXPECT_EQ(Y.shape(), (std::vector<int64_t>{3, 5}));
    auto exp = expected_3x4_4x5();
    for (int i = 0; i < 15; ++i) {
        EXPECT_NEAR(Y.data()[i], exp[i], 1e-5f) << "i=" << i;
    }
}

TEST(MatMul, IdentityLeft) {
    // A = I(3), B = any (3x2). Result == B.
    Tensor I = make<float>({3, 3}, {
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    });
    Tensor B = make<float>({3, 2}, {1, 2, 3, 4, 5, 6});
    Tensor Y = matmul(I, B);
    EXPECT_EQ(Y.shape(), (std::vector<int64_t>{3, 2}));
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(Y.data()[i], B.data()[i]);
    }
}

TEST(MatMul, IdentityRight) {
    // A = any (2x3), B = I(3). Result == A.
    Tensor A = make<float>({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor I = make<float>({3, 3}, {
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    });
    Tensor Y = matmul(A, I);
    EXPECT_EQ(Y.shape(), (std::vector<int64_t>{2, 3}));
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(Y.data()[i], A.data()[i]);
    }
}

TEST(MatMul, RectangularCases) {
    // (1, n) @ (n, 1) = (1, 1)
    Tensor a = make<float>({1, 4}, {1, 2, 3, 4});
    Tensor b = make<float>({4, 1}, {1, 1, 1, 1});
    Tensor y = matmul(a, b);
    EXPECT_EQ(y.shape(), (std::vector<int64_t>{1, 1}));
    EXPECT_FLOAT_EQ(y.at({0, 0}), 10.0f);
}

TEST(MatMul, ZeroMatrix) {
    Tensor A = make<float>({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor Z = make<float>({3, 2}, {0, 0, 0, 0, 0, 0});
    Tensor Y = matmul(A, Z);
    EXPECT_EQ(Y.shape(), (std::vector<int64_t>{2, 2}));
    for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(Y.data()[i], 0.0f);
}

TEST(MatMul, ShapeMismatchThrows) {
    Tensor A = make<float>({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor B = make<float>({4, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    EXPECT_THROW(matmul(A, B), std::invalid_argument);
}

TEST(MatMul, OneDimensionalNotSupported) {
    // Issue #5 scope is 2D only — vector inputs should throw.
    Tensor a = make<float>({4}, {1, 2, 3, 4});
    Tensor b = make<float>({4, 2}, {1, 0, 0, 1, 1, 0, 0, 1});
    EXPECT_THROW(matmul(a, b), std::invalid_argument);
    EXPECT_THROW(matmul(b, a), std::invalid_argument);
}

TEST(MatMul, SourceUnchanged) {
    Tensor A = make<float>({2, 2}, {1, 2, 3, 4});
    Tensor B = make<float>({2, 2}, {5, 6, 7, 8});
    Tensor A_before = A;
    Tensor B_before = B;
    (void)matmul(A, B);
    EXPECT_EQ(A.data(), A_before.data());
    EXPECT_EQ(B.data(), B_before.data());
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(A.data()[i], A_before.data()[i]);
        EXPECT_FLOAT_EQ(B.data()[i], B_before.data()[i]);
    }
}

// -----------------------------------------------------------------------
// gemm
// -----------------------------------------------------------------------

TEST(Gemm, NoTransNoBias) {
    // Y = A @ B
    Tensor A = make<float>({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor B = make<float>({3, 2}, {7, 8, 9, 10, 11, 12});
    Tensor Y = gemm(A, B, /*C=*/Tensor(), 1.0f, 1.0f, 0, 0);
    EXPECT_EQ(Y.shape(), (std::vector<int64_t>{2, 2}));
    // Row 0: 1*7+2*9+3*11=58, 1*8+2*10+3*12=64
    // Row 1: 4*7+5*9+6*11=139, 4*8+5*10+6*12=154
    EXPECT_FLOAT_EQ(Y.at({0, 0}), 58.0f);
    EXPECT_FLOAT_EQ(Y.at({0, 1}), 64.0f);
    EXPECT_FLOAT_EQ(Y.at({1, 0}), 139.0f);
    EXPECT_FLOAT_EQ(Y.at({1, 1}), 154.0f);
}

TEST(Gemm, WithBias) {
    // Y = A @ B + C
    Tensor A = make<float>({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor B = make<float>({3, 2}, {7, 8, 9, 10, 11, 12});
    Tensor C = make<float>({2, 2}, {0.1f, -0.1f, 0.5f, 0.0f});
    Tensor Y = gemm(A, B, C, 1.0f, 1.0f, 0, 0);
    EXPECT_FLOAT_EQ(Y.at({0, 0}), 58.1f);
    EXPECT_FLOAT_EQ(Y.at({0, 1}), 63.9f);
    EXPECT_FLOAT_EQ(Y.at({1, 0}), 139.5f);
    EXPECT_FLOAT_EQ(Y.at({1, 1}), 154.0f);
}

TEST(Gemm, AlphaAndBeta) {
    // Y = 2 * (A @ B) + 0.5 * C
    Tensor A = make<float>({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor B = make<float>({3, 2}, {1, 0, 0, 1, 0, 0});
    Tensor C = make<float>({2, 2}, {0, 0, 0, 0});
    Tensor Y = gemm(A, B, C, 2.0f, 0.5f, 0, 0);
    // A @ B = [1, 2; 4, 5]
    // 2 * (A@B) = [2, 4; 8, 10]; 0.5 * C = [0,0;0,0]
    EXPECT_FLOAT_EQ(Y.at({0, 0}), 2.0f);
    EXPECT_FLOAT_EQ(Y.at({0, 1}), 4.0f);
    EXPECT_FLOAT_EQ(Y.at({1, 0}), 8.0f);
    EXPECT_FLOAT_EQ(Y.at({1, 1}), 10.0f);
}

TEST(Gemm, TransBEqualsLinearLayer) {
    // nn.Linear: Y = X @ W^T + b corresponds to Gemm(X, W, b, transB=1).
    // X = (1x3) = [[1, 2, 3]]
    // W = (2x3) = [[1, 2, 3], [4, 5, 6]]
    // Expected Y = X @ W^T = [[1*1+2*2+3*3, 1*4+2*5+3*6]] = [[14, 32]]
    Tensor X = make<float>({1, 3}, {1, 2, 3});
    Tensor W = make<float>({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b = make<float>({2}, {0.1f, -0.1f});
    Tensor Y = gemm(X, W, b, 1.0f, 1.0f, 0, 1);
    EXPECT_EQ(Y.shape(), (std::vector<int64_t>{1, 2}));
    EXPECT_FLOAT_EQ(Y.at({0, 0}), 14.1f);
    EXPECT_FLOAT_EQ(Y.at({0, 1}), 31.9f);
}

TEST(Gemm, TransA) {
    // Y = A^T @ B
    // A = (2x3) = [[1,2,3],[4,5,6]] -> A^T = (3x2) = [[1,4],[2,5],[3,6]]
    // B = (3x2)
    Tensor A = make<float>({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor B = make<float>({3, 2}, {1, 0, 0, 1, 1, 1});
    Tensor Y = gemm(A, B, Tensor(), 1.0f, 1.0f, 1, 0);
    // A^T @ B:
    // row 0: 1*1+4*0=1, 1*0+4*1=4
    // row 1: 2*1+5*0=2, 2*0+5*1=5
    // row 2: 3*1+6*0=3, 3*0+6*1=6
    EXPECT_FLOAT_EQ(Y.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(Y.at({0, 1}), 4.0f);
    EXPECT_FLOAT_EQ(Y.at({1, 0}), 2.0f);
    EXPECT_FLOAT_EQ(Y.at({1, 1}), 5.0f);
    EXPECT_FLOAT_EQ(Y.at({2, 0}), 3.0f);
    EXPECT_FLOAT_EQ(Y.at({2, 1}), 6.0f);
}

TEST(Gemm, BiasBroadcast) {
    // C is (2,) — broadcasts to each row of result (2x2).
    Tensor A = make<float>({2, 2}, {1, 2, 3, 4});
    Tensor B = make<float>({2, 2}, {1, 0, 0, 1});
    Tensor C = make<float>({2}, {10.0f, 20.0f});
    Tensor Y = gemm(A, B, C, 1.0f, 1.0f, 0, 0);
    // A @ B = [1,2; 3,4] + [10,20; 10,20] = [11,22; 13,24]
    EXPECT_FLOAT_EQ(Y.at({0, 0}), 11.0f);
    EXPECT_FLOAT_EQ(Y.at({0, 1}), 22.0f);
    EXPECT_FLOAT_EQ(Y.at({1, 0}), 13.0f);
    EXPECT_FLOAT_EQ(Y.at({1, 1}), 24.0f);
}

TEST(Gemm, EmptyBiasIsAllowed) {
    // C may be empty (rank 0) — Gemm should ignore it.
    Tensor A = make<float>({2, 2}, {1, 2, 3, 4});
    Tensor B = make<float>({2, 2}, {1, 0, 0, 1});
    Tensor Y = gemm(A, B, Tensor(), 1.0f, 1.0f, 0, 0);
    EXPECT_FLOAT_EQ(Y.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(Y.at({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(Y.at({1, 0}), 3.0f);
    EXPECT_FLOAT_EQ(Y.at({1, 1}), 4.0f);
}

TEST(Gemm, ShapeMismatchThrows) {
    Tensor A = make<float>({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor B = make<float>({4, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    Tensor C;
    EXPECT_THROW(gemm(A, B, C, 1.0f, 1.0f, 0, 0), std::invalid_argument);
}