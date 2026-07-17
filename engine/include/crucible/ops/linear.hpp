// Linear operators — Issue #5.
//
// Implements:
//   * matmul(A, B)       — plain 2D matrix multiplication (ONNX MatMul)
//   * gemm(A, B, C, ...) — generalised Y = alpha * transA(A) @ transB(B) + beta * C
//
// Both ops accept float32 row-major Tensor inputs and return a freshly
// allocated Tensor. The inputs are not modified.
//
// Implementation uses Eigen 3.x under the hood; the public API only
// requires crucible::Tensor.

#pragma once

#include "crucible/tensor.hpp"

#include <string>

namespace crucible::ops {

// Compute Y = A @ B for 2D inputs.
// Throws std::invalid_argument if A or B is not 2D, or if A.cols != B.rows.
Tensor matmul(const Tensor& A, const Tensor& B);

// Compute Y = alpha * op(A) @ op(B) + beta * C
//   where op(X) is X if transX, X^T otherwise.
// C may be empty (rank 0) in which case the bias is omitted.
// C may broadcast: rank-1 of size M (rows of result) or rank-2 of size MxN.
// Throws std::invalid_argument on rank or shape mismatch.
Tensor gemm(const Tensor& A, const Tensor& B, const Tensor& C,
            float alpha = 1.0f, float beta = 1.0f,
            bool transA = false, bool transB = false);

}  // namespace crucible::ops