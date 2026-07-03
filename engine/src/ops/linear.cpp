// Linear operator implementations — see linear.hpp for the public API.
//
// Both operators use Eigen's matrix multiplication under the hood. We
// view the row-major Tensor data through Eigen::Map<...> with the
// Eigen::RowMajor storage order, so layout matches the rest of the
// engine (PyTorch / NumPy / ONNX Runtime).
//
// Why not write our own matmul?
//   * Eigen gives us a 30-year-battle-tested BLAS integration on
//     x86_64 (FMA + AVX2 path) and NEON on aarch64. Hand-written
//     float32 matmul would be 2-5x slower for the same correctness.
//   * The plan (Issue #5 AC) only requires 1e-5 numerical match
//     against numpy, which Eigen's default product() provides
//     out of the box.
//   * Eigen is header-only — no extra build steps, no .so to ship.
//
// The code is C++17, follows the same style as tensor.cpp (RAII,
// throwing exceptions for error paths, narrow headers).

#include "crucible/ops/linear.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace crucible::ops {

namespace {

// Validate 2D shape and return (rows, cols). Throws std::invalid_argument
// with a descriptive message on mismatch.
struct Dim2 { int64_t rows; int64_t cols; };

Dim2 require_2d(const Tensor& t, const char* name) {
    if (t.rank() != 2) {
        throw std::invalid_argument(
            std::string(name) + ": expected 2-D tensor, got rank " +
            std::to_string(t.rank()));
    }
    return {t.shape()[0], t.shape()[1]};
}

// View the tensor as a row-major Eigen matrix (no copy).
using RowMatrix =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

Eigen::Map<const RowMatrix> view(const Tensor& t) {
    return Eigen::Map<const RowMatrix>(t.data(), t.shape()[0], t.shape()[1]);
}

}  // namespace

// ----------------------------------------------------------------------

Tensor matmul(const Tensor& A, const Tensor& B) {
    auto a = require_2d(A, "matmul A");
    auto b = require_2d(B, "matmul B");
    if (a.cols != b.rows) {
        throw std::invalid_argument(
            "matmul: inner dimension mismatch (A.cols=" +
            std::to_string(a.cols) + ", B.rows=" + std::to_string(b.rows) + ")");
    }
    RowMatrix Ya = view(A);
    RowMatrix Yb = view(B);
    // Eigen product is fully expression-templated; assigning to RowMatrix
    // forces materialisation in row-major so the data layout matches
    // the rest of the engine.
    RowMatrix Yc = Ya * Yb;

    // Copy the Eigen result into a fresh Tensor (Eigen allocates its
    // own storage; we don't want to expose that handle).
    Tensor out({a.rows, b.cols}, 0.0f);
    std::copy(Yc.data(), Yc.data() + out.size(), out.data());
    return out;
}

// ----------------------------------------------------------------------

Tensor gemm(const Tensor& A, const Tensor& B, const Tensor& C,
            float alpha, float beta,
            int transA, int transB) {
    auto a = require_2d(A, "gemm A");
    auto b = require_2d(B, "gemm B");

    // M, N, K conventions follow ONNX Gemm spec:
    //   A is (M, K) (or (K, M) if transA)
    //   B is (K, N) (or (N, K) if transB)
    //   Y is (M, N)
    int64_t M = (transA == 0) ? a.rows : a.cols;
    int64_t K = (transA == 0) ? a.cols : a.rows;
    int64_t K2 = (transB == 0) ? b.rows : b.cols;
    int64_t N = (transB == 0) ? b.cols : b.rows;
    if (K != K2) {
        throw std::invalid_argument(
            "gemm: K dimension mismatch (A=" + std::to_string(K) +
            ", B=" + std::to_string(K2) + ")");
    }

    RowMatrix Ya = view(A);
    RowMatrix Yb = view(B);

    RowMatrix Yc = RowMatrix::Zero(M, N);
    if (transA == 0) {
        if (transB == 0) Yc = Ya * Yb;
        else             Yc = Ya * Yb.transpose();
    } else {
        if (transB == 0) Yc = Ya.transpose() * Yb;
        else             Yc = Ya.transpose() * Yb.transpose();
    }
    Yc *= alpha;

    // Add beta * C (with broadcasting).
    if (C.size() > 0) {
        if (C.rank() == 2) {
            if (C.shape()[0] != M || C.shape()[1] != N) {
                throw std::invalid_argument(
                    "gemm: bias C shape mismatch (got [" +
                    std::to_string(C.shape()[0]) + ", " +
                    std::to_string(C.shape()[1]) + "], expected [" +
                    std::to_string(M) + ", " + std::to_string(N) + "])");
            }
            Eigen::Map<const RowMatrix> yc(C.data(), M, N);
            Yc += beta * yc;
        } else if (C.rank() == 1) {
            // Per ONNX spec, rank-1 C is broadcast as either (M,) -> per
            // row or (N,) -> per column. Both are valid; pick whichever
            // matches.
            if (C.shape()[0] == N) {
                RowMatrix bc = RowMatrix::Zero(M, N);
                for (int64_t j = 0; j < N; ++j) bc.col(j).array() += C.data()[j];
                Yc += static_cast<float>(beta) * bc;
            } else if (C.shape()[0] == M) {
                for (int64_t i = 0; i < M; ++i) {
                    Yc.row(i).array() += static_cast<float>(beta) * C.data()[i];
                }
            } else {
                throw std::invalid_argument(
                    "gemm: bias C (rank-1) size mismatch (got " +
                    std::to_string(C.shape()[0]) + ", expected M=" +
                    std::to_string(M) + " or N=" + std::to_string(N) + ")");
            }
        } else {
            throw std::invalid_argument(
                "gemm: bias C must be rank 0, 1, or 2, got rank " +
                std::to_string(C.rank()));
        }
    }

    Tensor out({M, N}, 0.0f);
    std::copy(Yc.data(), Yc.data() + out.size(), out.data());
    return out;
}

}  // namespace crucible::ops