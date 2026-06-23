// 2D convolution operator — Issue #7.
//
// Implements the ONNX `Conv` operator (2D variant only) using the im2col
// algorithm: every receptive field in the input is materialised as a row
// of an expanded column matrix, the filters are unrolled into a wide
// weight matrix, and the convolution collapses into a single large
// matrix-matrix multiply.
//
// Why im2col?
//   * It turns the conv into a single GEMM — which Eigen has a tuned,
//     vectorised implementation for. Hand-rolled nested loops would be
//     3–10x slower on x86_64 and would require re-implementing tiling
//     and the FMA path.
//   * It keeps the operator API trivial: one matrix multiply per
//     output position, identical to Linear. The plan calls this out
//     in Issue #7 ("Conv2D took me 3 days and 47 test iterations to
//     get numerically correct to 1e-4 against PyTorch"). We don't
//     reinvent the wheel — we reuse Eigen.
//
// Public API:
//   struct ConvParams { int stride_h, stride_w, pad_h, pad_w, groups; };
//   Tensor conv2d_forward(input, weight, bias, params);
//
// Layout:
//   * Input X : (N, C_in, H, W)        row-major float32
//   * Weight W: (C_out, C_in, kH, kW)  row-major float32
//   * Bias  B : (C_out,)               row-major float32 (optional)
//   * Output Y: (N, C_out, out_h, out_w)
//
// Issue #7 scope: stride/pad on both spatial axes, groups == 1.
// groups > 1 is parsed but not implemented (matches the plan AC).

#pragma once

#include "crucible/tensor.hpp"

#include <cstdint>

namespace crucible::ops {

// Convolution hyperparameters. Defaults produce a "vanilla" cross-
// correlation conv: stride 1, no padding, single group (the standard
// behaviour of PyTorch's nn.Conv2d when called without explicit args).
struct ConvParams {
    int stride_h = 1;
    int stride_w = 1;
    int pad_h    = 0;
    int pad_w    = 0;
    int groups   = 1;
};

// Compute Y = X * W + B for the 2D cross-correlation convolution
// described in the file header.
//
//   X : (N, C_in,  H,  W)
//   W : (C_out, C_in, kH, kW)
//   B : (C_out,)                — pass an empty Tensor() to skip bias
//   p : stride/pad/groups
//   Y : (N, C_out, out_h, out_w)
//
// Throws std::invalid_argument on:
//   * rank != 4 for X or W
//   * C_in mismatch (X.shape[1] != W.shape[1] * p.groups)
//   * groups != 1 (out of scope for Issue #7)
//   * non-positive output spatial dim
Tensor conv2d_forward(const Tensor& input,
                     const Tensor& weight,
                     const Tensor& bias,
                     const ConvParams& p);

}  // namespace crucible::ops
