// 2D convolution operator.
//
// Implements the ONNX `Conv` operator (2D variant only). The general path
// uses im2col: every receptive field is laid out as a column of an
// expanded matrix and the convolution becomes one GEMM per (image, group).
// 1x1 stride-1 convolutions skip im2col, and depthwise convolutions use a
// direct sliding-window loop (see conv2d.cpp).
//
// Why im2col + GEMM?
//   * It reuses Eigen's blocked, vectorised GEMM instead of hand-tuning
//     nested loops, at the cost of an im2col buffer of
//     (C_in/groups * kH * kW) x (out_h * out_w) floats.
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
// Supported: stride/pad on both spatial axes and any groups that divide
// C_in and C_out (including depthwise, groups == C_in).

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
//   * groups <= 0, or C_in / C_out not divisible by groups
//   * non-positive output spatial dim
Tensor conv2d_forward(const Tensor& input,
                     const Tensor& weight,
                     const Tensor& bias,
                     const ConvParams& p);

}  // namespace crucible::ops
