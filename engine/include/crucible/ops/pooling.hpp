// 2D pooling operators — Issue #8.
//
// Implements the two pooling operators Crucible needs to run MobileNetV2:
//
//   * MaxPool        — sliding-window max
//   * AveragePool    — sliding-window arithmetic mean
//
// Both are NCHW in / NCHW out, with explicit kernel/stride/pad on each
// spatial axis. Issue #8 scope (per the prompt):
//   * rank-4 input (N, C, H, W)
//   * explicit kernel/stride/pad integers (no auto_pad, no ceil_mode,
//     no dilation — these belong to a later issue if we hit a model
//     that needs them)
//   * for AveragePool, count_include_pad = 1 (denominator is always
//     kH*kW, matching the ONNX v7 default that MobileNetV2 uses)
//
// Why no auto_pad?
//   MobileNetV2 doesn't use it. Adding the SAME_UPPER / SAME_LOWER
//   branches now would be ~30 LoC that no test exercises. Same for
//   ceil_mode and dilations — they show up in semantic segmentation
//   models (FCN, UNet) but never in classification.
//
// Why not Eigen for these?
//   Pooling is a strict element-wise reduction over a fixed-size window.
//   Eigen's tensor block operations would work, but the speedup over
//   a tight hand-written loop is in the single-digit percent range
//   for kH = kW = 2/3 windows, and the loop is far easier to read and
//   verify against the ONNX spec. We use Eigen where it pays (matmul,
//   conv) and avoid the dependency where it doesn't.

#pragma once

#include "crucible/tensor.hpp"

namespace crucible::ops {

// MaxPool with explicit kernel/stride/pad on each spatial axis.
//   X: (N, C, H, W)
//   out: (N, C, out_h, out_w)
//   out_h = (H + 2*pH - kH) / sH + 1
//   out_w = (W + 2*pW - kW) / sW + 1
// Padded positions are treated as -infinity so they never win the max.
// Throws std::invalid_argument on rank != 4, non-positive stride/kernel,
// or non-positive output spatial dim.
Tensor maxpool_forward(const Tensor& x,
                       int kH, int kW,
                       int sH, int sW,
                       int pH, int pW);

// AveragePool with the same signature. Denominator is always kH*kW
// (count_include_pad = 1) — matches MobileNetV2's behaviour under the
// ONNX v7 default.
Tensor avgpool_forward(const Tensor& x,
                       int kH, int kW,
                       int sH, int sW,
                       int pH, int pW);

}  // namespace crucible::ops
