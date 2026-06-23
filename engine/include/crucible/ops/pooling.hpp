// 2D pooling operators — Issue #8 + Issue #10.
//
// Implements the three pooling operators Crucible needs to run
// MobileNetV2 and ResNet18 end-to-end:
//
//   * MaxPool          — sliding-window max                       (Issue #8)
//   * AveragePool      — sliding-window arithmetic mean           (Issue #8)
//   * GlobalAveragePool — per-channel mean over the full HW plane (Issue #10)
//
// The first two are NCHW in / NCHW out, with explicit kernel/stride/pad
// on each spatial axis. GlobalAveragePool takes the rank-4 input and
// reduces the two spatial axes to length 1 — it's the canonical
// "average each feature map to a single number" operator used in the
// classifier head of every modern image-classification network.
//
// Issue #8 scope (per the prompt):
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

// GlobalAveragePool — reduce each (N, C) feature map to a single scalar
// by averaging every H*W element in the spatial plane.
//
//   X: (N, C, H, W)
//   out: (N, C, 1, 1)
//
// No kernel / stride / pad arguments: the kernel is implicitly the
// full H*W plane and the stride is implicit-1, so every (n, c) slab
// collapses to one number. This is what MobileNetV2 and ResNet both
// use as the head of their classifier (right before the FC layer).
//
// No padding: unlike AveragePool, there's no notion of "padded
// elements" because the window IS the input. count_include_pad has
// no effect.
//
// Throws std::invalid_argument on rank != 4. We do not validate that
// H > 0 / W > 0 — the natural loop computes mean over 0 elements as
// +inf / NaN depending on the sum, which is the same answer Eigen
// would give, so we don't second-guess the caller.
Tensor global_avgpool_forward(const Tensor& x);

}  // namespace crucible::ops
