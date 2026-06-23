// Normalization operator — Issue #8 (BatchNormalization only).
//
// Implements the ONNX `BatchNormalization` operator in **inference
// mode** (training_mode = 0), which is what Crucible needs to run
// exported ONNX models: the per-channel scale, bias, running mean and
// running variance are all stored as initializers in the .onnx file
// and applied directly to the input.
//
// Inference formula (per the ONNX v15 spec, training_mode = 0):
//     Y = scale * (X - running_mean) / sqrt(running_var + epsilon) + bias
//
// Crucible fuses this into a per-channel affine y = a * x + b where
//   a = scale / sqrt(running_var + epsilon)
//   b = bias - running_mean * a
// so the inner loop is one FMA per element — same shape as the conv
// post-processing pass in conv2d.cpp. Training mode (computing the
// mini-batch mean/variance and updating the running statistics) is
// out of scope for Issue #8: training happens in PyTorch, Crucible
// only runs the exported model.

#pragma once

#include "crucible/tensor.hpp"

namespace crucible::ops {

// Inference-mode BatchNormalization.
//
//   X           : (N, C, H, W)               input activations
//   scale       : (C,)                        per-channel multiplicative gain
//   bias        : (C,)                        per-channel additive shift
//   running_mean: (C,)                        per-channel moving average of mean
//   running_var : (C,)                        per-channel moving average of variance
//   epsilon     : small constant for numerical stability (default 1e-5
//                 — matches the ONNX spec default)
// Returns Y with the same shape as X.
//
// All inputs are float32 row-major. Throws std::invalid_argument on
// rank / channel-count mismatches.
Tensor batchnorm_forward(const Tensor& x,
                         const Tensor& scale,
                         const Tensor& bias,
                         const Tensor& running_mean,
                         const Tensor& running_var,
                         float epsilon = 1e-5f);

}  // namespace crucible::ops
