// Activation functions — Issue #6.
//
// Implements four element-wise (or per-axis) non-linearities from the
// ONNX spec:
//   * Relu    — max(0, x)
//   * Sigmoid — 1 / (1 + exp(-x))
//   * Softmax — exp(x - max(x)) / sum(exp(x - max(x)))   (numerically stable)
//   * GELU    — x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
//               (tanh approximation; this is the BERT/GPT-2 form,
//                the most common variant in modern transformers)
//
// Every function takes a row-major float32 Tensor and an optional
// `std::unordered_map<std::string, float>` of named attributes so the
// graph executor can dispatch ONNX attributes uniformly. Inputs are
// not modified; a fresh Tensor is returned.

#pragma once

#include "crucible/tensor.hpp"

#include <unordered_map>

namespace crucible::ops {

// ReLU: y = max(0, x). Element-wise. No attributes.
Tensor relu_forward(const Tensor& input,
                    const std::unordered_map<std::string, float>& attrs);

// Sigmoid: y = 1 / (1 + exp(-x)). Element-wise. No attributes.
Tensor sigmoid_forward(const Tensor& input,
                       const std::unordered_map<std::string, float>& attrs);

// Softmax: numerically stable softmax along `axis`.
//
// Recognised attributes:
//   axis — int dim along which to compute softmax. Default: -1 (the
//          last dim, matching the ONNX spec's default). Negative
//          values count from the back.
Tensor softmax_forward(const Tensor& input,
                       const std::unordered_map<std::string, float>& attrs);

// GELU using the tanh approximation (the form used by BERT, GPT-2,
// and most modern transformer stacks). See ENGINEERING_PLAN.md §6.
//
// Recognised attributes:
//   approximate — int; if non-zero, use tanh approximation (current
//                 behaviour). If zero, fall back to the exact form
//                 `0.5 * x * (1 + erf(x / sqrt(2)))`. We default to
//                 tanh because that is what almost every model uses
//                 in practice and matches the test specification.
Tensor gelu_forward(const Tensor& input,
                    const std::unordered_map<std::string, float>& attrs);

}  // namespace crucible::ops