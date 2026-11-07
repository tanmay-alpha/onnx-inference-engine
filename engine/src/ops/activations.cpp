// Activation operator implementations — see activations.hpp for the
// public API. Backed by Eigen 3.x element-wise expressions, just like
// the linear operators in linear.cpp.
//
// Style notes:
//   * Inputs are never modified. Every function allocates a fresh
//     output tensor of the same shape as the input.
//   * Errors throw std::invalid_argument with a descriptive message —
//     same convention as tensor.cpp and linear.cpp.
//   * Eigen's expression templates handle the inner loop; we only
//     orchestrate the buffer copy.

#include "crucible/ops/activations.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace crucible::ops {

namespace {

// Hard-coded constants for the GELU tanh approximation. We avoid
// using M_PI / M_SQRT2 from <cmath> because they are not guaranteed
// to be defined on every platform (MSVC needs a feature-test macro).
// These values are the long double-precision rounded-to-float versions
// of sqrt(2/pi) and 0.044715 — see Hendrycks & Gimpel 2016.
constexpr float kGeluCoef     = 0.7978845608f;   // sqrt(2 / pi)
constexpr float kGeluCubCoef  = 0.044715f;

// View a row-major tensor as an Eigen RowMajor matrix without copy.
//   rank-1 → RowVectorXf
//   rank-2 → MatrixXf
//   rank-N → Map over the flat buffer (1-D math is rank-agnostic for
//            element-wise ops, so we use the 1-D path for everything
//            and reshape on the way out).

inline Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 1>>
view1d(const Tensor& t) {
    return Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 1>>(
        t.data(), t.size());
}

inline Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>>
view1d(Tensor& t) {
    return Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>>(
        t.data(), t.size());
}

inline Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic,
                                       Eigen::Dynamic, Eigen::RowMajor>>
view2d(const Tensor& t) {
    return Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>>(
        t.data(), t.shape()[0], t.shape()[1]);
}

inline Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::RowMajor>>
view2d(Tensor& t) {
    return Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>>(
        t.data(), t.shape()[0], t.shape()[1]);
}

// Resolve an attribute key with a default value.
float get_attr(const std::unordered_map<std::string, float>& attrs,
               const std::string& key, float dflt) {
    auto it = attrs.find(key);
    return (it == attrs.end()) ? dflt : it->second;
}

// Resolve the ONNX axis attribute (default -1) to a non-negative
// index in the given shape.
int64_t resolve_axis(const Tensor& t, int64_t axis_attr) {
    int64_t rank = t.rank();
    int64_t axis = axis_attr;
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) {
        throw std::invalid_argument(
            "softmax: axis " + std::to_string(axis_attr) +
            " out of range for rank " + std::to_string(rank));
    }
    return axis;
}

}  // namespace

// ----------------------------------------------------------------------

Tensor relu_forward(const Tensor& input,
                    const std::unordered_map<std::string, float>& /*attrs*/) {
    Tensor out(input.shape(), 0.0f);
    if (input.size() == 0) return out;
    view1d(out) = view1d(input).cwiseMax(0.0f);
    return out;
}

// ----------------------------------------------------------------------

Tensor sigmoid_forward(const Tensor& input,
                       const std::unordered_map<std::string, float>& /*attrs*/) {
    Tensor out(input.shape(), 0.0f);
    if (input.size() == 0) return out;
    view1d(out) = view1d(input).unaryExpr(
        [](float x) { return 1.0f / (1.0f + std::exp(-x)); });
    return out;
}

// ----------------------------------------------------------------------

// Softmax over one axis of an arbitrary-rank tensor. The tensor is viewed
// as (outer, axis_dim, inner) in row-major order; each of the outer*inner
// fibres along `axis` is normalised independently with the max-shift trick
// for numerical stability. Accumulation is in double so long fibres do not
// lose precision relative to ONNX Runtime.

Tensor softmax_forward(const Tensor& input,
                       const std::unordered_map<std::string, float>& attrs) {
    if (input.rank() < 1) {
        throw std::invalid_argument("softmax: input must have rank >= 1");
    }
    const int64_t axis = resolve_axis(input, static_cast<int64_t>(
                                             get_attr(attrs, "axis", -1.0f)));
    const auto& shape = input.shape();
    const int64_t axis_dim = shape[static_cast<size_t>(axis)];
    int64_t outer = 1;
    for (int64_t d = 0; d < axis; ++d) outer *= shape[static_cast<size_t>(d)];
    int64_t inner = 1;
    for (int64_t d = axis + 1; d < input.rank(); ++d) inner *= shape[static_cast<size_t>(d)];

    Tensor out(shape, 0.0f);
    if (input.size() == 0) return out;
    const float* x = input.data();
    float* y = out.data();
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t i = 0; i < inner; ++i) {
            const int64_t base = o * axis_dim * inner + i;
            float m = x[base];
            for (int64_t k = 1; k < axis_dim; ++k) m = std::max(m, x[base + k * inner]);
            double sum = 0.0;
            for (int64_t k = 0; k < axis_dim; ++k) {
                const float e = std::exp(x[base + k * inner] - m);
                y[base + k * inner] = e;
                sum += e;
            }
            const float inv = static_cast<float>(1.0 / sum);
            for (int64_t k = 0; k < axis_dim; ++k) y[base + k * inner] *= inv;
        }
    }
    return out;
}

// ----------------------------------------------------------------------

Tensor gelu_forward(const Tensor& input,
                    const std::unordered_map<std::string, float>& attrs) {
    // ONNX Gelu (opset 20) has a string attribute approximate = "none" |
    // "tanh", default "none" (exact erf). The executor maps it onto this
    // float flag: non-zero = tanh approximation, zero/absent = exact.
    int approximate = static_cast<int>(get_attr(attrs, "approximate", 0.0f));

    Tensor out(input.shape(), 0.0f);
    if (input.size() == 0) return out;

    if (approximate != 0) {
        // tanh approximation: x * 0.5 * (1 + tanh(c * (x + 0.044715*x^3)))
        // where c = sqrt(2/pi).
        view1d(out) = view1d(input).unaryExpr([](float x) {
            float inner = kGeluCoef * (x + kGeluCubCoef * x * x * x);
            return x * 0.5f * (1.0f + std::tanh(inner));
        });
    } else {
        // Exact form: x * 0.5 * (1 + erf(x / sqrt(2)))
        view1d(out) = view1d(input).unaryExpr([](float x) {
            return x * 0.5f * (1.0f + std::erf(x * 0.7071067811865475f));
        });
    }
    return out;
}

// ----------------------------------------------------------------------

Tensor tanh_forward(const Tensor& input,
                    const std::unordered_map<std::string, float>& /*attrs*/) {
    Tensor out(input.shape(), 0.0f);
    if (input.size() == 0) return out;
    view1d(out) = view1d(input).unaryExpr([](float x) {
        return std::tanh(x);
    });
    return out;
}

// ----------------------------------------------------------------------

Tensor leaky_relu_forward(const Tensor& input,
                          const std::unordered_map<std::string, float>& attrs) {
    float alpha = get_attr(attrs, "alpha", 0.01f);
    Tensor out(input.shape(), 0.0f);
    if (input.size() == 0) return out;
    view1d(out) = view1d(input).unaryExpr([alpha](float x) {
        return x >= 0.0f ? x : alpha * x;
    });
    return out;
}

// ----------------------------------------------------------------------

Tensor elu_forward(const Tensor& input,
                   const std::unordered_map<std::string, float>& attrs) {
    float alpha = get_attr(attrs, "alpha", 1.0f);
    Tensor out(input.shape(), 0.0f);
    if (input.size() == 0) return out;
    view1d(out) = view1d(input).unaryExpr([alpha](float x) {
        return x >= 0.0f ? x : alpha * (std::exp(x) - 1.0f);
    });
    return out;
}

}  // namespace crucible::ops