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

// Softmax: shift by per-axis max for numerical stability, then
// exp / sum-of-exp. Works for any rank by reshaping the input into
// a 2D matrix (outer × axis_dim) and applying softmax along the
// inner dimension.
//
// This handles the common case (axis = -1, last dim) by treating
// the tensor as (rows = product of outer dims, cols = axis dim).
// A more general implementation would stride the input; for the
// ONNX opset we use (MobileNetV2 in particular), softmax is always
// along the last axis of a 2D or 4D tensor, so this covers it.

Tensor softmax_forward(const Tensor& input,
                       const std::unordered_map<std::string, float>& attrs) {
    int64_t axis = resolve_axis(input, static_cast<int64_t>(
                                       get_attr(attrs, "axis", -1.0f)));
    if (axis != input.rank() - 1) {
        // The current implementation only optimises for the common
        // axis=-1 case. For axis != last, the 2D-trick is incorrect.
        // We support a few higher-up positions via a simple explicit
        // path: collapse outer dims and treat axis as the inner one
        // (i.e. transpose via copy). For now we refuse anything other
        // than the last axis to keep the code honest.
        throw std::invalid_argument(
            "softmax: only axis = last dim (axis=-1 or axis=rank-1) "
            "is currently supported; got axis=" + std::to_string(axis));
    }
    if (input.rank() < 1) {
        throw std::invalid_argument("softmax: input must have rank >= 1");
    }

    int64_t axis_dim = input.shape()[axis];
    int64_t outer = input.size() / axis_dim;
    if (outer == 0) return Tensor(input.shape(), 0.0f);

    // View input as (outer x axis_dim) row-major. For rank-1 inputs
    // we use a VectorXf to avoid Eigen 3.3 Map base assertions on
    // (1, N) bindings.
    Tensor out(input.shape(), 0.0f);
    if (input.rank() == 1) {
        Eigen::VectorXf in = view1d(input);
        Eigen::VectorXf outv(axis_dim);
        float m = in.maxCoeff();
        Eigen::VectorXf shifted = in.array() - m;
        Eigen::VectorXf e = shifted.array().exp();
        outv = e / e.sum();
        std::copy(outv.data(), outv.data() + axis_dim, out.data());
    } else {
        auto in = view2d(input);
        auto outm = view2d(out);
        for (int64_t i = 0; i < outer; ++i) {
            auto in_row  = in.row(i);
            auto out_row = outm.row(i);
            float m = in_row.maxCoeff();
            Eigen::VectorXf shifted = in_row.array() - m;
            Eigen::VectorXf e = shifted.array().exp();
            out_row = e / e.sum();
        }
    }
    return out;
}

// ----------------------------------------------------------------------

Tensor gelu_forward(const Tensor& input,
                    const std::unordered_map<std::string, float>& attrs) {
    // ONNX GELU attribute is `approximate`; non-zero means use the
    // tanh approximation, zero means use the exact erf form.
    // The plan's AC and the test specification both target the tanh
    // approximation, so we default to it.
    int approximate = static_cast<int>(get_attr(attrs, "approximate", 1.0f));

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

}  // namespace crucible::ops