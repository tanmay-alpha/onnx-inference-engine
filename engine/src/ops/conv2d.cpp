// 2D convolution implementation — see conv2d.hpp for the public API.
//
// Algorithm: im2col + GEMM, with two specialisations.
//   * General: each group's receptive fields are laid out as a
//     (C_in_g*kH*kW, out_h*out_w) column matrix; the weight tensor viewed
//     in place as (C_out_g, C_in_g*kH*kW) multiplies it, and the product
//     is written directly into the NCHW output. Bias is added per row.
//   * 1x1 / stride 1 / no padding: the input slice already is the column
//     matrix, so im2col is skipped.
//   * Depthwise (one channel per group): direct sliding-window loop.
//
// Cross-correlation semantics (no kernel flip) and zero padding, as in
// ONNX / PyTorch.

#include "crucible/ops/conv2d.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace crucible::ops {

namespace {

// Row-major Eigen matrix alias. We force RowMajor to match the rest
// of the engine (PyTorch / NumPy / ONNX Runtime all use row-major).
// operator()(r, c) below honours this storage order.
using RowMatrix =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// im2col: fill `col` (C_in*kH*kW, out_h*out_w), row-major, with
//   col[c*kH*kW + kh*kW + kw, oh*out_w + ow] = X[c, h_in, w_in]
//     h_in = oh*stride_h + kh - pad_h,  w_in = ow*stride_w + kw - pad_w
// and 0 where (h_in, w_in) falls in the padding. `x` points at one image's
// (C_in, H, W) slice. Rows match the weight layout (C_out, C_in, kH, kW),
// so the product W * col needs no transpose.
void im2col(const float* x, int64_t C_in, int64_t H, int64_t W,
            int64_t kH, int64_t kW,
            int64_t out_h, int64_t out_w,
            int64_t stride_h, int64_t stride_w,
            int64_t pad_h, int64_t pad_w,
            RowMatrix& col) {
    float* dst = col.data();
    for (int64_t c = 0; c < C_in; ++c) {
        const float* xc = x + c * H * W;
        for (int64_t kh = 0; kh < kH; ++kh) {
            for (int64_t kw = 0; kw < kW; ++kw) {
                for (int64_t oh = 0; oh < out_h; ++oh) {
                    const int64_t h_in = oh * stride_h + kh - pad_h;
                    if (h_in < 0 || h_in >= H) {
                        std::fill(dst, dst + out_w, 0.0f);
                        dst += out_w;
                        continue;
                    }
                    const float* xrow = xc + h_in * W;
                    for (int64_t ow = 0; ow < out_w; ++ow) {
                        const int64_t w_in = ow * stride_w + kw - pad_w;
                        *dst++ = (w_in >= 0 && w_in < W) ? xrow[w_in] : 0.0f;
                    }
                }
            }
        }
    }
}

}  // namespace

// ----------------------------------------------------------------------

Tensor conv2d_forward(const Tensor& input,
                     const Tensor& weight,
                     const Tensor& bias,
                     const ConvParams& p) {
    // ---- 1. Validate inputs -------------------------------------------------
    if (input.rank() != 4) {
        throw std::invalid_argument(
            "conv2d_forward: input must be rank 4 (N,C,H,W), got rank " +
            std::to_string(input.rank()));
    }
    if (weight.rank() != 4) {
        throw std::invalid_argument(
            "conv2d_forward: weight must be rank 4 (C_out,C_in,kH,kW), got rank " +
            std::to_string(weight.rank()));
    }
    if (p.groups <= 0) {
        throw std::invalid_argument(
            "conv2d_forward: groups must be positive, got groups=" + std::to_string(p.groups));
    }
    if (p.stride_h <= 0 || p.stride_w <= 0) {
        throw std::invalid_argument(
            "conv2d_forward: stride must be positive, got (" +
            std::to_string(p.stride_h) + "," + std::to_string(p.stride_w) + ")");
    }
    if (p.pad_h < 0 || p.pad_w < 0) {
        throw std::invalid_argument(
            "conv2d_forward: pad must be non-negative, got (" +
            std::to_string(p.pad_h) + "," + std::to_string(p.pad_w) + ")");
    }

    const int64_t N     = input.shape()[0];
    const int64_t C_in  = input.shape()[1];
    const int64_t H     = input.shape()[2];
    const int64_t W     = input.shape()[3];
    const int64_t C_out = weight.shape()[0];
    const int64_t w_C   = weight.shape()[1];
    const int64_t kH    = weight.shape()[2];
    const int64_t kW    = weight.shape()[3];

    // ---- 2. Output spatial dimensions --------------------------------------
    const int64_t num_h = H + 2 * p.pad_h - kH;
    const int64_t num_w = W + 2 * p.pad_w - kW;
    if (num_h < 0 || num_w < 0) {
        throw std::invalid_argument(
            "conv2d_forward: kernel larger than (input + 2*pad) (got " +
            std::to_string(kH) + "x" + std::to_string(kW) +
            " vs " + std::to_string(H + 2 * p.pad_h) + "x" +
            std::to_string(W + 2 * p.pad_w) + ")");
    }
    const int64_t out_h = num_h / p.stride_h + 1;
    const int64_t out_w = num_w / p.stride_w + 1;
    if (out_h <= 0 || out_w <= 0) {
        throw std::invalid_argument(
            "conv2d_forward: output spatial dims must be positive, got (" +
            std::to_string(out_h) + "," + std::to_string(out_w) + ")");
    }

    // ---- 3. Optional bias ---------------------------------------------------
    const float* bias_ptr = nullptr;
    if (bias.size() > 0) {
        if (bias.rank() != 1 || bias.shape()[0] != C_out) {
            std::string got;
            for (size_t i = 0; i < bias.shape().size(); ++i) {
                if (i) got += ", ";
                got += std::to_string(bias.shape()[i]);
            }
            throw std::invalid_argument(
                "conv2d_forward: bias must be rank 1 of size C_out (got shape [" +
                got + "], expected [" + std::to_string(C_out) + "])");
        }
        bias_ptr = bias.data();
    }

    // ---- 4. Group bookkeeping ------------------------------------------------
    if (C_in % p.groups != 0) {
        throw std::invalid_argument(
            "conv2d_forward: input channels (" + std::to_string(C_in) +
            ") must be divisible by groups (" + std::to_string(p.groups) + ")");
    }
    if (C_out % p.groups != 0) {
        throw std::invalid_argument(
            "conv2d_forward: output channels (" + std::to_string(C_out) +
            ") must be divisible by groups (" + std::to_string(p.groups) + ")");
    }
    const int64_t C_in_g  = C_in / p.groups;
    const int64_t C_out_g = C_out / p.groups;
    if (w_C != C_in_g) {
        throw std::invalid_argument(
            "conv2d_forward: weight C_in (" + std::to_string(w_C) +
            ") != input channels per group (" + std::to_string(C_in_g) +
            ") for groups=" + std::to_string(p.groups));
    }

    Tensor output({N, C_out, out_h, out_w}, 0.0f);
    const int64_t HW_out = out_h * out_w;

    // ---- 5a. Depthwise (one input and one output channel per group) --------
    // A per-group GEMM would be a 1 x (kH*kW) by (kH*kW) x HW product — tiny
    // matrices dominated by im2col overhead — so compute it directly.
    if (C_in_g == 1 && C_out_g == 1) {
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C_out; ++c) {
                const float* xc = input.data() + (n * C_in + c) * H * W;
                const float* wc = weight.data() + c * kH * kW;
                float* yc = output.data() + (n * C_out + c) * HW_out;
                const float b = bias_ptr ? bias_ptr[c] : 0.0f;
                for (int64_t oh = 0; oh < out_h; ++oh) {
                    for (int64_t ow = 0; ow < out_w; ++ow) {
                        float acc = 0.0f;
                        for (int64_t kh = 0; kh < kH; ++kh) {
                            const int64_t h_in = oh * p.stride_h + kh - p.pad_h;
                            if (h_in < 0 || h_in >= H) continue;
                            const float* xrow = xc + h_in * W;
                            const float* wrow = wc + kh * kW;
                            for (int64_t kw = 0; kw < kW; ++kw) {
                                const int64_t w_in = ow * p.stride_w + kw - p.pad_w;
                                if (w_in < 0 || w_in >= W) continue;
                                acc += xrow[w_in] * wrow[kw];
                            }
                        }
                        yc[oh * out_w + ow] = acc + b;
                    }
                }
            }
        }
        return output;
    }

    // ---- 5b. General grouped convolution: im2col + GEMM ---------------------
    // Weight (C_out, C_in_g, kH, kW) row-major is already the
    // (C_out_g, C_in_g*kH*kW) matrix for each group, so it is viewed in
    // place. The GEMM result is written straight into the NCHW output.
    using ConstMap = Eigen::Map<const RowMatrix>;
    using OutMap = Eigen::Map<RowMatrix>;
    const int64_t ksize = C_in_g * kH * kW;
    const bool pointwise = kH == 1 && kW == 1 && p.stride_h == 1 &&
                           p.stride_w == 1 && p.pad_h == 0 && p.pad_w == 0;
    RowMatrix col;
    if (!pointwise) col.resize(ksize, HW_out);

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t g = 0; g < p.groups; ++g) {
            const float* x = input.data() + (n * C_in + g * C_in_g) * H * W;
            ConstMap wg(weight.data() + g * C_out_g * ksize, C_out_g, ksize);
            OutMap yg(output.data() + (n * C_out + g * C_out_g) * HW_out,
                      C_out_g, HW_out);
            if (pointwise) {
                // 1x1, stride 1, no pad: the input slice IS the (C_in_g, H*W)
                // column matrix.
                yg.noalias() = wg * ConstMap(x, C_in_g, HW_out);
            } else {
                im2col(x, C_in_g, H, W, kH, kW, out_h, out_w,
                       p.stride_h, p.stride_w, p.pad_h, p.pad_w, col);
                yg.noalias() = wg * col;
            }
            if (bias_ptr) {
                for (int64_t oc = 0; oc < C_out_g; ++oc) {
                    yg.row(oc).array() += bias_ptr[g * C_out_g + oc];
                }
            }
        }
    }
    return output;
}

}  // namespace crucible::ops
