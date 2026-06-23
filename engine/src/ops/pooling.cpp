// 2D pooling — see pooling.hpp for the public API.
//
// Both operators share the same nested-loop structure; the only thing
// that varies is the per-window reduction (max vs mean). We implement
// the reduction inline at the call site rather than abstracting it
// behind a function pointer — the compiler vectorises the max loop
// better when it can see the comparison in context, and the duplication
// is ~6 lines.
//
// Padding model: positions outside the input contribute 0.0 to the
// sum (AvgPool) and -infinity to the max (MaxPool). For sane
// (kernel, stride, pad) triples there is at least one in-bounds
// element per window, so the MaxPool branch never returns -inf in
// practice. We still leave the sentinel in for the case where the
// caller sets a degenerate config — the test suite verifies it.

#include "crucible/ops/pooling.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace crucible::ops {

namespace {

// Validate rank-4 input and the pooling hyperparameters. Returns the
// resolved (N, C, H, W) and computes the output spatial dims into the
// out-pointers. Throws std::invalid_argument with a descriptive
// message on any failure.
struct Dim4 { int64_t N, C, H, W; };
struct OutDims { int64_t out_h, out_w; };

void resolve_pool(const Tensor& x,
                  int kH, int kW, int sH, int sW, int pH, int pW,
                  Dim4& in, OutDims& out) {
    if (x.rank() != 4) {
        throw std::invalid_argument(
            "pool: input must be rank 4 (N,C,H,W), got rank " +
            std::to_string(x.rank()));
    }
    if (kH <= 0 || kW <= 0) {
        throw std::invalid_argument(
            "pool: kernel must be positive, got (" +
            std::to_string(kH) + "," + std::to_string(kW) + ")");
    }
    if (sH <= 0 || sW <= 0) {
        throw std::invalid_argument(
            "pool: stride must be positive, got (" +
            std::to_string(sH) + "," + std::to_string(sW) + ")");
    }
    if (pH < 0 || pW < 0) {
        throw std::invalid_argument(
            "pool: pad must be non-negative, got (" +
            std::to_string(pH) + "," + std::to_string(pW) + ")");
    }
    in = {x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]};

    const int64_t num_h = in.H + 2 * pH - kH;
    const int64_t num_w = in.W + 2 * pW - kW;
    if (num_h < 0 || num_w < 0) {
        throw std::invalid_argument(
            "pool: kernel larger than (input + 2*pad) (got " +
            std::to_string(kH) + "x" + std::to_string(kW) +
            " vs " + std::to_string(in.H + 2 * pH) + "x" +
            std::to_string(in.W + 2 * pW) + ")");
    }
    out.out_h = num_h / sH + 1;
    out.out_w = num_w / sW + 1;
    if (out.out_h <= 0 || out.out_w <= 0) {
        throw std::invalid_argument(
            "pool: output spatial dims must be positive, got (" +
            std::to_string(out.out_h) + "," + std::to_string(out.out_w) + ")");
    }
}

}  // namespace

// ----------------------------------------------------------------------

Tensor maxpool_forward(const Tensor& x,
                       int kH, int kW,
                       int sH, int sW,
                       int pH, int pW) {
    Dim4 in; OutDims out;
    resolve_pool(x, kH, kW, sH, sW, pH, pW, in, out);

    Tensor y({in.N, in.C, out.out_h, out.out_w}, 0.0f);
    const float neg_inf = -std::numeric_limits<float>::infinity();

    // Loop layout matches the conv2d implementation: outer batch,
    // outer channel, then per-window reduction. Padded positions
    // contribute neg_inf and lose every comparison.
    for (int64_t n = 0; n < in.N; ++n) {
        for (int64_t c = 0; c < in.C; ++c) {
            const float* xc = x.data() + (n * in.C + c) * in.H * in.W;
            for (int64_t oh = 0; oh < out.out_h; ++oh) {
                for (int64_t ow = 0; ow < out.out_w; ++ow) {
                    float m = neg_inf;
                    for (int64_t kh = 0; kh < kH; ++kh) {
                        const int64_t h_in = oh * sH + kh - pH;
                        if (h_in < 0 || h_in >= in.H) continue;
                        for (int64_t kw = 0; kw < kW; ++kw) {
                            const int64_t w_in = ow * sW + kw - pW;
                            if (w_in < 0 || w_in >= in.W) continue;
                            const float v = xc[h_in * in.W + w_in];
                            if (v > m) m = v;
                        }
                    }
                    y.at({n, c, oh, ow}) = m;
                }
            }
        }
    }
    return y;
}

// ----------------------------------------------------------------------

Tensor avgpool_forward(const Tensor& x,
                       int kH, int kW,
                       int sH, int sW,
                       int pH, int pW) {
    Dim4 in; OutDims out;
    resolve_pool(x, kH, kW, sH, sW, pH, pW, in, out);

    Tensor y({in.N, in.C, out.out_h, out.out_w}, 0.0f);
    // count_include_pad = 1: denominator is always kH*kW. This is the
    // ONNX v7 default and matches PyTorch's nn.AvgPool2d with
    // count_include_pad=True. MobileNetV2 uses this configuration.
    const float denom = static_cast<float>(kH) * static_cast<float>(kW);

    for (int64_t n = 0; n < in.N; ++n) {
        for (int64_t c = 0; c < in.C; ++c) {
            const float* xc = x.data() + (n * in.C + c) * in.H * in.W;
            for (int64_t oh = 0; oh < out.out_h; ++oh) {
                for (int64_t ow = 0; ow < out.out_w; ++ow) {
                    float sum = 0.0f;
                    for (int64_t kh = 0; kh < kH; ++kh) {
                        const int64_t h_in = oh * sH + kh - pH;
                        if (h_in < 0 || h_in >= in.H) continue;
                        for (int64_t kw = 0; kw < kW; ++kw) {
                            const int64_t w_in = ow * sW + kw - pW;
                            if (w_in < 0 || w_in >= in.W) continue;
                            sum += xc[h_in * in.W + w_in];
                        }
                    }
                    y.at({n, c, oh, ow}) = sum / denom;
                }
            }
        }
    }
    return y;
}

}  // namespace crucible::ops
