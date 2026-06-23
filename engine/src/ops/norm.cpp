// BatchNormalization (inference mode) — see norm.hpp for the public API.
//
// We fuse the inference formula
//     y = scale * (x - mean) / sqrt(var + eps) + bias
// into a per-channel affine
//     y = a * x + b
// with
//     a = scale / sqrt(var + eps)
//     b = bias - mean * a
// so the inner loop is one FMA per element. The fused form is the
// standard trick used by every fast inference engine (ONNX Runtime,
// TensorFlow Lite, TensorRT) — it cuts the per-element work from
// three ops to one and pulls the sqrt() out of the inner loop.
//
// We precompute a[c] and b[c] into small per-channel vectors once
// per call, then the four-deep loop is plain contiguous FMA. The
// hot loop accesses x[n,c,h,w] at offsets
//     n * C*H*W + c * H*W + h * W + w
// which is a stride-1 sweep along the inner W axis — the compiler
// can vectorise it.

#include "crucible/ops/norm.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace crucible::ops {

namespace {

// Validate that a 1-D tensor of length C is rank-1 with the expected
// size. Throws std::invalid_argument otherwise. The error message
// spells out the actual shape so a caller passing a wrong-rank
// initializer (e.g. a (C, 1) instead of (C,)) gets a clear pointer.
void require_channel_vector(const Tensor& t, int64_t C, const char* name) {
    if (t.rank() != 1 || t.shape()[0] != C) {
        std::string got;
        for (size_t i = 0; i < t.shape().size(); ++i) {
            if (i) got += ", ";
            got += std::to_string(t.shape()[i]);
        }
        throw std::invalid_argument(
            std::string("batchnorm_forward: ") + name +
            " must be rank 1 of size C (got shape [" + got +
            "], expected [" + std::to_string(C) + "])");
    }
}

}  // namespace

// ----------------------------------------------------------------------

Tensor batchnorm_forward(const Tensor& x,
                         const Tensor& scale,
                         const Tensor& bias,
                         const Tensor& running_mean,
                         const Tensor& running_var,
                         float epsilon) {
    if (x.rank() != 4) {
        throw std::invalid_argument(
            "batchnorm_forward: input must be rank 4 (N,C,H,W), got rank " +
            std::to_string(x.rank()));
    }
    if (!(epsilon >= 0.0f)) {
        // std::isfinite would also catch NaN; either is fine but the
        // sign check keeps the message specific.
        throw std::invalid_argument(
            "batchnorm_forward: epsilon must be non-negative, got " +
            std::to_string(epsilon));
    }

    const int64_t N = x.shape()[0];
    const int64_t C = x.shape()[1];
    const int64_t H = x.shape()[2];
    const int64_t W = x.shape()[3];

    require_channel_vector(scale,        C, "scale");
    require_channel_vector(bias,         C, "bias");
    require_channel_vector(running_mean, C, "running_mean");
    require_channel_vector(running_var,  C, "running_var");

    // Precompute the fused per-channel affine coefficients.
    //   a = scale / sqrt(var + epsilon)
    //   b = bias - mean * a
    // We do this in float64 to keep the sqrt argument accurate for
    // very small variances; the multiplication back to float32 at
    // use time preserves the precision the user expects.
    std::vector<float> a(C), b(C);
    for (int64_t c = 0; c < C; ++c) {
        const double var = static_cast<double>(running_var.data()[c]);
        const double inv = 1.0 / std::sqrt(var + static_cast<double>(epsilon));
        const double ac  = static_cast<double>(scale.data()[c]) * inv;
        a[static_cast<size_t>(c)] = static_cast<float>(ac);
        b[static_cast<size_t>(c)] = static_cast<float>(
            static_cast<double>(bias.data()[c]) -
            static_cast<double>(running_mean.data()[c]) * ac);
    }

    Tensor y({N, C, H, W}, 0.0f);

    // Hot loop: per-channel affine transform. We index x and y with
    // the row-major NCHW offset so the W axis is the contiguous inner
    // axis — the compiler can keep the load-port busy.
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            const float ac = a[static_cast<size_t>(c)];
            const float bc = b[static_cast<size_t>(c)];
            const float* xc = x.data() + (n * C + c) * H * W;
            float* yc = y.data() + (n * C + c) * H * W;
            for (int64_t hw = 0; hw < H * W; ++hw) {
                yc[hw] = ac * xc[hw] + bc;
            }
        }
    }

    return y;
}

}  // namespace crucible::ops
