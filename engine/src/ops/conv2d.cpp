// 2D convolution implementation — see conv2d.hpp for the public API.
//
// Algorithm: im2col + GEMM.
//   For each (output row, output col) the input receptive field of shape
//   (C_in, kH, kW) is laid out as one row of a wide matrix. The weight
//   tensor is reshaped to (C_out, C_in*kH*kW). One matrix multiply then
//   produces (C_out, out_h*out_w) which is reshaped back to the NCHW
//   output. Per-channel bias is broadcast and added in-place.
//
// We follow the PyTorch / ONNX semantics of cross-correlation (no kernel
// flip), and zero-pad positions outside the input.

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

// im2col: produce a (out_h*out_w, C_in*kH*kW) row-major matrix where row
// r holds the receptive field that produces output position
// (r / out_w, r % out_w).
//
// Memory layout of col_mat for batch b (the outer loop in conv2d_forward
// calls this once per batch):
//   col_mat[r, c * kH * kW + kh * kW + kw]
//     = X[b, c, h_in, w_in]   where
//         h_in = (r / out_w) * stride_h + kh - pad_h
//         w_in = (r % out_w) * stride_w + kw - pad_w
//     = 0.0 if (h_in, w_in) is outside the input bounds (zero padding).
void im2col(const float* x, int64_t C_in, int64_t H, int64_t W,
            int64_t kH, int64_t kW,
            int64_t out_h, int64_t out_w,
            int64_t stride_h, int64_t stride_w,
            int64_t pad_h, int64_t pad_w,
            RowMatrix& col_mat) {
    // Eigen indexing into the row-major matrix. operator()(r, c) honours
    // the RowMajor storage order we declared above.
    for (int64_t r = 0; r < out_h * out_w; ++r) {
        const int64_t oh = r / out_w;
        const int64_t ow = r % out_w;
        int64_t col = 0;
        for (int64_t c = 0; c < C_in; ++c) {
            for (int64_t kh = 0; kh < kH; ++kh) {
                const int64_t h_in = oh * stride_h + kh - pad_h;
                for (int64_t kw = 0; kw < kW; ++kw) {
                    const int64_t w_in = ow * stride_w + kw - pad_w;
                    float v = 0.0f;
                    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                        // Row-major offset into the (C_in, H, W) input
                        // slice for batch b.
                        v = x[c * H * W + h_in * W + w_in];
                    }
                    col_mat(static_cast<Eigen::Index>(r),
                            static_cast<Eigen::Index>(col)) = v;
                    ++col;
                }
            }
        }
    }
}

// Reshape weight (C_out, C_in, kH, kW) row-major → RowMatrix (C_out, C_in*kH*kW).
// The copy is unavoidable: weight may live in any Tensor whose storage we
// don't own, and Eigen::Map would alias it under our downstream multiply.
RowMatrix weight_to_2d(const Tensor& weight, int64_t C_out, int64_t C_in,
                       int64_t kH, int64_t kW) {
    const int64_t ksize = C_in * kH * kW;
    RowMatrix wmat(C_out, ksize);
    // The weight Tensor is row-major (C_out, C_in, kH, kW). For row c_out
    // and column index c * kH * kW + kh * kW + kw, the source offset is
    //   c_out * (C_in*kH*kW) + c * (kH*kW) + kh * kW + kw
    // which matches the layout we built in im2col for col_mat — same
    // column indexing convention. No transpose needed at multiplication
    // time.
    std::copy(weight.data(), weight.data() + C_out * ksize, wmat.data());
    return wmat;
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
    if (p.groups != 1) {
        throw std::invalid_argument(
            "conv2d_forward: groups != 1 is not implemented in Issue #7 "
            "(got groups=" + std::to_string(p.groups) + ")");
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

    if (w_C != C_in) {
        throw std::invalid_argument(
            "conv2d_forward: weight C_in (" + std::to_string(w_C) +
            ") != input C (" + std::to_string(C_in) + ")");
    }

    // ---- 2. Output spatial dimensions --------------------------------------
    // ONNX formula for stride/pad: out = floor((in + 2*pad - k) / stride) + 1
    // (no dilation in Issue #7 scope).
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
    // bias is rank-1 (C_out,) or empty. Validate the non-empty case.
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

    // ---- 4. Reshape weight to (C_out, C_in*kH*kW) row-major -----------------
    RowMatrix wmat = weight_to_2d(weight, C_out, C_in, kH, kW);

    // ---- 5. Allocate output -------------------------------------------------
    Tensor output({N, C_out, out_h, out_w}, 0.0f);
    const int64_t ksize = C_in * kH * kW;

    // ---- 6. Per-batch loop: im2col + GEMM + bias ----------------------------
    for (int64_t n = 0; n < N; ++n) {
        // Pointer to this batch's input slice (N, C_in, H, W).
        const float* x = input.data() + n * (C_in * H * W);

        // im2col → (out_h*out_w, ksize) row-major.
        RowMatrix col_mat(out_h * out_w, ksize);
        im2col(x, C_in, H, W, kH, kW,
               out_h, out_w,
               p.stride_h, p.stride_w, p.pad_h, p.pad_w,
               col_mat);

        // result = wmat * col_mat^T  →  (C_out, out_h*out_w)
        RowMatrix result = wmat * col_mat.transpose();

        // Add bias per channel and copy into the output tensor.
        for (int64_t oc = 0; oc < C_out; ++oc) {
            const float b = bias_ptr ? bias_ptr[oc] : 0.0f;
            for (int64_t r = 0; r < out_h * out_w; ++r) {
                const float v = result(static_cast<Eigen::Index>(oc),
                                       static_cast<Eigen::Index>(r)) + b;
                // Output offset for (n, oc, oh, ow):
                //   n * (C_out * out_h * out_w) + oc * (out_h * out_w)
                //     + oh * out_w + ow
                const int64_t oh = r / out_w;
                const int64_t ow = r % out_w;
                output.at({n, oc, oh, ow}) = v;
            }
        }
    }

    return output;
}

}  // namespace crucible::ops
