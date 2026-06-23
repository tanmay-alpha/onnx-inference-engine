// Google Benchmark — crucible::ops::conv2d_forward (Issue #11).
//
// Measures the throughput of our im2col + Eigen-GEMM Conv2D on a
// representative mid-sized tensor from MobileNetV2:
//
//   input  : (1, 32, 56, 56)   — 100,352 elements
//   weight : (64, 32,  3,  3)  — 18,432 elements
//   stride : 1,  pad : 1       (the standard "same-ish" 3x3 conv)
//   output : (1, 64, 56, 56)   — 200,704 elements
//
// That tensor pair is the second block of MobileNetV2's first
// bottleneck — a workload the engine will hit dozens of times during
// end-to-end inference. It's small enough that im2col's allocation
// overhead is non-trivial, but big enough that the GEMM dominates
// for 1024-iteration runs.
//
// We deliberately skip benchmarking multiple sizes here: conv2d has
// enough configuration axes (kernel, stride, pad, groups, channels)
// that a sweep is its own mini-project. Issue #11 just wants to see
// conv2d in the harness, prove it works, and compare Debug vs Release.

#include <benchmark/benchmark.h>

#include "crucible/ops/conv2d.hpp"
#include "crucible/tensor.hpp"

#include <vector>

namespace {

// Same deterministic LCG as bench_matmul.cpp — pseudo-random values
// in [-0.5, 0.5). Keeping the seed fixed makes the benchmark run
// reproducible, which matters when comparing Debug vs Release
// numbers across commits.
std::vector<float> make_data(size_t count, uint32_t seed) {
    std::vector<float> v(count);
    uint32_t s = seed ? seed : 0x9E3779B9u;
    for (auto& x : v) {
        s = s * 1664525u + 1013904223u;
        x = (static_cast<int32_t>(s) / 2147483647.0f) * 0.5f;
    }
    return v;
}

}  // namespace

static void BM_Conv2D(benchmark::State& state) {
    // Input  : (N=1, C_in=32, H=56, W=56) = 100,352 floats
    // Weight : (C_out=64, C_in=32, kH=3, kW=3) = 18,432 floats
    // Bias   : (C_out=64,) = 64 floats — present so the path is the
    //          same one MobileNetV2's 3x3 convolutions exercise.
    constexpr int64_t N = 1, C_in = 32, H = 56, W = 56;
    constexpr int64_t C_out = 64, kH = 3, kW = 3;

    const auto x_data = make_data(static_cast<size_t>(N * C_in * H * W),
                                  /*seed=*/0xBADC0DEu);
    const auto w_data = make_data(static_cast<size_t>(C_out * C_in * kH * kW),
                                  /*seed=*/0xFEEDFACEu);
    const auto b_data = make_data(static_cast<size_t>(C_out),
                                  /*seed=*/0xC0DEu);

    const crucible::Tensor X({N, C_in, H, W}, x_data);
    const crucible::Tensor Wt({C_out, C_in, kH, kW}, w_data);
    const crucible::Tensor B ({C_out}, b_data);

    // Stride 1, pad 1 — preserves spatial dims. Matches the standard
    // 3x3 conv config used in MobileNetV2's bottleneck blocks.
    crucible::ops::ConvParams p;
    p.stride_h = 1;
    p.stride_w = 1;
    p.pad_h    = 1;
    p.pad_w    = 1;
    p.groups   = 1;

    for (auto _ : state) {
        crucible::Tensor Y = crucible::ops::conv2d_forward(X, Wt, B, p);
        // DoNotOptimize so the compiler can't elide the call.
        benchmark::DoNotOptimize(Y.data());
    }

    // FLOPs for a Conv2D are 2 * N * C_out * C_in * kH * kW * out_h * out_w.
    // With the chosen params that's 2 * 1 * 64 * 32 * 3 * 3 * 56 * 56 =
    // 115,605,504 FLOPs per inference. Reporting as "items" so the
    // JSON consumers see a single number, and as "bytes" for the
    // memory-traffic story.
    constexpr int64_t flops_per_iter =
        2LL * N * C_out * C_in * kH * kW * H * W;
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * flops_per_iter);
    constexpr int64_t bytes_per_iter =
        N * C_in * H * W + C_out * C_in * kH * kW + C_out + N * C_out * H * W;
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * bytes_per_iter);
    state.SetLabel("conv2d 1x32x56x56 * 64x32x3x3 stride=1 pad=1");
}

BENCHMARK(BM_Conv2D)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();

BENCHMARK_MAIN();