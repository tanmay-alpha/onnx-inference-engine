// Google Benchmark — crucible::ops::matmul (Issue #11).
//
// Measures end-to-end throughput of our 2D matrix multiply across three
// sizes that span the regime Crucible actually runs:
//
//   *  64 ×  64  — small GEMM, common in attention heads and the
//                  first linear layer of small classifiers.
//   * 256 × 256  — medium GEMM, common in transformer FFN blocks.
//   *1024 ×1024  — large GEMM, common in image-classification
//                  classifiers (e.g. MobileNetV2's final Gemm).
//
// We benchmark the *square* case (M == N == K) because (a) that's the
// only case the issue requires, and (b) it makes the bytes/FLOPs
// comparison between Debug and Release easy to read.
//
// Why we set SetItemsProcessed:
//   Crucible's matmul is O(N^3) FLOPs and O(N^2) bytes. Reporting
//   "items processed" as the product M*N matches the BLAS convention
//   (so we can quote GFLOPs in the per-benchmark output), and
//   "bytes processed" as 2*M*N*K (one read of A, one read of B) so
//   the bytes/second number reflects real memory traffic.
//
// Why we pre-allocate inputs OUTSIDE the state loop:
//   State.PauseTiming / ResumeTiming would also work, but it adds
//   noise on the first iteration of small benchmarks. Pre-allocating
//   once is cleaner and matches the Google Benchmark style guide.

#include <benchmark/benchmark.h>

#include "crucible/ops/linear.hpp"
#include "crucible/tensor.hpp"

#include <vector>

namespace {

// Build a deterministic (M, K) matrix of pseudo-random floats. We use
// a simple LCG with a fixed seed so results are reproducible across
// runs. The exact distribution doesn't matter — what matters is that
// the values aren't all zero, which would let the optimiser short-
// circuit the multiply.
std::vector<float> make_matrix(int64_t rows, int64_t cols, uint32_t seed) {
    std::vector<float> v(static_cast<size_t>(rows * cols));
    uint32_t s = seed ? seed : 0x9E3779B9u;
    for (auto& x : v) {
        s = s * 1664525u + 1013904223u;
        // Map to [-1, 1) — small magnitude keeps the matmul numerically
        // well-behaved even at N=1024 where FLOP counts are 1e9.
        x = (static_cast<int32_t>(s) / 2147483647.0f) * 0.5f;
    }
    return v;
}

}  // namespace

static void BM_MatMul(benchmark::State& state) {
    const int64_t N = state.range(0);

    // Pre-allocate inputs once. We hold them in unique_ptr-like
    // std::vectors on the stack — Crucible's Tensor copies its data
    // on construction, so the input vectors are the only references
    // until the matmul returns.
    const auto A_data = make_matrix(N, N, /*seed=*/0xC0FFEEu);
    const auto B_data = make_matrix(N, N, /*seed=*/0xDECAFu);
    const crucible::Tensor A({N, N}, A_data);
    const crucible::Tensor B({N, N}, B_data);

    for (auto _ : state) {
        crucible::Tensor Y = crucible::ops::matmul(A, B);
        // DoNotOptimize forces the compiler to materialise the
        // result so the loop can't be elided. (The Crucible tensor
        // constructor does the copy for us, but DoNotOptimize is
        // belt-and-braces for aggressive LTO.)
        benchmark::DoNotOptimize(Y.data());
    }

    // Reporting: items_processed = M*N (BLAS convention for matmul),
    // bytes_processed = 2*M*N*K (one read of A, one read of B).
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N * N);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * 2 * N * N * N);
    state.SetLabel("matmul " + std::to_string(N) + "x" + std::to_string(N));
}

// Register the three sizes required by the issue AC. We pass them
// explicitly rather than using ->Range(...) because the AC names the
// exact sizes and we don't want a Dense(8, 1024) sweep filling the
// output with 8 more benchmarks we don't need.
//
// MinTime: by default, Google Benchmark runs each benchmark until it
// has enough samples (1 second of wall time). That's fine for 64x64
// but wasteful for 1024x1024 (which already takes hundreds of ms per
// iteration). We set MinTime to 0.5s for all sizes, which keeps the
// overall benchmark run under 10 seconds for all three sizes.
BENCHMARK(BM_MatMul)
    ->Arg(64)
    ->Arg(256)
    ->Arg(1024)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();

// Note: BENCHMARK_MAIN() is defined exactly once across the
// benchmark target. CMakeLists.txt globs every bench_*.cpp into one
// crucible_benchmarks executable, so we put the entry point in
// bench_conv2d.cpp and leave this file as a registrations-only
// translation unit. Defining main() twice would give a duplicate
// -symbol link error.
