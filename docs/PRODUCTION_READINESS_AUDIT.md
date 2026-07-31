# Production Readiness Audit Report — Crucible Engine

**Project Name**: Crucible (`tanmay-alpha/onnx-inference-engine`)  
**Audit Date**: July 31, 2026  
**Auditor**: Senior C++ Systems & ML Infrastructure Engineer  
**Branch**: `feature/production-readiness-audit-and-fixes`

---

## 1. Architecture Summary

Crucible is a multi-tier ONNX ML inference ecosystem comprising:
1. **Core C++ Engine (`engine/`)**: A C++17 library with a hermetic Protobuf binary graph decoder ([onnx_parser.cpp](file:///c:/Users/TANMAY/OneDrive/Desktop/onnx-inference-engine/engine/src/onnx_parser.cpp)), multi-dimensional row-major `Tensor` memory storage ([tensor.cpp](file:///c:/Users/TANMAY/OneDrive/Desktop/onnx-inference-engine/engine/src/tensor.cpp)), Kahn topological BFS graph executor ([executor.cpp](file:///c:/Users/TANMAY/OneDrive/Desktop/onnx-inference-engine/engine/src/executor.cpp)), and Eigen-backed operator implementations ([linear.cpp](file:///c:/Users/TANMAY/OneDrive/Desktop/onnx-inference-engine/engine/src/ops/linear.cpp), [conv2d.cpp](file:///c:/Users/TANMAY/OneDrive/Desktop/onnx-inference-engine/engine/src/ops/conv2d.cpp), [activations.cpp](file:///c:/Users/TANMAY/OneDrive/Desktop/onnx-inference-engine/engine/src/ops/activations.cpp)).
2. **C ABI & Shared Library ([c_api.h](file:///c:/Users/TANMAY/OneDrive/Desktop/onnx-inference-engine/engine/include/crucible/c_api.h))**: C-linkage export interface providing opaque `CrucibleModel*` handles and `crucible_run` functions for cross-language FFI bindings.
3. **Python Bindings (`engine/bindings/python/`)**: pybind11 C++ extension module (`crucible_py`) supporting model loading, input dict wrapping, and direct inference execution.
4. **Rust CLI (`cli/`)**: Command-line application using `clap` and safe Rust FFI wrappers around `crucible.dll`/`libcrucible.so` supporting `run`, `benchmark`, `validate`, and `info` subcommands.
5. **Rust WebAssembly Runtime (`wasm/`)**: Standalone pure-Rust ONNX decoder and executor compiled via `wasm-bindgen` for zero-network client-side browser execution.
6. **FastAPI Backend Server (`server/`)**: Python API providing model management, authentication (JWT & API keys), database storage (SQLAlchemy / Alembic / SQLite / PostgreSQL), transaction fraud detection endpoints, and ONNX conversion pipelines.
7. **TanStack Start Frontend (`web/`)**: React TypeScript web dashboard hosting an interactive ONNX playground, WASM privacy-first fraud detection UI, benchmark visualizations, and API key portal.

---

## 2. Build Systems & Supported Platforms
- **C++ Engine**: CMake 3.25+ with MSVC 2022 / GCC 11+ / Clang 14+ targets. Supported on Windows (x64), Linux (x86_64/aarch64), and macOS (arm64/x86_64).
- **Python**: Python 3.10–3.13 wheel/pyd compilation via pybind11 and setuptools/pip.
- **Rust / WASM**: Cargo toolchain with `wasm-pack` targeting `wasm32-unknown-unknown`.
- **Frontend**: Node.js 20+ / npm / Nitro / Vercel static build.

---

## 3. Findings Matrix & Categorization

| ID | Category | Severity | File Path | Finding Description & Impact |
| :--- | :--- | :---: | :--- | :--- |
| **AUD-01** | Build / Hygiene | **P0** | [CMakeLists.txt](file:///c:/Users/TANMAY/OneDrive/Desktop/onnx-inference-engine/engine/CMakeLists.txt) | Unknown `-O3` compiler flag warning emitted on MSVC builds; Python executable selection mismatch when multiple Python versions exist. |
| **AUD-02** | Performance | **P1** | [conv2d.cpp](file:///c:/Users/TANMAY/OneDrive/Desktop/onnx-inference-engine/engine/src/ops/conv2d.cpp#L191) | `im2col` allocates fresh `RowMatrix` heap memory on every batch loop iteration $n$, causing high latency on large vision models (821ms for MobileNetV2). |
| **AUD-03** | Performance | **P1** | [executor.cpp](file:///c:/Users/TANMAY/OneDrive/Desktop/onnx-inference-engine/engine/src/executor.cpp) | Lack of ahead-of-time graph optimization pass (BatchNorm folding into Conv weights), leading to redundant node passes and memory transfers. |
| **AUD-04** | Performance | **P2** | [conv2d.cpp](file:///c:/Users/TANMAY/OneDrive/Desktop/onnx-inference-engine/engine/src/ops/conv2d.cpp), [activations.cpp](file:///c:/Users/TANMAY/OneDrive/Desktop/onnx-inference-engine/engine/src/ops/activations.cpp) | Single-threaded execution without OpenMP multi-core parallelization over spatial outputs or batch dimensions. |
| **AUD-05** | Performance | **P2** | [crucible_py.cpp](file:///c:/Users/TANMAY/OneDrive/Desktop/onnx-inference-engine/engine/bindings/python/crucible_py.cpp#L70) | Input NumPy array buffers undergo forced memory copies in `ndarray_to_tensor` without zero-copy non-owning view support. |
| **AUD-06** | Performance | **P2** | [activations.cpp](file:///c:/Users/TANMAY/OneDrive/Desktop/onnx-inference-engine/engine/src/ops/activations.cpp#L97) | Activation functions (`ReLU`, `Sigmoid`, `Softmax`, `GELU`) use scalar single-element loops without explicit SIMD vectorization pragmas. |
| **AUD-07** | Operator Coverage | **P2** | [executor.cpp](file:///c:/Users/TANMAY/OneDrive/Desktop/onnx-inference-engine/engine/src/executor.cpp#L636) | Operators `Tanh`, `LeakyRelu`, `Elu`, `Clip`, `Shape`, `Gather`, `Unsqueeze`, `Squeeze`, `Resize` throw "unsupported" errors on graph dispatch. |
| **AUD-08** | Documentation | **P1** | [README.md](file:///c:/Users/TANMAY/OneDrive/Desktop/onnx-inference-engine/README.md) | Misleading README claims ("0.1ms AVX-512 execution", "3x faster than ONNX Runtime", "Zero-copy python", "Lock-free parallel executor") contradict empirical behavior. |
| **AUD-09** | CI/CD | **P2** | [.github/workflows/](file:///c:/Users/TANMAY/OneDrive/Desktop/onnx-inference-engine/.github/workflows/) | Missing GitHub Actions workflow for automated WASM build, C++ Release CTest verification, and full-stack integration testing. |
| **AUD-10** | Hygiene | **P2** | [.gitignore](file:///c:/Users/TANMAY/OneDrive/Desktop/onnx-inference-engine/.gitignore) | Generated benchmark results (`benchmarks/results.json`, `benchmarks/BENCHMARK_REPORT.md`) untracked or scattered in root repository. |

---

## 4. Itemized Actionable Findings

### P0 Findings (Blockers)
- **AUD-01: Build System Flag Warnings & Python Resolution**
  - **Evidence**: `cl : command line warning D9002: ignoring unknown option '-O3'` when building with MSVC.
  - **Proposed Fix**: Guard GCC/Clang specific compile flags (`-O3`, `-march=native`) under compiler ID checks (`GNU`, `Clang`, `AppleClang`) in `engine/CMakeLists.txt`.

### P1 Findings (Correctness, Security, Performance & Docs Integrity)
- **AUD-02: Conv2D Heap Allocation Overhead**
  - **Evidence**: `RowMatrix col_mat(out_h * out_w, ksize)` allocated per batch in `conv2d.cpp`.
  - **Proposed Fix**: Pre-allocate `col_mat` scratch buffer once per `conv2d_forward` invocation and reuse memory across batch loops.
- **AUD-03: Missing Graph Optimization Pass**
  - **Evidence**: `BatchNormalization` nodes executed as separate runtime dispatches after `Conv` nodes.
  - **Proposed Fix**: Implement `graph_optimizer.cpp` with a `fold_batchnorm` pass that merges $\gamma, \beta, \mu, \sigma^2$ into Conv weights $W'$ and bias $B'$.
- **AUD-08: Exaggerated Documentation Claims**
  - **Evidence**: `README.md` and `WRITEUP.md` claim 3× speedup over ONNX Runtime and 0.1ms AVX-512 execution.
  - **Proposed Fix**: Rewrite `README.md` to state exact, measured, reproducible baseline benchmark metrics and document real performance profiles.

### P2 Findings (Maintainability, Testing & Optimizations)
- **AUD-04: Multi-Core OpenMP Parallelization**
  - **Evidence**: Single-threaded loops in `conv2d.cpp`, `linear.cpp`, `activations.cpp`.
  - **Proposed Fix**: Enable CMake `OpenMP` package and add `#pragma omp parallel for` across spatial/batch loops.
- **AUD-05: Non-Owning Zero-Copy Tensor Views**
  - **Evidence**: `ndarray_to_tensor` in `crucible_py.cpp` performs `std::vector<float>` element copy.
  - **Proposed Fix**: Add zero-copy view support for C-contiguous float32 NumPy buffers in Python bindings.
- **AUD-06: SIMD Activation Kernel Vectorization**
  - **Evidence**: `activations.cpp` uses scalar loops for `ReLU`, `Sigmoid`, `Softmax`, `GELU`.
  - **Proposed Fix**: Add `#pragma omp simd` vectorization directives and AVX2-friendly loop structures.
- **AUD-07: Operator Coverage Expansion**
  - **Evidence**: Dispatch table in `executor.cpp` throws on `Tanh`, `LeakyRelu`, `Elu`, `Clip`, `Shape`, `Gather`, `Unsqueeze`, `Squeeze`, `Resize`.
  - **Proposed Fix**: Extend operator dispatches and add unit tests for missing ONNX operators.
- **AUD-09: Complete CI/CD GitHub Actions Workflows**
  - **Evidence**: Absence of unified GitHub Actions workflow running C++ Release builds, Python PyTest, Rust Clippy, and WASM build validation.
  - **Proposed Fix**: Create `.github/workflows/ci.yml` covering all subprojects.
