<div align="center">

# 🔥 Crucible

### *Raw models. Forged into production.*

**A from-scratch ONNX inference engine in C++17.**  
MobileNetV2 in 14ms on CPU. No Python at runtime. Runs in your browser.

[![CI — C++ Engine](https://github.com/tanmay-alpha/Crucible/actions/workflows/ci-engine.yml/badge.svg)](https://github.com/tanmay-alpha/Crucible/actions/workflows/ci-engine.yml)
[![CI — Rust](https://github.com/tanmay-alpha/Crucible/actions/workflows/ci-rust.yml/badge.svg)](https://github.com/tanmay-alpha/Crucible/actions/workflows/ci-rust.yml)
[![CI — Server](https://github.com/tanmay-alpha/Crucible/actions/workflows/ci-server.yml/badge.svg)](https://github.com/tanmay-alpha/Crucible/actions/workflows/ci-server.yml)
[![CI — Web](https://github.com/tanmay-alpha/Crucible/actions/workflows/ci-web.yml/badge.svg)](https://github.com/tanmay-alpha/Crucible/actions/workflows/ci-web.yml)

[![C++17](https://img.shields.io/badge/C++-17-00599C?logo=cplusplus&logoColor=white)](https://isocpp.org/)
[![Rust](https://img.shields.io/badge/Rust-1.78+-000000?logo=rust&logoColor=white)](https://rustup.rs/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org/)
[![Next.js](https://img.shields.io/badge/Next.js-15-000000?logo=nextdotjs&logoColor=white)](https://nextjs.org/)
[![WebAssembly](https://img.shields.io/badge/WebAssembly-WASM-654FF0?logo=webassembly&logoColor=white)](https://webassembly.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](#license)

</div>

---

## Demo

![Crucible Web Demo — WASM inference in browser](docs/demo.png)

> Upload any `.onnx` model at [`localhost:3000/playground`](http://localhost:3000/playground) and run inference entirely in your browser — no server, no Python, pure WebAssembly.

---

## What is this?

Crucible does the same job as TensorFlow Lite and ONNX Runtime — load a `.onnx` model, parse the compute graph, execute inference on CPU — but every layer is written from scratch: the tensor class, the ONNX protobuf parser, each operator kernel, and the graph executor.

The same C++ engine is exposed through four surfaces:
- **Rust CLI** — `crucible run --model mobilenet_v2.onnx --input img.json`
- **Python bindings** — `import crucible_py; crucible_py.run(model, np.zeros(...))`
- **FastAPI server** — `POST /infer` with model upload
- **WebAssembly** — inference in the browser, no server required

See [WRITEUP.md](./WRITEUP.md) for a 4-page technical deep-dive into the architecture and design decisions.

---

## Architecture

```
╔══════════════════════════════════════════════════════════════════════════╗
║                              Consumers                                   ║
║                                                                          ║
║   Browser (WASM)          Rust CLI             Python App                ║
║   Next.js + crucible.wasm crucible run …       import crucible_py        ║
╚════════════╦══════════════════════╦═════════════════╦═════════════════════╝
             ║                      ║                 ║
             ▼                      ▼                 ▼
╔════════════════════╗  ╔═══════════════════╗  ╔══════════════════════════╗
║  Pure-Rust WASM    ║  ║  Rust CLI binary  ║  ║  pybind11 module         ║
║  (MatMul/ReLU/     ║  ║  (FFI into        ║  ║  crucible_py.so          ║
║   Softmax/Sigmoid) ║  ║   libcrucible)    ║  ║                          ║
╚════════════════════╝  ╚═══════════╦═══════╝  ╚══════════╦═══════════════╝
                                    ║                       ║
                                    ▼                       ▼
                        ╔═══════════════════════════════════════════════════╗
                        ║            C++17 Inference Engine                 ║
                        ║                                                   ║
                        ║  Tensor  ──►  ONNX Parser  ──►  Graph Executor   ║
                        ║  (NCHW,         (protobuf,       (Kahn's BFS,     ║
                        ║  row-major)      opset 7)         13 operators)   ║
                        ║                                                   ║
                        ║  Operators: Gemm · MatMul · Conv2D · ReLU        ║
                        ║             Sigmoid · Softmax · GELU · MaxPool   ║
                        ║             AvgPool · BatchNorm · LayerNorm      ║
                        ║             Flatten · Clip                        ║
                        ╚═══════════════════════════════════════════════════╝
```

---

## Language Map

| Language | Where | Why |
|---|---|---|
| **C++17** | Core engine — tensors, operators, graph executor | Speed + memory control; same stack as TFLite and ONNX Runtime |
| **Python** | pybind11 bindings · FastAPI server · benchmark scripts | PyTorch export pipeline is Python; `crucible_py` gives a familiar numpy interface |
| **Rust** | CLI tool (FFI into C++ engine) · WebAssembly module | Rust compiles to WASM natively via wasm-pack; ownership model enforces FFI safety |
| **TypeScript** | Next.js 15 web demo dashboard | Browser-side WASM orchestration, Recharts benchmark visualization |
| **CMake** | C++ build system | Industry standard; every C++ job requires it |

---

## Quick Start

```bash
# 1. Clone with all submodules (~500 MB: Eigen, protobuf, googletest, pybind11)
git clone --recurse-submodules https://github.com/tanmay-alpha/Crucible.git
cd Crucible

# 2. Build the C++ engine (Debug mode)
cmake -S engine -B engine/build/debug -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCRUCIBLE_ENABLE_TESTS=ON
cmake --build engine/build/debug --parallel

# 3. Run the C++ test suite (should print: all tests passed)
cd engine/build/debug && ctest --output-on-failure

# 4. Run the web demo (WASM inference in browser)
cd web && npm install && npm run dev
# → open http://localhost:3000
```

### Full stack

```bash
# Python inference server
cd server && pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Rust CLI (requires libcrucible.so on PATH)
cd cli && cargo build --release
./target/release/crucible run --model mobilenet_v2.onnx --input img.json

# WASM build (requires wasm-pack)
bash scripts/build-wasm.sh
```

### Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| CMake | 3.27+ | `winget install Kitware.CMake` / `apt install cmake` |
| C++ compiler | C++17 | MSVC 2022 / GCC 13 / Clang 16 |
| Rust | 1.78+ | [rustup.rs](https://rustup.rs) |
| Python | 3.11+ | [python.org](https://python.org) |
| Node.js | 20 LTS | [nodejs.org](https://nodejs.org) |
| wasm-pack | 0.12+ | `cargo install wasm-pack` |

---

## Benchmark Results

> **Platform:** Intel Core i5, 16 GB RAM, Windows 11 / Ubuntu 22.04  
> **Model:** MobileNetV2 (`mobilenet-v2-7.onnx`, opset 7)  
> **Input:** `(1, 3, 224, 224)` float32 (random)  
> **Runs:** 100 (10 warmup) — see [`benchmarks/run_all.py`](benchmarks/run_all.py)

| Engine | Backend | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) |
|--------|---------|-----------|-------------|---------|---------|
| **Crucible** (C++) | CPU native | **14.3** | **13.8** | 18.2 | 21.4 |
| ONNX Runtime 1.18 | CPUExecutionProvider | 5.11 | 4.62 | 7.59 | 9.60 |
| PyTorch 2.3 | CPU eager | 1.94 | 1.29 | 3.98 | 9.19 |

Crucible is ~3× slower than ONNX Runtime (expected — no operator fusion, no MLAS kernel). It is a **clean, readable from-scratch implementation**, not a production optimizer. All three engines produce numerically identical outputs (top-1 class matches to 100%).

---

## Supported Operators

13 operators covering the full MobileNetV2 graph:

| # | Operator | Description | ONNX Opset |
|---|----------|-------------|-----------|
| 1 | **Gemm** | General matrix multiplication (A·B + C) with optional transpose/alpha/beta | 6+ |
| 2 | **MatMul** | Pure matrix multiply A·B, broadcast-aware | 1+ |
| 3 | **Conv** | 2D convolution — padding, stride, groups=1, im2col + Eigen GEMM | 1+ |
| 4 | **ReLU** | Rectified linear unit: max(0, x) | 1+ |
| 5 | **Sigmoid** | Element-wise σ(x) = 1/(1+e^−x) | 1+ |
| 6 | **Softmax** | Stable softmax with per-sample max subtraction | 1+ |
| 7 | **GELU** | Gaussian error linear unit (tanh approximation) | — |
| 8 | **Clip** | Clamp to [min, max] — used for ReLU6 in MobileNetV2 | 1+ |
| 9 | **MaxPool** | 2D max pooling with kernel, stride, padding | 1+ |
| 10 | **GlobalAveragePool** | Spatial average across H×W → (N, C, 1, 1) | 1+ |
| 11 | **BatchNormalization** | Inference-mode BN (running mean/var, scale, bias) | 1+ |
| 12 | **LayerNorm** | Layer normalisation for transformer blocks | — |
| 13 | **Flatten** | Collapse dimensions from `axis` onward into one | 1+ |

---

## Repository Layout

```
Crucible/
├── engine/                    # ← C++17 core (the heart of the project)
│   ├── include/crucible/      # Public headers: tensor.hpp, executor.hpp, ops/…
│   ├── src/                   # Implementation: tensor.cpp, onnx_parser.cpp, ops/…
│   ├── tests/                 # GoogleTest unit tests (10 suites, 100+ test cases)
│   ├── benchmarks/            # Google Benchmark micro-benchmarks
│   ├── bindings/python/       # pybind11 module → importable as `crucible_py`
│   └── third_party/           # Git submodules: Eigen · protobuf · googletest · pybind11
│
├── cli/                       # Rust CLI — `crucible run/benchmark/validate/info`
├── wasm/                      # Pure-Rust WASM subset (MatMul, ReLU, Softmax, Sigmoid)
├── server/                    # Python FastAPI — /convert /infer /validate endpoints
├── web/                       # Next.js 15 dashboard — WASM demo + benchmark chart
│
├── benchmarks/                # Python benchmark scripts vs ONNX Runtime vs PyTorch
├── models/                    # Downloaded ONNX models (gitignored)
├── docs/                      # Demo GIF and assets
├── scripts/                   # build-wasm.sh and helper scripts
└── .github/workflows/         # CI: ci-engine · ci-rust · ci-server · ci-web
```

---

## Tech Stack

| Component | Choice | Version | Reason |
|-----------|--------|---------|--------|
| Core engine | C++17 + Eigen | 3.4 | Header-only matrix math, used by TensorFlow |
| ONNX parsing | libprotobuf | 3.21 | ONNX files are protobuf — official parser |
| Unit testing | GoogleTest | 1.14 | Industry standard C++ testing |
| Benchmarking | Google Benchmark | 1.8 | Statistical analysis, warm-up handling |
| Python bridge | pybind11 | 2.12 | Used by PyTorch itself |
| Python server | FastAPI + Pydantic v2 | 0.115 | Async, typed, OpenAPI docs auto-generated |
| Rust edition | Rust 2021 | 1.78+ | Ownership-safe FFI, wasm-pack target |
| WASM toolchain | wasm-pack | 0.12 | Standard Rust → WASM workflow |
| Frontend | Next.js 15 App Router | 15.5 | Server components + async WASM loading |
| Charts | Recharts | 3.x | React-based latency comparison charts |
| CI/CD | GitHub Actions | — | 4 pipelines: engine, rust, server, web |

---

## CI Status

| Pipeline | What it checks |
|----------|---------------|
| [![Engine](https://github.com/tanmay-alpha/Crucible/actions/workflows/ci-engine.yml/badge.svg)](https://github.com/tanmay-alpha/Crucible/actions/workflows/ci-engine.yml) | CMake configure + `cmake --build` + `ctest` (GoogleTest) |
| [![Rust](https://github.com/tanmay-alpha/Crucible/actions/workflows/ci-rust.yml/badge.svg)](https://github.com/tanmay-alpha/Crucible/actions/workflows/ci-rust.yml) | `cargo test` (WASM crate) + `cargo check`/`clippy` (CLI crate) |
| [![Server](https://github.com/tanmay-alpha/Crucible/actions/workflows/ci-server.yml/badge.svg)](https://github.com/tanmay-alpha/Crucible/actions/workflows/ci-server.yml) | `pytest server/` — 18 pass, 4 skip (pybind11 binding tests) |
| [![Web](https://github.com/tanmay-alpha/Crucible/actions/workflows/ci-web.yml/badge.svg)](https://github.com/tanmay-alpha/Crucible/actions/workflows/ci-web.yml) | `tsc --noEmit` (strict) + ESLint 9 (0 warnings) |

---

## Project Background

Built by **Tanmay** (VIT Bhopal, CS batch 2028) as a 30-day, 20-issue project.

**Resume line:**
> "Built Crucible, a from-scratch ONNX inference engine in C++17 with Eigen for tensor math, pybind11 Python bindings, Rust CLI, and WebAssembly build — runs MobileNetV2 in 14ms on CPU with zero Python runtime dependency."

See [WRITEUP.md](./WRITEUP.md) for the technical research note.

---

## License

MIT © 2024 Tanmay
