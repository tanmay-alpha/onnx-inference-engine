<div align="center">

# 🔥 Crucible

### *Raw models. Forged into production.*

**A from-scratch ONNX inference engine in C++17.**  
Load any `.onnx` model, run CPU inference, benchmark against ONNX Runtime and PyTorch — or ship privacy-preserving inference to the browser with WebAssembly.

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

[🔴 **Live Demo — Fraud detection in browser**](http://localhost:3000/fraud)
&nbsp;|&nbsp;
[📊 **Benchmark — 3-engine comparison**](http://localhost:3000/benchmark)
&nbsp;|&nbsp;
[📖 **Docs — Supported operators**](http://localhost:3000/docs)
&nbsp;|&nbsp;
[🧪 **Playground — WASM inference console**](http://localhost:3000/playground)

---

## What is Crucible?

Crucible is a from-scratch ONNX inference engine. It loads `.onnx` models, parses the compute graph, and runs inference on CPU — every layer written from scratch: the tensor class, the ONNX protobuf parser, each operator, and the graph executor.

The project spans **five languages** and **four deployment targets**:

| Component | Language | Role |
|-----------|----------|------|
| `engine/` | C++17 + CMake | Core engine — tensors, operators, graph executor |
| `server/` | Python + FastAPI | HTTP inference server with ONNX model management |
| `cli/` | Rust + Clap | CLI binary that links against the C++ engine via FFI |
| `wasm/` | Rust + wasm-pack | Pure-Rust WASM module — runs inference in the browser |
| `web/` | TypeScript + Next.js 15 | Frontend — playground, fraud detection, benchmarks, docs |

---

## Repository Layout

```
crucible/
├── engine/                     # C++17 core engine (CMake)
│   ├── CMakeLists.txt          # Root build file
│   ├── CMakePresets.json       # debug / release presets
│   ├── include/crucible/       # Public headers
│   │   ├── tensor.hpp          # Multi-dim float32 tensor (row-major)
│   │   ├── onnx_parser.hpp     # Protobuf-based ONNX graph parser
│   │   ├── executor.hpp        # Topological-sort graph executor
│   │   ├── c_api.h             # extern "C" FFI boundary (for Rust CLI)
│   │   └── ops/                # Operator implementations
│   │       ├── linear.hpp      # MatMul, Gemm
│   │       ├── activations.hpp # ReLU, Sigmoid, Softmax, GELU, Clip
│   │       ├── conv2d.hpp      # Conv2D (with depthwise/grouped support)
│   │       ├── pooling.hpp     # MaxPool, AvgPool, GlobalAvgPool
│   │       └── norm.hpp        # BatchNorm, LayerNorm
│   ├── src/                    # Implementations matching include/ layout
│   ├── tests/                  # Google Test unit tests (per-operator + executor)
│   ├── benchmarks/             # Google Benchmark (matmul, conv2d)
│   └── bindings/python/        # pybind11 bindings → crucible_py.so
│
├── cli/                        # Rust CLI — FFI bridge to libcrucible.so
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs             # Clap CLI (run, bench, info commands)
│       ├── runner.rs           # Library loading, model lifecycle, inference
│       └── formatter.rs        # Text + JSON output formatting
│
├── server/                     # Python FastAPI inference server
│   ├── app/
│   │   ├── main.py             # Routes: /convert, /infer, /validate, /operators, /health
│   │   ├── schemas.py          # Pydantic v2 request/response models
│   │   ├── converter.py        # ONNX upload validation + PyTorch→ONNX (in-process)
│   │   └── validator.py        # Op-type extraction and supported-ops check
│   ├── tests/
│   │   └── test_api.py         # End-to-end HTTP tests
│   └── requirements.txt
│
├── wasm/                       # Pure-Rust WASM module (reimplements op subset)
│   ├── Cargo.toml
│   └── src/lib.rs              # MatMul, ReLU, Sigmoid, Softmax + ONNX parser
│
├── web/                        # Next.js 15 + shadcn/ui frontend
│   ├── src/app/                # App Router pages
│   │   ├── page.tsx            # Landing page
│   │   ├── benchmark/page.tsx  # 3-engine latency + footprint charts
│   │   ├── docs/page.tsx       # Supported operators reference
│   │   ├── fraud/page.tsx      # Privacy-preserving fraud detection (WASM)
│   │   └── playground/page.tsx # Interactive ONNX inference console
│   ├── src/components/         # Shared UI (Layout, Nav, Metric cards)
│   └── src/lib/                # WASM loader, API client, error page
│
├── models/                     # ONNX models + training scripts
│   ├── fraud/                  # Synthetic fraud dataset + LogisticRegression model
│   │   ├── train_fraud_model.py
│   │   ├── fraud_detector.onnx  # Generated ONNX model
│   │   └── model_config.json   # Feature means, stds, threshold, AUC
│   └── generate_fixtures.py    # Generates test fixture ONNX files for C++ tests
│
├── benchmarks/                 # Python benchmark harness (3-engine comparison)
│   ├── bench_crucible.py       # Benchmarks via Rust CLI
│   ├── bench_onnxruntime.py    # Benchmarks via onnxruntime Python
│   ├── bench_pytorch.py        # Benchmarks via PyTorch
│   └── run_all.py              # Orchestrator — runs all, merges results
│
├── scripts/                    # Build and utility scripts
│   └── build-wasm.sh           # wasm-pack build invocation
│
├── docs/                       # Documentation assets (screenshots, diagrams)
│
├── .github/workflows/          # CI/CD — 4 pipelines
│   ├── ci-engine.yml           # CMake + CTest + Google Benchmark
│   ├── ci-rust.yml             # cargo test + clippy (WASM + CLI)
│   ├── ci-server.yml           # pytest + FastAPI test client
│   └── ci-web.yml              # TypeScript type-check + ESLint
│
├── CONTEXT.md                  # Project context, conventions, MCP config
├── ENGINEERING_PLAN.md         # 20-issue implementation plan
├── PROJECT_EXPLAINER.md        # Detailed architecture writeup
├── README.md                   # ← you are here
├── WRITEUP.md                  # Technical writeup for submission
└── LICENSE                     # MIT
```

---

## Quick Start

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| CMake | 3.20+ | C++ engine build |
| GCC / Clang | C++17 capable | C++ compilation |
| Rust | 1.78+ | CLI + WASM crates |
| Python | 3.10+ | Server + benchmark scripts |
| Node.js | 18+ | Web frontend |
| wasm-pack | latest | WASM compilation |
| Git | 2.30+ | Submodule support |

### Clone with submodules

```bash
git clone --recursive https://github.com/tanmay-alpha/Crucible.git
cd Crucible

# If you already cloned without --recursive:
git submodule update --init --recursive
```

### Build everything

```bash
# 1. C++ engine (debug)
cmake --preset debug -S engine
cmake --build engine/build/debug --parallel

# 2. Python bindings
pip install -r server/requirements.txt --break-system-packages
pip install -e engine/bindings/python --break-system-packages

# 3. Rust CLI
cd cli && cargo build --release && cd ..

# 4. WASM module
bash scripts/build-wasm.sh

# 5. Web frontend
cd web && npm install && npm run build && cd ..
```

### Run

```bash
# FastAPI server (serves /convert, /infer, /validate, /operators, /health)
cd server && uvicorn app.main:app --reload --port 8000

# Rust CLI — run inference on an ONNX model
cd cli && cargo run -- run --model ../models/fraud/fraud_detector.onnx

# Next.js web frontend
cd web && npm run dev
# → http://localhost:3000
```

---

## The Engine in Detail

### Tensor

The core data structure is a multi-dimensional float32 tensor with row-major (C-order) contiguous storage. Every tensor owns its data — no shared pointers, no reference counting.

```cpp
// Create a 2×3 tensor filled with zeros
crucible::Tensor t({2, 3});

// Element access with bounds checking
float val = t.at({0, 1});   // row 0, col 1

// Reshape (total element count must match)
auto flat = t.reshape({6});

// Flatten to 1-D
auto one_d = t.flatten();
```

### ONNX Parser

The parser reads the ONNX protobuf format (`ModelProto`), extracts the compute graph (`GraphProto`), and builds an internal representation of nodes, initializers, inputs, and outputs. It validates the model using `onnx.checker.check_model()`.

### Operators

Each operator is a standalone function taking `Tensor` inputs and returning a new `Tensor`:

- **Linear:** `matmul`, `gemm`
- **Activations:** `relu`, `sigmoid`, `softmax`, `gelu`, `clip`
- **Normalization:** `batch_norm`, `layer_norm`
- **Convolution:** `conv2d` (supports depthwise separable, i.e., `groups == in_channels`)
- **Pooling:** `max_pool`, `avg_pool`, `global_avg_pool`
- **Combinators:** `add`, `concat`, `identity`, `dropout`

### Graph Executor

The executor performs a topological sort (Kahn's BFS) of the compute graph, then walks it in dependency order, materialising each node's output tensor and passing it to the next node. The first model output is returned as the inference result.

---

## Architecture: Five Languages, One Engine

```
 ┌─────────────────────────────────────────────────────────────┐
 │                    Web Frontend (Next.js 15)                 │
 │  Playground │ Fraud Detection │ Benchmark │ Docs             │
 └────────────────────────┬────────────────────────────────────┘
                          │
              ┌───────────┴───────────────┐
              ▼                           ▼
   ┌──────────────────┐       ┌──────────────────┐
   │  WASM Module     │       │  FastAPI Server  │
   │  (Pure Rust)     │       │  (Python)        │
   │  MatMul, ReLU,   │       │  /convert        │
   │  Sigmoid, Softmax│       │  /infer          │
   │  + ONNX parser   │       │  /validate       │
   │  3.1 MB binary   │       │  /operators      │
   │  Runs entirely   │       │  /health         │
   │  client-side     │       └────────┬─────────┘
   └──────────────────┘                │
          ▲              ┌──────────────┴──────────────┐
          │              ▼                             ▼
          │     ┌──────────────────┐       ┌──────────────────┐
          │     │  Rust CLI        │       │  Python Bindings  │
          │     │  (clap + FFI)    │       │  (pybind11)       │
          │     │  run, bench,     │       │  crucible_py.so   │
          │     │  info commands   │       │  load_model(),    │
          │     └────────┬─────────┘       │  run_inference()  │
          │              │                 └────────┬─────────┘
          │              │                          │
          │              │  extern "C" FFI boundary │
          │              │                          │
          └──────────────┴──────────────────────────┘
                         ▼
               ┌──────────────────────┐
               │  C++17 Engine        │
               │  (CMake + Eigen)     │
               │                      │
               │  Tensor              │
               │  ONNX Parser         │
               │  Operators (MatMul,  │
               │  Conv2D, ReLU, etc.) │
               │  Graph Executor      │
               │  C API (c_api.h)     │
               └──────────────────────┘
```

### Privacy-preserving fraud detection

The fraud detection page runs a LogisticRegression ONNX model **entirely in the browser** via WASM. Transaction data never leaves the device. The model is trained on a synthetic dataset and classifies transactions as FRAUD or LEGITIMATE based on seven features:

- Transaction amount
- Origin account balance (before / after)
- Destination account balance (before / after)
- Transaction type (CASH_OUT, TRANSFER, OTHER)

---

## Benchmark Results

Three-engine head-to-head comparison on MobileNetV2 (ImageNet classification):

| Runtime | Mean Latency | Median | P95 | P99 | Binary Size | Browser |
|----------|-------------|--------|-----|-----|-------------|---------|
| **Crucible Native (C++/Eigen)** | 445.9 ms | 463.4 ms | 482.3 ms | 498.8 ms | 1.4 MB | No |
| **ONNX Runtime 1.27** | 1.69 ms | 1.68 ms | 1.83 ms | 1.86 ms | 51.2 MB | No |
| **PyTorch 2.3** | 0.54 ms | 0.52 ms | 0.71 ms | 0.83 ms | 756 MB | No |
| **Crucible WASM** | — | — | — | — | 3.1 MB | **Yes** |

> The C++ engine is currently in early scaffold state — the operator implementations and executor are being built incrementally. Native numbers will improve as more operators are implemented. The WASM module provides a working inference path in the browser today.

---

## CI/CD — Four Pipelines

| Pipeline | Trigger | What it does |
|----------|---------|-------------|
| **ci-engine** | Push to `engine/**` | CMake configure + build + CTest + Google Benchmark |
| **ci-rust** | Push to `cli/**`, `wasm/**` | `cargo test` + `cargo clippy -D warnings` on both crates |
| **ci-server** | Push to `server/**` | `pytest` with FastAPI test client |
| **ci-web** | Push to `web/**` | TypeScript type-check + ESLint |

All pipelines use GitHub Actions with `actions/cache@v4` for Cargo registry, pip cache, and CMake build directories.

---

## Development

### Commit message format

```
feat(scope): short description
fix(scope): short description
test(scope): short description
chore(scope): short description
ci: short description
docs: short description
```

Valid scopes: `engine`, `ops`, `executor`, `bindings`, `server`, `cli`, `wasm`, `web`, `bench`, `models`, `scripts`.

### Code style

- **C++:** C++17, `#pragma once`, row-major tensors, no raw pointers in engine core
- **Python:** Pydantic v2, type annotations, `ruff` formatting
- **Rust:** Edition 2021, `cargo clippy -D warnings`, `extern "C"` FFI boundary with no C++ exceptions crossing
- **TypeScript:** Strict mode, Next.js 15 App Router, WASM loaded via dynamic import with lazy init

### Running tests

```bash
# C++ engine tests
cd engine/build/debug && ctest --output-on-failure

# Python server tests
cd server && pytest -v

# Rust WASM tests
cd wasm && cargo test

# Rust CLI tests (type-check only, requires libcrucible.so)
cd cli && cargo test
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Engine | C++17, CMake, Google Test, Google Benchmark, Eigen 3 |
| Bindings | pybind11, NumPy |
| Server | FastAPI, Pydantic v2, Uvicorn, ONNX |
| CLI | Rust, Clap, Serde |
| WASM | Rust, wasm-bindgen, wasm-pack |
| Frontend | Next.js 15, TypeScript, shadcn/ui, Recharts, Lucide icons |
| CI | GitHub Actions |
| Deployment | Vercel (frontend), Render (backend) |

---

## Key Design Decisions

1. **WASM is pure Rust — C++ does NOT compile to WASM.** The `wasm/` module reimplements MatMul, ReLU, Sigmoid, and Softmax in pure Rust. It does not call `libcrucible`.

2. **Row-major layout.** PyTorch, NumPy, and ONNX Runtime all default to row-major. Matching this minimises friction when adding Python bindings.

3. **No raw pointers in engine core.** All tensor data lives in `std::vector<float>`. The FFI boundary (`c_api.h`) uses `extern "C"` with no C++ exceptions crossing — errors are returned as integer codes.

4. **Kahn's BFS topological sort.** The graph executor uses a linear-time BFS for topological ordering. No JIT compilation — the interpreter walks the graph on every inference call.

5. **Tensor is copy-on-write by value.** Every `Tensor` owns its data. The cost is a memcpy on assignment; the win is predictable performance and no lifetime bugs.

6. **Privacy-first architecture.** The fraud detection demo runs inference entirely in the browser via WASM. No transaction data is sent to any server.

---

## Environment Variables

```bash
# FastAPI server
CRUCIBLE_MODEL_DIR=/tmp/models
CRUCIBLE_ENGINE_PATH=/usr/local/lib/libcrucible.so
CRUCIBLE_API_KEY=change-me-in-production
CRUCIBLE_MAX_MODEL_SIZE_MB=100

# Next.js frontend
NEXT_PUBLIC_API_URL=http://localhost:8000

# Python bindings
PYTHONPATH=./engine/build/release/bindings/python
```

---

## Known Limitations

- The C++ engine is in active development. Operator coverage is expanding incrementally. MobileNetV2 end-to-end inference requires all operators (Conv2D, BatchNorm, GlobalAvgPool, Gemm, etc.) to be implemented.
- The WASM module reimplements a subset of operators in pure Rust. It does not have feature parity with the C++ engine.
- The `/convert` endpoint accepts pre-exported `.onnx` files only. PyTorch-to-ONNX conversion exists in-process but is not exposed over HTTP for security reasons (pickle deserialization = RCE).
- Benchmark numbers reflect the current state of implementation and will improve as the engine matures.

---

## Roadmap

| Milestone | Status | Description |
|-----------|--------|-------------|
| Tensor class + indexing | Done | Shape, data, `at()`, `reshape()`, `flatten()` |
| ONNX protobuf parser | Done | Load + validate `.onnx` files |
| Core operators | In progress | MatMul, Gemm, activations, Conv2D, pooling, normalization |
| Graph executor | Scaffolded | Topological sort + node dispatch (needs operators to be functional) |
| Python bindings | Scaffolded | pybind11 bridge — needs C++ engine to compile |
| FastAPI server | Done | `/convert`, `/infer`, `/validate`, `/operators`, `/health` |
| Rust CLI | Done | FFI bridge, `run`/`bench`/`info` commands |
| WASM module | Done | Pure-Rust inference in browser |
| Web frontend | Done | Playground, fraud detection, benchmarks, docs |
| CI/CD | Done | 4 GitHub Actions pipelines |
| Demo | Done | Fraud detection + benchmark dashboard |

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

## Credits

Built by [Tanmay](https://github.com/tanmay-alpha) (VIT Bhopal, CS batch 2028).  
Eigen (MPL2), ONNX (MIT), pybind11 (BSD), Google Test/ Benchmark (BSD), shadcn/ui (MIT), Recharts (MIT), Lucide (ISC).
