<div align="center">

# ⚡ ONNX Inference Engine

### *Raw models. Forged into production.*

**A from-scratch ONNX inference engine in C++17 & WebAssembly.**  
Load any `.onnx` model, run CPU inference, benchmark against ONNX Runtime and PyTorch — or ship privacy-preserving inference to the browser with WebAssembly.

[![CI — C++ Engine](https://github.com/tanmay-alpha/onnx-inference-engine/actions/workflows/ci-engine.yml/badge.svg)](https://github.com/tanmay-alpha/onnx-inference-engine/actions/workflows/ci-engine.yml)
[![CI — Rust](https://github.com/tanmay-alpha/onnx-inference-engine/actions/workflows/ci-rust.yml/badge.svg)](https://github.com/tanmay-alpha/onnx-inference-engine/actions/workflows/ci-rust.yml)
[![CI — Server](https://github.com/tanmay-alpha/onnx-inference-engine/actions/workflows/ci-server.yml/badge.svg)](https://github.com/tanmay-alpha/onnx-inference-engine/actions/workflows/ci-server.yml)
[![CI — Web](https://github.com/tanmay-alpha/onnx-inference-engine/actions/workflows/ci-web.yml/badge.svg)](https://github.com/tanmay-alpha/onnx-inference-engine/actions/workflows/ci-web.yml)

[![C++17](https://img.shields.io/badge/C++-17-00599C?logo=cplusplus&logoColor=white)](https://isocpp.org/)
[![Rust](https://img.shields.io/badge/Rust-1.78+-000000?logo=rust&logoColor=white)](https://rustup.rs/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org/)
[![Vite](https://img.shields.io/badge/Vite-8-646CFF?logo=vite&logoColor=white)](https://vitejs.dev/)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=white)](https://react.dev/)
[![WebAssembly](https://img.shields.io/badge/WebAssembly-WASM-654FF0?logo=webassembly&logoColor=white)](https://webassembly.org/)
[![SQLite/PostgreSQL](https://img.shields.io/badge/Database-SQLite%2FPostgreSQL-4169E1?logo=postgresql&logoColor=white)](https://postgresql.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](#license)

</div>

---

[🌐 **Live Production App — crucible-ivory-three.vercel.app**](https://crucible-ivory-three.vercel.app/)
&nbsp;|&nbsp;
[🔴 **Fraud Demo**](https://crucible-ivory-three.vercel.app/fraud)
&nbsp;|&nbsp;
[📊 **Benchmark Engine**](https://crucible-ivory-three.vercel.app/benchmark)
&nbsp;|&nbsp;
[📖 **Operator Docs**](https://crucible-ivory-three.vercel.app/docs)
&nbsp;|&nbsp;
[🧪 **WASM Playground**](https://crucible-ivory-three.vercel.app/playground)

---

## ⚡ What is ONNX Inference Engine?

ONNX Inference Engine is a high-performance, multi-language **ONNX inference runtime built from first principles**. It parses `.onnx` compute graphs, performs topological graph execution, and evaluates tensor operations on CPU — with zero heavy external runtime dependencies. Every layer was written from scratch: the row-major float32 Tensor data structures, the protobuf ONNX graph decoder, individual operator kernels, and the execution engine.

The project spans **five languages** and **four deployment targets**:

| Component | Language / Framework | Key Responsibilities |
|-----------|----------------------|----------------------|
| `engine/` | **C++17 + CMake + Eigen** | Core inference engine — tensors, protobuf graph parser, ops, Kahn's algorithm executor, pybind11 bindings |
| `server/` | **Python 3.11 + FastAPI** | RESTful inference API, ONNX model management, security auth middleware, dual SQLite/PostgreSQL persistence |
| `cli/` | **Rust + Clap** | High-performance CLI tool interacting with the compiled C++ core engine via C-FFI bindings |
| `wasm/` | **Rust + wasm-pack** | Pure-Rust WebAssembly runtime compiled for zero-latency, privacy-preserving in-browser inference |
| `web/` | **React 19 + Vite 8 + TanStack Start** | Modern web dashboard — interactive WASM playground, live fraud detector, benchmark charts, operator docs |

---

## 🏗️ Repository Layout

```
onnx-inference-engine/
├── engine/                     # Core C++17 inference engine (CMake)
│   ├── CMakeLists.txt          # Root build configuration
│   ├── CMakePresets.json       # Debug & Release presets
│   ├── include/crucible/       # Public engine headers
│   │   ├── tensor.hpp          # Row-major float32 Tensor class
│   │   ├── onnx_parser.hpp     # Protobuf ONNX compute graph decoder
│   │   ├── executor.hpp        # Topological-sort (Kahn's BFS) graph executor
│   │   ├── c_api.h             # C-FFI export boundary (for Rust CLI bridge)
│   │   └── ops/                # Operator kernels (Conv2D, MatMul, Gemm, ReLU, Softmax, etc.)
│   ├── src/                    # C++ source code matching include layout
│   ├── tests/                  # Google Test suite (per-operator + graph tests)
│   ├── benchmarks/             # Google Benchmark suite (MatMul, Conv2D)
│   └── bindings/python/        # pybind11 C++ extension module (`crucible_py.so`)
│
├── server/                     # FastAPI inference server & persistence
│   ├── main.py                 # FastAPI application, authentication middleware, REST routes
│   ├── database.py             # Thread-safe dual-engine persistence (SQLite / Supabase PostgreSQL)
│   ├── schemas.py              # Pydantic v2 data validation schemas
│   ├── converter.py            # ONNX upload validator & PyTorch conversion helper
│   ├── validator.py            # ONNX graph operator extractor & compatibility checker
│   ├── tests/                  # pytest server suite (21/21 passing tests)
│   └── requirements.txt        # Server dependencies
│
├── cli/                        # Rust CLI binary (FFI bridge to C++ engine)
│   ├── Cargo.toml
│   └── src/                    # Clap CLI commands (`run`, `bench`, `info`)
│
├── wasm/                       # Pure-Rust WebAssembly module
│   ├── Cargo.toml
│   └── src/lib.rs              # In-browser WASM inference engine for fraud detection
│
├── web/                        # Vite 8 + TanStack Start frontend app
│   ├── src/routes/             # React 19 pages (`index`, `fraud`, `playground`, `benchmark`, `docs`)
│   ├── src/lib/                # API client (`api.ts`), WASM wrapper (`crucible-wasm.ts`)
│   └── scripts/postbuild.js    # Vercel Output API v3 copy automation
│
├── models/                     # Synthetic fraud detection ONNX models & fixture generators
├── benchmarks/                 # Python benchmarking harness (Crucible vs. ONNX Runtime vs. PyTorch)
├── scripts/                    # Helper build scripts (`build-wasm.sh`)
├── Dockerfile                  # Multi-stage container build (Debian builder + slim runtime)
├── render.yaml                 # Render Blueprint configuration (Docker API + Static Web)
└── vercel.json                 # Vercel deployment configuration
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites

- **C++ Compiler**: GCC 10+ or Clang 12+ (C++17 compliant)
- **CMake**: 3.20+ & Ninja / Make
- **Python**: 3.11+
- **Rust**: 1.78+ & `wasm-pack`
- **Node.js**: 20+ & `npm`

### 2. Clone Repository with Submodules

```bash
git clone --recursive https://github.com/tanmay-alpha/onnx-inference-engine.git
cd onnx-inference-engine

# If cloned without --recursive:
git submodule update --init --recursive
```

### 3. Build & Run Components

#### A. C++ Core Engine
```bash
# Configure & build C++ engine in Release mode
cmake --preset release -S engine -B build/release
cmake --build build/release --parallel

# Run Google Test suite
ctest --test-dir build/release --output-on-failure
```

#### B. Python FastAPI Inference Server
```bash
# Install dependencies
pip install -r server/requirements.txt

# Run server on port 8000 (starts SQLite db automatically)
python -m uvicorn server.main:app --reload --port 8000
```

#### C. Rust WebAssembly Engine
```bash
# Compile WASM module to web/public/wasm/
bash scripts/build-wasm.sh
```

#### D. Web Dashboard (Vite + React 19)
```bash
cd web
npm install
npm run dev
# → Open http://localhost:3000
```

#### E. Python Server Test Suite
```bash
python -m pytest server -v
```

---

## 🗄️ Database & REST API Architecture

Crucible features a persistent REST API with a thread-safe dual-engine database layer ([server/database.py](file:///C:/Users/TANMAY/OneDrive/Desktop/Crucible/server/database.py)):

- **Local Development**: Automatically initializes local SQLite database (`crucible.db`) with WAL mode.
- **Production / Cloud**: Seamlessly switches to **Supabase PostgreSQL** or **Render Managed PostgreSQL** when `DATABASE_URL` is set in environment variables.

### Main REST API Endpoints

| Endpoint | Method | Auth Required | Description |
| :--- | :---: | :---: | :--- |
| `/health` | `GET` | No | Liveness probe returning server status and active engine |
| `/operators` | `GET` | No | Catalogue of supported ONNX operators |
| `/convert` | `POST` | Yes (`X-API-Key`) | Upload `.onnx` model, validate ops, and register model ID |
| `/infer` | `POST` | Yes (`X-API-Key`) | Execute tensor inference & record execution metrics to DB |
| `/validate` | `POST` | Yes (`X-API-Key`) | Validate model file or registered `model_id` |
| `/models` | `GET` | No | Retrieve list of all registered models from database |
| `/models/{id}` | `DELETE` | Yes (`X-API-Key`) | Remove model record from database and storage |
| `/inference/logs` | `GET` | No | Fetch recent inference execution log history |
| `/fraud/log` | `POST` | No | Record fraud check outcome to database |
| `/fraud/history` | `GET` | No | Fetch recent fraud transaction evaluation records |
| `/benchmarks` | `GET` / `POST` | No | Retrieve and submit performance benchmark results |

---

## ☁️ Production Deployment

### Vercel (Web Frontend)
The frontend builds using Nitro with Vercel Output API v3 preset (`web/scripts/postbuild.js`), making SSR functions and static assets natively hostable on Vercel.

### Render & Supabase PostgreSQL (Full Stack)
The project includes a production-ready `render.yaml` Blueprint for Render deployment:
- **`crucible-api`**: Multi-stage Docker deployment (`Dockerfile`) running C++ engine & FastAPI on Render Web Service.
- **`crucible-web`**: Static Publish deployment serving Vite/TanStack build outputs.
- **`DATABASE_URL`**: Pointing to Supabase PostgreSQL for permanent zero-cost persistence.

---

## 📜 License

Crucible is open-source software licensed under the [MIT License](LICENSE).
