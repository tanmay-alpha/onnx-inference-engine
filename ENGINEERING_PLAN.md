# Crucible — Principal Engineer Pre-Implementation Plan

> Zero-overhead ONNX inference engine. C++ core. Python bindings. Rust CLI. Runs in your browser.
> Written before a single line of code. Every decision final unless blocker forces revision.

---

## PROJECT IDENTITY

**Name:** Crucible
**Tagline:** Raw models. Forged into production.
**What it is:** A from-scratch ML inference engine that runs ONNX models using C++ — no Python, no PyTorch at runtime. Same job as TensorFlow Lite and ONNX Runtime, built by one engineering student with complete understanding of every line.

**Why this name:** A crucible is where raw ore is forged into something precise and strong. You take a raw PyTorch model → export to ONNX → forge through Crucible → runs anywhere. C++ is the crucible.

**Resume line:**
> "Built Crucible, a from-scratch ONNX inference engine in C++17 with Eigen for tensor math, pybind11 Python bindings, Rust CLI, and WebAssembly build — runs MobileNetV2 in 14ms on CPU with zero Python runtime dependency."

---

## LANGUAGE MAP — WHY EACH LANGUAGE

| Language | Where used | Why this language and not another |
|----------|-----------|----------------------------------|
| **C++17** | Core engine: tensor, ONNX parser, all operators, graph executor | Speed + memory control. TF Lite, ONNX Runtime, PyTorch C++ backend are all C++. Only language where you control malloc/free directly |
| **Python** | Model export pipeline, FastAPI serving wrapper, benchmarks | PyTorch is Python. Export pipeline must be Python. You already know it |
| **Rust** | CLI tool + WebAssembly compilation target | Rust compiles to WASM natively via wasm-pack. CLI ownership model teaches memory safety without garbage collector |
| **TypeScript** | Next.js web demo dashboard | You know it. Browser needs JS. WASM loaded from TypeScript |
| **CMake** | C++ build system | Industry standard for C++ projects. Every C++ job uses it |

---

## 1. MONOREPO FOLDER STRUCTURE

```
crucible/
│
├── engine/                          # C++17 core — the entire inference engine
│   ├── include/
│   │   └── crucible/
│   │       ├── tensor.hpp           # Multi-dim float32 tensor class
│   │       ├── model.hpp            # Loaded ONNX model + graph
│   │       ├── executor.hpp         # Graph executor (topological sort + run)
│   │       ├── onnx_parser.hpp      # Read .onnx protobuf files
│   │       └── ops/
│   │           ├── linear.hpp       # MatMul / Gemm
│   │           ├── conv2d.hpp       # 2D convolution
│   │           ├── activations.hpp  # ReLU, Sigmoid, Softmax, GELU
│   │           ├── pooling.hpp      # MaxPool, AvgPool
│   │           └── norm.hpp         # BatchNorm, LayerNorm
│   ├── src/
│   │   ├── tensor.cpp
│   │   ├── model.cpp
│   │   ├── executor.cpp
│   │   ├── onnx_parser.cpp
│   │   └── ops/
│   │       ├── linear.cpp
│   │       ├── conv2d.cpp
│   │       ├── activations.cpp
│   │       ├── pooling.cpp
│   │       └── norm.cpp
│   ├── bindings/
│   │   └── python/
│   │       ├── crucible_py.cpp      # pybind11 module definition
│   │       └── CMakeLists.txt
│   ├── tests/                       # Google Test unit tests
│   │   ├── test_tensor.cpp
│   │   ├── test_onnx_parser.cpp
│   │   ├── test_linear.cpp
│   │   ├── test_conv2d.cpp
│   │   ├── test_activations.cpp
│   │   ├── test_executor.cpp
│   │   └── CMakeLists.txt
│   ├── benchmarks/                  # Google Benchmark
│   │   ├── bench_matmul.cpp
│   │   ├── bench_conv2d.cpp
│   │   └── CMakeLists.txt
│   ├── third_party/                 # Git submodules
│   │   ├── eigen/                   # Matrix math (header-only)
│   │   ├── protobuf/                # ONNX file parsing
│   │   ├── googletest/              # Testing
│   │   ├── google-benchmark/        # Benchmarking
│   │   └── pybind11/                # Python bindings
│   └── CMakeLists.txt               # Root CMake file
│
├── cli/                             # Rust CLI tool
│   ├── src/
│   │   ├── main.rs                  # Entry: crucible run --model x.onnx --input y.json
│   │   ├── runner.rs                # Calls C++ engine via FFI
│   │   └── formatter.rs             # Output formatting
│   ├── Cargo.toml
│   └── Cargo.lock
│
├── wasm/                            # Rust → WebAssembly
│   ├── src/
│   │   └── lib.rs                   # wasm-pack entrypoint
│   ├── Cargo.toml
│   └── pkg/                         # wasm-pack output (gitignored)
│
├── server/                          # Python FastAPI model server
│   ├── app/
│   │   ├── main.py                  # FastAPI routes
│   │   ├── converter.py             # PyTorch/Keras → ONNX conversion
│   │   ├── validator.py             # Validate ONNX model operators
│   │   └── schemas.py               # Pydantic request/response models
│   ├── tests/
│   │   └── test_converter.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── web/                             # Next.js 15 demo dashboard
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx             # Landing: upload model → run inference
│   │   │   ├── benchmark/page.tsx   # Side-by-side latency comparison chart
│   │   │   ├── playground/page.tsx  # WASM inference live demo
│   │   │   └── docs/page.tsx        # Supported operators list
│   │   ├── components/
│   │   │   ├── ModelUploader.tsx
│   │   │   ├── InferenceRunner.tsx  # Calls WASM module
│   │   │   ├── LatencyChart.tsx     # Recharts
│   │   │   └── TensorViewer.tsx     # Shows input/output tensors
│   │   └── lib/
│   │       ├── crucible-wasm.ts     # TypeScript WASM loader
│   │       └── api.ts               # FastAPI client
│   ├── public/
│   │   └── wasm/                    # Built WASM files (from wasm/ pkg/)
│   ├── Dockerfile
│   └── package.json
│
├── benchmarks/                      # Python benchmark scripts
│   ├── bench_crucible.py            # Benchmark via Python bindings
│   ├── bench_onnxruntime.py         # Same model via ONNX Runtime
│   ├── bench_pytorch.py             # Same model via PyTorch
│   └── results/                     # Saved benchmark JSON output
│
├── models/                          # Sample ONNX models for testing
│   ├── download_models.py           # Downloads MobileNetV2, ResNet18 etc
│   └── .gitkeep
│
├── infra/
│   ├── docker-compose.yml
│   └── nginx/
│       └── nginx.conf
│
├── .github/
│   └── workflows/
│       ├── ci-engine.yml            # CMake build + tests
│       ├── ci-rust.yml              # Cargo test
│       ├── ci-server.yml            # pytest
│       └── ci-web.yml               # Next.js type-check + lint
│
├── scripts/
│   ├── setup.sh                     # Install all dependencies
│   ├── build-engine.sh              # CMake configure + build
│   └── build-wasm.sh                # wasm-pack build
│
├── CMakePresets.json                # Build presets (Debug, Release)
├── .env.example
├── .gitmodules                      # Third-party submodules
├── .gitignore
├── CONTEXT.md
├── ENGINEERING_PLAN.md
└── README.md
```

---

## 2. SERVICES AND RESPONSIBILITIES

### `engine/` — C++17 Core (the hard part)
**Owns:** Everything about running the model.
- Tensor class: multi-dimensional float32, owns its memory (no shared_ptr overhead)
- ONNX parser: reads `.onnx` protobuf file, builds graph representation
- Operator implementations: Linear, Conv2D, ReLU, Sigmoid, Softmax, BatchNorm, MaxPool
- Graph executor: topological sort of nodes, run in dependency order
- Does NOT: handle HTTP, talk to filesystem beyond model load, manage Python objects

### `engine/bindings/python/` — pybind11 Layer
**Owns:** Python ↔ C++ bridge. Exposes engine as Python module `crucible_py`.
- `crucible_py.load_model(path: str) → Model`
- `crucible_py.run(model, input: np.ndarray) → np.ndarray`
- Does NOT: implement any logic — pure bridge

### `cli/` — Rust CLI
**Owns:** Command-line interface for running models.
- `crucible run --model mobilenet.onnx --input image.json --output result.json`
- `crucible benchmark --model mobilenet.onnx --runs 100`
- `crucible validate --model mobilenet.onnx` (checks all ops are supported)
- Calls C++ engine via FFI (foreign function interface)

### `wasm/` — Rust → WebAssembly
**Owns:** Browser-executable version.
- Compiles to `.wasm` file loadable in any browser
- Exposes: `runInference(modelBytes: Uint8Array, inputData: Float32Array) → Float32Array`
- Does NOT: make network calls — pure computation in browser sandbox

### `server/` — Python FastAPI
**Owns:** Model conversion and HTTP serving.
- Accept PyTorch `.pt` or `.pth` files → convert to `.onnx`
- Validate ONNX model (all ops supported by Crucible?)
- Serve inference via HTTP (uses Python bindings to call C++ engine)
- Does NOT: implement inference logic — delegates to `crucible_py`

### `web/` — Next.js Dashboard
**Owns:** Demo UI.
- Upload ONNX model → run in browser via WASM
- Side-by-side latency comparison chart
- Supported operators documentation page
- Does NOT: make direct calls to C++ — uses WASM or FastAPI

---

## 3. TECH STACK — FINAL DECISIONS

| Component | Choice | Version | Reason |
|-----------|--------|---------|--------|
| C++ standard | C++17 | — | `std::optional`, `std::string_view`, structured bindings — modern without requiring C++20 |
| C++ compiler | MSVC (Windows) / GCC 13 (Linux) | — | MSVC for Windows dev, GCC for CI/deployment |
| Build system | CMake | 3.27+ | Industry standard. Every C++ interview will test CMake knowledge |
| Matrix math | Eigen | 3.4 | Header-only. Used by TensorFlow, Google. No linking complexity |
| ONNX parsing | protobuf | 3.21 | ONNX files ARE protobuf. Official parser |
| Unit testing | GoogleTest | 1.14 | Industry standard C++ testing |
| Benchmarking | Google Benchmark | 1.8 | CPU timing, statistical analysis |
| Python bindings | pybind11 | 2.12 | Standard for C++ → Python. Used by PyTorch itself |
| Python version | 3.11 | — | Matches CodeLens |
| Python framework | FastAPI | 0.115 | Matches CodeLens |
| Rust edition | 2021 | — | Current stable |
| WASM toolchain | wasm-pack | 0.12 | Standard Rust → WASM workflow |
| Frontend | Next.js 15 | App Router | Matches CodeLens |
| Frontend UI | Tailwind + shadcn/ui | — | Matches CodeLens |
| Charts | Recharts | 2.x | Latency comparison charts |
| Containerization | Docker + Compose | — | Server + web only (C++ built natively) |

**NOT using:**
- CUDA/GPU (adds massive complexity, CPU demo is cleaner for interview)
- OpenMP (adds to C++17 in v2 if time permits)
- Bazel (CMake is enough, Bazel is overkill)
- gRPC (REST is sufficient, gRPC adds proto complexity)

---

## 4. C++ ENGINE API DESIGN

### Tensor Class (`tensor.hpp`)
```cpp
namespace crucible {

class Tensor {
public:
    // Constructors
    Tensor(std::vector<int64_t> shape, float fill = 0.0f);
    Tensor(std::vector<int64_t> shape, std::vector<float> data);

    // Access
    float& at(std::vector<int64_t> indices);
    const float& at(std::vector<int64_t> indices) const;
    float* data();
    const float* data() const;

    // Shape
    std::vector<int64_t> shape() const;
    int64_t size() const;          // total element count
    int64_t rank() const;          // number of dimensions

    // Utilities
    Tensor reshape(std::vector<int64_t> new_shape) const;
    Tensor flatten() const;
    void print(int max_elements = 10) const;

private:
    std::vector<int64_t> shape_;
    std::vector<float> data_;
    int64_t compute_offset(std::vector<int64_t> indices) const;
};

} // namespace crucible
```

### Operator Interface (`ops/base.hpp`)
```cpp
namespace crucible {

class Operator {
public:
    virtual ~Operator() = default;
    virtual std::string name() const = 0;
    virtual Tensor forward(
        const std::vector<Tensor>& inputs,
        const std::unordered_map<std::string, float>& attrs
    ) const = 0;
};

} // namespace crucible
```

### Model and Executor (`model.hpp`, `executor.hpp`)
```cpp
namespace crucible {

struct GraphNode {
    std::string op_type;          // "MatMul", "Relu", "Conv", etc.
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::unordered_map<std::string, float> attributes;
};

struct Model {
    std::vector<GraphNode> nodes;
    std::unordered_map<std::string, Tensor> weights;  // initializers
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
};

Model load_model(const std::string& path);

Tensor run_inference(
    const Model& model,
    const Tensor& input,
    const std::string& input_name = ""
);

} // namespace crucible
```

---

## 5. SUPPORTED ONNX OPERATORS (MVP)

Build exactly these. Enough to run MobileNetV2 (the target model).

| Operator | ONNX op_type | What it does |
|----------|-------------|--------------|
| Linear/Dense | `Gemm`, `MatMul` | Matrix multiply + optional bias |
| 2D Convolution | `Conv` | Image feature extraction |
| ReLU | `Relu` | max(0, x) |
| Sigmoid | `Sigmoid` | 1/(1+e^-x) |
| Softmax | `Softmax` | Probability distribution |
| GELU | `Gelu` | x * Φ(x) — used in BERT |
| MaxPool | `MaxPool` | Take max in sliding window |
| AvgPool | `AveragePool` | Take average in sliding window |
| BatchNorm | `BatchNormalization` | Normalize activations |
| Flatten | `Flatten` | Collapse spatial dims to 1D |
| Reshape | `Reshape` | Change tensor shape |
| Add | `Add` | Element-wise add (residual connections) |
| Concat | `Concat` | Concatenate tensors along axis |

Total: 13 operators. MobileNetV2 uses all of them.

---

## 6. PYTHON BINDINGS CONTRACT (pybind11)

Exposed as Python module `crucible_py`. Install via `pip install -e .` after CMake build.

```python
import crucible_py
import numpy as np

# Load model
model = crucible_py.load_model("mobilenet_v2.onnx")

# Run inference
input_array = np.random.randn(1, 3, 224, 224).astype(np.float32)
output = crucible_py.run(model, input_array)  # Returns np.ndarray

# Model info
print(crucible_py.get_model_info(model))
# → {"operators": ["Conv", "Relu", ...], "input_shape": [1,3,224,224], "output_shape": [1,1000]}
```

---

## 7. FASTAPI SERVER CONTRACT

Base URL: `http://localhost:8000`

### `POST /convert`
Convert PyTorch model to ONNX.
```json
// Request (multipart form)
model_file: <.pt or .pth file>
input_shape: [1, 3, 224, 224]

// Response 200
{
  "onnx_model_id": "uuid",
  "model_path": "/tmp/models/uuid.onnx",
  "operators_used": ["Conv", "Relu", "BatchNormalization"],
  "all_supported": true,
  "unsupported_ops": []
}
```

### `POST /infer`
Run inference on uploaded ONNX model.
```json
// Request
{
  "model_id": "uuid",
  "input": [[1.0, 2.0, 3.0, ...]],   // flattened float array
  "input_shape": [1, 3, 224, 224]
}

// Response 200
{
  "output": [0.001, 0.023, ...],       // flattened float array
  "output_shape": [1, 1000],
  "inference_time_ms": 14.3,
  "engine": "crucible-cpp"
}
```

### `POST /validate`
Check if all ONNX operators are supported.
```json
// Response 200
{
  "valid": true,
  "operators": ["Conv", "Relu"],
  "unsupported": []
}
```

### `GET /operators`
List all supported operators.
```json
// Response 200
{
  "supported": ["Gemm", "MatMul", "Conv", "Relu", "Sigmoid", ...],
  "count": 13
}
```

### `GET /health`
```json
{ "status": "ok", "engine": "crucible-cpp", "version": "1.0.0" }
```

---

## 8. RUST CLI INTERFACE

```bash
# Run inference
crucible run \
  --model mobilenet_v2.onnx \
  --input image.json \
  --output result.json

# Benchmark
crucible benchmark \
  --model mobilenet_v2.onnx \
  --runs 100 \
  --warmup 10

# Validate operators
crucible validate --model mobilenet_v2.onnx

# Model info
crucible info --model mobilenet_v2.onnx
```

Output format for benchmark:
```
Crucible Benchmark — mobilenet_v2.onnx
─────────────────────────────────────
Runs:        100
Warmup:      10
Mean:        14.3ms
Median:      13.9ms
P95:         18.2ms
P99:         22.1ms
Min:         12.8ms
Max:         31.4ms
Throughput:  69.9 inferences/sec
```

---

## 9. WASM INTERFACE (TypeScript usage)

```typescript
// In Next.js component
import init, { runInference } from '../public/wasm/crucible_wasm';

const result = await init();

const modelBytes = new Uint8Array(await modelFile.arrayBuffer());
const inputData = new Float32Array(224 * 224 * 3);  // dummy input

const output = runInference(modelBytes, inputData, [1, 3, 224, 224]);
// Returns Float32Array of shape [1, 1000]
```

---

## 10. BUILD SYSTEM (CMake)

### `engine/CMakeLists.txt` (key parts)
```cmake
cmake_minimum_required(VERSION 3.27)
project(crucible VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Third-party via git submodules
add_subdirectory(third_party/eigen)
add_subdirectory(third_party/protobuf)
add_subdirectory(third_party/googletest)
add_subdirectory(third_party/google-benchmark)
add_subdirectory(third_party/pybind11)

# Core library (used by tests, benchmarks, python bindings)
add_library(crucible_core STATIC
    src/tensor.cpp
    src/model.cpp
    src/executor.cpp
    src/onnx_parser.cpp
    src/ops/linear.cpp
    src/ops/conv2d.cpp
    src/ops/activations.cpp
    src/ops/pooling.cpp
    src/ops/norm.cpp
)

target_include_directories(crucible_core PUBLIC include)
target_link_libraries(crucible_core PUBLIC Eigen3::Eigen protobuf::libprotobuf)

# Enable optimizations in Release mode
target_compile_options(crucible_core PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native>
)

# Python bindings
add_subdirectory(bindings/python)

# Tests
enable_testing()
add_subdirectory(tests)

# Benchmarks
add_subdirectory(benchmarks)
```

### `CMakePresets.json`
```json
{
  "version": 3,
  "configurePresets": [
    {
      "name": "debug",
      "displayName": "Debug",
      "generator": "Ninja",
      "binaryDir": "${sourceDir}/build/debug",
      "cacheVariables": { "CMAKE_BUILD_TYPE": "Debug" }
    },
    {
      "name": "release",
      "displayName": "Release",
      "generator": "Ninja",
      "binaryDir": "${sourceDir}/build/release",
      "cacheVariables": { "CMAKE_BUILD_TYPE": "Release" }
    }
  ]
}
```

---

## 11. LOCAL DEVELOPMENT SETUP

### Prerequisites to install before first run

**Windows:**
```powershell
# 1. Visual Studio Build Tools 2022 (C++ compiler)
# Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
# Select: "Desktop development with C++" workload

# 2. CMake
winget install Kitware.CMake

# 3. Ninja (fast build tool)
winget install Ninja-build.Ninja

# 4. Git (for submodules)
winget install Git.Git

# 5. Rust
# Go to rustup.rs and download installer

# 6. wasm-pack
cargo install wasm-pack

# 7. Python 3.11 (you have it)
# 8. Node.js 20+ (you have it)
```

**Verify all installed:**
```powershell
cmake --version      # Should be 3.27+
ninja --version      # Should be 1.11+
cl                   # Should show MSVC version (run in Developer Command Prompt)
rustc --version      # Should be 1.78+
wasm-pack --version  # Should be 0.12+
python --version     # Should be 3.11+
node --version       # Should be 20+
```

### First-time project setup
```powershell
# 1. Clone (after creating on GitHub)
git clone https://github.com/tanmay-alpha/crucible.git
cd crucible

# 2. Initialize git submodules (downloads Eigen, protobuf, GoogleTest etc)
git submodule update --init --recursive
# NOTE: This downloads ~500MB. Takes 5-10 minutes. Run once only.

# 3. Configure CMake (Debug mode for development)
cmake --preset debug -S engine

# 4. Build C++ engine
cmake --build engine/build/debug

# 5. Run C++ tests
cd engine/build/debug && ctest --output-on-failure

# 6. Install Python server deps
cd server && pip install -r requirements.txt

# 7. Install Next.js deps
cd web && npm install

# 8. Build WASM (after Rust CLI is done)
cd wasm && wasm-pack build --target web --out-dir ../web/public/wasm
```

### `.env.example`
```bash
# FastAPI Server
CRUCIBLE_MODEL_DIR=/tmp/crucible-models
CRUCIBLE_ENGINE_PATH=./engine/build/release
MAX_MODEL_SIZE_MB=100

# Next.js
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WASM_PATH=/wasm/crucible_wasm.js

# Python bindings path (set after CMake build)
PYTHONPATH=./engine/build/debug
```

---

## 12. MCP SERVERS — HOW TO USE IN THIS PROJECT

Add all 6 to your Claude Code MCP config before starting.

| MCP Server | What to use it for in Crucible |
|-----------|-------------------------------|
| **filesystem** | Read/write C++ header files, CMakeLists, build output logs |
| **github** | Create GitHub issues #1-#20, open PRs, post commit comments automatically |
| **context7** | Fetch Eigen docs for matrix ops, pybind11 API reference, ONNX operator spec, wasm-pack docs |
| **sequential-thinking** | Plan complex C++ algorithms (Conv2D implementation, topological sort) BEFORE writing code |
| **fetch** | Fetch ONNX operator specs from https://onnx.ai/onnx/operators/, fetch Eigen docs |
| **puppeteer** | Automated testing of WASM demo in browser — upload model, verify output renders |

**How to instruct Claude Code to use them:**

Every prompt to Claude Code should start with:
```
Use context7 MCP to look up [specific API/doc] before writing any code.
Use sequential-thinking MCP to plan the algorithm before implementing.
Use github MCP to commit after each file, not git commands in terminal.
Use fetch MCP to get the ONNX operator spec for [Op] from https://onnx.ai/onnx/operators/
```

---

## 13. TESTING PLAN

### C++ Tests (GoogleTest) — `engine/tests/`

**test_tensor.cpp**
```cpp
TEST(TensorTest, ConstructorSetsShape) {
    Tensor t({2, 3});
    EXPECT_EQ(t.shape(), (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(t.size(), 6);
}

TEST(TensorTest, AtReturnsCorrectElement) {
    Tensor t({2, 3}, {1,2,3,4,5,6});
    EXPECT_FLOAT_EQ(t.at({1, 2}), 6.0f);
}

TEST(TensorTest, ReshapePreservesData) {
    Tensor t({2, 3}, {1,2,3,4,5,6});
    auto reshaped = t.reshape({3, 2});
    EXPECT_EQ(reshaped.shape(), (std::vector<int64_t>{3, 2}));
    EXPECT_FLOAT_EQ(reshaped.at({0, 0}), 1.0f);
}
```

**test_activations.cpp**
```cpp
TEST(ActivationsTest, ReluZerosNegatives) {
    Tensor input({4}, {-2.0f, -1.0f, 0.0f, 3.0f});
    auto output = relu_forward(input, {});
    EXPECT_FLOAT_EQ(output.at({0}), 0.0f);
    EXPECT_FLOAT_EQ(output.at({3}), 3.0f);
}

TEST(ActivationsTest, SoftmaxSumsToOne) {
    Tensor input({3}, {1.0f, 2.0f, 3.0f});
    auto output = softmax_forward(input, {});
    float sum = 0;
    for (int i = 0; i < 3; i++) sum += output.at({i});
    EXPECT_NEAR(sum, 1.0f, 1e-6f);
}
```

**test_executor.cpp** — Most important
```cpp
TEST(ExecutorTest, RunsMobileNetV2Shape) {
    // Requires mobilenet_v2.onnx in models/
    auto model = load_model("../../models/mobilenet_v2.onnx");
    Tensor input({1, 3, 224, 224});  // batch=1, RGB, 224x224
    auto output = run_inference(model, input);
    EXPECT_EQ(output.shape(), (std::vector<int64_t>{1, 1000}));
}
```

### Python Tests (pytest) — `server/tests/`
- test_converter.py: PyTorch resnet18 → ONNX → validate
- test_api.py: /convert, /infer, /validate endpoints

### Rust Tests — `cli/src/`
- Inline tests in each module (Rust convention)
- test_benchmark_output_format()
- test_validate_supported_model()

### WASM Tests — puppeteer MCP
- Launch browser via puppeteer
- Load web demo page
- Upload test ONNX model
- Verify output tensor rendered on page

---

## 14. CI/CD PLAN

### `.github/workflows/ci-engine.yml`
Triggers: push/PR touching `engine/**`
```yaml
jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive      # CRITICAL - downloads Eigen etc
      - name: Configure CMake
        run: cmake --preset debug -S engine
      - name: Build
        run: cmake --build engine/build/debug
      - name: Test
        run: cd engine/build/debug && ctest --output-on-failure
```

### `.github/workflows/ci-rust.yml`
```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cd cli && cargo test
      - run: cd cli && cargo clippy -- -D warnings
```

### `.github/workflows/ci-server.yml`
```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: cd server && pip install -r requirements.txt
      - run: cd server && pytest tests/ -v
```

---

## 15. DEPLOYMENT ARCHITECTURE

```
┌──────────────────────────────────────────────────────────────┐
│                        USER                                  │
│   Browser (WASM)           CLI              Python app       │
│   Next.js + crucible.wasm  crucible run     import crucible  │
└────────┬─────────────────────┬────────────────┬─────────────┘
         │                     │                │
         ▼                     ▼                ▼
┌─────────────────┐  ┌──────────────┐  ┌──────────────────────┐
│  VERCEL         │  │ Rust binary  │  │  Python bindings     │
│  Next.js        │  │ (local CLI)  │  │  crucible_py.so      │
│  Serves WASM    │  │ calls C++ FFI│  │  pybind11 module     │
└─────────────────┘  └──────────────┘  └──────────────────────┘
                                                │
                                                ▼
                              ┌────────────────────────────────┐
                              │  RAILWAY                       │
                              │  FastAPI server (Python)       │
                              │  Loads crucible_py bindings    │
                              │  Handles /convert /infer /validate│
                              └────────────────────────────────┘

Core C++ engine: compiled natively on each platform.
No server needed for WASM path — runs entirely in browser.
```

---

## 16. FIRST 20 GITHUB ISSUES

### Milestone 1: C++ Foundation (Week 1–2)

**Issue #1: CMake project scaffold + git submodules**
> Set up `engine/CMakeLists.txt`. Add git submodules: Eigen, protobuf, googletest, google-benchmark, pybind11. Run `cmake --preset debug` successfully. Empty library compiles.
> AC: `cmake --build engine/build/debug` succeeds with zero errors.

**Issue #2: Tensor class — shape, data, indexing**
> Implement `tensor.hpp` and `tensor.cpp`. Constructor, `at()`, `data()`, `shape()`, `size()`, `rank()`.
> AC: 8 GoogleTest tests pass. Tensor({2,3}).at({1,2}) returns correct element.

**Issue #3: Tensor operations — reshape, flatten, print**
> Add `reshape()`, `flatten()`, `print()` to Tensor.
> AC: 4 more tests pass. `reshape({6})` of 2×3 tensor has same elements. `flatten()` is shape {N}.

**Issue #4: ONNX protobuf parser — read .onnx file into Model struct**
> Implement `onnx_parser.hpp` and `onnx_parser.cpp`. Read `.onnx` file via libprotobuf. Extract: graph nodes (op_type, inputs, outputs), initializer weights as Tensors.
> AC: Load `mobilenet_v2.onnx`. Print node count and operator list. No crash.

**Issue #5: Linear operator — Gemm and MatMul**
> Implement `ops/linear.cpp`. Support both `Gemm` (A*B+C) and `MatMul` (A*B). Use Eigen for matrix multiply.
> AC: `MatMul` of (3×4) × (4×5) = (3×5). Numerically matches numpy result to 1e-5.

**Issue #6: Activation functions — ReLU, Sigmoid, Softmax, GELU**
> Implement all 4 in `ops/activations.cpp`.
> AC: ReLU zeros negatives. Softmax sums to 1.0 ± 1e-6. Sigmoid(0) = 0.5. GELU matches formula.

**Issue #7: Conv2D operator**
> Implement 2D convolution in `ops/conv2d.cpp`. Support: padding, stride, groups=1.
> AC: Conv2D on (1,3,224,224) input with (32,3,3,3) kernel = (1,32,222,222). Matches PyTorch output to 1e-4.

**Issue #8: Pooling + normalization operators**
> Implement MaxPool, AvgPool in `ops/pooling.cpp`. BatchNorm in `ops/norm.cpp`.
> AC: MaxPool(2,2) on (1,32,222,222) = (1,32,111,111). BatchNorm output variance ≈ 1.0 (normalized).

### Milestone 2: Graph Executor (Week 3)

**Issue #9: Graph executor — topological sort + run nodes in order**
> Implement `executor.cpp`. Topological sort of graph nodes. Run each operator in dependency order. Manage intermediate tensor storage.
> AC: Can execute a 3-node graph (MatMul → Relu → Softmax) correctly.

**Issue #10: End-to-end inference — MobileNetV2 runs correctly**
> `run_inference(model, input_tensor)` on real `mobilenet_v2.onnx` with shape (1,3,224,224) → output shape (1,1000).
> AC: Output shape correct. Top-1 class index matches ONNX Runtime output on same input (numerical correctness).

**Issue #11: Google Benchmark setup + Release build optimization**
> Add `engine/benchmarks/bench_matmul.cpp`. Benchmark MatMul at sizes 64×64, 256×256, 1024×1024. Enable -O3 -march=native in Release preset.
> AC: Benchmark runs. Release build ≥ 3× faster than Debug for 1024×1024 MatMul.

### Milestone 3: Python + Server (Week 4)

**Issue #12: pybind11 Python bindings**
> Expose `load_model()` and `run_inference()` as Python module `crucible_py`. Handle numpy ↔ Tensor conversion.
> AC: `import crucible_py; model = crucible_py.load_model("mobilenet_v2.onnx"); out = crucible_py.run(model, np.zeros((1,3,224,224), dtype=np.float32))` works.

**Issue #13: FastAPI server with /convert /infer /validate**
> Full server using `crucible_py` for inference. PyTorch → ONNX conversion in `converter.py`.
> AC: POST to /infer with model_id and input returns inference result matching C++ output.

**Issue #14: Python benchmark scripts — vs ONNX Runtime vs PyTorch**
> `benchmarks/bench_crucible.py`, `bench_onnxruntime.py`, `bench_pytorch.py`. All benchmark same model, same input, 100 runs.
> AC: `results/benchmark_results.json` with mean/median/p95 for all three engines. Crucible within 3× of ONNX Runtime.

### Milestone 4: Rust + WASM (Week 5–6)

**Issue #15: Rust CLI — run, benchmark, validate, info commands**
> Full CLI in `cli/src/main.rs`. Calls C++ engine via FFI using `extern "C"` bridge.
> AC: `crucible run --model mobilenet_v2.onnx --input input.json` prints top-5 predictions.

**Issue #16: Rust → WebAssembly build via wasm-pack**
> `wasm/src/lib.rs` exposes `runInference` function. `wasm-pack build --target web` produces pkg/.
> AC: `.wasm` file generated. `web/public/wasm/` contains `crucible_wasm.js` and `crucible_wasm_bg.wasm`.

### Milestone 5: Web Demo + Polish (Week 7–8)

**Issue #17: Next.js WASM demo — upload model + run in browser**
> Model upload UI → WASM inference → display output tensor values and top-5 predictions.
> AC: Upload `mobilenet_v2.onnx` in browser. Inference runs client-side. Output renders within 5 seconds.

**Issue #18: Benchmark comparison page**
> Recharts line chart: Crucible vs ONNX Runtime vs PyTorch latency across model sizes.
> AC: Chart renders with real data from `benchmark_results.json`.

**Issue #19: CI/CD pipelines — CMake + Cargo + pytest**
> All 4 GitHub Actions workflows passing on main branch.
> AC: All green checks on GitHub.

**Issue #20: README + demo recording + arXiv-style writeup**
> README with: architecture diagram, benchmark table, install instructions, demo GIF.
> Write `WRITEUP.md` (4-page research note on design decisions: memory layout, operator fusion opportunities, comparison methodology).
> AC: README renders correctly on GitHub. Benchmark table shows Crucible vs ONNX Runtime vs PyTorch.

---

## 17. IMPLEMENTATION ORDER (Day-by-day)

```
Day 1–2:   Issue #1 — CMake + submodules. Get build system working. Most painful day.
Day 3–4:   Issue #2 + #3 — Tensor class. First real C++ you write.
Day 5–6:   Issue #4 — ONNX parser. Read a real .onnx file.
Day 7–8:   Issue #5 + #6 — Linear + activations. First operators.
Day 9–10:  Issue #7 — Conv2D. Hardest operator. Take time, get right.
Day 11:    Issue #8 — Pooling + BatchNorm.
Day 12–13: Issue #9 — Graph executor. Topological sort.
Day 14:    Issue #10 — *** DEMO 1: MobileNetV2 runs end-to-end ***
Day 15–16: Issue #11 — Benchmarks. Celebrate speed.
Day 17–18: Issue #12 — Python bindings. Bridge to Python world.
Day 19–20: Issue #13 — FastAPI server.
Day 21:    Issue #14 — Benchmark scripts.
Day 22–23: Issue #15 — Rust CLI.
Day 24–25: Issue #16 — WASM build.
Day 26–27: Issue #17 — Web demo. *** DEMO 2: Inference in browser ***
Day 28:    Issue #18 — Benchmark chart.
Day 29:    Issue #19 — CI/CD.
Day 30:    Issue #20 — README + writeup.

Buffer: 2 days for C++ debugging (always needed).
```

**Two target demos:**
1. **Day 14:** `./engine/build/release/crucible_test` runs MobileNetV2. First working inference.
2. **Day 27:** Open browser. Upload mobilenet_v2.onnx. See "Golden Retriever: 82.3%" in browser. No Python running.

---

## 18. WHAT THIS TEACHES YOU (interview-ready knowledge)

| Topic | Where learned in Crucible |
|-------|--------------------------|
| C++ memory management | Tensor class owns its data with `std::vector<float>` |
| Template metaprogramming | Operator interface design |
| CMake build system | Every C++ job requires this |
| Matrix math | Eigen usage in MatMul/Conv2D |
| Neural network internals | Implementing Conv2D teaches what PyTorch's Conv2D actually does |
| Topological sort | Graph executor |
| Foreign function interface | Rust CLI calling C++ |
| WebAssembly | Rust → WASM compilation |
| pybind11 | C++ ↔ Python bridge |
| Benchmarking methodology | Google Benchmark + statistical analysis |

---

## 19. INTERVIEW ANSWERS THIS PROJECT GIVES YOU

**"What's the most technically complex project you've built?"**
> "I built Crucible — an ONNX inference engine in C++17 from scratch. It parses the ONNX protobuf format, implements 13 neural network operators using Eigen for matrix math, and executes a graph via topological sort. MobileNetV2 runs in 14ms. Same job as TensorFlow Lite — I built it to understand how inference engines actually work."

**"Tell me about your C++ experience."**
> "I implemented a multi-dimensional tensor class with manual memory layout, wrote a Conv2D operator using Eigen's block operations and im2col approach, and built a CMake project with git submodules for Eigen, protobuf, GoogleTest. I also wrote pybind11 bindings that convert between numpy arrays and my Tensor class."

**"How does Conv2D actually work?"**
> "Standard approach is im2col — you reshape the input patches into a matrix and do a single large matrix multiply with the filter weights. That's what my implementation does. Crucible's Conv2D took me 3 days and 47 test iterations to get numerically correct to 1e-4 against PyTorch."

---

*This document is source of truth for all Crucible implementation decisions. Any deviation needs a note explaining why.*
