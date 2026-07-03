# Crucible — User Guide
*How to use, run, and understand this project in the real world*

---

## What is Crucible?

Crucible is a **neural network inference engine** — it takes trained AI models in the ONNX format and runs them to produce predictions, without needing PyTorch or TensorFlow installed at runtime.

**Real-world analogy:** If PyTorch is a full car factory (training + deployment), Crucible is a purpose-built racing engine — it only does one thing (run the model) but does it with no unnecessary weight.

**Who this is for:**
- Developers who want to understand how inference engines work under the hood
- CS students learning about systems programming, CMake, Rust FFI, and WebAssembly
- Anyone who wants a reference implementation of ONNX inference in C++17

---

## Quick Start — 5 Minutes

### Requirement Check
```bash
cmake --version    # needs 3.27+
g++ --version      # needs GCC 13+ on Linux; MSVC 2022 on Windows
cargo --version    # needs Rust 1.78+
python --version   # needs Python 3.11+
node --version     # needs Node.js 20+ (for web demo only)
```

### 1. Clone and Pull All Submodules
```bash
git clone https://github.com/tanmay-alpha/Crucible.git
cd Crucible

# IMPORTANT: This downloads Eigen, protobuf, googletest, pybind11 (~500MB)
git submodule update --init --recursive
```

### 2. Build the C++ Engine
```bash
# Create build directory, configure, and build
cmake -S engine -B engine/build/debug -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCRUCIBLE_ENABLE_TESTS=ON
cmake --build engine/build/debug --parallel

# Run the unit test suite (should show all green)
ctest --test-dir engine/build/debug --output-on-failure
```

### 3. Try the Web Demo (No C++ Build Required)
```bash
cd web
npm install
npm run dev
# Open http://localhost:3000 in your browser
```

The web demo uses a **pure-Rust WebAssembly** module — no C++ needed in the browser.

---

## Paths for Different Goals

### Goal A: "I want to run neural network inference from the command line"

**Prerequisites:** C++ engine must be built (Step 2 above), plus the shared library:

```bash
# Build the shared library (libcrucible.so / crucible.dll)
cmake -S engine -B engine/build/release \
  -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCRUCIBLE_BUILD_SHARED=ON
cmake --build engine/build/release --parallel

# Build the Rust CLI
cd cli
# Set the library path (Linux/Mac)
export CRUCIBLE_LIB_DIR=$(pwd)/../engine/build/release
export LD_LIBRARY_PATH=$CRUCIBLE_LIB_DIR:$LD_LIBRARY_PATH
cargo build --release
```

**Usage:**
```bash
# Run inference on an ONNX model
./target/release/crucible run \
  --model ../models/mobilenet_v2.onnx \
  --input input.json

# Benchmark a model (100 runs)
./target/release/crucible benchmark \
  --model ../models/mobilenet_v2.onnx \
  --runs 100

# Check if all operators in a model are supported
./target/release/crucible validate \
  --model ../models/mobilenet_v2.onnx

# Show model info (input/output names, node count)
./target/release/crucible info \
  --model ../models/mobilenet_v2.onnx
```

**Input format** (`input.json`):
```json
{
  "shape": [1, 3, 224, 224],
  "data": [0.1, 0.2, 0.3, ...]
}
```

### Goal B: "I want to use Crucible from Python"

**Prerequisites:** C++ engine built (see Step 2), plus Python bindings:

```bash
# Build with Python bindings enabled
cmake -S engine -B engine/build/release \
  -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCRUCIBLE_ENABLE_PYTHON_BINDINGS=ON
cmake --build engine/build/release --parallel

# The .pyd/.so file will be in engine/build/release/
```

**Usage:**
```python
import sys
sys.path.insert(0, 'engine/build/release')  # path to crucible_py.so

import crucible_py
import numpy as np

# Load model
model = crucible_py.load_model("models/mobilenet_v2.onnx")

# Prepare input (batch=1, RGB, 224×224, ImageNet-normalized)
input_tensor = np.zeros((1, 3, 224, 224), dtype=np.float32)

# Run inference
output = crucible_py.run(model, input_tensor)
print(f"Output shape: {output.shape}")  # (1, 1000)
print(f"Top-1 class: {output.argmax()}")
```

### Goal C: "I want to run inference in the browser (no server)"

The web demo runs entirely client-side using WebAssembly. No Python or C++ needed.

```bash
# Option 1: Use the pre-built web demo
cd web && npm install && npm run dev
# Go to http://localhost:3000/playground

# Option 2: Build the WASM module yourself (requires wasm-pack)
cargo install wasm-pack
bash scripts/build-wasm.sh
# WASM artifacts appear in web/public/wasm/
```

**What the playground does:**
1. You drop a `.onnx` model file onto the upload area
2. The WASM module (running in your browser tab, no network calls) parses and runs it
3. Results appear: output tensor shape, top-5 class probabilities, timing

**Limitation:** The browser WASM module supports 4 operators (MatMul, ReLU, Softmax, Sigmoid). For the full 13-operator set, use the C++ engine locally.

### Goal D: "I want an HTTP API for model inference"

```bash
# Install Python server dependencies
cd server
pip install -r requirements.txt

# Start the FastAPI server
uvicorn app.main:app --reload --port 8000

# The API is now live at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

**API Endpoints:**

| Method | Endpoint | What it does |
|--------|----------|-------------|
| `GET` | `/health` | Server health check |
| `GET` | `/operators` | List all 13 supported operators |
| `POST` | `/validate` | Check if a model's ops are all supported |
| `POST` | `/convert` | Convert PyTorch .pt model → .onnx |
| `POST` | `/infer` | Run inference on a model + input |

**Example: Run inference via HTTP**
```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "mobilenet_v2",
    "input_data": [[[[0.1, 0.2, ...]]]]
  }'
```

**Example: Convert PyTorch model to ONNX**
```bash
curl -X POST http://localhost:8000/convert \
  -F "file=@my_model.pt" \
  -F "model_name=my_model"
```

### Goal E: "I want to benchmark Crucible vs ONNX Runtime vs PyTorch"

```bash
# Install benchmark dependencies
pip install onnxruntime torch numpy

# Download a sample model
python models/download_models.py

# Run all three benchmarks
python benchmarks/bench_crucible.py      # uses Python bindings
python benchmarks/bench_onnxruntime.py   # uses onnxruntime
python benchmarks/bench_pytorch.py       # uses PyTorch

# Results are saved to:
cat benchmarks/results/benchmark_results.json
```

---

## Understanding the Architecture

```
Your Data (image, text, etc.)
         │
         ▼
   ┌──────────────────────────────────────────────────┐
   │  Step 1: Load ONNX Model                        │
   │  parse_model("mobilenet_v2.onnx")               │
   │  → Reads protobuf format                        │
   │  → Extracts: nodes, weights, input/output names │
   └──────────────────┬───────────────────────────────┘
                      │
                      ▼
   ┌──────────────────────────────────────────────────┐
   │  Step 2: Topological Sort (Kahn's BFS)          │
   │  topological_sort(model.graph.nodes)             │
   │  → Orders operators so dependencies run first   │
   └──────────────────┬───────────────────────────────┘
                      │
                      ▼
   ┌──────────────────────────────────────────────────┐
   │  Step 3: Execute Graph                          │
   │  run_inference(model, {input_name: tensor})     │
   │  For each node in topological order:            │
   │    Look up inputs from tensor_map               │
   │    Dispatch to operator (Conv2D, ReLU, etc.)    │
   │    Store output in tensor_map                   │
   └──────────────────┬───────────────────────────────┘
                      │
                      ▼
        Output tensor: (1, 1000) probabilities
```

### How Operators Work

**Conv2D (the hardest operator):**
```
Input:  (batch=1, channels=3, height=224, width=224)
Filter: (out_channels=32, in_channels=3, kH=3, kW=3)
           ↓
    im2col transform
    (reshape each 3×3 receptive field into a row)
           ↓
    One big matrix multiply (Eigen handles this)
           ↓
Output: (batch=1, out_channels=32, height=222, width=222)
```

**BatchNorm (inference mode):**
```
Input:    x (any shape)
Scale/shift/running_mean/running_var from model weights
          ↓
  y = scale * (x - mean) / sqrt(var + eps) + bias
  (precomputed as y = a*x + b per channel — one FMA)
          ↓
Output: normalized tensor
```

---

## Supported Operators (13 total)

| Operator | What it does | Used in |
|----------|-------------|---------|
| `Gemm` | General matrix multiply: α·A·B + β·C | Dense layers, classifier heads |
| `MatMul` | Pure matrix multiply | Attention, linear projections |
| `Conv` | 2D convolution (im2col + Eigen) | Image feature extraction |
| `ReLU` | max(0, x) | After conv layers |
| `Sigmoid` | 1/(1+e^-x) | Gates, binary outputs |
| `Softmax` | Normalize to probability distribution | Final classification output |
| `GELU` | Gaussian error activation | Transformer models (BERT etc.) |
| `Clip` | Clamp to [min, max] | ReLU6 in MobileNetV2 |
| `MaxPool` | Spatial max reduction | Downsampling |
| `GlobalAveragePool` | Average across H×W → (N,C,1,1) | Before classifier head |
| `BatchNormalization` | Normalize + scale+shift (inference mode) | After conv in MobileNetV2 |
| `LayerNorm` | Normalize across feature dim | Transformer blocks |
| `Flatten` | Reshape dims to 1D from axis | Before dense layers |

---

## Downloading Sample Models

```bash
# Download MobileNetV2 and ResNet18 from ONNX Model Zoo
python models/download_models.py

# Or manually:
# MobileNetV2 (14MB): https://github.com/onnx/models/blob/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx
# ResNet18 (45MB):    https://github.com/onnx/models/blob/main/validated/vision/classification/resnet/model/resnet18-v2-7.onnx
```

Place them in the `models/` directory. They are gitignored (too large for the repo).

---

## Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `googletest submodule not initialized` | git submodule not pulled | Run `git submodule update --init --recursive` |
| `cmake: command not found` | CMake not installed | `sudo apt install cmake` / `winget install Kitware.CMake` |
| `cannot find -lcrucible` | Shared library not built | Add `-DCRUCIBLE_BUILD_SHARED=ON` to cmake configure |
| `ImportError: No module named 'crucible_py'` | pybind11 .so not on Python path | `sys.path.insert(0, 'engine/build/release')` |
| `unsupported op: DepthwiseConv` | Conv with groups>1 not implemented | Use a model with groups=1 only (plain MobileNetV2 first conv) |
| `Error: model file too large` | FastAPI server size limit | Set `MAX_MODEL_SIZE_MB=200` in `.env` |
| `wasm-pack: command not found` | wasm-pack not installed | `cargo install wasm-pack` |

---

## Real-World Use Cases

### Use Case 1: Edge Deployment Research
Crucible demonstrates the core of what **TFLite** and **ONNX Runtime Mobile** do for embedded systems. If you're building a custom inference engine for a microcontroller or FPGA, Crucible's clean source code shows exactly what you need: a tensor class, an operator table, and a graph executor — nothing more.

### Use Case 2: Understanding PyTorch Internals
When you call `torch.nn.Conv2d.forward()`, PyTorch ultimately calls into a cuDNN or MKLDNN kernel. Crucible's `conv2d.cpp` shows the *what* before the *how*: the im2col transformation, the GEMM, the reshape. Reading this before reading cuDNN makes the optimizations make sense.

### Use Case 3: Custom Operator Development
The operator dispatch table in `executor.cpp` is a direct model of how ONNX Runtime's custom operator API works. Adding a new operator to Crucible (e.g., `Mish`, `Swish`) follows the same pattern you'd use to add a custom op to ONNX Runtime:
1. Add the function signature to `ops/activations.hpp`
2. Implement in `ops/activations.cpp`
3. Add the `if (node.op_type == "Mish")` branch in `executor.cpp`
4. Add a test in `tests/test_activations.cpp`

### Use Case 4: Cross-Language Systems Design
Crucible's FFI stack — C++ core → extern C bridge → Rust CLI → WASM module → TypeScript web — is a real production pattern used by companies shipping AI products:
- Firefox uses Rust + C++ FFI for media codecs
- Figma ships a C++ renderer via WASM to the browser
- AWS Lambda functions call C++ extensions via Python ctypes

### Use Case 5: Interview Preparation
The project directly answers the most common systems + ML engineering interview questions:
- "Describe how a neural network runs on CPU" → `executor.cpp`
- "What is topological sort and where is it used?" → `executor.cpp::topological_sort`
- "How does Conv2D work under the hood?" → `ops/conv2d.cpp` + im2col explanation
- "What is WASM and how do you compile to it?" → `wasm/src/lib.rs` + `scripts/build-wasm.sh`
- "How do you safely call C++ from Rust?" → `engine/include/crucible/c_api.h` + `cli/src/runner.rs`
- "What is pybind11 and how does it work?" → `engine/bindings/python/`

---

## Project Structure Reference

```
Crucible/
├── engine/                 → C++17 inference engine (the core)
├── cli/                    → Rust command-line tool (calls C++ via FFI)
├── wasm/                   → Pure Rust → WebAssembly (browser inference)
├── server/                 → Python FastAPI (HTTP API + model conversion)
├── web/                    → Next.js 15 (interactive browser demo)
├── benchmarks/             → Python scripts comparing 3 engines
├── models/                 → ONNX models (gitignored, download separately)
├── docs/                   → Screenshots and demo assets
├── scripts/                → build-wasm.sh, setup.sh
├── .github/workflows/      → CI pipelines (4 total)
├── README.md               → Project overview + quick start
├── WRITEUP.md              → arXiv-style technical deep-dive
├── PROJECT_AUDIT.md        → Full issue-by-issue audit report
└── ENGINEERING_PLAN.md     → Original design document (source of truth)
```

---

## Environment Variables

Copy `.env.example` to `.env` and adjust:

```bash
# FastAPI Server
CRUCIBLE_MODEL_DIR=/tmp/crucible-models   # where uploaded models are stored
CRUCIBLE_ENGINE_PATH=./engine/build/release
MAX_MODEL_SIZE_MB=100

# Next.js
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WASM_PATH=/wasm/crucible_wasm.js

# Python bindings path (set after CMake build)
PYTHONPATH=./engine/build/release
```

---

## Frequently Asked Questions

**Q: Why is Crucible slower than ONNX Runtime?**
A: Three reasons: (1) No operator fusion — ONNX Runtime merges Conv+BatchNorm into one pass; Crucible runs them separately. (2) No MLAS kernels — ONNX Runtime uses Microsoft's hand-tuned AVX2/AVX-512 assembly for GEMM; Crucible uses Eigen's generic optimized GEMM. (3) No buffer reuse — each operator allocates fresh memory; ONNX Runtime reuses buffers via liveness analysis.

**Q: Why can't the C++ engine compile to WebAssembly?**
A: The C++ engine uses standard library file I/O (`std::fstream` for reading .onnx files), exceptions, and dynamic linking — none of which work in the WASM sandbox. The WASM module is a separate pure-Rust reimplementation of 4 core operators that work in-browser without any I/O.

**Q: Can I use Crucible with my own model?**
A: Yes, if your model uses only the 13 supported operators. Run `crucible validate --model mymodel.onnx` first. Common models that work: MobileNetV2, ResNet18, simple CNNs. Models that don't work yet: anything with depthwise conv (groups>1), LSTM, Transformer attention (missing Split/Concat operators).

**Q: How do I add a new operator?**
A: See "Use Case 3: Custom Operator Development" above. It takes about 30-60 minutes for a simple element-wise operator and 2-3 days for something like Conv2D.

**Q: Is this production-ready?**
A: No, and it's not designed to be. Crucible is an educational reference implementation. For production use, use ONNX Runtime or TFLite. Crucible's value is in teaching you how those engines work.

---

## Getting Help

- **Bug in build?** Check `git submodule update --init --recursive` first.
- **Test failures?** Run `ctest --test-dir engine/build/debug -V` for verbose output.
- **ONNX parsing error?** The model may use an unsupported opset or operator. Try `crucible validate`.
- **Performance questions?** Read `WRITEUP.md` §6 (Benchmarks) for the full analysis.
- **Architecture questions?** Read `ENGINEERING_PLAN.md` — it is the source of truth for every design decision.
