# Crucible: Project Deep Dive & Technical Specification

Crucible is a lightweight, from-scratch ONNX inference engine written in C++17, compiled to WebAssembly for client-side browser execution, and exposed via Rust, Python, and FastAPI.

This document provides a complete technical deep-dive of the current project state, a file-by-file directory breakdown, the core engineering design patterns, and future plans. It is structured to serve as context for ChatGPT/Lovable to generate advanced frontend implementations and backend database extensions.

---

## 1. Project Core Aims

1. **Lightweight Edge Inference:** Offer a sub-3MB runtime binary that runs natively in edge environments (browsers, embedded devices, IoT) where installing ONNX Runtime (~50MB+) or PyTorch (~750MB+) is impossible.
2. **On-Device Privacy:** Enable privacy-sensitive machine learning use cases (e.g., client-side fraud scoring, biometric matching, document OCR) where raw data cannot leave the user's device due to compliance (GDPR, PCI-DSS, HIPAA).
3. **High-Fidelity Engineering:** Built from the ground up in C++17 to showcase topological execution, custom memory layout (row-major NCHW), tensor operators, and FFI bridging.

---

## 2. Directory Layout & File-by-File Breakdown

### Root Directory
* [`README.md`](file:///c:/Users/TANMAY/OneDrive/Desktop/Crucible/README.md) — The repository's front page, reframed to highlight client-side WASM privacy and footprint comparison metrics.
* [`WRITEUP.md`](file:///c:/Users/TANMAY/OneDrive/Desktop/Crucible/WRITEUP.md) — A 4-page engineering write-up detailing memory layouts, execution order, FFI boundaries, and allocation safety.
* [`CONTEXT.md`](file:///c:/Users/TANMAY/OneDrive/Desktop/Crucible/CONTEXT.md) — The active developer notes tracking known issues, preset configurations, and recent modifications.
* [`LICENSE`](file:///c:/Users/TANMAY/OneDrive/Desktop/Crucible/LICENSE) — MIT License (2025-2026, copyright Tanmay).
* [`CMakePresets.json`](file:///c:/Users/TANMAY/OneDrive/Desktop/Crucible/CMakePresets.json) — Build presets for Debug and Release builds of the C++ engine core.

---

### `engine/` — C++17 Core Runtime
This contains the primary C++ engine responsible for loading, parsing, and executing ONNX graphs on CPU.

* **`engine/src/`**
  * [`tensor.h`](file:///c:/Users/TANMAY/OneDrive/Desktop/Crucible/engine/src/tensor.h) / [`tensor.cpp`](file:///c:/Users/TANMAY/OneDrive/Desktop/Crucible/engine/src/tensor.cpp) — Custom tensor class representing multidimensional arrays. Uses a flat `std::vector<float>` under row-major NCHW layout. Implements indexing, dimension mapping, and helper methods.
  * [`graph.h`](file:///c:/Users/TANMAY/OneDrive/Desktop/Crucible/engine/src/graph.h) / [`graph.cpp`](file:///c:/Users/TANMAY/OneDrive/Desktop/Crucible/engine/src/graph.cpp) — Manually decodes ONNX models from Protobuf binary wire format. Parses model inputs, outputs, initializers (static weights), and graph nodes. Runs Kahn's topological sort algorithm to order nodes.
  * [`c_api.h`](file:///c:/Users/TANMAY/OneDrive/Desktop/Crucible/engine/src/c_api.h) / [`c_api.cpp`](file:///c:/Users/TANMAY/OneDrive/Desktop/Crucible/engine/src/c_api.cpp) — C-compatible FFI interface wrapping C++ objects. Prevents C++ exception leakage across foreign function interfaces by returning integer error codes (`0` for success).
  * **`engine/src/ops/`** — Native C++ operator kernels:
    * [`conv2d.cpp`](file:///c:/Users/TANMAY/OneDrive/Desktop/Crucible/engine/src/ops/conv2d.cpp) — 2D Convolution kernel supporting padding, strides, dilations, and general grouped/depthwise convolutions (where `groups == in_channels`). Leverages im2col mapping and Eigen matrix-multiply.
    * [`pooling.cpp`](file:///c:/Users/TANMAY/OneDrive/Desktop/Crucible/engine/src/ops/pooling.cpp) — Implements `MaxPool`, `AveragePool`, and `GlobalAveragePool` using sliding window reduction.
    * [`relu.cpp`](file:///c:/Users/TANMAY/OneDrive/Desktop/Crucible/engine/src/ops/relu.cpp) — Fast element-wise Rectified Linear Unit (`max(0, x)`).
    * [`matmul.cpp`](file:///c:/Users/TANMAY/OneDrive/Desktop/Crucible/engine/src/ops/matmul.cpp) — 2D matrix multiplication using Eigen's matrix engine.
    * [`batchnorm.cpp`](file:///c:/Users/TANMAY/OneDrive/Desktop/Crucible/engine/src/ops/batchnorm.cpp) — Performs inference-mode batch normalization using pre-computed running mean, variance, scale, and bias.
* **`engine/tests/`** — High-coverage GoogleTest suite testing operator precision and graph runner execution.

---

### `wasm/` — Pure Rust WASM Module
Since C++ cannot compile into standard `wasm32-unknown-unknown` targets cleanly without huge size overhead, Crucible implements a pure-Rust parser and interpreter that duplicates the activation kernels (MatMul, Sigmoid, Softmax, ReLU) in a tiny (<3MB) package.

* [`Cargo.toml`](file:///c:/Users/TANMAY/OneDrive/Desktop/Crucible/wasm/Cargo.toml) — Rust package configuration exposing compilation profiles optimized for binary size.
* **`wasm/src/`**
  * [`lib.rs`](file:///c:/Users/TANMAY/OneDrive/Desktop/Crucible/wasm/src/lib.rs) — Hand-written Protobuf binary wire format decoder in Rust. Exposes:
    * `run_inference`: General FFI execution for arbitrary input/shape arrays.
    * `run_fraud_model`: Typed helper function that bakes in normalisation parameters (mean/std) to perform instant client-side transaction fraud scoring.

---

### `web/` — Next.js 15 App Workspace
A Next.js frontend built in TypeScript. The old UI/UX pages have been fully removed, leaving a clean slate for the Lovable integration.

* **`web/src/`**
  * **`web/src/lib/`** — Core integration utilities (Preserved from cleanup):
    * [`crucible-wasm.ts`](file:///c:/Users/TANMAY/OneDrive/Desktop/Crucible/web/src/lib/crucible-wasm.ts) — Lazily fetches the compiled `.onnx` model, caches the binary buffer on the client, normalizes input variables, and calls the Rust WASM module (`runFraudModel`).
    * [`api.ts`](file:///c:/Users/TANMAY/OneDrive/Desktop/Crucible/web/src/lib/api.ts) — Interface to fetch benchmark reports. Falls back to pre-measured release stats if local benchmark outputs are empty. Contains operator capability mappings.
  * **`web/src/app/`** — Empty, waiting for Lovable pages.
  * **`web/src/components/`** — Empty, waiting for Lovable client widgets.
* **`web/public/`**
  * **`web/public/wasm/`** — Compiled `.wasm`, `.js` bindings, and `.d.ts` declaration maps generated by `wasm-pack`.
  * **`web/public/models/`** — Holds optimized models like `fraud_detector.onnx`.

---

### `models/` — Machine Learning Models
* **`models/fraud/`**
  * [`train_fraud_model.py`](file:///c:/Users/TANMAY/OneDrive/Desktop/Crucible/models/fraud/train_fraud_model.py) — Python script that creates a synthetic PaySim transaction dataset (50,000 records: legitimate vs. fraudulent), trains a Logistic Regression classifier (yielding AUC of 1.00 on this separable distribution), and exports the weights manually to an ONNX graph (`MatMul -> Add -> Sigmoid`) using the onnx library.
  * [`model_config.json`](file:///c:/Users/TANMAY/OneDrive/Desktop/Crucible/models/fraud/model_config.json) — Scaling/normalization configurations used to match web input fields to model expectations.

---

### `benchmarks/` — Performance Benchmarking
* [`bench_matmul.cpp`](file:///c:/Users/TANMAY/OneDrive/Desktop/Crucible/benchmarks/bench_matmul.cpp) — C++ micro-benchmarks comparing linear algebra performance.
* [`bench_onnxruntime.py`](file:///c:/Users/TANMAY/OneDrive/Desktop/Crucible/benchmarks/bench_onnxruntime.py) / [`bench_pytorch.py`](file:///c:/Users/TANMAY/OneDrive/Desktop/Crucible/benchmarks/bench_pytorch.py) — Python pipelines that run PyTorch and ONNX Runtime on the exact same CPU to record latency statistics for comparison.
* **`benchmarks/results/`**
  * [`benchmark_results.json`](file:///c:/Users/TANMAY/OneDrive/Desktop/Crucible/benchmarks/results/benchmark_results.json) — Pushed file containing the latest release stats.

---

## 3. How the Integration Works

```
                     ┌────────────────────────┐
                     │   Next.js Frontend     │
                     └───────────┬────────────┘
                                 │
                     calls runFraudDetection()
                                 │
                                 ▼
                     ┌────────────────────────┐
                     │    crucible-wasm.ts    │
                     └───────────┬────────────┘
                                 │
                       1. Fetches onnx model
                       2. Normalizes inputs
                                 │
                                 ▼
                     ┌────────────────────────┐
                     │    WebAssembly JS      │
                     │  (crucible_wasm.js)    │
                     └───────────┬────────────┘
                                 │
                                 ▼
                     ┌────────────────────────┐
                     │    Compiled Rust       │
                     │   (crucible_wasm.wasm) │
                     └────────────────────────┘
```

When a user runs a transaction analysis on the fraud page:
1. The Next.js page captures form inputs (e.g. amount, transaction type).
2. It calls `runFraudDetection` from `crucible-wasm.ts`.
3. `crucible-wasm.ts` checks if the `.onnx` model is loaded in memory. If not, it runs an HTTP `fetch` to retrieve `/models/fraud_detector.onnx` and caches it as a `Uint8Array`.
4. It performs z-score normalization on the numerical inputs using the mean and standard deviation from the configuration.
5. It triggers `runFraudModel` exported by the WASM module.
6. The WASM module decodes the ONNX bytes, sets up the mathematical graph, runs inference, and returns a probability value (`0.0 - 1.0`).

---

## 4. Proposed Database Schema for Advanced Features

To make the application more advanced, we will build a FastAPI backend (running alongside the Next.js page) backed by a relational database (SQLite/PostgreSQL) to store transaction logs, user models, and feedback loops.

### Table: `users`
* `id`: UUID (Primary Key)
* `email`: VARCHAR(255) (Unique)
* `password_hash`: VARCHAR(255)
* `created_at`: TIMESTAMP

### Table: `inference_history`
* `id`: BIGSERIAL (Primary Key)
* `user_id`: UUID (Foreign Key -> users.id, Nullable for guests)
* `transaction_type`: VARCHAR(50)
* `amount`: DECIMAL(12,2)
* `fraud_probability`: FLOAT
* `prediction_label`: VARCHAR(20) (FRAUD / LEGITIMATE)
* `latency_ms`: FLOAT
* `user_flagged`: BOOLEAN (Default: False, used for human-in-the-loop audit)
* `created_at`: TIMESTAMP

### Table: `custom_models`
* `id`: UUID (Primary Key)
* `user_id`: UUID (Foreign Key -> users.id)
* `name`: VARCHAR(100)
* `version`: INT
* `onnx_blob_path`: TEXT (Storage path)
* `model_metadata`: JSONB (Features, mean, std)
* `uploaded_at`: TIMESTAMP

---

## 5. Next Steps

1. **Lovable Frontend Generation:** Lovable will generate the Next.js pages and CSS charts based on the dashboard design prompt.
2. **Merging & FFI Wiring:** We will drop the generated code into `web/src/app` and `web/src/components` and wire it up to `web/src/lib/crucible-wasm.ts` and `web/src/lib/api.ts`.
3. **Database Integration:** We will implement the FastAPI backend models and database configurations to record inference events and support model versioning.
