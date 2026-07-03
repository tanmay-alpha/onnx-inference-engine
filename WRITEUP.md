# Crucible: A From-Scratch ONNX Inference Engine in C++17

**Tanmay** · VIT Bhopal, Computer Science, Batch 2028  
`github.com/tanmay-alpha/Crucible`

---

## Abstract

We present **Crucible**, a from-scratch neural network inference engine that loads, parses, and executes ONNX models using C++17. Crucible implements 13 ONNX operators covering the complete MobileNetV2 [Howard et al., 2018] compute graph, a row-major tensor class with value semantics, an ONNX protobuf parser targeting opset 7, and a Kahn's-BFS graph executor. The same C++ core is exposed through four surfaces: a Python pybind11 module, a FastAPI HTTP server, a Rust CLI via `extern "C"` FFI, and a browser-executable pure-Rust WebAssembly subset. End-to-end MobileNetV2 inference runs in 14.3 ms on a commodity Intel CPU — approximately 3× slower than ONNX Runtime 1.18 (4.62 ms median), a gap explained entirely by the absence of operator fusion and MLAS micro-kernels. Numerical correctness is verified against ONNX Runtime: top-1 class agreement is 100% on a 1000-image ImageNet validation subset. Crucible is designed as a complete, readable reference implementation rather than a production optimizer.

---

## 1. Introduction

Large-scale ML frameworks like PyTorch [Paszke et al., 2019] and TensorFlow [Abadi et al., 2016] abstract inference behind highly optimised runtimes. TensorFlow Lite [David et al., 2021] and ONNX Runtime [ONNX Runtime, 2021] serve the deployment tier. The internals of these systems — how a Conv2D is dispatched, how graph execution is scheduled, how weight tensors are read from a protobuf file — remain opaque to most practitioners.

Crucible exists to close that gap. Every component is written from scratch with one primary constraint: **every design decision must be understandable from the source code alone**, without reference to the framework that inspired it.

**Scope.** Crucible supports CPU inference only. No GPU, no CUDA, no quantisation. The target models are MobileNetV2 and ResNet18 — small-to-medium vision backbones that stress the key operators (Conv2D, BatchNorm, GlobalAveragePool, Gemm) without requiring any exotic ops.

**Contributions.**
1. A readable, documented C++17 ONNX inference engine implementing 13 operators.
2. A multi-language surface (Python / Rust / WebAssembly) driven by the same C++ core.
3. A quantitative comparison methodology against ONNX Runtime and PyTorch on a standardised MobileNetV2 benchmark.
4. Design notes on the tradeoffs made at each layer that are not documented in production engines.

---

## 2. The Tensor Class

### 2.1 Memory Layout

Crucible tensors use **row-major (C-order) contiguous storage**: the rightmost index varies fastest. For a 4-D tensor of shape `(N, C, H, W)` — the NCHW layout used by MobileNetV2 — the element at position `(n, c, h, w)` lives at byte offset `(n·C·H·W + c·H·W + h·W + w) × sizeof(float)`.

This matches PyTorch's default CPU memory format and NumPy's default, making zero-copy transfers through pybind11 practical (§7.2).

### 2.2 Value Semantics

The `Tensor` class owns its data with a `std::vector<float>`. There is no `shared_ptr`, no reference counting, and no aliasing. Copy assignment does a full `memcpy`; move assignment transfers the underlying vector in O(1). The choice was made deliberately:

- **Predictable performance.** Shared ownership introduces cache-line contention when multiple tensors alias the same buffer. The graph executor (§5) holds many live tensors concurrently; shared ownership would make lifetime analysis non-local.
- **Exception safety.** Every constructor either succeeds or throws `std::invalid_argument`. There is no partially-constructed Tensor.
- **No raw pointers in the public API.** All pointer arithmetic is hidden inside `data()` (returns `float*`) and `at()` (computes offset, checks bounds). External code sees only shape queries and the raw buffer.

### 2.3 Bounds Checking

`at(indices)` validates that `indices.size() == rank()` and that each index is within the corresponding dimension. The check costs approximately 2 ns per call — negligible compared to operator work (µs to ms range). In the hot inner loops of Conv2D and MatMul, code accesses `data()` directly, bypassing `at()`.

### 2.4 Shape Operations

`reshape(new_shape)` and `flatten()` return **new** tensors by copying the data buffer. The source tensor is unchanged. This is intentionally safe-but-slow: copying a MobileNetV2 feature map (224×224×3 = 150,528 floats = 600 KB) at every reshape call costs roughly 150 µs. For the target latency regime (14 ms per full inference), this is a rounding error. A production engine would use strided views.

---

## 3. ONNX Graph Parsing

### 3.1 The ONNX Format

ONNX [ONNX, 2019] models are Protocol Buffer [Varda, 2008] serialisations of a `ModelProto` message defined in `onnx.proto3`. The top-level structure is:

```
ModelProto
  └── GraphProto
        ├── NodeProto[]     (operators: Conv, Gemm, Relu, …)
        ├── TensorProto[]   (initializers: weight tensors)
        ├── ValueInfoProto[] (input / output descriptors)
        └── AttributeProto  (per-node hyper-parameters: kernel_shape, pads, …)
```

### 3.2 Parser Design

`onnx_parser.cpp` reads a `.onnx` file into a `Model` struct containing:

```cpp
struct GraphNode {
    std::string op_type;           // "Conv", "Relu", "Gemm", …
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::map<std::string, Attribute> attrs;
};

struct Model {
    std::vector<GraphNode>          nodes;
    std::unordered_map<std::string, Tensor> initializers;  // weights
    std::vector<std::string>        input_names;
    std::vector<std::string>        output_names;
};
```

**Initializers** (weight tensors) are stored in `graph.initializer[]` in ONNX, not in `graph.input[]`. This is a common point of confusion: model inputs that appear in both `input[]` and `initializer[]` are weights, not user-provided data. The parser pre-loads all initializers into `tensor_map` before executing any node.

### 3.3 Attribute Handling

ONNX attributes are strongly typed (`FLOAT`, `INT`, `STRING`, `TENSOR`, `INTS`, `FLOATS`). The parser extracts them into a `std::variant`-based `Attribute` type. Operators query attributes by name with a default fallback — Conv2D defaults to `pads={0,0,0,0}`, `strides={1,1}`, `group=1`.

### 3.4 Opset Targeting

Crucible targets **opset 7** (MobileNetV2 uses opset 7). The parser does not attempt to handle opset upgrades. If a model uses a newer opset and the operator semantics changed (e.g., BatchNorm's training vs. inference mode in opset 9), the parser will silently use the opset 7 interpretation.

---

## 4. Operator Implementations

### 4.1 Gemm and MatMul (Linear)

The `Gemm` operator implements `Y = alpha * A' * B' + beta * C` where primes indicate optional transpositions controlled by `transA`/`transB` attributes. The matrix multiply is delegated to **Eigen** [Guennebaud et al., 2010]:

```cpp
using Map = Eigen::Map<Eigen::MatrixXf, Eigen::RowMajor>;
Map A_map(A.data(), M, K);
Map B_map(B.data(), K, N);
Eigen::MatrixXf C_mat = A_map * B_map;
```

Eigen's GEMM is highly optimised (blocked, vectorised with SSE/AVX on x86) and handles the heavy lifting. Crucible does not implement its own matrix multiply.

`MatMul` follows the NumPy broadcasting rules for tensors with `rank > 2`: only the last two dimensions participate in the multiply; leading dimensions are treated as batch dimensions.

### 4.2 Conv2D

2D convolution is the most complex operator. Crucible uses the **im2col** approach [Chellapilla et al., 2006]:

1. **im2col:** Reshape the input `(N, C_in, H, W)` into a matrix of shape `(N * H_out * W_out, C_in * kH * kW)` by extracting each receptive-field patch.
2. **GEMM:** Multiply by the filter matrix `(C_out, C_in * kH * kW)` transposed → `(N * H_out * W_out, C_out)`.
3. **Reshape + bias:** Reshape back to `(N, C_out, H_out, W_out)` and add per-channel bias.

This converts the O(N · C_out · H_out · W_out · C_in · kH · kW) nested loop into a single large matrix multiply, which Eigen can vectorise efficiently. The tradeoff is memory: the im2col buffer for a MobileNetV2 first conv layer (input 1×3×224×224, kernel 3×3, output 32 channels) is `(112×112)×(3×3×3) = 37,632 × 27 ≈ 4 MB` — acceptable for a batch size of 1.

Padding is implemented by zero-extending the input buffer before the im2col transform. Stride is handled by adjusting the output spatial dimensions.

### 4.3 Pooling

**MaxPool** slides a `kernel_shape × kernel_shape` window over the spatial dimensions, taking the maximum. Implemented as a direct nested loop — no im2col needed because the reduction is not a dot product.

**GlobalAveragePool** collapses the `(H, W)` spatial dimensions to `(1, 1)` by averaging. This is the final spatial aggregation in MobileNetV2 before the Gemm classifier head.

### 4.4 Batch Normalisation (Inference Mode)

In inference mode, BatchNorm fuses the running statistics:

```
y = (x - running_mean) / sqrt(running_var + eps) * scale + bias
```

The running mean and variance are stored as ONNX initializers (not computed at runtime). This is the correct inference-mode behaviour. Training-mode BatchNorm (which computes batch statistics) is not implemented.

### 4.5 Activation Functions

- **ReLU:** `max(0, x)` element-wise.
- **Sigmoid:** `1 / (1 + exp(-x))` — numerically stable for large negative inputs.
- **Softmax:** Subtract the row maximum before exponentiation for numerical stability: `exp(x - max(x)) / sum(exp(x - max(x)))`.
- **GELU:** `x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))` — the tanh approximation used by BERT-family models.
- **Clip:** Clamp to `[min, max]` — used to implement ReLU6 (`Clip(0, 6)`) in MobileNetV2's inverted residual blocks.

---

## 5. Graph Executor

### 5.1 Topological Sort

The executor implements **Kahn's BFS algorithm** [Kahn, 1962]:

1. Compute in-degree of each node (how many of its inputs are produced by other nodes).
2. Initialise the queue with all zero-in-degree nodes.
3. Dequeue a node, emit it to the sorted list, decrement the in-degree of all consumers.
4. Enqueue any consumer whose in-degree reaches zero.
5. If the sorted list length ≠ node count, the graph contains a cycle → throw `std::runtime_error`.

This is O(|V| + |E|) — linear in the number of nodes and edges. ONNX Runtime uses the same algorithm.

### 5.2 Operator Dispatch

Operator dispatch is an `if/else` chain on `op_type`:

```cpp
if (node.op_type == "Gemm")       result = gemm_forward(inputs, node.attrs);
else if (node.op_type == "Conv")   result = conv2d_forward(inputs, node.attrs);
else if (node.op_type == "Relu")   result = relu_forward(inputs, node.attrs);
// … 10 more operators
else throw std::invalid_argument("unknown op: " + node.op_type);
```

With 13 operators, a `std::function` dispatch table would add type-erasure overhead without benefit. When the operator count exceeds ~30, the chain should be replaced with a flat `std::unordered_map<std::string, OpFn>`.

### 5.3 Intermediate Tensor Management

The `tensor_map` — a `std::unordered_map<std::string, Tensor>` — holds every named tensor. After topological sort, the executor iterates the sorted nodes, looks up each input from `tensor_map`, runs the operator, and stores the output back. Because `Tensor` has move semantics, the `emplace` into the map moves the returned tensor — no copy.

Memory is not freed eagerly (no liveness analysis). A 224×224 MobileNetV2 inference materialises roughly 40 intermediate feature maps totalling ~120 MB at peak. A production engine would reuse buffers; this is left as future work.

---

## 6. Benchmarks

### 6.1 Methodology

All benchmarks use the same setup:
- Model: MobileNetV2 (`mobilenet-v2-7.onnx`, opset 7, ~14 MB)
- Input: `(1, 3, 224, 224)` float32 filled with random uniform values in `[-1, 1]`
- Runs: 100 timed inference calls after 10 warmup calls
- Timer: Python's `time.perf_counter()` (µs resolution)
- Platform: Intel Core i5, 16 GB RAM, no CUDA

Three engines benchmarked from `benchmarks/run_all.py`:
- **Crucible:** via Python bindings (`crucible_py.run`)
- **ONNX Runtime 1.18:** via `onnxruntime.InferenceSession`
- **PyTorch 2.3:** via `torch.nn.Module.forward` with `torch.no_grad()`

### 6.2 Results

| Engine | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) | Throughput (inf/s) |
|--------|-----------|-------------|---------|---------|-------------------|
| **Crucible** | **14.3** | **13.8** | 18.2 | 21.4 | 70.1 |
| ONNX Runtime 1.18 | 5.11 | 4.62 | 7.59 | 9.60 | 195.8 |
| PyTorch 2.3 | 1.94 | 1.29 | 3.98 | 9.19 | 516.4 |

### 6.3 Analysis

Crucible's 14.3 ms mean is approximately **3.1× slower than ONNX Runtime** on median latency. This gap is fully expected and attributable to three factors:

1. **No operator fusion.** ONNX Runtime fuses BatchNorm into the preceding Conv2D at graph-optimisation time, eliminating a full pass over feature map memory. Crucible runs BatchNorm as a separate pass.
2. **No MLAS kernels.** ONNX Runtime uses the MLAS (Microsoft Linear Algebra Subprograms) library for GEMM, which generates hand-tuned AVX2/AVX-512 assembly. Crucible uses Eigen's generic GEMM, which is good but not architecture-specifically tuned.
3. **No memory buffer reuse.** Each operator allocates a fresh output buffer. ONNX Runtime reuses buffers across operators using a live-range analysis.

The PyTorch number (1.29 ms median) reflects PyTorch's extremely aggressive model loading and execution pipeline, including AOT-compiled kernels and on-device profiling over the 10-call warmup. This is not a fair comparison to Crucible's interpreter, which has no compilation phase.

**Numerical agreement:** On 1000 random ImageNet-preprocessed inputs, the top-1 class predicted by Crucible matches ONNX Runtime in 100% of cases. The maximum absolute difference in output logits is < 1e-4, well within the float32 rounding budget.

---

## 7. Multi-Language Surface

### 7.1 Rust CLI via `extern "C"` FFI

The Rust CLI (`cli/`) calls into the C++ engine through a C ABI bridge (`engine/include/crucible/c_api.h`). All C++ exceptions are caught inside the C bridge and translated to integer error codes — exceptions cannot cross the FFI boundary safely.

```c
// c_api.h — the only header the Rust code sees
int crucible_load_model(const char* path, CrucibleModel** out_model);
int crucible_run(CrucibleModel* model, const float* input, int64_t n,
                 float** out_data, int64_t** out_shape, int64_t* out_rank);
void crucible_free_model(CrucibleModel* model);
void crucible_free_buffer(float* buf);
```

On the Rust side, `runner.rs` wraps these raw FFI calls into safe wrappers (`Model`, `Tensor`, `Status` types). All `unsafe` blocks are localised to `runner.rs` — every other file in the CLI crate is safe Rust.

### 7.2 pybind11 Python Bindings

The `crucible_py` module exposes two functions:

```python
model = crucible_py.load_model("mobilenet_v2.onnx")
output: np.ndarray = crucible_py.run(model, np.zeros((1, 3, 224, 224), dtype=np.float32))
```

The numpy ↔ Tensor bridge uses `py::array_t<float, py::array::c_style | py::array::forcecast>`. The `forcecast` flag ensures that non-contiguous arrays (e.g., PyTorch tensors converted to numpy with a non-unit stride) are made contiguous before being passed to the engine. The data is **copied** into a `Tensor` — zero-copy is intentionally not used to avoid aliasing issues when the Python object is garbage-collected mid-inference.

### 7.3 WebAssembly Pure-Rust Subset

The C++ engine cannot compile to WebAssembly — the standard library's file I/O, dynamic linking, and exception handling are not available in the WASM sandbox. Instead, `wasm/src/lib.rs` implements a **pure-Rust subset** of 4 operators (MatMul, ReLU, Softmax, Sigmoid) that covers the operators exposed in the web demo's emulation mode.

The WASM module is compiled with `wasm-pack build --target web` and produces `crucible_wasm_bg.wasm` + `crucible_wasm.js`. The TypeScript loader (`web/src/lib/crucible-wasm.ts`) lazily initialises the WASM instance on first call.

---

## 8. Future Work

### 8.1 Operator Fusion

The highest-impact optimisation is **Conv+BatchNorm fusion**: absorb the BatchNorm scale/shift into the Conv weights at graph-load time, eliminating a full pass over the feature map. This is a graph-level transformation done before execution, not during it.

### 8.2 Memory Buffer Reuse

Implement a liveness analysis pass after topological sort: if tensor T is consumed by node N and never used after N, its buffer can be freed immediately. A buddy-allocator pool across the inference would reduce peak memory from ~120 MB to ~30 MB.

### 8.3 OpenMP Parallelism

The inner loops of Conv2D (output channel iteration) are embarrassingly parallel. Adding `#pragma omp parallel for` to the outer output-channel loop in `conv2d_forward` would utilise all CPU cores with no algorithmic change.

### 8.4 INT8 Quantisation

Post-training static quantisation (per-tensor or per-channel) would reduce MobileNetV2 to INT8, cutting memory bandwidth by 4× and enabling use of VNNI/DPAS instructions on modern Intel CPUs.

### 8.5 Larger Model Support

Current limitations: no dynamic shapes, no LSTM/GRU ops, no Split/Concat along arbitrary axes. Adding these would enable BERT-family transformer inference.

---

## References

1. **Howard et al. (2018).** MobileNetV2: Inverted Residuals and Linear Bottlenecks. *CVPR 2018.* https://arxiv.org/abs/1801.04381

2. **Guennebaud, Jacob et al. (2010).** Eigen v3. http://eigen.tuxfamily.org

3. **Varda, Kenton (2008).** Protocol Buffers: Google's Data Interchange Format. *Google Open Source Blog.*

4. **ONNX (2019).** Open Neural Network Exchange. https://onnx.ai/

5. **Paszke et al. (2019).** PyTorch: An Imperative Style, High-Performance Deep Learning Library. *NeurIPS 2019.* https://arxiv.org/abs/1912.01703

6. **David et al. (2021).** TensorFlow Lite Micro: Embedded Machine Learning for TinyML Systems. *MLSys 2021.* https://arxiv.org/abs/2010.08678

7. **Chellapilla, Puri, Simard (2006).** High Performance Convolutional Neural Networks for Document Processing. *IWFHR 2006.*

8. **Kahn, A.B. (1962).** Topological Sorting of Large Networks. *Communications of the ACM, 5(11), 558–562.*

9. **ONNX Runtime (2021).** Microsoft ONNX Runtime. https://github.com/microsoft/onnxruntime

10. **Abadi et al. (2016).** TensorFlow: A System for Large-Scale Machine Learning. *OSDI 2016.* https://arxiv.org/abs/1605.08695
