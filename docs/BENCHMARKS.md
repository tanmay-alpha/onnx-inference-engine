# Benchmarks

Everything on this page comes from one file,
[`web/public/benchmarks/results.json`](../web/public/benchmarks/results.json),
written by [`benchmarks/run_all.py`](../benchmarks/run_all.py). The website's
benchmark page reads the same file. Nothing is typed in by hand: if the file is
missing, the site shows an empty state instead of numbers.

**Short version: Crucible is slower than ONNX Runtime on the CNNs.** The
measured ratios are in the table below. Crucible is a single-threaded engine
written for learning. It does not try to beat ORT's kernels.

## What is measured

| Runtime | What it is |
|---|---|
| `crucible-native` | `crucible_py` (pybind11 module over the C++ engine). Portable Release build (`-O3`, no `-march`). Single-threaded. |
| `crucible-native-avx2` | The same source built with `-DCRUCIBLE_MARCH_NATIVE=ON` (`-march=native`). **Host-specific.** The binary may not run on other CPUs. |
| `onnxruntime-cpu` | ONNX Runtime CPU EP with default `SessionOptions`, so ORT picks its own intra-op thread count. This is the parity reference. |
| `onnxruntime-cpu-1t` | ORT with `intra_op_num_threads = inter_op_num_threads = 1` and `ORT_SEQUENTIAL`. Compare this one with `crucible-native`. |
| `pytorch-cpu` | The **same ONNX file** converted with `onnx2torch`. onnx2torch has no converter for opset-7 `BatchNormalization`, so the opset-7 CNNs are first upgraded to opset 13 with `onnx.version_converter`. The graph and weights stay the same; only operator versions change. The upgrade is recorded in the row's `note`. The converted model is checked against ORT on the same input (max abs diff ≤ 1e-3 and the same arg-max). The row is marked `unsupported` if conversion fails or the outputs differ. The harness never swaps in a different network. `torch.set_num_threads` is set (default: detected physical cores) and recorded. |
| `crucible-wasm` | The pure-Rust WASM runtime shipped at `web/public/wasm` (wasm-bindgen `web` target). It runs under Node via `initSync` on the `.wasm` bytes and calls the same `runInference` export the browser uses. It only implements Add/MatMul/Relu/Sigmoid/Softmax, so in practice only the fraud model runs. On the CNNs, `runInference` throws and the row is recorded as `unsupported`. |

Models (from `models/`):

| Key | File | Inputs benchmarked |
|---|---|---|
| `resnet18` | `models/resnet18.onnx` | `[1,3,224,224]` |
| `mobilenet_v2` | `models/mobilenet_v2.onnx` | `[1,3,224,224]` (static batch 1) |
| `fraud` | `models/fraud/fraud_detector.onnx` (MatMul → Add → Sigmoid) | `[1,7]` and `[64,7]` |

## Methodology

- **Inputs:** float32 uniform `[0,1)` from `numpy.random.default_rng(0)`, generated fresh for every (model, shape). Every runtime gets byte-identical input.
- **Isolation:** each (runtime, model, shape) runs in its own subprocess, one at a time. Thread pools are never shared, and two `crucible_py` builds can be measured in one run.
- **Loop:** `warmup` untimed calls, then `iterations` timed calls. Each call is timed on its own with `time.perf_counter` (Python) or `process.hrtime.bigint` (Node). The timed region is one inference call from the host language, including the input/output copies across the binding.
- **Statistics:** mean, median, p95 (linear interpolation), min, and population std, all in milliseconds. `throughput_ips = batch / median`.
- **`load_ms`:** `crucible_py.load_model`, ORT `InferenceSession` construction (includes ORT graph optimisation), `onnx2torch` conversion, or WASM compile + instantiate, depending on the runtime.
- **Parity:** `max_abs_diff_vs_ort` compares the output of the last timed call with `onnxruntime-cpu` on the same input. `top1_match` compares the arg-max for each row. It is `null` for single-column outputs such as the fraud probability.
- **Environment:** the JSON records the CPU (`/proc/cpuinfo`), logical cores, OS and kernel, Python, compiler (taken from the build tree's `CMakeCache.txt`), ORT and torch versions, build flags, git commit (`-dirty` if tracked files were modified), and the byte sizes of the runtime artifacts.
- **Hard failures:** if `crucible_py` is missing or cannot be imported, the run stops with an error. The old numpy "fallback" that returned zeros has been removed.

## Reproduce

### Linux (or the dev container)

```bash
git submodule update --init engine/third_party/eigen engine/third_party/googletest engine/third_party/pybind11
pip install numpy==1.26.4 onnx onnxruntime
# optional, for pytorch-cpu (CPU wheels; torch and torchvision versions must match)
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install onnx2torch pillow
# Node >= 18 on PATH for crucible-wasm (optional)

# portable engine module (what crucible-native measures)
cmake -S engine -B build/portable -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCRUCIBLE_ENABLE_PYTHON_BINDINGS=ON -DPython_EXECUTABLE=$(which python)
cmake --build build/portable --target crucible_py

# optional: host-specific -march=native module and the C ABI shared library
cmake -S engine -B build/native -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCRUCIBLE_ENABLE_PYTHON_BINDINGS=ON -DCRUCIBLE_MARCH_NATIVE=ON -DPython_EXECUTABLE=$(which python)
cmake --build build/native --target crucible_py
cmake -S engine -B build/shared -G Ninja -DCMAKE_BUILD_TYPE=Release -DCRUCIBLE_BUILD_SHARED=ON
cmake --build build/shared --target crucible

python benchmarks/run_all.py --out web/public/benchmarks/results.json \
    --crucible-dir build/portable/python \
    --crucible-avx2-dir build/native/python \
    --libcrucible build/shared/libcrucible.so \
    --warmup 20 --iterations 200
python benchmarks/render_table.py web/public/benchmarks/results.json   # Markdown tables
```

`--models resnet18 mobilenet_v2 fraud` limits the run to some models. If
`--crucible-dir` is omitted, `$CRUCIBLE_PY_DIR` is used.

### Windows

The published numbers come from a Linux container (Docker Desktop, WSL2
backend) on a Windows laptop, because the engine's Python module is built
with GCC there:

```powershell
docker start crucible-dev
docker exec crucible-dev bash -c "cd /src && python benchmarks/run_all.py --out web/public/benchmarks/results.json --crucible-dir /build/portable/python --crucible-avx2-dir /build/native/python --libcrucible /build/shared/libcrucible.so --warmup 20 --iterations 200"
```

A native Windows run needs an MSVC build of `crucible_py`
(`cmake -S engine -B build/msvc -DCRUCIBLE_ENABLE_PYTHON_BINDINGS=ON`, then
`cmake --build build/msvc --config Release --target crucible_py`). After that,
run the same `run_all.py` command with `--crucible-dir` pointing at the folder
that contains the `.pyd`. Without `/proc/cpuinfo`, the harness falls back to
`platform.processor()` for the CPU name and uses the logical core count. This
path has not been used for the published numbers.

## Results

Run: `2026-11-07T19:03:55+00:00`, commit `d44218f-dirty`, 20 warmup + 200 timed iterations.
Host: Intel(R) Core(TM) i7-14650HX, 24 logical CPUs, Debian GNU/Linux 12 (bookworm) (kernel 6.18.33.2-microsoft-standard-WSL2), Python 3.11.18, c++ (Debian 12.2.0-14+deb12u1) 12.2.0,
onnxruntime 1.31.1, torch 2.14.1+cpu. Docker Desktop (WSL2 backend) container on a Windows 11 laptop.

### Headline: CNN median latency, batch 1 (lower is better)

| Model | crucible-native ms | ORT 1 thread ms | Crucible vs ORT-1t | ORT default ms | Crucible vs ORT default | crucible-native-avx2 ms (host-specific) | avx2 vs ORT-1t |
|---|---:|---:|---:|---:|---:|---:|---:|
| resnet18 | 107.6 | 26.9 | **4.0× slower** | 5.9 | 18.2× slower | 44.1 | 1.6× slower |
| mobilenet_v2 | 48.8 | 8.1 | **6.0× slower** | 2.2 | 22.0× slower | 33.9 | 4.2× slower |

On the CNNs, Crucible is **slower than ONNX Runtime**. With one thread on
both sides, it takes 4.0× as long as ORT on ResNet18 and
6.0× as long on MobileNetV2. Against ORT's default thread pool the gap is
much larger. Building with `-march=native` narrows the gap. That build is
host-specific and is not what ships. Every Crucible output matches ORT within
about 1e-5, with the same top-1 class.

On the 7-feature fraud model, every runtime finishes in a few microseconds, so
the numbers mostly measure per-call overhead. Crucible's per-call path is
lighter than ORT's Python binding here. That says nothing about kernel speed.

### Full table

Ratios are `median ÷ ORT median` for the same model and input: 2.00 means
twice as slow, 0.50 means twice as fast. "Load ms" is defined in the
Methodology section.

| Model | Input | Runtime | Threads | Median ms | Mean ms | p95 ms | Min ms | Std ms | Throughput (inf/s) | Load ms | Max abs diff vs ORT | Top-1 match | median ÷ ORT-1t median | median ÷ ORT median |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|---:|---:|
| resnet18 | 1×3×224×224 | onnxruntime-cpu | 12 | 5.925 | 6.232 | 7.644 | 5.094 | 0.7514 | 168.8 | 434.8 | — | — | 0.22 | 1.00 |
| resnet18 | 1×3×224×224 | onnxruntime-cpu-1t | 1 | 26.92 | 27.24 | 29.82 | 25.04 | 1.753 | 37.1 | 424.5 | 0 | yes | 1.00 | 4.54 |
| resnet18 | 1×3×224×224 | crucible-native | 1 | 107.6 | 110.9 | 130.1 | 100.1 | 9.733 | 9.3 | 471.1 | 7.15e-06 | yes | 4.00 | 18.17 |
| resnet18 | 1×3×224×224 | crucible-native-avx2 | 1 | 44.12 | 45.25 | 51.99 | 42.3 | 3.145 | 22.7 | 446.9 | 6.68e-06 | yes | 1.64 | 7.45 |
| resnet18 | 1×3×224×224 | pytorch-cpu | 12 | 13.93 | 14 | 15.84 | 11.19 | 1.041 | 71.8 | 1058 | 4.05e-06 | yes | 0.52 | 2.35 |
| resnet18 | 1×3×224×224 | crucible-wasm | 1 | unsupported | | | | | | | | | | |
| mobilenet_v2 | 1×3×224×224 | onnxruntime-cpu | 12 | 2.223 | 2.285 | 2.588 | 1.745 | 0.9623 | 449.9 | 142 | — | — | 0.27 | 1.00 |
| mobilenet_v2 | 1×3×224×224 | onnxruntime-cpu-1t | 1 | 8.083 | 8.293 | 9.816 | 7.251 | 0.6718 | 123.7 | 137.2 | 0 | yes | 1.00 | 3.64 |
| mobilenet_v2 | 1×3×224×224 | crucible-native | 1 | 48.82 | 50.17 | 58.18 | 46.53 | 3.862 | 20.5 | 126.2 | 1.41e-05 | yes | 6.04 | 21.97 |
| mobilenet_v2 | 1×3×224×224 | crucible-native-avx2 | 1 | 33.92 | 34.3 | 37.2 | 32.07 | 2.129 | 29.5 | 148.1 | 1.39e-05 | yes | 4.20 | 15.26 |
| mobilenet_v2 | 1×3×224×224 | pytorch-cpu | 12 | 12.61 | 12.97 | 16.93 | 9.401 | 2.035 | 79.3 | 313.4 | 1.22e-05 | yes | 1.56 | 5.67 |
| mobilenet_v2 | 1×3×224×224 | crucible-wasm | 1 | unsupported | | | | | | | | | | |
| fraud_detector | 1×7 | onnxruntime-cpu | 12 | 0.006301 | 0.008896 | 0.01228 | 0.005566 | 0.01232 | 158,705 | 9.755 | — | — | 1.95 | 1.00 |
| fraud_detector | 1×7 | onnxruntime-cpu-1t | 1 | 0.003233 | 0.003276 | 0.003596 | 0.003109 | 0.0002 | 309,358 | 7.216 | 0 | — | 1.00 | 0.51 |
| fraud_detector | 1×7 | crucible-native | 1 | 0.001829 | 0.001839 | 0.001916 | 0.001764 | 5.3e-05 | 546,597 | 2.09 | 9.31e-09 | — | 0.57 | 0.29 |
| fraud_detector | 1×7 | crucible-native-avx2 | 1 | 0.001818 | 0.001974 | 0.002359 | 0.001762 | 0.000256 | 550,206 | 5.885 | 9.31e-09 | — | 0.56 | 0.29 |
| fraud_detector | 1×7 | pytorch-cpu | 12 | 0.01136 | 0.01245 | 0.01533 | 0.01082 | 0.005907 | 88,055 | 17.72 | 9.31e-09 | — | 3.51 | 1.80 |
| fraud_detector | 1×7 | crucible-wasm | 1 | 0.01222 | 0.01299 | 0.01502 | 0.01052 | 0.004949 | 81,863 | 2.105 | 9.31e-09 | — | 3.78 | 1.94 |
| fraud_detector | 64×7 | onnxruntime-cpu | 12 | 0.004034 | 0.004697 | 0.006997 | 0.00361 | 0.002254 | 15,863,180 | 8.803 | — | — | 1.07 | 1.00 |
| fraud_detector | 64×7 | onnxruntime-cpu-1t | 1 | 0.003768 | 0.004258 | 0.005073 | 0.003486 | 0.004147 | 16,982,885 | 6.793 | 0 | — | 1.00 | 0.93 |
| fraud_detector | 64×7 | crucible-native | 1 | 0.00258 | 0.002564 | 0.002809 | 0.002278 | 0.00021 | 24,801,395 | 4.045 | 5.59e-08 | — | 0.68 | 0.64 |
| fraud_detector | 64×7 | crucible-native-avx2 | 1 | 0.002398 | 0.002522 | 0.003198 | 0.002252 | 0.00052 | 26,683,345 | 4.697 | 5.59e-08 | — | 0.64 | 0.59 |
| fraud_detector | 64×7 | pytorch-cpu | 12 | 0.01237 | 0.01292 | 0.0146 | 0.01164 | 0.002404 | 5,172,971 | 15.57 | 5.96e-08 | — | 3.28 | 3.07 |
| fraud_detector | 64×7 | crucible-wasm | 1 | 0.01606 | 0.01798 | 0.02631 | 0.01366 | 0.007161 | 3,985,925 | 2.782 | 5.59e-08 | — | 4.26 | 3.98 |

Unsupported rows:

- resnet18 1×3×224×224 / crucible-wasm: unsupported — runInference threw: Unsupported operator 'BatchNormalization' (WASM runtime supports: Add, MatMul, Relu, Sigmoid, Softmax)
- mobilenet_v2 1×3×224×224 / crucible-wasm: unsupported — runInference threw: Unsupported operator 'Conv' (WASM runtime supports: Add, MatMul, Relu, Sigmoid, Softmax)

Artifacts:

| Artifact | Bytes |
|---|---:|
| crucible_py (portable): crucible_py.cpython-311-x86_64-linux-gnu.so | 508,400 |
| crucible_py (march=native): crucible_py.cpython-311-x86_64-linux-gnu.so | 533,008 |
| crucible C ABI shared library: libcrucible.so | 400,800 |
| crucible-wasm: crucible_wasm_bg.wasm | 92,155 |
| onnxruntime native: onnxruntime_pybind11_state.cpython-311-x86_64-linux-gnu.so | 30,893,160 |
| pytorch native: libtorch_cpu.so | 438,232,064 |

Regenerate this section with `python benchmarks/render_table.py web/public/benchmarks/results.json`.

## Limitations and how to read the numbers

- **Crucible is single-threaded.** The fair comparison is `crucible-native` against `onnxruntime-cpu-1t`. The default-threads ORT row shows what a production runtime gets on this machine, not a like-for-like comparison.
- **Portable build.** `crucible-native` is built without `-march=native`, which is how the server and CI ship it. The `-avx2` row is host-specific and is only here to show how much the compiler flags matter.
- **Docker on a laptop.** These numbers come from Docker Desktop (WSL2 VM) on a Windows laptop with a hybrid Intel CPU (P- and E-cores), on wall power with other software running. The VM's CPU topology differs from the real chip (WSL2 reports the "physical core" count used for ORT/torch threads). Expect variance between runs and between machines. Only ratios measured in the same run are meaningful.
- **Microsecond-scale rows** (the fraud model) mostly measure per-call overhead: the Python/pybind11 or JS↔WASM boundary, input copies, and for WASM a protobuf re-parse on every call. They say little about kernel speed.
- **WASM numbers are Node/V8**, not a browser. Browser timings will differ.
- **PyTorch** runs the `onnx2torch` graph, not hand-written PyTorch code. Conversion overhead in the generated module is part of the measurement.

## Server engine time vs HTTP round trip

`POST /infer` returns `inference_time_ms`, measured with `time.perf_counter`
around the `crucible_py.run` call only. This is the same quantity as the
`crucible-native` rows above. The latency a client sees is larger. It adds
the network, TLS, auth (JWT/API-key lookup), JSON parsing of a flat `number[]`
input (about 150k numbers for a 224×224 image), building the numpy array,
writing the inference log row to the database, and serialising the output. For
the small fraud model the HTTP round trip is dominated by these costs, not by
the engine. Do not compare `inference_time_ms` with client-side timings, or
with the table above, as if they measured the same thing.
