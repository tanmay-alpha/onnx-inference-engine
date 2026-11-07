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

Run: `2026-11-07T19:14:51+00:00`, commit `e6d815e`, 20 warmup + 200 timed iterations.
Host: Intel(R) Core(TM) i7-14650HX, 24 logical CPUs, Debian GNU/Linux 12 (bookworm) (kernel 6.18.33.2-microsoft-standard-WSL2), Python 3.11.18, c++ (Debian 12.2.0-14+deb12u1) 12.2.0,
onnxruntime 1.31.1, torch 2.14.1+cpu. Docker Desktop (WSL2 backend) container on a Windows 11 laptop.

### Headline: CNN median latency, batch 1 (lower is better)

| Model | crucible-native ms | ORT 1 thread ms | Crucible vs ORT-1t | ORT default ms | Crucible vs ORT default | crucible-native-avx2 ms (host-specific) | avx2 vs ORT-1t |
|---|---:|---:|---:|---:|---:|---:|---:|
| resnet18 | 113.4 | 30.1 | **3.8× slower** | 7.3 | 15.5× slower | 46.4 | 1.5× slower |
| mobilenet_v2 | 55.6 | 8.3 | **6.7× slower** | 2.4 | 23.4× slower | 38.1 | 4.6× slower |

On the CNNs, Crucible is **slower than ONNX Runtime**. With one thread on
both sides, it takes 3.8× as long as ORT on ResNet18 and
6.7× as long on MobileNetV2. Against ORT's default thread pool the gap is
much larger. Building with `-march=native` narrows the gap. That build is
host-specific and is not what ships. Every Crucible output matches ORT within
about 1e-5, with the same top-1 class.

Ratios change from run to run. The previous run (commit `d44218f-dirty`)
measured 4.0× and 6.0× against ORT with one thread, so expect these ratios to
move by roughly ±10% between runs on this laptop.

On the 7-feature fraud model, every runtime finishes in a few microseconds, so
the numbers mostly measure per-call overhead. Crucible's per-call path is
lighter than ORT's Python binding here. That says nothing about kernel speed.

### Full table

Ratios are `median ÷ ORT median` for the same model and input: 2.00 means
twice as slow, 0.50 means twice as fast. "Load ms" is defined in the
Methodology section.

| Model | Input | Runtime | Threads | Median ms | Mean ms | p95 ms | Min ms | Std ms | Throughput (inf/s) | Load ms | Max abs diff vs ORT | Top-1 match | median ÷ ORT-1t median | median ÷ ORT median |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|---:|---:|
| resnet18 | 1×3×224×224 | onnxruntime-cpu | 12 | 7.314 | 7.262 | 8.057 | 5.85 | 0.5448 | 136.7 | 573.3 | — | — | 0.24 | 1.00 |
| resnet18 | 1×3×224×224 | onnxruntime-cpu-1t | 1 | 30.13 | 30.44 | 33.52 | 28.02 | 1.603 | 33.2 | 348.7 | 0 | yes | 1.00 | 4.12 |
| resnet18 | 1×3×224×224 | crucible-native | 1 | 113.4 | 114.3 | 122.2 | 105.1 | 4.05 | 8.8 | 353.7 | 7.15e-06 | yes | 3.76 | 15.50 |
| resnet18 | 1×3×224×224 | crucible-native-avx2 | 1 | 46.39 | 47.24 | 52.01 | 44.01 | 2.317 | 21.6 | 336.3 | 6.68e-06 | yes | 1.54 | 6.34 |
| resnet18 | 1×3×224×224 | pytorch-cpu | 12 | 15.11 | 15.17 | 18.97 | 11.6 | 2.112 | 66.2 | 928.4 | 4.05e-06 | yes | 0.50 | 2.07 |
| resnet18 | 1×3×224×224 | crucible-wasm | 1 | unsupported | | | | | | | | | | |
| mobilenet_v2 | 1×3×224×224 | onnxruntime-cpu | 12 | 2.375 | 2.373 | 2.821 | 1.851 | 0.2701 | 421.1 | 157 | — | — | 0.29 | 1.00 |
| mobilenet_v2 | 1×3×224×224 | onnxruntime-cpu-1t | 1 | 8.254 | 8.444 | 9.887 | 7.501 | 0.7671 | 121.2 | 113.5 | 0 | yes | 1.00 | 3.48 |
| mobilenet_v2 | 1×3×224×224 | crucible-native | 1 | 55.61 | 56.11 | 62.28 | 51.03 | 2.998 | 18.0 | 95.56 | 1.41e-05 | yes | 6.74 | 23.42 |
| mobilenet_v2 | 1×3×224×224 | crucible-native-avx2 | 1 | 38.1 | 38.74 | 43.06 | 34.37 | 2.412 | 26.2 | 103.2 | 1.39e-05 | yes | 4.62 | 16.05 |
| mobilenet_v2 | 1×3×224×224 | pytorch-cpu | 12 | 14.12 | 14.67 | 18.74 | 10.89 | 2.554 | 70.8 | 259.2 | 1.22e-05 | yes | 1.71 | 5.95 |
| mobilenet_v2 | 1×3×224×224 | crucible-wasm | 1 | unsupported | | | | | | | | | | |
| fraud_detector | 1×7 | onnxruntime-cpu | 12 | 0.01187 | 0.01269 | 0.01487 | 0.01124 | 0.003698 | 84,253 | 7.844 | — | — | 3.14 | 1.00 |
| fraud_detector | 1×7 | onnxruntime-cpu-1t | 1 | 0.003774 | 0.00383 | 0.004356 | 0.00325 | 0.000485 | 264,971 | 5.122 | 0 | — | 1.00 | 0.32 |
| fraud_detector | 1×7 | crucible-native | 1 | 0.002012 | 0.00212 | 0.002175 | 0.001779 | 0.001351 | 497,018 | 2.474 | 9.31e-09 | — | 0.53 | 0.17 |
| fraud_detector | 1×7 | crucible-native-avx2 | 1 | 0.001864 | 0.002097 | 0.002352 | 0.001777 | 0.001876 | 536,481 | 2.349 | 9.31e-09 | — | 0.49 | 0.16 |
| fraud_detector | 1×7 | pytorch-cpu | 12 | 0.01282 | 0.01307 | 0.01612 | 0.01115 | 0.001885 | 78,034 | 14.27 | 9.31e-09 | — | 3.40 | 1.08 |
| fraud_detector | 1×7 | crucible-wasm | 1 | 0.01302 | 0.01601 | 0.02268 | 0.01142 | 0.01193 | 76,817 | 2.527 | 9.31e-09 | — | 3.45 | 1.10 |
| fraud_detector | 64×7 | onnxruntime-cpu | 12 | 0.006487 | 0.006136 | 0.008727 | 0.003951 | 0.002237 | 9,865,125 | 7.401 | — | — | 1.55 | 1.00 |
| fraud_detector | 64×7 | onnxruntime-cpu-1t | 1 | 0.004173 | 0.004445 | 0.006234 | 0.003601 | 0.001737 | 15,334,851 | 5.979 | 0 | — | 1.00 | 0.64 |
| fraud_detector | 64×7 | crucible-native | 1 | 0.002398 | 0.002679 | 0.002893 | 0.002252 | 0.001965 | 26,688,906 | 2.799 | 5.59e-08 | — | 0.57 | 0.37 |
| fraud_detector | 64×7 | crucible-native-avx2 | 1 | 0.002608 | 0.0028 | 0.003318 | 0.00251 | 0.001217 | 24,535,174 | 2.576 | 5.59e-08 | — | 0.62 | 0.40 |
| fraud_detector | 64×7 | pytorch-cpu | 12 | 0.0133 | 0.01357 | 0.01505 | 0.01139 | 0.001905 | 4,810,945 | 13.91 | 5.96e-08 | — | 3.19 | 2.05 |
| fraud_detector | 64×7 | crucible-wasm | 1 | 0.01899 | 0.02406 | 0.03379 | 0.01594 | 0.01924 | 3,369,751 | 3.165 | 5.59e-08 | — | 4.55 | 2.93 |

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
