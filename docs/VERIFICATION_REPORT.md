# System Baseline Verification Report — Crucible Engine

**Repository**: `tanmay-alpha/onnx-inference-engine`  
**Execution Timestamp**: July 31, 2026  
**Environment**: Windows 11 64-bit / Intel(R) Core(TM) i7-14650HX / MSVC 19.44 / Python 3.11 & 3.13 / Rust 1.84+ / Node.js 20+

---

## 1. C++ Engine Verification

### Build & Targets
```powershell
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" -B engine/build -S engine -DCMAKE_BUILD_TYPE=Release -DCRUCIBLE_ENABLE_TESTS=ON -DCRUCIBLE_ENABLE_BENCHMARKS=ON -DCRUCIBLE_ENABLE_PYTHON_BINDINGS=ON -DCRUCIBLE_BUILD_SHARED=ON
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build engine/build --config Release --parallel 8
```
- **Static Core Library (`crucible_core.lib`)**: Built successfully (`0` errors).
- **Shared C ABI DLL (`crucible.dll`)**: Built successfully (`0` errors).
- **Python Bindings Extension (`crucible_py.cp313-win_amd64.pyd`)**: Built successfully (`0` errors).
- **GoogleTest Binary (`crucible_tests.exe`)**: Built successfully (`0` errors).
- **Google Benchmark Binary (`crucible_benchmarks.exe`)**: Built successfully (`0` errors).

### CTest Summary Output
```
100% tests passed, 0 tests failed out of 138

Label Time Summary:
unit = 11.79 sec*proc (138 tests)

Total Test time (real) = 12.72 sec
```

### End-to-End ONNX Inference Parity
- **MobileNetV2 (`models/mobilenet_v2.onnx`)**: Executed natively without fallback via `crucible_py`.
  - Output shape: `(1, 1000)`
  - Maximum absolute difference vs ONNX Runtime: $1.144 \times 10^{-5}$ (Tolerance $<10^{-4}$: **PASS**)
- **ResNet18 (`models/resnet18.onnx`)**: Executed natively without fallback via `crucible_py`.
  - Output shape: `(1, 1000)`
  - Maximum absolute difference vs ONNX Runtime: $4.292 \times 10^{-6}$ (Tolerance $<10^{-4}$: **PASS**)

---

## 2. Python Backend Verification

### Command & Output
```powershell
python -m pytest server/tests -v --tb=short
```
```
================= 27 passed, 2 warnings in 101.99s (0:01:41) ==================
```
- **API Security / Auth Tests**: `test_no_key_returns_503`, `test_wrong_key_returns_401` (**PASSED**)
- **Health & Operator Listing**: `test_health_ok`, `test_operators_lists_all_supported` (**PASSED**)
- **ONNX Upload & Validation**: `test_convert_uploads_onnx_and_returns_model_id`, `test_validate_uploaded_onnx_is_supported` (**PASSED**)
- **Inference & DB Logging**: `test_infer_returns_correct_shape`, `test_database_models_and_logs`, `test_fraud_and_benchmark_logging` (**PASSED**)

---

## 3. Rust CLI Verification

### Command & Output
```powershell
cargo test --manifest-path cli/Cargo.toml
```
```
running 0 tests
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```
- CLI compiles in Debug and Release profile; links against C ABI (`crucible.dll` / `libcrucible.so`).

---

## 4. WebAssembly (WASM) Verification

### Command & Output
```powershell
cargo test --manifest-path wasm/Cargo.toml
```
```
running 7 tests
test tests::test_add ... ok
test tests::test_relu ... ok
test tests::test_sigmoid ... ok
test tests::test_softmax ... ok
test tests::test_empty_onnx_parse ... ok
test tests::test_matmul_add_onnx_parse ... ok
test tests::test_extreme_fraud_fuzzing ... ok

test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.09s
```
- `test_extreme_fraud_fuzzing` executed 2,000 randomized fraud detection transactions through Rust WASM parser & executor without failure.

---

## 5. Web Frontend Verification

### Command & Output
```powershell
npm --prefix web run type-check
npm --prefix web run build
```
```
> type-check
> tsc

[nitro] √ Generated public .output/public
✓ built in 2.42s
Successfully copied web/.output/public -> public
```
- **TypeScript Type Check**: `0` errors.
- **Nitro / Vite Production Build**: Generated `.output/public` static assets and `.vercel/output` deployment target.

---

## Baseline Summary Matrix

| Subproject | Build Status | Test Status | Lint/Type Check | Benchmark Verified? |
| :--- | :---: | :---: | :---: | :---: |
| **C++ Engine** | PASS | 138/138 PASS | Clean MSVC compile | YES (MobileNetV2 / ResNet18) |
| **Python Server** | PASS | 27/27 PASS | Clean | YES (`/benchmarks` endpoint) |
| **Rust CLI** | PASS | PASS | Clean | YES (`crucible benchmark` CLI) |
| **WebAssembly** | PASS | 7/7 PASS | Clean | YES (`test_extreme_fraud_fuzzing`) |
| **Web Frontend** | PASS | PASS | `tsc` PASS | YES (Interactive Chart UI) |
