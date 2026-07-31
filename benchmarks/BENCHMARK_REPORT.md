# Crucible C++ Inference Engine Benchmark Report

## System Specification
- **CPU Model**: Intel(R) Core(TM) i7-14650HX
- **OS**: Microsoft Windows 11 Home Single Language (64-bit)
- **Compiler**: MSVC 19.44.35228.0 (Visual Studio 2022 Build Tools)
- **Build Configuration**: Release (`/O2` / `/O3`, `/W4`, C++17)
- **Execution Target**: Single-threaded C++ Graph Executor vs ONNX Runtime (CPUExecutionProvider)

## Benchmark Setup
- **Warmup Iterations**: 10
- **Timed Iterations**: 100
- **Correctness Tolerance**: $10^{-4}$ ($0.0001$) absolute error threshold against ONNX Runtime reference outputs

---

## Results Summary

### 1. MobileNetV2 (`models/mobilenet_v2.onnx`)
- **Input Shape**: `[1, 3, 224, 224]`
- **Correctness Check**: `PASS` (Max Abs Diff: $1.144 \times 10^{-5}$, Mean Abs Diff: $2.655 \times 10^{-6}$)

| Engine | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Min (ms) | Max (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ONNX Runtime 1.28 (CPU)** | **5.09** | **4.84** | **6.87** | **9.16** | **3.78** | **9.89** |
| **Crucible C++ Engine** | **895.85** | **821.68** | **1398.03** | **2037.73** | **770.68** | **2733.41** |

---

### 2. ResNet18 (`models/resnet18.onnx`)
- **Input Shape**: `[1, 3, 224, 224]`
- **Correctness Check**: `PASS` (Max Abs Diff: $4.292 \times 10^{-6}$, Mean Abs Diff: $1.043 \times 10^{-6}$)

| Engine | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Min (ms) | Max (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ONNX Runtime 1.28 (CPU)** | **11.71** | **11.39** | **14.19** | **15.10** | **10.41** | **15.80** |
| **Crucible C++ Engine** | **473.75** | **471.21** | **514.37** | **576.00** | **432.91** | **610.45** |
