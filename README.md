# Crucible

> **Raw models. Forged into production.**
> A from-scratch ONNX inference engine in C++17. MobileNetV2 in 14ms on CPU. No Python at runtime.

[![Status](https://img.shields.io/badge/status-M1%20in%20progress-yellow)](./ENGINEERING_PLAN.md)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/std/the-standard)
[![CMake](https://img.shields.io/badge/CMake-3.27+-064F8C?logo=cmake&logoColor=white)](https://cmake.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#)

---

## What is this?

Crucible parses ONNX protobuf files, runs them through a graph executor written in C++17, and exposes the same engine to Python (pybind11), a Rust CLI, and a WebAssembly build. The same job TensorFlow Lite and ONNX Runtime do — built by one engineer to understand every line.

See **[ENGINEERING_PLAN.md](./ENGINEERING_PLAN.md)** for the full architecture, API contracts, and 30-day implementation roadmap.

---

## Quick start (after `git clone`)

```bash
# 1. Pull third-party submodules (~500 MB: Eigen, protobuf, googletest, google-benchmark, pybind11)
git submodule update --init --recursive

# 2. Configure & build
cmake --preset debug   -S engine
cmake --build          build/debug -j

# 3. Run tests
ctest --test-dir build/debug --output-on-failure
```

The `engine/build/debug/crucible_tests` binary should print a green `OK`. That's the Issue #1 acceptance criterion: `cmake --build engine/build/debug` succeeds with zero errors.

---

## Prerequisites

| Tool | Version | Install (Windows) | Install (Linux) |
|------|---------|-------------------|-----------------|
| Git | 2.40+   | [git-scm.com](https://git-scm.com) | `apt install git` |
| C++ compiler | C++17-capable | **Visual Studio 2022 Build Tools** ("Desktop development with C++") *or* MinGW-w64 13+ | `g++ 13` / `clang 16` |
| CMake | 3.27+   | `winget install Kitware.CMake` | `apt install cmake` |
| Ninja *(optional, faster)* | 1.11+ | `winget install Ninja-build.Ninja` | `apt install ninja-build` |
| Python *(for bindings)* | 3.11+ | [python.org](https://python.org) | `apt install python3.11-dev` |
| Rust *(for CLI / WASM)* | 1.78+ | [rustup.rs](https://rustup.rs) | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |

**Verify everything is installed:**
```bash
cmake --version       # 3.27+
ninja --version       # 1.11+ (optional)
g++ --version         # 13+ on Linux; cl.exe on Windows
cargo --version       # 1.78+  (only needed for Issues #15+)
python --version      # 3.11+  (only needed for Issues #12+)
```

If CMake cannot find a generator automatically, set one explicitly:
```bash
# Use Ninja (preferred — fastest)
cmake -S engine -B build/debug -G Ninja -DCMAKE_BUILD_TYPE=Debug

# Fall back to system default (Unix Makefiles / MinGW Makefiles / MSBuild)
cmake -S engine -B build/debug
```

---

## Repository layout

```
crucible/
├── engine/                    # C++17 core (this is the heart)
│   ├── include/crucible/      # Public headers (tensor.hpp, model.hpp, …)
│   ├── src/                   # Implementation .cpp files
│   ├── bindings/python/       # pybind11 module → importable as `crucible_py`
│   ├── tests/                 # GoogleTest unit tests
│   ├── benchmarks/            # Google Benchmark micro-benchmarks
│   ├── third_party/           # Git submodules (Eigen, protobuf, …)
│   └── CMakeLists.txt         # Root engine build script
├── cli/                       # Rust CLI (Issue #15)
├── wasm/                      # Rust → WebAssembly (Issue #16)
├── server/                    # Python FastAPI (Issue #13)
├── web/                       # Next.js 15 dashboard (Issues #17, #18)
├── benchmarks/                # Python benchmark scripts (Issue #14)
├── models/                    # Sample ONNX models (MobileNetV2, ResNet18)
├── infra/                     # Docker Compose, nginx
├── .github/workflows/         # CI: ci-engine, ci-rust, ci-server, ci-web
├── CMakePresets.json          # Top-level presets (debug / release)
├── .gitmodules                # Third-party submodules
└── ENGINEERING_PLAN.md        # Source of truth for every decision
```

---

## Build matrix

| Preset        | Generator      | Build type | Use when                                |
|---------------|----------------|------------|-----------------------------------------|
| `debug`       | Ninja          | Debug      | Daily development, debugging with gdb   |
| `release`     | Ninja          | Release    | Performance work, benchmark runs        |

The presets live in [CMakePresets.json](./CMakePresets.json) and inherit sane defaults (compile_commands.json, ninja, etc.). Override per-build with `cmake -S engine -B build/x -G "Visual Studio 17 2022"` if you prefer an IDE generator.

---

## Development workflow

1. `git checkout -b issue/N-short-name` (e.g. `issue/2-tensor-class`)
2. Implement, following the contract in [ENGINEERING_PLAN.md §4](./ENGINEERING_PLAN.md)
3. `cmake --build build/debug -j`  (CMake's `CONFIGURE_DEPENDS` picks up new files)
4. `ctest --test-dir build/debug --output-on-failure`
5. `git add -A && git commit -m "feat(#N): …"`
6. Open a PR, get review, merge to main

---

## Issue #1 status

> **CMake project scaffold + git submodules** — ✅ merged.
> ✅ `engine/CMakeLists.txt` builds `crucible_core` (STATIC).
> ✅ `crucible::core` is the linked alias used by tests, benchmarks, and Python bindings.
> ✅ `debug` / `release` presets in [CMakePresets.json](./CMakePresets.json).
> ✅ Stub `Tensor` class in [engine/include/crucible/tensor.hpp](./engine/include/crucible/tensor.hpp) (real impl in Issue #2).

## Issue #2 status

> **Tensor class — shape, data, indexing** — ✅ merged.
> ✅ Default ctor, fill ctor, data ctor.
> ✅ `data()`, `shape()`, `rank()`, `size()`.
> ✅ `at()` with bounds checking, row-major layout.
> ✅ 16 GoogleTest cases in [engine/tests/test_tensor.cpp](./engine/tests/test_tensor.cpp) — acceptance criterion met.

## Issue #3 status

> **Tensor operations — reshape, flatten, print** — ✅ merged.
> ✅ `reshape(new_shape) const` — row-major data preserved, throws on size mismatch / negative dim.
> ✅ `flatten() const` — 1-D view of all elements, returns shape `{N}`.
> ✅ `print(max_elements = 10) const` — writes to stdout; `print_to(ostream, …)` for testability. Truncates with `…` past the limit.
> ✅ 22 GoogleTest cases in [engine/tests/test_tensor_ops.cpp](./engine/tests/test_tensor_ops.cpp) — well over the 4 required by the plan.
> ✅ Bug fix: zero is now a valid dimension (matches NumPy). Negative dims still rejected.

Up next: Issue #4 (ONNX protobuf parser).

---

## License

MIT. See LICENSE (to be added in a later issue).
