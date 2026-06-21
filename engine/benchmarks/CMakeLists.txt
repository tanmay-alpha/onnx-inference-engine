# =============================================================================
# Crucible — C++ micro-benchmarks (Google Benchmark)
# =============================================================================
# Issue #1: scaffold the benchmark target. Real benchmarks (bench_matmul
# for Issue #11, bench_conv2d later) drop into this directory and CMake
# picks them up automatically on next configure.
// =============================================================================

set(CRUCIBLE_BENCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/google-benchmark)

if(NOT EXISTS "${CRUCIBLE_BENCH_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "google-benchmark submodule not initialised.\n"
        "Run from repo root:  git submodule update --init --recursive"
    )
endif()

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(${CRUCIBLE_BENCH_DIR} benchmark_build EXCLUDE_FROM_ALL)

file(GLOB CRUCIBLE_BENCH_SOURCES CONFIGURE_DEPENDS "bench_*.cpp")

add_executable(crucible_benchmarks ${CRUCIBLE_BENCH_SOURCES})

target_link_libraries(crucible_benchmarks
    PRIVATE
        crucible::core
        benchmark::benchmark
)

target_include_directories(crucible_benchmarks PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Benchmarks are not part of `ctest` by default — they print to stdout and
# are run manually:  ./build/debug/engine/benchmarks/crucible_benchmarks
