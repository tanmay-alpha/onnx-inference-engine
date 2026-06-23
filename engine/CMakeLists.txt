# =============================================================================
# Crucible — ONNX Inference Engine (C++17)
# =============================================================================
# Build the core inference library, Python bindings, tests, and benchmarks.
#
# Quick start:
#   cmake --preset debug   -S engine
#   cmake --build          build/debug
#   ctest --test-dir       build/debug --output-on-failure
#
# Issue #1: this file establishes the build system. Subsequent issues add
# the Tensor class, ONNX parser, operators, and graph executor.
# =============================================================================

cmake_minimum_required(VERSION 3.27)
project(crucible
    VERSION 0.1.0
    DESCRIPTION "Zero-overhead ONNX inference engine"
    LANGUAGES CXX
)

# -----------------------------------------------------------------------------
# Global compile settings
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Crucible: no build type set, defaulting to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Project-level options
option(CRUCIBLE_ENABLE_TESTS          "Build C++ unit tests (GoogleTest)"   OFF)
option(CRUCIBLE_ENABLE_BENCHMARKS     "Build C++ micro-benchmarks"          OFF)
option(CRUCIBLE_ENABLE_PYTHON_BINDINGS "Build pybind11 Python bindings"      OFF)
option(CRUCIBLE_WARNINGS_AS_ERRORS    "Treat compiler warnings as errors"   OFF)

# -----------------------------------------------------------------------------
# Third-party dependencies (git submodules)
# -----------------------------------------------------------------------------
set(CRUCIBLE_THIRD_PARTY ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# Sanity-check that submodules are populated. add_subdirectory() will fail
# noisily if the directories are empty, but a clear upfront message helps.
foreach(submod eigen protobuf googletest google-benchmark pybind11)
    if(NOT EXISTS "${CRUCIBLE_THIRD_PARTY}/${submod}/CMakeLists.txt"
       AND NOT EXISTS "${CRUCIBLE_THIRD_PARTY}/${submod}/README.md"
       AND NOT EXISTS "${CRUCIBLE_THIRD_PARTY}/${submod}/Eigen")
        message(WARNING
            "Crucible: third_party/${submod} appears empty.\n"
            "          Run: git submodule update --init --recursive\n"
            "          (See engine/CMakeLists.txt for details.)"
        )
    endif()
endforeach()

# Eigen (header-only)
if(EXISTS "${CRUCIBLE_THIRD_PARTY}/eigen/CMakeLists.txt")
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    add_subdirectory(${CRUCIBLE_THIRD_PARTY}/eigen EXCLUDE_FROM_ALL)
endif()

# Protobuf (used by the ONNX parser in Issue #4)
if(EXISTS "${CRUCIBLE_THIRD_PARTY}/protobuf/CMakeLists.txt")
    set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(protobuf_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "" FORCE)
    add_subdirectory(${CRUCIBLE_THIRD_PARTY}/protobuf EXCLUDE_FROM_ALL)
endif()

# -----------------------------------------------------------------------------
# Core library: crucible_core
# -----------------------------------------------------------------------------
# Issue #1: only tensor.cpp is real. Other sources are intentionally absent
# until Issues #2-#10 introduce them. This keeps `cmake --build` green while
# the design is being implemented.
set(CRUCIBLE_CORE_SOURCES
    src/tensor.cpp
    src/onnx_parser.cpp
    src/ops/linear.cpp
    src/ops/activations.cpp
    src/ops/conv2d.cpp
)

add_library(crucible_core STATIC ${CRUCIBLE_CORE_SOURCES})

add_library(crucible::core ALIAS crucible_core)

target_include_directories(crucible_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Empty PUBLIC link block: third-party deps are attached conditionally below
# so that the scaffold builds even before submodules are initialised.
target_link_libraries(crucible_core PUBLIC)

if(EXISTS "${CRUCIBLE_THIRD_PARTY}/eigen/CMakeLists.txt")
    target_link_libraries(crucible_core PUBLIC Eigen3::Eigen)
endif()

if(EXISTS "${CRUCIBLE_THIRD_PARTY}/protobuf/CMakeLists.txt")
    target_link_libraries(crucible_core PUBLIC protobuf::libprotobuf)
endif()

# Compiler flags
target_compile_options(crucible_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive->
)

# Release-mode optimizations (auto-applied by CMake's RelWithDebInfo/Release)
target_compile_options(crucible_core PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:RelWithDebInfo>:-O2>
)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    # -march=native is intentionally OFF by default to keep binaries portable.
    # Enable per-build with -DCRUCIBLE_MARCH_NATIVE=ON.
    option(CRUCIBLE_MARCH_NATIVE "Enable -march=native for Release builds" OFF)
    if(CRUCIBLE_MARCH_NATIVE)
        target_compile_options(crucible_core PRIVATE $<$<CONFIG:Release>:-march=native>)
    endif()
endif()

if(CRUCIBLE_WARNINGS_AS_ERRORS)
    target_compile_options(crucible_core PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Werror>
        $<$<CXX_COMPILER_ID:MSVC>:/WX>
    )
endif()

# -----------------------------------------------------------------------------
# Optional sub-targets: tests / benchmarks / python bindings
# -----------------------------------------------------------------------------
if(CRUCIBLE_ENABLE_PYTHON_BINDINGS)
    add_subdirectory(bindings/python)
endif()

if(CRUCIBLE_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(CRUCIBLE_ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# -----------------------------------------------------------------------------
# Status banner
# -----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "============================================================")
message(STATUS "  Crucible ${PROJECT_VERSION}  (C++${CMAKE_CXX_STANDARD})")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  Compiler:          ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Tests:             ${CRUCIBLE_ENABLE_TESTS}")
message(STATUS "  Benchmarks:        ${CRUCIBLE_ENABLE_BENCHMARKS}")
message(STATUS "  Python bindings:   ${CRUCIBLE_ENABLE_PYTHON_BINDINGS}")
message(STATUS "============================================================")
message(STATUS "")
