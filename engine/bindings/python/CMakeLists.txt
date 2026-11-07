# =============================================================================
# Crucible — pybind11 Python bindings
# =============================================================================
# Builds the `crucible_py` extension (load_model, run, check_model,
# supported_ops, get_model_info). The module is written to <build>/python
# (<build>/python/<Config> with multi-config generators):
#   PYTHONPATH=build/release/python python -c "import crucible_py"
# =============================================================================

# pybind11 is provided as a git submodule under engine/third_party/pybind11.
# It is only required when this subdirectory is processed.
set(CRUCIBLE_PYBIND11_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/pybind11)

if(NOT EXISTS "${CRUCIBLE_PYBIND11_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "pybind11 submodule not initialised.\n"
        "Run from repo root:  git submodule update --init --depth 1 engine/third_party/pybind11"
    )
endif()

# Use the modern FindPython mode for robust interpreter detection.
set(PYBIND11_FINDPYTHON ON CACHE BOOL "" FORCE)
add_subdirectory(${CRUCIBLE_PYBIND11_DIR} pybind11_build EXCLUDE_FROM_ALL)

pybind11_add_module(crucible_py
    crucible_py.cpp
)

target_link_libraries(crucible_py PRIVATE crucible::core)

# Make the .so / .pyd discoverable: emit its location after build.
set_target_properties(crucible_py PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
)

message(STATUS "Crucible: Python bindings target = crucible_py")
