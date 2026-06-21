# =============================================================================
# Crucible — pybind11 Python bindings
# =============================================================================
# Issue #1: scaffold the CMake target. Real bindings (load_model, run,
# get_model_info) land in Issue #12.
#
# After building, install the produced .so/.pyd onto PYTHONPATH:
#   PYTHONPATH=./build/debug python -c "import crucible_py"
# =============================================================================

# pybind11 is provided as a git submodule under engine/third_party/pybind11.
# It is only required when this subdirectory is processed.
set(CRUCIBLE_PYBIND11_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/pybind11)

if(NOT EXISTS "${CRUCIBLE_PYBIND11_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "pybind11 submodule not initialised.\n"
        "Run from repo root:  git submodule update --init --recursive"
    )
endif()

# Use the modern FindPython mode for robust interpreter detection.
set(PYBIND11_FINDPYTHON ON CACHE BOOL "" FORCE)
add_subdirectory(${CRUCIBLE_PYBIND11_DIR} pybind11_build EXCLUDE_FROM_ALL)

# Issue #12 will replace this stub with crucible_py.cpp containing the real
# pybind11 module definition. For Issue #1 we expose an empty `_crucible_py`
# so the build stays green.
pybind11_add_module(crucible_py
    crucible_py.cpp
)

target_link_libraries(crucible_py PRIVATE crucible::core)

# Make the .so / .pyd discoverable: emit its location after build.
set_target_properties(crucible_py PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
)

message(STATUS "Crucible: Python bindings target = crucible_py")
