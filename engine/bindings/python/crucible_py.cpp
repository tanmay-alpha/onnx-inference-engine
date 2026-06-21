// =============================================================================
// Crucible — pybind11 module entrypoint (scaffold)
//
// Issue #1: stub module so the build target compiles.
// Issue #12: real bindings (load_model, run, get_model_info) live here.
// =============================================================================

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(crucible_py, m) {
    m.doc() = "Crucible — zero-overhead ONNX inference engine (scaffold)";
    m.attr("__version__") = "0.1.0";
}
