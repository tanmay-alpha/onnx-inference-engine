// =============================================================================
// Crucible — pybind11 module (Issue #12)
//
// Exposes three entry points to Python:
//
//   * load_model(path)     -> Model
//                             Parse an ONNX file into a Crucible Model.
//
//   * run(model, inputs)   -> numpy.ndarray
//                             Run inference. The `inputs` argument can be
//                             either a single numpy.ndarray (used as the
//                             model's first input) or a dict mapping input
//                             name -> numpy.ndarray (for models with several
//                             named inputs, e.g. some BERT variants).
//
//   * get_model_info(model) -> dict
//                             Return a plain Python dict with the model's
//                             metadata: input_names, output_names,
//                             num_nodes, num_initializers, ops_used, etc.
//
// Why py::array::c_style | py::array::forcecast?
//   c_style: Crucible's Tensor is row-major (C-style). A Fortran-ordered
//            array would copy the data with the wrong layout, so we
//            reject it up front and ask pybind11 to error out clearly.
//   forcecast: A user passing np.float64 is silently downcast to float32
//            (our supported dtype) rather than rejected. Strict dtype
//            rejection would be more "correct" but it's a footgun for
//            notebook users who often default to float64.
//
// Why we copy on the way in (and on the way out):
//   We do NOT use the zero-copy buffer protocol because Crucible's
//   Tensor owns its storage in a std::vector<float>, and that storage
//   is moved-from when we return a Tensor by value. A numpy array
//   that points into a std::vector<float> would dangle the moment
//   the Tensor went out of scope (e.g., at the end of run()). One
//   copy on the way in (numpy -> Tensor) and one on the way out
//   (Tensor -> numpy) is the honest cost.
//
// The GIL:
//   run() copies its inputs into C++ tensors, then releases the GIL for
//   the duration of run_inference. This is safe because run_inference
//   never mutates the Model (all per-run state is local), so concurrent
//   calls on one Model from several Python threads are allowed.
// =============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "crucible/executor.hpp"
#include "crucible/onnx_parser.hpp"
#include "crucible/tensor.hpp"

#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace {

// One numpy ndarray -> one crucible::Tensor. The shape is read from
// buf.shape and copied into the Tensor's std::vector<int64_t>; the
// data pointer is copied element-by-element. This is the "honest
// copy" path described in the file header.
crucible::Tensor ndarray_to_tensor(const py::array_t<float,
                                       py::array::c_style |
                                       py::array::forcecast>& arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim < 1) {
        throw std::invalid_argument(
            "Crucible: cannot convert 0-dim numpy array to Tensor");
    }
    std::vector<int64_t> shape;
    shape.reserve(static_cast<size_t>(buf.ndim));
    for (py::ssize_t d : buf.shape) {
        if (d < 0) {
            throw std::invalid_argument(
                "Crucible: numpy array has negative axis size");
        }
        shape.push_back(static_cast<int64_t>(d));
    }
    const float* src = static_cast<const float*>(buf.ptr);
    std::vector<float> data(src, src + buf.size);
    return crucible::Tensor(std::move(shape), std::move(data));
}

// One crucible::Tensor -> one numpy ndarray. We allocate a fresh
// numpy array and copy the data over so the returned array owns its
// storage independently of the C++ Tensor (which the caller is
// about to destroy).
py::array_t<float> tensor_to_ndarray(const crucible::Tensor& t) {
    std::vector<py::ssize_t> shape;
    shape.reserve(t.shape().size());
    for (int64_t d : t.shape()) {
        shape.push_back(static_cast<py::ssize_t>(d));
    }
    auto out = py::array_t<float>(shape);
    py::buffer_info buf = out.request();
    float* dst = static_cast<float*>(buf.ptr);
    const float* src = t.data();
    const size_t n = static_cast<size_t>(t.size());
    std::copy(src, src + n, dst);
    return out;
}

// Convert a Python dict of {str: ndarray} into the
// std::unordered_map<std::string, Tensor> that run_inference expects.
std::unordered_map<std::string, crucible::Tensor>
dict_to_inputs(const py::dict& inputs) {
    std::unordered_map<std::string, crucible::Tensor> out;
    out.reserve(static_cast<size_t>(inputs.size()));
    for (auto item : inputs) {
        const std::string name = item.first.cast<std::string>();
        // Re-wrap each ndarray through ndarray_to_tensor so the type
        // contract (c_style + forcecast) is enforced uniformly.
        py::array_t<float, py::array::c_style | py::array::forcecast> arr =
            item.second.cast<
                py::array_t<float, py::array::c_style | py::array::forcecast>>();
        out.emplace(name, ndarray_to_tensor(arr));
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(crucible_py, m) {
    m.doc() = "Crucible — zero-overhead ONNX inference engine";
    m.attr("__version__") = "0.1.0";

    // -----------------------------------------------------------------
    // Session (lightweight wrapper around crucible::Model)
    // -----------------------------------------------------------------
    // We expose the Model as a py::class_ rather than a bare struct
    // so the Python type system can distinguish "a Crucible model"
    // from "a dict that happens to look like one". The class is
    // non-copyable on the C++ side (crucible::Model owns an
    // unordered_map of Tensors), so we register a holder that
    // matches: `std::unique_ptr<crucible::Model>`.
    py::class_<crucible::Model>(m, "Model")
        .def(py::init([](const std::string& path) {
                 return std::make_unique<crucible::Model>(
                     crucible::load_model(path));
             }),
             py::arg("path"),
             "Load an ONNX model from `path` and return a Model wrapper.\n"
             "Throws RuntimeError on I/O failure or malformed ONNX.\n"
             "Example:\n"
             "  >>> model = crucible_py.load_model('mobilenet_v2.onnx')")
        .def_property_readonly("input_names",
            [](const crucible::Model& m) {
                return m.input_names;
            })
        .def_property_readonly("output_names",
            [](const crucible::Model& m) {
                return m.output_names;
            });

    // -----------------------------------------------------------------
    // Free functions
    // -----------------------------------------------------------------
    m.def("load_model",
        [](const std::string& path) {
            return std::make_unique<crucible::Model>(
                crucible::load_model(path));
        },
        py::arg("path"),
        "Parse `path` (an .onnx file) and return a Crucible Model.\n"
        "Equivalent to the Model(path) constructor; provided as a\n"
        "module-level function for the typical load-then-run idiom.");

    // run(model, inputs) — two overloads.
    //
    // 1) run(model, np.ndarray) — use the array as the model's FIRST
    //    input. This is the common case (ImageNet classifiers have
    //    a single input named "data") and matches the AC exactly.
    //
    // 2) run(model, dict) — name -> array. Required for multi-input
    //    models (BERT-style encoders, multi-tower networks).
    //
    // We register the dict overload FIRST so pybind11's overload
    // resolution prefers dict over ndarray when a dict is passed.
    m.def("run",
        [](const crucible::Model& model, const py::dict& inputs) {
            auto tensor_inputs = dict_to_inputs(inputs);
            crucible::Tensor out;
            {
                // The Model is read-only during a run and every input has
                // been copied out of Python memory, so other Python threads
                // may proceed while the graph executes.
                py::gil_scoped_release release;
                out = crucible::run_inference(model, tensor_inputs);
            }
            return tensor_to_ndarray(out);
        },
        py::arg("model"), py::arg("inputs"),
        "Run inference with a dict of named numpy array inputs.\n"
        "Returns the value of the model's FIRST output as a numpy array.");

    m.def("run",
        [](const crucible::Model& model,
           const py::array_t<float, py::array::c_style |
                              py::array::forcecast>& arr) {
            if (model.input_names.empty()) {
                throw std::runtime_error(
                    "Crucible.run: model has no input names");
            }
            std::unordered_map<std::string, crucible::Tensor> inputs;
            inputs.reserve(1);
            inputs.emplace(model.input_names.front(), ndarray_to_tensor(arr));
            crucible::Tensor out;
            {
                py::gil_scoped_release release;
                out = crucible::run_inference(model, inputs);
            }
            return tensor_to_ndarray(out);
        },
        py::arg("model"), py::arg("inputs"),
        "Run inference with a single numpy array input.\n"
        "The array is bound to the model's first declared input.");

    m.def("supported_ops",
        []() { return crucible::supported_ops(); },
        "Sorted list of ONNX op_types the native executor implements.");

    m.def("check_model",
        [](const crucible::Model& model) { return crucible::check_model(model); },
        py::arg("model"),
        "Static support check. Returns a list of human-readable reasons the\n"
        "model cannot run on Crucible (empty list = supported).");

    m.def("get_model_info",
        [](const crucible::Model& model) {
            // Build the dict on the Python side so the caller gets a
            // genuine py::dict (not a C++ map). py::dict preserves
            // insertion order in Python 3.7+, which makes the
            // repr stable for assertions in tests.
            py::dict info;
            info["input_names"]      = model.input_names;
            info["output_names"]     = model.output_names;
            info["opset"]            = model.graph.opset;
            info["num_nodes"]        = static_cast<int64_t>(model.graph.node.size());
            info["num_initializers"] =
                static_cast<int64_t>(model.graph.weights.size());
            info["num_int_initializers"] =
                static_cast<int64_t>(model.graph.int_initializers.size());

            // Declared IO types: shape uses -1 for symbolic/unknown dims;
            // shape is None when the model declares no shape at all.
            auto value_infos = [](const std::vector<crucible::ValueInfo>& vis) {
                py::list out;
                for (const auto& vi : vis) {
                    py::dict d;
                    d["name"] = vi.name;
                    d["elem_type"] = vi.elem_type;
                    d["shape"] = vi.has_shape ? py::cast(vi.shape) : py::none();
                    out.append(d);
                }
                return out;
            };
            info["inputs"]  = value_infos(model.graph.inputs);
            info["outputs"] = value_infos(model.graph.outputs);

            // ops_used: op_type strings in first-appearance order.
            std::set<std::string> seen;
            std::vector<std::string> ops;
            ops.reserve(model.graph.node.size());
            for (const auto& node : model.graph.node) {
                if (seen.insert(node.op_type).second) {
                    ops.push_back(node.op_type);
                }
            }
            info["ops_used"] = ops;
            return info;
        },
        py::arg("model"),
        "Return a dict of model metadata: input_names, output_names, opset,\n"
        "num_nodes, num_initializers, num_int_initializers, inputs/outputs\n"
        "(list of {name, elem_type, shape}), ops_used (first-appearance order).");
}