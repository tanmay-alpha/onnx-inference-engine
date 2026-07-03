// =============================================================================
// Crucible C ABI implementation — Issue #15 (Rust CLI bridge).
//
// The contract lives in engine/include/crucible/c_api.h. This file
// is a thin `extern "C"` wrapper over the C++ engine: it owns the
// opaque handles, translates exceptions into CrucibleStatus codes,
// and copies data across the boundary in the right direction.
//
// Why a wrapper instead of just `extern "C"` declarations on the
// C++ engine itself?
//   * The engine (engine/include/crucible/*.hpp) is a C++ API with
//     STL types — std::string, std::vector, exceptions. None of
//     those cross the FFI boundary cleanly.
//   * The wrapper translates exceptions into status codes once, so
//     every fallible function has the same error contract.
//   * Future ABI changes (e.g. switching run_inference to use a
//     pre-allocated output arena) need to touch exactly one file.
// =============================================================================

#include "crucible/c_api.h"

#include "crucible/executor.hpp"
#include "crucible/onnx_parser.hpp"
#include "crucible/tensor.hpp"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Internal handle definitions
// ---------------------------------------------------------------------------

namespace crucible {
namespace abi_detail {

// Owned copy of a parsed Crucible model plus its IO metadata. The
// strings stored in `input_names` / `output_names` point into this
// struct's `Model::graph.input_names` storage and stay valid until
// the model is destroyed.
struct ModelHolder {
    Model          model;
    // Keep the string storage alive as long as the holder exists.
    // `Model::graph.input_names` is `std::vector<std::string>`; the
    // holders below cache the c_str() pointers.
    std::vector<const char*> input_name_ptrs;
    std::vector<const char*> output_name_ptrs;
};

}  // namespace abi_detail
}  // namespace crucible

// CrucibleModel is the opaque type declared in the header. Here we
// give it a definition that's private to this translation unit.
struct CrucibleModel {
    std::unique_ptr<crucible::abi_detail::ModelHolder> holder;
};

// ---------------------------------------------------------------------------
// Thread-local error buffer
// ---------------------------------------------------------------------------
namespace {

// Per-thread buffer. We use thread_local so a multithreaded client
// can call fallible functions from worker threads without trampling
// each other's error messages. The buffer is overwritten on each
// fallible call, so the pointer stays valid only until the next
// call from the same thread — matching the contract in the header.
thread_local std::string g_last_error;

// Map an exception type to the most specific status code we know
// about. Anything we don't recognise is bucketed as INTERNAL so the
// caller knows the failure isn't part of the documented behaviour.
CrucibleStatus classify(const std::exception& e) {
    if (dynamic_cast<const std::invalid_argument*>(&e) != nullptr) {
        return CRUCIBLE_ERR_INVALID_ARGUMENT;
    }
    if (dynamic_cast<const std::runtime_error*>(&e) != nullptr) {
        // Both parse-time and run-time errors come through here;
        // the C++ engine does not distinguish them. We bucket by
        // message prefix when we can recognise one.
        const char* what = e.what();
        if (what != nullptr && std::strstr(what, "parse") != nullptr) {
            return CRUCIBLE_ERR_PARSE;
        }
        if (what != nullptr && std::strstr(what, "io") != nullptr) {
            return CRUCIBLE_ERR_IO;
        }
        if (what != nullptr && std::strstr(what, "unsupported") != nullptr) {
            return CRUCIBLE_ERR_UNSUPPORTED;
        }
        return CRUCIBLE_ERR_RUNTIME;
    }
    return CRUCIBLE_ERR_INTERNAL;
}

// Install `e.what()` as the thread-local error message.
void store_error(const std::exception& e) {
    g_last_error = e.what();
}
void store_error(CrucibleStatus s, const std::string& what) {
    g_last_error = std::string(crucible_status_str(s)) + ": " + what;
}

}  // namespace

// ---------------------------------------------------------------------------
// Status / error
// ---------------------------------------------------------------------------

extern "C" CRUCIBLE_API const char* crucible_status_str(CrucibleStatus s) {
    switch (s) {
        case CRUCIBLE_OK:                   return "CRUCIBLE_OK";
        case CRUCIBLE_ERR_INVALID_ARGUMENT: return "CRUCIBLE_ERR_INVALID_ARGUMENT";
        case CRUCIBLE_ERR_IO:               return "CRUCIBLE_ERR_IO";
        case CRUCIBLE_ERR_PARSE:            return "CRUCIBLE_ERR_PARSE";
        case CRUCIBLE_ERR_RUNTIME:          return "CRUCIBLE_ERR_RUNTIME";
        case CRUCIBLE_ERR_UNSUPPORTED:      return "CRUCIBLE_ERR_UNSUPPORTED";
        case CRUCIBLE_ERR_INTERNAL:         return "CRUCIBLE_ERR_INTERNAL";
    }
    return "CRUCIBLE_ERR_INTERNAL";
}

extern "C" CRUCIBLE_API const char* crucible_last_error(void) {
    return g_last_error.empty() ? nullptr : g_last_error.c_str();
}

// ---------------------------------------------------------------------------
// Load / free / info
// ---------------------------------------------------------------------------

extern "C" CRUCIBLE_API CrucibleModel* crucible_load(const char* path) {
    if (path == nullptr) {
        store_error(CRUCIBLE_ERR_INVALID_ARGUMENT, "path is null");
        return nullptr;
    }
    try {
        auto m = std::make_unique<crucible::abi_detail::ModelHolder>(
            crucible::abi_detail::ModelHolder{
                crucible::load_model(path),
                {},
                {}
            }
        );

        // Cache c_str() pointers once so model_info can fill
        // CrucibleModelInfo.input_names without copying.
        m->input_name_ptrs.reserve(m->model.input_names.size());
        for (const auto& n : m->model.input_names)  m->input_name_ptrs.push_back(n.c_str());
        m->output_name_ptrs.reserve(m->model.output_names.size());
        for (const auto& n : m->model.output_names) m->output_name_ptrs.push_back(n.c_str());

        auto* handle = new CrucibleModel();
        handle->holder = std::move(m);
        return handle;
    } catch (const std::exception& e) {
        store_error(classify(e), e.what());
        return nullptr;
    } catch (...) {
        store_error(CRUCIBLE_ERR_INTERNAL, "unknown exception during load");
        return nullptr;
    }
}

extern "C" CRUCIBLE_API void crucible_free(CrucibleModel* model) {
    delete model;  // unique_ptr destructor releases the ModelHolder
}

extern "C" CRUCIBLE_API CrucibleStatus crucible_model_info(const CrucibleModel* model,
                                                          CrucibleModelInfo* out) {
    if (model == nullptr || out == nullptr) {
        store_error(CRUCIBLE_ERR_INVALID_ARGUMENT, "null model or out");
        return CRUCIBLE_ERR_INVALID_ARGUMENT;
    }
    out->abi_version       = CRUCIBLE_ABI_VERSION;
    out->num_inputs        = static_cast<int32_t>(model->holder->input_name_ptrs.size());
    out->num_outputs       = static_cast<int32_t>(model->holder->output_name_ptrs.size());
    out->num_initializers  = static_cast<int32_t>(model->holder->model.weights.size());
    out->num_nodes         = static_cast<int32_t>(model->holder->model.graph.node.size());
    out->input_names       = model->holder->input_name_ptrs.data();
    out->output_names      = model->holder->output_name_ptrs.data();
    return CRUCIBLE_OK;
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

namespace {
// Materialise a C++ engine output Tensor into caller-owned float
// buffer + shape array. The shape is heap-allocated via malloc so
// the Rust caller can free() it through libc (which is what the
// `libc` crate does on Unix). On Windows, MSVC's malloc/free pair
// is matched. We do not use `new` / `delete` here precisely to keep
// the deallocation story under the caller's control.
void tensor_to_buffers(const crucible::Tensor& src,
                       float** out_buf,
                       CrucibleTensorDesc* out_desc) {
    const auto& shape = src.shape();
    out_desc->rank  = static_cast<int32_t>(shape.size());
    out_desc->size  = src.size();
    out_desc->shape = static_cast<int64_t*>(
        std::malloc(out_desc->rank * sizeof(int64_t)));
    std::memcpy(out_desc->shape, shape.data(),
                out_desc->rank * sizeof(int64_t));
    *out_buf = static_cast<float*>(std::malloc(src.size() * sizeof(float)));
    std::memcpy(*out_buf, src.data(), src.size() * sizeof(float));
}
}  // namespace

extern "C" CRUCIBLE_API CrucibleStatus crucible_run(
    CrucibleModel* model,
    const CrucibleTensorDesc* input_descs, int32_t num_inputs,
    float** outputs,                        /* out */
    CrucibleTensorDesc* output_descs,       /* out */
    int32_t num_outputs
) {
    if (model == nullptr) {
        store_error(CRUCIBLE_ERR_INVALID_ARGUMENT, "null model");
        return CRUCIBLE_ERR_INVALID_ARGUMENT;
    }
    if (num_inputs != static_cast<int32_t>(model->holder->input_name_ptrs.size())) {
        store_error(CRUCIBLE_ERR_INVALID_ARGUMENT,
            "num_inputs does not match model");
        return CRUCIBLE_ERR_INVALID_ARGUMENT;
    }
    if (num_outputs != static_cast<int32_t>(model->holder->output_name_ptrs.size())) {
        store_error(CRUCIBLE_ERR_INVALID_ARGUMENT,
            "num_outputs does not match model");
        return CRUCIBLE_ERR_INVALID_ARGUMENT;
    }
    try {
        // Build the C++ inputs map by name.
        std::unordered_map<std::string, crucible::Tensor> cpp_inputs;
        for (int32_t i = 0; i < num_inputs; ++i) {
            const auto& desc = input_descs[i];
            if (desc.data == nullptr || desc.size < 0 ||
                (desc.rank > 0 && desc.shape == nullptr)) {
                store_error(CRUCIBLE_ERR_INVALID_ARGUMENT,
                    "input descriptor is malformed");
                return CRUCIBLE_ERR_INVALID_ARGUMENT;
            }
            std::vector<int64_t> shape(desc.shape, desc.shape + desc.rank);
            std::vector<float>   data(desc.data, desc.data + desc.size);
            cpp_inputs.emplace(
                model->holder->input_name_ptrs[i],
                crucible::Tensor(std::move(shape), std::move(data)));
        }
        // The current executor returns the FIRST output. We run
        // once per output (wasteful but correct) — this will be
        // fixed when Issue #11's Session class exposes
        // multi-output run_inference.
        if (num_outputs > 0) {
            crucible::Tensor out = crucible::run_inference(
                model->holder->model, cpp_inputs);
            tensor_to_buffers(out, &outputs[0], &output_descs[0]);
            // For any additional declared outputs we don't have a
            // multi-output path yet; surface that by leaving the
            // buffer null and rank=0 so the Rust side knows the
            // value is absent. Future fix in Issue #11's Session.
            for (int32_t i = 1; i < num_outputs; ++i) {
                output_descs[i].rank = 0;
                output_descs[i].size = 0;
                output_descs[i].shape = nullptr;
                outputs[i] = nullptr;
            }
        }
        return CRUCIBLE_OK;
    } catch (const std::exception& e) {
        store_error(classify(e), e.what());
        return CRUCIBLE_ERR_RUNTIME;
    } catch (...) {
        store_error(CRUCIBLE_ERR_INTERNAL, "unknown exception during run");
        return CRUCIBLE_ERR_INTERNAL;
    }
}

extern "C" CRUCIBLE_API void crucible_free_array(void* ptr) {
    std::free(ptr);
}
