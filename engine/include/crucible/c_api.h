/*
 * Crucible C ABI — Issue #15 (Rust CLI bridge).
 *
 * This header is the contract between the Crucible C++ engine and any
 * non-C++ client (the Rust CLI in cli/, the upcoming WASM module in
 * wasm/, and any future Python/C#/Go bindings). Everything that crosses
 * the boundary goes through this file.
 *
 * Design rules:
 *   1. Pure C linkage. No C++ types in the public surface; no exceptions,
 *      no STL, no overloading. The FFI is `extern "C"` and uses only
 *      POD scalars, opaque pointers, and caller-owned buffers.
 *
 *   2. Opaque handles. `CrucibleModel` and `CrucibleTensor` are forward-
 *      declared structs; clients never see their layout. This means we
 *      can change the C++ representation (e.g. switch from
 *      `std::unordered_map` to a flat array) without recompiling
 *      downstream Rust / WASM crates.
 *
 *   3. Explicit status codes. Every fallible function returns
 *      `CrucibleStatus`. The C++ side swallows exceptions, stores the
 *      message in a thread-local buffer, and exposes it via
 *      `crucible_last_error()`. Clients never have to translate C++
 *      exception text.
 *
 *   4. Caller-owned buffers for tensor data. `crucible_run` accepts
 *      raw `float*` input pointers and a caller-allocated `float*`
 *      output buffer. The ABI does not allocate on the caller's
 *      behalf for hot-path data — only for the model handle and
 *      string metadata.
 *
 *   5. Not thread-safe. The engine reads thread-local error state, so
 *      two threads sharing a model and calling `crucible_run` would race
 *      on the error string. This is documented at the top of
 *      `crucible_load`. Multithreaded use needs a per-thread model or a
 *      future thread-safety layer (Issue #19 candidate).
 *
 * Stable API guarantee: functions and structs in this header are
 * versioned. The implementation may gain new fields at the END of
 * `CrucibleModelInfo`; existing fields and their order are fixed.
 */

#ifndef CRUCIBLE_C_API_H
#define CRUCIBLE_C_API_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  ifdef CRUCIBLE_BUILDING_DLL
#    define CRUCIBLE_API __declspec(dllexport)
#  else
#    define CRUCIBLE_API __declspec(dllimport)
#  endif
#else
#  ifdef __GNUC__
#    define CRUCIBLE_API __attribute__((visibility("default")))
#  else
#    define CRUCIBLE_API
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * Version
 * ============================================================ */

/** Bumped on every backwards-incompatible ABI change. */
#define CRUCIBLE_ABI_VERSION 1

/* ============================================================
 * Status codes
 * ============================================================ */

typedef enum CrucibleStatus {
    CRUCIBLE_OK                   = 0,
    CRUCIBLE_ERR_INVALID_ARGUMENT = 1,  /* null pointer, bad shape, etc. */
    CRUCIBLE_ERR_IO               = 2,  /* file not found, read failure */
    CRUCIBLE_ERR_PARSE            = 3,  /* malformed .onnx */
    CRUCIBLE_ERR_RUNTIME          = 4,  /* graph execution failure */
    CRUCIBLE_ERR_UNSUPPORTED      = 5,  /* unknown op, etc. */
    CRUCIBLE_ERR_INTERNAL         = 6   /* unexpected C++ exception */
} CrucibleStatus;

/** Human-readable name for a status code (e.g. "CRUCIBLE_ERR_PARSE"). */
CRUCIBLE_API const char* crucible_status_str(CrucibleStatus s);

/** Last error message set by any fallible function on the calling thread.
 *  Pointer is valid until the next fallible call on the same thread.
 *  Returns NULL if no error has been recorded on this thread. */
CRUCIBLE_API const char* crucible_last_error(void);

/* ============================================================
 * Opaque model handle
 * ============================================================ */

typedef struct CrucibleModel CrucibleModel;

/** Static model metadata, queried once after load. Layout is stable
 *  up to `abi_version`; fields can only be appended. */
typedef struct CrucibleModelInfo {
    uint32_t abi_version;          /* CRUCIBLE_ABI_VERSION at compile time */
    int32_t  num_inputs;
    int32_t  num_outputs;
    int32_t  num_initializers;
    int32_t  num_nodes;
    /* Caller-allocated buffers; populated by crucible_model_info.
     * Lengths match the *count fields above. The strings are owned
     * by the model and remain valid until crucible_free. */
    const char** input_names;
    const char** output_names;
} CrucibleModelInfo;

/** Load a .onnx model from disk. Returns NULL on failure; the error
 *  message is then available via crucible_last_error().
 *
 *  NOT THREAD-SAFE: the returned model shares thread-local state with
 *  crucible_run / crucible_free. A future revision will add a
 *  per-handle error buffer; until then, drive one model per thread. */
CRUCIBLE_API CrucibleModel* crucible_load(const char* path);

/** Populate a model-info struct. The caller is responsible for sizing
 *  the input_names / output_names arrays to at least num_inputs /
 *  num_outputs entries. The struct's count fields are filled even on
 *  failure, so the caller can grow and retry. */
CRUCIBLE_API CrucibleStatus crucible_model_info(const CrucibleModel* model,
                                               CrucibleModelInfo* out);

/** Free a model returned by crucible_load. NULL is a no-op. */
CRUCIBLE_API void crucible_free(CrucibleModel* model);

/* ============================================================
 * Run
 * ============================================================ */

typedef struct CrucibleTensorDesc {
    int64_t* shape;       /* row-major dimensions; may be NULL only if rank==0 */
    int32_t  rank;        /* number of dimensions; must equal length of shape */
    int64_t  size;        /* total element count, must equal prod(shape) */
    const float* data;    /* row-major floats; length must equal size */
} CrucibleTensorDesc;

/** Run a single inference. Inputs and outputs are matched by index
 *  against the model's input_names / output_names arrays
 *  (populated via crucible_model_info).
 *
 *  Semantics:
 *    - inputs and input_descs must each have length >= num_inputs.
 *      Only the first num_inputs are read.
 *    - outputs and output_descs must each have length >= num_outputs.
 *      The runtime fills the first num_outputs; the caller is
 *      responsible for the buffers. For each output:
 *        - the shape is written to output_descs[i].shape
 *          (engine-allocated; freed by the caller via free()).
 *        - output_descs[i].size is set to the number of elements.
 *        - output_descs[i].rank is set to the number of dimensions.
 *        - outputs[i] is filled with rank-major float32 data.
 *
 *  Returns CRUCIBLE_OK on success. On failure, the output buffers
 *  are in an undefined state; the caller should not read them. */
CRUCIBLE_API CrucibleStatus crucible_run(
    CrucibleModel* model,
    const CrucibleTensorDesc* input_descs,  int32_t num_inputs,
    float** outputs,                         /* out */
    CrucibleTensorDesc* output_descs,        /* out */
    int32_t num_outputs
);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* CRUCIBLE_C_API_H */
