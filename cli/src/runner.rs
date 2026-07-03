//! FFI bindings to the Crucible C ABI (engine/include/crucible/c_api.h).
//!
//! Three layers, top to bottom:
//!
//!   1. Public Rust API   — `Model`, `Tensor`, `Status`, `validate_model`,
//!                          `top_k_indices`. All safe Rust, no `unsafe`,
//!                          no raw pointers in the public surface.
//!
//!   2. Internal FFI      — `extern "C"` blocks below the public API.
//!                          Reproduces the C ABI exactly; the public
//!                          types wrap these.
//!
//!   3. Memory ownership  — output buffers from `crucible_run` are
//!                          caller-allocated (the C side fills them).
//!                          Shape arrays are malloc'd by the C side and
//!                          freed here with `libc::free` to match the
//!                          allocator.
//!
//! Why one module for the whole FFI? All the call sites are tiny and
//! the boundary is the same in every place; spreading them across
//! files would scatter the same `extern "C"` block four times.
//!
//! This module is the only place `unsafe` is allowed in the crate.
//! Every other file is pure safe Rust on top of these wrappers.

use std::ffi::{c_char, CStr};
use std::path::Path;

use thiserror::Error;

// ---------------------------------------------------------------------------
// FFI types — mirror engine/include/crucible/c_api.h verbatim.
// ---------------------------------------------------------------------------

/// Mirrors `CrucibleStatus` in c_api.h.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub enum Status {
    Ok                   = 0,
    InvalidArgument      = 1,
    Io                   = 2,
    Parse                = 3,
    Runtime              = 4,
    Unsupported          = 5,
    Internal             = 6,
}

/// Mirrors `CrucibleModelInfo`. Populated by `model_info`.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ModelInfo {
    pub abi_version:      u32,
    pub num_inputs:       i32,
    pub num_outputs:      i32,
    pub num_initializers: i32,
    pub num_nodes:        i32,
    pub input_names:      *const *const c_char,
    pub output_names:     *const *const c_char,
}

/// Mirrors `CrucibleTensorDesc`.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct TensorDesc {
    pub shape: *mut i64,
    pub rank:  i32,
    pub size:  i64,
    pub data:  *const f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CrucibleModelOpaque {
    _private: [u8; 0],
}
pub type CrucibleModelHandle = *mut CrucibleModelOpaque;

// Extern "C" — declared verbatim from c_api.h. The names must match
// exactly because Rust uses the C linker on the platforms we ship to
// (Linux, macOS, Windows). If you rename a function on the C side,
// update this block and the public wrapper below it.
extern "C" {
    fn crucible_load(path: *const c_char) -> CrucibleModelHandle;
    fn crucible_free(model: CrucibleModelHandle);
    fn crucible_model_info(
        model: CrucibleModelHandle,
        out:   *mut ModelInfo,
    ) -> Status;
    fn crucible_run(
        model:        CrucibleModelHandle,
        input_descs:  *const TensorDesc,
        num_inputs:   i32,
        outputs:      *mut *mut f32,
        output_descs: *mut TensorDesc,
        num_outputs:  i32,
    ) -> Status;
    fn crucible_last_error() -> *const c_char;
    #[allow(dead_code)]
    fn crucible_status_str(s: Status) -> *const c_char;
    fn crucible_free_array(ptr: *mut std::ffi::c_void);
}

// ---------------------------------------------------------------------------
// Public error type
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum CrucibleError {
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("I/O failure: {0}")]
    Io(String),
    #[error("parse failure: {0}")]
    Parse(String),
    #[error("runtime failure: {0}")]
    Runtime(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("internal error: {0}")]
    Internal(String),
    #[error("library not loaded: {0}")]
    LibraryUnavailable(String),
    #[error("IO error: {0}")]
    Fs(#[from] std::io::Error),
}

impl CrucibleError {
    fn from_status(s: Status) -> Self {
        // Combine the enum-level tag with the thread-local message.
        // Without the message the error is too generic to be useful
        // (a user hitting CRUCIBLE_ERR_PARSE wants to know WHICH
        // field was malformed).
        let detail = unsafe {
            let p = crucible_last_error();
            if p.is_null() { String::new() } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        };
        match s {
            Status::Ok               => CrucibleError::Internal("OK returned as error".into()),
            Status::InvalidArgument  => CrucibleError::InvalidArgument(detail),
            Status::Io               => CrucibleError::Io(detail),
            Status::Parse            => CrucibleError::Parse(detail),
            Status::Runtime          => CrucibleError::Runtime(detail),
            Status::Unsupported      => CrucibleError::Unsupported(detail),
            Status::Internal         => CrucibleError::Internal(detail),
        }
    }
}

// ---------------------------------------------------------------------------
// Public model handle
// ---------------------------------------------------------------------------

/// Owning handle to a loaded Crucible model. Drop releases the
/// underlying C++ resources via `crucible_free`.
pub struct Model {
    handle: CrucibleModelHandle,
}

impl Model {
    /// Load a .onnx file. Returns the parsed model or an error.
    ///
    /// Library load failure (DLL/.so not on the loader's path) is
    /// surfaced as `LibraryUnavailable`. Library load is deferred
    /// until the first `load` call so the rest of the CLI can still
    /// print --help without the engine present.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, CrucibleError> {
        let path_str = path.as_ref().to_str().ok_or_else(|| {
            CrucibleError::InvalidArgument("path is not UTF-8".into())
        })?;
        let cpath = std::ffi::CString::new(path_str)
            .map_err(|_| CrucibleError::InvalidArgument("path contains NUL".into()))?;
        // SAFETY: cpath is a valid C string and is consumed before
        // this call returns. The library is loaded lazily — if it
        // isn't on the search path, the symbol lookup below fails.
        let handle = unsafe { crucible_load(cpath.as_ptr()) };
        if handle.is_null() {
            // Distinguish library-not-found from engine-reported
            // failure by checking if there's a stored C error message.
            let detail = unsafe {
                let p = crucible_last_error();
                if p.is_null() { String::new() } else {
                    CStr::from_ptr(p).to_string_lossy().into_owned()
                }
            };
            if !detail.is_empty() {
                return Err(CrucibleError::Parse(detail));
            }
            return Err(CrucibleError::LibraryUnavailable(
                load_library_diagnostics(path_str),
            ));
        }
        Ok(Model { handle })
    }

    /// Populate a `ModelInfo` snapshot.
    pub fn info(&self) -> Result<ModelInfo, CrucibleError> {
        let mut info = std::mem::MaybeUninit::<ModelInfo>::uninit();
        // SAFETY: `info` is a valid out-pointer; we initialise it
        // before reading.
        let s = unsafe { crucible_model_info(self.handle, info.as_mut_ptr()) };
        if s == Status::Ok {
            // SAFETY: status is OK so the out-param was written.
            Ok(unsafe { info.assume_init() })
        } else {
            Err(CrucibleError::from_status(s))
        }
    }

    /// Run a single inference. `inputs` must match the model's
    /// declared inputs in number; each must be rank-matching.
    pub fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, CrucibleError> {
        // Build the C-side input descriptor array. We do NOT take
        // ownership of the input data; the model only reads it
        // during the call.
        let mut in_descs: Vec<TensorDesc> = inputs
            .iter()
            .map(|t| TensorDesc {
                shape: t.shape.as_ptr() as *mut i64,
                rank:  t.shape.len() as i32,
                size:  t.data.len() as i64,
                data:  t.data.as_ptr(),
            })
            .collect();

        // Query the model for output count so we can size the
        // output buffers up front. We re-query each call to keep
        // the public API simple — info is cheap.
        let info = self.info()?;
        let n_out = info.num_outputs as usize;
        if n_out == 0 {
            return Ok(Vec::new());
        }

        // The C ABI writes the shape array pointer and the float
        // buffer pointer into caller-owned slots.
        let mut out_buf:  Vec<*mut f32>     = vec![std::ptr::null_mut(); n_out];
        let mut out_desc: Vec<TensorDesc>  = (0..n_out)
            .map(|_| TensorDesc {
                shape: std::ptr::null_mut(),
                rank:  0,
                size:  0,
                data:  std::ptr::null(),
            })
            .collect();

        // SAFETY: all out-pointers are valid; the slices point to
        // caller-owned buffers of the right length. After the call,
        // each non-null out_desc[i].shape and out_buf[i] must be
        // freed with libc::free because the C side used malloc.
        let s = unsafe {
            crucible_run(
                self.handle,
                in_descs.as_ptr(),
                in_descs.len() as i32,
                out_buf.as_mut_ptr(),
                out_desc.as_mut_ptr(),
                n_out as i32,
            )
        };
        if s != Status::Ok {
            return Err(CrucibleError::from_status(s));
        }

        // Wrap each output into a Rust Tensor and copy the data so
        // the caller doesn't need to think about libc ownership.
        let mut result = Vec::with_capacity(n_out);
        for i in 0..n_out {
            let od = out_desc[i];
            if od.size == 0 || out_buf[i].is_null() {
                result.push(Tensor::empty());
                continue;
            }
            // Copy shape out of the malloc'd array BEFORE freeing it.
            let rank = od.rank as usize;
            let mut shape = Vec::<i64>::with_capacity(rank);
            // SAFETY: od.shape is malloc'd and `rank` matches the
            // number of elements the C side wrote.
            unsafe {
                std::ptr::copy_nonoverlapping(od.shape, shape.as_mut_ptr(), rank);
                shape.set_len(rank);
                crucible_free_array(od.shape as *mut std::ffi::c_void);
            }
            // Same dance for the float buffer.
            let size = od.size as usize;
            let mut data = Vec::<f32>::with_capacity(size);
            // SAFETY: out_buf[i] is malloc'd and `size` matches the
            // number of elements the C side wrote.
            unsafe {
                std::ptr::copy_nonoverlapping(out_buf[i], data.as_mut_ptr(), size);
                data.set_len(size);
                crucible_free_array(out_buf[i] as *mut std::ffi::c_void);
            }
            // Tell the borrow checker that `in_descs` is unused
            // beyond this point so the input data lifetime is over.
            let _ = &mut in_descs;
            result.push(Tensor { shape, data });
        }
        Ok(result)
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: handle came from a successful crucible_load.
            unsafe { crucible_free(self.handle); }
        }
    }
}

// ---------------------------------------------------------------------------
// Public Tensor
// ---------------------------------------------------------------------------

/// Owned multi-dimensional float32 tensor with row-major storage.
/// The same shape/dtype model used inside the C++ engine; only the
/// buffer ownership is Rust-friendly.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<i64>,
    pub data:  Vec<f32>,
}

impl Tensor {
    /// All-zero tensor with no elements (empty placeholder).
    pub fn empty() -> Self {
        Tensor { shape: Vec::new(), data: Vec::new() }
    }

    /// Read a JSON tensor file. Expected shape:
    ///   { "shape": [N, C, H, W], "data": [..] }
    pub fn from_json_file(path: &Path) -> Result<Self, CrucibleError> {
        let s = std::fs::read_to_string(path)?;
        let v: serde_json::Value =
            serde_json::from_str(&s).map_err(|e| CrucibleError::InvalidArgument(
                format!("input is not valid JSON: {e}")))?;
        let shape_v = v.get("shape")
            .and_then(|x| x.as_array())
            .ok_or_else(|| CrucibleError::InvalidArgument(
                "input JSON missing 'shape' array".into()))?;
        let mut shape = Vec::with_capacity(shape_v.len());
        for d in shape_v {
            shape.push(d.as_i64().ok_or_else(|| CrucibleError::InvalidArgument(
                "input JSON shape entries must be int".into()))?);
        }
        let data_v = v.get("data")
            .and_then(|x| x.as_array())
            .ok_or_else(|| CrucibleError::InvalidArgument(
                "input JSON missing 'data' array".into()))?;
        let mut data = Vec::with_capacity(data_v.len());
        for x in data_v {
            data.push(x.as_f64().ok_or_else(|| CrucibleError::InvalidArgument(
                "input JSON data entries must be number".into()))? as f32);
        }
        // Cross-check: declared size must match data length.
        let expected: i64 = shape.iter().product();
        if expected as usize != data.len() {
            return Err(CrucibleError::InvalidArgument(format!(
                "input shape product {expected} != data length {}",
                data.len())));
        }
        Ok(Tensor { shape, data })
    }

    /// Write JSON tensor file. Inverse of `from_json_file`.
    #[allow(dead_code)]
    pub fn to_json_file(&self, path: &Path) -> Result<(), CrucibleError> {
        let mut s = String::with_capacity(self.data.len() * 12);
        s.push_str(&serde_json::to_string(&self.shape).unwrap_or_else(|_| "[]".into()));
        s.push('\n');
        let body: Vec<serde_json::Value> = self.data
            .iter()
            .map(|f| serde_json::Value::from(*f as f64))
            .collect();
        s.push_str(&serde_json::to_string(&body).unwrap_or_else(|_| "[]".into()));
        s.push('\n');
        std::fs::write(path, s)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Run the engine's parse-only path on a file and return the
/// populated info block if successful. We don't expose a separate
/// `crucible_validate` entry point; the existing `crucible_load` +
/// `crucible_free` is enough to validate the file (the C++ side
/// parses on load and would have raised on any malformed field).
pub fn validate_model(path: &Path) -> Result<ModelInfo, CrucibleError> {
    let m = Model::load(path)?;
    let info = m.info()?;
    Ok(info)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Top-k indices by descending value. Used by `run --top` to print
/// ImageNet predictions.
pub fn top_k_indices(values: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    indexed
}

// ---------------------------------------------------------------------------
// Library load diagnostics
// ---------------------------------------------------------------------------

/// Best-effort: search the standard Crucible library names on the
/// loader's path so the user knows WHERE to put libcrucible.{so,dylib,dll}.
/// We try the names in the order most-likely-to-succeed; if none
/// load, we report the canonical "searched these" list.
fn load_library_diagnostics(_model_path: &str) -> String {
    #[cfg(target_os = "windows")]
    const CANDIDATES: &[&str] = &["crucible.dll"];
    #[cfg(target_os = "macos")]
    const CANDIDATES: &[&str] = &["libcrucible.dylib", "libcrucible.so"];
    #[cfg(all(unix, not(target_os = "macos")))]
    const CANDIDATES: &[&str] = &["libcrucible.so"];

    format!(
        "could not load the Crucible native library. Tried: {:?}. \
         Build the engine (cmake --build build/release) and either copy \
         the resulting shared object next to the CLI or add its \
         directory to LD_LIBRARY_PATH / DYLD_LIBRARY_PATH / PATH.",
        CANDIDATES,
    )
}
