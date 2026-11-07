//! FFI bindings to the Crucible C ABI (engine/include/crucible/c_api.h).
//!
//!   1. Public Rust API — `Model`, `Tensor`, `CrucibleError`,
//!      `top_k_indices`. Safe Rust; no raw pointers except the borrowed
//!      name arrays inside `ModelInfo` (read by formatter.rs).
//!   2. `extern "C"` block — mirrors c_api.h; the library is linked at build
//!      time by build.rs (`CRUCIBLE_LIB_DIR`).
//!   3. Ownership — `crucible_run` mallocs each output's shape array and data
//!      buffer; `OutputBuffers` frees every non-null pointer with
//!      `crucible_free_array` on every path (success, engine error, or a
//!      malformed descriptor detected here).
//!
//! Engine errors carry the thread-local `crucible_last_error()` text, which
//! is formatted by the engine as `"<CRUCIBLE_ERR_*>: <message>"`.

use std::ffi::{c_char, c_void, CStr, CString};
use std::path::Path;
use std::time::{Duration, Instant};

use thiserror::Error;

// ---------------------------------------------------------------------------
// FFI types — mirror engine/include/crucible/c_api.h verbatim.
// ---------------------------------------------------------------------------

/// `CrucibleStatus` as returned over the ABI. Kept as a raw `i32` so an
/// unexpected value from the library can never be UB.
type RawStatus = i32;
const STATUS_OK: RawStatus = 0;

/// Mirrors `CrucibleModelInfo`. Populated by `crucible_model_info`; the name
/// arrays are owned by the model and valid until it is freed.
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
struct TensorDesc {
    shape: *mut i64,
    rank:  i32,
    size:  i64,
    data:  *const f32,
}

#[repr(C)]
struct CrucibleModelOpaque {
    _private: [u8; 0],
}
type CrucibleModelHandle = *mut CrucibleModelOpaque;

extern "C" {
    fn crucible_load(path: *const c_char) -> CrucibleModelHandle;
    fn crucible_free(model: CrucibleModelHandle);
    fn crucible_model_info(model: CrucibleModelHandle, out: *mut ModelInfo) -> RawStatus;
    fn crucible_run(
        model:        CrucibleModelHandle,
        input_descs:  *const TensorDesc,
        num_inputs:   i32,
        outputs:      *mut *mut f32,
        output_descs: *mut TensorDesc,
        num_outputs:  i32,
    ) -> RawStatus;
    fn crucible_last_error() -> *const c_char;
    fn crucible_free_array(ptr: *mut c_void);
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
    #[error("IO error: {0}")]
    Fs(#[from] std::io::Error),
}

/// Text of `crucible_last_error()` for this thread, if any.
fn last_error() -> Option<String> {
    // SAFETY: the engine returns NULL or a NUL-terminated string valid until
    // the next fallible call on this thread; we copy it immediately.
    unsafe {
        let p = crucible_last_error();
        (!p.is_null()).then(|| CStr::from_ptr(p).to_string_lossy().into_owned())
    }
}

impl CrucibleError {
    /// Build an error from a status code plus the engine's last error text.
    fn from_status(s: RawStatus) -> Self {
        let detail = last_error().unwrap_or_else(|| format!("engine returned status {s} without a message"));
        match s {
            1 => CrucibleError::InvalidArgument(detail),
            2 => CrucibleError::Io(detail),
            3 => CrucibleError::Parse(detail),
            4 => CrucibleError::Runtime(detail),
            5 => CrucibleError::Unsupported(detail),
            _ => CrucibleError::Internal(detail),
        }
    }

    /// `crucible_load` returns NULL rather than a status; recover the status
    /// from the `"<CRUCIBLE_ERR_*>: ..."` prefix of the stored message.
    fn from_load_failure() -> Self {
        let Some(detail) = last_error() else {
            return CrucibleError::Internal("crucible_load returned NULL without an error message".into());
        };
        let status = [
            ("CRUCIBLE_ERR_INVALID_ARGUMENT", 1),
            ("CRUCIBLE_ERR_IO", 2),
            ("CRUCIBLE_ERR_PARSE", 3),
            ("CRUCIBLE_ERR_RUNTIME", 4),
            ("CRUCIBLE_ERR_UNSUPPORTED", 5),
        ]
        .iter()
        .find(|(prefix, _)| detail.strip_prefix(prefix).is_some_and(|rest| rest.starts_with(':')))
        .map_or(6, |&(_, s)| s);
        match status {
            1 => CrucibleError::InvalidArgument(detail),
            2 => CrucibleError::Io(detail),
            3 => CrucibleError::Parse(detail),
            4 => CrucibleError::Runtime(detail),
            5 => CrucibleError::Unsupported(detail),
            _ => CrucibleError::Internal(detail),
        }
    }
}

// ---------------------------------------------------------------------------
// Output buffer ownership
// ---------------------------------------------------------------------------

/// Output slots handed to `crucible_run`. Every non-null shape/data pointer
/// is engine-malloc'd and released exactly once in `Drop`.
struct OutputBuffers {
    data:  Vec<*mut f32>,
    descs: Vec<TensorDesc>,
}

impl OutputBuffers {
    fn new(n: usize) -> Self {
        OutputBuffers {
            data:  vec![std::ptr::null_mut(); n],
            descs: vec![
                TensorDesc { shape: std::ptr::null_mut(), rank: 0, size: 0, data: std::ptr::null() };
                n
            ],
        }
    }

    /// Copy output `i` into an owned tensor, validating the descriptor.
    fn to_tensor(&self, i: usize) -> Result<Tensor, CrucibleError> {
        let d = self.descs[i];
        let buf = self.data[i];
        if d.rank < 0 || d.size < 0 {
            return Err(CrucibleError::Internal(format!(
                "engine returned output {i} with rank {} and size {}", d.rank, d.size)));
        }
        let (rank, size) = (d.rank as usize, d.size as usize);
        if rank > 0 && d.shape.is_null() {
            return Err(CrucibleError::Internal(format!(
                "engine returned output {i} with rank {rank} but a null shape")));
        }
        if size > 0 && buf.is_null() {
            return Err(CrucibleError::Internal(format!(
                "engine returned output {i} with {size} elements but a null data buffer")));
        }
        let shape = if rank == 0 {
            Vec::new()
        } else {
            // SAFETY: non-null, engine wrote `rank` dims.
            unsafe { std::slice::from_raw_parts(d.shape, rank) }.to_vec()
        };
        let data = if size == 0 {
            Vec::new()
        } else {
            // SAFETY: non-null, engine wrote `size` floats.
            unsafe { std::slice::from_raw_parts(buf, size) }.to_vec()
        };
        let expected = shape.iter().try_fold(1usize, |acc, &x| usize::try_from(x).ok().and_then(|x| acc.checked_mul(x)));
        if expected != Some(size) {
            return Err(CrucibleError::Internal(format!(
                "engine returned output {i} with shape {shape:?} but {size} elements")));
        }
        Ok(Tensor { shape, data })
    }
}

impl Drop for OutputBuffers {
    fn drop(&mut self) {
        for (buf, desc) in self.data.iter().zip(&self.descs) {
            // SAFETY: each pointer is either NULL (a no-op for free) or was
            // malloc'd by the engine for this call and not yet freed. On
            // failure the engine leaves slots NULL or frees+NULLs them itself.
            unsafe {
                if !buf.is_null() {
                    crucible_free_array(*buf as *mut c_void);
                }
                if !desc.shape.is_null() {
                    crucible_free_array(desc.shape as *mut c_void);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public model handle
// ---------------------------------------------------------------------------

/// Owning handle to a loaded Crucible model. Drop releases it via `crucible_free`.
pub struct Model {
    handle: CrucibleModelHandle,
}

impl Model {
    /// Load and parse a .onnx file through `crucible_load`.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, CrucibleError> {
        let path_str = path.as_ref().to_str().ok_or_else(|| {
            CrucibleError::InvalidArgument("path is not UTF-8".into())
        })?;
        let cpath = CString::new(path_str)
            .map_err(|_| CrucibleError::InvalidArgument("path contains NUL".into()))?;
        // SAFETY: cpath is a valid C string for the duration of the call.
        let handle = unsafe { crucible_load(cpath.as_ptr()) };
        if handle.is_null() {
            return Err(CrucibleError::from_load_failure());
        }
        Ok(Model { handle })
    }

    /// Static metadata snapshot (`crucible_model_info`).
    pub fn info(&self) -> Result<ModelInfo, CrucibleError> {
        let mut info = std::mem::MaybeUninit::<ModelInfo>::uninit();
        // SAFETY: `info` is a valid out-pointer; read only on success.
        let s = unsafe { crucible_model_info(self.handle, info.as_mut_ptr()) };
        if s == STATUS_OK {
            // SAFETY: status OK means every field was written.
            Ok(unsafe { info.assume_init() })
        } else {
            Err(CrucibleError::from_status(s))
        }
    }

    /// Run one inference; returns all declared outputs (the engine currently
    /// materialises only the first; the rest come back empty).
    pub fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, CrucibleError> {
        self.run_timed(inputs).map(|(outs, _)| outs)
    }

    /// Like `run`, also returning the wall-clock time of the `crucible_run`
    /// FFI call alone (excludes descriptor setup and copying outputs out).
    pub fn run_timed(&self, inputs: &[Tensor]) -> Result<(Vec<Tensor>, Duration), CrucibleError> {
        let in_descs: Vec<TensorDesc> = inputs
            .iter()
            .map(|t| TensorDesc {
                // The engine only reads through this pointer.
                shape: t.shape.as_ptr() as *mut i64,
                rank:  t.shape.len() as i32,
                size:  t.data.len() as i64,
                data:  t.data.as_ptr(),
            })
            .collect();

        let n_out = usize::try_from(self.info()?.num_outputs).map_err(|_| {
            CrucibleError::Internal("engine reported a negative output count".into())
        })?;
        let mut outs = OutputBuffers::new(n_out);

        let t0 = Instant::now();
        // SAFETY: input descriptors borrow `inputs`, which outlives the call;
        // output arrays have exactly `n_out` slots as the ABI requires.
        let s = unsafe {
            crucible_run(
                self.handle,
                in_descs.as_ptr(),
                in_descs.len() as i32,
                outs.data.as_mut_ptr(),
                outs.descs.as_mut_ptr(),
                n_out as i32,
            )
        };
        let elapsed = t0.elapsed();
        if s != STATUS_OK {
            return Err(CrucibleError::from_status(s));
        }
        let tensors = (0..n_out).map(|i| outs.to_tensor(i)).collect::<Result<Vec<_>, _>>()?;
        Ok((tensors, elapsed))
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        // SAFETY: handle came from a successful crucible_load and is freed once.
        unsafe { crucible_free(self.handle) }
    }
}

// ---------------------------------------------------------------------------
// Public Tensor
// ---------------------------------------------------------------------------

/// Owned float32 tensor with row-major storage.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<i64>,
    pub data:  Vec<f32>,
}

impl Tensor {
    /// Tensor with no shape and no elements (absent output).
    pub fn empty() -> Self {
        Tensor { shape: Vec::new(), data: Vec::new() }
    }

    /// Read a JSON tensor file: `{ "shape": [N, C, H, W], "data": [..] }`.
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
            let val = d.as_i64().ok_or_else(|| CrucibleError::InvalidArgument(
                "input JSON shape entries must be int".into()))?;
            if val < 0 {
                return Err(CrucibleError::InvalidArgument(
                    format!("negative dimension in shape: {val}")));
            }
            shape.push(val);
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
        let expected = shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
            .ok_or_else(|| CrucibleError::InvalidArgument(
                format!("shape dimensions overflow: {shape:?}")))?;
        if expected != data.len() {
            return Err(CrucibleError::InvalidArgument(format!(
                "input shape product {expected} != data length {}",
                data.len())));
        }
        Ok(Tensor { shape, data })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Top-k (index, value) pairs by descending value; NaNs are skipped.
pub fn top_k_indices(values: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = values.iter()
        .copied()
        .enumerate()
        .filter(|(_, val)| !val.is_nan())
        .collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
    indexed.truncate(k);
    indexed
}
