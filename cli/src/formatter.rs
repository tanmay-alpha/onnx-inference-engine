//! Output formatting for the four CLI commands.
//!
//! Two modes: text (human-readable) and JSON (machine-readable).
//! JSON is selected with --json; the schemas are stable enough that
//! downstream tools (CI dashboards, the Issue #18 chart) can consume
//! them without a hand-rolled parser.
//!
//! Why one file? Every command emits different shapes; grouping
//! them by "thing being printed" (info, prediction, bench stats,
//! validate result) keeps the formatting decisions visible together.

use std::ffi::{c_char, CStr};
use std::io::Write;
use std::path::Path;

use crate::runner::{ModelInfo, Tensor};

/// Human-readable rendering of a model info block. Used by `info`
/// and the success path of `validate`.
pub fn format_model_info_text(info: &ModelInfo, path: &Path) -> String {
    let mut s = String::new();
    s.push_str(&format!("Model        : {}\n", path.display()));
    s.push_str(&format!("ABI version  : {}\n", info.abi_version));
    let input_names = unsafe { read_c_str_array(info.input_names, info.num_inputs) };
    let output_names = unsafe { read_c_str_array(info.output_names, info.num_outputs) };
    s.push_str(&format!("Inputs       : {}\n", input_names.len()));
    for (i, name) in input_names.iter().enumerate() {
        s.push_str(&format!("              [{i}] {name}\n"));
    }
    s.push_str(&format!("Outputs      : {}\n", output_names.len()));
    for (i, name) in output_names.iter().enumerate() {
        s.push_str(&format!("              [{i}] {name}\n"));
    }
    s.push_str(&format!("Initializers : {}\n", info.num_initializers));
    s.push_str(&format!("Nodes        : {}\n", info.num_nodes));
    s
}

/// JSON rendering of the same info block. The `shape` arrays are
/// represented as nested arrays so the JSON is unambiguous.
pub fn format_model_info_json(info: &ModelInfo, path: &Path) -> serde_json::Value {
    let input_names = unsafe { read_c_str_array(info.input_names, info.num_inputs) };
    let output_names = unsafe { read_c_str_array(info.output_names, info.num_outputs) };
    serde_json::json!({
        "model":          path.display().to_string(),
        "abi_version":    info.abi_version,
        "num_inputs":     info.num_inputs,
        "num_outputs":    info.num_outputs,
        "num_initializers": info.num_initializers,
        "num_nodes":      info.num_nodes,
        "input_names":    input_names,
        "output_names":   output_names,
    })
}

/// Format a single output tensor's top-k predictions as a table.
/// `class_names`, when supplied, is indexed by the same integers as
/// the values (e.g. ImageNet synset IDs). Without class names, the
/// row shows "class#N".
pub fn format_top_k_predictions(
    out: &Tensor,
    class_names: Option<&[String]>,
    k: usize,
) -> String {
    let top = crate::runner::top_k_indices(&out.data, k);
    let mut s = String::new();
    s.push_str("rank  class_idx  probability  name\n");
    s.push_str("----  ---------  ----------  ----\n");
    for (rank, (idx, val)) in top.iter().enumerate() {
        let name = class_names
            .and_then(|c| c.get(*idx))
            .cloned()
            .unwrap_or_else(|| format!("class#{idx}"));
        s.push_str(&format!(
            "{:>4}  {:>9}  {:>10.6}  {name}\n",
            rank + 1, idx, val,
        ));
    }
    s
}

/// Format full output tensor shape + first-N values. Used when the
/// user asks for the raw output (--print-output) rather than top-k.
pub fn format_tensor_summary(out: &Tensor, max_elements: usize) -> String {
    let mut s = String::new();
    s.push_str(&format!("shape: {:?}\n", out.shape));
    let n = out.data.len().min(max_elements);
    s.push_str(&format!("data  (first {n} of {}):\n", out.data.len()));
    for (i, v) in out.data.iter().take(n).enumerate() {
        s.push_str(&format!("  [{i:>6}] {v:>12.6}\n"));
    }
    if out.data.len() > n {
        s.push_str(&format!("  ... {} more\n", out.data.len() - n));
    }
    s
}

/// Format a `validate` command's outcome.
pub fn format_validate_text(path: &Path, info: &ModelInfo) -> String {
    let mut s = format!("OK: {}\n", path.display());
    s.push_str(&format!(
        "    {i} inputs, {o} outputs, {n} nodes, {init} initializers\n",
        i  = info.num_inputs,
        o  = info.num_outputs,
        n  = info.num_nodes,
        init = info.num_initializers,
    ));
    s
}

/// What `benchmark` measured: wall-clock time of each `crucible_run` call
/// (native C++ engine through the C ABI, including the ABI's input/output
/// copies), after `warmup` untimed calls.
pub const BENCH_METHOD: &str =
    "native Crucible engine (C++) via the C ABI; timer = std::time::Instant around each crucible_run call";

/// Latency statistics in milliseconds.
#[derive(Debug, Default, Clone, Copy)]
pub struct BenchStats {
    pub mean_ms:   f64,
    pub median_ms: f64,
    pub p95_ms:    f64,
    pub p99_ms:    f64,
    pub min_ms:    f64,
    pub max_ms:    f64,
}

/// Everything `benchmark` reports.
pub struct BenchReport<'a> {
    pub model:       &'a Path,
    pub input_shape: &'a [i64],
    pub warmup:      usize,
    pub runs:        usize,
    pub stats:       BenchStats,
}

fn calls_per_sec(mean_ms: f64) -> f64 {
    if mean_ms > 0.0 { 1000.0 / mean_ms } else { 0.0 }
}

pub fn format_bench_text(r: &BenchReport) -> String {
    let st = &r.stats;
    let mut s = String::new();
    s.push_str(&format!("Model        : {}\n", r.model.display()));
    s.push_str(&format!("Input shape  : {:?}\n", r.input_shape));
    s.push_str(&format!("Measures     : {BENCH_METHOD}\n"));
    s.push_str(&format!("Iterations   : {} warmup (untimed) + {} timed\n", r.warmup, r.runs));
    s.push_str(&format!("Mean (ms)    : {:.3}\n", st.mean_ms));
    s.push_str(&format!("Median (ms)  : {:.3}\n", st.median_ms));
    s.push_str(&format!("p95  (ms)    : {:.3}\n", st.p95_ms));
    s.push_str(&format!("p99  (ms)    : {:.3}\n", st.p99_ms));
    s.push_str(&format!("Min  (ms)    : {:.3}\n", st.min_ms));
    s.push_str(&format!("Max  (ms)    : {:.3}\n", st.max_ms));
    s.push_str(&format!("Calls/s      : {:.2} (1000 / mean)\n", calls_per_sec(st.mean_ms)));
    s
}

pub fn format_bench_json(r: &BenchReport) -> serde_json::Value {
    let st = &r.stats;
    serde_json::json!({
        "model":        r.model.display().to_string(),
        "input_shape":  r.input_shape,
        "runtime":      "crucible-native",
        "method":       BENCH_METHOD,
        "warmup":       r.warmup,
        "runs":         r.runs,
        "mean_ms":      st.mean_ms,
        "median_ms":    st.median_ms,
        "p95_ms":       st.p95_ms,
        "p99_ms":       st.p99_ms,
        "min_ms":       st.min_ms,
        "max_ms":       st.max_ms,
        "calls_per_sec": calls_per_sec(st.mean_ms),
    })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Read a `*const c_char` from the FFI into a Rust `String`.
///
/// SAFETY: caller guarantees the pointer is a valid C string owned
/// by the C model (the C side keeps the strings alive for the life
/// of the model). We copy immediately so the borrow is short.
unsafe fn read_c_str_ptr(p: *const std::os::raw::c_char) -> String {
    if p.is_null() { return "<null>".to_string(); }
    CStr::from_ptr(p).to_string_lossy().into_owned()
}

/// SAFETY: read a null-terminated string array. Returns an empty
/// Vec if the pointer itself is null or if num is 0.
unsafe fn read_c_str_array(ptr: *const *const c_char, num: i32) -> Vec<String> {
    if ptr.is_null() || num <= 0 {
        return Vec::new();
    }
    (0..num as isize)
        .map(|i| read_c_str_ptr(*ptr.offset(i)))
        .collect()
}

/// Flush stdout. Used by the CLI after every command so a piping
/// user sees output immediately even if the process is killed by
/// the next pipe stage.
pub fn flush() {
    let _ = std::io::stdout().flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bench_json() {
        let report = BenchReport {
            model: Path::new("m.onnx"),
            input_shape: &[1, 7],
            warmup: 3,
            runs: 10,
            stats: BenchStats { mean_ms: 10.0, median_ms: 9.5, p95_ms: 12.0, p99_ms: 13.0, min_ms: 8.0, max_ms: 14.0 },
        };
        let val = format_bench_json(&report);
        assert_eq!(val["runs"], 10);
        assert_eq!(val["warmup"], 3);
        assert_eq!(val["runtime"], "crucible-native");
        assert_eq!(val["mean_ms"], 10.0);
        assert_eq!(val["calls_per_sec"], 100.0);
        assert!(format_bench_text(&report).contains("3 warmup (untimed) + 10 timed"));
    }

    #[test]
    fn test_format_tensor_summary() {
        let t = Tensor {
            shape: vec![1, 3],
            data: vec![1.0, 2.0, 3.0],
        };
        let s = format_tensor_summary(&t, 2);
        assert!(s.contains("shape: [1, 3]"));
        assert!(s.contains("first 2 of 3"));
    }
}
