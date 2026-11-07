//! ONNX Runtime parity tests for the WASM interpreter.
//!
//! Fixtures are produced by `wasm/tests/gen_fixtures.py` (deterministic) and
//! committed under `wasm/tests/fixtures/`. Each JSON case either carries ORT's
//! output (`expected`, compared at max abs diff <= 1e-5) or an `expect_error`
//! substring the runtime must report.

use std::path::{Path, PathBuf};

use crucible_wasm::{load_model, run_inference_internal};
use serde_json::Value;

const TOLERANCE: f32 = 1e-5;

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures")
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number") as f32)
        .collect()
}

fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_i64().expect("int"))
        .collect()
}

fn run_case(name: &str) {
    let dir = fixtures_dir();
    let case: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join(format!("{name}.json")))
            .unwrap_or_else(|e| panic!("{name}: missing fixture ({e}); run wasm/tests/gen_fixtures.py")),
    )
    .expect("fixture JSON");
    let model_path = dir.join(case["model"].as_str().expect("model path"));
    let bytes = std::fs::read(&model_path).unwrap_or_else(|e| panic!("{name}: read {}: {e}", model_path.display()));
    let model = load_model(&bytes).unwrap_or_else(|e| panic!("{name}: parse failed: {e}"));
    let result = run_inference_internal(&model, &floats(&case["input"]), &ints(&case["input_shape"]));

    if let Some(substr) = case.get("expect_error").and_then(Value::as_str) {
        let err = result.err().unwrap_or_else(|| panic!("{name}: expected an error containing '{substr}'"));
        assert!(err.contains(substr), "{name}: error '{err}' does not contain '{substr}'");
        return;
    }

    let out = result.unwrap_or_else(|e| panic!("{name}: inference failed: {e}"));
    assert_eq!(out.shape, ints(&case["expected_shape"]), "{name}: output shape");
    let expected = floats(&case["expected"]);
    assert_eq!(out.data.len(), expected.len(), "{name}: output length");
    let max_diff = out
        .data
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff <= TOLERANCE, "{name}: max abs diff vs ORT {max_diff:e} > {TOLERANCE:e}");
}

macro_rules! parity {
    ($($name:ident),* $(,)?) => {
        $(#[test] fn $name() { run_case(stringify!($name)); })*
    };
}

parity!(
    mlp_sigmoid,
    add_3d_1d,
    add_col_row,
    add_init_first,
    add_scalar,
    softmax_opset11_rank3,
    softmax_opset13_rank3,
    softmax_opset13_axis1,
    fraud_detector,
    unsupported_tanh,
    add_incompatible,
    matmul_mismatch,
);
