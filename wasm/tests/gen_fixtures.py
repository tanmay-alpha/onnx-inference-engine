#!/usr/bin/env python3
"""Generate ONNX Runtime parity fixtures for the Crucible WASM runtime.

Deterministic: fixed RNG seed, fixed graphs. Writes small .onnx graphs and one
JSON file per case into wasm/tests/fixtures/. Each JSON holds the input, and
ONNX Runtime's (CPU EP) output as the expected value, or `expect_error` for
cases the WASM runtime must reject. `wasm/tests/parity.rs` consumes them.

Usage (from repo root):  python wasm/tests/gen_fixtures.py
Requires: numpy, onnx, onnxruntime.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
REPO = HERE.parent.parent
FRAUD_MODEL = REPO / "models" / "fraud" / "fraud_detector.onnx"

RNG = np.random.default_rng(20261108)


def rand(*shape: int) -> np.ndarray:
    return RNG.standard_normal(shape).astype(np.float32)


def vi(name: str, shape: list[int | str] | None) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def make(name: str, nodes, inputs, outputs, inits=(), opset: int = 13) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        name,
        inputs,
        outputs,
        initializer=[numpy_helper.from_array(v, k) for k, v in inits],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset)], producer_name="crucible-wasm-fixtures"
    )
    model.ir_version = 7
    return model


def ort_run(model_path: Path, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return np.asarray(sess.run(None, {sess.get_inputs()[0].name: x})[0], dtype=np.float32)


def floats(a: np.ndarray) -> list[float]:
    # float(np.float32) is exact as a double, so the JSON round-trips bit-exactly.
    return [float(v) for v in a.astype(np.float32).ravel()]


def write_case(name: str, model_file: str, x: np.ndarray, *, expect_error: str | None = None) -> None:
    model_path = (FIXTURES / model_file).resolve()
    case: dict = {
        "name": name,
        "model": model_file.replace("\\", "/"),
        "input_shape": list(x.shape),
        "input": floats(x),
    }
    if expect_error is None:
        y = ort_run(model_path, x)
        case["expected_shape"] = list(y.shape)
        case["expected"] = floats(y)
        case["reference"] = f"onnxruntime {ort.__version__} CPUExecutionProvider"
    else:
        case["expect_error"] = expect_error
    (FIXTURES / f"{name}.json").write_text(json.dumps(case, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {name}.json")


def save(model: onnx.ModelProto, file: str, check: bool = True) -> str:
    if check:
        onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(FIXTURES / file))
    return file


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for old in FIXTURES.glob("*"):
        old.unlink()

    # MatMul -> Add(bias [3]) -> Relu -> MatMul -> Add -> Sigmoid
    m = make(
        "mlp_sigmoid",
        [
            helper.make_node("MatMul", ["x", "w1"], ["h0"]),
            helper.make_node("Add", ["h0", "b1"], ["h1"]),
            helper.make_node("Relu", ["h1"], ["h2"]),
            helper.make_node("MatMul", ["h2", "w2"], ["h3"]),
            helper.make_node("Add", ["h3", "b2"], ["h4"]),
            helper.make_node("Sigmoid", ["h4"], ["y"]),
        ],
        [vi("x", ["N", 4])],
        [vi("y", ["N", 2])],
        [("w1", rand(4, 3)), ("b1", rand(3)), ("w2", rand(3, 2)), ("b2", rand(2))],
    )
    write_case("mlp_sigmoid", save(m, "mlp_sigmoid.onnx"), rand(5, 4))

    # Add broadcasting: [2,3,4] + [4]
    m = make(
        "add_3d_1d",
        [helper.make_node("Add", ["x", "b"], ["y"])],
        [vi("x", [2, 3, 4])],
        [vi("y", [2, 3, 4])],
        [("b", rand(4))],
    )
    write_case("add_3d_1d", save(m, "add_3d_1d.onnx"), rand(2, 3, 4))

    # Add broadcasting both directions: [4,1] + [1,5] -> [4,5]
    m = make(
        "add_col_row",
        [helper.make_node("Add", ["x", "b"], ["y"])],
        [vi("x", [4, 1])],
        [vi("y", [4, 5])],
        [("b", rand(1, 5))],
    )
    write_case("add_col_row", save(m, "add_col_row.onnx"), rand(4, 1))

    # Add with initializer as the FIRST operand broadcasting up: [3,1] + x[2,1,4]
    m = make(
        "add_init_first",
        [helper.make_node("Add", ["b", "x"], ["y"])],
        [vi("x", [2, 1, 4])],
        [vi("y", [2, 3, 4])],
        [("b", rand(3, 1))],
    )
    write_case("add_init_first", save(m, "add_init_first.onnx"), rand(2, 1, 4))

    # Add a rank-0 scalar initializer
    m = make(
        "add_scalar",
        [helper.make_node("Add", ["x", "s"], ["y"])],
        [vi("x", [2, 3])],
        [vi("y", [2, 3])],
        [("s", np.array(1.5, dtype=np.float32))],
    )
    write_case("add_scalar", save(m, "add_scalar.onnx"), rand(2, 3))

    # Softmax, rank 3: opset 11 default axis=1 (2-D coercion), opset 13 default axis=-1,
    # and opset 13 explicit axis=1 (strided).
    for name, opset, attrs in [
        ("softmax_opset11_rank3", 11, {}),
        ("softmax_opset13_rank3", 13, {}),
        ("softmax_opset13_axis1", 13, {"axis": 1}),
    ]:
        m = make(
            name,
            [helper.make_node("Softmax", ["x"], ["y"], **attrs)],
            [vi("x", [2, 3, 4])],
            [vi("y", [2, 3, 4])],
            opset=opset,
        )
        write_case(name, save(m, f"{name}.onnx"), rand(2, 3, 4) * 3)

    # Real model: fraud detector (referenced in place, not copied).
    fraud_rel = Path("..") / ".." / ".." / FRAUD_MODEL.relative_to(REPO)
    write_case("fraud_detector", fraud_rel.as_posix(), rand(8, 7) * 2)

    # Must-fail cases (no ORT reference).
    m = make(
        "unsupported_tanh",
        [helper.make_node("Tanh", ["x"], ["y"])],
        [vi("x", [1, 4])],
        [vi("y", [1, 4])],
    )
    write_case("unsupported_tanh", save(m, "unsupported_tanh.onnx"), rand(1, 4), expect_error="Tanh")

    m = make(
        "add_incompatible",
        [helper.make_node("Add", ["x", "b"], ["y"])],
        [vi("x", [2, 3])],
        [vi("y", None)],
        [("b", rand(4))],
    )
    write_case(
        "add_incompatible",
        save(m, "add_incompatible.onnx", check=False),
        rand(2, 3),
        expect_error="not broadcast-compatible",
    )

    m = make(
        "matmul_mismatch",
        [helper.make_node("MatMul", ["x", "w"], ["y"])],
        [vi("x", [2, 3])],
        [vi("y", None)],
        [("w", rand(4, 2))],
    )
    write_case(
        "matmul_mismatch",
        save(m, "matmul_mismatch.onnx", check=False),
        rand(2, 3),
        expect_error="inner dimensions differ",
    )


if __name__ == "__main__":
    main()
