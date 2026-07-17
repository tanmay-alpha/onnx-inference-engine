#!/usr/bin/env python3
"""
Generate tiny .onnx test fixtures for the Crucible C++ parser (Issue #4).

These are hand-crafted models that exercise the parser's full API surface
while staying under a few KB each. They are committed alongside the test
code so the parser tests are reproducible without network access.

Each fixture writes to engine/tests/fixtures/<filename>.onnx.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from onnx import (
    TensorProto,
    helper,
)


# engine/tests/fixtures, relative to repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES_DIR = os.path.join(REPO_ROOT, "engine", "tests", "fixtures")

# Validate expected layout: repo root should contain an engine/ directory.
engine_dir = os.path.join(REPO_ROOT, "engine")
if not os.path.isdir(engine_dir):
    raise FileNotFoundError(
        f"Expected Crucible layout: {engine_dir} not found. "
        f"The repo root appears to be {REPO_ROOT}. "
        f"Run generate_fixtures.py from the Crucible project root."
    )


def _make_tensor(name, dtype, dims, raw_bytes):
    """Create an initializer tensor with validated shape and native byte order."""
    # Validate dims
    for d in dims:
        if d <= 0:
            raise ValueError(f"Tensor '{name}': non-positive dimension {d}")
    # Ensure native byte order
    arr = np.frombuffer(raw_bytes, dtype=dtype)
    if arr.dtype.byteorder not in ('=', sys.byteorder):
        arr = arr.byteswap().newbyteorder('=')
    return helper.make_tensor(name, dtype, dims, arr.tobytes(), raw=True)


def make_matmul_add():
    """input -> MatMul(W) -> Add(b) -> output."""
    W = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]], dtype=np.float32)
    b = np.array([0.5, -0.5], dtype=np.float32)

    X_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
    Y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])
    W_init = helper.make_tensor("W", TensorProto.FLOAT, [2, 3], W.tobytes(), raw=True)
    b_init = helper.make_tensor("b", TensorProto.FLOAT, [2],     b.tobytes(), raw=True)

    nodes = [
        helper.make_node("MatMul", ["X", "W"], ["mm_out"], name="matmul1"),
        helper.make_node("Add",    ["mm_out", "b"], ["Y"], name="add1"),
    ]
    graph = helper.make_graph(
        nodes=nodes, name="matmul_add",
        inputs=[X_info], outputs=[Y_info],
        initializer=[W_init, b_init],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])


def make_gemm():
    """Single Gemm node with alpha=2.0, beta=0.5, transB=1."""
    X_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    Y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2])
    B = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    C = np.array([[0.1, -0.1]], dtype=np.float32)
    B_init = helper.make_tensor("B", TensorProto.FLOAT, [3, 2], B.tobytes(), raw=True)
    C_init = helper.make_tensor("C", TensorProto.FLOAT, [1, 2], C.tobytes(), raw=True)

    node = helper.make_node("Gemm", ["X", "B", "C"], ["Y"],
                            alpha=2.0, beta=0.5, transB=1, name="gemm1")
    graph = helper.make_graph(
        nodes=[node], name="gemm",
        inputs=[X_info], outputs=[Y_info],
        initializer=[B_init, C_init],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])


def make_reshape():
    """Single Reshape node using a target_shape initializer."""
    X_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 6])
    Y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
    shape_init = helper.make_tensor(
        "target_shape", TensorProto.INT64, [2],
        np.array([2, 3], dtype=np.int64).tobytes(), raw=True)
    node = helper.make_node("Reshape", ["X", "target_shape"], ["Y"], name="reshape1")
    graph = helper.make_graph(
        nodes=[node], name="reshape",
        inputs=[X_info], outputs=[Y_info],
        initializer=[shape_init],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])


def make_chain_4_nodes():
    """Four-node chain: MatMul -> Add -> MatMul -> Add."""
    W1 = np.eye(3, dtype=np.float32)
    b1 = np.zeros((3,), dtype=np.float32)
    W2 = np.ones((3, 3), dtype=np.float32)
    b2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    X_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
    Y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
    initializers = [
        helper.make_tensor("W1", TensorProto.FLOAT, [3, 3], W1.tobytes(), raw=True),
        helper.make_tensor("b1", TensorProto.FLOAT, [3],     b1.tobytes(), raw=True),
        helper.make_tensor("W2", TensorProto.FLOAT, [3, 3], W2.tobytes(), raw=True),
        helper.make_tensor("b2", TensorProto.FLOAT, [3],     b2.tobytes(), raw=True),
    ]
    nodes = [
        helper.make_node("MatMul", ["X",   "W1"], ["mm1"], name="mm1"),
        helper.make_node("Add",    ["mm1", "b1"], ["a1"],  name="a1"),
        helper.make_node("MatMul", ["a1",  "W2"], ["mm2"], name="mm2"),
        helper.make_node("Add",    ["mm2", "b2"], ["Y"],   name="a2"),
    ]
    graph = helper.make_graph(
        nodes=nodes, name="chain4", inputs=[X_info], outputs=[Y_info],
        initializer=initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])


def make_empty():
    """Smallest legal model — just input/output, no nodes, no initializers."""
    X_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])
    Y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])
    graph = helper.make_graph(nodes=[], name="empty", inputs=[X_info], outputs=[Y_info])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])


def write_model(model, name):
    path = os.path.join(FIXTURES_DIR, name + ".onnx")
    data = model.SerializeToString()
    with open(path, "wb") as f:
        f.write(data)
    print(f"  wrote {path}  ({len(data)} bytes)")


def _validate_opset(model, min_opset):
    """Verify the model's default ONNX opset is >= min_opset."""
    for imp in model.opset_import:
        if imp.domain == "" and int(imp.version) < min_opset:
            raise ValueError(
                f"Model '{model.graph.name}': default opset {imp.version} < required {min_opset}"
            )


def main():
    print(f"Generating fixtures in {FIXTURES_DIR}")
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    for make_fn in (make_matmul_add, make_gemm, make_reshape, make_chain_4_nodes, make_empty):
        model = make_fn()
        _validate_opset(model, 17)
        write_model(model, make_fn.__name__.removeprefix("make_"))
    print("Done.")


if __name__ == "__main__":
    main()