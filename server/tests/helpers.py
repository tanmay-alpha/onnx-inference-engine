"""Generated ONNX models and request helpers for the server tests."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import onnx
from fastapi.testclient import TestClient
from onnx import TensorProto, helper, numpy_helper


# ---------------------------------------------------------------------------
# Small generated ONNX models
# ---------------------------------------------------------------------------
def _finish(graph: onnx.GraphProto, opset: int = 13) -> bytes:
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model.SerializeToString()


def mlp_bytes() -> bytes:
    """X[N,4] -> Gemm(8) -> Relu -> Gemm(3) -> Softmax. Dynamic batch dim."""
    rng = np.random.RandomState(0)
    w1 = numpy_helper.from_array(rng.randn(4, 8).astype(np.float32), "W1")
    b1 = numpy_helper.from_array(rng.randn(8).astype(np.float32), "B1")
    w2 = numpy_helper.from_array(rng.randn(8, 3).astype(np.float32), "W2")
    b2 = numpy_helper.from_array(rng.randn(3).astype(np.float32), "B2")
    nodes = [
        helper.make_node("Gemm", ["X", "W1", "B1"], ["H"]),
        helper.make_node("Relu", ["H"], ["R"]),
        helper.make_node("Gemm", ["R", "W2", "B2"], ["L"]),
        helper.make_node("Softmax", ["L"], ["Y"], axis=1),
    ]
    graph = helper.make_graph(
        nodes, "mlp",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["N", 3])],
        [w1, b1, w2, b2],
    )
    return _finish(graph)


def conv_bytes() -> bytes:
    """X[1,3,8,8] -> Conv(4,3x3,pad1) -> Relu -> GlobalAveragePool -> Flatten -> Gemm(2). Static shape."""
    rng = np.random.RandomState(1)
    w = numpy_helper.from_array(rng.randn(4, 3, 3, 3).astype(np.float32), "CW")
    b = numpy_helper.from_array(rng.randn(4).astype(np.float32), "CB")
    fw = numpy_helper.from_array(rng.randn(4, 2).astype(np.float32), "FW")
    fb = numpy_helper.from_array(rng.randn(2).astype(np.float32), "FB")
    nodes = [
        helper.make_node("Conv", ["X", "CW", "CB"], ["C"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["C"], ["R"]),
        helper.make_node("GlobalAveragePool", ["R"], ["P"]),
        helper.make_node("Flatten", ["P"], ["F"], axis=1),
        helper.make_node("Gemm", ["F", "FW", "FB"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes, "conv",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])],
        [w, b, fw, fb],
    )
    return _finish(graph)


def unsupported_bytes() -> bytes:
    """Uses Cos, which Crucible does not implement."""
    graph = helper.make_graph(
        [helper.make_node("Cos", ["X"], ["Y"])], "cos",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])],
    )
    return _finish(graph)


def upload(client: TestClient, headers: Dict[str, str], data: bytes,
           filename: str = "model.onnx", **form) -> dict:
    r = client.post("/convert", files={"model_file": (filename, data, "application/octet-stream")},
                    data=form or None, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


def ort_run(data: bytes, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
    return sess.run(None, {sess.get_inputs()[0].name: x})[0]


def files_in(d: Path) -> List[Path]:
    return sorted(p for p in d.iterdir()) if d.exists() else []
