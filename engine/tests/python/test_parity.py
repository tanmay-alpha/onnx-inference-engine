"""Numerical parity: Crucible native engine (crucible_py) vs ONNX Runtime CPU.

Every case builds a small ONNX graph with onnx.helper (or loads a real model
from models/), runs it through both engines on identical seeded inputs, and
compares with numpy.testing.assert_allclose.

Tolerances (float32):
  * single-operator graphs:  rtol=1e-4, atol=1e-5
  * full CNNs (ResNet18, MobileNetV2): rtol=1e-3, atol=1e-3 — accumulation
    order differs between Eigen GEMM and MLAS over 10^5-10^6 term dot
    products, so bit-exactness is not expected. Top-1 class must match.

This module requires the native extension. It is NOT skipped when the
extension is missing — a missing build is a failure, not a pass.
Point CRUCIBLE_PY_DIR at the directory containing crucible_py*.so/.pyd.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

REPO = Path(__file__).resolve().parents[3]
if os.environ.get("CRUCIBLE_PY_DIR"):
    sys.path.insert(0, os.environ["CRUCIBLE_PY_DIR"])

import crucible_py  # noqa: E402  (hard dependency, see module docstring)

OP_RTOL, OP_ATOL = 1e-4, 1e-5
MODEL_RTOL, MODEL_ATOL = 1e-3, 1e-3


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _make_model(nodes, inputs, outputs, initializers=(), opset=17):
    graph = helper.make_graph(nodes, "g", inputs, outputs, initializer=list(initializers))
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    # The checker insists on output shapes; several cases leave the output
    # shape for ONNX Runtime to infer, so only check fully-typed graphs.
    if all(o.type.tensor_type.HasField("shape") for o in outputs):
        onnx.checker.check_model(model)
    return model


def _run_both(model: onnx.ModelProto, feeds: dict[str, np.ndarray], tmp_path: Path):
    path = tmp_path / "m.onnx"
    onnx.save(model, str(path))
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    expected = sess.run(None, feeds)[0]
    m = crucible_py.load_model(str(path))
    actual = crucible_py.run(m, feeds)
    return actual, expected


def _vi(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _init(name, arr):
    return numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name)


def _rand(*shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def _check_single(model, feeds, tmp_path, rtol=OP_RTOL, atol=OP_ATOL):
    actual, expected = _run_both(model, feeds, tmp_path)
    assert actual.shape == expected.shape, f"shape {actual.shape} != ORT {expected.shape}"
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


# --------------------------------------------------------------------------
# elementwise / activations
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "op,attrs,opset",
    [
        ("Relu", {}, 17),
        ("Sigmoid", {}, 17),
        ("Tanh", {}, 17),
        ("LeakyRelu", {"alpha": 0.2}, 17),
        ("LeakyRelu", {}, 17),
        ("Elu", {"alpha": 0.7}, 17),
        ("Gelu", {}, 20),
        ("Gelu", {"approximate": "tanh"}, 20),
        ("Identity", {}, 17),
    ],
)
def test_unary_ops(op, attrs, opset, tmp_path):
    x = _rand(2, 3, 4, 5)
    node = helper.make_node(op, ["X"], ["Y"], **attrs)
    model = _make_model([node], [_vi("X", x.shape)], [_vi("Y", x.shape)], opset=opset)
    _check_single(model, {"X": x}, tmp_path)


def test_dropout_is_identity_at_inference(tmp_path):
    x = _rand(4, 8)
    node = helper.make_node("Dropout", ["X"], ["Y"])
    model = _make_model([node], [_vi("X", x.shape)], [_vi("Y", x.shape)])
    _check_single(model, {"X": x}, tmp_path)


@pytest.mark.parametrize("shape,axis", [((3, 10), -1), ((2, 3, 7), -1), ((2, 3, 7), 2)])
def test_softmax_opset13(shape, axis, tmp_path):
    x = _rand(*shape) * 5
    node = helper.make_node("Softmax", ["X"], ["Y"], axis=axis)
    model = _make_model([node], [_vi("X", shape)], [_vi("Y", shape)], opset=13)
    _check_single(model, {"X": x}, tmp_path)


@pytest.mark.parametrize("shape,axis", [((2, 3, 4), None), ((2, 3, 4), 1), ((2, 3, 4, 5), 2), ((4, 6), None)])
def test_softmax_opset11_coerces_to_2d(shape, axis, tmp_path):
    # Opset < 13: input is flattened to [prod(shape[:axis]), prod(shape[axis:])]
    # and softmax runs over the second dimension. Default axis = 1.
    x = _rand(*shape) * 3
    kw = {} if axis is None else {"axis": axis}
    node = helper.make_node("Softmax", ["X"], ["Y"], **kw)
    model = _make_model([node], [_vi("X", shape)], [_vi("Y", shape)], opset=11)
    _check_single(model, {"X": x}, tmp_path)


# --------------------------------------------------------------------------
# Add broadcasting (numpy semantics)
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "a_shape,b_shape",
    [
        ((2, 3, 4, 5), (2, 3, 4, 5)),
        ((2, 3, 4, 5), (5,)),
        ((2, 3, 4, 5), (1, 3, 1, 1)),
        ((2, 3, 4, 5), (3, 1, 1)),
        ((2, 1, 4), (3, 1)),
        ((4, 1), (1, 6)),
        ((5,), (2, 3, 5)),
        ((2, 3), ()),
    ],
)
def test_add_broadcast(a_shape, b_shape, tmp_path):
    a = _rand(*a_shape, seed=1)
    b = _rand(*b_shape, seed=2) if b_shape else np.array(1.5, dtype=np.float32)
    out_shape = np.broadcast_shapes(a_shape, b_shape)
    node = helper.make_node("Add", ["A", "B"], ["Y"])
    model = _make_model(
        [node], [_vi("A", a_shape)], [_vi("Y", out_shape)], initializers=[_init("B", b)]
    )
    _check_single(model, {"A": a}, tmp_path)


# --------------------------------------------------------------------------
# MatMul / Gemm
# --------------------------------------------------------------------------

def test_matmul_add_sigmoid(tmp_path):
    x = _rand(16, 7)
    w = _rand(7, 1, seed=3)
    b = _rand(1, seed=4)
    nodes = [
        helper.make_node("MatMul", ["X", "W"], ["h"]),
        helper.make_node("Add", ["h", "B"], ["z"]),
        helper.make_node("Sigmoid", ["z"], ["Y"]),
    ]
    model = _make_model(nodes, [_vi("X", [16, 7])], [_vi("Y", [16, 1])], [_init("W", w), _init("B", b)])
    _check_single(model, {"X": x}, tmp_path)


@pytest.mark.parametrize(
    "transA,transB,alpha,beta,c_shape",
    [
        (0, 0, 1.0, 1.0, (6,)),
        (0, 1, 2.0, 0.5, (4, 6)),
        (1, 0, 1.0, 1.0, (1, 6)),
        (1, 1, 0.5, 2.0, (4, 1)),
        (0, 1, 1.0, 1.0, ()),
        (0, 0, 1.0, 1.0, None),
    ],
)
def test_gemm(transA, transB, alpha, beta, c_shape, tmp_path):
    M, K, N = 4, 5, 6
    a = _rand(*((K, M) if transA else (M, K)), seed=5)
    b = _rand(*((N, K) if transB else (K, N)), seed=6)
    inits = [_init("B", b)]
    ins = ["A", "B"]
    if c_shape is not None:
        c = _rand(*c_shape, seed=7) if c_shape else np.array(0.25, dtype=np.float32)
        inits.append(_init("C", c))
        ins.append("C")
    node = helper.make_node("Gemm", ins, ["Y"], alpha=alpha, beta=beta, transA=transA, transB=transB)
    model = _make_model([node], [_vi("A", a.shape)], [_vi("Y", [M, N])], inits)
    _check_single(model, {"A": a}, tmp_path)


# --------------------------------------------------------------------------
# Conv / pooling / norm
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "cin,cout,k,stride,pad,group,bias",
    [
        (3, 8, 3, 1, 1, 1, True),
        (3, 8, 3, 2, 1, 1, False),
        (4, 8, 1, 1, 0, 1, True),
        (8, 8, 3, 1, 1, 8, True),   # depthwise
        (4, 6, 3, 1, 0, 2, True),   # grouped
        (3, 4, 5, 2, 2, 1, True),
        (3, 4, (3, 1), 1, (1, 0), 1, True),  # non-square kernel
    ],
)
def test_conv(cin, cout, k, stride, pad, group, bias, tmp_path):
    kh, kw = (k, k) if isinstance(k, int) else k
    ph, pw = (pad, pad) if isinstance(pad, int) else pad
    x = _rand(2, cin, 9, 11, seed=8)
    w = _rand(cout, cin // group, kh, kw, seed=9)
    inits = [_init("W", w)]
    ins = ["X", "W"]
    if bias:
        inits.append(_init("Bc", _rand(cout, seed=10)))
        ins.append("Bc")
    node = helper.make_node(
        "Conv", ins, ["Y"], kernel_shape=[kh, kw], strides=[stride, stride],
        pads=[ph, pw, ph, pw], group=group,
    )
    model = _make_model([node], [_vi("X", x.shape)], [_vi("Y", None)], inits)
    _check_single(model, {"X": x}, tmp_path)


@pytest.mark.parametrize(
    "op,k,s,p,extra",
    [
        ("MaxPool", 2, 2, 0, {}),
        ("MaxPool", 3, 2, 1, {}),
        ("AveragePool", 2, 2, 0, {}),
        ("AveragePool", 3, 1, 1, {}),                        # count_include_pad=0 (default)
        ("AveragePool", 3, 1, 1, {"count_include_pad": 1}),
    ],
)
def test_pooling(op, k, s, p, extra, tmp_path):
    x = _rand(2, 3, 8, 9, seed=11)
    node = helper.make_node(op, ["X"], ["Y"], kernel_shape=[k, k], strides=[s, s], pads=[p, p, p, p], **extra)
    model = _make_model([node], [_vi("X", x.shape)], [_vi("Y", None)])
    _check_single(model, {"X": x}, tmp_path)


def test_global_average_pool(tmp_path):
    x = _rand(2, 5, 7, 7)
    node = helper.make_node("GlobalAveragePool", ["X"], ["Y"])
    model = _make_model([node], [_vi("X", x.shape)], [_vi("Y", [2, 5, 1, 1])])
    _check_single(model, {"X": x}, tmp_path)


def test_batchnorm(tmp_path):
    C = 4
    x = _rand(2, C, 5, 5)
    rng = np.random.default_rng(12)
    inits = [
        _init("s", rng.standard_normal(C)),
        _init("b", rng.standard_normal(C)),
        _init("m", rng.standard_normal(C)),
        _init("v", rng.uniform(0.5, 2.0, C)),
    ]
    node = helper.make_node("BatchNormalization", ["X", "s", "b", "m", "v"], ["Y"], epsilon=1e-3)
    model = _make_model([node], [_vi("X", x.shape)], [_vi("Y", x.shape)], inits)
    _check_single(model, {"X": x}, tmp_path)


# --------------------------------------------------------------------------
# shape ops
# --------------------------------------------------------------------------

@pytest.mark.parametrize("axis", [0, 1, 2, 3, -1, -3])
def test_flatten(axis, tmp_path):
    x = _rand(2, 3, 4, 5)
    node = helper.make_node("Flatten", ["X"], ["Y"], axis=axis)
    model = _make_model([node], [_vi("X", x.shape)], [_vi("Y", None)])
    _check_single(model, {"X": x}, tmp_path)


@pytest.mark.parametrize("target", [[2, -1], [0, 12, -1], [-1], [4, 0, 2, 0]])
def test_reshape(target, tmp_path):
    x = _rand(4, 6, 2, 5)[:, :, :, :] if len(target) == 4 else _rand(2, 12, 5)
    shape_init = numpy_helper.from_array(np.array(target, dtype=np.int64), "shape")
    expected_out = None
    node = helper.make_node("Reshape", ["X", "shape"], ["Y"])
    model = _make_model([node], [_vi("X", x.shape)], [_vi("Y", expected_out)], [shape_init])
    _check_single(model, {"X": x}, tmp_path)


@pytest.mark.parametrize("axis", [0, 1, 2, -1])
def test_concat(axis, tmp_path):
    a = _rand(2, 3, 4, seed=13)
    b_shape = list(a.shape)
    b_shape[axis] = 5
    b = _rand(*b_shape, seed=14)
    node = helper.make_node("Concat", ["A", "B"], ["Y"], axis=axis)
    model = _make_model([node], [_vi("A", a.shape)], [_vi("Y", None)], [_init("B", b)])
    _check_single(model, {"A": a}, tmp_path)


def test_scalar_initializer_loads(tmp_path):
    # Rank-0 initializers are common in PyTorch exports (e.g. `x + 1.0`).
    x = _rand(3, 4)
    node = helper.make_node("Add", ["X", "one"], ["Y"])
    model = _make_model([node], [_vi("X", x.shape)], [_vi("Y", x.shape)], [_init("one", 1.0)])
    _check_single(model, {"X": x}, tmp_path)


# --------------------------------------------------------------------------
# explicit rejection of unsupported features (never silently wrong)
# --------------------------------------------------------------------------

def _expect_reject(model, feeds, tmp_path, match):
    path = tmp_path / "bad.onnx"
    onnx.save(model, str(path))
    with pytest.raises(Exception, match=match):
        m = crucible_py.load_model(str(path))
        crucible_py.run(m, feeds)


def test_rejects_unsupported_op(tmp_path):
    x = _rand(2, 3)
    node = helper.make_node("Mul", ["X", "X"], ["Y"])
    model = _make_model([node], [_vi("X", x.shape)], [_vi("Y", x.shape)])
    _expect_reject(model, {"X": x}, tmp_path, "Mul")


def test_rejects_dilated_conv(tmp_path):
    x = _rand(1, 2, 8, 8)
    w = _rand(2, 2, 3, 3)
    node = helper.make_node("Conv", ["X", "W"], ["Y"], dilations=[2, 2])
    model = _make_model([node], [_vi("X", x.shape)], [_vi("Y", None)], [_init("W", w)])
    _expect_reject(model, {"X": x}, tmp_path, "dilations")


def test_rejects_asymmetric_pads(tmp_path):
    x = _rand(1, 2, 8, 8)
    w = _rand(2, 2, 3, 3)
    node = helper.make_node("Conv", ["X", "W"], ["Y"], pads=[0, 0, 1, 1])
    model = _make_model([node], [_vi("X", x.shape)], [_vi("Y", None)], [_init("W", w)])
    _expect_reject(model, {"X": x}, tmp_path, "pads")


def test_rejects_ceil_mode(tmp_path):
    x = _rand(1, 2, 7, 7)
    node = helper.make_node("MaxPool", ["X"], ["Y"], kernel_shape=[2, 2], strides=[2, 2], ceil_mode=1)
    model = _make_model([node], [_vi("X", x.shape)], [_vi("Y", None)])
    _expect_reject(model, {"X": x}, tmp_path, "ceil_mode")


def test_rejects_wrong_input_shape(tmp_path):
    x = _rand(2, 3)
    node = helper.make_node("Relu", ["X"], ["Y"])
    model = _make_model([node], [_vi("X", [2, 3])], [_vi("Y", [2, 3])])
    _expect_reject(model, {"X": _rand(3, 3)}, tmp_path, "shape")


# --------------------------------------------------------------------------
# real models
# --------------------------------------------------------------------------

def test_fraud_model(tmp_path):
    path = REPO / "models" / "fraud" / "fraud_detector.onnx"
    x = _rand(32, 7, seed=21)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    expected = sess.run(None, {"input": x})[0]
    actual = crucible_py.run(crucible_py.load_model(str(path)), {"input": x})
    np.testing.assert_allclose(actual, expected, rtol=OP_RTOL, atol=OP_ATOL)


@pytest.mark.parametrize("name", ["resnet18", "mobilenet_v2"])
def test_imagenet_models(name):
    path = REPO / "models" / f"{name}.onnx"
    x = np.random.default_rng(0).uniform(0, 1, (1, 3, 224, 224)).astype(np.float32)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    expected = sess.run(None, {in_name: x})[0]
    actual = crucible_py.run(crucible_py.load_model(str(path)), {in_name: x})
    assert actual.shape == expected.shape == (1, 1000)
    np.testing.assert_allclose(actual, expected, rtol=MODEL_RTOL, atol=MODEL_ATOL)
    assert int(actual.argmax()) == int(expected.argmax())


def test_model_info_reports_io_and_ops():
    m = crucible_py.load_model(str(REPO / "models" / "resnet18.onnx"))
    info = crucible_py.get_model_info(m)
    assert info["input_names"] == ["data"]
    assert info["inputs"][0]["shape"] == [-1, 3, 224, 224]
    assert info["opset"] == 7
    assert set(info["ops_used"]) <= set(crucible_py.supported_ops())
