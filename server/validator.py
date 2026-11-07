"""Static analysis of uploaded ONNX models.

Every upload goes through :func:`analyze_model_file`:

1. ``onnx.load`` (external data is never followed) + ``onnx.checker`` —
   malformed files raise ``ValueError`` (HTTP 400).
2. If the native engine is importable: ``crucible_py.load_model`` +
   ``check_model`` + ``get_model_info`` provide the authoritative support
   verdict, operator list and I/O signature. A model the native loader
   rejects is still stored, marked unsupported with the loader's reason.
3. If the native engine is missing: metadata comes from the ``onnx``
   package and support is judged against :data:`SUPPORTED_OPS`, a static
   mirror of ``crucible_py.supported_ops()``. ``native_checked`` is False
   so callers can tell the verdict is approximate.

No inference ever happens here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import onnx
from onnx import TensorProto

from server import engine

# Mirror of crucible_py.supported_ops() used only when the native module is
# unavailable. A test asserts equality with the native list when it is built.
SUPPORTED_OPS: frozenset[str] = frozenset({
    "Add", "AveragePool", "BatchNormalization", "Concat", "Conv", "Dropout",
    "Elu", "Flatten", "Gelu", "Gemm", "GlobalAveragePool", "Identity",
    "LeakyRelu", "MatMul", "MaxPool", "Relu", "Reshape", "Sigmoid",
    "Softmax", "Tanh",
})

_DTYPE_NAMES = {
    TensorProto.FLOAT: "float32",
    TensorProto.UINT8: "uint8",
    TensorProto.INT8: "int8",
    TensorProto.UINT16: "uint16",
    TensorProto.INT16: "int16",
    TensorProto.INT32: "int32",
    TensorProto.INT64: "int64",
    TensorProto.STRING: "string",
    TensorProto.BOOL: "bool",
    TensorProto.FLOAT16: "float16",
    TensorProto.DOUBLE: "float64",
    TensorProto.UINT32: "uint32",
    TensorProto.UINT64: "uint64",
    TensorProto.BFLOAT16: "bfloat16",
}


def dtype_name(elem_type: int) -> str:
    if elem_type in _DTYPE_NAMES:
        return _DTYPE_NAMES[elem_type]
    try:
        return TensorProto.DataType.Name(elem_type).lower()
    except ValueError:
        return f"elem_type_{elem_type}"


@dataclass
class ModelAnalysis:
    opset: Optional[int]
    operators_used: List[str]
    unsupported_ops: List[str]
    unsupported_reasons: List[str]
    inputs: List[Dict[str, Any]]
    outputs: List[Dict[str, Any]]
    native_checked: bool
    all_supported: bool = field(init=False)

    def __post_init__(self) -> None:
        self.all_supported = not self.unsupported_ops and not self.unsupported_reasons


def extract_op_types(model: onnx.ModelProto) -> List[str]:
    """Distinct op_types in first-appearance order."""
    seen: set[str] = set()
    out: List[str] = []
    for node in model.graph.node:
        if node.op_type not in seen:
            seen.add(node.op_type)
            out.append(node.op_type)
    return out


def _default_opset(model: onnx.ModelProto) -> Optional[int]:
    for imp in model.opset_import:
        if imp.domain in ("", "ai.onnx"):
            return int(imp.version)
    return None


def _value_info_spec(vi: onnx.ValueInfoProto) -> Dict[str, Any]:
    tt = vi.type.tensor_type
    shape: Optional[List[int]] = None
    if tt.HasField("shape"):
        shape = [int(d.dim_value) if d.HasField("dim_value") else -1 for d in tt.shape.dim]
    return {"name": vi.name, "shape": shape, "dtype": dtype_name(tt.elem_type)}


def _native_spec(d: Dict[str, Any]) -> Dict[str, Any]:
    shape = d.get("shape")
    return {
        "name": str(d["name"]),
        "shape": None if shape is None else [int(x) for x in shape],
        "dtype": dtype_name(int(d.get("elem_type", 0))),
    }


def _uses_external_data(model: onnx.ModelProto) -> bool:
    return any(t.data_location == TensorProto.EXTERNAL for t in model.graph.initializer)


def analyze_model_file(path: Path) -> ModelAnalysis:
    """Validate and describe the ONNX file at ``path``. Raises ValueError if malformed."""
    try:
        proto = onnx.load(str(path), load_external_data=False)
    except Exception as exc:  # noqa: BLE001 - any parse failure is a client error
        raise ValueError(f"Invalid ONNX (could not parse): {type(exc).__name__}") from exc
    try:
        onnx.checker.check_model(proto)
    except onnx.checker.ValidationError as exc:
        raise ValueError(f"Invalid ONNX (checker): {exc}") from exc

    initializer_names = {t.name for t in proto.graph.initializer}
    onnx_inputs = [_value_info_spec(v) for v in proto.graph.input if v.name not in initializer_names]
    onnx_outputs = [_value_info_spec(v) for v in proto.graph.output]
    onnx_ops = extract_op_types(proto)
    opset = _default_opset(proto)

    native = engine.native_module()
    if native is None:
        unsupported = sorted(set(onnx_ops) - SUPPORTED_OPS)
        reasons = [f"operator {op} is not implemented by Crucible" for op in unsupported]
        if _uses_external_data(proto):
            reasons.append("initializers stored as external data are not supported")
        return ModelAnalysis(opset, onnx_ops, unsupported, reasons, onnx_inputs, onnx_outputs,
                             native_checked=False)

    native_ops = set(native.supported_ops())
    unsupported = sorted(set(onnx_ops) - native_ops)
    try:
        model = native.load_model(str(path))
    except (RuntimeError, ValueError) as exc:
        return ModelAnalysis(opset, onnx_ops, unsupported,
                             [f"native loader rejected the model: {exc}"],
                             onnx_inputs, onnx_outputs, native_checked=True)

    reasons = [str(r) for r in native.check_model(model)]
    info = native.get_model_info(model)
    ops = [str(op) for op in info.get("ops_used", onnx_ops)]
    native_opset = info.get("opset")
    return ModelAnalysis(
        opset=int(native_opset) if native_opset is not None and int(native_opset) > 0 else opset,
        operators_used=ops,
        unsupported_ops=sorted(set(ops) - native_ops),
        unsupported_reasons=reasons,
        inputs=[_native_spec(d) for d in info.get("inputs", [])],
        outputs=[_native_spec(d) for d in info.get("outputs", [])],
        native_checked=True,
    )
