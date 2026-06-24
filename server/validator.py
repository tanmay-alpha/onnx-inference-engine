"""ONNX structural validation for the Crucible server.

The server has two validation surfaces:

  * /operators returns the static catalogue of op_types Crucible
    can dispatch (a Python frozenset baked into this file). It is
    small and stable — it matches the if/else chain in
    engine/src/executor.cpp. The /operators test asserts the count
    is 20, which is the same number as the C++ dispatcher's cases.

  * /validate parses an uploaded ONNX file (or a previously saved
    model_id), walks its nodes, and reports which op_types appear
    and which of those are unsupported.

We use the official `onnx` package to parse the model rather than
re-implementing the protobuf wire format in Python. The C++ engine
uses a hand-rolled reader (see onnx_parser.hpp), but here the cost
of one extra dep is fine — onnx is already in requirements.txt
because the /convert endpoint needs it for the PyTorch exporter.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import onnx


# ---------------------------------------------------------------------------
# The dispatch table. Keep this in sync with executor.cpp's if/else chain.
# ---------------------------------------------------------------------------
# Adding an op to the C++ dispatcher is a one-line change; this set is
# the single source of truth on the Python side. We use a frozenset
# rather than a list because /operators calls len() on it and iterates
# it; both are O(n) on a frozenset, but membership checks (which we
# do for every node in /validate) are O(1).
SUPPORTED_OPS: frozenset[str] = frozenset({
    # Linear / matmul
    "MatMul",
    "Gemm",
    # Activations
    "Relu",
    "Sigmoid",
    "Softmax",
    "Gelu",
    "Tanh",
    "LeakyRelu",
    "Elu",
    # Convolution / pooling
    "Conv",
    "MaxPool",
    "AveragePool",
    "GlobalAveragePool",
    # Normalisation
    "BatchNormalization",
    # Elementwise
    "Add",
    "Concat",
    # Shape / metadata
    "Flatten",
    "Reshape",
    # No-ops (pass-through, used by MobileNetV2 dropout in inference mode)
    "Identity",
    "Dropout",
})


@dataclass(frozen=True)
class ValidationResult:
    """Result of /validate.

    We use a frozen dataclass rather than a Pydantic model here
    because (a) it never crosses the JSON boundary (main.py converts
    it to ValidateResponse), and (b) Pydantic models are 10-50x
    slower to construct than dataclasses.
    """
    valid: bool
    operators: List[str]       # distinct op_types in first-appearance order
    unsupported: List[str]     # operators \ SUPPORTED_OPS, sorted for stability


def list_supported_ops() -> List[str]:
    """Sorted list of supported op_type strings.

    Sorted so the JSON response is stable across calls. /operators
    tests assert on the sorted form.
    """
    return sorted(SUPPORTED_OPS)


def supported_ops_count() -> int:
    """len(SUPPORTED_OPS). Convenience for callers that don't want
    to import the frozenset directly."""
    return len(SUPPORTED_OPS)


def extract_op_types(model: onnx.ModelProto) -> List[str]:
    """Walk the model's nodes and return distinct op_types in
    first-appearance order.

    Why first-appearance order and not alphabetical?
        MobileNetV2 starts with Conv and ends with Gemm — first-
        appearance order reads top-to-bottom like the source graph,
        which is what humans want when skimming the /validate
        output. Alphabetical is what tooling wants for diffing;
        those callers can sort() the result themselves.
    """
    seen: set[str] = set()
    out: List[str] = []
    for node in model.graph.node:
        op = node.op_type
        if op not in seen:
            seen.add(op)
            out.append(op)
    return out


def validate_model_bytes(data: bytes) -> ValidationResult:
    """Parse ONNX from a raw byte buffer (multipart upload)."""
    model = onnx.load_from_string(data)
    return _validate_model(model)


def validate_model_path(path: str | Path) -> ValidationResult:
    """Parse ONNX from a filesystem path (model_id lookup)."""
    model = onnx.load(str(path))
    return _validate_model(model)


def _validate_model(model: onnx.ModelProto) -> ValidationResult:
    """Common path: take a parsed ModelProto and produce a result."""
    # onnx.checker.check_model raises on malformed graphs. We catch
    # and translate so the HTTP layer can return 400 instead of 500.
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as exc:
        raise ValueError(f"Malformed ONNX: {exc}") from exc

    ops = extract_op_types(model)
    unsupported = sorted(set(ops) - SUPPORTED_OPS)
    return ValidationResult(
        valid=not unsupported,
        operators=ops,
        unsupported=unsupported,
    )


def partition_ops(ops: Iterable[str]) -> tuple[Sequence[str], Sequence[str]]:
    """Split an iterable of op_types into (supported, unsupported).

    Used by /convert to populate the all_supported / unsupported_ops
    fields of ConvertResponse.
    """
    supported: List[str] = []
    unsupported: List[str] = []
    for op in ops:
        if op in SUPPORTED_OPS:
            supported.append(op)
        else:
            unsupported.append(op)
    return supported, unsupported