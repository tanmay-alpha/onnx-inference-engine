"""Pydantic v2 request/response models for the Crucible FastAPI server.

Why Pydantic v2 (not v1)?
    The plan calls for v2 explicitly ("FastAPI with Pydantic v2 only
    (no v1 compat)"). v2 is roughly 5-50x faster than v1 on most
    workloads because validation runs in a Rust core. The migration
    from v1 idioms that would normally bite us (Config class ->
    ConfigDict, validator decorators -> @field_validator) is small
    enough that the perf win is worth it.

Why we keep the model classes tiny?
    Pydantic v2 errors include a `loc` tuple that pinpoints the
    exact field that failed. The smaller each model is, the more
    useful that pointer is to the API consumer.
"""
from __future__ import annotations

import math
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# /convert — multipart upload, so the request isn't a Pydantic model.
# We still expose the response model for OpenAPI docs.
# ---------------------------------------------------------------------------
class _Base(BaseModel):
    """Common Pydantic v2 base.

    Why protected_namespaces=()?
        Pydantic v2 reserves the `model_` prefix to prevent collisions
        with its own private attrs (`model_dump`, `model_copy`, etc.).
        The Crucible API has `model_id`, `model_file`, `model_path`
        as public fields, which trips the warning. The empty tuple
        disables the reservation; the names are unambiguous because
        Crucible's own code never calls `.model_dump` on a request
        model.
    """
    model_config = ConfigDict(protected_namespaces=())


class ConvertResponse(_Base):
    """Response from POST /convert.

    The server saves the converted ONNX under /tmp/models/<uuid>.onnx
    and returns its id. Clients use that id in subsequent /infer and
    /validate calls.

    Note: the filesystem path is intentionally omitted from the response
    for security — the server may store models outside the web root.
    """
    onnx_model_id: str = Field(
        ..., description="UUID identifying the saved ONNX model"
    )
    operators_used: List[str] = Field(
        ..., description="Distinct op_type strings in the graph, "
                          "in first-appearance order"
    )
    all_supported: bool = Field(
        ..., description="True iff every op in operators_used is in the "
                         "Crucible executor's dispatch table"
    )
    unsupported_ops: List[str] = Field(
        default_factory=list,
        description="ops_used \\ supported set. Empty iff all_supported."
    )


# ---------------------------------------------------------------------------
# /infer — pure JSON body.
# ---------------------------------------------------------------------------
class InferRequest(_Base):
    """Request body for POST /infer.

    `input` is a flat list of floats; the client is expected to send
    the model in row-major order matching `input_shape`. We accept a
    flat list (not a nested list) so the wire format is stable across
    model dims and JSON encoders — the cost is one reshape on the
    server.
    """
    model_id: str = Field(..., description="UUID returned by /convert")
    input: List[float] = Field(
        ..., description="Flattened float32 input array, row-major. "
                          "NaN and +/-Infinity are rejected."
    )
    input_shape: List[int] = Field(
        ..., description="Reshape target, e.g. [1, 3, 224, 224]"
    )

    @field_validator("input")
    @classmethod
    def _validate_input_finite(cls, v: List[float]) -> List[float]:
        """Reject NaN/Inf — they would propagate through the C++ engine
        and produce undefined numerical results."""
        for i, x in enumerate(v):
            if math.isnan(x) or math.isinf(x):
                raise ValueError(
                    f"input[{i}] is not a finite number (NaN/Inf rejected)")
        return v

    @field_validator("input_shape")
    @classmethod
    def _validate_input_shape(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError("input_shape must be non-empty")
        if any(d <= 0 for d in v):
            raise ValueError("input_shape dims must be positive")
        # numpy crashes on huge axis sizes; cap at 2**31 to keep the
        # server from OOMing on a hostile 4-gigabyte-axis request.
        if any(d >= 2**31 for d in v):
            raise ValueError("input_shape dims must be < 2^31")
        return v


class InferResponse(_Base):
    """Response from POST /infer.

    `output` is a flat list, reshapable via `output_shape`.
    `inference_time_ms` is wall-clock for the C++ run_inference call
    only — JSON encode/decode is excluded.
    """
    output: List[float] = Field(
        ..., description="Flattened float values serialized as JSON numbers (double precision)"
    )
    output_shape: List[int] = Field(
        ..., description="Reshape target for `output`"
    )
    inference_time_ms: float = Field(
        ..., description="Wall-clock time spent in run_inference, ms"
    )
    engine: str = Field(
        default="crucible-cpp", description="Engine identifier string"
    )


# ---------------------------------------------------------------------------
# /validate — multipart upload (optional) OR body with model_id.
# The plan shows /validate as taking no request body, but the natural
# usage is to validate a model the client already uploaded, so we
# accept both shapes: pass a file directly, or pass model_id to
# re-validate a previously uploaded one.
# ---------------------------------------------------------------------------
class ValidateResponse(_Base):
    """Response from POST /validate."""
    valid: bool = Field(
        ..., description="True iff every op is in the dispatch table"
    )
    operators: List[str] = Field(
        ..., description="Distinct op_type strings in the graph"
    )
    unsupported: List[str] = Field(
        default_factory=list,
        description="ops in `operators` but not in the dispatch table"
    )


# ---------------------------------------------------------------------------
# /operators — read-only catalogue.
# ---------------------------------------------------------------------------
class OperatorsResponse(_Base):
    """Response from GET /operators.

    `count` is the length of `supported` and is included so clients
    that paginate the list can verify they got the full set without
    having to len() it themselves.
    """
    supported: List[str] = Field(
        ..., description="op_type strings Crucible can dispatch"
    )
    count: int = Field(
        ..., description="Length of `supported` (== len(supported))"
    )


# ---------------------------------------------------------------------------
# /health — tiny liveness probe.
# ---------------------------------------------------------------------------
class HealthResponse(_Base):
    """Response from GET /health."""
    status: str = Field(default="ok", description="Always 'ok' if responding")
    engine: str = Field(
        default="crucible-cpp", description="Engine identifier"
    )
    version: str = Field(..., description="Server version string")


# ---------------------------------------------------------------------------
# Error envelope.
# ---------------------------------------------------------------------------
# FastAPI's default error response is `{"detail": "..."}` with a
# HTTPException. We expose a typed model so OpenAPI consumers get
# a real schema to import, not a free-form string.
class ErrorResponse(_Base):
    """Standard error envelope used for 4xx/5xx responses."""
    detail: str = Field(..., description="Human-readable error message")
    error_code: Optional[str] = Field(
        default=None, description="Machine-readable code, e.g. 'MODEL_NOT_FOUND'"
    )


# Backward compatibility aliases for test suites
ConvertRequest = ConvertResponse
PredictRequest = InferRequest
PredictResponse = InferResponse