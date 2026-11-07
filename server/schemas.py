"""Pydantic v2 request/response models for the Crucible server API."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Base(BaseModel):
    # Fields such as model_id / model_file would otherwise trip Pydantic's
    # reserved "model_" namespace warning.
    model_config = ConfigDict(protected_namespaces=())


class TensorSpec(_Base):
    name: str
    shape: Optional[List[int]] = Field(..., description="Dims; -1 = dynamic; null = unknown rank")
    dtype: str = Field(..., description="e.g. float32, int64")


class ModelAnalysisFields(_Base):
    opset: Optional[int] = Field(..., description="Default-domain opset, if declared")
    operators_used: List[str] = Field(..., description="Distinct op_types, first-appearance order")
    all_supported: bool = Field(..., description="True iff the engine reports no blocking issue")
    unsupported_ops: List[str] = Field(default_factory=list)
    unsupported_reasons: List[str] = Field(
        default_factory=list, description="Human-readable reasons the model cannot run on Crucible"
    )
    inputs: List[TensorSpec]
    outputs: List[TensorSpec]


# ---------------------------------------------------------------------------
# /convert, /validate, /models
# ---------------------------------------------------------------------------
class ConvertResponse(ModelAnalysisFields):
    onnx_model_id: str = Field(..., description="Id of the stored model (owned by the caller)")
    name: str
    size_bytes: int


class ValidateResponse(_Base):
    valid: bool = Field(..., description="Same as all_supported")
    operators: List[str]
    unsupported: List[str] = Field(default_factory=list)
    unsupported_reasons: List[str] = Field(default_factory=list)
    opset: Optional[int] = None
    inputs: List[TensorSpec] = Field(default_factory=list)
    outputs: List[TensorSpec] = Field(default_factory=list)


class ModelItem(ModelAnalysisFields):
    id: str
    name: str
    size_bytes: int
    created_at: str


class ModelListResponse(_Base):
    models: List[ModelItem]
    count: int


# ---------------------------------------------------------------------------
# /infer
# ---------------------------------------------------------------------------
class InferRequest(_Base):
    model_id: str = Field(..., description="Id returned by /convert")
    input: List[float] = Field(..., description="Flattened float32 input, row-major. NaN/Inf rejected.")
    input_shape: List[int] = Field(..., description="Shape of `input`, e.g. [1, 3, 224, 224]")

    @field_validator("input_shape")
    @classmethod
    def _validate_input_shape(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError("input_shape must be non-empty")
        if len(v) > 8:
            raise ValueError("input_shape rank must be <= 8")
        if any(d <= 0 for d in v):
            raise ValueError("input_shape dims must be positive")
        if any(d >= 2**31 for d in v):
            raise ValueError("input_shape dims must be < 2^31")
        return v


class InferResponse(_Base):
    model_id: str
    output: List[float] = Field(..., description="First model output, flattened")
    output_shape: List[int]
    inference_time_ms: float = Field(..., description="Time inside crucible_py.run only (perf_counter)")
    backend: Literal["crucible-native"] = "crucible-native"
    engine: Literal["crucible-native"] = "crucible-native"


class BatchItemResult(_Base):
    model_id: str
    status: Literal["success", "error"]
    output: Optional[List[float]] = None
    output_shape: Optional[List[int]] = None
    inference_time_ms: Optional[float] = None
    backend: Literal["crucible-native"] = "crucible-native"
    error: Optional[str] = None
    status_code: Optional[int] = Field(default=None, description="HTTP status the item would have produced")


class BatchResponse(_Base):
    batch_size: int
    success_count: int
    error_count: int
    results: List[BatchItemResult]


# ---------------------------------------------------------------------------
# /inference/logs
# ---------------------------------------------------------------------------
class InferenceLogItem(_Base):
    id: str
    model_id: str
    latency_ms: float
    backend: str
    input_shape: List[int]
    output_shape: List[int]
    status: str
    created_at: str


class InferenceLogListResponse(_Base):
    logs: List[InferenceLogItem]
    count: int


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------
class OperatorsResponse(_Base):
    supported: List[str]
    count: int


class HealthResponse(_Base):
    status: str = "ok"
    engine: Literal["crucible-native", "unavailable"]
    native_available: bool
    version: str


class ReadyResponse(HealthResponse):
    database: bool


# ---------------------------------------------------------------------------
# Fraud check history (scores are computed client-side, e.g. in WASM)
# ---------------------------------------------------------------------------
class FraudTxRequest(_Base):
    tx_type: str = Field(default="", max_length=64)
    amount: float = 0.0
    orig_before: float = 0.0
    orig_after: float = 0.0
    dest_before: float = 0.0
    dest_after: float = 0.0
    probability: float = Field(default=0.0, ge=0.0, le=1.0)
    verdict: str = Field(default="", max_length=64)
    execution_mode: str = Field(default="wasm", max_length=32)
    latency_ms: float = Field(default=0.0, ge=0.0)


class FraudTxItem(FraudTxRequest):
    id: str
    created_at: str


class FraudHistoryResponse(_Base):
    history: List[FraudTxItem]
    count: int
