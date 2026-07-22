"""Prometheus metrics for Crucible inference server.

Exposes GET /metrics endpoint with:
  - crucible_requests_total — counter of all inference requests
  - crucible_request_duration_seconds — histogram of inference latency
  - crucible_errors_total — counter of failed requests by error type
  - crucible_fraud_detections_total — counter of fraud detections by risk level
  - crucible_active_models — gauge of currently loaded models
  - crucible_fraud_confidence — histogram of fraud prediction confidence scores
"""
from __future__ import annotations

from typing import Optional

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    REGISTRY,
    generate_latest,
    multiprocess,
)
from starlette.responses import Response

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
_inference_requests_total = Counter(
    "crucible_inference_requests_total",
    "Total number of inference requests",
    ["model_id", "engine", "status"],
)

_inference_duration = Histogram(
    "crucible_inference_duration_seconds",
    "Inference request latency in seconds",
    ["model_id", "engine"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

_errors_total = Counter(
    "crucible_errors_total",
    "Total number of errors by type",
    ["error_type"],
)

_fraud_detections_total = Counter(
    "crucible_fraud_detections_total",
    "Total number of fraud detections by risk level",
    ["risk_level", "is_fraud"],
)

_active_models = Gauge(
    "crucible_active_models",
    "Number of currently active/loaded models",
)

_fraud_confidence = Histogram(
    "crucible_fraud_confidence",
    "Distribution of fraud prediction confidence scores",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)

_upload_bytes_total = Counter(
    "crucible_upload_bytes_total",
    "Total bytes uploaded for model conversion",
)

_batch_jobs_total = Counter(
    "crucible_batch_jobs_total",
    "Total batch inference jobs",
    ["status"],
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def record_inference(model_id: str, engine: str, latency_ms: float, status: str = "success") -> None:
    _inference_requests_total.labels(model_id=model_id, engine=engine, status=status).inc()
    _inference_duration.labels(model_id=model_id, engine=engine).observe(latency_ms / 1000.0)
    if status != "success":
        _errors_total.labels(error_type="inference").inc()


def record_error(error_type: str) -> None:
    _errors_total.labels(error_type=error_type).inc()


def record_fraud_detection(risk_level: str, is_fraud: bool, probability: float) -> None:
    _fraud_detections_total.labels(risk_level=risk_level, is_fraud=str(is_fraud)).inc()
    _fraud_confidence.observe(probability)


def set_active_models(count: int) -> None:
    _active_models.set(count)


def record_upload(num_bytes: int) -> None:
    _upload_bytes_total.inc(num_bytes)


def record_batch_job(status: str) -> None:
    _batch_jobs_total.labels(status=status).inc()


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
def metrics_response() -> Response:
    """Return Prometheus metrics as plain text."""
    data = generate_latest(REGISTRY)
    return Response(content=data, media_type="text/plain; version=0.0.4")
