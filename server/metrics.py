"""Prometheus metrics exposed at GET /metrics.

  - crucible_inference_requests_total{engine,status}
  - crucible_inference_duration_seconds{engine}   (engine-only time, successes)
  - crucible_errors_total{error_type}
  - crucible_batch_jobs_total{status}

Model ids are deliberately not used as labels (unbounded cardinality).
"""
from prometheus_client import REGISTRY, Counter, Histogram, generate_latest
from starlette.responses import Response

_inference_requests_total = Counter(
    "crucible_inference_requests_total",
    "Inference requests that reached the native engine",
    ["engine", "status"],
)

_inference_duration = Histogram(
    "crucible_inference_duration_seconds",
    "Time spent inside the native engine run call",
    ["engine"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

_errors_total = Counter("crucible_errors_total", "Errors by type", ["error_type"])

_batch_jobs_total = Counter("crucible_batch_jobs_total", "Batch inference items", ["status"])


def record_inference(engine: str, latency_ms: float, status: str = "success") -> None:
    _inference_requests_total.labels(engine=engine, status=status).inc()
    if status == "success":
        _inference_duration.labels(engine=engine).observe(latency_ms / 1000.0)
    else:
        _errors_total.labels(error_type=f"inference_{status}").inc()


def record_error(error_type: str) -> None:
    _errors_total.labels(error_type=error_type).inc()


def record_batch_job(status: str) -> None:
    _batch_jobs_total.labels(status=status).inc()


def metrics_response() -> Response:
    return Response(content=generate_latest(REGISTRY), media_type="text/plain; version=0.0.4")
