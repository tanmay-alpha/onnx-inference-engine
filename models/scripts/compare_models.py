"""
Compare two ONNX models on identical test data.

Outputs a comparison report covering:
  - Latency (mean, median, p95, p99)
  - Output shape and value ranges
  - Numerical agreement between models
  - Memory usage (approximate)

Usage:
    python models/scripts/compare_models.py \\
        models/registry/fraud_detector-v1.onnx \\
        models/registry/fraud_detector-v2.onnx \\
        --test-data data/processed/test.csv

Or with synthetic data:
    python models/scripts/compare_models.py model_a.onnx model_b.onnx
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ModelBenchmark:
    """Benchmark results for a single ONNX model."""

    name: str
    path: str
    n_runs: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    output_shape: list[int]
    output_min: float
    output_max: float
    output_mean: float
    output_std: float
    memory_mb: float


@dataclass
class ComparisonReport:
    """Full comparison between two models."""

    model_a: ModelBenchmark
    model_b: ModelBenchmark
    cosine_similarity: Optional[float] = None
    max_abs_diff: Optional[float] = None
    mean_abs_diff: Optional[float] = None
    latency_winner: Optional[str] = None
    recommendation: str = ""

    def report(self) -> str:
        """Return a human-readable comparison report."""
        lines = [
            "=" * 60,
            "ONNX MODEL COMPARISON REPORT",
            "=" * 60,
            "",
            f"  Model A: {self.model_a.name}",
            f"    Path:        {self.model_a.path}",
            f"    Output:      {self.model_a.output_shape}",
            f"    Latency:     mean={self.model_a.mean_ms:.3f}ms  "
            f"median={self.model_a.median_ms:.3f}ms  "
            f"p95={self.model_a.p95_ms:.3f}ms  p99={self.model_a.p99_ms:.3f}ms",
            f"    Output range:[{self.model_a.output_min:.4f}, {self.model_a.output_max:.4f}]",
            f"    Memory:      ~{self.model_a.memory_mb:.1f} MB",
            "",
            f"  Model B: {self.model_b.name}",
            f"    Path:        {self.model_b.path}",
            f"    Output:      {self.model_b.output_shape}",
            f"    Latency:     mean={self.model_b.mean_ms:.3f}ms  "
            f"median={self.model_b.median_ms:.3f}ms  "
            f"p95={self.model_b.p95_ms:.3f}ms  p99={self.model_b.p99_ms:.3f}ms",
            f"    Output range:[{self.model_b.output_min:.4f}, {self.model_b.output_max:.4f}]",
            f"    Memory:      ~{self.model_b.memory_mb:.1f} MB",
            "",
        ]

        if self.cosine_similarity is not None:
            lines += [
                "  Agreement:",
                f"    Cosine similarity: {self.cosine_similarity:.6f}",
                f"    Max abs diff:      {self.max_abs_diff:.6f}",
                f"    Mean abs diff:     {self.mean_abs_diff:.6f}",
            ]

        if self.latency_winner:
            faster = self.model_a if self.latency_winner == "A" else self.model_b
            slower = self.model_b if self.latency_winner == "A" else self.model_a
            speedup = slower.mean_ms / faster.mean_ms
            lines.append(
                f"\n  Latency winner: {self.latency_winner} ({faster.name}) — "
                f"{speedup:.2f}x faster"
            )

        lines += [
            "",
            "  Recommendation:",
            f"    {self.recommendation}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to a dict for JSON export."""
        return {
            "model_a": {
                "name": self.model_a.name,
                "path": self.model_a.path,
                "mean_ms": round(self.model_a.mean_ms, 4),
                "median_ms": round(self.model_a.median_ms, 4),
                "p95_ms": round(self.model_a.p95_ms, 4),
                "p99_ms": round(self.model_a.p99_ms, 4),
                "output_shape": self.model_a.output_shape,
                "memory_mb": round(self.model_a.memory_mb, 1),
            },
            "model_b": {
                "name": self.model_b.name,
                "path": self.model_b.path,
                "mean_ms": round(self.model_b.mean_ms, 4),
                "median_ms": round(self.model_b.median_ms, 4),
                "p95_ms": round(self.model_b.p95_ms, 4),
                "p99_ms": round(self.model_b.p99_ms, 4),
                "output_shape": self.model_b.output_shape,
                "memory_mb": round(self.model_b.memory_mb, 1),
            },
            "agreement": {
                "cosine_similarity": (
                    round(self.cosine_similarity, 6) if self.cosine_similarity else None
                ),
                "max_abs_diff": (
                    round(self.max_abs_diff, 6) if self.max_abs_diff else None
                ),
                "mean_abs_diff": (
                    round(self.mean_abs_diff, 6) if self.mean_abs_diff else None
                ),
            },
            "latency_winner": self.latency_winner,
            "recommendation": self.recommendation,
        }


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def _load_session(model_path: Path) -> ort.InferenceSession:
    """Load an ONNX Runtime inference session."""
    if not ONNX_AVAILABLE:
        raise RuntimeError("onnxruntime not installed. pip install onnxruntime")
    return ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )


def _benchmark_model(
    session: ort.InferenceSession,
    test_input: np.ndarray,
    n_runs: int = 100,
    warmup: int = 10,
) -> ModelBenchmark:
    """Run multiple inference passes and collect timing statistics.

    Args:
        session: ONNX Runtime session.
        test_input: Input array for inference.
        n_runs: Number of timed runs.
        warmup: Number of untimed warmup runs.

    Returns:
        ModelBenchmark with timing and output statistics.
    """
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(warmup):
        session.run(None, {input_name: test_input})

    # Timed runs
    timings: list[float] = []
    outputs_list = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        output = session.run(None, {input_name: test_input})[0]
        elapsed = (time.perf_counter() - t0) * 1000.0
        timings.append(elapsed)
        outputs_list.append(output)

    timings_arr = np.array(timings)
    all_outputs = np.concatenate([o.flatten() for o in outputs_list])

    # Estimate memory: sum of all tensor sizes in the graph
    mem_bytes = sum(
        np.prod([d.dim_value for d in inp.type.tensor_type.shape.dim]) * 4
        for inp in session.get_inputs()
    )
    # Also account for intermediate tensors (rough estimate: 3x input)
    mem_mb = (mem_bytes * 3) / (1024 * 1024)

    return ModelBenchmark(
        name=Path(session._model_path).stem,
        path=str(session._model_path),
        n_runs=n_runs,
        mean_ms=float(timings_arr.mean()),
        median_ms=float(np.median(timings_arr)),
        p95_ms=float(np.percentile(timings_arr, 95)),
        p99_ms=float(np.percentile(timings_arr, 99)),
        min_ms=float(timings_arr.min()),
        max_ms=float(timings_arr.max()),
        output_shape=list(outputs_list[0].shape),
        output_min=float(all_outputs.min()),
        output_max=float(all_outputs.max()),
        output_mean=float(all_outputs.mean()),
        output_std=float(all_outputs.std()),
        memory_mb=mem_mb,
    )


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D arrays."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def compare_two_models(
    model_a_path: Path,
    model_b_path: Path,
    test_input: np.ndarray,
    *,
    n_runs: int = 100,
) -> ComparisonReport:
    """Benchmark and compare two ONNX models.

    Args:
        model_a_path: Path to the first ONNX model.
        model_b_path: Path to the second ONNX model.
        test_input: Input array for inference.
        n_runs: Number of timed inference runs per model.

    Returns:
        ComparisonReport with full results.
    """
    session_a = _load_session(model_a_path)
    session_b = _load_session(model_b_path)

    logger.info("Benchmarking Model A (%s)...", model_a_path.name)
    bench_a = _benchmark_model(session_a, test_input, n_runs=n_runs)

    logger.info("Benchmarking Model B (%s)...", model_b_path.name)
    bench_b = _benchmark_model(session_b, test_input, n_runs=n_runs)

    # Agreement check: run once more to compare outputs
    out_a = session_a.run(None, {session_a.get_inputs()[0].name: test_input})[0]
    out_b = session_b.run(None, {session_b.get_inputs()[0].name: test_input})[0]

    cos_sim = _cosine_similarity(out_a, out_b)
    abs_diff = np.abs(out_a.astype(np.float64) - out_b.astype(np.float64))

    # Latency winner
    if bench_a.mean_ms < bench_b.mean_ms:
        latency_winner = "A"
        speedup = bench_b.mean_ms / bench_a.mean_ms
    else:
        latency_winner = "B"
        speedup = bench_a.mean_ms / bench_b.mean_ms

    # Recommendation
    if cos_sim > 0.999:
        agreement_note = "models produce nearly identical outputs"
    elif cos_sim > 0.95:
        agreement_note = "models produce similar outputs"
    else:
        agreement_note = "models produce meaningfully different outputs"

    recommendation = (
        f"Model {latency_winner} is faster ({speedup:.2f}x). "
        f"{agreement_note} (cosine similarity: {cos_sim:.4f}). "
        f"If outputs are equivalent, choose Model {latency_winner} for lower latency."
    )

    return ComparisonReport(
        model_a=bench_a,
        model_b=bench_b,
        cosine_similarity=cos_sim,
        max_abs_diff=float(abs_diff.max()),
        mean_abs_diff=float(abs_diff.mean()),
        latency_winner=latency_winner,
        recommendation=recommendation,
    )


# ---------------------------------------------------------------------------
# Test data loading
# ---------------------------------------------------------------------------

def load_test_input(
    source: Optional[str] = None,
    n_features: int = 30,
    n_samples: int = 1,
) -> np.ndarray:
    """Load or generate test input for model comparison.

    Args:
        source: Path to a CSV file (last n_features columns are features).
        n_features: Number of features (for synthetic data).
        n_samples: Number of samples to generate (batch size).

    Returns:
        Float32 array of shape ``(n_samples, n_features)``.
    """
    if source:
        with open(source, newline="") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            rows = list(reader)
        data = np.array(rows, dtype=np.float32)
        # Take all columns except last (assumed label)
        X = data[:, :-1]
        # Use a small subset for speed
        return X[:min(n_samples, len(X))]
    else:
        rng = np.random.default_rng(42)
        return rng.standard_normal((n_samples, n_features)).astype(np.float32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two ONNX models on identical test data."
    )
    parser.add_argument("model_a", help="Path to the first ONNX model.")
    parser.add_argument("model_b", help="Path to the second ONNX model.")
    parser.add_argument(
        "--test-data",
        help="Path to CSV test data (optional; generates synthetic data if omitted).",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=30,
        help="Number of features (for synthetic data generation).",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=100,
        help="Number of timed inference runs per model.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    model_a_path = Path(args.model_a)
    model_b_path = Path(args.model_b)

    if not model_a_path.exists():
        logger.error("Model A not found: %s", model_a_path)
        return 1
    if not model_b_path.exists():
        logger.error("Model B not found: %s", model_b_path)
        return 1

    test_input = load_test_input(
        source=args.test_data,
        n_features=args.n_features,
        n_samples=1,
    )
    logger.info("Test input shape: %s", test_input.shape)

    report = compare_two_models(
        model_a_path,
        model_b_path,
        test_input,
        n_runs=args.n_runs,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.report())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
