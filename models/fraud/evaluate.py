"""
Model evaluation for the fraud detection pipeline.

Computes precision, recall, F1, AUC-ROC, confusion matrix, and compares
against three baselines: random guessing, always-fraud, and never-fraud.

Usage:
    python fraud/evaluate.py [--model path/to/model.onnx] [--config path/to/model_config.json]

Or programmatically:
    from fraud.evaluate import evaluate_model
    report = evaluate_model(
        model_path="models/fraud/fraud_detector.onnx",
        config_path="models/fraud/model_config.json",
        X_test=X_test, y_test=y_test,
    )
    print(report.summary())
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

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
class EvalMetrics:
    """Container for all evaluation metrics."""

    precision: float
    recall: float
    f1: float
    auc_roc: float
    confusion: dict[str, int] = field(default_factory=dict)
    n_samples: int = 0
    inference_ms: float = 0.0
    threshold: float = 0.5

    def summary(self) -> str:
        """Return a human-readable report string."""
        lines = [
            "=" * 50,
            "MODEL EVALUATION REPORT",
            "=" * 50,
            f"  Samples:       {self.n_samples}",
            f"  Precision:     {self.precision:.4f}",
            f"  Recall:        {self.recall:.4f}",
            f"  F1 Score:      {self.f1:.4f}",
            f"  AUC-ROC:       {self.auc_roc:.4f}",
            f"  Threshold:     {self.threshold}",
            f"  Inference:     {self.inference_ms:.2f} ms",
            "",
            "  Confusion Matrix:",
            f"    TN={self.confusion.get('tn', 0):,}  FP={self.confusion.get('fp', 0):,}",
            f"    FN={self.confusion.get('fn', 0):,}  TP={self.confusion.get('tp', 0):,}",
            "=" * 50,
        ]
        return "\n".join(lines)


@dataclass
class BaselineResult:
    """Metrics for a baseline strategy."""

    name: str
    precision: float
    recall: float
    f1: float
    auc_roc: float


# ---------------------------------------------------------------------------
# ONNX inference helper
# ---------------------------------------------------------------------------

def _load_onnx_session(model_path: Path) -> "ort.InferenceSession":
    """Create an ONNX Runtime inference session."""
    if not ONNX_AVAILABLE:
        raise RuntimeError(
            "onnxruntime is not installed. Install it with: pip install onnxruntime"
        )
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    return ort.InferenceSession(
        str(model_path), sess_options=sess_options, providers=["CPUExecutionProvider"]
    )


def _predict_onnx(
    session: "ort.InferenceSession",
    X: np.ndarray,
    input_name: str = "input",
) -> np.ndarray:
    """Run batched inference through an ONNX model.

    Args:
        session: ONNX Runtime session.
        X: Float32 array of shape ``(n_samples, n_features)``.
        input_name: ONNX graph input tensor name.

    Returns:
        1-D array of fraud probabilities (floats in [0, 1]).
    """
    raw = session.run(None, {input_name: X.astype(np.float32)})[0]
    probs = raw.flatten().astype(np.float64)

    # If the model outputs logits (values outside [0,1]), apply sigmoid.
    if probs.min() < -1e-6 or probs.max() > 1.0 + 1e-6:
        probs = 1.0 / (1.0 + np.exp(-probs))

    return probs


# ---------------------------------------------------------------------------
# Baseline strategies
# ---------------------------------------------------------------------------

def _random_baseline(y_true: np.ndarray, rng: np.random.Generator) -> BaselineResult:
    """Random guessing baseline."""
    n = len(y_true)
    y_pred = rng.integers(0, 2, size=n)
    y_prob = rng.random(size=n)
    return _compute_baseline_metrics("random_guess", y_true, y_pred, y_prob)


def _always_fraud_baseline(y_true: np.ndarray) -> BaselineResult:
    """Always predict fraud."""
    n = len(y_true)
    y_pred = np.ones(n, dtype=int)
    y_prob = np.ones(n) * 0.5
    return _compute_baseline_metrics("always_fraud", y_true, y_pred, y_prob)


def _never_fraud_baseline(y_true: np.ndarray) -> BaselineResult:
    """Never predict fraud."""
    n = len(y_true)
    y_pred = np.zeros(n, dtype=int)
    y_prob = np.ones(n) * 0.5
    return _compute_baseline_metrics("never_fraud", y_true, y_pred, y_prob)


def _compute_baseline_metrics(
    name: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> BaselineResult:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        auc_val = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc_val = 0.5
    return BaselineResult(name=name, precision=precision, recall=recall, f1=f1, auc_roc=auc_val)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model_path: str | Path,
    config_path: str | Path,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    threshold: float = 0.5,
    input_name: str = "input",
    seed: int = 42,
) -> EvalMetrics:
    """Evaluate an ONNX fraud detection model on test data.

    Args:
        model_path: Path to the ``.onnx`` model file.
        config_path: Path to ``model_config.json`` for metadata.
        X_test: Feature matrix of shape ``(n_samples, n_features)``.
        y_test: Binary labels of shape ``(n_samples,)``.
        threshold: Decision threshold for classifying fraud.
        input_name: ONNX graph input tensor name.
        seed: Random seed for baseline generation.

    Returns:
        EvalMetrics with all computed metrics.
    """
    model_path = Path(model_path)
    config_path = Path(config_path)

    # Load config for context
    with open(config_path) as f:
        config = json.load(f)
    effective_threshold = config.get("threshold", threshold)

    # Load model and run inference
    session = _load_onnx_session(model_path)
    t0 = time.perf_counter()
    probs = _predict_onnx(session, X_test, input_name)
    inference_ms = (time.perf_counter() - t0) * 1000.0

    y_pred = (probs >= effective_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    try:
        auc_val = roc_auc_score(y_test, probs)
    except ValueError:
        auc_val = 0.5

    logger.info(
        "Evaluated %d samples in %.2f ms — P=%.4f R=%.4f F1=%.4f AUC=%.4f",
        len(y_test),
        inference_ms,
        precision,
        recall,
        f1,
        auc_val,
    )

    return EvalMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        auc_roc=auc_val,
        confusion={"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        n_samples=len(y_test),
        inference_ms=inference_ms,
        threshold=effective_threshold,
    )


def evaluate_with_baselines(
    model_path: str | Path,
    config_path: str | Path,
    X_test: np.ndarray,
    y_test: np.ndarray,
    **kwargs,
) -> tuple[EvalMetrics, list[BaselineResult]]:
    """Evaluate model and compare against baselines.

    Args:
        model_path: Path to the ONNX model.
        config_path: Path to ``model_config.json``.
        X_test: Test feature matrix.
        y_test: Test labels.
        **kwargs: Forwarded to :func:`evaluate_model`.

    Returns:
        Tuple of (model metrics, list of baseline results).
    """
    model_metrics = evaluate_model(model_path, config_path, X_test, y_test, **kwargs)
    rng = np.random.default_rng(42)
    baselines = [
        _random_baseline(y_test, rng),
        _always_fraud_baseline(y_test),
        _never_fraud_baseline(y_test),
    ]
    return model_metrics, baselines


def full_report(
    model_path: str | Path,
    config_path: str | Path,
    X_test: np.ndarray,
    y_test: np.ndarray,
    **kwargs,
) -> str:
    """Generate a complete evaluation report including baselines.

    Args:
        model_path: Path to the ONNX model.
        config_path: Path to ``model_config.json``.
        X_test: Test feature matrix.
        y_test: Test labels.
        **kwargs: Forwarded to :func:`evaluate_model`.

    Returns:
        Multi-line report string.
    """
    model_metrics, baselines = evaluate_with_baselines(
        model_path, config_path, X_test, y_test, **kwargs
    )

    lines = [model_metrics.summary(), "", "BASELINE COMPARISON", "-" * 50]
    for b in baselines:
        lines.append(
            f"  {b.name:20s}  P={b.precision:.4f}  R={b.recall:.4f}  "
            f"F1={b.f1:.4f}  AUC={b.auc_roc:.4f}"
        )
    lines.append(
        f"\n  Model vs. best baseline F1: "
        f"{model_metrics.f1:.4f} vs {max(b.f1 for b in baselines):.4f}"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_synthetic_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate a small synthetic test set for evaluation without external data."""
    rng = np.random.default_rng(42)
    N = 5000
    n_fraud = int(N * 0.013)
    X = np.random.randn(N, 30).astype(np.float32)
    y = np.zeros(N, dtype=int)
    fraud_idx = rng.choice(N, n_fraud, replace=False)
    y[fraud_idx] = 1
    # Add some signal
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    X[y == 1] += 0.5  # shift fraud samples
    return X, y


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a fraud detection ONNX model."
    )
    parser.add_argument(
        "--model",
        default=str(Path(__file__).parent / "fraud_detector.onnx"),
        help="Path to the ONNX model file.",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "model_config.json"),
        help="Path to model_config.json.",
    )
    parser.add_argument("--data", help="Path to test data CSV (optional).")
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Classification threshold."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.data:
        # Load from CSV — expects 30 feature columns, last column is label
        import csv

        with open(args.data) as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        data = np.array(rows, dtype=np.float32)
        X_test = data[:, :-1]
        y_test = data[:, -1].astype(int)
    else:
        logger.info("No --data provided; using synthetic test set.")
        X_test, y_test = _build_synthetic_data()

    print(full_report(args.model, args.config, X_test, y_test, threshold=args.threshold))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
