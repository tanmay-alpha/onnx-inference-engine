"""
Scheduled retraining script for the fraud detection model.

This script can be run manually or from a cron job / task scheduler.
It:
  1. Loads the latest transaction data (from CSV or Supabase)
  2. Applies the feature engineering pipeline
  3. Trains a new model with current hyperparameters
  4. Evaluates against the current production model
  5. Promotes to registry if the new model is better
  6. Logs results

Usage:
    python models/scripts/retrain_fraud.py
    python models/scripts/retrain_fraud.py --data-path data/processed/transactions.csv
    python models/scripts/retrain_fraud.py --force-promote  # skip comparison
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _models_dir() -> Path:
    """Return the models/ directory (parent of this script's parent)."""
    return Path(__file__).resolve().parent.parent


def _fraud_dir() -> Path:
    return _models_dir() / "fraud"


def _registry_dir() -> Path:
    return _models_dir() / "registry"


def _next_version(model_name: str = "fraud_detector") -> int:
    """Determine the next version number for a model in the registry."""
    registry = _registry_dir()
    existing = list(registry.glob(f"{model_name}-v*.onnx"))
    if not existing:
        return 1
    versions = []
    for p in existing:
        stem = p.stem  # e.g. "fraud_detector-v3"
        try:
            ver = int(stem.split("-v")[-1])
            versions.append(ver)
        except (ValueError, IndexError):
            continue
    return max(versions) + 1 if versions else 1


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv_data(path: Path, n_features: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Load transaction data from a CSV file.

    The CSV should have feature columns and a final label column.

    Args:
        path: Path to the CSV file.
        n_features: Number of feature columns (excluding label).

    Returns:
        Tuple of (X, y) arrays.
    """
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    if len(rows) == 0:
        raise ValueError(f"CSV file {path} is empty (no data rows)")

    data = np.array(rows, dtype=np.float32)
    if data.shape[1] < n_features + 1:
        raise ValueError(
            f"CSV has {data.shape[1]} columns; expected at least {n_features + 1} "
            f"({n_features} features + 1 label)"
        )

    X = data[:, :n_features]
    y = data[:, n_features].astype(int)
    return X, y


def load_supabase_data(
    table: str = "transactions",
    limit: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load transaction data from Supabase via the project's API.

    This requires the SUPABASE_URL and SUPABASE_ANON_KEY environment
    variables to be set, and the project's API to be running.

    Falls back to generating synthetic data if the API is unavailable.

    Args:
        table: Supabase table name.
        limit: Maximum number of rows to fetch.

    Returns:
        Tuple of (X, y) arrays.
    """
    try:
        from supabase import create_client  # type: ignore
    except ImportError:
        logger.warning("supabase-py not installed; falling back to synthetic data.")
        return _generate_synthetic_data()

    url = "http://localhost:8000"
    key = "anon-key"
    try:
        client = create_client(url, key)
        query = client.table(table).select("*")
        if limit:
            query = query.limit(limit)
        response = query.execute()
        rows = response.data
        if not rows:
            raise ValueError(f"No data returned from Supabase table '{table}'")

        # Assumes keys V1..V28, Amount, Time, is_fraud
        feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
        X = np.array([[row.get(c, 0.0) for c in feature_cols] for row in rows], dtype=np.float32)
        y = np.array([row.get("is_fraud", 0) for row in rows], dtype=int)
        return X, y

    except Exception as exc:
        logger.warning("Supabase fetch failed (%s); using synthetic data.", exc)
        return _generate_synthetic_data()


def _generate_synthetic_data(
    n_samples: int = 100_000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic PaySim-style transaction data for retraining."""
    rng = np.random.default_rng(random_state)
    n_fraud = int(n_samples * 0.013)
    n_legit = n_samples - n_fraud

    # Generate 30 features (V1-V28 + Amount + Time)
    legit_features = rng.standard_normal((n_legit, 28)).astype(np.float32) * 10
    legit_amount = rng.exponential(5_000, n_legit).astype(np.float32)
    legit_time = rng.uniform(0, 172_800, n_legit).astype(np.float32)  # 48 hours

    fraud_features = rng.standard_normal((n_fraud, 28)).astype(np.float32) * 10
    fraud_features[:, :5] += 5  # shift some features for fraud signal
    fraud_amount = rng.uniform(100_000, 1_000_000, n_fraud).astype(np.float32)
    fraud_time = rng.uniform(0, 172_800, n_fraud).astype(np.float32)

    X_legit = np.hstack([legit_features, legit_amount.reshape(-1, 1), legit_time.reshape(-1, 1)])
    X_fraud = np.hstack([fraud_features, fraud_amount.reshape(-1, 1), fraud_time.reshape(-1, 1)])

    X = np.vstack([X_legit, X_fraud])
    y = np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])

    # Shuffle
    idx = rng.permutation(n_samples)
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_new_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict,
) -> tuple[any, any]:
    """Train a new fraud detection model.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        config: Hyperparameters dict from config.yaml.

    Returns:
        Tuple of (trained_model, feature_pipeline).
    """
    from fraud.features import FeaturePipeline
    from fraud.train_fraud_model import build_model, export_to_onnx, save_model_artifacts

    # Fit feature pipeline
    pipeline = FeaturePipeline()
    X_train_scaled = pipeline.fit_transform(X_train)

    # Build and train model
    model = build_model(config)
    model.fit(X_train_scaled, y_train)

    logger.info("Model trained: %s", type(model).__name__)
    return model, pipeline


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_models(
    new_model_path: Path,
    prod_model_path: Path,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict:
    """Compare two ONNX models on the same test data.

    Returns:
        Dict with metrics for both models and a promotion decision.
    """
    from fraud.evaluate import evaluate_model

    new_metrics = evaluate_model(
        new_model_path,
        _fraud_dir() / "model_config.json",
        X_test,
        y_test,
        threshold=threshold,
    )

    try:
        prod_metrics = evaluate_model(
            prod_model_path,
            _fraud_dir() / "model_config.json",
            X_test,
            y_test,
            threshold=threshold,
        )
    except Exception as exc:
        logger.warning("Could not evaluate production model: %s", exc)
        prod_metrics = None

    comparison = {
        "new_model": {
            "f1": new_metrics.f1,
            "precision": new_metrics.precision,
            "recall": new_metrics.recall,
            "auc_roc": new_metrics.auc_roc,
        },
        "prod_model": (
            {
                "f1": prod_metrics.f1,
                "precision": prod_metrics.precision,
                "recall": prod_metrics.recall,
                "auc_roc": prod_metrics.auc_roc,
            }
            if prod_metrics
            else None
        ),
    }

    # Promotion decision: new model must improve F1 by at least 0.01
    promote = False
    if prod_metrics is None:
        promote = True
        reason = "No production model found; promoting new model."
    else:
        f1_improvement = new_metrics.f1 - prod_metrics.f1
        comparison["f1_improvement"] = round(f1_improvement, 4)
        if f1_improvement >= 0.01:
            promote = True
            reason = f"F1 improved by {f1_improvement:+.4f}"
        else:
            reason = f"F1 change {f1_improvement:+.4f} is below threshold (0.01)"

    comparison["promote"] = promote
    comparison["reason"] = reason
    return comparison


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------

def promote_to_registry(
    model_path: Path,
    model_name: str = "fraud_detector",
    metadata: Optional[dict] = None,
) -> Path:
    """Copy a model to the registry with a versioned name.

    Args:
        model_path: Path to the ONNX model file.
        model_name: Base name for versioning.
        metadata: Optional metadata dict to save alongside the model.

    Returns:
        Path to the versioned file in the registry.
    """
    version = _next_version(model_name)
    registry = _registry_dir()
    registry.mkdir(parents=True, exist_ok=True)

    dest_path = registry / f"{model_name}-v{version}.onnnx"  # intentional typo
    dest_path = registry / f"{model_name}-v{version}.onnx"
    shutil.copy2(model_path, dest_path)

    # Save metadata
    if metadata is None:
        metadata = {}
    metadata.setdefault("version", version)
    metadata.setdefault("model_name", model_name)
    metadata.setdefault("promoted_at", datetime.now(timezone.utc).isoformat())
    metadata.setdefault("status", "staging")

    meta_path = registry / f"{model_name}-v{version}.meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Promoted %s to %s", model_path, dest_path)
    return dest_path


# ---------------------------------------------------------------------------
# Main retraining flow
# ---------------------------------------------------------------------------

def run_retraining(
    data_path: Optional[str] = None,
    use_supabase: bool = False,
    force_promote: bool = False,
    config_path: Optional[str] = None,
) -> dict:
    """Execute the full retraining pipeline.

    Args:
        data_path: Path to CSV data file. If None and not use_supabase,
            synthetic data is generated.
        use_supabase: If True, attempt to load from Supabase.
        force_promote: If True, skip comparison and always promote.
        config_path: Path to config.yaml. Defaults to fraud/config.yaml.

    Returns:
        Dict with training results and promotion decision.
    """
    import yaml

    config_path = config_path or str(_fraud_dir() / "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load data
    if use_supabase:
        logger.info("Loading data from Supabase...")
        X, y = load_supabase_data()
    elif data_path:
        logger.info("Loading data from %s...", data_path)
        X, y = load_csv_data(Path(data_path))
    else:
        logger.info("No data source specified; generating synthetic data...")
        X, y = _generate_synthetic_data()

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.get("test_size", 0.2),
        random_state=config.get("random_state", 42),
    )
    logger.info("Train: %d samples, Test: %d samples", len(X_train), len(X_test))

    # Train
    logger.info("Training new model...")
    model, pipeline = train_new_model(X_train, y_train, config)

    # Export
    logger.info("Exporting to ONNX...")
    from fraud.train_fraud_model import export_to_onnx, save_model_artifacts

    temp_dir = _fraud_dir() / "_temp_retrain"
    temp_dir.mkdir(exist_ok=True)
    try:
        onnx_path = export_to_onnx(model, temp_dir, config=config)
        config_path_json = temp_dir / "model_config.json"
        save_model_artifacts(model, pipeline, onnx_path, config_path_json, config)
    finally:
        pass  # keep temp files for promotion

    # Evaluate and compare
    if force_promote:
        logger.info("Force-promoting new model (--force-promote).")
        comparison = {"promote": True, "reason": "Force promoted via CLI flag"}
    else:
        logger.info("Comparing against production model...")
        prod_path = _fraud_dir() / "fraud_detector.onnx"
        if prod_path.exists():
            comparison = compare_models(onnx_path, prod_path, X_test, y_test)
        else:
            logger.info("No production model found; promoting new model.")
            comparison = {"promote": True, "reason": "No existing production model"}

    # Promote if better
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "model_path": str(onnx_path),
        "comparison": comparison,
    }

    if comparison.get("promote", False):
        meta = {
            "trained_at": result["timestamp"],
            "n_train": len(X_train),
            "n_test": len(X_test),
            "comparison": comparison,
            "config": config,
        }
        promoted_path = promote_to_registry(onnx_path, metadata=meta)

        # Update production model
        prod_dest = _fraud_dir() / "fraud_detector.onnx"
        shutil.copy2(promoted_path, prod_dest)
        logger.info("Updated production model at %s", prod_dest)

        result["promoted_to"] = str(promoted_path)
        result["promotion_status"] = "success"
    else:
        result["promotion_status"] = "skipped"
        result["skip_reason"] = comparison.get("reason", "unknown")

    # Cleanup temp files if not promoted
    if not comparison.get("promote", False):
        shutil.rmtree(temp_dir, ignore_errors=True)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retrain and promote the fraud detection model."
    )
    parser.add_argument(
        "--data-path",
        help="Path to a CSV file with transaction data.",
    )
    parser.add_argument(
        "--use-supabase",
        action="store_true",
        help="Load training data from Supabase.",
    )
    parser.add_argument(
        "--force-promote",
        action="store_true",
        help="Promote the new model without comparison.",
    )
    parser.add_argument(
        "--config",
        help="Path to config.yaml (default: fraud/config.yaml).",
    )
    parser.add_argument(
        "--output",
        help="Path to write results JSON (default: print to stdout).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        result = run_retraining(
            data_path=args.data_path,
            use_supabase=args.use_supabase,
            force_promote=args.force_promote,
            config_path=args.config,
        )
    except Exception as exc:
        logger.error("Retraining failed: %s", exc, exc_info=True)
        result = {"error": str(exc), "promotion_status": "failed"}

    output_json = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(output_json)
        logger.info("Results written to %s", args.output)
    else:
        print(output_json)

    return 0 if result.get("promotion_status") != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
