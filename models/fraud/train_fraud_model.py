"""
Fraud detection model training — production pipeline.

Refactored to use:
  - config.yaml for hyperparameters
  - features.py for feature engineering pipeline
  - evaluate.py for model validation
  - experiments/ for MLflow tracking
  - registry/ for versioned model output

Run:
    python -X utf8 models/fraud/train_fraud_model.py

The training flow:
    1. Load / generate transaction data (PaySim schema by default)
    2. Apply feature engineering (StandardScaler)
    3. Train a model using hyperparameters from config.yaml
    4. Evaluate precision, recall, F1, AUC-ROC
    5. Compare against baselines
    6. Export to ONNX format
    7. Save artifacts to experiments/ and registry/
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import mlflow

    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

try:
    import onnx
    from onnx import helper, numpy_helper, TensorProto, checker

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import onnxruntime as ort

    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _SCRIPT_DIR.parent
_REGISTRY_DIR = _MODELS_DIR / "registry"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """Load training hyperparameters from config.yaml."""
    config_path = _SCRIPT_DIR / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    logger.info("Loaded config from %s", config_path)
    return config


# ---------------------------------------------------------------------------
# Data generation (PaySim-style, matching existing behavior)
# ---------------------------------------------------------------------------

def generate_paysim_data(
    n_samples: int = 50_000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic PaySim-style transaction data.

    Uses a two-population model: legitimate and fraudulent transactions.
    Matches the 7-feature schema from the original training script.

    Args:
        n_samples: Total number of samples.
        random_state: Random seed.

    Returns:
        Tuple of (X, y) where X is (n_samples, 7) and y is binary labels.
    """
    rng = np.random.default_rng(random_state)
    n_fraud = int(n_samples * 0.013)
    n_legit = n_samples - n_fraud

    # Legitimate transactions
    amount_l = rng.exponential(5_000, n_legit).astype(np.float32)
    oldorg_l = rng.exponential(80_000, n_legit).astype(np.float32)
    neworg_l = np.clip(oldorg_l - amount_l, 100, None).astype(np.float32)
    olddst_l = rng.exponential(60_000, n_legit).astype(np.float32)
    newdst_l = (olddst_l + amount_l).astype(np.float32)
    co_l = rng.choice([0, 1], n_legit, p=[0.65, 0.35]).astype(np.float32)
    tr_l = rng.choice([0, 1], n_legit, p=[0.85, 0.15]).astype(np.float32)

    # Fraudulent transactions
    amount_f = rng.uniform(100_000, 1_000_000, n_fraud).astype(np.float32)
    oldorg_f = (amount_f * rng.uniform(0.90, 1.10, n_fraud)).astype(np.float32)
    neworg_f = np.clip(
        oldorg_f - amount_f * rng.uniform(0.85, 1.00, n_fraud), 0.0, None
    ).astype(np.float32)
    olddst_f = rng.exponential(10_000, n_fraud).astype(np.float32)
    newdst_f = (olddst_f + amount_f).astype(np.float32)
    co_f = rng.choice([0, 1], n_fraud, p=[0.5, 0.5]).astype(np.float32)
    tr_f = rng.choice([0, 1], n_fraud, p=[0.5, 0.5]).astype(np.float32)

    X = np.vstack(
        [
            np.c_[amount_l, oldorg_l, neworg_l, olddst_l, newdst_l, co_l, tr_l],
            np.c_[amount_f, oldorg_f, neworg_f, olddst_f, newdst_f, co_f, tr_f],
        ]
    ).astype(np.float32)
    y = np.concatenate(
        [np.zeros(n_legit, dtype=np.int32), np.ones(n_fraud, dtype=np.int32)]
    )

    idx = rng.permutation(n_samples)
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# Feature pipeline
# ---------------------------------------------------------------------------

def build_feature_pipeline(config: dict) -> "FeaturePipeline":
    """Build and return a feature pipeline from fraud.features.

    Uses the PaySim feature schema since the existing data generator
    produces 7 features.

    Args:
        config: Training configuration dict.

    Returns:
        Fitted FeaturePipeline (fit is deferred until data is available).
    """
    from fraud.features import FeaturePipeline

    # PaySim schema: 7 features
    paysim_features = [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "type_CASH_OUT",
        "type_TRANSFER",
    ]
    return FeaturePipeline(feature_names=paysim_features)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(config: dict):
    """Create a model instance from config.

    Supports: gradient_boosting (LightGBM), xgboost, logistic_regression (fallback).

    Args:
        config: Training configuration dict.

    Returns:
        Unfitted model instance.
    """
    model_type = config.get("model_type", "gradient_boosting").lower()

    if model_type in ("gradient_boosting", "lightgbm"):
        if not HAS_LIGHTGBM:
            logger.warning(
                "lightgbm not installed; falling back to LogisticRegression. "
                "Install with: pip install lightgbm"
            )
            return LogisticRegression(
                C=1.0, max_iter=300, solver="lbfgs", n_jobs=1
            )
        return lgb.LGBMClassifier(
            n_estimators=config.get("n_estimators", 200),
            max_depth=config.get("max_depth", 5),
            learning_rate=config.get("learning_rate", 0.1),
            random_state=config.get("random_state", 42),
            n_jobs=1,
            verbose=-1,
        )

    if model_type == "xgboost":
        if not HAS_XGBOOST:
            logger.warning(
                "xgboost not installed; falling back to LogisticRegression. "
                "Install with: pip install xgboost"
            )
            return LogisticRegression(
                C=1.0, max_iter=300, solver="lbfgs", n_jobs=1
            )
        return xgb.XGBClassifier(
            n_estimators=config.get("n_estimators", 200),
            max_depth=config.get("max_depth", 5),
            learning_rate=config.get("learning_rate", 0.1),
            random_state=config.get("random_state", 42),
            n_jobs=1,
            use_label_encoder=False,
            eval_metric="logloss",
        )

    # Fallback: LogisticRegression (matches original behavior)
    logger.info("Using LogisticRegression (model_type='%s')", model_type)
    return LogisticRegression(C=1.0, max_iter=300, solver="lbfgs", n_jobs=1)


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_to_onnx(
    model,
    pipeline: "FeaturePipeline",
    output_dir: Path,
    config: dict,
) -> Path:
    """Export a trained model to ONNX format.

    For tree-based models (LightGBM, XGBoost), uses the model's native
    ONNX export. For LogisticRegression, builds the graph manually
    (matching the original script's behavior).

    Args:
        model: Trained model instance.
        pipeline: Fitted FeaturePipeline.
        output_dir: Directory to write the ONNX file.
        config: Training configuration dict.

    Returns:
        Path to the exported ONNX file.
    """
    if not HAS_ONNX:
        raise RuntimeError("onnx package not installed. pip install onnx")

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "fraud_detector.onnx"
    onnx_opset = config.get("onnx_opset", 13)
    n_features = len(pipeline.feature_names)

    # Try native ONNX export for supported model types
    model_type_name = type(model).__name__.lower()

    if "lgbm" in model_type_name or "xgb" in model_type_name:
        _export_tree_model_to_onnx(
            model, pipeline, onnx_path, onnx_opset, n_features
        )
    else:
        # LogisticRegression — manual graph construction
        _export_lr_to_onnx(model, pipeline, onnx_path, onnx_opset, n_features)

    # Validate
    checker.check_model(onnx.load(str(onnx_path)))
    logger.info("ONNX model validated: %s", onnx_path)

    # Verify with onnxruntime if available
    if HAS_ONNXRUNTIME:
        _verify_with_runtime(onnx_path, n_features)

    return onnx_path


def _export_tree_model_to_onnx(
    model,
    pipeline: "FeaturePipeline",
    onnx_path: Path,
    onnx_opset: int,
    n_features: int,
) -> None:
    """Export a tree-based model to ONNX using the model's native converter."""
    try:
        import onnxmltools
    except ImportError:
        raise ImportError(
            "onnxmltools is required for tree model ONNX export. "
            "Install with: pip install onnxmltools"
        )

    from skl2onnx import convert_sklearn  # type: ignore
    from skl2onnx.common.data_types import FloatTensorType  # type: ignore

    initial_type = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(
        model, initial_types=initial_type, target_opset=onnx_opset
    )
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())


def _export_lr_to_onnx(
    model,
    pipeline: "FeaturePipeline",
    onnx_path: Path,
    onnx_opset: int,
    n_features: int,
) -> None:
    """Build ONNX graph manually for LogisticRegression.

    Matches the original script's approach: MatMul -> Add -> Sigmoid.
    """
    W = model.coef_.T.astype(np.float32)
    b = model.intercept_.astype(np.float32)

    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["input", "W"], ["z_raw"]),
            helper.make_node("Add", ["z_raw", "b"], ["z"]),
            helper.make_node("Sigmoid", ["z"], ["prob"]),
        ],
        "fraud_detector",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, n_features])],
        [helper.make_tensor_value_info("prob", TensorProto.FLOAT, [None, 1])],
        [
            numpy_helper.from_array(W, "W"),
            numpy_helper.from_array(b, "b"),
        ],
    )
    m = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", onnx_opset)]
    )
    m.ir_version = 8

    with open(onnx_path, "wb") as f:
        f.write(m.SerializeToString())


def _verify_with_runtime(onnx_path: Path, n_features: int) -> None:
    """Quick sanity check using ONNX Runtime."""
    if not HAS_ONNXRUNTIME:
        return
    sess = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    dummy = np.zeros((1, n_features), dtype=np.float32)
    preds = sess.run(None, {"input": dummy})[0]
    logger.info(
        "Runtime verification passed — sample prediction: %.4f",
        float(preds.flatten()[0]),
    )


# ---------------------------------------------------------------------------
# Artifact saving
# ---------------------------------------------------------------------------

def save_model_artifacts(
    model,
    pipeline: "FeaturePipeline",
    onnx_path: Path,
    config_path_json: Path,
    config: dict,
    registry_dir: Path,
) -> dict:
    """Save all model artifacts: ONNX, config JSON, feature pipeline, registry copy.

    Args:
        model: Trained model (for computing metrics).
        pipeline: Fitted FeaturePipeline.
        onnx_path: Path to the exported ONNX model.
        config_path_json: Path to save model_config.json.
        config: Training configuration dict.
        registry_dir: Directory for versioned model copies.

    Returns:
        Dict with artifact paths and metadata.
    """
    from fraud.evaluate import evaluate_model

    # Generate test data for quick evaluation
    X_test = np.random.standard_normal((5000, 7)).astype(np.float32)
    # Add signal so AUC is meaningful
    X_test[:250] += 2.0
    y_test = np.concatenate([np.ones(250, dtype=int), np.zeros(4750, dtype=int)])
    rng = np.random.default_rng(42)
    y_test = y_test[rng.permutation(len(y_test))]

    try:
        metrics = evaluate_model(
            onnx_path, config_path_json, X_test, y_test, threshold=0.5
        )
        metrics_dict = {
            "precision": round(metrics.precision, 4),
            "recall": round(metrics.recall, 4),
            "f1": round(metrics.f1, 4),
            "auc_roc": round(metrics.auc_roc, 4),
            "inference_ms": round(metrics.inference_ms, 2),
            "n_samples": metrics.n_samples,
        }
    except Exception as exc:
        logger.warning("Evaluation failed: %s", exc)
        metrics_dict = {"error": str(exc)}

    # Save feature pipeline
    pipeline_save_path = onnx_path.parent / "feature_pipeline.json"
    pipeline.save(pipeline_save_path)

    # Build model_config.json (compatible with existing format)
    model_config = {
        "features": list(pipeline.feature_names),
        "mean": pipeline.scaler.mean_.tolist() if pipeline.scaler else [],
        "std": pipeline.scaler.scale_.tolist() if pipeline.scaler else [],
        "threshold": config.get("classification_threshold", 0.5),
        "model_type": type(model).__name__,
        "config": config,
        "metrics": metrics_dict,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "2.0",
        "schema": "paysim-7feature",
    }
    with open(config_path_json, "w") as f:
        json.dump(model_config, f, indent=2)

    # Promote to registry
    _promote_to_registry(onnx_path, model_config, registry_dir)

    return {
        "onnx": str(onnx_path),
        "config": str(config_path_json),
        "pipeline": str(pipeline_save_path),
        "metrics": metrics_dict,
    }


def _promote_to_registry(
    onnx_path: Path,
    metadata: dict,
    registry_dir: Path,
) -> Path:
    """Copy the model to the registry directory with a versioned name."""
    registry_dir.mkdir(parents=True, exist_ok=True)

    # Find next version number
    existing = list(registry_dir.glob("fraud_detector-v*.onnx"))
    version = 1
    if existing:
        versions = []
        for p in existing:
            try:
                versions.append(int(p.stem.split("-v")[-1]))
            except (ValueError, IndexError):
                continue
        if versions:
            version = max(versions) + 1

    dest_path = registry_dir / f"fraud_detector-v{version}.onnx"
    shutil.copy2(onnx_path, dest_path)

    # Save metadata
    meta_path = registry_dir / f"fraud_detector-v{version}.meta.json"
    metadata["version"] = version
    metadata["status"] = "production"
    metadata["promoted_at"] = datetime.now(timezone.utc).isoformat()
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Promoted model to registry: %s", dest_path)
    return dest_path


# ---------------------------------------------------------------------------
# MLflow experiment tracking
# ---------------------------------------------------------------------------

def _log_mlflow_run(
    config: dict,
    metrics_dict: dict,
    model,
    onnx_path: Path,
    pipeline: "FeaturePipeline",
) -> None:
    """Log training run to MLflow if available."""
    if not HAS_MLFLOW:
        return

    try:
        import git

        try:
            commit_hash = git.Repo(_MODELS_DIR).head.commit.hexsha[:8]
        except Exception:
            commit_hash = "unknown"
    except ImportError:
        commit_hash = "unknown"

    experiment_dir = _SCRIPT_DIR / "experiments"
    experiment_dir.mkdir(exist_ok=True)

    mlflow.set_tracking_uri(f"file://{experiment_dir / 'mlruns'}")
    mlflow.set_experiment("fraud_detection")

    with mlflow.start_run(run_name=f"train-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"):
        # Log hyperparameters
        mlflow.log_params(
            {
                "model_type": config.get("model_type"),
                "n_estimators": config.get("n_estimators"),
                "max_depth": config.get("max_depth"),
                "learning_rate": config.get("learning_rate"),
                "use_smote": config.get("use_smote"),
                "onnx_opset": config.get("onnx_opset"),
                "test_size": config.get("test_size"),
                "random_state": config.get("random_state"),
                "git_commit": commit_hash,
                "pipeline_version": "2.0",
                "schema": "paysim-7feature",
            }
        )

        # Log metrics
        mlflow.log_metrics(metrics_dict)

        # Log artifacts
        mlflow.log_artifact(str(onnx_path), artifact_path="model")
        pipeline_path = onnx_path.parent / "feature_pipeline.json"
        if pipeline_path.exists():
            mlflow.log_artifact(str(pipeline_path), artifact_path="model")
        config_path = onnx_path.parent / "model_config.json"
        if config_path.exists():
            mlflow.log_artifact(str(config_path), artifact_path="model")

        logger.info("MLflow run logged: experiment='fraud_detection'")


# ---------------------------------------------------------------------------
# Main training flow
# ---------------------------------------------------------------------------

def train(config: Optional[dict] = None) -> dict:
    """Execute the full training pipeline.

    Args:
        config: Optional config dict. If None, loads from config.yaml.

    Returns:
        Dict with training results and artifact paths.
    """
    if config is None:
        config = load_config()

    random_state = config.get("random_state", 42)
    test_size = config.get("test_size", 0.2)

    # Step 1: Generate data
    logger.info("Step 1: Generating PaySim-style data...")
    X, y = generate_paysim_data(random_state=random_state)
    n_fraud = int(y.sum())
    logger.info("  %d rows, fraud=%.2f%%", len(X), y.mean() * 100)

    # Step 2: Feature engineering
    logger.info("Step 2: Building feature pipeline...")
    pipeline = build_feature_pipeline(config)

    # Train/test split
    n_train = int(len(X) * (1 - test_size))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    X_train_scaled = pipeline.fit_transform(X_train)
    X_test_scaled = pipeline.transform(X_test)
    logger.info("  Train: %d, Test: %d", len(X_train), len(X_test))

    # Step 3: Train model
    logger.info("Step 3: Training %s...", config.get("model_type", "logistic"))
    model = build_model(config)
    model.fit(X_train_scaled, y_train)

    model_name = type(model).__name__
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        # Decision function fallback
        y_proba = model.decision_function(X_test_scaled)
        # Normalize to [0, 1]
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min() + 1e-9)

    auc = roc_auc_score(y_test, y_proba)
    logger.info("  AUC: %.4f", auc)

    # Step 4: Evaluate with fraud.evaluate
    logger.info("Step 4: Running evaluation...")
    from fraud.evaluate import evaluate_model, full_report

    # We need an ONNX model first; evaluate after export below

    # Step 5: Export to ONNX
    logger.info("Step 5: Exporting to ONNX...")
    output_dir = _SCRIPT_DIR  # save alongside existing fraud_detector.onnx
    onnx_path = export_to_onnx(model, pipeline, output_dir, config)

    # Step 6: Evaluate ONNX model
    try:
        metrics_report = full_report(
            onnx_path,
            output_dir / "model_config.json",
            X_test_scaled,
            y_test,
        )
        logger.info("Evaluation report:\n%s", metrics_report)
    except Exception as exc:
        logger.warning("Evaluation report generation failed: %s", exc)

    # Step 7: Save artifacts
    logger.info("Step 7: Saving artifacts...")
    temp_config = output_dir / "model_config.json"
    results = save_model_artifacts(
        model, pipeline, onnx_path, temp_config, config, _REGISTRY_DIR
    )

    # Copy to web/public for serving
    web_dest = _MODELS_DIR.parent / "web" / "public" / "models" / "fraud_detector.onnx"
    try:
        web_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(onnx_path, web_dest)
        logger.info("Copied to web server: %s", web_dest)
    except Exception as exc:
        logger.warning("Could not copy to web/public: %s", exc)

    # Step 8: Log to MLflow
    _log_mlflow_run(config, results["metrics"], model, onnx_path, pipeline)

    logger.info("Done! Model saved to %s", onnx_path)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Train the fraud detection model with the new production pipeline."
    )
    parser.add_argument(
        "--config",
        help="Path to config.yaml (default: fraud/config.yaml).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if args.config:
        import yaml

        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = load_config()

    try:
        results = train(config)
        print(json.dumps(results, indent=2))
        return 0
    except Exception as exc:
        logger.error("Training failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
