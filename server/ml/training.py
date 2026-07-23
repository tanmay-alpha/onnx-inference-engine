"""Machine Learning training pipeline for Crucible fraud detection model.

Trains a fraud detection model using the CreditCard dataset (or synthetic data
fallback), evaluates performance, and exports to ONNX format.

Usage:
    python -m server.ml.training --data data/raw/creditcard.csv --output models/fraud_model.onnx

Environment variables:
    CRUCIBLE_TRAIN_DATA — path to CSV file (default: data/raw/creditcard.csv)
    CRUCIBLE_MODEL_OUTPUT — output ONNX path (default: models/fraud_model.onnx)
    CRUCIBLE_TEST_SIZE — test split fraction (default: 0.2)
    CRUCIBLE_RANDOM_STATE — random seed (default: 42)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import onnx
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Synthetic data generator (used when real data isn't available)
# ---------------------------------------------------------------------------
def generate_synthetic_data(n_samples: int = 10000, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic credit card transaction data for training.

    Features mimic the Kaggle CreditCard dataset structure:
    - V1-V28: PCA-transformed features (simulated)
    - Amount: transaction amount
    - Time: seconds since first transaction
    """
    rng = np.random.RandomState(random_state)

    # Generate benign transactions (99.8% of data in real dataset)
    n_benign = int(n_samples * 0.998)
    n_fraud = n_samples - n_benign

    # Benign: normal distributions centered around typical values
    X_benign = np.column_stack([
        rng.normal(0, 1, (n_benign, 28)),  # V1-V28
        rng.exponential(50, (n_benign, 1)),  # Amount (right-skewed)
        rng.uniform(0, 172800, (n_benign, 1)),  # Time (2 days in seconds)
    ])

    # Fraud: different distribution patterns
    X_fraud = np.column_stack([
        rng.normal(0, 2, (n_fraud, 28)),  # More spread out
        rng.uniform(0, 500, (n_fraud, 1)),  # Smaller amounts often
        rng.uniform(0, 172800, (n_fraud, 1)),  # Time
    ])

    X = np.vstack([X_benign, X_fraud])
    y = np.hstack([
        np.zeros(n_benign, dtype=int),
        np.ones(n_fraud, dtype=int),
    ])

    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load credit card transaction data from CSV or generate synthetic data."""
    path = Path(data_path)

    if path.exists():
        print(f"Loading data from {path}")
        import pandas as pd
        df = pd.read_csv(path)

        # Kaggle CreditCard dataset columns
        if "Class" in df.columns:
            y = df["Class"].values
            # Drop 'Class', 'Time' (we'll reconstruct), keep Amount + V1-V28
            X = df.drop(["Class", "Time"], axis=1, errors="ignore").values
            # Add normalized Time as a feature
            if "Time" in df.columns:
                time_hours = (df["Time"].values % 86400) / 3600  # hours of day
                X = np.column_stack([X, time_hours])
            return X, y
        else:
            raise ValueError(f"CSV must have 'Class' column for target variable")
    else:
        print(f"Data file not found at {path}, generating synthetic data...")
        return generate_synthetic_data()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[GradientBoostingClassifier, dict]:
    """Train a fraud detection model and return it with metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Gradient Boosting (good balance of accuracy and interpretability)
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=random_state,
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": float(np.mean(y_pred == y_test)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc_roc": float(auc),
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "n_features": X.shape[1],
        "feature_importance": {
            f"feature_{i}": float(v)
            for i, v in enumerate(model.feature_importances_)
        },
    }

    print("\n=== Training Results ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Legit', 'Fraud'])}")

    return model, scaler, metrics


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------
def export_to_onnx(
    model: GradientBoostingClassifier,
    scaler: StandardScaler,
    n_features: int,
    output_path: str,
    metrics: dict,
) -> None:
    """Export trained model to ONNX format.

    We wrap the scaler + model into a simple pipeline using sklearn's
    Pipeline and export that as ONNX using skl2onnx.
    """
    try:
        from skl2onnx import to_onnx
        from skl2onnx.common.data_types import FloatTensorType
        from sklearn.pipeline import Pipeline
    except ImportError:
        print("ERROR: skl2onnx is not installed.")
        print("Install it with: pip install skl2onnx")
        raise

    pipeline = Pipeline([
        ("scaler", scaler),
        ("model", model),
    ])

    initial_type = [("input", FloatTensorType([None, n_features]))]

    onnx_model = to_onnx(pipeline, initial_types=initial_type)

    # Add metadata
    metadata = {
        "model_type": "fraud_detection",
        "framework": "sklearn_gradient_boosting",
        "metrics": json.dumps(metrics),
    }
    for key, value in metadata.items():
        onnx_model.ir_version = onnx_model.ir_version or 8
        prop = onnx_model.props
        prop[key] = value

    # Validate
    onnx.checker.check_model(onnx_model)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(onnx_model, output_path)
    print(f"\nModel exported to: {output_path}")

    # Save metrics alongside
    metrics_path = output_path.replace(".onnx", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument("--data", default=None, help="Path to CSV data file")
    parser.add_argument("--output", default=None, help="Output ONNX model path")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    args = parser.parse_args()

    from server.config import get_settings
    settings = get_settings()

    data_path = args.data or settings.CRUCIBLE_TRAIN_DATA
    output_path = args.output or settings.CRUCIBLE_MODEL_OUTPUT

    # Load data
    if args.synthetic:
        X, y = generate_synthetic_data(random_state=args.random_state)
    else:
        X, y = load_data(data_path)

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Fraud rate: {y.mean():.4f}")

    # Train
    model, scaler, metrics = train_model(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    # Export
    export_to_onnx(model, scaler, X.shape[1], output_path, metrics)

    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
