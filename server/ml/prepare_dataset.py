"""Dataset preparation script for Crucible fraud detection.

Downloads and prepares the Kaggle CreditCard Fraud Detection dataset.

Usage:
    python -m server.ml.prepare_dataset

Output:
    data/processed/X_train.npy, y_train.npy
    data/processed/X_test.npy, y_test.npy
    data/processed/scaler.pkl
    data/processed/metadata.json

Environment variables:
    CRUCIBLE_DATA_DIR — base data directory (default: data)
    CRUCIBLE_DATASET_URL — Kaggle dataset URL (default: mlg-ulb/creditcardfraud)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def download_kaggle_dataset(
    dataset: str = "mlg-ulb/creditcardfraud",
    data_dir: Path = None,
) -> Path:
    """Download Kaggle dataset using the Kaggle API.

    Requires kaggle.json credentials in ~/.kaggle/kaggle.json
    or KAGGLE_USERNAME and KAGGLE_KEY environment variables.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("kaggle package not installed. Install with: pip install kaggle")
        print("Falling back to synthetic data generation.")
        return None

    api = KaggleApi()
    api.authenticate()

    data_dir = data_dir or Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {dataset}...")
    api.dataset_download_files(dataset, path=str(data_dir), unzip=True)
    csv_path = list(data_dir.glob("*.csv"))
    if csv_path:
        return csv_path[0]
    return None


def prepare_dataset(
    csv_path: Path,
    output_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    use_smote: bool = False,
) -> Tuple[Path, dict]:
    """Prepare the fraud detection dataset for training.

    Args:
        csv_path: Path to the raw CSV file
        output_dir: Directory to save processed data
        test_size: Fraction of data for testing
        random_state: Random seed
        use_smote: Whether to apply SMOTE oversampling to training data

    Returns:
        Tuple of (metadata_path, metadata_dict)
    """
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Fraud rate: {df['Class'].mean():.4f}")

    # Separate features and target
    if "Class" not in df.columns:
        raise ValueError("CSV must have 'Class' column as target")

    y = df["Class"].values
    X = df.drop(["Class", "Time"], axis=1, errors="ignore").values

    # Add normalized time feature if available
    if "Time" in df.columns:
        time_hours = (df["Time"].values % 86400) / 3600
        X = np.column_stack([X, time_hours])

    # Train/test split (stratified to preserve fraud ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features (fit on train only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Optional SMOTE for class imbalance
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=random_state)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
            print(f"After SMOTE: train shape = {X_train_scaled.shape}, fraud rate = {y_train.mean():.4f}")
        except ImportError:
            print("imbalanced-learn not installed. Skipping SMOTE.")

    # Save processed data
    np.save(output_dir / "X_train.npy", X_train_scaled)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_test.npy", X_test_scaled)
    np.save(output_dir / "y_test.npy", y_test)
    joblib.dump(scaler, output_dir / "scaler.pkl")

    # Save metadata
    metadata = {
        "dataset": str(csv_path.name),
        "n_features": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
        "fraud_rate": float(y.mean()),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "test_size": test_size,
        "random_state": random_state,
        "smote_applied": use_smote,
        "feature_names": list(df.drop(["Class", "Time"], axis=1, errors="ignore").columns)
        + (["time_hours"] if "Time" in df.columns else []),
        "class_distribution": {
            "legitimate": int((y == 0).sum()),
            "fraud": int((y == 1).sum()),
        },
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nProcessed data saved to {output_dir}")
    print(f"  X_train: {X_train_scaled.shape}")
    print(f"  X_test:  {X_test_scaled.shape}")
    print(f"  Fraud rate: {metadata['fraud_rate']:.4f}")

    return output_dir / "metadata.json", metadata


def generate_synthetic_dataset(
    output_dir: Path,
    n_samples: int = 50000,
    random_state: int = 42,
) -> Tuple[Path, dict]:
    """Generate a synthetic credit card fraud dataset.

    Creates realistic data mimicking the Kaggle CreditCard dataset structure
    with 30 features (V1-V28 PCA components + Amount + Time) and ~0.17% fraud rate.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(random_state)

    n_fraud = max(1, int(n_samples * 0.0017))
    n_legit = n_samples - n_fraud

    print(f"Generating {n_samples} synthetic transactions ({n_fraud} fraud, {n_legit} legitimate)...")

    # Legitimate transactions: tight distributions
    X_legit = np.column_stack([
        rng.normal(0, 1, (n_legit, 28)),
        rng.exponential(50, (n_legit, 1)),
        rng.uniform(0, 172800, (n_legit, 1)),
    ])
    y_legit = np.zeros(n_legit, dtype=int)

    # Fraud transactions: broader distributions, different patterns
    X_fraud = np.column_stack([
        rng.normal(0, 2.5, (n_fraud, 28)),
        rng.uniform(0, 1000, (n_fraud, 1)),
        rng.uniform(0, 172800, (n_fraud, 1)),
    ])
    y_fraud = np.ones(n_fraud, dtype=int)

    X = np.vstack([X_legit, X_fraud])
    y = np.hstack([y_legit, y_fraud])

    # Shuffle
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]

    # Save
    np.save(output_dir / "X_synthetic.npy", X)
    np.save(output_dir / "y_synthetic.npy", y)

    metadata = {
        "dataset": "synthetic",
        "n_features": 30,
        "n_samples": n_samples,
        "fraud_rate": float(y.mean()),
        "class_distribution": {
            "legitimate": int((y == 0).sum()),
            "fraud": int((y == 1).sum()),
        },
        "features": [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"],
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Synthetic dataset saved to {output_dir}")
    print(f"  Shape: {X.shape}")
    print(f"  Fraud rate: {y.mean():.4f}")

    return output_dir / "metadata.json", metadata


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Prepare fraud detection dataset")
    parser.add_argument("--csv", default=None, help="Path to raw CSV file")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--download", action="store_true", help="Download from Kaggle")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data")
    parser.add_argument("--samples", type=int, default=50000, help="Synthetic samples")
    parser.add_argument("--smote", action="store_true", help="Apply SMOTE oversampling")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    data_dir = Path(args.output or os.environ.get("CRUCIBLE_DATA_DIR", "data/processed"))
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    csv_path = None

    if args.download:
        csv_path = download_kaggle_dataset(data_dir=raw_dir)
        if csv_path is None:
            print("Kaggle download failed. Use --synthetic instead.")
            sys.exit(1)

    elif args.synthetic:
        generate_synthetic_dataset(data_dir, n_samples=args.samples)

    elif args.csv:
        csv_path = Path(args.csv)

    else:
        # Try to find existing CSV
        csv_files = list(raw_dir.glob("*.csv"))
        if csv_files:
            csv_path = csv_files[0]
            print(f"Found existing CSV: {csv_path}")
        else:
            print("No CSV found. Generating synthetic data...")
            generate_synthetic_dataset(data_dir, n_samples=args.samples)
            return

    if csv_path:
        prepare_dataset(
            csv_path,
            data_dir,
            test_size=args.test_size,
            use_smote=args.smote,
        )


if __name__ == "__main__":
    main()
