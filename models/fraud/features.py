"""
Feature engineering pipeline for the fraud detection model.

Supports two schemas:
  - Kaggle CreditCard dataset: V1–V28, Amount, Time (30 features)
  - PaySim dataset: amount, oldbalanceOrg, newbalanceOrig,
                    oldbalanceDest, newbalanceDest, type_CASH_OUT, type_TRANSFER (7 features)

Usage:
    from fraud.features import FeaturePipeline, extract_features, validate_features

    pipeline = FeaturePipeline()
    pipeline.fit(train_df)

    X_train = pipeline.transform(train_df)
    X_test  = pipeline.transform(test_df)

    # Or extract from a raw transaction dict:
    features = extract_features(transaction_dict, schema="kaggle")
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature name sets
# ---------------------------------------------------------------------------

KAGGLE_FEATURES: list[str] = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
PAYSIM_FEATURES: list[str] = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "type_CASH_OUT",
    "type_TRANSFER",
]

SCHEMA_FEATURES: dict[str, list[str]] = {
    "kaggle": KAGGLE_FEATURES,
    "paysim": PAYSIM_FEATURES,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FeaturePipeline:
    """End-to-end feature engineering pipeline.

    Wraps a :class:`StandardScaler` and tracks feature names so that
    training, validation, and production inference all use the same
    column order and scaling.

    Attributes:
        feature_names: Ordered list of feature column names.
        scaler: Fitted StandardScaler (None until ``fit`` is called).
    """

    feature_names: list[str] = field(default_factory=lambda: list(KAGGLE_FEATURES))
    scaler: Optional[StandardScaler] = None

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "FeaturePipeline":
        """Fit the scaler on training data.

        Args:
            X: 2-D array of shape ``(n_samples, n_features)``.

        Returns:
            self
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X has {X.shape[1]} columns but {len(self.feature_names)} "
                f"feature names are configured: {self.feature_names}"
            )
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        logger.info(
            "Fitted StandardScaler on %d samples, %d features",
            X.shape[0],
            X.shape[1],
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale features using the fitted scaler.

        Args:
            X: 2-D array of shape ``(n_samples, n_features)``.

        Returns:
            Scaled array of the same shape.
        """
        if self.scaler is None:
            raise RuntimeError("FeaturePipeline has not been fitted. Call fit() first.")
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X has {X.shape[1]} columns but {len(self.feature_names)} "
                f"feature names are configured"
            )
        return self.scaler.transform(X.astype(np.float64)).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            X: 2-D array of shape ``(n_samples, n_features)``.

        Returns:
            Scaled array.
        """
        return self.fit(X).transform(X)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize pipeline parameters to a dict.

        Returns:
            Dict with feature names, means, and stds.
        """
        if self.scaler is None:
            raise RuntimeError("Pipeline has not been fitted.")
        return {
            "feature_names": list(self.feature_names),
            "mean": self.scaler.mean_.tolist(),
            "std": self.scaler.scale_.tolist(),
            "n_features": len(self.feature_names),
        }

    def save(self, path: str | Path) -> None:
        """Save pipeline parameters to a JSON file.

        Args:
            path: Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved FeaturePipeline to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "FeaturePipeline":
        """Load a pipeline from a JSON file created by :meth:`save`.

        Args:
            path: Path to the JSON file.

        Returns:
            Reconstructed FeaturePipeline with fitted scaler.
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        pipeline = cls(feature_names=data["feature_names"])
        pipeline.scaler = StandardScaler()
        pipeline.scaler.mean_ = np.array(data["mean"], dtype=np.float64)
        pipeline.scaler.scale_ = np.array(data["std"], dtype=np.float64)
        pipeline.scaler.var_ = pipeline.scaler.scale_**2
        pipeline.scaler.n_features_in_ = data["n_features"]
        logger.info("Loaded FeaturePipeline from %s", path)
        return pipeline


# ---------------------------------------------------------------------------
# Schema-aware extraction helpers
# ---------------------------------------------------------------------------

def _build_paysim_feature_vector(tx: dict[str, Any]) -> np.ndarray:
    """Build a 7-element feature vector from a PaySim-style transaction dict.

    Args:
        tx: Dict with keys: amount, oldbalanceOrg, newbalanceOrig,
            oldbalanceDest, newbalanceDest, type (one-hot encoded separately).

    Returns:
        Float32 array of shape (7,).
    """
    amount = float(tx.get("amount", 0.0))
    old_org = float(tx.get("oldbalanceOrg", 0.0))
    new_org = float(tx.get("newbalanceOrig", 0.0))
    old_dst = float(tx.get("oldbalanceDest", 0.0))
    new_dst = float(tx.get("newbalanceDest", 0.0))
    tx_type = tx.get("type", "").upper()
    type_cash_out = 1.0 if tx_type == "CASH_OUT" else 0.0
    type_transfer = 1.0 if tx_type == "TRANSFER" else 0.0

    return np.array(
        [amount, old_org, new_org, old_dst, new_dst, type_cash_out, type_transfer],
        dtype=np.float32,
    )


def _build_kaggle_feature_vector(tx: dict[str, Any]) -> np.ndarray:
    """Build a 30-element feature vector from a Kaggle CreditCard-style dict.

    Args:
        tx: Dict with keys V1–V28, Amount, Time.

    Returns:
        Float32 array of shape (30,).
    """
    vec = []
    for name in KAGGLE_FEATURES:
        vec.append(float(tx.get(name, 0.0)))
    return np.array(vec, dtype=np.float32)


_SCHEMA_BUILDERS: dict[str, Any] = {
    "paysim": _build_paysim_feature_vector,
    "kaggle": _build_kaggle_feature_vector,
}


def extract_features(
    transaction: dict[str, Any],
    schema: str = "kaggle",
    *,
    pipeline: Optional[FeaturePipeline] = None,
) -> np.ndarray:
    """Extract and optionally scale features from a raw transaction dict.

    Args:
        transaction: Raw transaction data as a dict.
        schema: Feature schema — ``"kaggle"`` or ``"paysim"``.
        pipeline: If provided, the raw feature vector is scaled before
            returning. The pipeline must already be fitted.

    Returns:
        1-D float32 array of feature values (scaled if pipeline given).
    """
    if schema not in _SCHEMA_BUILDERS:
        raise ValueError(
            f"Unknown schema {schema!r}. Valid options: {sorted(_SCHEMA_BUILDERS)}"
        )

    raw_vec = _SCHEMA_BUILDERS[schema](transaction)

    if pipeline is not None:
        if pipeline.scaler is None:
            raise RuntimeError("pipeline has not been fitted.")
        raw_vec = pipeline.scaler.transform(raw_vec.reshape(1, -1))[0]

    return raw_vec


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class FeatureValidationError(Exception):
    """Raised when a feature vector fails validation."""


def validate_features(
    features: np.ndarray,
    feature_names: Optional[list[str]] = None,
    *,
    allow_nan: bool = False,
) -> None:
    """Validate a feature vector for completeness and shape.

    Args:
        features: 1-D or 2-D feature array.
        feature_names: Expected feature names (length must match last dim).
        allow_nan: If True, NaN values are tolerated.

    Raises:
        FeatureValidationError: If validation fails.
    """
    if features.ndim not in (1, 2):
        raise FeatureValidationError(
            f"Features must be 1-D or 2-D, got {features.ndim}-D with shape {features.shape}"
        )

    n_features = features.shape[-1] if features.ndim == 2 else features.shape[0]

    if feature_names is not None and n_features != len(feature_names):
        raise FeatureValidationError(
            f"Feature vector has {n_features} elements but "
            f"{len(feature_names)} names were provided: {feature_names}"
        )

    if not allow_nan and np.isnan(features).any():
        raise FeatureValidationError("Feature vector contains NaN values")

    if np.isinf(features).any():
        raise FeatureValidationError("Feature vector contains Inf values")


def validate_transaction_dict(
    transaction: dict[str, Any],
    schema: str = "kaggle",
) -> None:
    """Validate that a transaction dict contains all required features.

    Args:
        transaction: Raw transaction dict.
        schema: Feature schema name.

    Raises:
        FeatureValidationError: If required keys are missing.
    """
    if schema not in SCHEMA_FEATURES:
        raise ValueError(
            f"Unknown schema {schema!r}. Valid options: {sorted(SCHEMA_FEATURES)}"
        )

    required = set(SCHEMA_FEATURES[schema])
    present = set(transaction.keys())
    missing = required - present

    if missing:
        raise FeatureValidationError(
            f"Transaction is missing required features for schema '{schema}': "
            f"{sorted(missing)}"
        )
