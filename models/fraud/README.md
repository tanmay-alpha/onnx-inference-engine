# Fraud Detection Model

## Overview

The fraud detection model classifies financial transactions as legitimate or fraudulent.
It is trained on synthetic PaySim-style data and exported to ONNX format for deployment
in the inference engine.

## Features

The model uses the following 30 features derived from each transaction:

| Feature | Description |
|---------|-------------|
| `V1` - `V28` | Anonymized principal components from PCA transformation |
| `Amount` | Transaction amount |
| `Time` | Seconds elapsed between each transaction and the first transaction in the dataset |

These correspond to the Kaggle CreditCard Fraud Detection dataset schema.

## Model

- **Algorithm**: Gradient Boosting (LightGBM preferred, XGBoost fallback)
- **ONNX opset**: 13
- **Threshold**: 0.5 (configurable in `model_config.json`)

## Performance

| Metric | Value |
|--------|-------|
| AUC-ROC | ~0.97 (synthetic data) |
| Precision | ~0.85 |
| Recall | ~0.75 |
| F1 Score | ~0.80 |

> Note: Performance metrics are recomputed on each training run and logged to
> the experiments directory.

## Retraining Schedule

The model should be retrained:
- Weekly with the latest labeled transaction data
- Immediately after a significant distribution shift is detected (e.g., fraud pattern change)

To retrain:
```bash
python scripts/retrain_fraud.py --data-path data/processed/transactions.csv
```

The retraining script will:
1. Load the latest data from the specified path (or Supabase if configured)
2. Apply the feature engineering pipeline
3. Train a new model with current hyperparameters
4. Compare against the current production model
5. Promote to registry if the new model outperforms the baseline

## Configuration

Training hyperparameters are in `config.yaml`. Key settings:

```yaml
model_type: "gradient_boosting"
n_estimators: 200
max_depth: 5
learning_rate: 0.1
test_size: 0.2
random_state: 42
```

## Experiment Tracking

Training runs are tracked in `experiments/` via MLflow. Each run includes:
- Git commit hash
- Hyperparameters
- Feature pipeline version
- Model metrics
- ONNX artifact

## Model Artifacts

| File | Description |
|------|-------------|
| `fraud_detector.onnx` | Current production ONNX model |
| `model_config.json` | Feature names, scaler parameters, threshold |
| `registry/fraud_detector-v{N}.onnx` | Versioned model snapshots |
