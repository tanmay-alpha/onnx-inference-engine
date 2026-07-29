# Experiments Directory

This directory stores MLflow experiment tracking artifacts for the fraud detection model.

## Setup

### Prerequisites

```bash
pip install mlflow
```

### Start MLflow UI

```bash
mlflow ui --backend-store-uri file://./mlruns --port 5000
```

Open http://localhost:5000 to browse experiments.

## What Gets Logged

Each training run logs the following to MLflow:

| Category | Items |
|----------|-------|
| **Parameters** | `model_type`, `n_estimators`, `max_depth`, `learning_rate`, `test_size`, `random_state`, `use_smote`, `onnx_opset` |
| **Metrics** | `precision`, `recall`, `f1_score`, `auc_roc`, `inference_ms`, `n_samples` |
| **Artifacts** | ONNX model file, feature pipeline JSON, training log |
| **Tags** | `git_commit`, `pipeline_version`, `schema_version` |

## Naming Convention

- **Experiment**: `fraud_detection`
- **Run name**: `{model_type}-v{N}_{timestamp}`
- **Artifact key**: `model` → contains the ONNX file and config

## Integration

The training script (`train_fraud_model.py`) logs runs automatically if MLflow
is installed. To disable tracking, set the environment variable:

```bash
export MLFLOW_TRACKING_URI=""  # disables tracking
```
