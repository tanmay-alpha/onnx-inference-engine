# Model Registry

The model registry is the single source of truth for production-ready ONNX models.
All models deployed to the inference engine are promoted through this directory.

## Directory Layout

```
registry/
├── .gitkeep
├── README.md              # This file
├── fraud_detector-v1.onnx # Current production model
├── fraud_detector-v2.onnx # Next candidate (pending promotion)
└── mobilenet_v2-v1.onnx   # Image classification model
```

## Naming Convention

All models follow the pattern:

```
{model_name}-v{Major}.onnx
```

Examples:
- `fraud_detector-v1.onnx` — Fraud detection, version 1
- `mobilenet_v2-v1.onnx` — Image classification, version 1

Version numbers increment when the model architecture, training data, or
hyperparameters change significantly. Patch-level changes (scaler updates,
minor feature additions) reuse the same major version.

## Promoting a Model to Production

### Step 1: Train the new model

```bash
python models/fraud/train_fraud_model.py
```

This saves the new model to `models/fraud/fraud_detector.onnx`.

### Step 2: Copy to registry with versioned name

```bash
# Determine the next version number
ls models/registry/fraud_detector-*.onnx
# If highest is v2, next is v3

cp models/fraud/fraud_detector.onnx models/registry/fraud_detector-v3.onnx
```

### Step 3: Validate

```bash
python models/fraud/evaluate.py --model models/registry/fraud_detector-v3.onnx
```

Compare the new model's metrics against the current production model.

### Step 4: Update production pointer

```bash
# Copy the new version as the canonical production model
cp models/registry/fraud_detector-v3.onnx models/fraud/fraud_detector.onnx

# Also update any deployed copies
cp models/registry/fraud_detector-v3.onnx web/public/models/fraud_detector.onnx
```

## Rolling Back

If a newly promoted model causes issues:

```bash
# Revert to the previous version
cp models/registry/fraud_detector-v2.onnx models/fraud/fraud_detector.onnx
cp models/registry/fraud_detector-v2.onnx web/public/models/fraud_detector.onnx
```

Always keep at least the last two versions in the registry for quick rollback.

## Model Metadata

Each model version should have a companion metadata file:

```
fraud_detector-v1.onnx
fraud_detector-v1.meta.json   # training date, metrics, data version, git commit
```

The `.meta.json` file should contain:

```json
{
  "version": 1,
  "model_name": "fraud_detector",
  "trained_at": "2026-07-24T10:00:00Z",
  "git_commit": "abc1234",
  "training_data": "paysim-synthetic-v2",
  "metrics": {
    "precision": 0.85,
    "recall": 0.75,
    "f1": 0.80,
    "auc_roc": 0.97
  },
  "hyperparameters": {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.1
  },
  "status": "production"
}
```

## Automated Promotion

The retraining script (`models/scripts/retrain_fraud.py`) automates this process:

1. Trains a new model
2. Evaluates it against the current production model
3. If the new model is better, automatically promotes it to the registry
4. Logs the promotion decision with metrics

## Lifecycle

```
Train → Evaluate → Registry (staging) → Validate → Promote → Production
                                                │
                                                └─ Rollback if issues
```
