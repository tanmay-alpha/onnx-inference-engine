# Models Directory

This directory contains all ML models, training pipelines, and data management for the ONNX inference engine.

## Directory Structure

```
models/
├── README.md                    # This file
├── .gitkeep
├── download_models.py           # Download pretrained models from ONNX model zoo
├── generate_fixtures.py         # Generate test fixtures for the C++ parser
├── mobilenet_v2.onnx            # MobileNetV2 pretrained model
├── resnet18.onnx                # ResNet18 pretrained model
│
├── fraud/                       # Fraud detection pipeline
│   ├── README.md
│   ├── config.yaml              # Training hyperparameters
│   ├── train_fraud_model.py     # Training entry point (refactored)
│   ├── features.py              # Feature engineering pipeline
│   ├── evaluate.py              # Model evaluation and benchmarking
│   ├── fraud_detector.onnx      # Production model
│   ├── model_config.json        # Feature metadata and scaler params
│   └── experiments/             # MLflow tracking artifacts
│       ├── .gitkeep
│       └── README.md
│
├── image_classifier/            # Image classification pipeline
│   ├── README.md
│   ├── config.yaml
│   ├── train.py                 # PyTorch -> ONNX export
│   ├── evaluate.py
│
├── data/                        # Data management
│   ├── raw/                     # Original datasets (gitignored)
│   ├── processed/               # Cleaned, split, scaled data
│   └── external/                # Third-party datasets
│
├── registry/                    # Production model registry
│   ├── .gitkeep
│   └── README.md
│
└── scripts/                     # Utility scripts
    ├── retrain_fraud.py         # Scheduled retraining script
    └── compare_models.py        # Model comparison utility
```

## Quick Start

### Download Pretrained Models

```bash
python models/download_models.py
python models/download_models.py --model mobilenet_v2
python models/download_models.py --model resnet18
```

### Train Fraud Detection Model

```bash
python models/fraud/train_fraud_model.py
```

This will:
1. Generate synthetic PaySim-style transaction data
2. Apply feature engineering from `features.py`
3. Train a gradient boosting model using hyperparameters from `config.yaml`
4. Evaluate using `evaluate.py` (precision, recall, F1, AUC-ROC)
5. Export the model to ONNX format
6. Save to `registry/fraud_detector-v1.onnx`

### Train Image Classifier

```bash
python models/image_classifier/train.py --model mobilenet_v2
python models/image_classifier/train.py --model resnet18
```

### Retrain Fraud Model (Scheduled)

```bash
python models/scripts/retrain_fraud.py --data-path data/processed/latest.csv
```

### Compare Two ONNX Models

```bash
python models/scripts/compare_models.py \
  models/registry/fraud_detector-v1.onnx \
  models/registry/fraud_detector-v2.onnx \
  --test-data data/processed/test.csv
```

## Experiment Tracking

Training runs log to `models/fraud/experiments/` via MLflow. Each run captures:
- Hyperparameters (from `config.yaml`)
- Feature pipeline version
- Model metrics (precision, recall, F1, AUC-ROC)
- ONNX model artifact

## Model Registry

The `models/registry/` directory is the single source of truth for production models.

- Models follow the naming convention: `{model_name}-v{N}.onnx`
- The current production model is symlinked or copied from the latest version
- See `models/registry/README.md` for promotion and rollback procedures

## Requirements

```bash
pip install numpy scikit-learn onnx onnxruntime pyyaml
```

For gradient boosting (preferred):
```bash
pip install lightgbm
```

Optional (for experiment tracking):
```bash
pip install mlflow
```
