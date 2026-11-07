# Models

```
models/
├── download_models.py       # Re-download the two ONNX model zoo files below
├── generate_fixtures.py     # Writes the tiny parser fixtures in engine/tests/fixtures/
├── mobilenet_v2.onnx        # ONNX model zoo MobileNetV2 (opset 7), not trained here
├── resnet18.onnx            # ONNX model zoo ResNet18 v2 (opset 7), not trained here
├── fraud/                   # 7-feature logistic-regression demo model + its training script
├── image_classifier/        # Optional: export torchvision MobileNetV2/ResNet18 to ONNX
└── data/                    # Empty; where to put the PaySim CSV if you train on real data
```

## Pretrained CNNs

`mobilenet_v2.onnx` and `resnet18.onnx` are the ONNX model zoo's
`mobilenetv2-7.onnx` and `resnet18-v2-7.onnx`, committed unmodified so the parity
tests and benchmarks run offline. `models/*.onnx` is gitignored, so these two
files were force-added. To fetch them again (SHA-256 checked):

```bash
python models/download_models.py                 # both
python models/download_models.py --model resnet18
```

## Fraud model

`fraud/fraud_detector.onnx` is a `MatMul → Add → Sigmoid` logistic regression
trained on synthetic data. `python models/fraud/train_fraud_model.py`
reproduces it byte for byte. See [`fraud/README.md`](fraud/README.md) for the
data, the feature order and normalisation, and how to train on real PaySim data.

## Requirements

```bash
pip install numpy scikit-learn onnx          # fraud model
pip install torch torchvision onnx onnxruntime  # image_classifier/ only
```
