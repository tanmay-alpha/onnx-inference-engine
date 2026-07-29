# Image Classification Pipeline

This directory contains the pipeline for exporting pretrained image classification models to ONNX format.

## Supported Models

| Model | Size | Speed | Top-1 Accuracy |
|-------|------|-------|----------------|
| MobileNetV2 | ~14 MB | Fast | ~71% |
| ResNet18 | ~45 MB | Medium | ~70% |

## Usage

### Export a Model to ONNX

```bash
python models/image_classifier/train.py --model mobilenet_v2
python models/image_classifier/train.py --model resnet18
python models/image_classifier/train.py --both
```

### Evaluate an Exported Model

```bash
python models/image_classifier/evaluate.py --model mobilenet_v2
python models/image_classifier/evaluate.py --model resnet18 --json
```

### With Custom Settings

```bash
python models/image_classifier/train.py \
  --model mobilenet_v2 \
  --opset 18 \
  --output-dir models/
```

## Input Format

Models expect images normalized to ImageNet statistics:

- **Size**: 224 x 224 pixels
- **Channels**: RGB (3 channels)
- **Normalization**: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- **Tensor layout**: NCHW (batch, channels, height, width)

Example preprocessing:
```python
from PIL import Image
import numpy as np

img = Image.open("image.jpg").resize((224, 224))
arr = np.array(img, dtype=np.float32) / 255.0
arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
arr = arr.transpose(2, 0, 1)  # HWC -> CHW
arr = arr[np.newaxis, ...]    # add batch dim
```

## Output Format

Models output raw logits of shape `(batch_size, 1000)` corresponding to ImageNet classes.

To get probabilities, apply softmax:
```python
probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
top5_idx = probs.argsort()[-5:][::-1]
```

## Requirements

```bash
pip install torch torchvision onnx onnxruntime numpy Pillow
```

## Integration with Inference Engine

The exported ONNX models can be loaded directly by the ONNX inference engine:

```bash
# Engine loads the model at startup
python engine/server.py --model models/mobilenet_v2.onnx
```
