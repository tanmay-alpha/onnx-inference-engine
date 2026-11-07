# Image classifier export (optional)

Nothing is trained here. The committed `models/mobilenet_v2.onnx` and
`models/resnet18.onnx` come from the ONNX model zoo (`models/download_models.py`),
not from these scripts.

* `train.py` loads torchvision's pretrained `IMAGENET1K_V1` MobileNetV2 or
  ResNet18 weights and exports them to ONNX with `torch.onnx.export`, then
  checks the file and runs one random input through ONNX Runtime.
* `evaluate.py` loads an ONNX file in ONNX Runtime, feeds it random input and
  reports output shape/range checks and latency. It does not measure accuracy.

No accuracy figures are produced or claimed in this repository.

```bash
pip install torch torchvision onnx onnxruntime numpy
python models/image_classifier/train.py --model mobilenet_v2 --output-dir /tmp/export
python models/image_classifier/train.py --both --opset 18 --output-dir /tmp/export
python models/image_classifier/evaluate.py --model models/mobilenet_v2.onnx
```

## Input and output

* Input: float32 NCHW `[1, 3, 224, 224]`, RGB, scaled to [0, 1] and then
  normalised with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
* Output: `[1, 1000]` ImageNet logits (apply softmax for probabilities).

```python
from PIL import Image
import numpy as np

img = Image.open("image.jpg").convert("RGB").resize((224, 224))
arr = np.asarray(img, dtype=np.float32) / 255.0
arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
x = arr.transpose(2, 0, 1)[np.newaxis].astype(np.float32)  # [1, 3, 224, 224]
```

## Running with Crucible

Python bindings (build the engine first; see the root README):

```python
import crucible_py
m = crucible_py.load_model("models/mobilenet_v2.onnx")
name = crucible_py.get_model_info(m)["inputs"][0]["name"]
logits = crucible_py.run(m, {name: x})
```

CLI (`cli/`). The input is a JSON tensor file `{"shape": [1, 3, 224, 224], "data": [...]}`:

```bash
crucible run --model models/mobilenet_v2.onnx --input input.json
```
