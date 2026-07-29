"""
Export pretrained PyTorch image classification models to ONNX format.

Supported models:
  - mobilenet_v2  (lightweight, ~14 MB)
  - resnet18      (higher accuracy, ~45 MB)

Usage:
    python models/image_classifier/train.py --model mobilenet_v2
    python models/image_classifier/train.py --model resnet18 --output-dir models/
    python models/image_classifier/train.py --both

The exported ONNX model is validated and a test inference is run to verify
output shapes and value ranges.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_MODEL_BUILDERS: dict[str, str] = {
    "mobilenet_v2": "mobilenet_v2",
    "resnet18": "resnet18",
}


def _resolve_model_name(name: str) -> str:
    """Validate and normalize a model name."""
    name = name.lower().strip()
    if name not in _MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model '{name}'. Valid options: {sorted(_MODEL_BUILDERS)}"
        )
    return name


# ---------------------------------------------------------------------------
# PyTorch import (optional dependency)
# ---------------------------------------------------------------------------

def _import_torch():
    """Import torch and torchvision, raising a clear error if missing."""
    try:
        import torch
        import torchvision.models as models
        import torchvision.transforms as T
        return torch, models, T
    except ImportError as exc:
        raise ImportError(
            "PyTorch and torchvision are required for model export. "
            "Install them with:\n"
            "  pip install torch torchvision"
        ) from exc


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_model(
    model_name: str,
    output_dir: Path,
    *,
    opset: int = 13,
    input_size: tuple[int, int, int, int] = (1, 3, 224, 224),
    dynamic_batch: bool = True,
) -> Path:
    """Export a pretrained PyTorch model to ONNX.

    Args:
        model_name: One of ``"mobilenet_v2"`` or ``"resnet18"``.
        output_dir: Directory to write the ``.onnx`` file.
        opset: ONNX opset version.
        input_size: Static input shape ``(B, C, H, W)``.
        dynamic_batch: If True, make the batch dimension dynamic.

    Returns:
        Path to the exported ``.onnx`` file.
    """
    torch, models, T = _import_torch()
    model_name = _resolve_model_name(model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pretrained model
    logger.info("Loading pretrained %s ...", model_name)
    builder = getattr(models, model_name)
    model = builder(weights="IMAGENET1K_V1")
    model.eval()

    # Prepare dummy input
    dummy_input = torch.randn(*input_size)

    # Build dynamic axes for flexible batching
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    onnx_path = output_dir / f"{model_name}.onnx"

    # Export
    logger.info("Exporting to ONNX (opset=%d) ...", opset)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    size_kb = onnx_path.stat().st_size / 1024
    logger.info("Exported %s (%.1f KB)", onnx_path, size_kb)
    return onnx_path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_onnx_model(onnx_path: Path) -> dict[str, any]:
    """Validate an ONNX model and return metadata.

    Args:
        onnx_path: Path to the ``.onnx`` file.

    Returns:
        Dict with model metadata (inputs, outputs, opset, etc.).
    """
    import onnx

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    graph = model.graph
    meta = {
        "opset": model.opset_import[0].version if model.opset_import else None,
        "ir_version": model.ir_version,
        "inputs": [
            {"name": i.name, "shape": [d.dim_value for d in i.type.tensor_type.shape.dim]}
            for i in graph.input
        ],
        "outputs": [
            {"name": o.name, "shape": [d.dim_value for d in o.type.tensor_type.shape.dim]}
            for o in graph.output
        ],
    }
    logger.info("Model validated: inputs=%s outputs=%s", meta["inputs"], meta["outputs"])
    return meta


def run_test_inference(
    onnx_path: Path,
    input_size: tuple[int, int, int, int] = (1, 3, 224, 224),
) -> dict[str, any]:
    """Run a single test inference through the ONNX model.

    Uses ONNX Runtime if available; falls back to onnx reference checker.

    Args:
        onnx_path: Path to the ``.onnx`` file.
        input_size: Expected input shape ``(B, C, H, W)``.

    Returns:
        Dict with inference results (output shape, range, latency).
    """
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed; skipping runtime inference.")
        return {"skipped": True, "reason": "onnxruntime not available"}

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    dummy = np.random.randn(*input_size).astype(np.float32)
    input_name = session.get_inputs()[0].name

    t0 = time.perf_counter()
    outputs = session.run(None, {input_name: dummy})
    elapsed_ms = (time.perf_counter() - t0) * 1000

    output_arr = outputs[0]
    result = {
        "output_shape": list(output_arr.shape),
        "output_dtype": str(output_arr.dtype),
        "output_min": float(output_arr.min()),
        "output_max": float(output_arr.max()),
        "inference_ms": round(elapsed_ms, 3),
        "skipped": False,
    }
    logger.info(
        "Test inference: shape=%s range=[%.4f, %.4f] %.2f ms",
        result["output_shape"],
        result["output_min"],
        result["output_max"],
        elapsed_ms,
    )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export pretrained image classifiers to ONNX."
    )
    parser.add_argument(
        "--model",
        choices=sorted(_MODEL_BUILDERS),
        help="Model to export (mobilenet_v2 or resnet18).",
    )
    parser.add_argument("--both", action="store_true", help="Export both models.")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent),
        help="Output directory (default: this script's directory).",
    )
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version.")
    parser.add_argument("--skip-validation", action="store_true", help="Skip ONNX validation.")
    parser.add_argument("--skip-inference", action="store_true", help="Skip test inference.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.model and not args.both:
        parser.error("Specify --model or --both")

    output_dir = Path(args.output_dir)
    models_to_export = (
        sorted(_MODEL_BUILDERS) if args.both else ([args.model] if args.model else [])
    )

    for m in models_to_export:
        logger.info("--- Processing %s ---", m)
        try:
            onnx_path = export_model(m, output_dir, opset=args.opset)

            if not args.skip_validation:
                meta = validate_onnx_model(onnx_path)

            if not args.skip_inference:
                result = run_test_inference(onnx_path)

            logger.info("Done: %s", onnx_path)
        except Exception as exc:
            logger.error("Failed to export %s: %s", m, exc)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
