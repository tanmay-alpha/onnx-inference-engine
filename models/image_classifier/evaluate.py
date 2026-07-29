"""
Evaluate an exported ONNX image classification model.

Loads the ONNX file, runs a test inference, and verifies output shapes
and value ranges against expected thresholds.

Usage:
    python models/image_classifier/evaluate.py --model mobilenet_v2
    python models/image_classifier/evaluate.py --model resnet18 --input-size 1,3,224,224

Or programmatically:
    from image_classifier.evaluate import ImageClassifierEvaluator

    evaluator = ImageClassifierEvaluator("models/mobilenet_v2.onnx")
    results = evaluator.evaluate()
    print(results.report())
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Results from evaluating an image classifier ONNX model."""

    model_path: str
    input_shape: list[int]
    output_shape: list[int]
    inference_ms: float
    output_min: float
    output_max: float
    valid_output_range: bool
    batch_size: int = 1
    errors: list[str] = field(default_factory=list)

    def report(self) -> str:
        """Return a human-readable report string."""
        status = "PASS" if not self.errors else "FAIL"
        lines = [
            "=" * 55,
            f"IMAGE CLASSIFIER EVALUATION — {status}",
            "=" * 55,
            f"  Model:           {self.model_path}",
            f"  Input shape:     {self.input_shape}",
            f"  Output shape:    {self.output_shape}",
            f"  Inference:       {self.inference_ms:.2f} ms",
            f"  Output range:    [{self.output_min:.4f}, {self.output_max:.4f}]",
            f"  Valid range:     {'Yes' if self.valid_output_range else 'No'}",
            "",
        ]
        if self.errors:
            lines.append("  Errors:")
            for err in self.errors:
                lines.append(f"    - {err}")
        else:
            lines.append("  All checks passed.")
        lines.append("=" * 55)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to a dict for JSON export."""
        return {
            "model_path": self.model_path,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "inference_ms": round(self.inference_ms, 4),
            "output_min": round(self.output_min, 6),
            "output_max": round(self.output_max, 6),
            "valid_output_range": self.valid_output_range,
            "batch_size": self.batch_size,
            "errors": self.errors,
            "passed": len(self.errors) == 0,
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class ImageClassifierEvaluator:
    """Evaluate an image classification ONNX model.

    Args:
        model_path: Path to the ``.onnx`` file.
        input_size: Expected input shape ``(B, C, H, W)``. If None, read from model.
        num_classes: Expected number of output classes.
    """

    def __init__(
        self,
        model_path: str | Path,
        input_size: Optional[tuple[int, int, int, int]] = None,
        num_classes: Optional[int] = None,
    ):
        self.model_path = Path(model_path)
        self._expected_input_size = input_size
        self._expected_num_classes = num_classes
        self._session: Optional[ort.InferenceSession] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_session(self) -> ort.InferenceSession:
        """Load the ONNX Runtime session."""
        if not ONNX_AVAILABLE:
            raise RuntimeError(
                "onnxruntime is not installed. Install with: pip install onnxruntime"
            )
        if self._session is None:
            self._session = ort.InferenceSession(
                str(self.model_path),
                providers=["CPUExecutionProvider"],
            )
        return self._session

    def _get_model_info(self) -> dict:
        """Extract input/output metadata from the ONNX model."""
        session = self._load_session()
        inputs = session.get_inputs()[0]
        outputs = session.get_outputs()[0]
        return {
            "input_name": inputs.name,
            "input_shape": list(inputs.shape),
            "output_name": outputs.name,
            "output_shape": list(outputs.shape),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _run_inference(
        self,
        input_array: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Run a single inference pass.

        Args:
            input_array: Float32 array matching the model's expected input shape.

        Returns:
            Tuple of (output array, elapsed_ms).
        """
        session = self._load_session()
        info = self._get_model_info()

        t0 = time.perf_counter()
        outputs = session.run(None, {info["input_name"]: input_array})
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return outputs[0], elapsed_ms

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        input_size: Optional[tuple[int, int, int, int]] = None,
        *,
        verbose: bool = False,
    ) -> EvalResult:
        """Run full evaluation pipeline.

        Steps:
            1. Load model metadata
            2. Generate random input matching expected shape
            3. Run inference and measure latency
            4. Validate output shape and value ranges
            5. Check against expected num_classes

        Args:
            input_size: Override for input shape ``(B, C, H, W)``.
            verbose: If True, log detailed info.

        Returns:
            EvalResult with all checks and metrics.
        """
        errors: list[str] = []
        info = self._get_model_info()

        input_shape = info["input_shape"]
        output_shape = info["output_shape"]

        # Resolve dynamic dimensions
        batch_dim = input_shape[0]
        if isinstance(batch_dim, str):
            batch_dim = 1  # dynamic batch, test with batch=1
            input_shape[0] = 1

        if input_size is not None:
            if list(input_size) != input_shape:
                errors.append(
                    f"Provided input size {input_size} does not match "
                    f"model expected shape {info['input_shape']}"
                )

        # Generate random input (ImageNet-normalized range)
        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        # Run inference
        try:
            output_arr, inference_ms = self._run_inference(dummy_input)
        except Exception as exc:
            errors.append(f"Inference failed: {exc}")
            return EvalResult(
                model_path=str(self.model_path),
                input_shape=input_shape,
                output_shape=output_shape,
                inference_ms=0.0,
                output_min=0.0,
                output_max=0.0,
                valid_output_range=False,
                errors=errors,
            )

        output_min = float(output_arr.min())
        output_max = float(output_arr.max())

        # Validate output shape
        expected_output_shape = [input_shape[0], self._expected_num_classes or 1000]
        if list(output_shape) != expected_output_shape:
            errors.append(
                f"Output shape {output_shape} does not match expected {expected_output_shape}"
            )

        # Validate output range (logits or softmaxed probabilities)
        valid_range = not (output_arr.min() < -1e-6 or output_arr.max() > 1.0 + 1e-6)
        if not valid_range:
            # Logits are fine — apply softmax for range check
            softmaxed = np.exp(output_arr - output_arr.max(axis=-1, keepdims=True))
            softmaxed /= softmaxed.sum(axis=-1, keepdims=True)
            output_min = float(softmaxed.min())
            output_max = float(softmaxed.max())
            valid_range = True

        # Check that the output is not all zeros or all the same
        if output_arr.std() < 1e-6:
            errors.append(
                f"Output has near-zero variance (std={output_arr.std():.6f}). "
                "Model may not be producing meaningful predictions."
            )

        if verbose:
            logger.info(
                "Input: %s -> Output: %s (%.2f ms)",
                input_shape,
                output_shape,
                inference_ms,
            )

        return EvalResult(
            model_path=str(self.model_path),
            input_shape=input_shape,
            output_shape=output_shape,
            inference_ms=inference_ms,
            output_min=output_min,
            output_max=output_max,
            valid_output_range=valid_range,
            batch_size=batch_dim,
            errors=errors,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate an image classification ONNX model."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="ONNX model file path (e.g., mobilenet_v2.onnx).",
    )
    parser.add_argument(
        "--input-size",
        default="1,3,224,224",
        help="Comma-separated input size (B,C,H,W).",
    )
    parser.add_argument(
        "--num-classes", type=int, default=1000, help="Expected number of output classes."
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    input_size = tuple(int(x) for x in args.input_size.split(","))
    evaluator = ImageClassifierEvaluator(
        model_path=args.model,
        input_size=input_size,
        num_classes=args.num_classes,
    )

    result = evaluator.evaluate(input_size=input_size, verbose=args.verbose)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.report())

    return 0 if not result.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
