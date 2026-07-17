"""PyTorch eager-mode benchmark (Issue #14).

Same shape as bench_crucible.py / bench_onnxruntime.py: 100 timed
runs after 10 warmup iterations, reported as JSON.

The natural PyTorch comparison for the .onnx-based Crucible and
ONNX Runtime benches is to run the *same architecture* under
PyTorch's eager mode. We have two options:

  1. Load torchvision.models.mobilenet_v2 (if torchvision is
     installed) and run it directly. The architecture is the
     same as the .onnx we benchmark Crucible on, so the
     comparison is apples-to-apples.

  2. If torchvision is not installed, build a small ConvNet
     stub that uses the same input/output shapes. The numbers
     are still informative — they tell you "PyTorch's eager
     dispatch on CPU" — but they're not directly comparable to
     a real MobileNetV2 forward pass.

We try option 1 first and fall back to option 2 with a warning.
The backend name in the JSON tells downstream tooling which
path was taken.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import warnings
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Constants — kept in lockstep with the other benches.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_SHAPE = (1, 3, 224, 224)
DEFAULT_RUNS = 100
DEFAULT_WARMUP = 10


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
class _TinyClassifier(nn.Module):
    """Fallback model used when torchvision isn't installed.

    It mirrors the input shape (1, 3, 224, 224) and produces a
    1000-class output, which is what the comparison consumers
    (recharts plot, etc.) expect. It is NOT a real MobileNetV2.
    """

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        # Keep parameter count small so the bench finishes in a
        # reasonable time on a developer laptop. The point of the
        # PyTorch bench is to measure dispatch overhead, not to
        # win at MobileNetV2.
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def _load_model() -> tuple[nn.Module, str, str]:
    """Try torchvision's MobileNetV2, fall back to a tiny stub.

    Returns (model, backend_name, model_name).
    """
    try:
        from torchvision.models import mobilenet_v2  # type: ignore[import-not-found]
        m = mobilenet_v2(weights=None)
        m.eval()
        return m, "torchvision-mobilenet_v2", "mobilenet_v2 (torchvision)"
    except ImportError:
        warnings.warn(
            "torchvision not installed; falling back to a tiny stub. "
            "The PyTorch numbers will not be directly comparable to "
            "Crucible / ONNX Runtime MobileNetV2 numbers.",
            stacklevel=2,
        )
        m = _TinyClassifier()
        m.eval()
        return m, "torch-stub", "tiny_convnet"


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------
def _bench(model: nn.Module, x: torch.Tensor,
           runs: int, warmup: int) -> List[float]:
    """Eager-mode forward with no_grad for fair comparison.

    We disable autograd because inference is what we're measuring.
    Re-enabling it would add operator-level bookkeeping and inflate
    every other engine's apparent latency.
    """
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        timings: List[float] = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            timings.append((time.perf_counter() - t0) * 1000.0)
    return timings


def percentile(sorted_data, p):
    """Standard percentile with linear interpolation."""
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    k = (n - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


def _summarise(timings_ms: List[float]) -> dict:
    sorted_t = sorted(timings_ms)
    n = len(sorted_t)
    return {
        "runs": n,
        "mean_ms":   statistics.fmean(sorted_t),
        "median_ms": statistics.median(sorted_t),
        "p95_ms": percentile(sorted_t, 95),
        "p99_ms": percentile(sorted_t, 99),
        "min_ms": sorted_t[0],
        "max_ms": sorted_t[-1],
        "throughput_inf_per_sec": min(1e6, 1000.0 / statistics.fmean(sorted_t)),
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    model, backend, model_name = _load_model()
    print(f"[bench_pytorch] backend={backend} model={model_name} "
          f"runs={args.runs} warmup={args.warmup}", file=sys.stderr)

    # Match the other benches: same seed, same shape, same dtype.
    rng = np.random.default_rng(args.seed)
    x_np = rng.standard_normal(DEFAULT_INPUT_SHAPE).astype(np.float32)
    x = torch.from_numpy(x_np)

    timings = _bench(model, x, args.runs, args.warmup)
    summary = _summarise(timings)

    result = {
        "engine": "pytorch",
        "backend": backend,
        "model": model_name,
        "input_shape": list(DEFAULT_INPUT_SHAPE),
        "stats": summary,
    }

    out = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out + "\n")
        print(f"[bench_pytorch] wrote {args.output}", file=sys.stderr)
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())