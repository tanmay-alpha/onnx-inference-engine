"""ONNX Runtime benchmark (Issue #14).

Same shape as bench_crucible.py: 100 timed runs after 10 warmup
iterations of MobileNetV2 forward inference, reported as JSON.

ONNX Runtime is the most direct comparison for Crucible because
both engines consume the same .onnx file; the only difference
is the dispatch / kernel implementations. PyTorch's eager mode
also consumes .onnx files (via onnx -> torch.onnx under the
hood) but goes through a different runtime, so its numbers are
informative rather than directly competitive.

Output format is identical to bench_crucible.py so run_all.py
can concatenate them without a separate parser.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Constants — kept in lockstep with bench_crucible.py so the JSONs
# can be plotted side-by-side.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO_ROOT / "models" / "mobilenet_v2.onnx"
DEFAULT_INPUT_SHAPE = (1, 3, 224, 224)
DEFAULT_RUNS = 100
DEFAULT_WARMUP = 10


# ---------------------------------------------------------------------------
# Engine loading
# ---------------------------------------------------------------------------
def _load_session(model_path: Path):
    """Load an ONNX Runtime InferenceSession.

    We force CPUExecutionProvider here. The other common choice
    is CUDAExecutionProvider, but that requires CUDA + cuDNN on
    the host and would skew the comparison across machines that
    have different GPUs. CPU is the lowest common denominator
    and matches the deployment target of the FastAPI server.
    """
    import onnxruntime as ort  # imported lazily so a missing dep
                              # only blows up at bench time, not at
                              # import time of every other script
    sess_options = ort.SessionOptions()
    # Disable ORT's built-in graph optimisation logs. They're
    # useful the first time, noisy the 100th time.
    sess_options.log_severity_level = 3
    return ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )


# ---------------------------------------------------------------------------
# Benchmark loop — exact same structure as bench_crucible.py.
# ---------------------------------------------------------------------------
def _bench(sess, x: np.ndarray, input_name: str,
           runs: int, warmup: int) -> List[float]:
    for _ in range(warmup):
        sess.run(None, {input_name: x})
    timings: List[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: x})
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
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.model.is_file():
        print(f"error: model not found at {args.model}", file=sys.stderr)
        return 2

    sess = _load_session(args.model)
    input_name = sess.get_inputs()[0].name
    providers = sess.get_providers()
    print(f"[bench_onnxruntime] providers={providers} model={args.model.name} "
          f"runs={args.runs} warmup={args.warmup}", file=sys.stderr)

    rng = np.random.default_rng(args.seed)
    x = rng.standard_normal(DEFAULT_INPUT_SHAPE).astype(np.float32)

    timings = _bench(sess, x, input_name, args.runs, args.warmup)
    summary = _summarise(timings)

    result = {
        "engine": "onnxruntime",
        "backend": providers[0] if providers else "unknown",
        "model": str(args.model),
        "input_shape": list(DEFAULT_INPUT_SHAPE),
        "stats": summary,
    }

    out = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out + "\n")
        print(f"[bench_onnxruntime] wrote {args.output}", file=sys.stderr)
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())