"""Crucible C++ engine benchmark (Issue #14).

Runs N=100 forward passes through the compiled crucible_py module
on a fixed MobileNetV2 input, after a configurable warm-up
(default 10). Reports mean / median / p95 / p99 / min / max
latency plus throughput in inferences/sec.

If crucible_py is not importable (the C++ engine hasn't been built
and installed yet), we fall back to the numpy-fallback path the
FastAPI server uses. The fallback is not representative of real
performance — the latency it reports is dominated by onnx.load
and numpy.zeros, not by graph execution — but it lets the
benchmark runner produce a complete JSON for the comparison
plot without forcing every developer to build the C++ engine.

Each script (this one, bench_onnxruntime.py, bench_pytorch.py)
is independently runnable:

    PYTHONPATH=./build/release/python python benchmarks/bench_crucible.py

Or wired together via run_all.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time


from pathlib import Path
from typing import List

import numpy as np


def percentile(sorted_data, p):
    """Standard percentile with linear interpolation (numpy-style)."""
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    k = (n - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    d0 = sorted_data[f] * (c - k)
    d1 = sorted_data[c] * (k - f)
    return d0 + d1


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO_ROOT / "models" / "mobilenet_v2.onnx"
DEFAULT_INPUT_SHAPE = (1, 3, 224, 224)
DEFAULT_RUNS = 100
DEFAULT_WARMUP = 10


# ---------------------------------------------------------------------------
# Engine loading
# ---------------------------------------------------------------------------
def _load_crucible_model(model_path: Path):
    """Load a Crucible Model, falling back to a numpy stub if the
    C++ binding isn't importable.

    Returns (callable, backend_name) where callable takes an
    ndarray and returns an ndarray of the model's output.
    """
    try:
        # Make sure the build directory is on sys.path. We try the
        # conventional locations: build/release/python (the
        # default preset output), build/debug/python, or anything
        # already on PYTHONPATH.
        build_dirs = [
            REPO_ROOT / "engine" / "build" / "release" / "python",
            REPO_ROOT / "engine" / "build" / "debug" / "python",
            REPO_ROOT / "build" / "release" / "python",
            REPO_ROOT / "build" / "debug" / "python",
        ]
        for d in build_dirs:
            if d.is_dir():
                sys.path.insert(0, str(d))
        import crucible_py  # type: ignore[import-not-found]
    except ImportError:
        # Fall back to the numpy stub. The fallback cannot run a
        # real MobileNetV2 graph — it just returns zeros of the
        # right output shape — so the latency it reports is
        # completely synthetic. We mark it with a backend name so
        # the JSON reader can see this.
        import onnx
        proto = onnx.load(str(model_path))
        out_shape = tuple(
            d.dim_value if d.dim_value > 0 else 1
            for d in proto.graph.output[0].type.tensor_type.shape.dim
        )
        # Allocate the output once and reuse it. We don't care
        # about correctness for the fallback — only that it
        # returns an ndarray of the right shape quickly.
        out_buf = np.zeros(out_shape, dtype=np.float32)
        return (lambda x: out_buf), "numpy-fallback"

    model = crucible_py.load_model(str(model_path))
    return (lambda x: np.asarray(crucible_py.run(model, x), dtype=np.float32),
            "cpp")


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------
def _bench(forward, x: np.ndarray, runs: int, warmup: int) -> List[float]:
    """Run `forward(x)` `runs` times after `warmup` warmup iterations.

    Each iteration's wall-clock time (in milliseconds) is recorded.
    Warmup is excluded from the sample so JIT-style one-off setup
    costs don't pollute the percentiles.
    """
    for _ in range(warmup):
        forward(x)
    timings: List[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        forward(x)
        timings.append((time.perf_counter() - t0) * 1000.0)
    return timings


def _summarise(timings_ms: List[float]) -> dict:
    """Compute mean / median / p95 / p99 / min / max + throughput."""
    sorted_t = sorted(timings_ms)
    n = len(sorted_t)
    return {
        "runs": n,
        "mean_ms":   statistics.fmean(sorted_t),
        "median_ms": statistics.median(sorted_t),
        # Linear-interpolation percentile (numpy-style, ~IEEE 754 pctl).
        "p95_ms": percentile(sorted_t, 95),
        "p99_ms": percentile(sorted_t, 99),
        "min_ms": sorted_t[0],
        "max_ms": sorted_t[-1],
        # Throughput = 1000 ms / mean_ms. Capped at 1e6 to keep
        # the JSON readable when a backend is broken and reports
        # mean_ms near zero.
        "throughput_inf_per_sec": min(1e6, 1000.0 / statistics.fmean(sorted_t)),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="Path to .onnx model (default: MobileNetV2)")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                        help=f"Number of timed runs (default: {DEFAULT_RUNS})")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                        help=f"Warmup iterations excluded from stats (default: {DEFAULT_WARMUP})")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for the input tensor")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional path to write JSON. Default: stdout")
    args = parser.parse_args(argv)

    if not args.model.is_file():
        print(f"error: model not found at {args.model}", file=sys.stderr)
        return 2

    forward, backend = _load_crucible_model(args.model)
    print(f"[bench_crucible] backend={backend} model={args.model.name} "
          f"runs={args.runs} warmup={args.warmup}", file=sys.stderr)

    # Fixed seed so every backend runs the same input.
    rng = np.random.default_rng(args.seed)
    x = rng.standard_normal(DEFAULT_INPUT_SHAPE).astype(np.float32)

    timings = _bench(forward, x, args.runs, args.warmup)
    summary = _summarise(timings)

    result = {
        "engine": "crucible",
        "backend": backend,
        "model": str(args.model),
        "input_shape": list(DEFAULT_INPUT_SHAPE),
        "stats": summary,
    }

    out = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out + "\n")
        print(f"[bench_crucible] wrote {args.output}", file=sys.stderr)
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())