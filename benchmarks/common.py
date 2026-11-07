"""Shared helpers for the benchmark workers and the run_all driver.

Every runtime worker (bench_crucible.py, bench_onnxruntime.py,
bench_pytorch.py, bench_wasm.mjs) follows the same protocol so the
driver can treat them uniformly and run each one in its own process
(no shared thread pools, and two different ``crucible_py`` builds can be
measured in the same run):

    worker --model M.onnx --input in.npy --out-npy out.npy --result res.json
           --warmup W --iterations N [runtime-specific flags]

The worker writes ``res.json``:

    {"status": "ok", "load_ms": float|null, "timings_ms": [float, ...],
     "threads": int, "note": str|null}

or, when the runtime genuinely cannot execute the model,

    {"status": "unsupported", "note": "<reason>"}

and saves the output tensor of the *last* timed call to ``out.npy`` so
the driver can compute parity against ONNX Runtime. Anything else
(missing module, crash) is a hard error: non-zero exit, no result file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def worker_args(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path,
                        help=".npy file holding the float32 input tensor")
    parser.add_argument("--out-npy", required=True, type=Path,
                        help="where to save the output of the last timed call")
    parser.add_argument("--result", required=True, type=Path,
                        help="where to write the worker JSON result")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    return parser


def time_calls(fn: Callable[[], np.ndarray], warmup: int,
               iterations: int, clock: Callable[[], float]) -> tuple[List[float], np.ndarray]:
    """Run ``fn`` warmup times untimed, then ``iterations`` times timing
    each call individually. Returns (per-call ms, last output)."""
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    out = None
    for _ in range(warmup):
        out = fn()
    timings: List[float] = []
    for _ in range(iterations):
        t0 = clock()
        out = fn()
        timings.append((clock() - t0) * 1000.0)
    return timings, np.asarray(out, dtype=np.float32)


def write_ok(path: Path, *, load_ms: Optional[float], timings_ms: List[float],
             threads: int, note: Optional[str] = None) -> None:
    path.write_text(json.dumps({
        "status": "ok", "load_ms": load_ms, "timings_ms": timings_ms,
        "threads": threads, "note": note,
    }))


def write_unsupported(path: Path, note: str, threads: Optional[int] = None) -> None:
    path.write_text(json.dumps({"status": "unsupported", "note": note,
                                "threads": threads}))


def summarise(timings_ms: List[float], batch: int) -> dict:
    """mean/median/p95/min/std (population) + throughput = batch / median."""
    arr = np.asarray(timings_ms, dtype=np.float64)
    median = float(np.median(arr))
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": median,
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "std_ms": float(arr.std()),
        "throughput_ips": float(batch / (median / 1000.0)) if median > 0 else None,
    }
