"""Run all three Issue #14 benchmarks and write a single JSON.

Wiring order:
  1. Run bench_crucible.py
  2. Run bench_onnxruntime.py
  3. Run bench_pytorch.py

Each sub-script writes a single-engine JSON to a tmp file.
We then merge those into the canonical
`benchmarks/results/benchmark_results.json` along with a
cross-engine comparison block (mean_ms ratio, fastest engine,
Crucible-vs-ORL p99 ratio) so the Issue #18 web chart can render
without re-deriving anything.

The AC from Issue #14:
  "results/benchmark_results.json with mean/median/p95 for all
   three engines. Crucible within 3× of ONNX Runtime."

We surface that ratio in the summary and assert it (warning,
not failure) at the end so a regression is visible without
breaking the build.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO_ROOT / "benchmarks"
RESULTS_DIR = BENCH_DIR / "results"
DEFAULT_OUTPUT = RESULTS_DIR / "benchmark_results.json"

# The plan's "Crucible within 3× of ONNX Runtime" AC. We use it
# for the summary block; the actual constraint is enforced as a
# warning (see _emit_ac_check), not a hard failure, because
# the bench needs the C++ build to make the comparison fair.
AC_RATIO_LIMIT = 3.0


# ---------------------------------------------------------------------------
# Subprocess driver
# ---------------------------------------------------------------------------
def _run_one(script_name: str, *args: str) -> dict:
    """Run a single bench script in a subprocess and parse its JSON output.

    Why subprocess and not direct import?
      Each script is independently runnable from the command
      line, and we want to honour the same argv contract from
      the orchestrator. Subprocess also gives us a clean process
      boundary so a torch.onnx / onnxruntime init error in one
      bench doesn't poison the others.
    """
    script_path = BENCH_DIR / script_name
    with tempfile.NamedTemporaryFile(
        suffix=".json", delete=False, mode="w",
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        cmd = [sys.executable, str(script_path), "--output", str(tmp_path),
               *args]
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            print(
                f"[run_all] {script_name} failed (rc={result.returncode}):\n"
                f"  stdout: {result.stdout}\n  stderr: {result.stderr}",
                file=sys.stderr,
            )
            return {
                "engine": script_name.replace("bench_", "").replace(".py", ""),
                "error": result.stderr or result.stdout or "subprocess failed",
            }
        # Forward the script's progress to the user. Each sub-script
        # prints "[bench_xxx] ..." lines to stderr.
        sys.stderr.write(result.stderr)
        return json.loads(tmp_path.read_text())
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Cross-engine summary
# ---------------------------------------------------------------------------
def _summarise(results: List[dict]) -> dict:
    """Compute the AC-required comparison block.

    Returns:
        {
          "engines":            [...],          # engine names that ran cleanly
          "fastest_mean":       <engine>,       # engine with smallest mean_ms
          "fastest_p95":        <engine>,
          "crucible_vs_ort":    <float|None>,   # mean_ms(Crucible) / mean_ms(ORL)
          "crucible_vs_pytorch":<float|None>,   # mean_ms(Crucible) / mean_ms(PyTorch)
          "ac_within_3x":       <bool|None>,    # AC from Issue #14
        }

    Important: if Crucible ran with the numpy-fallback backend,
    its latency is dominated by zero-cost buffer reuse, not by
    real graph execution. In that case we report
    `crucible_vs_ort = None` and `ac_within_3x = null` so the
    chart doesn't render Crucible as 50,000x faster than ORT.
    """
    by_engine = {r["engine"]: r for r in results if "stats" in r}
    if not by_engine:
        return {"engines": [], "fastest_mean": None, "fastest_p95": None,
                "crucible_vs_ort": None, "crucible_vs_pytorch": None,
                "ac_within_3x": None}

    engines = list(by_engine.keys())
    fastest_mean = min(engines, key=lambda e: by_engine[e]["stats"]["mean_ms"])
    fastest_p95 = min(engines, key=lambda e: by_engine[e]["stats"]["p95_ms"])

    crucible = by_engine.get("crucible", {})
    ort = by_engine.get("onnxruntime", {})
    pytorch = by_engine.get("pytorch", {})

    # Crucible on the numpy fallback isn't a real forward pass.
    # Don't let it win the AC by cheating.
    cru_is_real = crucible.get("backend", "").startswith(("cpp", "crucible"))

    def _ratio(a: dict, b: dict, a_real: bool = True) -> float | None:
        if not a_real:
            return None
        a_ms = a.get("stats", {}).get("mean_ms")
        b_ms = b.get("stats", {}).get("mean_ms")
        if a_ms is None or b_ms is None or b_ms <= 0:
            return None
        return a_ms / b_ms

    cru_vs_ort = _ratio(crucible, ort, cru_is_real)
    cru_vs_torch = _ratio(crucible, pytorch, cru_is_real)

    ac_ok: bool | None
    if cru_vs_ort is None:
        ac_ok = None  # not measurable without the C++ build
    else:
        ac_ok = cru_vs_ort <= AC_RATIO_LIMIT

    summary = {
        "engines": engines,
        "fastest_mean": fastest_mean,
        "fastest_p95": fastest_p95,
        "crucible_vs_ort": cru_vs_ort,
        "crucible_vs_pytorch": cru_vs_torch,
        "ac_within_3x": ac_ok,
        "ac_ratio_limit": AC_RATIO_LIMIT,
    }
    if not cru_is_real:
        summary["note"] = (
            "crucible ran with backend 'numpy-fallback' because the "
            "C++ module is not built. Build the engine and rerun "
            "for an AC measurement."
        )
    return summary


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _emit_ac_check(summary: dict, output_stream=sys.stderr) -> None:
    """Surface the AC result in a human-readable form.

    The AC from Issue #14 is "Crucible within 3× of ONNX
    Runtime". This is the comparison that determines whether
    the engine is competitive, so we always print it.

    A WARNING (not an error) is emitted on regression because
    the fallback Crucible backend cannot meet the AC; we don't
    want a missing C++ build to break the benchmark pipeline.
    """
    cru = summary.get("crucible_vs_ort")
    if cru is None:
        print("[run_all] AC check skipped (Crucible not measurable; "
              "see summary.note if present)", file=output_stream)
        return
    ok = summary.get("ac_within_3x")
    label = "PASS" if ok else "WARN"
    print(f"[run_all] AC: Crucible vs ONNX Runtime = {cru:.2f}x "
          f"(limit {AC_RATIO_LIMIT}x) -> {label}", file=output_stream)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=100,
                        help="Forward runs per engine (default 100)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup iterations (default 10)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for the input tensor")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output JSON path")
    parser.add_argument("--only", choices=["crucible", "onnxruntime", "pytorch"],
                        action="append", default=None,
                        help="Limit which engines run (repeatable)")
    args = parser.parse_args(argv)

    # Ensure the results directory exists so the per-engine JSON
    # files inside it don't surprise `git status`.
    args.output.parent.mkdir(parents=True, exist_ok=True)

    engines = args.only or ["crucible", "onnxruntime", "pytorch"]
    script_for = {
        "crucible": "bench_crucible.py",
        "onnxruntime": "bench_onnxruntime.py",
        "pytorch": "bench_pytorch.py",
    }

    print(f"[run_all] engines={engines} runs={args.runs} "
          f"warmup={args.warmup} seed={args.seed}", file=sys.stderr)

    t0 = time.perf_counter()
    results: List[dict] = []
    for engine in engines:
        script = script_for[engine]
        print(f"[run_all] -> {script}", file=sys.stderr)
        result = _run_one(script,
                          "--runs", str(args.runs),
                          "--warmup", str(args.warmup),
                          "--seed", str(args.seed))
        results.append(result)
    elapsed = time.perf_counter() - t0

    summary = _summarise(results)
    _emit_ac_check(summary)

    # Resolve CPU model and today's date
    import datetime
    import platform
    today = datetime.date.today().isoformat()
    cpu_model = "Unknown CPU"
    if platform.system() == "Windows":
        try:
            out = subprocess.check_output("wmic cpu get name", shell=True).decode().strip().split("\n")
            if len(out) > 1:
                cpu_model = out[1].strip()
        except Exception:
            pass
    elif platform.system() == "Linux":
        try:
            for line in open("/proc/cpuinfo"):
                if "model name" in line:
                    cpu_model = line.split(":", 1)[1].strip()
                    break
        except Exception:
            pass
    elif platform.system() == "Darwin":
        try:
            cpu_model = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        except Exception:
            pass

    # Top-level shape matches what the Issue #18 chart and the
    # /workspace/OWNER plan reference. Add a `meta` block for
    # provenance so a downstream reader can tell when the file
    # was produced without consulting git.
    output = {
        "meta": {
            "generated_at_unix": int(time.time()),
            "wall_clock_seconds": elapsed,
            "runs": args.runs,
            "warmup": args.warmup,
            "seed": args.seed,
            "measured_on": f"{today} on {cpu_model}",
        },
        "results": results,
        "summary": summary,
    }

    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"[run_all] wrote {args.output} ({elapsed:.1f}s wall)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())