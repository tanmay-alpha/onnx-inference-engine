"""Run every runtime on every model and write one results JSON.

    python benchmarks/run_all.py --out web/public/benchmarks/results.json \
        [--models resnet18 mobilenet_v2 fraud] [--warmup 10 --iterations 50]

Runtimes (each in its own subprocess, one after another):
  onnxruntime-cpu       ORT CPU EP, default threads (parity reference)
  onnxruntime-cpu-1t    ORT CPU EP, intra=inter=1, sequential
  crucible-native       crucible_py from --crucible-dir (portable build)
  crucible-native-avx2  crucible_py from --crucible-avx2-dir, only if given
                        (-DCRUCIBLE_MARCH_NATIVE=ON, host-specific)
  pytorch-cpu           onnx2torch conversion of the same ONNX, verified vs ORT
  crucible-wasm         web/public/wasm via Node (initSync on the .wasm bytes)

Inputs are float32 uniform[0, 1) drawn from numpy.random.default_rng(0)
for every (model, shape) pair, so every runtime sees identical bytes.
A missing crucible_py module is a hard error. Runtimes that genuinely
cannot run a model produce a row with status "unsupported" and a note.
See docs/BENCHMARKS.md.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from common import REPO_ROOT, summarise

BENCH_DIR = Path(__file__).resolve().parent
SCHEMA = 1
SEED = 0

MODELS: Dict[str, dict] = {
    "resnet18": {"name": "resnet18", "path": "models/resnet18.onnx",
                 "shapes": [[1, 3, 224, 224]]},
    "mobilenet_v2": {"name": "mobilenet_v2", "path": "models/mobilenet_v2.onnx",
                     "shapes": [[1, 3, 224, 224]]},
    "fraud": {"name": "fraud_detector", "path": "models/fraud/fraud_detector.onnx",
              "shapes": [[1, 7], [64, 7]]},
}

STAT_KEYS = ("mean_ms", "median_ms", "p95_ms", "min_ms", "std_ms", "throughput_ips")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
def _cpuinfo() -> List[Dict[str, str]]:
    path = Path("/proc/cpuinfo")
    if not path.exists():
        return []
    blocks: List[Dict[str, str]] = []
    for block in path.read_text().split("\n\n"):
        entry = {}
        for line in block.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                entry[k.strip()] = v.strip()
        if entry:
            blocks.append(entry)
    return blocks


def cpu_name() -> str:
    for entry in _cpuinfo():
        if "model name" in entry:
            return entry["model name"]
    return platform.processor() or platform.machine()


def physical_cores() -> int:
    ids = {(e.get("physical id"), e.get("core id")) for e in _cpuinfo()
           if "core id" in e}
    return len(ids) if ids else (os.cpu_count() or 1)


def os_name() -> str:
    release = Path("/etc/os-release")
    if release.exists():
        for line in release.read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                pretty = line.split("=", 1)[1].strip('"')
                return f"{pretty} (kernel {platform.release()})"
    return platform.platform()


def git_commit() -> str:
    def git(*a: str) -> str:
        return subprocess.run(["git", "-c", "safe.directory=*", *a], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    try:
        sha = git("rev-parse", "--short", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def _cmake_cache(module_dir: Path) -> Dict[str, str]:
    """Read CMakeCache.txt of the build tree that produced crucible_py
    (module lives in <build>/python)."""
    cache = module_dir.parent / "CMakeCache.txt"
    out: Dict[str, str] = {}
    if cache.exists():
        for line in cache.read_text(errors="replace").splitlines():
            if line and not line.startswith(("#", "//")) and "=" in line and ":" in line:
                key_type, value = line.split("=", 1)
                out[key_type.split(":", 1)[0]] = value
    return out


def compiler(module_dir: Path) -> str:
    cxx = _cmake_cache(module_dir).get("CMAKE_CXX_COMPILER")
    if not cxx:
        return f"unknown (no CMakeCache.txt in {module_dir.parent})"
    try:
        first = subprocess.run([cxx, "--version"], capture_output=True, text=True,
                               check=True).stdout.splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return cxx
    return first


def build_flags(module_dir: Path) -> str:
    c = _cmake_cache(module_dir)
    if not c:
        return f"unknown (no CMakeCache.txt in {module_dir.parent})"
    build_type = c.get("CMAKE_BUILD_TYPE", "?")
    flags = c.get(f"CMAKE_CXX_FLAGS_{build_type.upper()}", "")
    return (f"{build_type} '{flags}' CRUCIBLE_MARCH_NATIVE={c.get('CRUCIBLE_MARCH_NATIVE', '?')}"
            f" ({module_dir})")


def dist_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def node_version(node: Optional[str]) -> Optional[str]:
    if not node:
        return None
    return subprocess.run([node, "--version"], capture_output=True, text=True,
                          check=True).stdout.strip()


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------
def _package_dir(name: str) -> Optional[Path]:
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0])


def artifacts(crucible_dirs: Dict[str, Path], libcrucible: Optional[Path],
              wasm_dir: Path) -> List[dict]:
    found: List[dict] = []

    def add(label: str, path: Path) -> None:
        found.append({"name": f"{label}: {path.name}", "bytes": path.stat().st_size})

    for label, d in crucible_dirs.items():
        for p in sorted(d.glob("crucible_py*")):
            if p.suffix in (".so", ".pyd"):
                add(label, p)
    if libcrucible is not None:
        add("crucible C ABI shared library", libcrucible)
    wasm = wasm_dir / "crucible_wasm_bg.wasm"
    if wasm.exists():
        add("crucible-wasm", wasm)
    ort_dir = _package_dir("onnxruntime")
    if ort_dir is not None:
        for p in sorted((ort_dir / "capi").glob("onnxruntime_pybind11_state*")):
            add("onnxruntime native", p)
    torch_dir = _package_dir("torch")
    if torch_dir is not None:
        for p in sorted((torch_dir / "lib").glob("libtorch_cpu*")):
            add("pytorch native", p)
    return found


# ---------------------------------------------------------------------------
# Running workers
# ---------------------------------------------------------------------------
def run_worker(cmd: List[str], result_path: Path) -> dict:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"benchmark worker failed ({proc.returncode}): {' '.join(cmd)}\n"
                           f"{proc.stdout}\n{proc.stderr}")
    return json.loads(result_path.read_text())


def make_row(model: str, shape: List[int], runtime: str, res: dict,
             out: Optional[np.ndarray], ref: Optional[np.ndarray],
             is_reference: bool) -> dict:
    row = {"model": model, "input_shape": shape, "runtime": runtime,
           "threads": res.get("threads"), "status": res["status"]}
    note = res.get("note")
    if res["status"] != "ok":
        row.update({k: None for k in STAT_KEYS})
        row.update({"load_ms": None, "max_abs_diff_vs_ort": None, "top1_match": None,
                    "note": note})
        return row

    stats = summarise(res["timings_ms"], batch=shape[0])
    row.update({k: (round(v, 6) if v is not None else None) for k, v in stats.items()})
    row["load_ms"] = round(res["load_ms"], 3) if res.get("load_ms") is not None else None
    if is_reference or ref is None or out is None:
        row["max_abs_diff_vs_ort"] = None
        row["top1_match"] = None
        if is_reference:
            note = "parity reference" + (f"; {note}" if note else "")
    else:
        if out.shape != ref.shape:
            raise RuntimeError(f"{runtime} {model} output shape {out.shape} != ORT {ref.shape}")
        row["max_abs_diff_vs_ort"] = float(np.max(np.abs(out - ref)))
        classes = ref.shape[-1]
        row["top1_match"] = (bool(np.array_equal(out.reshape(-1, classes).argmax(-1),
                                                 ref.reshape(-1, classes).argmax(-1)))
                             if classes > 1 else None)
    row["note"] = note
    return row


def bench_model(key: str, shape: List[int], args, tmp: Path, node: Optional[str],
                cores: int) -> List[dict]:
    spec = MODELS[key]
    model_path = REPO_ROOT / spec["path"]
    if not model_path.exists():
        raise SystemExit(f"model file missing: {model_path} (see models/README.md)")
    tag = f"{key}_{'x'.join(map(str, shape))}"
    x = np.random.default_rng(SEED).uniform(0.0, 1.0, size=shape).astype(np.float32)
    x_path = tmp / f"{tag}_in.npy"
    np.save(x_path, x)

    common = ["--model", str(model_path), "--input", str(x_path),
              "--warmup", str(args.warmup), "--iterations", str(args.iterations)]
    py = sys.executable
    rows: List[dict] = []
    ref: Optional[np.ndarray] = None

    def py_worker(runtime: str, script: str, extra: List[str]) -> None:
        nonlocal ref
        log(f"  {runtime:22s} {spec['name']} {shape}")
        out_path, res_path = tmp / f"{tag}_{runtime}.npy", tmp / f"{tag}_{runtime}.json"
        res = run_worker([py, str(BENCH_DIR / script), *common, "--out-npy", str(out_path),
                          "--result", str(res_path), *extra], res_path)
        out = np.load(out_path) if res["status"] == "ok" else None
        is_ref = runtime == "onnxruntime-cpu"
        if is_ref:
            if out is None:
                raise RuntimeError(f"onnxruntime could not run {model_path}")
            ref = out
        rows.append(make_row(spec["name"], shape, runtime, res, out, ref, is_ref))

    py_worker("onnxruntime-cpu", "bench_onnxruntime.py",
              ["--threads", "default", "--physical-cores", str(cores)])
    py_worker("onnxruntime-cpu-1t", "bench_onnxruntime.py",
              ["--threads", "1", "--physical-cores", str(cores)])
    py_worker("crucible-native", "bench_crucible.py",
              ["--module-dir", str(args.crucible_dir)])
    if args.crucible_avx2_dir is not None:
        py_worker("crucible-native-avx2", "bench_crucible.py",
                  ["--module-dir", str(args.crucible_avx2_dir)])
    ref_path = tmp / f"{tag}_onnxruntime-cpu.npy"
    py_worker("pytorch-cpu", "bench_pytorch.py",
              ["--threads", str(args.torch_threads), "--reference", str(ref_path)])

    # crucible-wasm via Node
    runtime = "crucible-wasm"
    log(f"  {runtime:22s} {spec['name']} {shape}")
    wasm_js = args.wasm_dir / "crucible_wasm.js"
    if node is None or not wasm_js.exists():
        why = "node not found on PATH" if node is None else f"{wasm_js} not built"
        rows.append(make_row(spec["name"], shape, runtime,
                             {"status": "unsupported", "note": why, "threads": 1},
                             None, ref, False))
        return rows
    in_bin, out_bin = tmp / f"{tag}_in.f32", tmp / f"{tag}_wasm_out.f32"
    res_path = tmp / f"{tag}_wasm.json"
    x.tofile(in_bin)
    res = run_worker([node, str(BENCH_DIR / "bench_wasm.mjs"),
                      "--wasm-dir", str(args.wasm_dir), "--model", str(model_path),
                      "--input-bin", str(in_bin), "--shape", ",".join(map(str, shape)),
                      "--out-bin", str(out_bin), "--result", str(res_path),
                      "--warmup", str(args.warmup), "--iterations", str(args.iterations)],
                     res_path)
    out = None
    if res["status"] == "ok":
        flat = np.fromfile(out_bin, dtype=np.float32)
        if flat.size != ref.size:
            raise RuntimeError(f"crucible-wasm returned {flat.size} values, ORT {ref.size}")
        out = flat.reshape(ref.shape)
    rows.append(make_row(spec["name"], shape, runtime, res, out, ref, False))
    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--crucible-dir", type=Path,
                        default=Path(os.environ["CRUCIBLE_PY_DIR"])
                        if os.environ.get("CRUCIBLE_PY_DIR") else None,
                        help="directory containing crucible_py (default: $CRUCIBLE_PY_DIR)")
    parser.add_argument("--crucible-avx2-dir", type=Path, default=None,
                        help="optional crucible_py built with -DCRUCIBLE_MARCH_NATIVE=ON")
    parser.add_argument("--libcrucible", type=Path, default=None,
                        help="optional libcrucible shared library, reported in artifacts")
    parser.add_argument("--wasm-dir", type=Path, default=REPO_ROOT / "web" / "public" / "wasm")
    parser.add_argument("--torch-threads", type=int, default=None,
                        help="torch.set_num_threads value (default: physical core count)")
    parser.add_argument("--note", action="append", default=[],
                        help="extra methodology note (repeatable), e.g. host description")
    args = parser.parse_args(argv)

    if args.crucible_dir is None:
        parser.error("--crucible-dir (or CRUCIBLE_PY_DIR) is required")
    for label, path in (("--crucible-dir", args.crucible_dir),
                        ("--crucible-avx2-dir", args.crucible_avx2_dir),
                        ("--libcrucible", args.libcrucible)):
        if path is not None and not path.exists():
            parser.error(f"{label} {path} does not exist")
    if args.iterations < 1 or args.warmup < 0:
        parser.error("--iterations must be >= 1 and --warmup >= 0")

    cores = physical_cores()
    if args.torch_threads is None:
        args.torch_threads = cores
    node = shutil.which("node")

    results: List[dict] = []
    with tempfile.TemporaryDirectory(prefix="crucible-bench-") as td:
        for key in args.models:
            for shape in MODELS[key]["shapes"]:
                log(f"[{key} {shape}]")
                results.extend(bench_model(key, shape, args, Path(td), node, cores))

    build = build_flags(args.crucible_dir)
    if args.crucible_avx2_dir is not None:
        build += f"; crucible-native-avx2: {build_flags(args.crucible_avx2_dir)}"
    doc = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "environment": {
            "cpu": cpu_name(),
            "cores": os.cpu_count() or 1,
            "os": os_name(),
            "python": platform.python_version(),
            "compiler": compiler(args.crucible_dir),
            "onnxruntime": dist_version("onnxruntime") or "not installed",
            "torch": dist_version("torch"),
            "crucible_build": build,
        },
        "methodology": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "timer": "time.perf_counter per call (Python workers); "
                     "process.hrtime.bigint per call (Node WASM worker)",
            "notes": [
                f"Inputs: float32 uniform[0,1) from numpy.random.default_rng({SEED}); "
                "identical bytes for every runtime.",
                "Each runtime runs in its own subprocess, sequentially; nothing else "
                "is benchmarked concurrently.",
                "Timed region is one inference call from the host language, including "
                "input/output copies across the binding boundary.",
                "throughput_ips = batch / median latency.",
                f"cores = logical CPUs visible to the process; {cores} physical cores "
                "detected (used as ORT default and torch thread count).",
                "max_abs_diff_vs_ort / top1_match compare the last timed output with "
                "onnxruntime-cpu on the same input; top1_match is null for "
                "single-column outputs.",
                "crucible-native is single-threaded; compare it with "
                "onnxruntime-cpu-1t for a like-for-like number.",
                "load_ms: crucible_py.load_model / ORT InferenceSession construction "
                "(incl. graph optimisation) / onnx2torch conversion / WASM "
                "compile+instantiate.",
                f"Node: {node_version(node) or 'not found'}.",
                *args.note,
            ],
        },
        "results": results,
        "artifacts": artifacts(
            {"crucible_py (portable)": args.crucible_dir,
             **({"crucible_py (march=native)": args.crucible_avx2_dir}
                if args.crucible_avx2_dir is not None else {})},
            args.libcrucible, args.wasm_dir),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n")
    log(f"wrote {args.out} ({len(results)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
