"""Render a results JSON (from run_all.py) as the Markdown tables used in
docs/BENCHMARKS.md. Prints to stdout; nothing is computed here that is not
in the JSON except the plain latency ratios against ONNX Runtime.

    python benchmarks/render_table.py web/public/benchmarks/results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def fmt(v, kind: str = "ms") -> str:
    """kind: 'ms' -> 4 significant digits, 'ips' -> integer, 'diff' -> scientific."""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "NO"
    if isinstance(v, float):
        if kind == "ips":
            return f"{v:,.1f}" if v < 1000 else f"{v:,.0f}"
        if kind == "diff":
            return "0" if v == 0 else f"{v:.2e}"
        return f"{v:.4g}"
    return str(v)


def main() -> int:
    doc = json.loads(Path(sys.argv[1]).read_text())
    rows = doc["results"]
    out = ["| Model | Input | Runtime | Threads | Median ms | Mean ms | p95 ms | Min ms "
           "| Std ms | Throughput (inf/s) | Load ms | Max abs diff vs ORT | Top-1 match "
           "| median ÷ ORT-1t median | median ÷ ORT median |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|---:|---:|"]
    medians = {(r["model"], tuple(r["input_shape"]), r["runtime"]): r["median_ms"]
               for r in rows if r["status"] == "ok"}
    notes = []
    for r in rows:
        key = (r["model"], tuple(r["input_shape"]))
        shape = "×".join(map(str, r["input_shape"]))
        if r["status"] != "ok":
            notes.append(f"- {r['model']} {shape} / {r['runtime']}: unsupported — {r['note']}")
            out.append(f"| {r['model']} | {shape} | {r['runtime']} | {fmt(r['threads'])} "
                       "| unsupported | | | | | | | | | | |")
            continue
        ort1 = medians.get(key + ("onnxruntime-cpu-1t",))
        ort = medians.get(key + ("onnxruntime-cpu",))
        ratio1 = f"{r['median_ms'] / ort1:.2f}" if ort1 else "—"
        ratio = f"{r['median_ms'] / ort:.2f}" if ort else "—"
        out.append(
            f"| {r['model']} | {shape} | {r['runtime']} | {r['threads']} "
            f"| {fmt(r['median_ms'])} | {fmt(r['mean_ms'])} | {fmt(r['p95_ms'])} "
            f"| {fmt(r['min_ms'])} | {fmt(r['std_ms'])} | {fmt(r['throughput_ips'], 'ips')} "
            f"| {fmt(r['load_ms'])} | {fmt(r['max_abs_diff_vs_ort'], 'diff')} "
            f"| {fmt(r['top1_match'])} | {ratio1} | {ratio} |")
    print("\n".join(out))
    if notes:
        print("\nUnsupported rows:\n")
        print("\n".join(notes))
    print("\nArtifacts:\n")
    print("| Artifact | Bytes |\n|---|---:|")
    for a in doc["artifacts"]:
        print(f"| {a['name']} | {a['bytes']:,} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
