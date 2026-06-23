#!/usr/bin/env python3
"""Merge Debug and Release JSON benchmark outputs into one file.

Usage:
    python benchmarks/results/merge.py \\
        --debug    benchmarks/results/matmul_debug_vs_release_debug.json \\
        --release  benchmarks/results/matmul_debug_vs_release_release.json \\
        --output   benchmarks/results/matmul_debug_vs_release.json

Reads the two native Google Benchmark JSON outputs, extracts the
relevant fields from each benchmark entry, and writes a flat list
of records into the output file. The flat layout makes it trivial
to diff two rows by `name` and `build` (the AC for Issue #11 is
"Release build >= 3x faster than Debug for 1024x1024 MatMul", which
is one row comparison).

Why not just concatenate the two files?
   Google's native format wraps every record in `{"benchmarks": [...],
   "context": {...}}`. Concatenating would yield invalid JSON, and
   the duplicate `context` key would lose information anyway.
   Flattening keeps the file human-readable AND parseable.
"""
import argparse
import json
import sys


def parse_google_benchmark_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def extract_records(data: dict, build_label: str) -> list:
    """Pull the fields we care about out of one Google Benchmark
    JSON file. Each benchmark becomes one record in the flat output."""
    out = []
    for b in data.get("benchmarks", []):
        out.append({
            "name":         b["name"],
            "build":        build_label,
            "iterations":   b.get("iterations"),
            "time_unit":    b["time_unit"],
            "real_time":    b["real_time"],
            "cpu_time":     b["cpu_time"],
            "items_per_second":     b.get("items_per_second"),
            "bytes_per_second":     b.get("bytes_per_second"),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--debug",   required=True, help="Debug build JSON")
    ap.add_argument("--release", required=True, help="Release build JSON")
    ap.add_argument("--output",  required=True, help="Merged JSON output")
    args = ap.parse_args()

    debug_data = parse_google_benchmark_json(args.debug)
    release_data = parse_google_benchmark_json(args.release)

    merged = (extract_records(debug_data, "debug")
              + extract_records(release_data, "release"))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
        f.write("\n")

    print(f"wrote {len(merged)} records -> {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())