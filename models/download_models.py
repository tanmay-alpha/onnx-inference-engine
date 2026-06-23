"""Download ONNX models from the official ONNX model zoo.

Run from the repo root:

    python models/download_models.py
    python models/download_models.py --model mobilenet_v2
    python models/download_models.py --model resnet18

The files land in this directory (`models/`) and are gitignored.
We pin to opset 7 for both models because Crucible's executor
currently targets that opset — see ENGINEERING_PLAN.md §6.

Why opset 7?
  * It is the highest version for which MobileNetV2 and ResNet18
    are published as standalone pre-converted .onnx files in the
    ONNX model zoo (newer versions require exporting from PyTorch).
  * The hand-rolled protobuf reader in `engine/src/onnx_parser.cpp`
    covers the opset-7 schema's exact attribute set, so we don't
    have to extend it for these two models.

We use `urllib.request` (stdlib only) so the script runs in any
Python 3.8+ without a `pip install` step. The official ONNX
model-zoo S3 bucket has served these files reliably for years.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

# ResNet18 v2 is published as part of the ONNX model zoo under
# `onnx-models/.../resnet18-v2-7.onnx`. The MobileNetV2 file is at
# `onnx-models/.../mobilenetv2/mobilenet-v2-7.onnx`. The hosting
# arrangement changed in 2023:
#
#   * The S3 mirror (`s3.amazonaws.com/onnx-model-zoo/...`) was
#     decommissioned; URLs return 404.
#   * The `onnx/models` GitHub repo serves the files via Git LFS.
#     `git clone` retrieves the real binary, but the raw URL serves
#     only the 133-byte LFS pointer — useless for direct download.
#   * The ONNX model zoo redirected to HuggingFace
#     (`huggingface.co/onnxmodelzoo/legacy_models`); the redirect is
#     a 302 to an Xet-backed S3 mirror that serves the real binary.
#
# So our mirror list is:
URLS = {
    "mobilenet_v2": [
        # HuggingFace (primary, current location of the model zoo)
        "https://huggingface.co/onnxmodelzoo/legacy_models/resolve/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx",
    ],
    "resnet18": [
        # HuggingFace (primary)
        "https://huggingface.co/onnxmodelzoo/legacy_models/resolve/main/validated/vision/classification/resnet/model/resnet18-v2-7.onnx",
    ],
}

# Approximate expected sizes (bytes). We don't pin a SHA-256 because
# the model-zoo re-publishes occasionally; the size is good enough
# to detect a truncated or HTML error page download.
EXPECTED_SIZES = {
    "mobilenet_v2": 13_900_000,   # ~13.3 MB
    "resnet18":     44_700_000,   # ~44.6 MB
}

DEFAULT_TOLERANCE = 0.20  # ±20% — covers the 7.x re-uploads


def _download(url: str, dst: Path) -> int:
    """Download `url` to `dst`. Returns bytes written. Raises on
    HTTP errors so the caller can fall back to the next URL."""
    print(f"  GET {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "crucible-downloader/1.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        if resp.status != 200:
            raise urllib.error.HTTPError(
                url, resp.status, "non-200 from server",
                resp.headers, None)
        with open(dst, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)  # 1 MiB
                if not chunk:
                    break
                f.write(chunk)
    return dst.stat().st_size


def download_model(name: str, dst_dir: Path) -> Path:
    if name not in URLS:
        raise ValueError(f"unknown model {name!r}; valid: {sorted(URLS)}")
    dst = dst_dir / f"{name}.onnx"
    expected = EXPECTED_SIZES[name]
    lo = int(expected * (1 - DEFAULT_TOLERANCE))
    hi = int(expected * (1 + DEFAULT_TOLERANCE))

    last_error: Exception | None = None
    for url in URLS[name]:
        try:
            size = _download(url, dst)
        except Exception as e:  # noqa: BLE001 — we genuinely want
                                # to catch every failure (timeout,
                                # 404, network unreachable) and try
                                # the next URL.
            last_error = e
            print(f"  -> failed: {e}")
            continue
        if not (lo <= size <= hi):
            print(f"  -> size {size} outside expected [{lo}, {hi}]; trying next mirror")
            try:
                dst.unlink()
            except OSError:
                pass
            continue
        sha = hashlib.sha256(dst.read_bytes()).hexdigest()
        print(f"  -> wrote {dst} ({size:,} bytes) sha256={sha[:16]}...")
        return dst
    raise RuntimeError(
        f"all mirrors failed for {name}; last error: {last_error}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", "-m", action="append", choices=sorted(URLS),
        help="model(s) to download; pass multiple times for several. "
             "default: download both")
    p.add_argument(
        "--dir", default=str(Path(__file__).resolve().parent),
        help="output directory (default: this script's directory)")
    args = p.parse_args()
    models = args.model or sorted(URLS)
    dst_dir = Path(args.dir).resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)

    failed: list[str] = []
    for name in models:
        print(f"[{name}]")
        try:
            download_model(name, dst_dir)
        except Exception as e:  # noqa: BLE001 — keep going so a single
                                # mirror outage doesn't block the rest
            print(f"  FAILED: {e}", file=sys.stderr)
            failed.append(name)

    if failed:
        print(f"\n{len(failed)} model(s) failed: {failed}", file=sys.stderr)
        return 1
    print("\nAll models downloaded successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
