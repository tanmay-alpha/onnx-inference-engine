"""Download the two ONNX model zoo files committed in models/.

Run from the repo root:

    python models/download_models.py
    python models/download_models.py --model mobilenet_v2
    python models/download_models.py --model resnet18

`models/mobilenet_v2.onnx` and `models/resnet18.onnx` are committed unmodified
(force-added; `models/*.onnx` is otherwise gitignored). This script fetches
the same files again and checks their SHA-256 against the committed copies.

Both are the model zoo's opset-7 exports (`mobilenetv2-7.onnx`,
`resnet18-v2-7.onnx`), the versions the zoo publishes as standalone files. The
engine is not limited to opset 7; see docs/ARCHITECTURE.md for the parser and
the opset-dependent semantics it handles.

Stdlib only (`urllib.request`), so no `pip install` is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

# The ONNX model zoo moved to HuggingFace (`onnxmodelzoo/legacy_models`); the
# old S3 bucket is gone and the `onnx/models` GitHub raw URLs serve only Git LFS
# pointers. HuggingFace answers with a 302 to a mirror that serves the file.
_ZOO = "https://huggingface.co/onnxmodelzoo/legacy_models/resolve/main/validated/vision/classification"
URLS = {
    "mobilenet_v2": f"{_ZOO}/mobilenet/model/mobilenetv2-7.onnx",
    "resnet18": f"{_ZOO}/resnet/model/resnet18-v2-7.onnx",
}

# SHA-256 of the committed files, which match HuggingFace's X-Linked-ETag.
SHA256 = {
    "mobilenet_v2": "c1c513582d56afceff8516c73804e484c81c6a830712ab6d682253f4a3cd042f",
    "resnet18": "3b4d048057854022232aa20bf68262e31718ab7119a17f920b8a2dc1433f824b",
}


def _download(url: str, dst: Path) -> int:
    """Download `url` to `dst`. Returns bytes written; raises on HTTP errors."""
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
    tmp = dst.with_suffix(".onnx.part")
    size = _download(URLS[name], tmp)
    sha = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if sha != SHA256[name]:
        tmp.unlink()
        raise RuntimeError(f"sha256 mismatch for {name}: got {sha}, expected {SHA256[name]}")
    tmp.replace(dst)
    print(f"  -> wrote {dst} ({size:,} bytes, sha256 verified)")
    return dst


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
