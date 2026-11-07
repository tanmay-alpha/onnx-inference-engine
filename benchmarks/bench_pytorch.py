"""PyTorch (CPU, eager) benchmark worker.

PyTorch is only benchmarked on a *genuinely equivalent* network: the
exact ONNX file is converted to a ``torch.nn.Module`` with onnx2torch
and its output is checked against ONNX Runtime's output for the same
input (``--reference``). onnx2torch has no converters for some old
opset versions (e.g. BatchNormalization-7 in the opset-7 CNNs), so if
the direct conversion fails the graph is first upgraded with
``onnx.version_converter`` to ``--upgrade-opset`` -- same nodes, same
weights -- and the upgrade is noted in the row. If conversion still
fails or the outputs diverge beyond ``--tolerance`` (or the arg-max
class differs), the row is recorded as ``unsupported`` with the reason
-- a different network (e.g. a torchvision model with other weights)
is never substituted.

``torch.set_num_threads(--threads)`` is applied before anything runs and
the effective ``torch.get_num_threads()`` is recorded. ``load_ms`` is the
onnx2torch conversion time (including any opset upgrade). Timing is
``time.perf_counter`` around each forward call under
``torch.inference_mode()``, including the numpy -> tensor -> numpy
conversions. See common.py for the protocol.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from common import time_calls, worker_args, write_ok, write_unsupported


def main() -> int:
    parser = worker_args(__doc__.splitlines()[0])
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--reference", required=True, type=Path,
                        help=".npy output of onnxruntime-cpu for the same input")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--upgrade-opset", type=int, default=13)
    args = parser.parse_args()

    try:
        import onnx
        import onnx.version_converter
        import torch
        from onnx2torch import convert
    except ImportError as exc:
        write_unsupported(args.result,
                          f"torch/onnx2torch not installed in the benchmark environment ({exc})",
                          threads=args.threads)
        return 0

    torch.set_num_threads(args.threads)
    threads = torch.get_num_threads()

    x_np = np.load(args.input)
    ref = np.load(args.reference)

    t0 = time.perf_counter()
    proto = onnx.load(str(args.model))
    note = f"ONNX converted with onnx2torch; torch.set_num_threads({args.threads})"
    try:
        module = convert(proto).eval()
    except Exception as direct_exc:  # onnx2torch raises NotImplementedError and friends
        opset = next((o.version for o in proto.opset_import if o.domain in ("", "ai.onnx")),
                     None)
        try:
            upgraded = onnx.version_converter.convert_version(proto, args.upgrade_opset)
            module = convert(upgraded).eval()
        except Exception as exc:
            write_unsupported(args.result,
                              f"onnx2torch could not convert the model ({direct_exc}); "
                              f"after upgrading opset {opset} -> {args.upgrade_opset}: {exc}",
                              threads=threads)
            return 0
        note = (f"ONNX opset {opset} upgraded to {args.upgrade_opset} with "
                f"onnx.version_converter (same weights), then converted with onnx2torch; "
                f"torch.set_num_threads({args.threads})")
    load_ms = (time.perf_counter() - t0) * 1000.0

    def forward() -> np.ndarray:
        with torch.inference_mode():
            return module(torch.from_numpy(x_np)).numpy()

    check = forward()
    if check.shape != ref.shape:
        write_unsupported(args.result,
                          f"onnx2torch output shape {list(check.shape)} != ORT {list(ref.shape)}",
                          threads=threads)
        return 0
    diff = float(np.max(np.abs(check.astype(np.float32) - ref)))
    top1_ok = ref.shape[-1] < 2 or bool(
        np.array_equal(check.reshape(-1, ref.shape[-1]).argmax(-1),
                       ref.reshape(-1, ref.shape[-1]).argmax(-1)))
    if diff > args.tolerance or not top1_ok:
        write_unsupported(args.result,
                          f"onnx2torch conversion is not equivalent: max_abs_diff vs ORT "
                          f"{diff:.3g} (tolerance {args.tolerance:g}), top1_match={top1_ok}",
                          threads=threads)
        return 0

    timings, out = time_calls(forward, args.warmup, args.iterations, time.perf_counter)
    np.save(args.out_npy, out)
    write_ok(args.result, load_ms=load_ms, timings_ms=timings, threads=threads, note=note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
