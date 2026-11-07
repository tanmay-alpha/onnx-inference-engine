"""ONNX Runtime (CPUExecutionProvider) benchmark worker.

``--threads default``: untouched ``SessionOptions`` -- ORT picks its own
intra-op pool size (one thread per physical core). The driver passes the
physical core count it detected via ``--physical-cores`` so the result
records the thread count ORT is expected to use.

``--threads 1``: ``intra_op_num_threads = inter_op_num_threads = 1`` and
``ORT_SEQUENTIAL`` execution -- the like-for-like comparison with the
single-threaded Crucible engine.

Graph optimisations are left at ORT's default (ORT_ENABLE_ALL).
``load_ms`` is the ``InferenceSession`` construction time, which includes
those optimisations. Timing is ``time.perf_counter`` around each
``session.run`` call. See common.py for the worker protocol.
"""
from __future__ import annotations

import time

import numpy as np
import onnxruntime as ort

from common import time_calls, worker_args, write_ok


def main() -> int:
    parser = worker_args(__doc__.splitlines()[0])
    parser.add_argument("--threads", choices=["default", "1"], required=True)
    parser.add_argument("--physical-cores", type=int, required=True)
    args = parser.parse_args()

    opts = ort.SessionOptions()
    if args.threads == "1":
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        threads = 1
        note = "intra_op=inter_op=1, ORT_SEQUENTIAL"
    else:
        threads = args.physical_cores
        note = ("ORT default SessionOptions (intra_op_num_threads=0 -> one thread "
                "per physical core)")

    x = np.load(args.input)
    t0 = time.perf_counter()
    sess = ort.InferenceSession(str(args.model), sess_options=opts,
                                providers=["CPUExecutionProvider"])
    load_ms = (time.perf_counter() - t0) * 1000.0

    feeds = {sess.get_inputs()[0].name: x}
    timings, out = time_calls(lambda: sess.run(None, feeds)[0],
                              args.warmup, args.iterations, time.perf_counter)
    np.save(args.out_npy, out)
    write_ok(args.result, load_ms=load_ms, timings_ms=timings, threads=threads,
             note=note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
