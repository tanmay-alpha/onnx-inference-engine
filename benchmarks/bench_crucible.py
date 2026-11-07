"""Crucible native engine benchmark worker (pybind11 ``crucible_py``).

Imports ``crucible_py`` from ``--module-dir`` only. A missing or
unimportable module is a hard error: there is no fallback, because a
fallback would produce numbers that are not Crucible's.

The engine is single-threaded (no OpenMP / thread pool), so ``threads``
is always 1. Timing is ``time.perf_counter`` around each
``crucible_py.run`` call, which includes the pybind11 ndarray -> tensor
copy and the output copy back to numpy. ``crucible_py.run`` releases
the GIL.

See common.py for the worker protocol.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from common import time_calls, worker_args, write_ok, write_unsupported


def _import_crucible(module_dir: Path):
    if not module_dir.is_dir():
        raise SystemExit(f"crucible_py module dir does not exist: {module_dir}")
    sys.path.insert(0, str(module_dir.resolve()))
    import crucible_py  # noqa: E402  (hard error if missing)

    loaded_from = Path(crucible_py.__file__).resolve().parent
    if loaded_from != module_dir.resolve():
        raise SystemExit(
            f"crucible_py was imported from {loaded_from}, expected {module_dir}")
    return crucible_py


def main() -> int:
    parser = worker_args(__doc__.splitlines()[0])
    parser.add_argument("--module-dir", required=True, type=Path,
                        help="directory containing the crucible_py extension module")
    args = parser.parse_args()

    crucible_py = _import_crucible(args.module_dir)
    x = np.load(args.input)

    t0 = time.perf_counter()
    model = crucible_py.load_model(str(args.model))
    load_ms = (time.perf_counter() - t0) * 1000.0

    problems = crucible_py.check_model(model)
    if problems:
        write_unsupported(args.result, "; ".join(problems), threads=1)
        return 0

    input_name = crucible_py.get_model_info(model)["input_names"][0]
    feeds = {input_name: x}
    try:
        timings, out = time_calls(lambda: crucible_py.run(model, feeds),
                                  args.warmup, args.iterations, time.perf_counter)
    except (ValueError, RuntimeError) as exc:
        # ValueError: shape/dtype rejected; RuntimeError: unsupported op/attr.
        write_unsupported(args.result, f"crucible_py.run rejected the input: {exc}",
                          threads=1)
        return 0

    np.save(args.out_npy, out)
    write_ok(args.result, load_ms=load_ms, timings_ms=timings, threads=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
