"""Native engine access for the server (crucible_py, the pybind11 module).

There is no fallback executor: if ``crucible_py`` cannot be imported the
server still starts (uploads are analysed with the ``onnx`` package), but
every inference request fails with :class:`EngineUnavailable` (HTTP 503).

Timeouts and concurrency
------------------------
A native ``crucible_py.run`` call cannot be cancelled once it has started.
Runs therefore execute on a dedicated thread pool with
``MAX_CONCURRENT_INFERENCES`` workers, and a slot semaphore of the same size
is acquired *before* submission and released by the worker only when the
native call actually returns. Consequences:

* at most ``MAX_CONCURRENT_INFERENCES`` native runs are ever in flight;
* when every slot is busy the request is rejected immediately with
  :class:`EngineBusy` (HTTP 503 + Retry-After) instead of queueing;
* a request whose run exceeds ``INFERENCE_TIMEOUT_SEC`` gets
  :class:`EngineTimeout` (HTTP 504), but the run keeps executing in the
  background and keeps its slot until it finishes.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from server.config import get_settings

_log = logging.getLogger("crucible.engine")

BACKEND = "crucible-native"

try:
    import crucible_py  # type: ignore[import-not-found]
except ImportError as _exc:  # pragma: no cover - depends on the build
    crucible_py = None  # type: ignore[assignment]
    _log.warning("crucible_py not importable (%s); /infer will return 503.", _exc)


class EngineUnavailable(RuntimeError):
    """crucible_py is not importable in this process."""


class EngineBusy(RuntimeError):
    """All native inference slots are occupied."""


class EngineTimeout(RuntimeError):
    """The native run did not finish within the configured timeout."""


class EngineRejected(ValueError):
    """The engine refused the model or input (unsupported op, bad shape...)."""


def native_module() -> Optional[Any]:
    """Return the crucible_py module, or None when it is not available."""
    return crucible_py


def native_available() -> bool:
    return crucible_py is not None


_settings = get_settings()
MAX_CONCURRENT = _settings.MAX_CONCURRENT_INFERENCES
_POOL = ThreadPoolExecutor(max_workers=MAX_CONCURRENT, thread_name_prefix="crucible-infer")
_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT)


def _run_blocking(module: Any, model_path: str, input_name: str,
                  arr: np.ndarray) -> Tuple[np.ndarray, float]:
    """Load + run on a worker thread. Returns (output, engine-only ms)."""
    try:
        model = module.load_model(model_path)
        t0 = time.perf_counter()
        out = module.run(model, {input_name: arr})
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return np.asarray(out, dtype=np.float32), elapsed_ms
    finally:
        _SLOTS.release()


async def run_native(model_path: Path, input_name: str, arr: np.ndarray,
                     timeout_sec: float) -> Tuple[np.ndarray, float]:
    """Run one inference through crucible_py.

    ``inference_time_ms`` covers only ``crucible_py.run`` (model loading,
    JSON decoding and queueing are excluded).

    Raises EngineUnavailable, EngineBusy, EngineTimeout, EngineRejected.
    """
    module = crucible_py
    if module is None:
        raise EngineUnavailable("native engine (crucible_py) is not available on this server")
    if not _SLOTS.acquire(blocking=False):
        raise EngineBusy(f"all {MAX_CONCURRENT} inference slots are busy")
    try:
        fut = _POOL.submit(_run_blocking, module, str(model_path), input_name, arr)
    except BaseException:
        _SLOTS.release()
        raise
    try:
        return await asyncio.wait_for(asyncio.wrap_future(fut), timeout=timeout_sec)
    except asyncio.TimeoutError as exc:
        raise EngineTimeout(
            f"native inference exceeded {timeout_sec:g}s; the run cannot be cancelled "
            "and continues in the background"
        ) from exc
    except (ValueError, RuntimeError) as exc:
        # crucible_py: ValueError = bad input (missing/shape/dtype),
        # RuntimeError = unsupported op/attribute or malformed model.
        raise EngineRejected(str(exc)) from exc


def supported_ops() -> List[str]:
    """Operator list from the native engine, or the static mirror when absent."""
    if crucible_py is not None:
        return sorted(crucible_py.supported_ops())
    from server.validator import SUPPORTED_OPS
    return sorted(SUPPORTED_OPS)
