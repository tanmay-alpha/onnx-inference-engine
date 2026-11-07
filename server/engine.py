"""Native engine access for the server (crucible_py, the pybind11 module).

There is no fallback executor: if ``crucible_py`` cannot be imported the
server still starts (uploads are analysed with the ``onnx`` package), but
every inference request fails with :class:`EngineUnavailable` (HTTP 503).

Deadlines, memory limits and concurrency
----------------------------------------
Every ``crucible_py.run`` call gets ``timeout_ms`` (from
``INFERENCE_TIMEOUT_SEC``), ``max_tensor_elements`` and ``max_total_elements``.
The engine checks the deadline between nodes and inside the outer loops of
Conv/pooling/MatMul, so a run that overruns stops shortly after the deadline
(best-effort granularity: one node or one outer-loop iteration) and frees its
worker. ``crucible_py.ResourceLimitError`` mentioning the deadline becomes
:class:`EngineTimeout` (HTTP 504); any other resource limit becomes
:class:`EngineRejected` (HTTP 422).

Runs execute on a dedicated thread pool with ``MAX_CONCURRENT_INFERENCES``
workers, and a slot semaphore of the same size is acquired *before*
submission and released by the worker only when the native call returns:

* at most ``MAX_CONCURRENT_INFERENCES`` native runs are ever in flight;
* when every slot is busy the request is rejected immediately with
  :class:`EngineBusy` (HTTP 503 + Retry-After) instead of queueing;
* ``asyncio.wait_for`` with the same timeout is kept as a backstop: the
  request gets :class:`EngineTimeout` even if one node overruns the deadline,
  and the slot is released when the engine reaches its next deadline check.

Loaded models are kept in an LRU cache (``MODEL_CACHE_SIZE`` entries, keyed
by resolved path + mtime + size); ``DELETE /models/{id}`` evicts its entry.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
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

# Loaded crucible_py models, LRU-bounded. The key includes mtime/size so a
# replaced file is reloaded; Model objects are immutable during run() (the
# engine allocates per-call buffers), so one instance may serve concurrent runs.
MODEL_CACHE_SIZE = _settings.MODEL_CACHE_SIZE
# Uploaded models are untrusted: a tiny graph can request huge tensors. Every
# tensor the engine allocates is capped at MAX_TENSOR_ELEMENTS, and the live
# intermediates of one run together at MAX_TOTAL_ELEMENTS.
MAX_TENSOR_ELEMENTS = _settings.MAX_TENSOR_ELEMENTS
MAX_TOTAL_ELEMENTS = _settings.MAX_TOTAL_ELEMENTS
_CacheKey = Tuple[int, str, int, int]
_cache: "OrderedDict[_CacheKey, Any]" = OrderedDict()
_cache_lock = threading.Lock()


def _load_cached(module: Any, model_path: Path) -> Any:
    resolved = model_path.resolve()
    st = resolved.stat()
    key: _CacheKey = (id(module), str(resolved), st.st_mtime_ns, st.st_size)
    with _cache_lock:
        model = _cache.get(key)
        if model is not None:
            _cache.move_to_end(key)
            return model
    model = module.load_model(str(resolved))  # outside the lock: loads can be slow
    if MODEL_CACHE_SIZE > 0:
        with _cache_lock:
            _cache[key] = model
            _cache.move_to_end(key)
            while len(_cache) > MODEL_CACHE_SIZE:
                _cache.popitem(last=False)
    return model


def evict_model(model_path: Path) -> None:
    """Drop every cached model loaded from ``model_path`` (any mtime/size)."""
    resolved = str(model_path.resolve())
    with _cache_lock:
        for key in [k for k in _cache if k[1] == resolved]:
            del _cache[key]


def cached_paths() -> List[str]:
    with _cache_lock:
        return [k[1] for k in _cache]


def _run_blocking(module: Any, model_path: Path, input_name: str,
                  arr: np.ndarray, timeout_ms: int) -> Tuple[np.ndarray, float]:
    """Load (cached) + run on a worker thread. Returns (output, engine-only ms)."""
    try:
        model = _load_cached(module, model_path)
        t0 = time.perf_counter()
        out = module.run(model, {input_name: arr}, max_tensor_elements=MAX_TENSOR_ELEMENTS,
                         max_total_elements=MAX_TOTAL_ELEMENTS, timeout_ms=timeout_ms)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return np.asarray(out, dtype=np.float32), elapsed_ms
    finally:
        _SLOTS.release()


async def run_native(model_path: Path, input_name: str, arr: np.ndarray,
                     timeout_sec: float) -> Tuple[np.ndarray, float]:
    """Run one inference through crucible_py with a ``timeout_sec`` deadline.

    ``inference_time_ms`` covers only ``crucible_py.run`` (model loading,
    JSON decoding and queueing are excluded).

    Raises EngineUnavailable, EngineBusy, EngineTimeout, EngineRejected.
    """
    module = crucible_py
    if module is None:
        raise EngineUnavailable("native engine (crucible_py) is not available on this server")
    timeout_ms = max(1, int(timeout_sec * 1000))
    if not _SLOTS.acquire(blocking=False):
        raise EngineBusy(f"all {MAX_CONCURRENT} inference slots are busy")
    try:
        fut = _POOL.submit(_run_blocking, module, Path(model_path), input_name, arr, timeout_ms)
    except BaseException:
        _SLOTS.release()
        raise
    limit_error = getattr(module, "ResourceLimitError", ())
    try:
        return await asyncio.wait_for(asyncio.wrap_future(fut), timeout=timeout_sec)
    except asyncio.TimeoutError as exc:
        # Backstop: one node overran the deadline; the engine stops at its next check.
        raise EngineTimeout(f"native inference exceeded {timeout_sec:g}s and was stopped") from exc
    except limit_error as exc:  # type: ignore[misc]
        if "deadline" in str(exc):
            raise EngineTimeout(f"native inference exceeded {timeout_sec:g}s and was stopped") from exc
        raise EngineRejected(str(exc)) from exc
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
