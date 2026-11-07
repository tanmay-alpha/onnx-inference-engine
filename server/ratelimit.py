"""In-process sliding-window rate limiting.

State lives in this process only: with several instances (or workers) each
one enforces its own limits, so the effective limit is multiplied by the
instance count. Good enough for a single small instance; use a shared store
(e.g. Redis) when scaling out.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Deque, Dict, Hashable, Sequence, Tuple

from starlette.requests import Request

from server.config import get_settings

_SWEEP_EVERY = 1024


class SlidingWindowLimiter:
    """Counts events per key over the last ``window_sec`` seconds."""

    def __init__(self, window_sec: float) -> None:
        self.window = window_sec
        self._events: Dict[Hashable, Deque[float]] = {}
        self._lock = threading.Lock()
        self._calls = 0

    def acquire(self, rules: Sequence[Tuple[Hashable, int]], cost: int = 1) -> float:
        """Record ``cost`` events on every key if all ``(key, limit)`` rules allow it.

        Returns 0.0 when recorded, else the seconds until the request would
        fit (nothing is recorded then, so rejected attempts don't extend the wait).
        """
        now = time.monotonic()
        cutoff = now - self.window
        with self._lock:
            self._calls += 1
            if self._calls % _SWEEP_EVERY == 0:
                self._sweep(cutoff)
            retry_after = 0.0
            for key, limit in rules:
                q = self._events.get(key)
                if q is None:
                    if cost > limit:
                        retry_after = max(retry_after, self.window)
                    continue
                while q and q[0] <= cutoff:
                    q.popleft()
                excess = len(q) + cost - limit
                if excess > 0:
                    wait = self.window if excess > len(q) else q[excess - 1] + self.window - now
                    retry_after = max(retry_after, wait)
            if retry_after > 0:
                return retry_after
            for key, _ in rules:
                self._events.setdefault(key, deque()).extend([now] * cost)
            return 0.0

    def _sweep(self, cutoff: float) -> None:
        for key in [k for k, q in self._events.items() if not q or q[-1] <= cutoff]:
            del self._events[key]

    def reset(self) -> None:
        with self._lock:
            self._events.clear()


def retry_after_header(seconds: float) -> Dict[str, str]:
    return {"Retry-After": str(max(1, math.ceil(seconds)))}


def client_ip(request: Request) -> str:
    """The caller's IP; the first X-Forwarded-For entry only if TRUST_PROXY_HEADERS."""
    if get_settings().TRUST_PROXY_HEADERS:
        forwarded = request.headers.get("x-forwarded-for", "")
        first = forwarded.split(",", 1)[0].strip()
        if first:
            return first
    return request.client.host if request.client else "unknown"


_settings = get_settings()
# /auth/login and /auth/register attempts.
auth_limiter = SlidingWindowLimiter(_settings.AUTH_RATE_WINDOW_SEC)
# Per-API-key requests per minute (ApiKey.rate_limit) on /infer and /inference/batch.
api_key_limiter = SlidingWindowLimiter(60.0)


def reset_all() -> None:
    auth_limiter.reset()
    api_key_limiter.reset()
