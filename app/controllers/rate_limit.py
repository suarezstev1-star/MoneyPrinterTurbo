"""Lightweight in-process request throttling for the HTTP API.

State lives in this process only: it bounds floods against a single API
instance, it is not a distributed limiter. All limits come from ``config.app``
and ``0`` disables the corresponding limit.
"""

import threading
import time
from collections import deque

from app.config import config

_DEFAULT_REQUESTS_PER_MINUTE = 120
_DEFAULT_AUTH_FAILURES = 10
_DEFAULT_AUTH_WINDOW_SECONDS = 300
_MAX_TRACKED_CLIENTS = 10000


def _setting(name: str, default: int) -> int:
    value = config.app.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return max(int(value), 0)


class SlidingWindowLimiter:
    """Allow at most ``limit`` events per ``window`` seconds for each client."""

    def __init__(self) -> None:
        self._events: dict[str, deque] = {}
        self._lock = threading.Lock()

    def _prune(self, bucket: deque, now: float, window: float) -> None:
        while bucket and now - bucket[0] >= window:
            bucket.popleft()

    def _evict_idle(self, now: float, window: float) -> None:
        if len(self._events) < _MAX_TRACKED_CLIENTS:
            return
        for client in list(self._events):
            bucket = self._events[client]
            self._prune(bucket, now, window)
            if not bucket:
                del self._events[client]
        # Still full of active clients: drop the oldest entries so memory
        # stays bounded under a many-source flood.
        while len(self._events) >= _MAX_TRACKED_CLIENTS:
            self._events.pop(next(iter(self._events)))

    def hit(self, client: str, limit: int, window: float) -> bool:
        """Record an event; return False when the client is over the limit."""
        if limit <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            self._evict_idle(now, window)
            bucket = self._events.setdefault(client, deque())
            self._prune(bucket, now, window)
            if len(bucket) >= limit:
                return False
            bucket.append(now)
            return True

    def is_blocked(self, client: str, limit: int, window: float) -> bool:
        """Return True when ``limit`` events were already recorded in the window."""
        if limit <= 0:
            return False
        now = time.monotonic()
        with self._lock:
            bucket = self._events.get(client)
            if bucket is None:
                return False
            self._prune(bucket, now, window)
            return len(bucket) >= limit

    def record(self, client: str, window: float) -> None:
        now = time.monotonic()
        with self._lock:
            self._evict_idle(now, window)
            bucket = self._events.setdefault(client, deque())
            self._prune(bucket, now, window)
            bucket.append(now)

    def reset(self) -> None:
        with self._lock:
            self._events.clear()


request_limiter = SlidingWindowLimiter()
auth_failure_limiter = SlidingWindowLimiter()


def requests_per_minute() -> int:
    return _setting("api_rate_limit_per_minute", _DEFAULT_REQUESTS_PER_MINUTE)


def allow_request(client: str) -> bool:
    return request_limiter.hit(client, requests_per_minute(), 60)


def auth_failure_settings() -> tuple[int, int]:
    return (
        _setting("api_auth_failure_limit", _DEFAULT_AUTH_FAILURES),
        _setting("api_auth_failure_window_seconds", _DEFAULT_AUTH_WINDOW_SECONDS),
    )


def is_auth_blocked(client: str) -> bool:
    limit, window = auth_failure_settings()
    return auth_failure_limiter.is_blocked(client, limit, window)


def record_auth_failure(client: str) -> None:
    limit, window = auth_failure_settings()
    if limit > 0:
        auth_failure_limiter.record(client, window)


def reset_all() -> None:
    request_limiter.reset()
    auth_failure_limiter.reset()
