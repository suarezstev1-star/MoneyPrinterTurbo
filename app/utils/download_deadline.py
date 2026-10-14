"""Total wall-clock deadline for streamed downloads.

``requests`` timeouts apply per socket read, so a server that trickles one byte
every few seconds can keep a download alive forever. Streaming loops use this
helper to enforce an overall limit and abort so callers can clean up.
"""

from __future__ import annotations

import math
import time

from app.config import config

DEFAULT_DOWNLOAD_DEADLINE_SECONDS = 600.0
DEADLINE_CONFIG_KEY = "download_total_timeout_seconds"


class DownloadDeadlineExceeded(TimeoutError):
    """The overall wall-clock budget for a download was used up."""


def configured_deadline_seconds() -> float:
    try:
        value = float(
            config.app.get(DEADLINE_CONFIG_KEY, DEFAULT_DOWNLOAD_DEADLINE_SECONDS)
        )
    except (TypeError, ValueError):
        return DEFAULT_DOWNLOAD_DEADLINE_SECONDS
    if not math.isfinite(value) or value <= 0:
        return DEFAULT_DOWNLOAD_DEADLINE_SECONDS
    return value


class DownloadDeadline:
    """Start the clock on construction; call ``check`` inside the read loop."""

    def __init__(self, seconds: float | None = None):
        self.seconds = (
            configured_deadline_seconds() if seconds is None else float(seconds)
        )
        self._expires_at = time.monotonic() + self.seconds

    def expired(self) -> bool:
        return time.monotonic() > self._expires_at

    def check(self, what: str = "download") -> None:
        if self.expired():
            raise DownloadDeadlineExceeded(
                f"{what} exceeded the total time limit of {self.seconds:.0f}s"
            )
