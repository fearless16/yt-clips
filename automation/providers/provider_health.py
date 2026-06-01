"""Provider health tracking with thread-safe success/failure recording."""

import enum
import threading
from typing import Any


class ProviderStatus(enum.Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"


class ProviderHealth:
    """Thread-safe health tracker for external providers.

    Provider is HEALTHY if >= 2 consecutive successes, DEGRADED if exactly 1
    failure in the last 3, DOWN if >= 3 consecutive failures.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._data: dict[str, dict[str, Any]] = {}

    def _ensure(self, provider_name: str) -> dict[str, Any]:
        if provider_name not in self._data:
            self._data[provider_name] = {
                "total_calls": 0,
                "successes": 0,
                "failures": 0,
                "consecutive_failures": 0,
                "recent": [],
            }
        return self._data[provider_name]

    def record_success(self, provider_name: str) -> None:
        with self._lock:
            d = self._ensure(provider_name)
            d["total_calls"] += 1
            d["successes"] += 1
            d["consecutive_failures"] = 0
            d["recent"].append("success")
            if len(d["recent"]) > 3:
                d["recent"].pop(0)

    def record_failure(self, provider_name: str, status_code: int | None = None) -> None:
        with self._lock:
            d = self._ensure(provider_name)
            d["total_calls"] += 1
            d["failures"] += 1
            d["consecutive_failures"] += 1
            d["recent"].append("failure")
            if len(d["recent"]) > 3:
                d["recent"].pop(0)

    def get_status(self, provider_name: str) -> ProviderStatus:
        with self._lock:
            d = self._ensure(provider_name)
            if d["consecutive_failures"] >= 3:
                return ProviderStatus.DOWN
            if d["consecutive_failures"] >= 1:
                return ProviderStatus.DEGRADED
            return ProviderStatus.HEALTHY

    def get_stats(self, provider_name: str) -> dict[str, Any]:
        with self._lock:
            d = self._ensure(provider_name)
            status = self.get_status(provider_name)
            return {
                "total_calls": d["total_calls"],
                "successes": d["successes"],
                "failures": d["failures"],
                "consecutive_failures": d["consecutive_failures"],
                "status": status,
            }

    def reset(self, provider_name: str) -> None:
        with self._lock:
            self._data.pop(provider_name, None)
