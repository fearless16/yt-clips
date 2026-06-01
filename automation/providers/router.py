"""Provider response classification and routing with health recording."""

from collections.abc import Callable
from typing import Any

from automation.providers.fallback_policy import FallbackPolicy
from automation.providers.provider_health import ProviderHealth


class Taxonomy:
    SUCCESS = "SUCCESS"
    RETRIABLE = "RETRIABLE"
    DEFERRED = "DEFERRED"
    INFRA_FAILED = "INFRA_FAILED"
    HARD_FAIL = "HARD_FAIL"


_QUOTA_KEYWORDS = [
    "quota", "rate limit", "insufficient", "exceeded",
    "billing", "payment required",
]


def _has_quota_message(body: dict | None) -> bool:
    if body is None:
        return False
    body_str = str(body).lower()
    for kw in _QUOTA_KEYWORDS:
        if kw in body_str:
            return True
    return False


def classify_response(status_code: int, response_body: dict | None = None) -> str:
    if status_code == 200:
        return Taxonomy.SUCCESS
    if status_code == 429:
        return Taxonomy.RETRIABLE
    if status_code >= 500:
        return Taxonomy.RETRIABLE
    if status_code == 402:
        return Taxonomy.DEFERRED
    if status_code in (401, 403):
        if _has_quota_message(response_body):
            return Taxonomy.DEFERRED
        return Taxonomy.INFRA_FAILED
    if status_code in (400, 404):
        return Taxonomy.HARD_FAIL
    return Taxonomy.INFRA_FAILED


class ProviderRouter:
    """Wraps a provider call with response classification and health tracking."""

    def __init__(self, fallback_policy: FallbackPolicy, health_tracker: ProviderHealth) -> None:
        self._policy = fallback_policy
        self._health = health_tracker

    def call(self, provider: str, fn: Callable, *args: Any, **kwargs: Any) -> tuple[Any, str]:
        try:
            result = fn(*args, **kwargs)
        except Exception:
            status = Taxonomy.INFRA_FAILED
            self._health.record_failure(provider)
            self._policy.on_result(provider, status)
            return None, status

        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
            body, status_code = result
            status = classify_response(status_code, body)
        else:
            status = Taxonomy.SUCCESS

        if status == Taxonomy.SUCCESS:
            self._health.record_success(provider)
        else:
            self._health.record_failure(provider)
        self._policy.on_result(provider, status)
        return result, status
