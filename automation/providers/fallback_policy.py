"""Ordered fallback policy with retry counting for provider routing."""

from automation.providers.provider_health import ProviderHealth, ProviderStatus


class FallbackPolicy:
    """Manages provider selection with fallback ordering and retry limits.

    Args:
        providers: Ordered list of provider names (first is preferred).
        max_retries: Maximum consecutive RETRIABLE results before INFRA_FAILED.
    """

    def __init__(self, providers: list[str], max_retries: int = 3) -> None:
        self._providers = list(providers)
        self._max_retries = max_retries
        self._health = ProviderHealth()
        self._retries: dict[str, int] = {}
        self._infra_failed: set[str] = set()

    def select_provider(self, preferred: str | None = None) -> str:
        if preferred is not None and preferred not in self._infra_failed and self._health.get_status(preferred) != ProviderStatus.DOWN:
            return preferred
        for p in self._providers:
            if p not in self._infra_failed and self._health.get_status(p) != ProviderStatus.DOWN:
                return p
        raise RuntimeError("all providers down")

    def on_result(self, provider: str, status: str) -> None:
        from automation.providers.router import Taxonomy

        if status == Taxonomy.SUCCESS:
            self._health.record_success(provider)
            self._retries[provider] = 0
            self._infra_failed.discard(provider)
        elif status == Taxonomy.RETRIABLE:
            self._health.record_failure(provider)
            count = self._retries.get(provider, 0) + 1
            self._retries[provider] = count
            if count >= self._max_retries:
                self._infra_failed.add(provider)
                self._retries[provider] = 0
        elif status == Taxonomy.INFRA_FAILED or status == Taxonomy.HARD_FAIL:
            self._health.record_failure(provider)
            self._retries[provider] = 0
            self._infra_failed.add(provider)
        elif status == Taxonomy.DEFERRED:
            self._health.record_failure(provider)

    def providers_available(self) -> list[str]:
        return [
            p for p in self._providers
            if p not in self._infra_failed and self._health.get_status(p) != ProviderStatus.DOWN
        ]

    def retry_count(self, provider: str) -> int:
        return self._retries.get(provider, 0)
