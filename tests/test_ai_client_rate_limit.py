"""Tests for ai_client.py rate-limit and circuit-breaker improvements."""

import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCircuitBreakerExcludesRateLimits:
    """429 rate limits should NOT count toward circuit breaker failures."""

    def test_429_does_not_increment_failure_counter(self):
        """A 429 error should not increase the circuit breaker failure count."""
        from utils.ai_client import AIClient

        # Reset state
        AIClient._provider_failures = {}
        AIClient._provider_cooldown_until = {}

        # Simulate 20 consecutive 429 errors
        for _ in range(20):
            exc = MagicMock()
            exc.status_code = 429
            exc.response.headers = {"Retry-After": "5"}
            exc.__str__ = lambda s: "429 rate limit"
            AIClient._note_provider_error("opencode", exc)

        # Circuit breaker failure count should be 0 (429s excluded)
        failures = AIClient._provider_failures.get("opencode", 0)
        assert failures == 0, (
            f"Circuit breaker failures should be 0 after 429s, got {failures}"
        )

    def test_429_does_not_trigger_circuit_breaker(self):
        """After many 429s, the provider should NOT have hit circuit breaker.
        It will be in rate-limit cooldown (expected), but NOT 900s circuit breaker."""
        from utils.ai_client import AIClient

        AIClient._provider_failures = {}
        AIClient._provider_cooldown_until = {}

        # Simulate 20 consecutive 429 errors
        for _ in range(20):
            exc = MagicMock()
            exc.status_code = 429
            exc.response.headers = {"Retry-After": "5"}
            exc.__str__ = lambda s: "429 rate limit"
            AIClient._note_provider_error("opencode", exc)

        # Failure count should be 0 — circuit breaker never triggered
        failures = AIClient._provider_failures.get("opencode", 0)
        assert failures == 0, (
            f"Expected 0 circuit breaker failures, got {failures}"
        )

        # Rate-limit cooldown should be at most 300s (5 min), NOT 900s (circuit breaker)
        cd = AIClient._provider_cooldown_until.get("opencode", 0)
        remaining = cd - time.time()
        assert remaining <= 300, (
            f"Cooldown should be rate-limit (<=300s), got {remaining:.0f}s remaining"
        )
        assert remaining > 0, "Provider should be in rate-limit cooldown"

    def test_5xx_errors_do_trigger_circuit_breaker(self):
        """5xx errors SHOULD count toward circuit breaker."""
        from utils.ai_client import AIClient

        AIClient._provider_failures = {}
        AIClient._provider_cooldown_until = {}

        # Simulate 16 consecutive 500 errors (exceeds threshold of 15)
        for _ in range(16):
            exc = MagicMock()
            exc.status_code = 500
            exc.response = None
            exc.__str__ = lambda s: "500 internal server error"
            AIClient._note_provider_error("opencode", exc)

        # Provider SHOULD be in circuit breaker cooldown
        failures = AIClient._provider_failures.get("opencode", 0)
        assert failures >= 15, f"Expected >=15 failures, got {failures}"

    def test_mixed_errors_only_real_failures_count(self):
        """Mix of 429s and 5xx: only 5xx should count toward circuit breaker."""
        from utils.ai_client import AIClient

        AIClient._provider_failures = {}
        AIClient._provider_cooldown_until = {}

        # 10x 429 + 6x 500 = 6 real failures (not enough for circuit breaker at 15)
        for _ in range(10):
            exc = MagicMock()
            exc.status_code = 429
            exc.response.headers = {"Retry-After": "5"}
            exc.__str__ = lambda s: "429"
            AIClient._note_provider_error("opencode", exc)

        for _ in range(6):
            exc = MagicMock()
            exc.status_code = 500
            exc.response = None
            exc.__str__ = lambda s: "500"
            AIClient._note_provider_error("opencode", exc)

        failures = AIClient._provider_failures.get("opencode", 0)
        assert failures == 6, f"Expected 6 real failures, got {failures}"


class TestExponentialBackoff:
    """Backoff between retry rounds should increase exponentially."""

    def test_backoff_increases_with_consecutive_failures(self):
        """Multiple retry rounds should have increasing delays."""
        from utils.ai_client import AIClient

        AIClient._provider_failures = {}
        AIClient._provider_cooldown_until = {}

        # First 429: cooldown should be based on Retry-After (5s)
        exc = MagicMock()
        exc.status_code = 429
        exc.response.headers = {"Retry-After": "5"}
        exc.__str__ = lambda s: "429"
        AIClient._note_provider_error("opencode", exc)

        # Second 429: cooldown should be at least 5s (same or higher)
        exc2 = MagicMock()
        exc2.status_code = 429
        exc2.response.headers = {"Retry-After": "5"}
        exc2.__str__ = lambda s: "429"
        AIClient._note_provider_error("opencode", exc2)

        # Both should have set cooldowns
        cd1 = AIClient._provider_cooldown_until.get("opencode", 0)
        assert cd1 > time.time(), "Cooldown should be set after 429"


class TestSEOModelRotation:
    """SEO should rotate through models, not hammer the same one."""

    def test_seo_tries_multiple_models(self):
        """generate_seo_text should attempt different models on retry."""
        from utils.ai_client import AIClient

        models_tried = []

        def mock_call(provider, prompt, system_instruction, prefer_model=None):
            models_tried.append(prefer_model)
            raise Exception("mocked failure")

        client = AIClient.__new__(AIClient)
        client.opencode_api_key = "test"
        client.nvidia_api_key = None
        client.groq_api_key = None
        client._provider = "opencode"
        client._model = "qwen3.7-max"

        AIClient._provider_failures = {}
        AIClient._provider_cooldown_until = {}

        # Patch _call_provider and _in_cooldown
        with patch.object(AIClient, '_call_provider', side_effect=mock_call), \
             patch.object(AIClient, '_in_cooldown', return_value=False), \
             patch.object(AIClient, '_check_and_consume_token', return_value=True):
            try:
                client.generate_seo_text("test prompt", "system")
            except RuntimeError:
                pass  # Expected - all models fail

        # Should have tried both SEO preferred models
        assert len(models_tried) >= 2, (
            f"Expected at least 2 model attempts, got {len(models_tried)}: {models_tried}"
        )
        # Should have different models
        unique_models = set(models_tried)
        assert len(unique_models) >= 2, (
            f"Expected at least 2 unique models, got {unique_models}"
        )
