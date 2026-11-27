import threading
import time
import pytest

from automation.providers.provider_health import ProviderStatus, ProviderHealth
from automation.providers.fallback_policy import FallbackPolicy
from automation.providers.router import Taxonomy, classify_response, ProviderRouter


# =============================================================================
# ProviderHealth Tests
# =============================================================================

class TestProviderStatus:
    def test_enum_values(self):
        assert ProviderStatus.HEALTHY.value == "healthy"
        assert ProviderStatus.DEGRADED.value == "degraded"
        assert ProviderStatus.DOWN.value == "down"


class TestProviderHealth:

    def test_starts_with_no_data(self):
        h = ProviderHealth()
        assert h.get_status("nonexistent") == ProviderStatus.HEALTHY

    def test_healthy_after_one_success(self):
        h = ProviderHealth()
        h.record_success("p1")
        assert h.get_status("p1") == ProviderStatus.HEALTHY

    def test_degraded_after_one_failure(self):
        h = ProviderHealth()
        h.record_failure("p1", 500)
        assert h.get_status("p1") == ProviderStatus.DEGRADED

    def test_down_after_three_consecutive_failures(self):
        h = ProviderHealth()
        h.record_failure("p1", 500)
        h.record_failure("p1", 502)
        h.record_failure("p1", 503)
        assert h.get_status("p1") == ProviderStatus.DOWN

    def test_recovers_after_success_following_failures(self):
        h = ProviderHealth()
        h.record_failure("p1", 500)
        h.record_failure("p1", 502)
        assert h.get_status("p1") == ProviderStatus.DEGRADED
        h.record_success("p1")
        assert h.get_status("p1") == ProviderStatus.HEALTHY

    def test_get_stats_shape(self):
        h = ProviderHealth()
        h.record_success("p1")
        h.record_failure("p1", 500)
        stats = h.get_stats("p1")
        assert "total_calls" in stats
        assert "successes" in stats
        assert "failures" in stats
        assert "consecutive_failures" in stats
        assert "status" in stats
        assert stats["total_calls"] == 2
        assert stats["successes"] == 1
        assert stats["failures"] == 1
        assert isinstance(stats["status"], ProviderStatus)

    def test_reset_clears_provider(self):
        h = ProviderHealth()
        h.record_failure("p1", 500)
        h.record_failure("p1", 502)
        h.record_failure("p1", 503)
        assert h.get_status("p1") == ProviderStatus.DOWN
        h.reset("p1")
        assert h.get_status("p1") == ProviderStatus.HEALTHY
        stats = h.get_stats("p1")
        assert stats["total_calls"] == 0

    def test_multiple_providers_independent(self):
        h = ProviderHealth()
        h.record_success("p1")
        h.record_success("p2")
        h.record_failure("p2", 500)
        assert h.get_status("p1") == ProviderStatus.HEALTHY
        assert h.get_status("p2") == ProviderStatus.DEGRADED

    def test_thread_safety(self):
        h = ProviderHealth()
        errors = []

        def worker(name):
            try:
                for _ in range(100):
                    h.record_success(name)
                    h.record_failure(name, 500)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        for i in range(10):
            name = f"t{i}"
            stats = h.get_stats(name)
            assert stats["total_calls"] == 200
            assert stats["successes"] == 100
            assert stats["failures"] == 100


# =============================================================================
# FallbackPolicy Tests
# =============================================================================

class TestFallbackPolicy:

    def test_select_preferred_when_healthy(self):
        policy = FallbackPolicy(["p1", "p2"])
        policy.on_result("p1", "SUCCESS")
        assert policy.select_provider("p1") == "p1"

    def test_fallback_to_next_when_preferred_down(self):
        policy = FallbackPolicy(["p1", "p2", "p3"])
        policy.on_result("p1", "HARD_FAIL")
        policy.on_result("p1", "HARD_FAIL")
        policy.on_result("p1", "HARD_FAIL")
        result = policy.select_provider("p1")
        assert result == "p2"

    def test_raises_runtime_error_when_all_down(self):
        policy = FallbackPolicy(["p1", "p2"])
        for p in ["p1", "p2"]:
            policy.on_result(p, "HARD_FAIL")
            policy.on_result(p, "HARD_FAIL")
            policy.on_result(p, "HARD_FAIL")
        with pytest.raises(RuntimeError, match="all providers down"):
            policy.select_provider("p1")

    def test_retriable_increments_retry_count(self):
        policy = FallbackPolicy(["p1", "p2"])
        assert policy.retry_count("p1") == 0
        policy.on_result("p1", "RETRIABLE")
        assert policy.retry_count("p1") == 1
        policy.on_result("p1", "RETRIABLE")
        assert policy.retry_count("p1") == 2

    def test_providers_available_returns_healthy_only(self):
        policy = FallbackPolicy(["p1", "p2", "p3"])
        policy.on_result("p1", "SUCCESS")
        policy.on_result("p2", "HARD_FAIL")
        policy.on_result("p2", "HARD_FAIL")
        policy.on_result("p2", "HARD_FAIL")
        av = policy.providers_available()
        assert "p1" in av
        assert "p2" not in av
        assert "p3" in av

    def test_after_max_retries_marked_infra_failed(self):
        policy = FallbackPolicy(["p1", "p2"])
        policy.on_result("p1", "RETRIABLE")
        policy.on_result("p1", "RETRIABLE")
        policy.on_result("p1", "RETRIABLE")
        assert policy.retry_count("p1") == 0  # reset after max
        assert "p1" not in policy.providers_available()
        # Knock out p2 as well to verify both are unreachable
        policy.on_result("p2", "HARD_FAIL")
        policy.on_result("p2", "HARD_FAIL")
        policy.on_result("p2", "HARD_FAIL")
        with pytest.raises(RuntimeError, match="all providers down"):
            policy.select_provider("p1")


# =============================================================================
# Router Tests
# =============================================================================

class TestTaxonomy:
    def test_constants(self):
        assert Taxonomy.SUCCESS == "SUCCESS"
        assert Taxonomy.RETRIABLE == "RETRIABLE"
        assert Taxonomy.DEFERRED == "DEFERRED"
        assert Taxonomy.INFRA_FAILED == "INFRA_FAILED"
        assert Taxonomy.HARD_FAIL == "HARD_FAIL"


class TestClassifyResponse:

    def test_200_is_success(self):
        assert classify_response(200) == Taxonomy.SUCCESS

    def test_429_is_retriable(self):
        assert classify_response(429) == Taxonomy.RETRIABLE

    def test_5xx_is_retriable(self):
        assert classify_response(500) == Taxonomy.RETRIABLE
        assert classify_response(502) == Taxonomy.RETRIABLE
        assert classify_response(503) == Taxonomy.RETRIABLE

    def test_401_403_infra_failed(self):
        assert classify_response(401) == Taxonomy.INFRA_FAILED
        assert classify_response(403) == Taxonomy.INFRA_FAILED

    def test_400_404_hard_fail(self):
        assert classify_response(400) == Taxonomy.HARD_FAIL
        assert classify_response(404) == Taxonomy.HARD_FAIL

    def test_402_is_deferred(self):
        assert classify_response(402) == Taxonomy.DEFERRED

    def test_quota_message_in_body_is_deferred(self):
        body = {"error": {"message": "quota exceeded"}}
        assert classify_response(403, body) == Taxonomy.DEFERRED

    def test_quota_message_no_error_key(self):
        body = {"message": "quota limit reached"}
        assert classify_response(403, body) == Taxonomy.DEFERRED


class TestProviderRouter:

    def test_call_returns_result_and_success(self):
        policy = FallbackPolicy(["p1"])
        health = ProviderHealth()
        router = ProviderRouter(policy, health)

        def my_fn(x):
            return x * 2

        result, status = router.call("p1", my_fn, 5)
        assert result == 10
        assert status == Taxonomy.SUCCESS

    def test_call_returns_none_and_infra_failed_on_exception(self):
        policy = FallbackPolicy(["p1"])
        health = ProviderHealth()
        router = ProviderRouter(policy, health)

        def broken_fn():
            raise ConnectionError("network error")

        result, status = router.call("p1", broken_fn)
        assert result is None
        assert status == Taxonomy.INFRA_FAILED

    def test_call_records_failure_on_non_success(self):
        policy = FallbackPolicy(["p1"])
        health = ProviderHealth()
        router = ProviderRouter(policy, health)

        def fail_fn():
            raise ValueError("bad")

        router.call("p1", fail_fn)
        stats = health.get_stats("p1")
        assert stats["failures"] >= 1
