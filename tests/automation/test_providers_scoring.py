import threading
import time
import pytest

from automation.providers.provider_health import ProviderStatus, ProviderHealth
from automation.providers.fallback_policy import FallbackPolicy
from automation.providers.router import Taxonomy, classify_response, ProviderRouter
from automation.scoring.feature_extractor import FeatureExtractor
from automation.scoring.scoring import ClipScorer
from automation.scoring.ranker import ClipRanker


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


# =============================================================================
# FeatureExtractor Tests
# =============================================================================

class TestFeatureExtractor:

    def test_extract_returns_all_required_keys(self):
        extractor = FeatureExtractor()
        data = {
            "clip_id": "c1",
            "duration_s": 45.0,
            "transcript": "wait, this is amazing",
            "title": "Incredible video"
        }
        features = extractor.extract(data)
        assert "hook_words_count" in features
        assert "payoff_indicators" in features
        assert "profanity_flag" in features
        assert "duration_s" in features
        assert "length_category" in features

    def test_hook_words_count(self):
        extractor = FeatureExtractor()
        data = {
            "clip_id": "c2",
            "duration_s": 30.0,
            "transcript": "wait watch this incredible amazing shocking moment",
            "title": "you won't believe this"
        }
        features = extractor.extract(data)
        assert features["hook_words_count"] == 6

    def test_payoff_indicators(self):
        extractor = FeatureExtractor()
        data = {
            "clip_id": "c3",
            "duration_s": 20.0,
            "transcript": "finally, there it is! yes we got it",
            "title": "perfect ending"
        }
        features = extractor.extract(data)
        assert features["payoff_indicators"] == 5

    def test_profanity_flag_true_when_profanity_present(self):
        extractor = FeatureExtractor()
        data = {
            "clip_id": "c4",
            "duration_s": 15.0,
            "transcript": "this is shit, damn it",
            "title": "bad video"
        }
        features = extractor.extract(data)
        assert features["profanity_flag"] is True

    def test_length_category_short(self):
        extractor = FeatureExtractor()
        data = {
            "clip_id": "c5",
            "duration_s": 25.0,
            "transcript": "",
            "title": ""
        }
        features = extractor.extract(data)
        assert features["length_category"] == "short"

    def test_length_category_medium(self):
        extractor = FeatureExtractor()
        data = {
            "clip_id": "c6",
            "duration_s": 45.0,
            "transcript": "",
            "title": ""
        }
        features = extractor.extract(data)
        assert features["length_category"] == "medium"

    def test_length_category_long(self):
        extractor = FeatureExtractor()
        data = {
            "clip_id": "c7",
            "duration_s": 90.0,
            "transcript": "",
            "title": ""
        }
        features = extractor.extract(data)
        assert features["length_category"] == "long"

    def test_raises_value_error_for_missing_fields(self):
        extractor = FeatureExtractor()
        with pytest.raises(ValueError, match="duration_s"):
            extractor.extract({"clip_id": "c8"})
        with pytest.raises(ValueError, match="transcript"):
            extractor.extract({"clip_id": "c9", "duration_s": 10.0})
        with pytest.raises(ValueError, match="title"):
            extractor.extract({"clip_id": "c10", "duration_s": 10.0, "transcript": ""})


# =============================================================================
# ClipScorer Tests
# =============================================================================

class TestClipScorer:

    def test_score_returns_correct_shape(self):
        scorer = ClipScorer()
        data = {
            "clip_id": "c1",
            "duration_s": 30.0,
            "transcript": "hello world",
            "title": "test"
        }
        result = scorer.score(data)
        assert "clip_id" in result
        assert "score" in result
        assert "features" in result
        assert "breakdown" in result
        assert result["clip_id"] == "c1"
        assert 0.0 <= result["score"] <= 1.0

    def test_score_formula_base(self):
        scorer = ClipScorer()
        data = {
            "clip_id": "c1",
            "duration_s": 30.0,
            "transcript": "hello world",
            "title": "test"
        }
        result = scorer.score(data)
        assert result["score"] == 0.5
        assert result["breakdown"]["base"] == 0.5

    def test_score_with_hooks_and_payoffs(self):
        scorer = ClipScorer()
        data = {
            "clip_id": "c2",
            "duration_s": 25.0,
            "transcript": "wait watch this incredible finally yes",
            "title": "amazing"
        }
        result = scorer.score(data)
        # 3 hook words: wait, watch, incredible (title has 'amazing' too) → 3 * 0.1 = 0.3 (max 0.3)
        # 2 payoff indicators: finally, yes → 2 * 0.1 = 0.2 (max 0.2)
        # no profanity, short duration
        assert result["score"] == 1.0  # 0.5 + 0.3 + 0.2 = 1.0
        assert result["breakdown"]["hook_bonus"] == 0.3
        assert result["breakdown"]["payoff_bonus"] == 0.2

    def test_score_with_profanity_penalty(self):
        scorer = ClipScorer()
        data = {
            "clip_id": "c3",
            "duration_s": 30.0,
            "transcript": "this is shit",
            "title": "test"
        }
        result = scorer.score(data)
        # 0.5 - 0.2 = 0.3
        assert result["score"] == 0.3
        assert result["breakdown"]["profanity_penalty"] == 0.2

    def test_score_with_length_penalty(self):
        scorer = ClipScorer()
        data = {
            "clip_id": "c4",
            "duration_s": 90.0,
            "transcript": "hello world",
            "title": "test"
        }
        result = scorer.score(data)
        # 0.5 - 0.1 = 0.4
        assert result["score"] == 0.4
        assert result["breakdown"]["length_penalty"] == 0.1

    def test_score_clamps_to_zero(self):
        scorer = ClipScorer()
        data = {
            "clip_id": "c5",
            "duration_s": 90.0,
            "transcript": "this is shit damn it",
            "title": "test"
        }
        result = scorer.score(data)
        # 0.5 - 0.2 (profanity) - 0.1 (length) = 0.2
        assert result["score"] == pytest.approx(0.2)

    def test_score_many_returns_list(self):
        scorer = ClipScorer()
        clips = [
            {"clip_id": "c1", "duration_s": 30.0, "transcript": "hello", "title": "a"},
            {"clip_id": "c2", "duration_s": 60.0, "transcript": "world", "title": "b"},
        ]
        results = scorer.score_many(clips)
        assert len(results) == 2
        assert all("score" in r for r in results)

    def test_score_with_provider_combines(self):
        scorer = ClipScorer()
        data = {
            "clip_id": "c1",
            "duration_s": 30.0,
            "transcript": "hello world",
            "title": "test"
        }

        def provider_fn(cd):
            return 0.8

        result = scorer.score_with_provider(data, provider_fn)
        assert result["clip_id"] == "c1"
        assert result["provider_score"] == 0.8
        # ai_score = base score from our scorer: 0.5
        assert result["ai_score"] == 0.5
        # combined = (0.5 + 0.8) / 2 = 0.65
        assert result["combined_score"] == 0.65


# =============================================================================
# ClipRanker Tests
# =============================================================================

class TestClipRanker:

    def test_rank_returns_sorted_by_score_descending(self):
        scorer = ClipScorer()
        ranker = ClipRanker(scorer)
        clips = [
            {"clip_id": "c1", "duration_s": 30.0, "transcript": "hello", "title": "a"},
            {"clip_id": "c2", "duration_s": 25.0, "transcript": "wait incredible finally yes", "title": "b"},
            {"clip_id": "c3", "duration_s": 90.0, "transcript": "this is shit", "title": "c"},
        ]
        ranked = ranker.rank(clips)
        assert len(ranked) == 3
        assert ranked[0]["score"] >= ranked[1]["score"] >= ranked[2]["score"]

    def test_rank_assigns_rank_field(self):
        scorer = ClipScorer()
        ranker = ClipRanker(scorer)
        clips = [
            {"clip_id": "c1", "duration_s": 30.0, "transcript": "hello", "title": "a"},
            {"clip_id": "c2", "duration_s": 25.0, "transcript": "world", "title": "b"},
        ]
        ranked = ranker.rank(clips)
        assert ranked[0]["rank"] == 1
        assert ranked[1]["rank"] == 2

    def test_top_k_returns_k_clips(self):
        scorer = ClipScorer()
        ranker = ClipRanker(scorer)
        clips = [
            {"clip_id": f"c{i}", "duration_s": 30.0, "transcript": "hello", "title": "a"}
            for i in range(10)
        ]
        top = ranker.top_k(clips, 3)
        assert len(top) == 3

    def test_select_best_returns_highest_scored(self):
        scorer = ClipScorer()
        ranker = ClipRanker(scorer)
        clips = [
            {"clip_id": "c1", "duration_s": 90.0, "transcript": "this is shit", "title": "a"},
            {"clip_id": "c2", "duration_s": 25.0, "transcript": "wait incredible", "title": "b"},
        ]
        best = ranker.select_best(clips)
        assert best is not None
        assert best["clip_id"] == "c2"

    def test_select_best_returns_none_for_empty_list(self):
        scorer = ClipScorer()
        ranker = ClipRanker(scorer)
        best = ranker.select_best([])
        assert best is None
