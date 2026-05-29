"""Tests for the self-learning improvements (feat/self-learning)."""
import pytest
from unittest.mock import MagicMock, patch

from automation.seo.seo_learner import SEOLearner, _stable_pattern_key, MIN_CLIPS_FOR_PATTERN
import automation.seo.analytics as analytics


def _fresh_learner(tmp_path):
    learner = SEOLearner()
    learner.performance_db = tmp_path / "perf.json"
    learner.learned_insights = {
        "clips": [], "title_patterns": {}, "hooks_performance": {},
        "ctas_performance": {}, "hashtag_performance": {}, "feature_importance": {},
        "model_performance": {}, "benchmark_history": [],
        "current_best_provider": None, "current_best_model": None,
        "last_updated": None, "llm_insights": [], "version": 2,
    }
    return learner


def test_pattern_key_excludes_numeric_features():
    """Two clips differing ONLY in numeric features must share a pattern key."""
    base = {"has_pipe_format": True, "has_emoji": False, "hashtag_count_bin": "4-5"}
    a = {**base, "title_length": 47, "description_length": 1200, "title_words": 9}
    b = {**base, "title_length": 88, "description_length": 3400, "title_words": 14}
    assert _stable_pattern_key(a) == _stable_pattern_key(b)
    # ...but differing booleans/bins still produce distinct keys.
    c = {**base, "has_emoji": True, "title_length": 47}
    assert _stable_pattern_key(c) != _stable_pattern_key(a)


def test_record_performance_stores_title(tmp_path):
    learner = _fresh_learner(tmp_path)
    learner.record_performance(
        clip_id="v1", title="LIVE RCB vs CSK | Kohli 67",
        description="d", hashtags=["#Shorts"],
        analytics={"viewCount": 5000, "likeCount": 200, "commentCount": 30},
        provider="groq", model="m1",
    )
    stored = learner.learned_insights["clips"][-1]
    assert stored["analytics"]["title"] == "LIVE RCB vs CSK | Kohli 67"


def test_retention_ctr_branch_activates():
    """When retention+ctr are present the scorer must use the richer formula
    (previously dead code because only views/likes/comments were ever supplied)."""
    learner = SEOLearner()
    base = {"viewCount": 5000, "likeCount": 200, "commentCount": 30}
    score_basic = learner._calculate_performance_score(base)
    score_rich = learner._calculate_performance_score(
        {**base, "retention": 0.95, "ctr": 0.09}
    )
    # Strong retention+CTR should pull the score meaningfully higher.
    assert score_rich != score_basic
    assert score_rich > score_basic


def test_recompute_best_model_prefers_real_over_benchmark(tmp_path):
    learner = _fresh_learner(tmp_path)
    # Synthetic benchmark says nvidia is best...
    learner.learned_insights["benchmark_history"] = [
        {"top_result": {"provider": "nvidia", "model": "nemo", "score": 80}}
    ]
    # ...but real performance data (>= MIN clips) says groq is best.
    learner.learned_insights["model_performance"] = {
        "groq/scout": {"count": MIN_CLIPS_FOR_PATTERN, "avg_score": 0.8,
                       "provider": "groq", "model": "scout"},
        "deepseek/chat": {"count": MIN_CLIPS_FOR_PATTERN, "avg_score": 0.4,
                          "provider": "deepseek", "model": "chat"},
    }
    source = learner._recompute_best_model()
    assert source == "real"
    assert learner.learned_insights["current_best_provider"] == "groq"
    assert learner.learned_insights["current_best_model"] == "scout"


def test_recompute_best_model_benchmark_coldstart(tmp_path):
    learner = _fresh_learner(tmp_path)
    learner.learned_insights["benchmark_history"] = [
        {"top_result": {"provider": "groq", "model": "scout", "score": 70}}
    ]
    source = learner._recompute_best_model()
    assert source == "benchmark"
    assert learner.learned_insights["current_best_provider"] == "groq"


def test_fetch_advanced_metrics_parses_analytics_api():
    fake_service = MagicMock()
    fake_service.reports().query().execute.return_value = {
        "columnHeaders": [
            {"name": "video"}, {"name": "estimatedMinutesWatched"},
            {"name": "averageViewPercentage"}, {"name": "impressions"},
            {"name": "impressionClickThroughRate"},
        ],
        "rows": [["vidA", 1234.0, 82.5, 10000, 7.5]],
    }
    with patch.object(analytics, "_auth_analytics", return_value=fake_service):
        analytics.ANALYTICS_CACHE.clear() if hasattr(analytics.ANALYTICS_CACHE, "clear") else None
        out = analytics.fetch_advanced_metrics(["vidA"])
    assert "vidA" in out
    assert out["vidA"]["retention"] == pytest.approx(0.825)
    assert out["vidA"]["ctr"] == pytest.approx(0.075)
    assert out["vidA"]["impressions"] == 10000


def test_fetch_advanced_metrics_degrades_without_auth():
    with patch.object(analytics, "_auth_analytics", return_value=None):
        assert analytics.fetch_advanced_metrics(["x"]) == {}



def _empty_insights():
    return {
        "clips": [], "title_patterns": {}, "hooks_performance": {},
        "ctas_performance": {}, "hashtag_performance": {}, "feature_importance": {},
        "model_performance": {}, "benchmark_history": [],
        "current_best_provider": None, "current_best_model": None,
        "last_updated": None, "llm_insights": [], "version": 2,
    }


def test_get_best_model_reloads_on_cross_instance_write(tmp_path):
    """Bug 4: a reader must observe best-model updates written by a *different*
    instance/process (benchmark writer) to the same perf DB, not stale data."""
    db = tmp_path / "perf.json"

    reader = SEOLearner()
    reader.performance_db = db
    reader.learned_insights = _empty_insights()
    reader._loaded_mtime = 0.0
    assert reader.get_best_model() == (None, None)

    # Separate writer instance commits a best model to the SAME file.
    writer = SEOLearner()
    writer.performance_db = db
    writer.learned_insights = _empty_insights()
    writer.learned_insights["current_best_provider"] = "groq"
    writer.learned_insights["current_best_model"] = "scout"
    writer._save_performance_data()

    # Reader picks it up via mtime-triggered reload.
    assert reader.get_best_model() == ("groq", "scout")


def test_get_best_model_no_reload_when_unchanged(tmp_path):
    """No disk change => reader keeps its in-memory state (no needless reload)."""
    db = tmp_path / "perf.json"
    reader = SEOLearner()
    reader.performance_db = db
    reader.learned_insights = _empty_insights()
    reader.learned_insights["current_best_provider"] = "deepseek"
    reader.learned_insights["current_best_model"] = "deepseek-chat"
    # File never written; mtime stays 0 -> no reload, in-memory value preserved.
    reader._loaded_mtime = reader._db_mtime()
    assert reader.get_best_model() == ("deepseek", "deepseek-chat")
