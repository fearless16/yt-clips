"""
TDD for analytics: performance dashboard, SEO feedback loop, AI analysis.
"""

from datetime import datetime, timedelta
from pathlib import Path
import pytest

# ═══════════════════════════════════════════════════════════════════════
# 1. Performance scoring (reused by analytics + seo_learner)
# ═══════════════════════════════════════════════════════════════════════

class TestPerformanceScoring:
    def test_seo_learner_initializes(self):
        from seo_learner import SEOLearner
        learner = SEOLearner()
        assert learner.learned_insights is not None
        assert "clips" in learner.learned_insights

    def test_record_and_retrieve_performance(self):
        from seo_learner import SEOLearner
        learner = SEOLearner()
        learner.record_performance(
            clip_id="test123",
            title="Kya baat hai! Kohli Six!",
            description="Kya shot tha bhai!",
            hashtags=["#IPL2026", "#Kohli"],
            analytics={"viewCount": 1500, "likeCount": 120, "commentCount": 15},
        )
        assert len(learner.learned_insights["clips"]) >= 1
        last = learner.learned_insights["clips"][-1]
        assert last["clip_id"] == "test123"
        assert last["features"]["has_shock_hook"] is True

    def test_performance_score_bounds(self):
        from seo_learner import SEOLearner
        learner = SEOLearner()
        learner.record_performance(
            clip_id="high",
            title="cricket live: Kohli Six!",
            description="Kya shot!",
            hashtags=["#IPL"],
            analytics={"viewCount": 100000, "likeCount": 15000, "commentCount": 500},
        )
        last = learner.learned_insights["clips"][-1]
        assert 0.0 <= last["performance_score"] <= 1.0

    def test_suggestions_generated(self):
        from seo_learner import SEOLearner, generate_performance_report
        learner = SEOLearner()
        # Add some varied data
        for i in range(5):
            learner.record_performance(
                clip_id=f"clip{i}",
                title=f"cricket live: Kohli Six {i}!",
                description="Kya shot!",
                hashtags=["#IPL"],
                analytics={"viewCount": 1000 * (i + 1), "likeCount": 100 * (i + 1), "commentCount": 10 * (i + 1)},
            )
        suggestions = learner.get_seo_improvement_suggestions()
        assert isinstance(suggestions, list)

    def test_report_generates(self):
        from seo_learner import generate_performance_report
        report = generate_performance_report()
        assert isinstance(report, str)
        assert len(report) > 0


# ═══════════════════════════════════════════════════════════════════════
# 2. Analytics data processing (no auth needed)
# ═══════════════════════════════════════════════════════════════════════

class TestAnalyticsProcessing:
    def test_dashboard_print_no_crash(self):
        """Print function should not crash with empty or valid data."""
        from analytics import print_performance_dashboard
        print_performance_dashboard([])
        print_performance_dashboard([
            {"id": "1", "title": "test", "published": datetime.now(),
             "views": 100, "likes": 10, "comments": 2, "tags": ["#test"]},
        ])
        print_performance_dashboard([
            {"id": str(i), "title": f"short {i}", "published": datetime.now(),
             "views": i * 100, "likes": i * 10, "comments": i, "tags": []}
            for i in range(10)
        ])

    def test_seo_learner_feed(self):
        """Feeding analytics into SEOLearner should update learned patterns."""
        from analytics import feed_seo_learner, seo_learner as analytics_learner

        initial_count = len(analytics_learner.learned_insights["clips"])
        shorts = [
            {"id": "a1", "title": "Kya baat! Six!", "published": datetime.now(),
             "views": 5000, "likes": 400, "comments": 50, "tags": ["#IPL"]},
        ]
        feed_seo_learner(shorts)
        assert len(analytics_learner.learned_insights["clips"]) > initial_count

    def test_ai_analysis_returns_none_with_few_shorts(self):
        from analytics import ai_analyze
        result = ai_analyze([{"id": "1", "title": "t", "published": datetime.now(),
                               "views": 10, "likes": 1, "comments": 0, "tags": []}])
        assert result is None  # Need at least 4 shorts
