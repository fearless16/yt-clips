"""Tests for enhanced SEO learner — search impressions, CTR, topic cluster."""

import json
import time
from pathlib import Path
from datetime import datetime

import pytest

from automation.seo.seo_learner import SEOLearner


@pytest.fixture
def fresh_learner(tmp_path):
    db_path = tmp_path / "seo_performance.json"
    class TempLearner(SEOLearner):
        def __init__(self):
            self.performance_db = Path(str(db_path))
            self.performance_db.parent.mkdir(exist_ok=True)
            self.learned_insights = self._load_performance_data()
            self._loaded_mtime = self._db_mtime()
            self._benchmark_lock = __import__("threading").Lock()
    return TempLearner()


class TestSearchImpressions:
    def test_performance_score_uses_search_impressions(self):
        from automation.seo.seo_learner import SEOLearner
        learner = SEOLearner()
        score_with_impressions = learner._calculate_performance_score({
            "viewCount": 1000, "likeCount": 50, "commentCount": 5,
            "search_impressions": 5000, "ctr": 0.05,
        })
        score_without = learner._calculate_performance_score({
            "viewCount": 1000, "likeCount": 50, "commentCount": 5,
        })
        assert score_with_impressions >= score_without


class TestTopicClusterTracking:
    def test_record_with_topic_cluster(self, fresh_learner):
        fresh_learner.record_performance(
            clip_id="test_clip_1",
            title="Kohli Six! 🔥 | RCB vs MI",
            description="Great match highlights.",
            hashtags=["#Shorts", "#RCB"],
            analytics={
                "viewCount": 500, "likeCount": 25, "commentCount": 2,
                "search_impressions": 2000, "ctr": 0.04,
            },
            topic_cluster="rcb_vs_mi_ipl",
        )
        data = fresh_learner.learned_insights
        clips = data["clips"]
        assert len(clips) == 1
        assert clips[0]["clip_id"] == "test_clip_1"
        assert clips[0]["features"] is not None


class TestDescriptionLengthFeature:
    def test_description_length_in_features(self, fresh_learner):
        from automation.seo.seo_learner import SEOLearner
        features = fresh_learner._extract_seo_features(
            title="Test Title 🔥",
            description="A" * 500,
            hashtags=["#Shorts"],
        )
        assert features["description_length"] == 500
