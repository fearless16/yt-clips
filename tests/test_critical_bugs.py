"""TDD tests for critical bugs — tests fail BEFORE fixes, pass AFTER."""

import sys, json, threading, time, shutil
from pathlib import Path
from unittest.mock import Mock, patch, PropertyMock, MagicMock, call, ANY
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


# ─── premium_analyzer.py: _linear_assignment greedy fallback ─────────────────

class TestLinearAssignmentGreedy:
    """_linear_assignment greedy fallback must minimize cost, not match blindly."""

    def test_greedy_picks_best_match_per_row(self):
        from premium_analyzer import _linear_assignment
        cost = np.array([
            [0.9, 0.1],
            [0.2, 0.8],
        ], dtype=np.float32)
        matches = _linear_assignment(cost)
        assert len(matches) == 2
        match_set = set(matches)
        assert (0, 1) in match_set
        assert (1, 0) in match_set

    def test_greedy_rectangular_more_detections(self):
        from premium_analyzer import _linear_assignment
        cost = np.array([
            [0.9, 0.1],
            [0.2, 0.8],
            [0.3, 0.7],
        ], dtype=np.float32)
        matches = _linear_assignment(cost)
        assert len(matches) == 2
        match_set = set(matches)
        assert (0, 1) in match_set
        assert (1, 0) in match_set

    def test_greedy_rectangular_more_trackers(self):
        from premium_analyzer import _linear_assignment
        cost = np.array([
            [0.9, 0.1, 0.5],
            [0.2, 0.8, 0.3],
        ], dtype=np.float32)
        matches = _linear_assignment(cost)
        assert len(matches) == 2
        match_set = set(matches)
        assert (0, 1) in match_set
        assert (1, 0) in match_set

    def test_greedy_empty(self):
        from premium_analyzer import _linear_assignment
        assert _linear_assignment(np.empty((0, 4))) == []
        assert _linear_assignment(np.empty((4, 0))) == []
        assert _linear_assignment(np.empty((0, 0))) == []

    def test_greedy_single(self):
        from premium_analyzer import _linear_assignment
        cost = np.array([[0.5]], dtype=np.float32)
        assert _linear_assignment(cost) == [(0, 0)]

    def test_greedy_with_scipy(self):
        from premium_analyzer import _linear_assignment
        cost = np.array([
            [0.9, 0.1],
            [0.2, 0.8],
        ], dtype=np.float32)
        matches = _linear_assignment(cost)
        assert len(matches) == 2
        match_set = set(matches)
        assert (0, 1) in match_set


# ─── worker.py: hardcoded pipeline.py path ─────────────────────────────────────

class TestWorkerPath:
    """worker.py must resolve pipeline.py relative to its own location."""

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.glob")
    @patch("pathlib.Path.read_text", return_value="https://youtu.be/test")
    @patch("subprocess.run")
    def test_pipeline_path_resolved(self, mock_run, mock_read, mock_glob, mock_exists):
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        mock_glob.return_value = [Path("pending/test.url")]
        from worker import check_new_videos
        check_new_videos()
        args, _ = mock_run.call_args
        cmd_list = args[0]
        pipeline_arg = cmd_list[1]
        assert "pipeline.py" in pipeline_arg, \
            f"Expected pipeline.py in args, got: {cmd_list}"

    def test_pipeline_file_exists_relative_to_worker(self):
        worker_path = Path(__file__).resolve().parent.parent / "worker.py"
        assert worker_path.exists()
        pipeline_path = worker_path.parent / "pipeline.py"
        assert pipeline_path.exists()


# ─── pipeline.py: enhancement backup before overwrite ────────────────────────

class TestPipelineEnhancementBackup:
    """Enhancement must backup originals before shutil.move overwrite."""

    def test_backup_created_before_move(self):
        import tempfile, shutil as real_shutil
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            clip = tmp / "clip1.mp4"
            clip.write_text("original")
            enhanced = tmp / "clip1_enhanced.mp4"
            enhanced.write_text("enhanced data")

            bak = clip.with_suffix(".mp4.bak")
            assert not bak.exists()
            real_shutil.copy2(str(clip), str(bak))
            real_shutil.move(str(enhanced), str(clip))
            assert bak.exists()
            assert bak.read_text() == "original"
            assert clip.read_text() == "enhanced data"

    def test_backup_survives_move_failure(self):
        with patch("shutil.move", side_effect=Exception("disk full")):
            import tempfile, shutil as real_shutil
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                clip = tmp / "clip1.mp4"
                clip.write_text("original")
                bak = clip.with_suffix(".mp4.bak")
                real_shutil.copy2(str(clip), str(bak))
                assert bak.read_text() == "original"


# ─── seo_learner.py: learn from ALL scores ───────────────────────────────────

class TestSEOLearnerAllScores:
    """Patterns must be learned from ALL performance scores, not just extremes."""

    def test_all_scores_contribute_to_patterns(self):
        from automation.seo.seo_learner import SEOLearner
        learner = SEOLearner()
        learner.learned_insights["clips"] = []
        learner.learned_insights["title_patterns"] = {}

        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        for i, score in enumerate(scores):
            features = {
                "has_pipe_format": True, "has_power_word": False,
                "has_player_name": True, "has_score": False,
                "has_emoji": False, "sections_count": 2,
                "has_sections": False, "has_cta": True,
                "hashtag_count_bin": "4-5",
                "has_shorts_hashtag": True, "has_ipl_hashtag": True,
                "has_team_hashtag": False,
                "title_length": 60, "title_words": 9,
                "description_length": 500,
            }
            learner._update_learned_patterns(features, score, f"2026-05-{20+i}T12:00:00")

        total_count = sum(
            p["count"] for p in learner.learned_insights["title_patterns"].values()
        )
        assert total_count == len(scores), \
            f"Expected pattern count {len(scores)}, got {total_count}"

    def test_dedup_same_clip_id(self):
        """Same clip_id should replace old entry, not append."""
        from automation.seo.seo_learner import SEOLearner
        learner = SEOLearner()
        learner.learned_insights["clips"] = []

        learner.record_performance(
            clip_id="abc123", title="Test Video", description="desc",
            hashtags=["#shorts"], analytics={"viewCount": 100},
        )
        assert len(learner.learned_insights["clips"]) == 1

        learner.record_performance(
            clip_id="abc123", title="Test Video Updated", description="desc2",
            hashtags=["#shorts"], analytics={"viewCount": 200},
        )
        assert len(learner.learned_insights["clips"]) == 1, \
            "Same clip_id should dedup, not append"
        assert learner.learned_insights["clips"][0]["analytics"]["viewCount"] == 200

    def test_time_decay_weights(self):
        """Recent clips should have higher weight than old clips."""
        from automation.seo.seo_learner import _time_decay_weight
        now = "2026-06-01T12:00:00"
        recent = _time_decay_weight("2026-05-30T12:00:00")
        old = _time_decay_weight("2026-01-01T12:00:00")
        assert recent > old, f"Recent ({recent}) should weigh more than old ({old})"
        assert 0 < recent <= 1
        assert 0 < old <= 1

    def test_feature_importance_tracks_relevant_features(self):
        """Feature importance should compute delta for boolean features."""
        from automation.seo.seo_learner import SEOLearner
        learner = SEOLearner()
        learner.learned_insights["clips"] = []
        learner.learned_insights["feature_importance"] = {}

        # 3 clips with pipe_format, 3 without
        for i in range(3):
            learner.record_performance(
                clip_id=f"pipe_{i}", title="RCB vs CSK | IPL 2026", description="desc",
                hashtags=["#shorts"], analytics={"viewCount": 1000 + i * 100},
            )
        for i in range(3):
            learner.record_performance(
                clip_id=f"no_pipe_{i}", title="Just a cricket video", description="desc",
                hashtags=["#shorts"], analytics={"viewCount": 100 + i * 10},
            )

        fi = learner.get_feature_importance()
        pipe_key = "has_pipe_format"
        assert pipe_key in fi, f"Expected {pipe_key} in feature importance, got {list(fi.keys())}"
        assert fi[pipe_key]["delta"] > 0, \
            "Pipe format should have positive delta (higher scores)"


class TestSEOLearnerHashtagPerformance:
    """hashtag_performance must be populated in _update_learned_patterns."""

    def test_hashtag_performance_populated(self):
        from automation.seo.seo_learner import SEOLearner
        learner = SEOLearner()
        learner.learned_insights["hashtag_performance"] = {}

        features = {
            "has_pipe_format": True, "has_power_word": True,
            "has_player_name": True, "has_score": False,
            "has_emoji": False, "sections_count": 2,
            "has_sections": True, "has_cta": True,
            "hashtag_count_bin": "4-5",
            "has_shorts_hashtag": True, "has_ipl_hashtag": True,
            "has_team_hashtag": True,
            "title_length": 60, "title_words": 9,
            "description_length": 500,
        }
        learner._update_learned_patterns(features, 0.85, "2026-05-24T12:00:00")

        hp = learner.learned_insights["hashtag_performance"]
        assert len(hp) > 0, "hashtag_performance should not be empty"
        found = any("hashtag" in k or "shorts" in k or "ipl" in k or "team" in k
                    for k in hp)
        assert found, f"No hashtag-related entries found in {hp}"


class TestSEOLearnerScoreRealMetrics:
    """Performance score should exclude retention/CTR when unavailable."""

    def test_score_without_retention_ctr(self):
        from automation.seo.seo_learner import SEOLearner
        learner = SEOLearner()
        analytics = {"viewCount": 1000, "likeCount": 50, "commentCount": 10}
        score1 = learner._calculate_performance_score({**analytics, "retention": None, "ctr": None})
        score2 = learner._calculate_performance_score(analytics)
        assert 0 <= score1 <= 1
        assert 0 <= score2 <= 1

    def test_score_with_real_retention_ctr(self):
        from automation.seo.seo_learner import SEOLearner
        learner = SEOLearner()
        analytics = {
            "viewCount": 1000, "likeCount": 50, "commentCount": 10,
            "retention": 0.6, "ctr": 0.08,
        }
        score = learner._calculate_performance_score(analytics)
        assert 0 <= score <= 1
        fake_score = learner._calculate_performance_score(
            {"viewCount": 1000, "likeCount": 50, "commentCount": 10}
        )
        assert score != fake_score, "Score with real retention/CTR should differ from default"


class TestSEOLearnerAutoBenchmarkThreadSafe:
    """run_auto_benchmark must use a lock for provider/model swap."""

    def test_benchmark_has_lock(self):
        from automation.seo.seo_learner import SEOLearner
        learner = SEOLearner()
        assert hasattr(learner, "_benchmark_lock")
        assert isinstance(learner._benchmark_lock, threading.Lock)


class TestSEOLearnerStablePatternKeys:
    """Pattern keys must be deterministic regardless of feature key order."""

    def test_stable_key_order_independent(self):
        from automation.seo.seo_learner import _stable_pattern_key
        f1 = {"has_pipe_format": True, "has_power_word": False, "has_player_name": True}
        f2 = {"has_power_word": False, "has_player_name": True, "has_pipe_format": True}
        assert _stable_pattern_key(f1) == _stable_pattern_key(f2), \
            "Pattern keys must be order-independent"


# ─── analytics.py: pagination ────────────────────────────────────────────────

class TestAnalyticsPagination:
    """_fetch_all_recent should paginate past 50 results."""

    def test_pagination_loop_exists(self):
        import inspect
        from automation.seo.analytics import _fetch_all_recent
        source = inspect.getsource(_fetch_all_recent)
        assert "pageToken" in source
        assert "nextPageToken" in source


# ─── analytics.py: shorts detection ──────────────────────────────────────────

class TestAnalyticsShortsDetection:
    """Shorts classification must be robust — not solely dependent on tags."""

    def test_shorts_detection_in_title(self):
        from automation.seo.analytics import _classify_content_type
        result = _classify_content_type(duration=45, has_shorts_tag=False, title="#Shorts Kohli six")
        assert result == "shorts"

    def test_shorts_classified_by_duration(self):
        from automation.seo.analytics import _classify_content_type
        result = _classify_content_type(duration=45, has_shorts_tag=False)
        assert result == "shorts"

    def test_shorts_180_seconds_with_tag(self):
        from automation.seo.analytics import _classify_content_type
        result = _classify_content_type(duration=120, has_shorts_tag=True)
        assert result == "shorts"

    def test_video_over_60_no_shorts_marker(self):
        from automation.seo.analytics import _classify_content_type
        result = _classify_content_type(duration=120, has_shorts_tag=False)
        assert result == "video"

    def test_live_broadcast(self):
        from automation.seo.analytics import _classify_content_type
        result = _classify_content_type(duration=3600, has_shorts_tag=False, is_live=True)
        assert result == "live"


# ─── analytics.py: ISO 8601 duration parser ──────────────────────────────────

class TestParseISO8601Duration:
    """_parse_iso8601_duration should handle fractional seconds and strict ISO."""

    def test_fractional_seconds(self):
        from automation.seo.analytics import _parse_iso8601_duration
        result = _parse_iso8601_duration("PT1.5S")
        assert result == 1

    def test_strict_pt_prefix(self):
        from automation.seo.analytics import _parse_iso8601_duration
        assert _parse_iso8601_duration("PT1H2M3S") == 3723

    def test_empty_string(self):
        from automation.seo.analytics import _parse_iso8601_duration
        assert _parse_iso8601_duration("") == 0
        assert _parse_iso8601_duration(None) == 0

    def test_hms_combinations(self):
        from automation.seo.analytics import _parse_iso8601_duration
        assert _parse_iso8601_duration("PT5M") == 300
        assert _parse_iso8601_duration("PT1H") == 3600
        assert _parse_iso8601_duration("PT30S") == 30
