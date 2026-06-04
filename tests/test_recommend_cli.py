"""Tests for automation.learner.recommend CLI module."""

import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from automation.learner.recommend import (
    show_recommendations,
    score_candidate,
    show_stats,
    _build_engine,
    _format_recommendation,
    main,
)
from automation.learner.recommender import CricketRecommendation
from automation.learner.state_store import PersistentStateStore

from tests._learner_fixtures import seed_with_facts


class TestFormatRecommendation(unittest.TestCase):
    """Test recommendation formatting."""

    def test_basic_format(self):
        rec = CricketRecommendation(
            category="player",
            priority="high",
            recommendation="Cover bumrah more",
            reason="avg 1500 views (5x baseline)",
            confidence=0.85,
        )
        output = _format_recommendation(rec)
        self.assertIn("Cover bumrah more", output)
        self.assertIn("PLAYER", output)
        self.assertIn("high", output)
        self.assertIn("0.85", output)

    def test_with_supporting_data(self):
        rec = CricketRecommendation(
            category="hook",
            priority="medium",
            recommendation="Avoid question hooks",
            reason="low views",
            confidence=0.70,
            supporting_data={
                "hook_type": "question", "n": 5, "avg_views": 100,
                "score": 0.3, "extra": "x"
            },
        )
        output = _format_recommendation(rec)
        self.assertIn("hook_type=question", output)
        self.assertIn("n=5", output)
        # Only first 4 fields shown — "extra" should be omitted
        self.assertNotIn("extra=x", output)

    def test_with_expires_at(self):
        rec = CricketRecommendation(
            category="topic",
            priority="high",
            recommendation="Cover trending X",
            reason="active trend",
            confidence=0.90,
            expires_at="2026-06-05T00:00:00",
        )
        output = _format_recommendation(rec)
        self.assertIn("2026-06-05", output)


class TestBuildEngine(unittest.TestCase):
    """Test engine construction helper."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.tmp_dir) / "test.db"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_builds_all_components(self):
        state, fmt, ent, tim, dur, trend, scorer, rec_engine = _build_engine(str(self.db_path))
        try:
            self.assertIsNotNone(state)
            self.assertIsNotNone(fmt)
            self.assertIsNotNone(ent)
            self.assertIsNotNone(tim)
            self.assertIsNotNone(dur)
            self.assertIsNotNone(trend)
            self.assertIsNotNone(scorer)
            self.assertIsNotNone(rec_engine)
        finally:
            state.close()


class TestShowRecommendations(unittest.TestCase):
    """Test recommendation display."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.tmp_dir) / "test.db"
        seed_with_facts(self.tmp_dir, str(self.db_path))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_shows_recommendations_text(self):
        state, fmt, ent, tim, dur, trend, scorer, rec_engine = _build_engine(str(self.db_path))
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = show_recommendations(rec_engine, as_json=False)
            output = buf.getvalue()
            self.assertEqual(rc, 0)
            self.assertIn("RECOMMENDATIONS", output)
        finally:
            state.close()

    def test_json_output(self):
        state, fmt, ent, tim, dur, trend, scorer, rec_engine = _build_engine(str(self.db_path))
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = show_recommendations(rec_engine, as_json=True)
            output = buf.getvalue()
            self.assertEqual(rc, 0)
            data = json.loads(output)
            self.assertIsInstance(data, list)
        finally:
            state.close()

    def test_category_filter(self):
        state, fmt, ent, tim, dur, trend, scorer, rec_engine = _build_engine(str(self.db_path))
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                show_recommendations(rec_engine, category="player", as_json=True)
            data = json.loads(buf.getvalue())
            for rec in data:
                self.assertEqual(rec["category"], "player")
        finally:
            state.close()

    def test_empty_state(self):
        # Empty DB
        empty_db = Path(self.tmp_dir) / "empty.db"
        state = PersistentStateStore(str(empty_db))
        state._conn.execute("DELETE FROM learned_state")
        state._conn.execute("DELETE FROM processed_events")
        state._conn.commit()

        class FakeFmt:
            def get_scores(self):
                return {"hook_types": {}, "title_patterns": {}}

        class FakeEnt:
            def get_top_entities(self, t, limit=5):
                return []
            def get_entity_score(self, *a, **k):
                return 0.5
            def extract_entities(self, *a, **k):
                return {"players": [], "teams": [], "series": []}
            def fatigue_penalty(self, *a, **k):
                return 0.0

        class FakeTim:
            def best_hour(self):
                return 19
            def best_day(self):
                return "saturday"
            def best_match_phase(self):
                return "post_match"
            def get_schedule_recommendation(self):
                return {"total_observations": 0, "confidence": 0.0,
                        "hour": 19, "day": "saturday", "reason": "default",
                        "avg_views_at_hour": 0}

        class FakeDur:
            def select_duration(self):
                return 30
            def get_scores(self):
                return {"buckets": {}, "sweet_spot": "25-34s", "total_n": 0}

        class FakeTrend:
            def get_active(self):
                return []
            def get_hot_topics(self, limit=3):
                return []
            def get_trend_for_entities(self, **k):
                return 0.0

        class FakeScorer:
            def get_weights(self):
                return {"trend_weight": 0.3, "format_weight": 0.25,
                        "entity_weight": 0.2, "historical_weight": 0.15,
                        "timing_weight": 0.1}

        class FakeRec:
            def generate_all(self):
                return []

        rec_engine = FakeRec()

        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                show_recommendations(rec_engine, as_json=True)
            data = json.loads(buf.getvalue())
            self.assertEqual(data, [])
        finally:
            state.close()


class TestScoreCandidate(unittest.TestCase):
    """Test candidate scoring."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.tmp_dir) / "test.db"
        seed_with_facts(self.tmp_dir, str(self.db_path))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_score_text_output(self):
        state, fmt, ent, tim, dur, trend, scorer, rec_engine = _build_engine(str(self.db_path))
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = score_candidate(
                    fmt, ent, trend, tim, dur, scorer,
                    title="Bumrah magic spell",
                    players=["bumrah"],
                    teams=["mi"],
                    as_json=False,
                )
            output = buf.getvalue()
            self.assertEqual(rc, 0)
            self.assertIn("CANDIDATE SCORE", output)
            self.assertIn("Bumrah magic spell", output)
            self.assertIn("bumrah", output)
            self.assertIn("mi", output)
        finally:
            state.close()

    def test_score_json_output(self):
        state, fmt, ent, tim, dur, trend, scorer, rec_engine = _build_engine(str(self.db_path))
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                score_candidate(
                    fmt, ent, trend, tim, dur, scorer,
                    title="Test Title",
                    players=["kohli"],
                    teams=["rcb"],
                    series="ipl",
                    as_json=True,
                )
            data = json.loads(buf.getvalue())
            self.assertIn("score", data)
            self.assertIn("breakdown", data)
            self.assertIn("suggestions", data)
            self.assertIn("weights", data)
            self.assertGreaterEqual(data["score"], 0.0)
            self.assertLessEqual(data["score"], 1.0)
        finally:
            state.close()

    def test_score_with_no_entities(self):
        state, fmt, ent, tim, dur, trend, scorer, rec_engine = _build_engine(str(self.db_path))
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                score_candidate(
                    fmt, ent, trend, tim, dur, scorer,
                    title="Some random clip",
                    players=[],
                    teams=[],
                    as_json=False,
                )
            output = buf.getvalue()
            self.assertIn("(none)", output)
        finally:
            state.close()

    def test_score_verdict_strong(self):
        state, fmt, ent, tim, dur, trend, scorer, rec_engine = _build_engine(str(self.db_path))
        try:
            # bumrah is top player with 1326 views, mi is top team
            buf = io.StringIO()
            with redirect_stdout(buf):
                score_candidate(
                    fmt, ent, trend, tim, dur, scorer,
                    title="Bumrah stunning catch! 😱 | MI vs RCB rivalry",
                    players=["bumrah", "kohli"],
                    teams=["mi", "rcb"],
                    series="ipl",
                    as_json=False,
                )
            output = buf.getvalue()
            # Should produce some verdict
            self.assertIn("Verdict:", output)
        finally:
            state.close()


class TestShowStats(unittest.TestCase):
    """Test raw stats display."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.tmp_dir) / "test.db"
        seed_with_facts(self.tmp_dir, str(self.db_path))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_stats_text_output(self):
        state, fmt, ent, tim, dur, trend, scorer, rec_engine = _build_engine(str(self.db_path))
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = show_stats(state, fmt, ent, tim, dur, trend, scorer, as_json=False)
            output = buf.getvalue()
            self.assertEqual(rc, 0)
            self.assertIn("RAW SCORES", output)
            self.assertIn("Scorer weights", output)
            self.assertIn("Format", output)
            self.assertIn("Entities", output)
        finally:
            state.close()

    def test_stats_json_output(self):
        state, fmt, ent, tim, dur, trend, scorer, rec_engine = _build_engine(str(self.db_path))
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                show_stats(state, fmt, ent, tim, dur, trend, scorer, as_json=True)
            data = json.loads(buf.getvalue())
            self.assertIn("weights", data)
            self.assertIn("format", data)
            self.assertIn("entity", data)
            self.assertIn("timing", data)
            self.assertIn("duration", data)
            self.assertIn("trends", data)
        finally:
            state.close()


class TestCLIMain(unittest.TestCase):
    """Test the CLI main() entry point."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.tmp_dir) / "test.db"
        seed_with_facts(self.tmp_dir, str(self.db_path))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_main_default_shows_recommendations(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["--db", str(self.db_path)])
        self.assertEqual(rc, 0)
        self.assertIn("RECOMMENDATIONS", buf.getvalue())

    def test_main_json_output(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["--db", str(self.db_path), "--json"])
        self.assertEqual(rc, 0)
        data = json.loads(buf.getvalue())
        self.assertIsInstance(data, list)

    def test_main_category_filter(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["--db", str(self.db_path), "--category", "player", "--json"])
        data = json.loads(buf.getvalue())
        for rec in data:
            self.assertEqual(rec["category"], "player")

    def test_main_stats(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["--db", str(self.db_path), "--stats"])
        self.assertEqual(rc, 0)
        self.assertIn("RAW SCORES", buf.getvalue())

    def test_main_score(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main([
                "--db", str(self.db_path),
                "--score",
                "--title", "Bumrah magic spell",
                "--players", "bumrah",
                "--teams", "mi",
            ])
        self.assertEqual(rc, 0)
        self.assertIn("CANDIDATE SCORE", buf.getvalue())

    def test_main_score_without_title_errors(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["--db", str(self.db_path), "--score"])
        self.assertEqual(rc, 1)

    def test_main_score_json(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main([
                "--db", str(self.db_path),
                "--score",
                "--title", "Test title",
                "--players", "kohli",
                "--teams", "rcb",
                "--json",
            ])
        self.assertEqual(rc, 0)
        data = json.loads(buf.getvalue())
        self.assertIn("score", data)


if __name__ == "__main__":
    unittest.main()
