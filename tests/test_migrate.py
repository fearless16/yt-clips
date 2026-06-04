"""Tests for automation.learner.migrate — one-time migration from old memories table."""

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from automation.learner.migrate import (
    classify_hook_type,
    extract_title_features,
    parse_publish_timing,
    has_devanagari,
    has_emoji,
    read_old_facts,
    build_metrics_event,
    build_dispatcher,
    migrate,
)
from automation.learner.entity_learner import EntityLearner
from automation.learner.state_store import PersistentStateStore
from automation.memory.event_models import EventType


class TestHookClassifier(unittest.TestCase):
    """Test hook type classification from title."""

    def test_classify_reaction_emoji(self):
        self.assertEqual(classify_hook_type("OMG! Speechless! 😱"), "reaction")

    def test_classify_reaction_word(self):
        self.assertEqual(classify_hook_type("Stunned by this catch"), "reaction")

    def test_classify_fan_war_vs(self):
        self.assertEqual(classify_hook_type("MI vs CSK - The Real Battle"), "fan_war")

    def test_classify_shock_keyword(self):
        self.assertEqual(classify_hook_type("Shocking umpire decision"), "shock")

    def test_classify_live_keyword(self):
        self.assertEqual(classify_hook_type("Cricket Live Streaming Now"), "live")

    def test_classify_question_mark(self):
        self.assertEqual(classify_hook_type("Kohli ka form wapas aayega?"), "question")

    def test_classify_prediction_keyword(self):
        self.assertEqual(classify_hook_type("India will win the World Cup"), "prediction")

    def test_classify_debate_keyword(self):
        self.assertEqual(classify_hook_type("Should Rohit retire after T20 WC?"), "debate")

    def test_classify_returns_none_for_generic(self):
        self.assertIsNone(classify_hook_type("GT ne RCB ko haraya"))

    def test_classify_priority_reaction_over_question(self):
        self.assertEqual(
            classify_hook_type("Stunned reaction! Can you believe it?"),
            "reaction",
        )

    def test_classify_priority_fan_war_over_live(self):
        self.assertEqual(
            classify_hook_type("Live: MI vs CSK rivalry continues"),
            "fan_war",
        )

    def test_classify_empty_string(self):
        self.assertIsNone(classify_hook_type(""))


class TestTitleFeatures(unittest.TestCase):
    """Test title feature extraction."""

    def test_length(self):
        title = "RCB won the match"
        feats = extract_title_features(title)
        self.assertEqual(feats["length"], len(title))

    def test_has_pipe(self):
        feats = extract_title_features("Title A | Title B")
        self.assertTrue(feats["has_pipe"])

    def test_no_pipe(self):
        feats = extract_title_features("Title without pipe")
        self.assertFalse(feats["has_pipe"])

    def test_has_question(self):
        feats = extract_title_features("What a match?")
        self.assertTrue(feats["has_question"])

    def test_no_question(self):
        feats = extract_title_features("What a match")
        self.assertFalse(feats["has_question"])

    def test_has_emoji(self):
        feats = extract_title_features("Wow 😱 amazing catch")
        self.assertTrue(feats["has_emoji"])

    def test_no_emoji(self):
        feats = extract_title_features("Wow amazing catch")
        self.assertFalse(feats["has_emoji"])

    def test_is_hinglish_true(self):
        # Hinglish = mix of Devanagari (Hindi) + Latin script
        feats = extract_title_features("Bumrah ने maara six")
        self.assertTrue(feats["is_hinglish"])

    def test_is_hinglish_false_hindi_only(self):
        feats = extract_title_features("बुमराह ने मारा छक्का")
        self.assertFalse(feats["is_hinglish"])

    def test_is_hinglish_false_english_only(self):
        feats = extract_title_features("Bumrah hits a six")
        self.assertFalse(feats["is_hinglish"])

    def test_is_hinglish_transliteration_not_mixed(self):
        # Transliterated Hindi (Latin chars only) is not Hinglish
        feats = extract_title_features("Bumrah ne maara six")
        self.assertFalse(feats["is_hinglish"])

    def test_has_caps(self):
        feats = extract_title_features("RCB vs MI FINAL")
        self.assertTrue(feats["has_caps"])

    def test_no_caps(self):
        feats = extract_title_features("rcb vs mi final")
        self.assertFalse(feats["has_caps"])


class TestPublishTiming(unittest.TestCase):
    """Test publish date/hour parsing."""

    def test_valid_date(self):
        # 2026-06-02 is Tuesday
        hour, dow = parse_publish_timing("20260602")
        self.assertEqual(hour, 19)
        self.assertEqual(dow, 1)

    def test_valid_date_sunday(self):
        # 2026-06-07 is Sunday
        hour, dow = parse_publish_timing("20260607")
        self.assertEqual(hour, 19)
        self.assertEqual(dow, 6)

    def test_valid_date_monday(self):
        # 2026-06-01 is Monday
        hour, dow = parse_publish_timing("20260601")
        self.assertEqual(hour, 19)
        self.assertEqual(dow, 0)

    def test_invalid_date(self):
        hour, dow = parse_publish_timing("not-a-date")
        self.assertIsNone(hour)
        self.assertIsNone(dow)

    def test_empty_string(self):
        hour, dow = parse_publish_timing("")
        self.assertIsNone(hour)
        self.assertIsNone(dow)


class TestDevanagariAndEmoji(unittest.TestCase):
    """Test script/emoji detection helpers."""

    def test_has_devanagari_true(self):
        self.assertTrue(has_devanagari("बुमराह"))

    def test_has_devanagari_false(self):
        self.assertFalse(has_devanagari("bumrah"))

    def test_has_emoji_true(self):
        self.assertTrue(has_emoji("Wow 😱"))

    def test_has_emoji_false(self):
        self.assertFalse(has_emoji("Wow amazing"))


class TestReadOldFacts(unittest.TestCase):
    """Test reading old memories table from self_learner.db."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.tmp_dir) / "test.db"
        self._create_fake_db()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _create_fake_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE memories (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                timestamp REAL NOT NULL,
                confidence REAL DEFAULT 1.0,
                source TEXT DEFAULT 'manual',
                ttl REAL DEFAULT NULL
            );
        """)
        facts = [
            ("fact:upload:was_observed:001", {
                "subject": "upload", "relation": "was_observed",
                "object": {
                    "clip_id": "clipA1", "title": "RCB vs MI Final",
                    "duration": 30, "view_count": 1500, "like_count": 50,
                    "comment_count": 5, "engagement_rate": 3.67,
                    "tag_count": 5, "hashtag_count": 1, "upload_date": "20260601",
                },
            }, 100.0),
            ("fact:upload:was_observed:002", {
                "subject": "upload", "relation": "was_observed",
                "object": {
                    "clip_id": "clipB2", "title": "Bumrah magic spell",
                    "duration": 25, "view_count": 200, "like_count": 5,
                    "comment_count": 0, "engagement_rate": 2.5,
                    "tag_count": 3, "hashtag_count": 0, "upload_date": "20260602",
                },
            }, 200.0),
            ("fact:seo_outcome:has_performance:001", {
                "subject": "seo_outcome", "relation": "has_performance",
                "object": {
                    "clip_id": "clipA1", "title": "RCB vs MI Final",
                    "description": "MI vs RCB rivalry match analysis",
                    "tags": ["RCB", "MI", "IPL"],
                    "is_shorts": True, "model": "minimax-m3",
                },
            }, 150.0),
            ("fact:trend:views:has_data_point:001", {
                "subject": "trend:views", "relation": "has_data_point",
                "object": {"value": 1500.0, "metadata": {"clip_id": "clipA1"}},
            }, 160.0),
        ]
        for key, fact, ts in facts:
            conn.execute(
                "INSERT INTO memories (key, value, timestamp, source) VALUES (?, ?, ?, ?)",
                (key, json.dumps(fact), ts, "test")
            )
        conn.commit()
        conn.close()

    def test_read_uploads(self):
        uploads, seo = read_old_facts(str(self.db_path))
        self.assertEqual(len(uploads), 2)
        self.assertIn("clipA1", uploads)
        self.assertEqual(uploads["clipA1"]["view_count"], 1500)

    def test_read_seo(self):
        uploads, seo = read_old_facts(str(self.db_path))
        self.assertEqual(len(seo), 1)
        self.assertIn("clipA1", seo)
        self.assertEqual(len(seo["clipA1"]["tags"]), 3)

    def test_trend_facts_excluded(self):
        uploads, seo = read_old_facts(str(self.db_path))
        # trend:views fact should be ignored
        self.assertNotIn("trend", uploads)
        self.assertNotIn("trend", seo)


class TestBuildMetricsEvent(unittest.TestCase):
    """Test synthetic ClipEvent construction."""

    def setUp(self):
        self.entity_learner = EntityLearner(state_store=None)  # no state needed for extract

    def test_event_id_deterministic(self):
        upload_obj = {
            "clip_id": "test123", "title": "RCB vs MI Final",
            "duration": 30, "view_count": 1000, "engagement_rate": 1.0,
            "upload_date": "20260601",
        }
        event = build_metrics_event("test123", upload_obj, None, self.entity_learner, 0)
        self.assertEqual(event.event_id, "migrate-0000-test123")
        self.assertEqual(event.event_type, EventType.metrics_received)
        self.assertEqual(event.clip_id, "test123")

    def test_payload_contains_views(self):
        upload_obj = {
            "clip_id": "test", "title": "Title",
            "duration": 25, "view_count": 500, "engagement_rate": 0.5,
            "upload_date": "20260601",
        }
        event = build_metrics_event("test", upload_obj, None, self.entity_learner, 0)
        payload = json.loads(event.payload_json)
        self.assertEqual(payload["views"], 500)
        self.assertEqual(payload["duration_seconds"], 25)

    def test_payload_extracts_hook_type(self):
        upload_obj = {
            "clip_id": "test", "title": "Stunned by this catch! 😱",
            "duration": 25, "view_count": 500, "engagement_rate": 0.5,
            "upload_date": "20260601",
        }
        event = build_metrics_event("test", upload_obj, None, self.entity_learner, 0)
        payload = json.loads(event.payload_json)
        self.assertEqual(payload["hook_type"], "reaction")

    def test_payload_extracts_title_features(self):
        upload_obj = {
            "clip_id": "test",
            "title": "Bumrah ने maara six! | Stunning spell",
            "duration": 25, "view_count": 500, "engagement_rate": 0.5,
            "upload_date": "20260601",
        }
        event = build_metrics_event("test", upload_obj, None, self.entity_learner, 0)
        payload = json.loads(event.payload_json)
        feats = payload["title_pattern"]
        self.assertTrue(feats["has_pipe"])
        self.assertTrue(feats["is_hinglish"])
        self.assertGreater(feats["length"], 30)

    def test_payload_extracts_entities_from_seo(self):
        upload_obj = {
            "clip_id": "test", "title": "Great match today",
            "duration": 25, "view_count": 500, "engagement_rate": 0.5,
            "upload_date": "20260601",
        }
        seo_obj = {
            "description": "Bumrah and Kohli played well for MI vs RCB",
            "tags": ["bumrah", "kohli", "mi", "rcb"],
        }
        event = build_metrics_event("test", upload_obj, seo_obj, self.entity_learner, 0)
        payload = json.loads(event.payload_json)
        self.assertIn("bumrah", payload["player_tags"])
        self.assertIn("kohli", payload["player_tags"])
        self.assertIn("mi", payload["team_tags"])
        self.assertIn("rcb", payload["team_tags"])

    def test_payload_derives_publish_timing(self):
        upload_obj = {
            "clip_id": "test", "title": "Match",
            "duration": 25, "view_count": 500, "engagement_rate": 0.5,
            "upload_date": "20260602",  # Tuesday
        }
        event = build_metrics_event("test", upload_obj, None, self.entity_learner, 0)
        payload = json.loads(event.payload_json)
        self.assertEqual(payload["publish_hour"], 19)
        self.assertEqual(payload["publish_dow"], 1)  # 2026-06-02 is Tuesday

    def test_payload_completion_rate_default(self):
        upload_obj = {
            "clip_id": "test", "title": "Match",
            "duration": 25, "view_count": 500, "engagement_rate": 0.0,
            "upload_date": "20260601",
        }
        event = build_metrics_event("test", upload_obj, None, self.entity_learner, 0)
        payload = json.loads(event.payload_json)
        self.assertEqual(payload["completion_rate"], 0.5)

    def test_payload_marks_as_migrated(self):
        upload_obj = {
            "clip_id": "test", "title": "Match",
            "duration": 25, "view_count": 500, "engagement_rate": 0.5,
            "upload_date": "20260601",
        }
        event = build_metrics_event("test", upload_obj, None, self.entity_learner, 0)
        payload = json.loads(event.payload_json)
        self.assertEqual(payload["migrated_from"], "self_learner.db:memories")

    def test_handles_missing_seo_obj(self):
        upload_obj = {
            "clip_id": "test", "title": "Bumrah amazing",
            "duration": 25, "view_count": 500, "engagement_rate": 0.5,
            "upload_date": "20260601",
        }
        event = build_metrics_event("test", upload_obj, None, self.entity_learner, 0)
        payload = json.loads(event.payload_json)
        # Should still extract entities from title
        self.assertIn("bumrah", payload["player_tags"])


class TestEndToEndMigration(unittest.TestCase):
    """End-to-end test: build fake DB, run migrate(), verify learned_state populated."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.tmp_dir) / "test.db"
        self._create_fake_db()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _create_fake_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE memories (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                timestamp REAL NOT NULL,
                confidence REAL DEFAULT 1.0,
                source TEXT DEFAULT 'manual',
                ttl REAL DEFAULT NULL
            );
        """)
        facts = [
            ("fact:upload:1", {
                "subject": "upload", "relation": "was_observed",
                "object": {
                    "clip_id": "vidAAA", "title": "Bumrah magic spell",
                    "duration": 25, "view_count": 1500, "like_count": 50,
                    "engagement_rate": 3.5, "tag_count": 3,
                    "hashtag_count": 1, "upload_date": "20260601",
                },
            }, 100.0),
            ("fact:upload:2", {
                "subject": "upload", "relation": "was_observed",
                "object": {
                    "clip_id": "vidBBB", "title": "RCB vs MI Final Match",
                    "duration": 35, "view_count": 200, "like_count": 2,
                    "engagement_rate": 1.0, "tag_count": 5,
                    "hashtag_count": 1, "upload_date": "20260602",
                },
            }, 200.0),
            ("fact:upload:3", {
                "subject": "upload", "relation": "was_observed",
                "object": {
                    "clip_id": "vidCCC", "title": "Stunned by this catch 😱",
                    "duration": 20, "view_count": 800, "like_count": 30,
                    "engagement_rate": 3.75, "tag_count": 4,
                    "hashtag_count": 1, "upload_date": "20260602",
                },
            }, 300.0),
            ("fact:seo_outcome:1", {
                "subject": "seo_outcome", "relation": "has_performance",
                "object": {
                    "clip_id": "vidAAA", "title": "Bumrah magic spell",
                    "description": "Bumrah bowled a great spell for MI",
                    "tags": ["bumrah", "mi", "ipl"],
                    "is_shorts": True,
                },
            }, 110.0),
            ("fact:seo_outcome:2", {
                "subject": "seo_outcome", "relation": "has_performance",
                "object": {
                    "clip_id": "vidBBB", "title": "RCB vs MI Final Match",
                    "description": "MI vs RCB rivalry",
                    "tags": ["rcb", "mi", "ipl"],
                    "is_shorts": True,
                },
            }, 210.0),
            ("fact:seo_outcome:3", {
                "subject": "seo_outcome", "relation": "has_performance",
                "object": {
                    "clip_id": "vidCCC", "title": "Stunned by this catch",
                    "description": "Amazing catch by kohli",
                    "tags": ["kohli", "rcb"],
                    "is_shorts": True,
                },
            }, 310.0),
        ]
        for key, fact, ts in facts:
            conn.execute(
                "INSERT INTO memories (key, value, timestamp, source) VALUES (?, ?, ?, ?)",
                (key, json.dumps(fact), ts, "test")
            )
        conn.commit()
        conn.close()

    def test_dry_run_does_not_modify_db(self):
        summary = migrate(db_path=str(self.db_path), dry_run=True)
        self.assertEqual(summary["events_built"], 3)
        self.assertTrue(summary["dry_run"])

        # Verify NO new tables created
        conn = sqlite3.connect(self.db_path)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )]
        conn.close()
        self.assertNotIn("learned_state", tables)
        self.assertNotIn("processed_events", tables)
        self.assertNotIn("trends", tables)

    def test_migrate_creates_learned_state(self):
        summary = migrate(db_path=str(self.db_path), dry_run=False)
        self.assertEqual(summary["events_built"], 3)
        self.assertEqual(summary["events_dispatched"], 3)
        self.assertGreater(summary["learners_updated"], 0)
        self.assertFalse(summary["dry_run"])

        # Verify tables created
        conn = sqlite3.connect(self.db_path)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )]
        conn.close()
        self.assertIn("learned_state", tables)
        self.assertIn("processed_events", tables)
        self.assertIn("trends", tables)

    def test_migrate_populates_entity_scores(self):
        migrate(db_path=str(self.db_path), dry_run=False)

        # Re-open and inspect
        state = PersistentStateStore(str(self.db_path))
        try:
            entity_state = state.get("entity_scores")
            self.assertIsNotNone(entity_state)
            players = entity_state.get("players", {})
            teams = entity_state.get("teams", {})

            # bumrah should be in top players (1500 views)
            self.assertIn("bumrah", players)
            self.assertGreaterEqual(players["bumrah"]["n"], 1)
            self.assertEqual(players["bumrah"]["avg_views"], 1500)

            # mi and rcb should be in teams
            self.assertIn("mi", teams)
            self.assertIn("rcb", teams)
            # mi appears in 2 clips, avg = (1500 + 200) / 2 = 850
            self.assertAlmostEqual(teams["mi"]["avg_views"], 850, delta=1)
        finally:
            state.close()

    def test_migrate_populates_format_scores(self):
        migrate(db_path=str(self.db_path), dry_run=False)

        state = PersistentStateStore(str(self.db_path))
        try:
            format_scores = state.get("format_scores")
            self.assertIsNotNone(format_scores)
            hook_types = format_scores.get("hook_types", {})

            # vidAAA = "Bumrah magic spell" -> no hook type
            # vidBBB = "RCB vs MI Final Match" -> fan_war
            # vidCCC = "Stunned by this catch" -> reaction
            self.assertIn("fan_war", hook_types)
            self.assertIn("reaction", hook_types)
            self.assertEqual(hook_types["fan_war"]["n"], 1)
            self.assertEqual(hook_types["reaction"]["n"], 1)
        finally:
            state.close()

    def test_migrate_populates_duration_scores(self):
        migrate(db_path=str(self.db_path), dry_run=False)

        state = PersistentStateStore(str(self.db_path))
        try:
            duration_scores = state.get("duration_scores")
            self.assertIsNotNone(duration_scores)
            buckets = duration_scores.get("buckets", {})

            # vidAAA = 25s -> 25-34s
            # vidBBB = 35s -> 35-44s
            # vidCCC = 20s -> 15-24s
            self.assertEqual(duration_scores["total_n"], 3)
            self.assertIn("15-24s", buckets)
            self.assertIn("25-34s", buckets)
            self.assertIn("35-44s", buckets)
        finally:
            state.close()

    def test_migrate_populates_timing_scores(self):
        migrate(db_path=str(self.db_path), dry_run=False)

        state = PersistentStateStore(str(self.db_path))
        try:
            # All events defaulted to hour=19
            hour_19 = state.get("timing:hour:19")
            self.assertIsNotNone(hour_19)
            self.assertEqual(hour_19["n"], 3)
        finally:
            state.close()

    def test_migrate_marks_processed_events(self):
        migrate(db_path=str(self.db_path), dry_run=False)

        state = PersistentStateStore(str(self.db_path))
        try:
            log_table = state._conn.execute(
                "SELECT learner, COUNT(*) as n FROM processed_events GROUP BY learner"
            ).fetchall()
            learners_seen = {row[0] for row in log_table}
            # Should have entries for format, entity, timing, duration
            self.assertIn("format", learners_seen)
            self.assertIn("entity", learners_seen)
            self.assertIn("timing", learners_seen)
            self.assertIn("duration", learners_seen)
        finally:
            state.close()

    def test_migrate_idempotent(self):
        # Run twice; second run should be a no-op (events already processed)
        first = migrate(db_path=str(self.db_path), dry_run=False)
        second = migrate(db_path=str(self.db_path), dry_run=False)

        # Second run should have same entity counts (not double-counted)
        state = PersistentStateStore(str(self.db_path))
        try:
            entity_state = state.get("entity_scores")
            players = entity_state.get("players", {})
            # First migration: bumrah n=1. Second: still n=1 (idempotent)
            self.assertEqual(players["bumrah"]["n"], 1)
        finally:
            state.close()

    def test_migrate_with_reset_clears_state(self):
        # First run to populate
        migrate(db_path=str(self.db_path), dry_run=False)

        # Reset and re-run
        summary = migrate(db_path=str(self.db_path), dry_run=False, reset=True)
        self.assertEqual(summary["events_built"], 3)

        state = PersistentStateStore(str(self.db_path))
        try:
            entity_state = state.get("entity_scores")
            # After reset+rebuild, bumrah should still be there
            self.assertIsNotNone(entity_state)
            self.assertIn("bumrah", entity_state.get("players", {}))
        finally:
            state.close()


class TestBuildDispatcher(unittest.TestCase):
    """Test dispatcher construction helper."""

    def test_build_dispatcher_creates_all_components(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            state = PersistentStateStore(db_path)
            processed_log, dispatcher, fmt, ent, tim, dur, trend = build_dispatcher(state)
            try:
                self.assertIsNotNone(processed_log)
                self.assertIsNotNone(dispatcher)
                self.assertEqual(dispatcher._format, fmt)
                self.assertEqual(dispatcher._entity, ent)
                self.assertEqual(dispatcher._trend, trend)
            finally:
                state.close()
        finally:
            import os
            if os.path.exists(db_path):
                os.unlink(db_path)


if __name__ == "__main__":
    unittest.main()
