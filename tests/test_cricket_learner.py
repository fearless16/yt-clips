"""Tests for the new self-learning engine components."""

import json
import os
import tempfile
import pytest
from datetime import datetime, timezone, timedelta

from automation.learner.state_store import PersistentStateStore
from automation.learner.processed_log import ProcessedEventLog
from automation.learner.format_learner import FormatLearner
from automation.learner.entity_learner import EntityLearner
from automation.learner.trend_engine import TrendEngine, TrendInput
from automation.learner.timing_learner import TimingLearner
from automation.learner.duration_learner import DurationLearner
from automation.learner.dispatcher import LearningDispatcher
from automation.learner.scorer import CricketScorer
from automation.learner.recommender import CricketRecommendationEngine
from automation.memory.event_models import ClipEvent, EventType


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def state_store(temp_db):
    """Create a PersistentStateStore with temp DB."""
    store = PersistentStateStore(temp_db)
    yield store
    store.close()


@pytest.fixture
def processed_log(state_store):
    """Create a ProcessedEventLog."""
    return ProcessedEventLog(state_store._conn)


class TestPersistentStateStore:
    """Tests for PersistentStateStore."""

    def test_get_set(self, state_store):
        """Test basic get/set operations."""
        state_store.set("test_key", {"score": 0.8, "n": 5})
        result = state_store.get("test_key")
        assert result is not None
        assert result["score"] == 0.8
        assert result["n"] == 5

    def test_get_nonexistent(self, state_store):
        """Test getting a key that doesn't exist."""
        result = state_store.get("nonexistent")
        assert result is None

    def test_set_overwrite(self, state_store):
        """Test overwriting an existing key."""
        state_store.set("key", {"v": 1})
        state_store.set("key", {"v": 2})
        result = state_store.get("key")
        assert result["v"] == 2

    def test_get_all(self, state_store):
        """Test getting all entries."""
        state_store.set("k1", {"a": 1})
        state_store.set("k2", {"b": 2})
        all_data = state_store.get_all()
        assert len(all_data) == 2
        assert "k1" in all_data
        assert "k2" in all_data

    def test_delete(self, state_store):
        """Test deleting a key."""
        state_store.set("key", {"v": 1})
        state_store.delete("key")
        assert state_store.get("key") is None

    def test_clear(self, state_store):
        """Test clearing all state."""
        state_store.set("k1", {"a": 1})
        state_store.set("k2", {"b": 2})
        state_store.clear()
        assert state_store.get_all() == {}

    def test_entity_helpers(self, state_store):
        """Test entity-specific get/set."""
        state_store.set_entity("players", "bumrah", {
            "score": 0.95, "n": 3, "avg_views": 1285, "trend": "rising"
        })
        result = state_store.get_entity("players", "bumrah")
        assert result["score"] == 0.95
        assert result["n"] == 3

    def test_entity_default(self, state_store):
        """Test getting a non-existent entity returns defaults."""
        result = state_store.get_entity("players", "unknown")
        assert result["score"] == 0.5
        assert result["n"] == 0

    def test_get_all_entities(self, state_store):
        """Test getting all entities of a type."""
        state_store.set_entity("players", "bumrah", {"score": 0.95, "n": 3})
        state_store.set_entity("players", "kohli", {"score": 0.80, "n": 5})
        all_players = state_store.get_all_entities("players")
        assert len(all_players) == 2
        assert "bumrah" in all_players
        assert "kohli" in all_players

    def test_count_recent(self, state_store):
        """Test counting recent uploads."""
        state_store.set_entity("players", "jadeja", {
            "score": 0.5, "n": 10, "avg_views": 200,
            "recent_uploads": [
                datetime.now(timezone.utc).timestamp(),
                (datetime.now(timezone.utc) - timedelta(days=5)).timestamp(),
                (datetime.now(timezone.utc) - timedelta(days=20)).timestamp(),
            ]
        })
        count = state_store.count_recent("players", "jadeja", 14)
        assert count == 2

    def test_add_recent_upload(self, state_store):
        """Test adding a recent upload timestamp."""
        state_store.set_entity("players", "gill", {"score": 0.8, "n": 2})
        state_store.add_recent_upload("players", "gill")
        entity = state_store.get_entity("players", "gill")
        assert "recent_uploads" in entity
        assert len(entity["recent_uploads"]) == 1


class TestProcessedEventLog:
    """Tests for ProcessedEventLog."""

    def test_mark_and_check(self, processed_log):
        """Test marking an event as processed."""
        assert not processed_log.is_processed("evt1", "format")
        processed_log.mark_processed("evt1", "format")
        assert processed_log.is_processed("evt1", "format")

    def test_different_learners(self, processed_log):
        """Test that different learners are tracked independently."""
        processed_log.mark_processed("evt1", "format")
        assert processed_log.is_processed("evt1", "format")
        assert not processed_log.is_processed("evt1", "entity")

    def test_clear_learner(self, processed_log):
        """Test clearing a specific learner."""
        processed_log.mark_processed("evt1", "format")
        processed_log.mark_processed("evt1", "entity")
        cleared = processed_log.clear_learner("format")
        assert cleared == 1
        assert not processed_log.is_processed("evt1", "format")
        assert processed_log.is_processed("evt1", "entity")

    def test_clear_all(self, processed_log):
        """Test clearing all processed events."""
        processed_log.mark_processed("evt1", "format")
        processed_log.mark_processed("evt2", "entity")
        cleared = processed_log.clear_all()
        assert cleared == 2
        assert not processed_log.is_processed("evt1", "format")
        assert not processed_log.is_processed("evt2", "entity")

    def test_count(self, processed_log):
        """Test counting processed events."""
        processed_log.mark_processed("evt1", "format")
        processed_log.mark_processed("evt1", "entity")
        processed_log.mark_processed("evt2", "format")
        assert processed_log.count() == 3
        assert processed_log.count("format") == 2
        assert processed_log.count("entity") == 1

    def test_get_learners(self, processed_log):
        """Test getting list of learners."""
        processed_log.mark_processed("evt1", "format")
        processed_log.mark_processed("evt1", "entity")
        learners = processed_log.get_learners()
        assert set(learners) == {"format", "entity"}


class TestFormatLearner:
    """Tests for FormatLearner."""

    def test_process_metrics(self, state_store):
        """Test processing metrics updates hook scores."""
        learner = FormatLearner(state_store, baseline_views=290)
        learner.process_metrics("clip1", {
            "views": 552,
            "hook_type": "reaction",
            "title_pattern": {"length": 65, "is_hinglish": True}
        })
        scores = learner.get_scores()
        assert "reaction" in scores["hook_types"]
        assert scores["hook_types"]["reaction"]["n"] == 1

    def test_bayesian_ema(self, state_store):
        """Test that scores update via EMA."""
        learner = FormatLearner(state_store, baseline_views=290, alpha=0.15)
        learner.process_metrics("c1", {"views": 580, "hook_type": "shock"})
        score1 = learner.get_hook_score("shock")
        learner.process_metrics("c2", {"views": 580, "hook_type": "shock"})
        score2 = learner.get_hook_score("shock")
        assert score2 > score1

    def test_select_hook_type(self, state_store):
        """Test Thompson Sampling hook selection."""
        learner = FormatLearner(state_store, baseline_views=290)
        learner.process_metrics("c1", {"views": 500, "hook_type": "reaction"})
        selected = learner.select_hook_type()
        assert selected in FormatLearner.HOOK_TYPES

    def test_title_score(self, state_store):
        """Test composite title score."""
        learner = FormatLearner(state_store, baseline_views=290)
        learner.process_metrics("c1", {
            "views": 400,
            "hook_type": "shock",
            "title_pattern": {"length": 70, "is_hinglish": True, "has_pipe": True}
        })
        score = learner.get_title_score({
            "length": 70, "is_hinglish": True, "has_pipe": True
        })
        assert 0.0 <= score <= 1.0

    def test_zero_views_ignored(self, state_store):
        """Test that zero views don't update scores."""
        learner = FormatLearner(state_store, baseline_views=290)
        learner.process_metrics("c1", {"views": 0, "hook_type": "shock"})
        scores = learner.get_scores()
        assert scores["hook_types"] == {}


class TestEntityLearner:
    """Tests for EntityLearner."""

    def test_process_metrics(self, state_store):
        """Test processing metrics updates entity scores."""
        learner = EntityLearner(state_store, baseline_views=290)
        learner.process_metrics("clip1", {
            "views": 1285,
            "player_tags": ["bumrah"],
            "team_tags": ["mi"]
        })
        bumrah = state_store.get_entity("players", "bumrah")
        assert bumrah["n"] == 1
        assert bumrah["avg_views"] == 1285

    def test_confidence_weighted_ema(self, state_store):
        """Test that more data = slower learning."""
        learner = EntityLearner(state_store, baseline_views=290)
        for i in range(10):
            learner.process_metrics(f"c{i}", {
                "views": 300,
                "player_tags": ["kohli"]
            })
        entity = state_store.get_entity("players", "kohli")
        assert entity["n"] == 10

    def test_fatigue_penalty(self, state_store):
        """Test fatigue penalty for over-covered entities."""
        learner = EntityLearner(state_store, baseline_views=290)
        state_store.set_entity("players", "jadeja", {
            "score": 0.40, "n": 17, "avg_views": 233,
            "recent_uploads": [datetime.now(timezone.utc).timestamp()] * 15
        })
        penalty = learner.fatigue_penalty("players", "jadeja", 14)
        assert penalty > 0
        assert penalty <= 0.30

    def test_no_fatigue_for_good_performance(self, state_store):
        """Test no fatigue penalty for well-performing entities."""
        learner = EntityLearner(state_store, baseline_views=290)
        state_store.set_entity("players", "bumrah", {
            "score": 0.95, "n": 15, "avg_views": 1285,
            "recent_uploads": [datetime.now(timezone.utc).timestamp()] * 15
        })
        penalty = learner.fatigue_penalty("players", "bumrah", 14)
        assert penalty == 0.0

    def test_get_entity_score(self, state_store):
        """Test composite entity score."""
        learner = EntityLearner(state_store, baseline_views=290)
        state_store.set_entity("players", "bumrah", {"score": 0.95, "n": 3})
        state_store.set_entity("teams", "mi", {"score": 0.82, "n": 14})
        score = learner.get_entity_score(["bumrah"], ["mi"], None)
        assert 0.0 <= score <= 1.0

    def test_extract_entities(self, state_store):
        """Test entity extraction from title."""
        learner = EntityLearner(state_store, baseline_views=290)
        found = learner.extract_entities("Bumrah ne maara yorker! MI vs CSK")
        assert "bumrah" in found["players"]
        assert "mi" in found["teams"]
        assert "csk" in found["teams"]

    def test_get_top_entities(self, state_store):
        """Test getting top-performing entities."""
        learner = EntityLearner(state_store, baseline_views=290)
        state_store.set_entity("players", "bumrah", {"score": 0.95, "n": 3, "avg_views": 1285, "trend": "rising"})
        state_store.set_entity("players", "kohli", {"score": 0.80, "n": 5, "avg_views": 800, "trend": "stable"})
        top = learner.get_top_entities("players", limit=2)
        assert len(top) == 2
        assert top[0]["name"] == "bumrah"


class TestTrendEngine:
    """Tests for TrendEngine."""

    def test_ingest(self, state_store):
        """Test trend ingestion."""
        engine = TrendEngine(state_store)
        trend = TrendInput(
            trend_id="ipl_final_2026",
            source="manual",
            query="IPL 2026 Final",
            trend_score=0.95,
            velocity=0.8,
            category="match",
            entities={"teams": ["rcb", "gt"]},
            half_life_hours=72.0
        )
        engine.ingest(trend)
        active = engine.get_active(min_score=0.1)
        assert len(active) == 1
        assert active[0]["trend_id"] == "ipl_final_2026"

    def test_decay(self, state_store):
        """Test trend decay over time."""
        engine = TrendEngine(state_store)
        old_time = (datetime.now(timezone.utc) - timedelta(hours=72)).isoformat()
        state_store.set("trend_scores", {
            "active_trends": {
                "test_trend": {
                    "initial_score": 0.95,
                    "current_score": 0.95,
                    "created_at": old_time,
                    "half_life_hours": 72.0,
                    "velocity": 0.0,
                    "entities": {},
                    "query": "test",
                    "source": "test",
                    "category": "match"
                }
            }
        })
        engine.decay_all()
        active = engine.get_active(min_score=0.0)
        assert len(active) == 1
        assert active[0]["score"] < 0.50

    def test_get_trend_for_entities(self, state_store):
        """Test getting trend score for entities."""
        engine = TrendEngine(state_store)
        trend = TrendInput(
            trend_id="bumrah_trend",
            source="manual",
            query="Bumrah",
            trend_score=0.88,
            velocity=0.5,
            category="player",
            entities={"players": ["bumrah"]},
            half_life_hours=48.0
        )
        engine.ingest(trend)
        score = engine.get_trend_for_entities(["bumrah"], [])
        assert score >= 0.88

    def test_no_match_returns_minimum(self, state_store):
        """Test that no matching trend returns minimum score."""
        engine = TrendEngine(state_store)
        score = engine.get_trend_for_entities(["unknown_player"], [])
        assert score == 0.1

    def test_prune_expired(self, state_store):
        """Test pruning low-score trends."""
        engine = TrendEngine(state_store)
        state_store.set("trend_scores", {
            "active_trends": {
                "old_trend": {
                    "current_score": 0.02,
                    "initial_score": 0.5,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "half_life_hours": 72.0,
                    "velocity": 0.0,
                    "entities": {},
                    "query": "old",
                    "source": "test",
                    "category": "match"
                }
            }
        })
        pruned = engine.prune_expired()
        assert pruned == 1
        active = engine.get_active(min_score=0.0)
        assert len(active) == 0


class TestTimingLearner:
    """Tests for TimingLearner."""

    def test_process_metrics(self, state_store):
        """Test processing timing metrics."""
        learner = TimingLearner(state_store, baseline_views=290)
        learner.process_metrics("c1", {
            "views": 400,
            "publish_hour": 19,
            "publish_dow": 5
        })
        hour_data = state_store.get("timing:hour:19")
        assert hour_data is not None
        assert hour_data["n"] == 1

    def test_best_hour(self, state_store):
        """Test getting best upload hour."""
        learner = TimingLearner(state_store, baseline_views=290)
        learner.process_metrics("c1", {"views": 500, "publish_hour": 19})
        learner.process_metrics("c2", {"views": 200, "publish_hour": 10})
        best = learner.best_hour()
        assert best == 19

    def test_best_day(self, state_store):
        """Test getting best upload day."""
        learner = TimingLearner(state_store, baseline_views=290)
        learner.process_metrics("c1", {"views": 500, "publish_dow": 5})
        learner.process_metrics("c2", {"views": 200, "publish_dow": 1})
        best = learner.best_day()
        assert best == "saturday"

    def test_schedule_recommendation(self, state_store):
        """Test schedule recommendation generation."""
        learner = TimingLearner(state_store, baseline_views=290)
        learner.process_metrics("c1", {"views": 500, "publish_hour": 19, "publish_dow": 5})
        rec = learner.get_schedule_recommendation()
        assert "hour" in rec
        assert "day" in rec
        assert "confidence" in rec


class TestDurationLearner:
    """Tests for DurationLearner."""

    def test_process_metrics(self, state_store):
        """Test processing duration metrics."""
        learner = DurationLearner(state_store, baseline_views=290)
        learner.process_metrics("c1", {
            "views": 400,
            "duration_seconds": 28,
            "completion_rate": 0.65
        })
        scores = learner.get_scores()
        assert "25-34s" in scores["buckets"]
        assert scores["buckets"]["25-34s"]["n"] == 1

    def test_select_duration(self, state_store):
        """Test UCB1 duration selection."""
        learner = DurationLearner(state_store, baseline_views=290)
        learner.process_metrics("c1", {"views": 500, "duration_seconds": 30, "completion_rate": 0.7})
        duration = learner.select_duration()
        assert 15 <= duration <= 59

    def test_bucket_assignment(self, state_store):
        """Test correct bucket assignment."""
        learner = DurationLearner(state_store, baseline_views=290)
        learner.process_metrics("c1", {"views": 300, "duration_seconds": 20, "completion_rate": 0.8})
        learner.process_metrics("c2", {"views": 400, "duration_seconds": 35, "completion_rate": 0.6})
        scores = learner.get_scores()
        assert "15-24s" in scores["buckets"]
        assert "35-44s" in scores["buckets"]


class TestLearningDispatcher:
    """Tests for LearningDispatcher."""

    def test_dispatch_metrics(self, state_store, processed_log):
        """Test dispatching metrics_received events."""
        fmt = FormatLearner(state_store, baseline_views=290)
        ent = EntityLearner(state_store, baseline_views=290)
        tim = TimingLearner(state_store, baseline_views=290)
        dur = DurationLearner(state_store, baseline_views=290)
        trend = TrendEngine(state_store)

        dispatcher = LearningDispatcher(fmt, ent, tim, dur, trend, processed_log)

        event = ClipEvent(
            event_id="evt1",
            clip_id="clip1",
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=EventType.metrics_received,
            payload_json=json.dumps({
                "views": 500,
                "hook_type": "reaction",
                "player_tags": ["bumrah"],
                "team_tags": ["mi"],
                "publish_hour": 19,
                "publish_dow": 5,
                "duration_seconds": 28,
                "completion_rate": 0.7
            })
        )

        updated = dispatcher.dispatch(event)
        assert updated == 4

    def test_idempotency(self, state_store, processed_log):
        """Test that dispatching same event twice doesn't double-process."""
        fmt = FormatLearner(state_store, baseline_views=290)
        ent = EntityLearner(state_store, baseline_views=290)
        tim = TimingLearner(state_store, baseline_views=290)
        dur = DurationLearner(state_store, baseline_views=290)
        trend = TrendEngine(state_store)

        dispatcher = LearningDispatcher(fmt, ent, tim, dur, trend, processed_log)

        event = ClipEvent(
            event_id="evt1",
            clip_id="clip1",
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=EventType.metrics_received,
            payload_json=json.dumps({
                "views": 500,
                "hook_type": "shock",
                "player_tags": ["kohli"]
            })
        )

        dispatcher.dispatch(event)
        score1 = fmt.get_hook_score("shock")

        dispatcher.dispatch(event)
        score2 = fmt.get_hook_score("shock")

        assert score1 == score2

    def test_dispatch_trend(self, state_store, processed_log):
        """Test dispatching trend_ingested events."""
        fmt = FormatLearner(state_store, baseline_views=290)
        ent = EntityLearner(state_store, baseline_views=290)
        tim = TimingLearner(state_store, baseline_views=290)
        dur = DurationLearner(state_store, baseline_views=290)
        trend = TrendEngine(state_store)

        dispatcher = LearningDispatcher(fmt, ent, tim, dur, trend, processed_log)

        event = ClipEvent(
            event_id="evt2",
            clip_id="trend1",
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=EventType.trend_ingested,
            payload_json=json.dumps({
                "trend_id": "ipl_final",
                "source": "manual",
                "query": "IPL Final",
                "trend_score": 0.90,
                "velocity": 0.7,
                "category": "match",
                "entities": {"teams": ["rcb", "gt"]},
                "half_life_hours": 72.0
            })
        )

        updated = dispatcher.dispatch(event)
        assert updated == 1
        active = trend.get_active(min_score=0.1)
        assert len(active) == 1


class TestCricketScorer:
    """Tests for CricketScorer."""

    def test_score_candidate(self, state_store):
        """Test scoring a candidate clip."""
        fmt = FormatLearner(state_store, baseline_views=290)
        ent = EntityLearner(state_store, baseline_views=290)
        trend = TrendEngine(state_store)
        tim = TimingLearner(state_store, baseline_views=290)
        dur = DurationLearner(state_store, baseline_views=290)

        scorer = CricketScorer(fmt, ent, trend, tim, dur, state_store)

        candidate = {
            "detected_players": ["bumrah"],
            "detected_teams": ["mi"],
            "detected_hook_type": "shock",
            "weighted_score": 8.0,
            "title_pattern": {"length": 65, "is_hinglish": True}
        }

        score = scorer.score_candidate(candidate)
        assert 0.0 <= score <= 1.0

    def test_score_many(self, state_store):
        """Test scoring multiple candidates."""
        fmt = FormatLearner(state_store, baseline_views=290)
        ent = EntityLearner(state_store, baseline_views=290)
        trend = TrendEngine(state_store)
        tim = TimingLearner(state_store, baseline_views=290)
        dur = DurationLearner(state_store, baseline_views=290)

        scorer = CricketScorer(fmt, ent, trend, tim, dur, state_store)

        candidates = [
            {"detected_players": ["bumrah"], "detected_teams": ["mi"], "weighted_score": 8.0},
            {"detected_players": ["kohli"], "detected_teams": ["rcb"], "weighted_score": 7.0}
        ]

        scored = scorer.score_many(candidates)
        assert len(scored) == 2
        assert scored[0][1] >= scored[1][1]


class TestCricketRecommendationEngine:
    """Tests for CricketRecommendationEngine."""

    def test_generate_all(self, state_store):
        """Test generating recommendations."""
        fmt = FormatLearner(state_store, baseline_views=290)
        ent = EntityLearner(state_store, baseline_views=290)
        trend = TrendEngine(state_store)
        tim = TimingLearner(state_store, baseline_views=290)
        dur = DurationLearner(state_store, baseline_views=290)

        state_store.set_entity("players", "bumrah", {
            "score": 0.95, "n": 3, "avg_views": 1285, "trend": "rising"
        })

        engine = CricketRecommendationEngine(
            fmt, ent, trend, tim, dur, state_store, baseline_views=290
        )

        recs = engine.generate_all()
        assert len(recs) > 0
        assert any(r.category == "player" for r in recs)

    def test_player_recommendation(self, state_store):
        """Test player-specific recommendations."""
        fmt = FormatLearner(state_store, baseline_views=290)
        ent = EntityLearner(state_store, baseline_views=290)
        trend = TrendEngine(state_store)
        tim = TimingLearner(state_store, baseline_views=290)
        dur = DurationLearner(state_store, baseline_views=290)

        state_store.set_entity("players", "bumrah", {
            "score": 0.95, "n": 3, "avg_views": 1285, "trend": "rising"
        })

        engine = CricketRecommendationEngine(
            fmt, ent, trend, tim, dur, state_store, baseline_views=290
        )

        recs = engine.generate_all()
        bumrah_recs = [r for r in recs if "bumrah" in r.recommendation.lower()]
        assert len(bumrah_recs) > 0
        assert bumrah_recs[0].priority == "high"
