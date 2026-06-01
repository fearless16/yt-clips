"""Tests for Learner, PolicyUpdater, PreferenceEngine, ReplayEngine, Analytics, SEOGenerator."""

import json
import pytest

from automation.memory.event_models import EventType, ClipEvent
from automation.memory.decision_store import DecisionStore, LearnedStateStore
from automation.memory.feedback_schema import FeedbackPayload
from automation.learner.learner import Learner
from automation.learner.policy_updater import PolicyUpdater
from automation.learner.preference_engine import PreferenceEngine
from automation.learner.replay import ReplayEngine
from automation.seo.analytics import Analytics
from automation.seo.seo import SEOGenerator


# =============================================================================
# Learner Tests
# =============================================================================

class TestLearner:
    def test_process_feedback_stores_event_in_decision_store(self):
        store = DecisionStore()
        state = LearnedStateStore()
        learner = Learner(store, state)
        payload = FeedbackPayload(
            clip_id="clip-001",
            rating=5,
            feedback_type="like",
            details=None,
            timestamp="2025-01-01T00:00:00Z",
        )
        learner.process_feedback(payload)
        assert store.count() == 1
        event = store.get_all_events()[0]
        assert event.event_type == EventType.metrics_received
        assert event.clip_id == "clip-001"

    def test_process_feedback_updates_learned_state_hook_weight(self):
        store = DecisionStore()
        state = LearnedStateStore()
        learner = Learner(store, state)
        payload = FeedbackPayload(
            clip_id="clip-001",
            rating=5,
            feedback_type="like",
            details=None,
            timestamp="2025-01-01T00:00:00Z",
        )
        learner.process_feedback(payload)
        hook = learner.get_state("hook_weight")
        assert hook is not None
        assert float(hook) == pytest.approx(1.1)

    def test_process_manual_override_stores_manual_override_event(self):
        store = DecisionStore()
        state = LearnedStateStore()
        learner = Learner(store, state)
        learner.process_manual_override("clip-001", "keep")
        assert store.count() == 1
        event = store.get_all_events()[0]
        assert event.event_type == EventType.manual_override
        assert event.clip_id == "clip-001"

    def test_process_manual_override_increases_weights_more_than_regular_feedback(self):
        store = DecisionStore()
        state = LearnedStateStore()
        learner = Learner(store, state)

        fb = FeedbackPayload(
            clip_id="clip-001",
            rating=5,
            feedback_type="like",
            details=None,
            timestamp="2025-01-01T00:00:00Z",
        )
        learner.process_feedback(fb)
        hook_after_feedback = float(learner.get_state("hook_weight"))

        store2 = DecisionStore()
        state2 = LearnedStateStore()
        learner2 = Learner(store2, state2)
        learner2.process_manual_override("clip-001", "keep")
        hook_after_override = float(learner2.get_state("hook_weight"))

        assert (hook_after_override - 1.0) > (hook_after_feedback - 1.0)

    def test_get_state_reads_from_learned_state(self):
        store = DecisionStore()
        state = LearnedStateStore()
        learner = Learner(store, state)
        state.set("test_key", '"test_value"')
        result = learner.get_state("test_key")
        assert result == '"test_value"'

    def test_get_state_returns_none_for_missing_key(self):
        store = DecisionStore()
        state = LearnedStateStore()
        learner = Learner(store, state)
        assert learner.get_state("nonexistent") is None

    def test_reset_learned_state_clears_all_state(self):
        store = DecisionStore()
        state = LearnedStateStore()
        learner = Learner(store, state)
        state.set("hook_weight", "1.5")
        state.set("payoff_weight", "1.3")
        learner.reset_learned_state()
        assert learner.get_state("hook_weight") is None
        assert learner.get_state("payoff_weight") is None

    def test_negative_feedback_decreases_weights(self):
        store = DecisionStore()
        state = LearnedStateStore()
        learner = Learner(store, state)

        fb = FeedbackPayload(
            clip_id="clip-001",
            rating=1,
            feedback_type="dislike",
            details=None,
            timestamp="2025-01-01T00:00:00Z",
        )
        learner.process_feedback(fb)
        hook = float(learner.get_state("hook_weight"))
        payoff = float(learner.get_state("payoff_weight"))
        assert hook == pytest.approx(0.9)
        assert payoff == pytest.approx(0.9)


# =============================================================================
# PolicyUpdater Tests
# =============================================================================

class TestPolicyUpdater:
    def test_update_from_event_metrics_received_calls_process_feedback(self):
        store = DecisionStore()
        state = LearnedStateStore()
        learner = Learner(store, state)
        updater = PolicyUpdater(learner)

        event = ClipEvent(
            event_id="e1",
            clip_id="clip-001",
            timestamp="2025-01-01T00:00:00Z",
            event_type=EventType.metrics_received,
            payload_json='{"rating": 5, "feedback_type": "like"}',
        )
        updater.update_from_event(event)

        assert store.count() >= 1
        assert float(learner.get_state("hook_weight")) == pytest.approx(1.1)

    def test_update_from_event_manual_override_calls_process_manual_override(self):
        store = DecisionStore()
        state = LearnedStateStore()
        learner = Learner(store, state)
        updater = PolicyUpdater(learner)

        event = ClipEvent(
            event_id="e2",
            clip_id="clip-001",
            timestamp="2025-01-01T00:00:00Z",
            event_type=EventType.manual_override,
            payload_json='{"override_type": "keep"}',
        )
        updater.update_from_event(event)

        assert store.count() >= 1
        assert float(learner.get_state("hook_weight")) == pytest.approx(1.3)

    def test_update_from_event_other_types_are_noop(self):
        store = DecisionStore()
        state = LearnedStateStore()
        learner = Learner(store, state)
        updater = PolicyUpdater(learner)

        event = ClipEvent(
            event_id="e3",
            clip_id="clip-001",
            timestamp="2025-01-01T00:00:00Z",
            event_type=EventType.published,
            payload_json="{}",
        )
        updater.update_from_event(event)

        assert store.count() == 0

    def test_update_from_events_processes_all_events(self):
        store = DecisionStore()
        state = LearnedStateStore()
        learner = Learner(store, state)
        updater = PolicyUpdater(learner)

        events = [
            ClipEvent("e1", "clip-001", "ts1", EventType.metrics_received, '{"rating": 5, "feedback_type": "like"}'),
            ClipEvent("e2", "clip-001", "ts2", EventType.metrics_received, '{"rating": 2, "feedback_type": "dislike"}'),
        ]
        updater.update_from_events(events)

        assert float(learner.get_state("hook_weight")) == pytest.approx(1.0)


# =============================================================================
# PreferenceEngine Tests
# =============================================================================

class TestPreferenceEngine:
    def test_compute_preferences_returns_dict_with_required_keys(self):
        store = DecisionStore()
        state = LearnedStateStore()
        learner = Learner(store, state)
        engine = PreferenceEngine(learner)

        prefs = engine.compute_preferences()
        assert "preferred_duration" in prefs
        assert "preferred_pacing" in prefs
        assert "topic_fatigue_penalty" in prefs

    def test_get_recommendation_returns_proper_shape(self):
        store = DecisionStore()
        state = LearnedStateStore()
        learner = Learner(store, state)
        engine = PreferenceEngine(learner)

        clip_data = {"clip_id": "clip-001", "duration_s": 30.0}
        rec = engine.get_recommendation(clip_data)
        assert "clip_id" in rec
        assert "recommended" in rec
        assert "reason" in rec
        assert rec["clip_id"] == "clip-001"
        assert isinstance(rec["recommended"], bool)
        assert isinstance(rec["reason"], str)

    def test_preferences_change_based_on_learned_weights(self):
        store = DecisionStore()
        state = LearnedStateStore()
        learner = Learner(store, state)
        engine = PreferenceEngine(learner)

        prefs_before = engine.compute_preferences()
        assert prefs_before["preferred_duration"] == "medium"

        learner.process_manual_override("clip-001", "keep")
        prefs_after = engine.compute_preferences()
        assert prefs_after["preferred_duration"] == "short"


# =============================================================================
# ReplayEngine Tests
# =============================================================================

class TestReplayEngine:
    def test_replay_processes_all_events_and_returns_count(self):
        store = DecisionStore()
        state = LearnedStateStore()
        learner = Learner(store, state)

        store.append_event(ClipEvent(
            "e1", "clip-001", "ts1", EventType.metrics_received,
            '{"rating": 5, "feedback_type": "like"}',
        ))
        store.append_event(ClipEvent(
            "e2", "clip-001", "ts2", EventType.metrics_received,
            '{"rating": 2, "feedback_type": "dislike"}',
        ))

        engine = ReplayEngine(store, state, learner)
        count = engine.replay()

        assert count == 2
        assert float(state.get("hook_weight")) == pytest.approx(1.0)

    def test_replay_produces_same_learned_state_as_original_processing(self):
        store = DecisionStore()
        state = LearnedStateStore()
        learner = Learner(store, state)

        fb1 = FeedbackPayload("clip-001", 5, "like", None, "ts1")
        fb2 = FeedbackPayload("clip-001", 2, "dislike", None, "ts2")
        learner.process_feedback(fb1)
        learner.process_feedback(fb2)

        expected_state = dict(state.get_all())

        state.clear()
        engine = ReplayEngine(store, state, learner)
        count = engine.replay()

        assert count == 2
        assert state.get_all() == expected_state

    def test_verify_returns_true_when_replay_matches(self):
        store = DecisionStore()
        state = LearnedStateStore()
        learner = Learner(store, state)

        fb = FeedbackPayload("clip-001", 5, "like", None, "ts1")
        learner.process_feedback(fb)

        engine = ReplayEngine(store, state, learner)
        assert engine.verify() is True


# =============================================================================
# Analytics Tests
# =============================================================================

class TestAnalytics:
    def test_get_metrics_returns_correct_dict_shape_for_clip_with_events(self):
        store = DecisionStore()
        store.append_event(ClipEvent(
            "e1", "clip-001", "ts1", EventType.metrics_received,
            '{"rating": 5, "feedback_type": "like"}',
        ))
        store.append_event(ClipEvent(
            "e2", "clip-001", "ts2", EventType.candidate_created, "{}",
        ))
        store.append_event(ClipEvent(
            "e3", "clip-002", "ts3", EventType.published, "{}",
        ))

        analytics = Analytics(store)
        metrics = analytics.get_metrics("clip-001")

        assert metrics["total_events"] == 2
        assert metrics["events_by_type"]["metrics_received"] == 1
        assert metrics["events_by_type"]["candidate_created"] == 1
        assert metrics["feedback_count"] == 1
        assert metrics["avg_rating"] == 5.0

    def test_get_metrics_returns_zero_values_for_clip_with_no_events(self):
        store = DecisionStore()
        analytics = Analytics(store)
        metrics = analytics.get_metrics("clip-nonexistent")

        assert metrics["total_events"] == 0
        assert metrics["events_by_type"] == {}
        assert metrics["feedback_count"] == 0
        assert metrics["avg_rating"] == 0.0

    def test_get_summary_returns_aggregated_stats(self):
        store = DecisionStore()
        store.append_event(ClipEvent("e1", "clip-001", "ts1", EventType.candidate_created, "{}"))
        store.append_event(ClipEvent("e2", "clip-001", "ts2", EventType.published, "{}"))
        store.append_event(ClipEvent("e3", "clip-002", "ts3", EventType.candidate_created, "{}"))
        store.append_event(ClipEvent("e4", "clip-002", "ts4", EventType.candidate_scored, '{"score": 0.8}'))

        analytics = Analytics(store)
        summary = analytics.get_summary()

        assert summary["total_clips"] == 2
        assert summary["total_events"] == 4
        assert summary["published_count"] == 1
        assert summary["avg_score"] == 0.8

    def test_get_trends_returns_dict_with_trend_data(self):
        store = DecisionStore()
        store.append_event(ClipEvent("e1", "clip-001", "ts1", EventType.candidate_created, "{}"))
        store.append_event(ClipEvent("e2", "clip-001", "ts2", EventType.published, "{}"))
        store.append_event(ClipEvent("e3", "clip-002", "ts3", EventType.candidate_created, "{}"))

        analytics = Analytics(store)
        trends = analytics.get_trends()

        assert isinstance(trends, dict)
        assert trends["candidate_created"] == 2
        assert trends["published"] == 1


# =============================================================================
# SEOGenerator Tests
# =============================================================================

class TestSEOGenerator:
    def test_generate_returns_dict_with_all_required_keys(self):
        store = DecisionStore()
        gen = SEOGenerator(store)
        clip_data = {
            "clip_id": "clip-001",
            "title": "Amazing Video Title",
            "transcript_summary": "This is a great video about something interesting.",
            "duration_s": 45.0,
        }
        result = gen.generate(clip_data)
        assert "clip_id" in result
        assert "title" in result
        assert "description" in result
        assert "tags" in result
        assert "category" in result
        assert result["clip_id"] == "clip-001"
        assert result["category"] == "Entertainment"

    def test_title_is_truncated_properly(self):
        store = DecisionStore()
        gen = SEOGenerator(store)
        long_title = "This is a really long title that should definitely be truncated to sixty characters for shorts purposes"
        clip_data = {
            "clip_id": "clip-001",
            "title": long_title,
            "transcript_summary": "Summary text.",
            "duration_s": 30.0,
        }
        result = gen.generate(clip_data)
        assert result["title"] == long_title[:60] + " - Shorts"
        assert len(result["title"]) == 60 + len(" - Shorts")

    def test_description_includes_transcript_summary(self):
        store = DecisionStore()
        gen = SEOGenerator(store)
        clip_data = {
            "clip_id": "clip-001",
            "title": "My Video",
            "transcript_summary": "This is the transcript summary content.",
            "duration_s": 30.0,
        }
        result = gen.generate(clip_data)
        assert "This is the transcript summary content." in result["description"]
        assert "#shorts #youtubeshorts" in result["description"]

    def test_tags_include_shorts_and_youtubeshorts(self):
        store = DecisionStore()
        gen = SEOGenerator(store)
        clip_data = {
            "clip_id": "clip-001",
            "title": "My Amazing Video",
            "transcript_summary": "Summary.",
            "duration_s": 30.0,
        }
        result = gen.generate(clip_data)
        assert "shorts" in result["tags"]
        assert "youtubeshorts" in result["tags"]
        assert "viral" in result["tags"]

    def test_tags_include_words_from_title(self):
        store = DecisionStore()
        gen = SEOGenerator(store)
        clip_data = {
            "clip_id": "clip-001",
            "title": "Amazing Funny Viral Clip",
            "transcript_summary": "Summary.",
            "duration_s": 30.0,
        }
        result = gen.generate(clip_data)
        assert "Amazing" in result["tags"]
        assert "Funny" in result["tags"]
        assert "Viral" in result["tags"]
        assert len([t for t in result["tags"] if t not in ("shorts", "youtubeshorts", "viral")]) == 3

    def test_generate_batch_returns_list_of_results(self):
        store = DecisionStore()
        gen = SEOGenerator(store)
        clips = [
            {"clip_id": "clip-001", "title": "First", "transcript_summary": "Sum1.", "duration_s": 30.0},
            {"clip_id": "clip-002", "title": "Second", "transcript_summary": "Sum2.", "duration_s": 60.0},
        ]
        results = gen.generate_batch(clips)
        assert len(results) == 2
        assert results[0]["clip_id"] == "clip-001"
        assert results[1]["clip_id"] == "clip-002"

    def test_enhance_with_analytics_adds_tags_when_analytics_provided(self):
        store = DecisionStore()
        from automation.memory.event_models import ClipEvent
        e1 = ClipEvent(
            event_id="evt-001", clip_id="clip-001",
            timestamp="2025-01-01T00:00:00Z",
            event_type=EventType.candidate_scored,
            payload_json='{"score": 0.85}',
        )
        e2 = ClipEvent(
            event_id="evt-002", clip_id="clip-001",
            timestamp="2025-01-01T00:00:00Z",
            event_type=EventType.published,
        )
        store.append_event(e1)
        store.append_event(e2)
        analytics = Analytics(store)
        gen = SEOGenerator(store, analytics=analytics)
        clip_data = {
            "clip_id": "clip-001",
            "title": "Test Video",
            "transcript_summary": "Summary.",
            "duration_s": 30.0,
        }
        result = gen.enhance_with_analytics(clip_data)
        assert "highly_rated" in result["tags"]

    def test_enhance_with_analytics_does_not_add_tags_without_analytics(self):
        store = DecisionStore()
        gen = SEOGenerator(store)
        clip_data = {
            "clip_id": "clip-001",
            "title": "Test Video",
            "transcript_summary": "Summary.",
            "duration_s": 30.0,
        }
        result = gen.enhance_with_analytics(clip_data)
        assert "highly_rated" not in result["tags"]
        assert "popular_channel" not in result["tags"]
