"""Tests for the core data layer — event models, config, feedback schema, decision store."""

import pytest
import json
from dataclasses import fields

from automation.memory.event_models import EventType, ClipEvent, LearnedStateEntry
from automation.config import load, get
from automation.memory.feedback_schema import (
    FeedbackPayload,
    ValidationResult,
    validate_feedback,
    validate_feedback_payload,
)
from automation.memory.decision_store import DecisionStore, LearnedStateStore


# ── EventType ──────────────────────────────────────────────────────────────────


class TestEventType:
    def test_has_all_required_members(self):
        expected = {
            "candidate_created",
            "candidate_scored",
            "candidate_ranked",
            "selected",
            "rejected",
            "exported",
            "published",
            "metrics_received",
            "manual_override",
            "validation_failed",
            "infra_failed",
            "deferred",
            "policy_updated",
            "trend_ingested",
            "trend_decayed",
        }
        actual = {e.value for e in EventType}
        assert actual == expected
        assert len(EventType) == 15

    def test_no_extra_members(self):
        expected_count = 15
        assert len(EventType) == expected_count


# ── ClipEvent ─────────────────────────────────────────────────────────────────


class TestClipEvent:
    def test_can_be_constructed_with_all_fields(self):
        event = ClipEvent(
            event_id="evt-001",
            clip_id="clip-abc",
            timestamp="2025-01-01T00:00:00Z",
            event_type=EventType.candidate_created,
            payload_json='{"source": "youtube"}',
        )
        assert event.event_id == "evt-001"
        assert event.clip_id == "clip-abc"
        assert event.timestamp == "2025-01-01T00:00:00Z"
        assert event.event_type == EventType.candidate_created
        assert event.payload_json == '{"source": "youtube"}'

    def test_requires_event_type_to_be_eventtype_enum(self):
        with pytest.raises(TypeError):
            ClipEvent(
                event_id="evt-002",
                clip_id="clip-def",
                timestamp="2025-01-01T00:00:00Z",
                event_type="candidate_created",
                payload_json="{}",
            )

    def test_event_type_must_be_valid_enum_member(self):
        with pytest.raises(ValueError):
            EventType("invalid_type")

    def test_payload_json_defaults_to_empty_dict_json(self):
        event = ClipEvent(
            event_id="evt-003",
            clip_id="clip-ghi",
            timestamp="2025-01-01T00:00:00Z",
            event_type=EventType.selected,
        )
        assert event.payload_json == "{}"

    def test_all_field_types_are_strings(self):
        event = ClipEvent(
            event_id="evt-004",
            clip_id="clip-jkl",
            timestamp="2025-01-01T00:00:00Z",
            event_type=EventType.rejected,
            payload_json='{"reason": "low_score"}',
        )
        assert isinstance(event.event_id, str)
        assert isinstance(event.clip_id, str)
        assert isinstance(event.timestamp, str)
        assert isinstance(event.payload_json, str)


# ── LearnedStateEntry ─────────────────────────────────────────────────────────


class TestLearnedStateEntry:
    def test_constructed_correctly(self):
        entry = LearnedStateEntry(
            state_key="clip:abc:score",
            value_json='{"score": 0.95}',
            derived_from_event_id="evt-001",
            updated_at="2025-01-01T00:00:00Z",
        )
        assert entry.state_key == "clip:abc:score"
        assert entry.value_json == '{"score": 0.95}'
        assert entry.derived_from_event_id == "evt-001"
        assert entry.updated_at == "2025-01-01T00:00:00Z"

    def test_allows_none_derived_from_event_id(self):
        entry = LearnedStateEntry(
            state_key="clip:abc:score",
            value_json='{"score": 0.95}',
            derived_from_event_id=None,
            updated_at="2025-01-01T00:00:00Z",
        )
        assert entry.derived_from_event_id is None

    def test_state_key_and_value_json_are_strings(self):
        entry = LearnedStateEntry(
            state_key="clip:abc:score",
            value_json='{"score": 0.95}',
            derived_from_event_id=None,
            updated_at="2025-01-01T00:00:00Z",
        )
        assert isinstance(entry.state_key, str)
        assert isinstance(entry.value_json, str)
        assert isinstance(entry.updated_at, str)


# ── Config ─────────────────────────────────────────────────────────────────────


class TestConfigLoad:
    def test_load_returns_dict(self, tmp_path):
        d = tmp_path / "sub"
        d.mkdir()
        p = d / "config.yaml"
        p.write_text("key: value\nnested:\n  inner: 42\n")
        cfg = load(path=str(p))
        assert isinstance(cfg, dict)
        assert cfg["key"] == "value"

    def test_load_missing_file_returns_empty_dict(self):
        cfg = load(path="/tmp/nonexistent_config_file_xyz.yaml")
        assert cfg == {}

    def test_load_defaults_to_config_yaml(self):
        # Should not crash; config.yaml exists in cwd
        cfg = load()
        assert isinstance(cfg, dict)


class TestConfigGet:
    def test_get_with_dot_notation_returns_value(self, tmp_path):
        d = tmp_path / "sub"
        d.mkdir()
        p = d / "config.yaml"
        p.write_text("paths:\n  input: /data/in\n  output: /data/out\n")
        cfg = load(path=str(p))
        val = get(cfg, "paths.input")
        assert val == "/data/in"

    def test_get_with_missing_key_returns_default(self):
        cfg = {"paths": {"input": "/data/in"}}
        val = get(cfg, "paths.output", default="/tmp/fallback")
        assert val == "/tmp/fallback"

    def test_get_returns_none_when_no_default_and_key_missing(self):
        cfg = {"paths": {"input": "/data/in"}}
        val = get(cfg, "paths.output")
        assert val is None

    def test_get_nested_missing_parent_returns_default(self):
        cfg = {"paths": {"input": "/data/in"}}
        val = get(cfg, "nonexistent.deep.key", default="fallback")
        assert val == "fallback"

    def test_get_empty_config_returns_default(self):
        val = get({}, "anything", default="fallback")
        assert val == "fallback"


# ── Feedback Schema ────────────────────────────────────────────────────────────


class TestFeedbackPayload:
    def test_valid_feedback_dict_passes_validation(self):
        payload = {
            "clip_id": "clip-001",
            "rating": 5,
            "feedback_type": "like",
            "details": {"source": "web"},
            "timestamp": "2025-01-01T00:00:00Z",
        }
        result = validate_feedback_payload(payload)
        assert isinstance(result, FeedbackPayload)
        assert result.clip_id == "clip-001"
        assert result.rating == 5
        assert result.feedback_type == "like"
        assert result.details == {"source": "web"}
        assert result.timestamp == "2025-01-01T00:00:00Z"

    def test_valid_feedback_without_details(self):
        payload = {
            "clip_id": "clip-002",
            "rating": 3,
            "feedback_type": "skip",
            "timestamp": "2025-01-01T00:00:00Z",
        }
        result = validate_feedback_payload(payload)
        assert result.details is None

    def test_missing_clip_id_raises_valueerror(self):
        payload = {
            "rating": 4,
            "feedback_type": "like",
            "timestamp": "2025-01-01T00:00:00Z",
        }
        with pytest.raises(ValueError, match="clip_id"):
            validate_feedback_payload(payload)

    def test_rating_0_raises_valueerror(self):
        payload = {
            "clip_id": "clip-003",
            "rating": 0,
            "feedback_type": "like",
            "timestamp": "2025-01-01T00:00:00Z",
        }
        with pytest.raises(ValueError, match="rating"):
            validate_feedback_payload(payload)

    def test_rating_5_works(self):
        payload = {
            "clip_id": "clip-004",
            "rating": 5,
            "feedback_type": "dislike",
            "timestamp": "2025-01-01T00:00:00Z",
        }
        result = validate_feedback_payload(payload)
        assert result.rating == 5

    def test_rating_6_raises_valueerror(self):
        payload = {
            "clip_id": "clip-005",
            "rating": 6,
            "feedback_type": "like",
            "timestamp": "2025-01-01T00:00:00Z",
        }
        with pytest.raises(ValueError, match="rating"):
            validate_feedback_payload(payload)

    def test_invalid_feedback_type_raises_valueerror(self):
        payload = {
            "clip_id": "clip-006",
            "rating": 4,
            "feedback_type": "invalid_type",
            "timestamp": "2025-01-01T00:00:00Z",
        }
        with pytest.raises(ValueError, match="feedback_type"):
            validate_feedback_payload(payload)

    def test_extra_unknown_keys_raises_valueerror(self):
        payload = {
            "clip_id": "clip-007",
            "rating": 4,
            "feedback_type": "like",
            "timestamp": "2025-01-01T00:00:00Z",
            "unknown_key": "should_not_be_here",
        }
        with pytest.raises(ValueError, match="unknown"):
            validate_feedback_payload(payload)

    def test_empty_dict_raises_valueerror(self):
        with pytest.raises(ValueError, match="clip_id"):
            validate_feedback_payload({})


class TestValidateFeedback:
    def test_valid_returns_success_true(self):
        payload = {
            "clip_id": "clip-001",
            "rating": 4,
            "feedback_type": "watch_full",
            "timestamp": "2025-01-01T00:00:00Z",
        }
        result = validate_feedback(payload)
        assert result.success is True
        assert isinstance(result.feedback, FeedbackPayload)
        assert result.error is None

    def test_valid_returns_success_true_for_comment(self):
        payload = {
            "clip_id": "clip-002",
            "rating": 5,
            "feedback_type": "comment",
            "timestamp": "2025-01-01T00:00:00Z",
        }
        result = validate_feedback(payload)
        assert result.success is True

    def test_valid_returns_success_true_for_share(self):
        payload = {
            "clip_id": "clip-003",
            "rating": 5,
            "feedback_type": "share",
            "timestamp": "2025-01-01T00:00:00Z",
        }
        result = validate_feedback(payload)
        assert result.success is True

    def test_invalid_returns_success_false_and_error(self):
        payload = {
            "clip_id": "clip-004",
            "rating": 7,
            "feedback_type": "like",
            "timestamp": "2025-01-01T00:00:00Z",
        }
        result = validate_feedback(payload)
        assert result.success is False
        assert result.feedback is None
        assert isinstance(result.error, str)
        assert "rating" in result.error.lower()

    def test_missing_clip_id_returns_failure(self):
        payload = {
            "rating": 4,
            "feedback_type": "like",
            "timestamp": "2025-01-01T00:00:00Z",
        }
        result = validate_feedback(payload)
        assert result.success is False
        assert result.feedback is None
        assert isinstance(result.error, str)


# ── DecisionStore ──────────────────────────────────────────────────────────────


class TestDecisionStore:
    def test_starts_empty(self):
        store = DecisionStore()
        assert store.count() == 0
        assert store.get_all_events() == []

    def test_append_event_raises_typeerror_for_non_clipevent(self):
        store = DecisionStore()
        with pytest.raises(TypeError, match="ClipEvent"):
            store.append_event("not_a_clip_event")
        with pytest.raises(TypeError, match="ClipEvent"):
            store.append_event(123)
        with pytest.raises(TypeError, match="ClipEvent"):
            store.append_event(None)

    def test_append_event_stores_event_and_count_increases(self):
        store = DecisionStore()
        event = ClipEvent(
            event_id="evt-001",
            clip_id="clip-abc",
            timestamp="2025-01-01T00:00:00Z",
            event_type=EventType.candidate_created,
            payload_json="{}",
        )
        store.append_event(event)
        assert store.count() == 1
        event2 = ClipEvent(
            event_id="evt-002",
            clip_id="clip-def",
            timestamp="2025-01-01T00:00:00Z",
            event_type=EventType.selected,
            payload_json="{}",
        )
        store.append_event(event2)
        assert store.count() == 2

    def test_get_events_filters_by_clip_id(self):
        store = DecisionStore()
        e1 = ClipEvent(
            event_id="evt-001",
            clip_id="clip-abc",
            timestamp="2025-01-01T00:00:00Z",
            event_type=EventType.candidate_created,
        )
        e2 = ClipEvent(
            event_id="evt-002",
            clip_id="clip-def",
            timestamp="2025-01-01T00:00:00Z",
            event_type=EventType.candidate_created,
        )
        e3 = ClipEvent(
            event_id="evt-003",
            clip_id="clip-abc",
            timestamp="2025-01-02T00:00:00Z",
            event_type=EventType.selected,
        )
        store.append_event(e1)
        store.append_event(e2)
        store.append_event(e3)

        results = store.get_events(clip_id="clip-abc")
        assert len(results) == 2
        assert all(e.clip_id == "clip-abc" for e in results)

    def test_get_events_filters_by_event_type(self):
        store = DecisionStore()
        e1 = ClipEvent(
            event_id="evt-001",
            clip_id="clip-abc",
            timestamp="2025-01-01T00:00:00Z",
            event_type=EventType.candidate_created,
        )
        e2 = ClipEvent(
            event_id="evt-002",
            clip_id="clip-abc",
            timestamp="2025-01-01T00:00:00Z",
            event_type=EventType.selected,
        )
        store.append_event(e1)
        store.append_event(e2)

        results = store.get_events(event_type=EventType.selected)
        assert len(results) == 1
        assert results[0].event_id == "evt-002"

    def test_get_events_filters_by_both_clip_id_and_event_type(self):
        store = DecisionStore()
        e1 = ClipEvent(
            event_id="evt-001",
            clip_id="clip-abc",
            timestamp="2025-01-01T00:00:00Z",
            event_type=EventType.candidate_created,
        )
        e2 = ClipEvent(
            event_id="evt-002",
            clip_id="clip-abc",
            timestamp="2025-01-01T00:00:00Z",
            event_type=EventType.selected,
        )
        e3 = ClipEvent(
            event_id="evt-003",
            clip_id="clip-def",
            timestamp="2025-01-01T00:00:00Z",
            event_type=EventType.candidate_created,
        )
        store.append_event(e1)
        store.append_event(e2)
        store.append_event(e3)

        results = store.get_events(clip_id="clip-abc", event_type=EventType.selected)
        assert len(results) == 1
        assert results[0].event_id == "evt-002"

    def test_get_events_returns_all_with_no_filters(self):
        store = DecisionStore()
        e1 = ClipEvent(
            event_id="evt-001",
            clip_id="clip-abc",
            timestamp="2025-01-01T00:00:00Z",
            event_type=EventType.candidate_created,
        )
        e2 = ClipEvent(
            event_id="evt-002",
            clip_id="clip-def",
            timestamp="2025-01-01T00:00:00Z",
            event_type=EventType.selected,
        )
        store.append_event(e1)
        store.append_event(e2)

        results = store.get_events()
        assert len(results) == 2

    def test_get_all_events_returns_all(self):
        store = DecisionStore()
        e1 = ClipEvent(
            event_id="evt-001",
            clip_id="clip-abc",
            timestamp="2025-01-01T00:00:00Z",
            event_type=EventType.candidate_created,
        )
        e2 = ClipEvent(
            event_id="evt-002",
            clip_id="clip-def",
            timestamp="2025-01-01T00:00:00Z",
            event_type=EventType.selected,
        )
        store.append_event(e1)
        store.append_event(e2)

        all_events = store.get_all_events()
        assert len(all_events) == 2
        assert all_events[0].event_id == "evt-001"
        assert all_events[1].event_id == "evt-002"

    def test_get_all_events_returns_copy(self):
        store = DecisionStore()
        e1 = ClipEvent(
            event_id="evt-001",
            clip_id="clip-abc",
            timestamp="2025-01-01T00:00:00Z",
            event_type=EventType.candidate_created,
        )
        store.append_event(e1)
        all_events = store.get_all_events()
        all_events.append("should_not_affect_store")
        assert store.count() == 1

    def test_clear_resets_store(self):
        store = DecisionStore()
        event = ClipEvent(
            event_id="evt-001",
            clip_id="clip-abc",
            timestamp="2025-01-01T00:00:00Z",
            event_type=EventType.candidate_created,
        )
        store.append_event(event)
        assert store.count() == 1
        store.clear()
        assert store.count() == 0
        assert store.get_all_events() == []

    def test_clear_on_empty_store_does_not_error(self):
        store = DecisionStore()
        store.clear()
        assert store.count() == 0


# ── LearnedStateStore ──────────────────────────────────────────────────────────


class TestLearnedStateStore:
    def test_set_get_roundtrip(self):
        store = LearnedStateStore()
        store.set("clip:abc:score", '{"score": 0.95}', derived_from_event_id="evt-001")
        result = store.get("clip:abc:score")
        assert result == '{"score": 0.95}'

    def test_set_without_derived_from(self):
        store = LearnedStateStore()
        store.set("clip:abc:tags", '["funny", "short"]')
        result = store.get("clip:abc:tags")
        assert result == '["funny", "short"]'

    def test_get_returns_none_for_missing_key(self):
        store = LearnedStateStore()
        assert store.get("nonexistent") is None

    def test_delete_removes_key(self):
        store = LearnedStateStore()
        store.set("clip:abc:score", '{"score": 0.95}')
        store.delete("clip:abc:score")
        assert store.get("clip:abc:score") is None

    def test_delete_nonexistent_key_does_not_error(self):
        store = LearnedStateStore()
        store.delete("nonexistent")

    def test_get_all_returns_dict(self):
        store = LearnedStateStore()
        store.set("a", "1")
        store.set("b", "2")
        all_states = store.get_all()
        assert all_states == {"a": "1", "b": "2"}

    def test_get_all_returns_copy(self):
        store = LearnedStateStore()
        store.set("a", "1")
        all_states = store.get_all()
        all_states["b"] = "2"
        assert "b" not in store.get_all()

    def test_clear_removes_all_state(self):
        store = LearnedStateStore()
        store.set("a", "1")
        store.set("b", "2")
        store.clear()
        assert store.get_all() == {}
        assert store.get("a") is None

    def test_clear_on_empty_store_does_not_error(self):
        store = LearnedStateStore()
        store.clear()
        assert store.get_all() == {}

    def test_overwrite_existing_key(self):
        store = LearnedStateStore()
        store.set("clip:abc:score", '{"score": 0.95}')
        store.set("clip:abc:score", '{"score": 0.99}')
        assert store.get("clip:abc:score") == '{"score": 0.99}'

    def test_multiple_keys_independent(self):
        store = LearnedStateStore()
        store.set("key1", "val1")
        store.set("key2", "val2")
        store.delete("key1")
        assert store.get("key1") is None
        assert store.get("key2") == "val2"
