"""Learner — processes feedback and manual overrides into learned state."""

import json
import uuid
from datetime import datetime, timezone

from automation.memory.decision_store import DecisionStore, LearnedStateStore
from automation.memory.event_models import ClipEvent, EventType
from automation.memory.feedback_schema import FeedbackPayload


class Learner:
    def __init__(self, decision_store: DecisionStore, learned_state: LearnedStateStore) -> None:
        self._store = decision_store
        self._state = learned_state
        self._init_defaults()

    def _init_defaults(self) -> None:
        if self._state.get("hook_weight") is None:
            self._state.set("hook_weight", "1.0")
        if self._state.get("payoff_weight") is None:
            self._state.set("payoff_weight", "1.0")

    def process_feedback(self, payload: FeedbackPayload) -> None:
        event = ClipEvent(
            event_id=uuid.uuid4().hex[:12],
            clip_id=payload.clip_id,
            timestamp=payload.timestamp,
            event_type=EventType.metrics_received,
            payload_json=json.dumps({
                "rating": payload.rating,
                "feedback_type": payload.feedback_type,
            }),
        )
        self._store.append_event(event)

        hook_weight = float(self._state.get("hook_weight") or "1.0")
        payoff_weight = float(self._state.get("payoff_weight") or "1.0")

        if payload.feedback_type in ("like", "share"):
            hook_weight += 0.1
            payoff_weight += 0.1
        elif payload.feedback_type in ("dislike", "skip"):
            hook_weight -= 0.1
            payoff_weight -= 0.1

        self._state.set("hook_weight", str(hook_weight))
        self._state.set("payoff_weight", str(payoff_weight))

    def process_manual_override(self, clip_id: str, override_type: str) -> None:
        event = ClipEvent(
            event_id=uuid.uuid4().hex[:12],
            clip_id=clip_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=EventType.manual_override,
            payload_json=json.dumps({"override_type": override_type}),
        )
        self._store.append_event(event)

        if override_type == "keep":
            hook_weight = float(self._state.get("hook_weight") or "1.0")
            hook_weight += 0.3
            self._state.set("hook_weight", str(hook_weight))

    def get_state(self, key: str) -> str | None:
        return self._state.get(key)

    def reset_learned_state(self) -> None:
        self._state.clear()
