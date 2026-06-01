"""PolicyUpdater — processes ClipEvents and delegates to Learner."""

import json

from automation.memory.event_models import ClipEvent, EventType
from automation.memory.feedback_schema import validate_feedback_payload
from automation.learner.learner import Learner


class PolicyUpdater:
    def __init__(self, learner: Learner) -> None:
        self._learner = learner

    def update_from_event(self, event: ClipEvent) -> None:
        if event.event_type == EventType.metrics_received:
            payload = json.loads(event.payload_json)
            full_payload = {
                "clip_id": event.clip_id,
                "rating": payload.get("rating"),
                "feedback_type": payload.get("feedback_type"),
                "details": payload.get("details"),
                "timestamp": event.timestamp,
            }
            try:
                fb = validate_feedback_payload(full_payload)
            except ValueError:
                return
            self._learner.process_feedback(fb)
        elif event.event_type == EventType.manual_override:
            payload = json.loads(event.payload_json)
            override_type = payload.get("override_type", "")
            self._learner.process_manual_override(event.clip_id, override_type)

    def update_from_events(self, events: list[ClipEvent]) -> None:
        for event in events:
            self.update_from_event(event)
