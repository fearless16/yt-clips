"""Event models — immutable event types and learned state entries for the append-only event store."""

import json
from dataclasses import dataclass, field
from enum import Enum


class EventType(str, Enum):
    candidate_created = "candidate_created"
    candidate_scored = "candidate_scored"
    candidate_ranked = "candidate_ranked"
    selected = "selected"
    rejected = "rejected"
    exported = "exported"
    published = "published"
    metrics_received = "metrics_received"
    manual_override = "manual_override"
    validation_failed = "validation_failed"
    infra_failed = "infra_failed"
    deferred = "deferred"
    policy_updated = "policy_updated"


@dataclass(frozen=True)
class ClipEvent:
    event_id: str
    clip_id: str
    timestamp: str
    event_type: EventType
    payload_json: str = field(default="{}")

    def __post_init__(self) -> None:
        if not isinstance(self.event_type, EventType):
            raise TypeError(
                f"event_type must be an EventType enum, got {type(self.event_type).__name__}"
            )
        if not isinstance(self.payload_json, str):
            raise TypeError(
                f"payload_json must be a string, got {type(self.payload_json).__name__}"
            )
        try:
            json.loads(self.payload_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"payload_json is not valid JSON: {e}")


@dataclass(frozen=True)
class LearnedStateEntry:
    state_key: str
    value_json: str
    derived_from_event_id: str | None
    updated_at: str
