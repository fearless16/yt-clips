"""Feedback schema — validated feedback payloads with strict field checking."""

import json
import uuid
from dataclasses import dataclass
from typing import Any


VALID_FEEDBACK_TYPES = frozenset({
    "like", "dislike", "skip", "watch_full", "comment", "share",
})

ALLOWED_KEYS = frozenset({
    "clip_id", "rating", "feedback_type", "details", "timestamp",
})


@dataclass(frozen=True)
class FeedbackPayload:
    clip_id: str
    rating: int
    feedback_type: str
    details: dict | None
    timestamp: str


@dataclass(frozen=True)
class ValidationResult:
    success: bool
    feedback: FeedbackPayload | None
    error: str | None
    error_type: str | None = None


def _categorize_error(message: str) -> str:
    message_lower = message.lower()
    if "unknown keys" in message_lower:
        return "unknown_keys"
    if "missing required field" in message_lower:
        return "missing_field"
    if "feedback_type" in message_lower:
        return "invalid_value"
    if "rating" in message_lower:
        return "invalid_value"
    if "details" in message_lower:
        return "invalid_value"
    if "timestamp" in message_lower:
        return "invalid_value"
    if "payload must be a dict" in message_lower:
        return "invalid_payload"
    return "validation_error"


def _build_validation_failed_event(payload: dict, error: str, clip_id: str):
    from automation.memory.event_models import ClipEvent, EventType

    return ClipEvent(
        event_id=uuid.uuid4().hex[:12],
        clip_id=clip_id,
        timestamp="",
        event_type=EventType.validation_failed,
        payload_json=json.dumps({
            "error": error,
            "original_payload": {k: v for k, v in payload.items() if k != "details" or v is None},
        }),
    )


def validate_feedback_payload(payload: dict) -> FeedbackPayload:
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")

    unknown = set(payload.keys()) - ALLOWED_KEYS
    if unknown:
        raise ValueError(f"unknown keys: {', '.join(sorted(unknown))}")

    clip_id = payload.get("clip_id")
    if not clip_id:
        raise ValueError("missing required field: clip_id")

    rating = payload.get("rating")
    if not isinstance(rating, int) or rating < 1 or rating > 5:
        raise ValueError(f"rating must be an integer between 1 and 5, got {rating}")

    feedback_type = payload.get("feedback_type")
    if feedback_type not in VALID_FEEDBACK_TYPES:
        raise ValueError(
            f"invalid feedback_type: {feedback_type}; "
            f"must be one of {sorted(VALID_FEEDBACK_TYPES)}"
        )

    details = payload.get("details")
    if details is not None and not isinstance(details, dict):
        raise ValueError("details must be a dict or None")

    timestamp = payload.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp.strip():
        raise ValueError("missing required field: timestamp")

    return FeedbackPayload(
        clip_id=clip_id,
        rating=rating,
        feedback_type=feedback_type,
        details=details,
        timestamp=timestamp,
    )


def validate_feedback(payload: dict, decision_store=None) -> ValidationResult:
    try:
        feedback = validate_feedback_payload(payload)
        return ValidationResult(success=True, feedback=feedback, error=None)
    except ValueError as e:
        error_msg = str(e)
        error_type = _categorize_error(error_msg)
        clip_id = payload.get("clip_id", "unknown") if isinstance(payload, dict) else "unknown"
        result = ValidationResult(
            success=False, feedback=None, error=error_msg, error_type=error_type,
        )
        if decision_store is not None:
            event = _build_validation_failed_event(payload, error_msg, clip_id)
            decision_store.append_event(event)
        return result
