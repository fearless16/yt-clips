"""automation — self-correcting clip selection and learning system.

Append-only event store.
Derived state only.
Human overrides outrank models.
Infrastructure noise is not learning data.

Usage::

    from automation import decision_store, emit_event, emit_infra_failed
    from automation.memory.event_models import EventType

    emit_event("clip1", EventType.candidate_created, {"text": "..."})
    events = decision_store.get_events(clip_id="clip1")
"""

import json
import os
import uuid
from datetime import datetime, timezone

from automation.memory.event_models import ClipEvent, EventType
from automation.memory.decision_store import DecisionStore, LearnedStateStore
from automation.providers.provider_health import ProviderHealth
from automation.learner.learner import Learner as AutomationLearner
from automation.learner.policy_updater import PolicyUpdater
from automation.learner.preference_engine import PreferenceEngine
from automation.learner.replay import ReplayEngine

VERSION = "3.0.0"

_AUTOMATION_DB = os.environ.get("AUTOMATION_DB", "automation.db")

decision_store = DecisionStore(db_path=_AUTOMATION_DB)
provider_health = ProviderHealth()

_learned_state = LearnedStateStore(db_path=_AUTOMATION_DB)
automation_learner = AutomationLearner(decision_store, _learned_state)
policy_updater = PolicyUpdater(automation_learner)
preference_engine = PreferenceEngine(automation_learner, decision_store)
replay_engine = ReplayEngine(decision_store, _learned_state, automation_learner)


def emit_event(
    clip_id: str,
    event_type: EventType,
    payload: dict | None = None,
    timestamp: str | None = None,
) -> ClipEvent:
    """Create a ClipEvent, append it to the global decision_store, and return it."""
    event = ClipEvent(
        event_id=uuid.uuid4().hex[:12],
        clip_id=clip_id,
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        event_type=event_type,
        payload_json=json.dumps(payload or {}),
    )
    decision_store.append_event(event)
    return event


def emit_infra_failed(clip_id: str, error: str, stage: str = "") -> ClipEvent:
    return emit_event(clip_id, EventType.infra_failed, {"error": error, "stage": stage})


def emit_validation_failed(clip_id: str, error: str, stage: str = "") -> ClipEvent:
    return emit_event(clip_id, EventType.validation_failed, {"error": error, "stage": stage})
