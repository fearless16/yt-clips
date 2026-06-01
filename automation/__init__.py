"""automation — self-correcting clip selection and learning system.

Append-only event store.
Derived state only.
Human overrides outrank models.
Infrastructure noise is not learning data.
"""

VERSION = "3.0.0"


# ── Backward-compat aliases ────────────────────────────────────────────────
# These let legacy code import from `automation` directly.

def _get_decision_store():
    from automation.memory.decision_store import DecisionStore as _DS
    return _DS()


def _get_provider_health():
    from automation.providers.provider_health import ProviderHealth as _PH
    return _PH()


decision_store = _get_decision_store()
provider_health = _get_provider_health()


def emit_event(clip_id, event_type, payload=None):
    import json
    import uuid
    from datetime import datetime, timezone
    from automation.memory.event_models import ClipEvent
    event = ClipEvent(
        event_id=f"evt-{uuid.uuid4().hex[:8]}",
        clip_id=clip_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        event_type=event_type,
        payload_json=json.dumps(payload or {}),
    )
    decision_store.append_event(event)
    return event


def emit_infra_failed(clip_id, error, stage=""):
    from automation.memory.event_models import EventType
    return emit_event(clip_id, EventType.infra_failed, {
        "error": error, "stage": stage,
    })
