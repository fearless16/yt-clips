"""automation — self-correcting clip selection and learning system.

Append-only event store.
Derived state only.
Human overrides outrank models.
Infrastructure noise is not learning data.
"""

VERSION = "3.0.0"


# ── Backward-compat aliases (lazy) ─────────────────────────────────────────
# These let legacy code import from `automation` directly.
# Lazily initialized to avoid import-time side effects.

_decision_store = None
_provider_health = None


def _ensure_decision_store():
    global _decision_store
    if _decision_store is None:
        from automation.memory.decision_store import DecisionStore as _DS
        _decision_store = _DS()
    return _decision_store


def _ensure_provider_health():
    global _provider_health
    if _provider_health is None:
        from automation.providers.provider_health import ProviderHealth as _PH
        _provider_health = _PH()
    return _provider_health


def __getattr__(name):
    if name == "decision_store":
        return _ensure_decision_store()
    if name == "provider_health":
        return _ensure_provider_health()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    _ensure_decision_store().append_event(event)
    return event


def emit_infra_failed(clip_id, error, stage=""):
    from automation.memory.event_models import EventType
    return emit_event(clip_id, EventType.infra_failed, {
        "error": error, "stage": stage,
    })
