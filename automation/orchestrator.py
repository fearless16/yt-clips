"""Orchestrator — runs pipeline stages and emits events to the decision store."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from automation.memory.event_models import EventType, ClipEvent
from automation.memory.decision_store import DecisionStore

_STAGES = [
    "download", "transcribe", "score", "rank",
    "export", "seo", "upload", "cleanup",
]

_STAGE_EVENT_MAP: dict[str, str] = {
    "score": "candidate_scored",
    "rank": "candidate_ranked",
    "export": "exported",
    "upload": "published",
}


class Orchestrator:
    def __init__(
        self,
        decision_store: DecisionStore | None = None,
        clip_scorer: Any = None,
        clip_ranker: Any = None,
    ) -> None:
        self._decision_store: DecisionStore = (
            decision_store if decision_store is not None else DecisionStore()
        )
        self._clip_scorer = clip_scorer
        self._clip_ranker = clip_ranker
        self._pipeline_running: bool = False
        self._last_run: str | None = None
        self._events_emitted: int = 0

    def run_pipeline(
        self,
        url: str,
        clip_data: dict | None = None,
        stages: dict[str, bool] | None = None,
    ) -> dict[str, Any]:
        self._pipeline_running = True
        effective_stages = stages if stages is not None else {}
        clip_id = f"clip-{uuid.uuid4().hex[:8]}"
        data: dict[str, Any] = {"clip_id": clip_id, "url": url}
        if clip_data is not None:
            data.update(clip_data)
        self.emit_event(clip_id, "candidate_created", {"url": url})
        stages_completed: list[str] = []
        errors: list[dict[str, str]] = []
        for stage_name in _STAGES:
            if effective_stages.get(stage_name, True):
                try:
                    stage_result = self._run_stage(stage_name, data)
                    stages_completed.append(stage_name)
                    data.update(stage_result)
                    event_type = _STAGE_EVENT_MAP.get(stage_name)
                    if event_type is not None:
                        self.emit_event(clip_id, event_type, {"stage": stage_name})
                except Exception as exc:
                    errors.append({"stage": stage_name, "error": str(exc)})
        self._pipeline_running = False
        self._last_run = datetime.now(timezone.utc).isoformat()
        return {
            "url": url,
            "stages_completed": stages_completed,
            "events_emitted": self._events_emitted,
            "errors": errors,
        }

    def _run_stage(self, name: str, data: dict[str, Any]) -> dict[str, Any]:
        if name == "score" and self._clip_scorer is not None:
            return self._clip_scorer.score(data)
        if name == "rank" and self._clip_ranker is not None:
            clips = data.get("clips", [data])
            ranked = self._clip_ranker.rank(clips)
            return {
                "ranked": ranked,
                "rank": ranked[0].get("rank") if ranked else None,
            }
        return {**data, name: True}

    def emit_event(
        self,
        clip_id: str,
        event_type_str: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        try:
            event_type = EventType(event_type_str)
        except ValueError:
            raise ValueError(
                f"Invalid event type: '{event_type_str}'. "
                f"Valid types: {[e.value for e in EventType]}"
            )
        event = ClipEvent(
            event_id=f"evt-{uuid.uuid4().hex[:8]}",
            clip_id=clip_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            payload_json=json.dumps(payload or {}),
        )
        self._decision_store.append_event(event)
        self._events_emitted += 1

    def get_pipeline_status(self) -> dict[str, Any]:
        return {
            "pipeline_running": self._pipeline_running,
            "last_run": self._last_run,
            "total_events": self._events_emitted,
        }
