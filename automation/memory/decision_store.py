"""Decision store — append-only event store and derived learned state store."""

from copy import deepcopy
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from automation.memory.event_models import ClipEvent


class DecisionStore:
    def __init__(self) -> None:
        self._events: list[ClipEvent] = []
        self._lock = Lock()

    def append_event(self, event: "ClipEvent") -> None:
        from automation.memory.event_models import ClipEvent as _ClipEvent

        if not isinstance(event, _ClipEvent):
            raise TypeError(f"expected ClipEvent, got {type(event).__name__}")
        with self._lock:
            self._events.append(event)

    def get_events(
        self,
        clip_id: str | None = None,
        event_type: object = None,
    ) -> list["ClipEvent"]:
        with self._lock:
            results = list(self._events)
        if clip_id is not None:
            results = [e for e in results if e.clip_id == clip_id]
        if event_type is not None:
            results = [e for e in results if e.event_type == event_type]
        return results

    def get_all_events(self) -> list["ClipEvent"]:
        with self._lock:
            return list(self._events)

    def count(self) -> int:
        with self._lock:
            return len(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


class LearnedStateStore:
    def __init__(self) -> None:
        self._state: dict[str, str] = {}
        self._lock = Lock()

    def set(
        self,
        key: str,
        value_json: str,
        derived_from_event_id: str | None = None,
    ) -> None:
        with self._lock:
            self._state[key] = value_json

    def get(self, key: str) -> str | None:
        with self._lock:
            return self._state.get(key)

    def get_all(self) -> dict[str, str]:
        with self._lock:
            return dict(self._state)

    def delete(self, key: str) -> None:
        with self._lock:
            self._state.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._state.clear()
