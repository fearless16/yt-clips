"""Bounded selection policy; semantic completeness always wins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class PolicyResult:
    """Auditable outcome of one selection-policy evaluation."""

    eligible: bool
    applied_adjustment: float
    shadow_adjustment: float
    matched_segments: tuple[str, ...]
    reason: str


class SelectionPolicy:
    """Convert supported model effects into a small, safe score adjustment."""

    def __init__(
        self,
        recommendations: Iterable[dict[str, Any]],
        *,
        shadow_mode: bool,
        max_adjustment_points: float = 3.0,
    ) -> None:
        self.recommendations = list(recommendations)
        self.shadow_mode = shadow_mode
        self.max_adjustment_points = max(0.0, float(max_adjustment_points))

    def evaluate(
        self,
        *,
        duration_seconds: float,
        complete_thought: bool,
        features: dict[str, str] | None = None,
    ) -> PolicyResult:
        """Evaluate without changing boundaries, transcript, or playback speed."""
        if not complete_thought:
            return PolicyResult(False, 0.0, 0.0, (), "incomplete_thought")
        candidate = dict(features or {})
        candidate["duration_bucket"] = _duration_bucket(duration_seconds)
        effect = 0.0
        matched: list[str] = []
        for segment in self.recommendations:
            name = str(segment.get("feature_name", ""))
            value = str(segment.get("feature_value", ""))
            if candidate.get(name) == value:
                effect += max(0.0, float(segment.get("effect", 0.0)))
                matched.append(f"{name}={value}")
        adjustment = min(self.max_adjustment_points, effect * 100.0)
        return PolicyResult(
            True,
            0.0 if self.shadow_mode else adjustment,
            adjustment,
            tuple(matched),
            "supported_evidence" if matched else "no_supported_segment",
        )


def _duration_bucket(seconds: float) -> str:
    if seconds < 15:
        return "under_15"
    if seconds < 25:
        return "15_24"
    if seconds < 40:
        return "25_39"
    if seconds < 60:
        return "40_59"
    return "60_plus"

