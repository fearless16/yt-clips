"""Boundary models for catalog, production, and outcome data."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ShortRecord:
    """One video proven to be present on the channel's Shorts shelf."""

    video_id: str
    channel_id: str
    title: str
    description: str
    published_at: str
    duration_seconds: int
    is_short: bool
    category_id: str = ""
    tags: tuple[str, ...] = ()
    source_url: str = ""
    local_metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PerformanceSnapshot:
    """Immutable point-in-time YouTube outcome counters."""

    video_id: str
    captured_at: str
    engaged_views: int = 0
    views: int = 0
    estimated_minutes_watched: float = 0.0
    average_view_duration_seconds: float = 0.0
    average_view_percentage: float = 0.0
    likes: int = 0
    comments: int = 0
    shares: int = 0
    subscribers_gained: int = 0
    subscribers_lost: int = 0
    source: str = "youtube_analytics"
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        counters = (
            self.engaged_views,
            self.views,
            self.likes,
            self.comments,
            self.shares,
            self.subscribers_gained,
            self.subscribers_lost,
        )
        if any(value < 0 for value in counters):
            raise ValueError("performance counters cannot be negative")
        if self.average_view_duration_seconds < 0 or self.average_view_percentage < 0:
            raise ValueError("watch metrics cannot be negative")
