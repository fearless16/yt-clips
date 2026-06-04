"""TrendEngine — cricket trend tracking with half-life decay.

30% weight in scoring formula. Trends decay over time based on
category-specific half-lives. Velocity boosts accelerating trends.
"""

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class TrendInput:
    """Input data for trend ingestion."""
    trend_id: str
    source: str
    query: str
    trend_score: float
    velocity: float
    category: str
    entities: dict
    half_life_hours: float
    expires_at: str | None = None

    @classmethod
    def from_payload(cls, payload: dict) -> "TrendInput":
        """Create TrendInput from event payload."""
        return cls(
            trend_id=payload.get("trend_id", ""),
            source=payload.get("source", "manual"),
            query=payload.get("query", ""),
            trend_score=payload.get("trend_score", 0.5),
            velocity=payload.get("velocity", 0.0),
            category=payload.get("category", "match"),
            entities=payload.get("entities", {}),
            half_life_hours=payload.get("half_life_hours", 72.0),
            expires_at=payload.get("expires_at")
        )


HALF_LIVES = {
    "live_moment": 6.0,
    "match_result": 36.0,
    "match": 72.0,
    "series": 120.0,
    "tournament": 168.0,
    "controversy": 180.0,
    "selection": 180.0,
    "evergreen": 336.0,
}


class TrendEngine:
    """Cricket trend tracking with half-life decay. 30% weight."""

    def __init__(self, state_store):
        """Initialize TrendEngine.

        Args:
            state_store: PersistentStateStore instance
        """
        self._state = state_store

    def ingest(self, trend: TrendInput) -> None:
        """Add or update a trend. Idempotent by trend_id.

        Args:
            trend: TrendInput with trend data
        """
        trends = self._state.get("trend_scores")
        if trends is None:
            trends = {"active_trends": {}}

        active = trends.get("active_trends", {})

        now = datetime.now(timezone.utc).isoformat()
        active[trend.trend_id] = {
            "score": trend.trend_score,
            "initial_score": trend.trend_score,
            "velocity": trend.velocity,
            "created_at": now,
            "half_life_hours": trend.half_life_hours,
            "current_score": trend.trend_score,
            "entities": trend.entities,
            "query": trend.query,
            "source": trend.source,
            "category": trend.category,
            "expires_at": trend.expires_at
        }

        trends["active_trends"] = active
        self._state.set("trend_scores", trends)

    def decay_all(self) -> int:
        """Recompute current_score for all active trends using half-life decay.

        Returns:
            Number of trends updated
        """
        trends = self._state.get("trend_scores")
        if trends is None:
            return 0

        active = trends.get("active_trends", {})
        if not active:
            return 0

        now = datetime.now(timezone.utc)
        updated = 0

        for trend_id, data in active.items():
            created_str = data.get("created_at", "")
            if not created_str:
                continue

            try:
                created = datetime.fromisoformat(created_str)
            except (ValueError, TypeError):
                continue

            elapsed_hours = (now - created).total_seconds() / 3600.0
            half_life = data.get("half_life_hours", 72.0)

            if half_life <= 0:
                half_life = 72.0

            initial = data.get("initial_score", 0.5)
            decayed = initial * math.pow(2, -elapsed_hours / half_life)

            velocity = data.get("velocity", 0.0)
            velocity_boost = min(velocity * 0.3, 0.25)
            effective = min(1.0, decayed + velocity_boost)

            data["current_score"] = effective
            updated += 1

        trends["active_trends"] = active
        self._state.set("trend_scores", trends)
        return updated

    def get_active(self, min_score: float = 0.10) -> list[dict]:
        """Return active trends above threshold, sorted by current_score desc.

        Args:
            min_score: Minimum score threshold

        Returns:
            List of trend dicts
        """
        trends = self._state.get("trend_scores")
        if trends is None:
            return []

        active = trends.get("active_trends", {})
        result = []

        for trend_id, data in active.items():
            score = data.get("current_score", 0.0)
            if score >= min_score:
                result.append({
                    "trend_id": trend_id,
                    "score": score,
                    "query": data.get("query", ""),
                    "category": data.get("category", ""),
                    "entities": data.get("entities", {}),
                    "created_at": data.get("created_at", ""),
                    "half_life_hours": data.get("half_life_hours", 72.0)
                })

        result.sort(key=lambda x: x["score"], reverse=True)
        return result

    def get_trend_for_entities(self, players: list[str],
                                teams: list[str]) -> float:
        """Get max trend score matching any of the given entities.

        Args:
            players: List of player names
            teams: List of team names

        Returns:
            Max trend score (0-1), minimum 0.1 if no match
        """
        trends = self._state.get("trend_scores")
        if trends is None:
            return 0.1

        active = trends.get("active_trends", {})
        if not active:
            return 0.1

        search_entities = set()
        for p in players:
            search_entities.add(p.lower())
        for t in teams:
            search_entities.add(t.lower())

        if not search_entities:
            return 0.1

        max_score = 0.1

        for trend_id, data in active.items():
            trend_entities = data.get("entities", {})
            trend_players = set(p.lower() for p in trend_entities.get("players", []))
            trend_teams = set(t.lower() for t in trend_entities.get("teams", []))
            all_trend_entities = trend_players | trend_teams

            if search_entities & all_trend_entities:
                score = data.get("current_score", 0.0)
                max_score = max(max_score, score)

        return max_score

    def prune_expired(self) -> int:
        """Remove trends where current_score < 0.05.

        Returns:
            Number of trends pruned
        """
        trends = self._state.get("trend_scores")
        if trends is None:
            return 0

        active = trends.get("active_trends", {})
        to_remove = []

        for trend_id, data in active.items():
            if data.get("current_score", 0.0) < 0.05:
                to_remove.append(trend_id)

        for trend_id in to_remove:
            del active[trend_id]

        trends["active_trends"] = active
        self._state.set("trend_scores", trends)
        return len(to_remove)

    def get_hot_topics(self, limit: int = 5) -> list[dict]:
        """Get top trending topics with entity context for recommendations.

        Args:
            limit: Maximum number of topics to return

        Returns:
            List of dicts with query, score, entities
        """
        active = self.get_active(min_score=0.15)
        return active[:limit]
