"""CricketScorer — combines all learners into a single candidate score.

FinalScore = (
    0.30 * TrendScore +
    0.25 * FormatScore +
    0.20 * EntityScore +
    0.15 * HistoricalScore +
    0.10 * TimingScore
) * (1 - FatiguePenalty)
"""

import json
from datetime import datetime, timezone
from typing import Any


DEFAULT_WEIGHTS = {
    "trend_weight": 0.30,
    "format_weight": 0.25,
    "entity_weight": 0.20,
    "historical_weight": 0.15,
    "timing_weight": 0.10,
}


class CricketScorer:
    """Combines all learners into a single candidate score."""

    def __init__(self, format_learner, entity_learner, trend_engine,
                 timing_learner, duration_learner, state_store):
        """Initialize CricketScorer.

        Args:
            format_learner: FormatLearner instance
            entity_learner: EntityLearner instance
            trend_engine: TrendEngine instance
            timing_learner: TimingLearner instance
            duration_learner: DurationLearner instance
            state_store: PersistentStateStore instance
        """
        self._format = format_learner
        self._entity = entity_learner
        self._trend = trend_engine
        self._timing = timing_learner
        self._duration = duration_learner
        self._state = state_store

        weights = self._state.get("scoring_weights")
        self._weights = weights if weights else dict(DEFAULT_WEIGHTS)

    def score_candidate(self, candidate: dict) -> float:
        """Score a single candidate clip.

        Args:
            candidate: Dict with detected_players, detected_teams,
                       detected_hook_type, duration_seconds,
                       weighted_score (from highlight.py), title_pattern

        Returns:
            Final score (0-1)
        """
        players = candidate.get("detected_players") or candidate.get("players", [])
        teams = candidate.get("detected_teams") or candidate.get("teams", [])
        hook_type = candidate.get("detected_hook_type") or candidate.get("hook_type", "shock")
        weighted_score = candidate.get("weighted_score", 5.0)
        title_pattern = candidate.get("title_pattern", {})

        trend_score = max(
            self._trend.get_trend_for_entities(players, teams),
            0.1
        )

        hook_score = self._format.get_hook_score(hook_type)
        title_score = self._format.get_title_score(title_pattern)
        format_score = hook_score * title_score

        series = None
        format_tags = candidate.get("format_tags", [])
        if format_tags:
            series = format_tags[0]
        entity_score = self._entity.get_entity_score(players, teams, series)

        historical_score = min(weighted_score / 10.0, 1.0)

        schedule = self._timing.get_schedule_recommendation()
        timing_score = schedule.get("confidence", 0.5)

        fatigue = 0.0
        for p in players:
            fatigue = max(fatigue, self._entity.fatigue_penalty("players", p))
        for t in teams:
            fatigue = max(fatigue, self._entity.fatigue_penalty("teams", t))

        w = self._weights
        raw_score = (
            w["trend_weight"] * trend_score +
            w["format_weight"] * format_score +
            w["entity_weight"] * entity_score +
            w["historical_weight"] * historical_score +
            w["timing_weight"] * timing_score
        )

        final = raw_score * (1.0 - fatigue)
        return max(0.0, min(1.0, final))

    def score_many(self, candidates: list[dict]) -> list[tuple[dict, float]]:
        """Score multiple candidates and return sorted by score desc.

        Args:
            candidates: List of candidate dicts

        Returns:
            List of (candidate, score) tuples sorted by score descending
        """
        scored = [(c, self.score_candidate(c)) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def get_weights(self) -> dict:
        """Get current scoring weights.

        Returns:
            Dict of weight_name -> weight_value
        """
        return dict(self._weights)

    def recalibrate_weights(self, events: list) -> None:
        """Recalibrate weights based on recent metrics_received events.

        Compares predicted component scores vs actual view performance.
        Blends new weights with current weights (70/30) to avoid over-correction.

        Args:
            events: List of ClipEvent (metrics_received type)
        """
        if len(events) < 20:
            return

        component_totals = {
            "trend_weight": 0.0,
            "format_weight": 0.0,
            "entity_weight": 0.0,
            "historical_weight": 0.0,
            "timing_weight": 0.0,
        }

        # Only count events that have real view data (productive events).
        # Using len(events) as divisor dilutes the signal with zero-view skips.
        productive_n = 0

        for event in events:
            payload = json.loads(event.payload_json)
            views = payload.get("views", 0)
            if views == 0:
                continue

            players = payload.get("player_tags", [])
            teams = payload.get("team_tags", [])
            hook_type = payload.get("hook_type", "shock")
            title_pattern = payload.get("title_pattern", {})

            trend = self._trend.get_trend_for_entities(players, teams)
            hook = self._format.get_hook_score(hook_type)
            title = self._format.get_title_score(title_pattern)
            fmt = hook * title
            series = payload.get("format_tags", [None])[0] if payload.get("format_tags") else None
            entity = self._entity.get_entity_score(players, teams, series)
            schedule = self._timing.get_schedule_recommendation()
            timing = schedule.get("confidence", 0.5)

            weighted_score = payload.get("weighted_score", views / 100.0)
            historical = min(weighted_score / 10.0, 1.0)
            total = trend + fmt + entity + historical + timing
            if total == 0:
                continue

            component_totals["trend_weight"] += trend / total
            component_totals["format_weight"] += fmt / total
            component_totals["entity_weight"] += entity / total
            component_totals["historical_weight"] += historical / total
            component_totals["timing_weight"] += timing / total
            productive_n += 1

        # No productive events — keep current weights unchanged.
        if productive_n == 0:
            return

        new_weights = {}
        for k, v in component_totals.items():
            raw = v / productive_n
            new_weights[k] = max(0.05, raw)

        total = sum(new_weights.values())
        for k in new_weights:
            new_weights[k] /= total

        for k in new_weights:
            self._weights[k] = 0.7 * self._weights.get(k, DEFAULT_WEIGHTS[k]) + 0.3 * new_weights[k]

        # Re-normalise core keys only (exclude metadata like last_recalibrated).
        core_total = sum(self._weights.get(k, 0) for k in DEFAULT_WEIGHTS)
        for k in DEFAULT_WEIGHTS:
            self._weights[k] /= core_total

        self._weights["last_recalibrated"] = datetime.now(timezone.utc).isoformat()
        self._state.set("scoring_weights", self._weights)
