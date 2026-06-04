"""TimingLearner — upload timing optimization.

10% weight in scoring formula. Tracks best upload hours, days,
and match phases using count-weighted EMA with recency bias.
"""

from datetime import datetime, timezone
from typing import Any


DAYS_OF_WEEK = ["monday", "tuesday", "wednesday", "thursday",
                "friday", "saturday", "sunday"]

MATCH_PHASES = ["pre_toss", "during_innings", "innings_break",
                "post_match", "selection", "press_conference", "off_season"]


class TimingLearner:
    """Upload timing optimization. 10% weight."""

    def __init__(self, state_store, baseline_views: int = 290):
        """Initialize TimingLearner.

        Args:
            state_store: PersistentStateStore instance
            baseline_views: Channel average views for normalization
        """
        self._state = state_store
        self._baseline = baseline_views

    def process_metrics(self, clip_id: str, payload: dict) -> None:
        """Process a metrics_received event and update timing scores.

        Args:
            clip_id: The clip ID
            payload: Event payload with publish_hour, publish_dow, views
        """
        views = payload.get("views", 0)
        if views == 0:
            return

        signal = min(views / self._baseline, 3.0) / 3.0

        hour = payload.get("publish_hour")
        if hour is not None:
            self._update_timing("hour", str(hour), signal, views)

        dow = payload.get("publish_dow")
        if dow is not None:
            day_name = DAYS_OF_WEEK[dow] if isinstance(dow, int) and 0 <= dow <= 6 else str(dow)
            self._update_timing("day", day_name, signal, views)

        match_phase = payload.get("match_phase")
        if match_phase:
            self._update_timing("phase", match_phase, signal, views)

    def _update_timing(self, key_prefix: str, value: str,
                       signal: float, views: int) -> None:
        """Update timing score for a specific hour/day/phase.

        Args:
            key_prefix: "hour", "day", or "phase"
            value: The specific value (e.g., "19", "saturday")
            signal: Normalized view signal (0-1)
            views: Raw view count
        """
        state_key = f"timing:{key_prefix}:{value}"
        current = self._state.get(state_key)

        if current is None:
            current = {"score": 0.5, "n": 0, "avg_views": 0}

        n = current["n"]
        alpha = max(0.10, 0.25 / (1 + n * 0.08))

        old_score = current["score"]
        new_score = (1 - alpha) * old_score + alpha * signal

        old_avg = current["avg_views"]
        new_avg = (old_avg * n + views) / (n + 1)

        self._state.set(state_key, {
            "score": new_score,
            "n": n + 1,
            "avg_views": new_avg
        })

    def best_hour(self, day: int | None = None) -> int:
        """Return optimal upload hour.

        Args:
            day: Optional day of week (0=Monday, 6=Sunday)

        Returns:
            Best hour (0-23), defaults to 19 if no data
        """
        best_score = 0.0
        best_hour = 19

        for h in range(24):
            data = self._state.get(f"timing:hour:{h}")
            if data and data.get("n", 0) > 0:
                score = data.get("score", 0.5)
                if score > best_score:
                    best_score = score
                    best_hour = h

        return best_hour

    def best_day(self) -> str:
        """Return optimal day of week.

        Returns:
            Best day name, defaults to "saturday" if no data
        """
        best_score = 0.0
        best_day = "saturday"

        for day in DAYS_OF_WEEK:
            data = self._state.get(f"timing:day:{day}")
            if data and data.get("n", 0) > 0:
                score = data.get("score", 0.5)
                if score > best_score:
                    best_score = score
                    best_day = day

        return best_day

    def best_match_phase(self) -> str:
        """Return optimal match phase for upload.

        Returns:
            Best match phase, defaults to "post_match" if no data
        """
        best_score = 0.0
        best_phase = "post_match"

        for phase in MATCH_PHASES:
            data = self._state.get(f"timing:phase:{phase}")
            if data and data.get("n", 0) > 0:
                score = data.get("score", 0.5)
                if score > best_score:
                    best_score = score
                    best_phase = phase

        return best_phase

    def get_schedule_recommendation(self) -> dict:
        """Generate upload schedule recommendation.

        Returns:
            Dict with hour, day, match_phase, confidence, reason
        """
        hour = self.best_hour()
        day = self.best_day()
        phase = self.best_match_phase()

        hour_data = self._state.get(f"timing:hour:{hour}")
        day_data = self._state.get(f"timing:day:{day}")
        phase_data = self._state.get(f"timing:phase:{phase}")

        hour_n = hour_data.get("n", 0) if hour_data else 0
        day_n = day_data.get("n", 0) if day_data else 0
        phase_n = phase_data.get("n", 0) if phase_data else 0

        total_n = hour_n + day_n + phase_n
        confidence = min(0.9, total_n / 30.0) if total_n > 0 else 0.3

        avg_views = 0
        if hour_data:
            avg_views = hour_data.get("avg_views", 0)

        return {
            "hour": hour,
            "day": day,
            "match_phase": phase,
            "confidence": confidence,
            "avg_views_at_hour": avg_views,
            "total_observations": total_n,
            "reason": f"Best performing: {hour}:00 on {day} during {phase}"
        }
