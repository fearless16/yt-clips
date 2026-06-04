"""DurationLearner — optimal clip duration via UCB1.

Tracks duration buckets and uses Upper Confidence Bound to
balance exploration of under-sampled durations vs exploitation
of known-good durations.
"""

import math
from datetime import datetime, timezone
from typing import Any


DURATION_BUCKETS = {
    "15-24s": (15, 24),
    "25-34s": (25, 34),
    "35-44s": (35, 44),
    "45-59s": (45, 59),
}


class DurationLearner:
    """Optimal clip duration via UCB1."""

    def __init__(self, state_store, baseline_views: int = 290):
        """Initialize DurationLearner.

        Args:
            state_store: PersistentStateStore instance
            baseline_views: Channel average views for normalization
        """
        self._state = state_store
        self._baseline = baseline_views

    def process_metrics(self, clip_id: str, payload: dict) -> None:
        """Process a metrics_received event and update duration scores.

        Args:
            clip_id: The clip ID
            payload: Event payload with duration_seconds, views, completion_rate
        """
        duration = payload.get("duration_seconds", 0)
        views = payload.get("views", 0)
        completion = payload.get("completion_rate", 0.5)

        if duration == 0 or views == 0:
            return

        bucket = self._get_bucket(duration)
        if bucket is None:
            return

        signal = min(views / self._baseline, 3.0) / 3.0
        self._update_bucket(bucket, signal, views, completion)

    def _get_bucket(self, duration: int) -> str | None:
        """Get the duration bucket for a given duration.

        Args:
            duration: Duration in seconds

        Returns:
            Bucket name or None if out of range
        """
        for name, (low, high) in DURATION_BUCKETS.items():
            if low <= duration <= high:
                return name
        return None

    def _update_bucket(self, bucket: str, signal: float,
                       views: int, completion: float) -> None:
        """Update score for a specific duration bucket.

        Args:
            bucket: Bucket name (e.g., "25-34s")
            signal: Normalized view signal (0-1)
            views: Raw view count
            completion: Completion rate (0-1)
        """
        scores = self._state.get("duration_scores")
        if scores is None:
            scores = {"buckets": {}, "sweet_spot": "25-34s", "total_n": 0}

        buckets = scores.get("buckets", {})
        current = buckets.get(bucket, {
            "score": 0.5,
            "n": 0,
            "avg_completion": 0.5,
            "avg_views": 0,
            "total_reward": 0.0
        })

        n = current["n"]
        alpha = max(0.10, 0.25 / (1 + n * 0.08))

        old_score = current["score"]
        new_score = (1 - alpha) * old_score + alpha * signal

        old_avg = current["avg_views"]
        new_avg = (old_avg * n + views) / (n + 1)

        old_comp = current["avg_completion"]
        new_comp = (old_comp * n + completion) / (n + 1)

        buckets[bucket] = {
            "score": new_score,
            "n": n + 1,
            "avg_completion": new_comp,
            "avg_views": new_avg,
            "total_reward": current["total_reward"] + signal
        }

        sweet_spot = max(buckets.keys(), key=lambda b: buckets[b]["score"])

        scores["buckets"] = buckets
        scores["sweet_spot"] = sweet_spot
        scores["total_n"] = scores.get("total_n", 0) + 1
        self._state.set("duration_scores", scores)

    def select_duration(self) -> int:
        """Select optimal duration using UCB1.

        Returns:
            Target duration in seconds (midpoint of selected bucket)
        """
        scores = self._state.get("duration_scores")
        if scores is None:
            return 30

        buckets = scores.get("buckets", {})
        if not buckets:
            return 30

        total_n = scores.get("total_n", 1)

        best_bucket = None
        best_ucb = -1.0

        for bucket_name, data in buckets.items():
            n = data.get("n", 0)
            if n == 0:
                return self._bucket_midpoint(bucket_name)

            avg_reward = data.get("total_reward", 0.0) / n
            exploration = math.sqrt(2.0 * math.log(total_n) / n)
            ucb = avg_reward + exploration

            if ucb > best_ucb:
                best_ucb = ucb
                best_bucket = bucket_name

        if best_bucket is None:
            return 30

        return self._bucket_midpoint(best_bucket)

    def _bucket_midpoint(self, bucket: str) -> int:
        """Get midpoint duration for a bucket.

        Args:
            bucket: Bucket name

        Returns:
            Midpoint in seconds
        """
        bounds = DURATION_BUCKETS.get(bucket)
        if bounds is None:
            return 30
        return (bounds[0] + bounds[1]) // 2

    def get_scores(self) -> dict:
        """Get all duration scores.

        Returns:
            Dict with buckets and sweet_spot
        """
        scores = self._state.get("duration_scores")
        if scores is None:
            return {"buckets": {}, "sweet_spot": "25-34s", "total_n": 0}
        return scores
