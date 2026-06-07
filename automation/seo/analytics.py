"""Analytics — aggregated metrics and trends from the event store."""

import json
import logging

from automation.memory.event_models import EventType
from automation.memory.decision_store import DecisionStore

log = logging.getLogger("analytics")


class Analytics:
    def __init__(self, decision_store: DecisionStore) -> None:
        self._store = decision_store

    def get_metrics(self, clip_id: str) -> dict:
        events = self._store.get_events(clip_id=clip_id)
        events_by_type: dict[str, int] = {}
        feedback_count = 0
        ratings: list[float] = []

        for event in events:
            et = event.event_type.value
            events_by_type[et] = events_by_type.get(et, 0) + 1
            if event.event_type == EventType.metrics_received:
                feedback_count += 1
                try:
                    payload = json.loads(event.payload_json)
                    rating = payload.get("rating")
                    if rating is not None:
                        ratings.append(float(rating))
                except (json.JSONDecodeError, TypeError):
                    pass

        avg_rating = sum(ratings) / len(ratings) if ratings else 0.0

        return {
            "total_events": len(events),
            "events_by_type": events_by_type,
            "feedback_count": feedback_count,
            "avg_rating": avg_rating,
        }

    def get_summary(self) -> dict:
        events = self._store.get_all_events()
        clip_ids = set(e.clip_id for e in events)
        published_count = sum(1 for e in events if e.event_type == EventType.published)
        scores: list[float] = []

        for e in events:
            if e.event_type == EventType.candidate_scored:
                try:
                    payload = json.loads(e.payload_json)
                    score = payload.get("score")
                    if score is not None:
                        scores.append(float(score))
                except (json.JSONDecodeError, TypeError):
                    pass

        avg_score = sum(scores) / len(scores) if scores else 0.0

        return {
            "total_clips": len(clip_ids),
            "total_events": len(events),
            "published_count": published_count,
            "avg_score": avg_score,
        }

    def get_trends(self, days: int = 7) -> dict:
        events = self._store.get_all_events()
        trends: dict[str, int] = {}
        for event in events:
            et = event.event_type.value
            trends[et] = trends.get(et, 0) + 1
        return trends


def generate_daily_insights() -> dict:
    """Convenience function: instantiate Analytics and return summary.

    Called by pipeline.py and worker.py for analytics cycle.
    """
    try:
        store = DecisionStore()
        analytics = Analytics(store)
        summary = analytics.get_summary()
        log.info("analytics summary: %s", summary)
        return summary
    except Exception as e:
        log.warning("Analytics unavailable: %s", e)
        return {}
