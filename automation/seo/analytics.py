"""Runtime telemetry plus the canonical Shorts Intelligence adapter."""

from __future__ import annotations

import json

from automation.memory.decision_store import DecisionStore
from automation.memory.event_models import EventType


class Analytics:
    """In-process pipeline telemetry; not a learning database."""

    def __init__(self, decision_store: DecisionStore) -> None:
        self._store = decision_store

    def get_metrics(self, clip_id: str) -> dict:
        events = self._store.get_events(clip_id=clip_id)
        events_by_type: dict[str, int] = {}
        ratings: list[float] = []
        for event in events:
            name = event.event_type.value
            events_by_type[name] = events_by_type.get(name, 0) + 1
            if event.event_type == EventType.metrics_received:
                try:
                    rating = json.loads(event.payload_json).get("rating")
                    if rating is not None:
                        ratings.append(float(rating))
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
        return {
            "total_events": len(events),
            "events_by_type": events_by_type,
            "feedback_count": events_by_type.get(EventType.metrics_received.value, 0),
            "avg_rating": sum(ratings) / len(ratings) if ratings else 0.0,
        }

    def get_summary(self) -> dict:
        events = self._store.get_all_events()
        scores: list[float] = []
        for event in events:
            if event.event_type == EventType.candidate_scored:
                try:
                    score = json.loads(event.payload_json).get("score")
                    if score is not None:
                        scores.append(float(score))
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
        return {
            "total_clips": len({event.clip_id for event in events}),
            "total_events": len(events),
            "published_count": sum(
                event.event_type == EventType.published for event in events
            ),
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
        }

    def get_trends(self, days: int = 7) -> dict:
        del days
        trends: dict[str, int] = {}
        for event in self._store.get_all_events():
            name = event.event_type.value
            trends[name] = trends.get(name, 0) + 1
        return trends


def sync_clip_performance_from_youtube(*, config_path: str = "config.yaml") -> int:
    """Sync the exact channel Shorts shelf and real Analytics outcomes."""
    from shorts_intelligence.config import RuntimeConfig
    from shorts_intelligence.learner import BayesianSegmentLearner
    from shorts_intelligence.service import ShortsIntelligence
    from shorts_intelligence.store import ShortsStore
    from shorts_intelligence.youtube_source import YouTubeShortsSource

    config = RuntimeConfig.from_yaml(config_path)
    if not config.enabled:
        return 0
    with ShortsStore(config.db_path, channel_id=config.channel_id) as store:
        result = ShortsIntelligence(
            store=store,
            source=YouTubeShortsSource(config.source_config()),
            learner=BayesianSegmentLearner(config.learner),
        ).sync_and_fit()
    return int(result["snapshots_added"])


def generate_daily_insights(*, config_path: str = "config.yaml") -> dict:
    """Return compact canonical status without hidden network calls."""
    from shorts_intelligence.config import RuntimeConfig
    from shorts_intelligence.store import ShortsStore

    config = RuntimeConfig.from_yaml(config_path)
    if not config.enabled:
        return {"status": "disabled"}
    with ShortsStore(config.db_path, channel_id=config.channel_id) as store:
        summary = store.summary()
        model = store.latest_model()
        return {
            **summary,
            "production_records": store.production_count(),
            "model_observations": int(model["observations"]) if model else 0,
            "recommendations": len(store.active_recommendations()),
        }
