"""Analytics — aggregated metrics and trends from persistent stores."""

import json
import logging
import sqlite3
from pathlib import Path

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


def _count_from_clip_learner() -> dict:
    db = Path("clip_learner.db")
    if not db.exists():
        return {"total_clips": 0, "published": 0, "avg_score": 0.0, "total_views": 0}
    try:
        conn = sqlite3.connect(str(db))
        total = conn.execute("SELECT COUNT(*) FROM clip_performance").fetchone()[0]
        published = conn.execute("SELECT COUNT(*) FROM clip_performance WHERE youtube_video_id IS NOT NULL").fetchone()[0]
        avg = conn.execute("SELECT COALESCE(AVG(final_score), 0) FROM clip_performance").fetchone()[0]
        views = conn.execute("SELECT COALESCE(SUM(views), 0) FROM clip_performance").fetchone()[0]
        conn.close()
        return {"total_clips": total, "published": published, "avg_score": round(avg, 3), "total_views": views}
    except Exception:
        return {"total_clips": 0, "published": 0, "avg_score": 0.0, "total_views": 0}


def _count_from_self_learner() -> dict:
    db = Path("self_learner.db")
    if not db.exists():
        return {"memories": 0, "processed_events": 0}
    try:
        conn = sqlite3.connect(str(db))
        memories = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        events = conn.execute("SELECT COUNT(*) FROM processed_events").fetchone()[0]
        conn.close()
        return {"memories": memories, "processed_events": events}
    except Exception:
        return {"memories": 0, "processed_events": 0}


def _latest_exports(count: int = 5) -> list[dict]:
    export_dir = Path("shorts")
    if not export_dir.exists():
        return []
    dirs = sorted([d for d in export_dir.iterdir() if d.is_dir()], reverse=True)[:count]
    results = []
    for d in dirs:
        metas = list(d.glob("*_metadata.json"))
        clips = 0
        uploaded = 0
        for m in metas:
            clips += 1
            try:
                data = json.loads(m.read_text())
                if data.get("youtube_video_id"):
                    uploaded += 1
            except Exception:
                pass
        results.append({
            "export_id": d.name,
            "clips": clips,
            "uploaded": uploaded,
        })
    return results


def generate_daily_insights() -> dict:
    """Aggregate analytics from all persistent stores.

    Reads from clip_learner.db, self_learner.db, and export directories
    to provide real analytics data instead of depending on the in-memory
    DecisionStore (which was always empty cross-session).
    """
    try:
        clip_data = _count_from_clip_learner()
        learner_data = _count_from_self_learner()
        exports = _latest_exports(5)

        summary = {
            "total_clips": clip_data["total_clips"],
            "published": clip_data["published"],
            "avg_score": clip_data["avg_score"],
            "total_views": clip_data["total_views"],
            "memories": learner_data["memories"],
            "processed_events": learner_data["processed_events"],
            "recent_exports": len(exports),
        }

        log.info("analytics summary: %s", summary)
        return summary
    except Exception as e:
        log.warning("Analytics unavailable: %s", e)
        return {}
