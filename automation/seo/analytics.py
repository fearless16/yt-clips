"""Analytics — aggregated metrics and trends from persistent stores."""

import json
import logging
import re
import sqlite3
from datetime import date
from pathlib import Path

from automation._cache import TTLCache
from automation.memory.event_models import EventType
from automation.memory.decision_store import DecisionStore

log = logging.getLogger("analytics")
YT_API_CACHE = TTLCache(maxsize=4, ttl=300)
_YOUTUBE_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def _build_youtube_analytics_service():
    """Build the authorized Analytics API client, or return ``None``."""
    token_path = Path("yt_analytics_token.json")
    if not token_path.exists():
        log.warning("YouTube Analytics token not found at %s", token_path)
        return None
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        scopes = [
            "https://www.googleapis.com/auth/youtube.readonly",
            "https://www.googleapis.com/auth/yt-analytics.readonly",
        ]
        credentials = Credentials.from_authorized_user_file(str(token_path), scopes)
        if not credentials.valid and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
            token_path.write_text(credentials.to_json(), encoding="utf-8")
        if not credentials.valid:
            log.warning("YouTube Analytics credentials are invalid")
            return None
        return build("youtubeAnalytics", "v2", credentials=credentials)
    except Exception as exc:
        log.warning("YouTube Analytics client unavailable: %s", exc)
        return None


def _analytics_rows(service, video_ids: list[str]) -> list[dict]:
    """Fetch one analytics row per video ID with a five-minute response cache."""
    cache_key = f"{date.today().isoformat()}:{','.join(video_ids)}"
    cached = YT_API_CACHE.get(cache_key)
    if cached is not None:
        return cached

    response = service.reports().query(
        ids="channel==MINE",
        startDate="2005-01-01",
        endDate=date.today().isoformat(),
        metrics="views,engagedViews,averageViewDuration,averageViewPercentage",
        dimensions="video",
        filters=f"video=={','.join(video_ids)}",
    ).execute()
    headers = [header.get("name") for header in response.get("columnHeaders", [])]
    rows = [dict(zip(headers, row)) for row in response.get("rows", [])]
    YT_API_CACHE.set(cache_key, rows)
    return rows


def sync_clip_performance_from_youtube(learner=None, service=None) -> int:
    """Sync uploaded clip views and retention metrics into ``clip_learner.db``.

    Dependencies may be injected for tests. Authentication or API failures are
    non-fatal because analytics feedback must never block clip generation.
    """
    owns_learner = learner is None
    if owns_learner:
        from automation.clip_selection.clip_learner import ClipLearner
        learner = ClipLearner()
    try:
        clips = learner.get_all_clips()
        by_video: dict[str, list[str]] = {}
        for clip in clips:
            video_id = clip.get("youtube_video_id")
            clip_id = clip.get("clip_id")
            if (
                isinstance(video_id, str)
                and _YOUTUBE_VIDEO_ID_RE.fullmatch(video_id)
                and isinstance(clip_id, str)
            ):
                by_video.setdefault(video_id, []).append(clip_id)
        if not by_video:
            return 0

        service = service or _build_youtube_analytics_service()
        if service is None:
            return 0

        updated = 0
        video_ids = sorted(by_video)
        for offset in range(0, len(video_ids), 200):
            batch = video_ids[offset:offset + 200]
            for row in _analytics_rows(service, batch):
                video_id = str(row.get("video", ""))
                if video_id not in by_video:
                    continue
                views = max(0, int(row.get("views", 0) or 0))
                engaged = max(0, int(row.get("engagedViews", 0) or 0))
                retention = max(0.0, min(1.0, float(
                    row.get("averageViewPercentage", 0.0) or 0.0
                ) / 100.0))
                continued = max(0.0, min(1.0, engaged / views)) if views else 0.0
                avg_duration = max(0.0, float(row.get("averageViewDuration", 0.0) or 0.0))
                for clip_id in by_video[video_id]:
                    learner.update_performance(
                        clip_id=clip_id,
                        youtube_video_id=video_id,
                        views=views,
                        estimated_retention=retention,
                        completion_rate=continued,
                        avg_view_duration_seconds=avg_duration,
                    )
                    updated += 1
        log.info("Synced YouTube performance for %d clips", updated)
        return updated
    except Exception as exc:
        log.warning("YouTube performance sync failed: %s", exc)
        return 0
    finally:
        if owns_learner:
            learner.close()


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
        sync_clip_performance_from_youtube()
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
