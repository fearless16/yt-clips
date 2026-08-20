"""YouTube adapter: exact Shorts shelf joined to owner Analytics metrics."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from shorts_intelligence.models import PerformanceSnapshot, ShortRecord


WATCH_METRICS = (
    "engagedViews", "views", "estimatedMinutesWatched",
    "averageViewDuration", "averageViewPercentage",
)
ENGAGEMENT_METRICS = ("likes", "comments", "shares")
SUBSCRIBER_METRICS = ("subscribersGained", "subscribersLost")


@dataclass(frozen=True, slots=True)
class SourceConfig:
    """Runtime settings for one owned YouTube channel."""

    channel_id: str
    handle: str
    token_path: str = "yt_analytics_token.json"
    local_metadata_root: str = "shorts"
    yt_dlp_path: str = ""
    analytics_start_date: str = "2006-01-01"


@dataclass(slots=True)
class IngestionBatch:
    """One complete source pull ready for an atomic store transaction."""

    records: list[ShortRecord]
    snapshots: list[PerformanceSnapshot]
    shelf_count: int
    issues: list[str] = field(default_factory=list)


class YouTubeShortsSource:
    """Fetch only shelf-proven Shorts, then enrich them from owner APIs."""

    def __init__(
        self,
        config: SourceConfig,
        *,
        shelf_loader: Callable[[], list[dict[str, Any]]] | None = None,
        video_loader: Callable[[list[str]], list[dict[str, Any]]] | None = None,
        analytics_loader: Callable[[list[str], tuple[str, ...]], list[dict[str, Any]]] | None = None,
        local_metadata_loader: Callable[[], dict[str, dict[str, Any]]] | None = None,
    ) -> None:
        self.config = config
        self._shelf_loader = shelf_loader or self._load_shelf
        self._video_loader = video_loader or self._load_videos
        self._analytics_loader = analytics_loader or self._load_analytics
        self._local_metadata_loader = local_metadata_loader or self._load_local_metadata
        self._youtube = None
        self._analytics = None

    def fetch(self, *, captured_at: str | None = None) -> IngestionBatch:
        """Fetch the full Shorts catalog and a point-in-time metric snapshot."""
        captured = captured_at or datetime.now(timezone.utc).isoformat()
        shelf = self._shelf_loader()
        ids = list(dict.fromkeys(str(item.get("id", "")) for item in shelf if item.get("id")))
        local = self._local_metadata_loader()
        issues: list[str] = []

        videos: list[dict[str, Any]] = []
        for chunk in _chunks(ids, 50):
            videos.extend(self._video_loader(chunk))

        records: list[ShortRecord] = []
        by_id: dict[str, dict[str, Any]] = {}
        for video in videos:
            video_id = str(video.get("id", ""))
            snippet = video.get("snippet", {})
            if snippet.get("channelId") != self.config.channel_id:
                issues.append(f"cross_channel:{video_id}")
                continue
            if video_id not in ids:
                issues.append(f"not_on_shelf:{video_id}")
                continue
            statistics = video.get("statistics", {})
            by_id[video_id] = video
            records.append(ShortRecord(
                video_id=video_id,
                channel_id=self.config.channel_id,
                title=str(snippet.get("title", "")),
                description=str(snippet.get("description", "")),
                published_at=str(snippet.get("publishedAt", "")),
                duration_seconds=self.parse_duration(
                    str(video.get("contentDetails", {}).get("duration", "PT0S"))
                ),
                is_short=True,
                category_id=str(snippet.get("categoryId", "")),
                tags=tuple(str(tag) for tag in snippet.get("tags", []) if tag),
                source_url=f"https://www.youtube.com/shorts/{video_id}",
                local_metadata=local.get(video_id, {}),
                raw={"statistics": statistics, "status": video.get("status", {})},
            ))

        metrics: dict[str, dict[str, Any]] = {record.video_id: {} for record in records}
        metric_ids = list(metrics)
        for group in (WATCH_METRICS, ENGAGEMENT_METRICS, SUBSCRIBER_METRICS):
            for chunk in _chunks(metric_ids, 200):
                for row in self._analytics_loader(chunk, group):
                    video_id = str(row.get("video", ""))
                    if video_id in metrics:
                        metrics[video_id].update(row)

        snapshots: list[PerformanceSnapshot] = []
        for record in records:
            row = metrics[record.video_id]
            public = by_id[record.video_id].get("statistics", {})
            snapshots.append(PerformanceSnapshot(
                video_id=record.video_id,
                captured_at=captured,
                engaged_views=_int(row.get("engagedViews")),
                views=_int(row.get("views", public.get("viewCount"))),
                estimated_minutes_watched=_float(row.get("estimatedMinutesWatched")),
                average_view_duration_seconds=_float(row.get("averageViewDuration")),
                average_view_percentage=_float(row.get("averageViewPercentage")),
                likes=_int(row.get("likes", public.get("likeCount"))),
                comments=_int(row.get("comments", public.get("commentCount"))),
                shares=_int(row.get("shares")),
                subscribers_gained=_int(row.get("subscribersGained")),
                subscribers_lost=_int(row.get("subscribersLost")),
                raw=row,
            ))
        missing = set(ids) - {record.video_id for record in records}
        issues.extend(f"metadata_missing:{video_id}" for video_id in sorted(missing))
        return IngestionBatch(records, snapshots, len(ids), issues)

    @staticmethod
    def parse_duration(value: str) -> int:
        """Parse the ISO-8601 duration form returned by videos.list."""
        match = re.fullmatch(r"P(?:(\d+)D)?T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", value)
        if not match:
            raise ValueError(f"invalid YouTube duration: {value}")
        days, hours, minutes, seconds = (int(part or 0) for part in match.groups())
        return days * 86_400 + hours * 3_600 + minutes * 60 + seconds

    def _load_shelf(self) -> list[dict[str, Any]]:
        executable = self.config.yt_dlp_path or _default_ytdlp()
        url = f"https://www.youtube.com/{self.config.handle}/shorts"
        result = subprocess.run(
            [executable, "--flat-playlist", "--dump-single-json", url, "--no-warnings"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
            check=True,
        )
        payload = json.loads(result.stdout)
        if payload.get("id") not in {self.config.channel_id, None}:
            raise ValueError("Shorts shelf belongs to a different channel")
        return list(payload.get("entries") or [])

    def _services(self):
        if self._youtube is not None and self._analytics is not None:
            return self._youtube, self._analytics
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        scopes = [
            "https://www.googleapis.com/auth/youtube.readonly",
            "https://www.googleapis.com/auth/yt-analytics.readonly",
        ]
        credentials = Credentials.from_authorized_user_file(self.config.token_path, scopes=scopes)
        if not credentials.valid:
            credentials.refresh(Request())
            Path(self.config.token_path).write_text(
                credentials.to_json(), encoding="utf-8"
            )
        self._youtube = build("youtube", "v3", credentials=credentials, cache_discovery=False)
        self._analytics = build("youtubeAnalytics", "v2", credentials=credentials, cache_discovery=False)
        return self._youtube, self._analytics

    def _load_videos(self, ids: list[str]) -> list[dict[str, Any]]:
        youtube, _ = self._services()
        response = youtube.videos().list(
            part="snippet,contentDetails,statistics,status",
            id=",".join(ids),
        ).execute()
        return list(response.get("items") or [])

    def _load_analytics(
        self,
        ids: list[str],
        metrics: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        _, analytics = self._services()
        response = analytics.reports().query(
            ids="channel==MINE",
            startDate=self.config.analytics_start_date,
            endDate=date.today().isoformat(),
            dimensions="video",
            filters="video==" + ",".join(ids),
            metrics=",".join(metrics),
            maxResults=200,
        ).execute()
        headers = [header["name"] for header in response.get("columnHeaders", [])]
        return [dict(zip(headers, row)) for row in response.get("rows", [])]

    def _load_local_metadata(self) -> dict[str, dict[str, Any]]:
        root = Path(self.config.local_metadata_root)
        result: dict[str, dict[str, Any]] = {}
        if not root.exists():
            return result
        for path in root.glob("**/*_metadata.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            video_id = payload.get("youtube_video_id")
            if video_id:
                result[str(video_id)] = payload
        return result


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for offset in range(0, len(values), size):
        yield values[offset:offset + size]


def _int(value: Any) -> int:
    return max(0, int(value or 0))


def _float(value: Any) -> float:
    return max(0.0, float(value or 0.0))


def _default_ytdlp() -> str:
    bundled = Path(".venv/Scripts/yt-dlp.exe")
    return str(bundled) if bundled.exists() else "yt-dlp"
