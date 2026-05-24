"""
analytics.py — YouTube Performance Analytics + SEO Feedback Loop.
Fetches videos, shorts, and lives separately → feeds SEOLearner.
Run standalone:  python analytics.py
Worker hook:     auto-runs from worker.py
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import re

from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from utils.config import load_config
from utils.logger import get_logger
from .seo_learner import SEOLearner
from automation._cache import TTLCache

YT_API_CACHE = TTLCache(maxsize=4, ttl=300)
ANALYTICS_CACHE = TTLCache(maxsize=2, ttl=60)


def _parse_iso8601_duration(dur: str) -> int:
    """Parse ISO 8601 duration string to seconds (handles fractional seconds)."""
    if not dur:
        return 0
    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?', dur)
    if not match:
        return 0
    h, m, s = match.groups()
    total = int(h or 0) * 3600 + int(m or 0) * 60
    if s:
        total += int(float(s))
    return total


cfg = load_config()
log = get_logger("analytics", cfg["logging"]["log_file"], cfg["logging"]["level"])


def _auth() -> Optional[any]:
    """Authenticate with YouTube Data API via OAuth token."""
    cache_key = "yt_auth"
    cached = YT_API_CACHE.get(cache_key)
    if cached is not None:
        return cached
    token_path = cfg.get("youtube", {}).get("token_path", "yt_token.json")
    if not Path(token_path).exists():
        log.warning("Token not found: %s", token_path)
        return None
    try:
        creds = Credentials.from_authorized_user_file(token_path)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        if not creds or not creds.valid:
            return None
        service = build("youtube", "v3", credentials=creds, cache_discovery=False)
        YT_API_CACHE.set(cache_key, service)
        return service
    except Exception as e:
        log.warning("Auth failed: %s", e)
        return None


def _classify_content_type(
    duration: int,
    has_shorts_tag: bool = False,
    title: str = "",
    is_live: bool = False,
) -> str:
    """Classify a video as shorts / video / live based on duration + markers."""
    if is_live:
        return "live"
    title_lower = title.lower()
    has_shorts_in_title = "#shorts" in title_lower or "shorts" in title_lower.split()
    if has_shorts_tag or has_shorts_in_title:
        return "shorts"
    if duration <= 60:
        return "shorts"
    return "video"


def _fetch_all_recent(days: int = 30) -> Dict[str, List[Dict]]:
    """Fetch recent videos, shorts, and lives."""
    cache_key = f"all_recent_{days}"
    cached = ANALYTICS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    service = _auth()
    if not service:
        return {"shorts": [], "videos": [], "lives": []}
    channel_id = cfg.get("youtube", {}).get("channel_id", "")
    if not channel_id:
        log.warning("No channel_id configured")
        return {"shorts": [], "videos": [], "lives": []}
    try:
        published_after = (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z"
        all_video_ids = []
        page_token = None
        while True:
            req_kwargs = dict(
                part="snippet",
                channelId=channel_id,
                order="date",
                maxResults=50,
                publishedAfter=published_after,
            )
            if page_token:
                req_kwargs["pageToken"] = page_token
            request = service.search().list(**req_kwargs)
            response = request.execute()
            for item in response.get("items", []):
                all_video_ids.append(item["id"]["videoId"])
            page_token = response.get("nextPageToken")
            if not page_token:
                break

        if not all_video_ids:
            return {"shorts": [], "videos": [], "lives": []}

        shorts, videos, lives = [], [], []
        for i in range(0, len(all_video_ids), 50):
            batch = all_video_ids[i:i + 50]
            vid_request = service.videos().list(
                part="snippet,statistics,contentDetails",
                id=",".join(batch),
            )
            vid_response = vid_request.execute()
            for vid_item in vid_response.get("items", []):
                snippet = vid_item.get("snippet", {})
                stats = vid_item.get("statistics", {})
                duration = _parse_iso8601_duration(vid_item.get("contentDetails", {}).get("duration", ""))
                entry = {
                    "id": vid_item["id"],
                    "title": snippet.get("title", ""),
                    "published_at": snippet.get("publishedAt", ""),
                    "views": int(stats.get("viewCount", 0)),
                    "likes": int(stats.get("likeCount", 0)),
                    "comments": int(stats.get("commentCount", 0)),
                    "duration_seconds": duration,
                }
                ctype = _classify_content_type(
                    duration=duration,
                    has_shorts_tag="#shorts" in snippet.get("tags", []),
                    title=snippet.get("title", ""),
                    is_live=snippet.get("liveBroadcastContent") == "upcoming",
                )
                if ctype == "shorts":
                    shorts.append(entry)
                elif ctype == "live":
                    lives.append(entry)
                else:
                    videos.append(entry)

        result = {"shorts": shorts, "videos": videos, "lives": lives}
        ANALYTICS_CACHE.set(cache_key, result)
        return result
    except Exception as e:
        log.warning("Fetch failed: %s", e)
        return {"shorts": [], "videos": [], "lives": []}


def fetch_shorts(days: int = 30) -> List[Dict]:
    """Fetch recent Shorts."""
    data = _fetch_all_recent(days)
    return data.get("shorts", [])


def fetch_videos(days: int = 30) -> List[Dict]:
    """Fetch recent videos (non-Shorts, non-Live)."""
    data = _fetch_all_recent(days)
    return data.get("videos", [])


def fetch_lives(days: int = 30) -> List[Dict]:
    """Fetch recent Live streams."""
    data = _fetch_all_recent(days)
    return data.get("lives", [])


def print_performance_dashboard(shorts: List[Dict], videos: List[Dict] = None, lives: List[Dict] = None):
    """Print a human-readable performance dashboard."""
    videos = videos or []
    lives = lives or []
    log.info("===== YouTube Performance Dashboard =====")
    if shorts:
        total_shorts_views = sum(s["views"] for s in shorts)
        avg_shorts_views = total_shorts_views / len(shorts)
        log.info("Shorts (%d): total_views=%d, avg_views=%.1f", len(shorts), total_shorts_views, avg_shorts_views)
        top_short = max(shorts, key=lambda s: s["views"])
        log.info("  Top Short: %s (%d views)", top_short["title"][:60], top_short["views"])
    if videos:
        total_vid_views = sum(v["views"] for v in videos)
        avg_vid_views = total_vid_views / len(videos) if videos else 0
        log.info("Videos (%d): total_views=%d, avg_views=%.1f", len(videos), total_vid_views, avg_vid_views)
    if lives:
        log.info("Lives: %d upcoming", len(lives))
    log.info("=" * 40)


def feed_seo_learner(shorts: List[Dict], videos: List[Dict] = None, lives: List[Dict] = None):
    """Feed analytics data to the SEO learner for pattern learning.

    Attempts to load per-clip metadata JSON to extract provider/model,
    hashtags, and search_terms for richer learning.
    """
    videos = videos or []
    lives = lives or []
    learner = SEOLearner()
    feeds = 0
    for item in shorts + videos + lives:
        try:
            clip_id = item["id"]
            desc = ""
            hashtags = []
            search_terms = []
            tags = []
            provider = None
            model = None

            shorts_dir = Path("shorts")
            for meta_path in shorts_dir.rglob(f"{clip_id}_metadata.json"):
                if meta_path.is_file():
                    try:
                        meta = json.loads(meta_path.read_text())
                        desc = meta.get("description", "")
                        hashtags = meta.get("hashtags", [])
                        search_terms = meta.get("search_terms", [])
                        tags = meta.get("tags", [])
                        provider = meta.get("_generated_by_provider")
                        model = meta.get("_generated_by_model")
                    except Exception:
                        pass
                    break

            learner.record_performance(
                clip_id=clip_id,
                title=item["title"],
                description=desc,
                hashtags=hashtags,
                tags=tags,
                search_terms=search_terms,
                analytics={
                    "viewCount": item["views"],
                    "likeCount": item["likes"],
                    "commentCount": item["comments"],
                },
                provider=provider,
                model=model,
            )
            feeds += 1
        except Exception as e:
            log.warning("Feed error for %s: %s", item.get("id", "?"), e)
    log.info("Fed %d items to SEOLearner", feeds)


def print_seo_learnings():
    """Print current SEO learning insights."""
    try:
        from .seo_learner import generate_performance_report
        report = generate_performance_report()
        print(report)
    except Exception as e:
        log.warning("SEO learnings not available: %s", e)


def generate_daily_insights(output_dir: str = "logs"):
    """Generate daily analytics report and feed SEO learner."""
    data = _fetch_all_recent()
    shorts = data.get("shorts", [])
    videos = data.get("videos", [])
    lives = data.get("lives", [])

    print_performance_dashboard(shorts, videos, lives)
    feed_seo_learner(shorts, videos, lives)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report_path = Path(output_dir) / f"analytics_{datetime.now().strftime('%Y-%m-%d')}.json"
    with open(report_path, "w") as f:
        json.dump(data, f, indent=2)
    log.info("Analytics saved → %s", report_path)


if __name__ == "__main__":
    generate_daily_insights()
