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
from seo_learner import SEOLearner

def _parse_iso8601_duration(dur: str) -> int:
    """Parse ISO 8601 duration string to seconds (robust to H/M/S combos)."""
    if not dur:
        return 0
    match = re.match(r'PT?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', dur)
    if not match:
        return 0
    h, m, s = match.groups()
    return int(h or 0) * 3600 + int(m or 0) * 60 + int(s or 0)


cfg = load_config()
log = get_logger("analytics", cfg["logging"]["log_file"], cfg["logging"]["level"])


def _auth() -> Optional[any]:
    """Authenticate using yt_token.json (shared with upload)."""
    tf = "yt_token.json"
    if not os.path.exists(tf):
        log.error("yt_token.json missing — run OAuth setup first")
        return None

    try:
        token_data = json.loads(open(tf).read())
        creds = Credentials(
            token=token_data.get("token"),
            refresh_token=token_data.get("refresh_token"),
            token_uri=token_data.get("token_uri"),
            client_id=token_data.get("client_id"),
            client_secret=token_data.get("client_secret"),
            scopes=token_data.get("scopes", []),
        )
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            # Save refreshed token
            token_data["token"] = creds.token
            json.dump(token_data, open(tf, "w"), indent=2)
            log.info("Token refreshed and saved")
        yt = build("youtube", "v3", credentials=creds)
        return yt
    except Exception as e:
        log.error(f"Auth failed: {e}")
        return None


def _fetch_all_recent(days: int = 30) -> Dict[str, List[Dict]]:
    """Fetch all recent content, split by type: videos, shorts, lives."""
    yt = _auth()
    if not yt:
        return {"videos": [], "shorts": [], "lives": []}

    ch = yt.channels().list(part="contentDetails", mine=True).execute()["items"][0]
    pid = ch["contentDetails"]["relatedPlaylists"]["uploads"]
    items = yt.playlistItems().list(part="snippet", playlistId=pid, maxResults=50).execute()["items"]
    ids = [i["snippet"]["resourceId"]["videoId"] for i in items]

    cutoff = datetime.now() - timedelta(days=days)
    videos, shorts, lives = [], [], []

    for i in range(0, len(ids), 20):
        batch = ids[i:i+20]
        vids = yt.videos().list(
            part="snippet,statistics,contentDetails,liveStreamingDetails",
            id=",".join(batch)
        ).execute()

        for v in vids.get("items", []):
            pub = datetime.strptime(v["snippet"]["publishedAt"], "%Y-%m-%dT%H:%M:%SZ")
            if pub < cutoff:
                continue

            stats = v.get("statistics", {})
            dur = v["contentDetails"]["duration"]
            seconds = _parse_iso8601_duration(dur)
            is_live = v.get("liveStreamingDetails") is not None

            entry = {
                "id": v["id"],
                "title": v["snippet"]["title"],
                "published": pub,
                "views": int(stats.get("viewCount", 0)),
                "likes": int(stats.get("likeCount", 0)),
                "comments": int(stats.get("commentCount", 0)),
                "tags": v["snippet"].get("tags", []),
                "duration_sec": seconds,
                "is_live": is_live,
            }

            if is_live:
                lives.append(entry)
            elif seconds <= 180:
                shorts.append(entry)
            else:
                videos.append(entry)

    videos.sort(key=lambda x: x["views"], reverse=True)
    shorts.sort(key=lambda x: x["views"], reverse=True)
    lives.sort(key=lambda x: x["views"], reverse=True)

    return {"videos": videos, "shorts": shorts, "lives": lives}


def fetch_shorts(days: int = 30) -> List[Dict]:
    """Fetch only shorts (< 3 min)."""
    return _fetch_all_recent(days)["shorts"]


def fetch_videos(days: int = 30) -> List[Dict]:
    """Fetch only full videos (> 3 min, not live)."""
    return _fetch_all_recent(days)["videos"]


def fetch_lives(days: int = 30) -> List[Dict]:
    """Fetch only live streams."""
    return _fetch_all_recent(days)["lives"]


def print_performance_dashboard(shorts: List[Dict], videos: List[Dict] = None, lives: List[Dict] = None):
    """Log performance summary (works headless — no rich console needed)."""
    for label, items in [("Shorts", shorts), ("Videos", videos or []), ("Lives", lives or [])]:
        if not items:
            continue
        total_v = sum(x["views"] for x in items)
        total_l = sum(x["likes"] for x in items)
        avg_v = total_v / max(1, len(items))
        top = items[0]
        log.info(f"📊 {label}: {len(items)} items, {total_v:,} total views, avg {avg_v:,.0f}, "
                 f"top: '{top['title'][:50]}' ({top['views']:,} views)")

    if shorts:
        for i, s in enumerate(shorts[:10], 1):
            lr = s["likes"] / max(1, s["views"]) * 100
            log.info(f"  #{i} {s['title'][:50]} — {s['views']:,} views, {lr:.1f}% like rate")


def feed_seo_learner(shorts: List[Dict], videos: List[Dict] = None, lives: List[Dict] = None):
    """Feed all content types into the SEO learner for pattern recognition."""
    learner = SEOLearner()
    all_items = []
    for label, items in [("short", shorts), ("video", videos or []), ("live", lives or [])]:
        for item in items:
            item["_content_type"] = label
            all_items.append(item)

    for s in all_items:
        provider = None
        model = None
        desc = ""
        tags = s.get("tags", [])

        # Search for local metadata in shorts directories
        meta_paths = [
            Path("shorts") / s["id"] / f"{s['id']}_metadata.json",
            Path("shorts") / f"{s['id']}_metadata.json",
        ]
        for d in Path("shorts").glob("*/"):
            meta_paths.append(d / f"{s['id']}_metadata.json")

        for mp in meta_paths:
            if mp.exists():
                try:
                    meta = json.loads(mp.read_text())
                    provider = meta.get("_generated_by_provider")
                    model = meta.get("_generated_by_model")
                    desc = meta.get("description", "")
                    if meta.get("hashtags"):
                        tags = meta["hashtags"]
                except Exception:
                    pass
                break

        learner.record_performance(
            clip_id=s["id"],
            title=s["title"],
            description=desc,
            hashtags=tags,
            analytics={
                "viewCount": s["views"],
                "likeCount": s["likes"],
                "commentCount": s["comments"],
                "content_type": s.get("_content_type", "short"),
            },
            provider=provider,
            model=model,
        )
    log.info(f"🧠 Fed {len(all_items)} items into SEO learner "
             f"({len(shorts)} shorts, {len(videos or [])} videos, {len(lives or [])} lives)")


def print_seo_learnings():
    learner = SEOLearner()
    suggestions = learner.get_seo_improvement_suggestions()
    for s in suggestions:
        log.info(f"  💡 {s}")


def generate_daily_insights(output_dir: str = "logs"):
    """Fetch all content types, feed learner, save report."""
    log.info("📊 Starting daily analytics fetch...")

    all_data = _fetch_all_recent(days=30)
    shorts = all_data["shorts"]
    videos = all_data["videos"]
    lives = all_data["lives"]

    total = len(shorts) + len(videos) + len(lives)
    if total == 0:
        log.warning("No content found in last 30 days")
        return None

    print_performance_dashboard(shorts, videos, lives)
    feed_seo_learner(shorts, videos, lives)
    print_seo_learnings()

    # Save report
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "shorts": shorts,
        "videos": videos,
        "lives": lives,
    }
    rp = Path(output_dir) / f"analytics_{datetime.now():%Y%m%d_%H%M%S}.json"
    rp.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log.info(f"📄 Analytics saved: {rp}")

    return rp


if __name__ == "__main__":
    generate_daily_insights()
