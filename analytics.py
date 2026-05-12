"""
analytics.py — YouTube Performance Analytics + SEO Feedback Loop.
Fetches shorts performance → Rich terminal dashboard → feeds SEOLearner.
Run standalone:  python analytics.py
Pipeline hook:  auto-runs after upload
"""

import os
import json
import time
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import httplib2
import google_auth_httplib2

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.columns import Columns
from rich.layout import Layout

from utils.config import load_config
from utils.logger import get_logger, phase_tracker
from utils.ai_client import AIClient
from seo_learner import SEOLearner

console = Console()
cfg = load_config()
log = get_logger("analytics", cfg["logging"]["log_file"], cfg["logging"]["level"])
ai = AIClient()
seo_learner = SEOLearner()

SCOPES = [
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]


def _auth() -> Tuple[Optional[any], Optional[any]]:
    creds = None
    tf = "yt_analytics_token.json"
    if os.path.exists(tf):
        try:
            creds = Credentials.from_authorized_user_file(tf, SCOPES)
        except Exception:
            creds = None
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None
        if not creds:
            if not os.path.exists("client_secrets.json"):
                log.error("client_secrets.json missing — run OAuth setup first")
                return None, None
            flow = InstalledAppFlow.from_client_secrets_file("client_secrets.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open(tf, "w") as f:
            f.write(creds.to_json())
    base_http = httplib2.Http(timeout=60)
    auth_http = google_auth_httplib2.AuthorizedHttp(creds, http=base_http)
    yt = build("youtube", "v3", http=auth_http)
    ya = build("youtubeAnalytics", "v2", http=auth_http)
    return yt, ya


def fetch_shorts(days: int = 30) -> List[Dict]:
    yt, ya = _auth()
    if not yt:
        return []
    # Get uploads
    ch = yt.channels().list(part="contentDetails", mine=True).execute()["items"][0]
    pid = ch["contentDetails"]["relatedPlaylists"]["uploads"]
    items = yt.playlistItems().list(part="snippet", playlistId=pid, maxResults=50).execute()["items"]
    ids = [i["snippet"]["resourceId"]["videoId"] for i in items]

    cutoff = datetime.now() - timedelta(days=days)
    shorts = []
    for i in range(0, len(ids), 20):
        batch = ids[i:i+20]
        vids = yt.videos().list(part="snippet,statistics,contentDetails", id=",".join(batch)).execute()
        for v in vids.get("items", []):
            pub = datetime.strptime(v["snippet"]["publishedAt"], "%Y-%m-%dT%H:%M:%SZ")
            if pub < cutoff:
                continue
            dur = v["contentDetails"]["duration"]  # ISO 8601
            if "M" in dur and int(dur.split("M")[0].split("PT")[-1]) > 3:
                continue  # Not a short (>3 min)
            stats = v.get("statistics", {})
            shorts.append({
                "id": v["id"],
                "title": v["snippet"]["title"],
                "published": pub,
                "views": int(stats.get("viewCount", 0)),
                "likes": int(stats.get("likeCount", 0)),
                "comments": int(stats.get("commentCount", 0)),
                "tags": v["snippet"].get("tags", []),
            })
    shorts.sort(key=lambda x: x["views"], reverse=True)
    return shorts


def print_performance_dashboard(shorts: List[Dict]):
    if not shorts:
        console.print("[yellow]No shorts data to show[/]")
        return

    total_views = sum(s["views"] for s in shorts)
    total_likes = sum(s["likes"] for s in shorts)
    avg_views = total_views / max(1, len(shorts))
    avg_like_rate = sum(s["likes"] / max(1, s["views"]) for s in shorts) / max(1, len(shorts))

    # ─── HEADER ────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        Text("📈 SHORTS PERFORMANCE ANALYTICS", style="bold green", justify="center"),
        border_style="green", width=80,
    ))

    # ─── METRICS ───────────────────────────────────────────────────────────
    m = Table(title="Channel Summary (Last 30 Days)", box=box.ROUNDED, header_style="bold cyan")
    m.add_column("Metric", style="cyan")
    m.add_column("Value", justify="right")
    m.add_row("Shorts Analyzed", f"[bold]{len(shorts)}[/]")
    m.add_row("Total Views", f"[bold]{total_views:,}[/]")
    m.add_row("Avg Views/Short", f"{avg_views:,.0f}")
    m.add_row("Total Likes", f"{total_likes:,}")
    m.add_row("Avg Like Rate", f"{avg_like_rate*100:.1f}%")
    m.add_row("Top Short", f"'{shorts[0]['title'][:40]}' — {shorts[0]['views']:,} views")
    console.print(m)

    # ─── PER-SHORT TABLE ──────────────────────────────────────────────────
    t = Table(title="📊 Per-Short Breakdown", box=box.SIMPLE, header_style="bold magenta")
    t.add_column("#", justify="right", style="dim")
    t.add_column("Title", style="cyan", max_width=42)
    t.add_column("Views", justify="right")
    t.add_column("Likes", justify="right")
    t.add_column("Like %", justify="right")
    t.add_column("Engagement", justify="right")
    t.add_column("Published", style="dim")

    for i, s in enumerate(shorts, 1):
        lr = s["likes"] / max(1, s["views"]) * 100
        eng = (s["likes"] + s["comments"]) / max(1, s["views"]) * 100
        vc = "green" if s["views"] > avg_views * 1.5 else "yellow" if s["views"] > avg_views * 0.5 else "white"
        lc = "green" if lr > 5 else "yellow" if lr > 2 else "white"
        t.add_row(
            str(i),
            s["title"][:42],
            f"[{vc}]{s['views']:,}[/]",
            f"{s['likes']:,}",
            f"[{lc}]{lr:.1f}%[/]",
            f"{eng:.1f}%",
            s["published"].strftime("%d %b"),
        )
    console.print()
    console.print(t)


def ai_analyze(shorts: List[Dict]) -> Optional[str]:
    if len(shorts) < 4:
        return None
    top3 = shorts[:3]
    bot3 = shorts[-3:]
    prompt = f"""You are a YouTube Shorts performance analyst. Analyze these shorts and give 3-5 bullet points of what's working and what's not. Use Hinglish. Be specific — mention actual titles.

TOP PERFORMING (highest views):
{chr(10).join(f"- '{s['title']}' — {s['views']} views, {s['likes']} likes" for s in top3)}

BOTTOM PERFORMING (lowest views):
{chr(10).join(f"- '{s['title']}' — {s['views']} views, {s['likes']} likes" for s in bot3)}

Return ONLY 3-5 bullet points, no preamble:
- What patterns do the top shorts share?
- What's different about bottom shorts?
- Specific recommendations for next shorts."""
    try:
        return ai.generate_text(prompt, "You are a YouTube analytics expert. Return only bullet points. Use Hinglish.")
    except Exception as e:
        log.warning(f"AI analysis failed: {e}")
        return None


def feed_seo_learner(shorts: List[Dict]):
    for s in shorts:
        seo_learner.record_performance(
            clip_id=s["id"],
            title=s["title"],
            description="",
            hashtags=s.get("tags", []),
            analytics={
                "viewCount": s["views"],
                "likeCount": s["likes"],
                "commentCount": s["comments"],
            },
        )
    log.info(f"Fed {len(shorts)} shorts into SEO learner")


def print_seo_learnings():
    suggestions = seo_learner.get_seo_improvement_suggestions()
    if suggestions:
        console.print(Panel(
            Text("🧠 SEO LEARNINGS", style="bold yellow", justify="center"),
            border_style="yellow", width=80,
        ))
        for s in suggestions:
            emoji = "✅" if "perform" in s.lower() or "use more" in s.lower() else "⚠️"
            console.print(f"  {emoji} {s}")
        console.print()


def generate_daily_insights(output_dir: str = "logs"):
    phase_tracker.begin("Analytics — Fetching Shorts Performance")
    console.print()

    shorts = fetch_shorts(days=30)
    if not shorts:
        console.print("[yellow]No shorts found in last 30 days[/]")
        phase_tracker.end("done")
        return None

    print_performance_dashboard(shorts)
    feed_seo_learner(shorts)
    print_seo_learnings()

    analysis = ai_analyze(shorts)
    if analysis:
        console.print(Panel(
            Text("🤖 AI PERFORMANCE ANALYSIS", style="bold magenta", justify="center"),
            border_style="magenta", width=80,
        ))
        for line in analysis.strip().split("\n"):
            if line.strip():
                console.print(f"  {line.strip()}")
        console.print()

    # Save to file
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rp = Path(output_dir) / f"analytics_{datetime.now():%Y%m%d_%H%M%S}.json"
    rp.write_text(json.dumps(shorts, indent=2, default=str), encoding="utf-8")
    console.print(f"[dim]📄 Data saved: {rp}[/]")

    phase_tracker.end("done")
    return rp


if __name__ == "__main__":
    generate_daily_insights()
