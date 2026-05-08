import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import httplib2
import google_auth_httplib2

try:
    from googleapiclient.discovery import build
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
except ImportError:
    pass

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("analytics", cfg["logging"]["log_file"], cfg["logging"]["level"])

# Need read-only access to channel videos and analytics
SCOPES = [
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/yt-analytics.readonly"
]

def get_analytics_services():
    """Authenticate and return both YouTube Data API and YouTube Analytics API services."""
    creds = None
    token_file = "yt_analytics_token.json"

    if os.path.exists(token_file):
        try:
            creds = Credentials.from_authorized_user_file(token_file, SCOPES)
        except Exception as e:
            log.warning(f"Failed to load {token_file}: {e}")
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                log.warning(f"Token refresh failed: {e}. Starting new auth flow.")
                creds = None

        if not creds:
            if not os.path.exists("client_secrets.json"):
                log.error("❌ client_secrets.json not found! Cannot fetch analytics.")
                return None, None
            try:
                flow = InstalledAppFlow.from_client_secrets_file("client_secrets.json", SCOPES)
                creds = flow.run_local_server(port=0)
            except Exception as e:
                log.error(f"OAuth flow failed: {e}")
                return None, None

        with open(token_file, "w") as token:
            token.write(creds.to_json())

    base_http = httplib2.Http(timeout=60)
    auth_http = google_auth_httplib2.AuthorizedHttp(creds, http=base_http)
    
    youtube = build("youtube", "v3", http=auth_http)
    youtube_analytics = build("youtubeAnalytics", "v2", http=auth_http)
    
    return youtube, youtube_analytics


def generate_daily_insights(output_dir: str = "logs"):
    """
    Fetch recent Shorts performance and generate daily insights.
    Compares views, likes, and analyzes title/hashtag performance.
    """
    log.info("📊 Generating daily YouTube performance insights...")
    youtube, yt_analytics = get_analytics_services()
    if not youtube:
        return
        
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report_path = Path(output_dir) / f"daily_insights_{time.strftime('%Y%m%d')}.md"

    try:
        # 1. Get Channel ID and Uploads Playlist
        channel_resp = youtube.channels().list(part="contentDetails", mine=True).execute()
        if not channel_resp.get("items"):
            log.error("No channel found for the authenticated user.")
            return
            
        uploads_playlist_id = channel_resp["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
        channel_id = channel_resp["items"][0]["id"]
        
        # 2. Get last 20 videos
        playlist_resp = youtube.playlistItems().list(
            part="snippet",
            playlistId=uploads_playlist_id,
            maxResults=20
        ).execute()
        
        video_ids = [item["snippet"]["resourceId"]["videoId"] for item in playlist_resp.get("items", [])]
        
        if not video_ids:
            log.warning("No videos found on the channel.")
            return

        # 3. Get basic stats for these videos
        videos_resp = youtube.videos().list(
            part="snippet,statistics",
            id=",".join(video_ids)
        ).execute()
        
        videos_data = []
        for v in videos_resp.get("items", []):
            stats = v.get("statistics", {})
            videos_data.append({
                "id": v["id"],
                "title": v["snippet"]["title"],
                "published_at": v["snippet"]["publishedAt"],
                "views": int(stats.get("viewCount", 0)),
                "likes": int(stats.get("likeCount", 0)),
                "comments": int(stats.get("commentCount", 0)),
                "tags": v["snippet"].get("tags", [])
            })
            
        # 4. Try to get Analytics data (last 7 days channel level)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        
        channel_metrics = {}
        try:
            analytics_resp = yt_analytics.reports().query(
                ids=f"channel==MINE",
                startDate=start_date,
                endDate=end_date,
                metrics="views,estimatedMinutesWatched,averageViewDuration",
                dimensions="day",
                sort="day"
            ).execute()
            channel_metrics = analytics_resp
        except Exception as e:
            log.warning(f"Could not fetch advanced analytics (might need 24h data propagation): {e}")

        # 5. Build Insights Report
        videos_data.sort(key=lambda x: x["views"], reverse=True)
        top_videos = videos_data[:3]
        
        lines = [
            f"# 📈 YouTube Daily Insights ({datetime.now().strftime('%Y-%m-%d')})",
            "## 🏆 Top Performing Recent Uploads",
        ]
        
        for i, v in enumerate(top_videos, 1):
            lines.append(f"### {i}. {v['title']}")
            lines.append(f"- **Views:** {v['views']:,}")
            lines.append(f"- **Likes:** {v['likes']:,} ({(v['likes']/max(1, v['views'])*100):.1f}% Like Rate)")
            lines.append(f"- **Tags used:** {', '.join(v['tags'][:5])}...")
            lines.append("")
            
        # Analyze hashtag performance (simple frequency of tags in top 5 vs bottom 5)
        lines.append("## 🔍 Style & Tag Analysis")
        if len(videos_data) >= 6:
            top_tags = [t for v in videos_data[:3] for t in v["tags"]]
            bot_tags = [t for v in videos_data[-3:] for t in v["tags"]]
            
            # Very basic insight
            lines.append("Top performing tags recently:")
            lines.append(f" `{', '.join(list(set(top_tags))[:10])}`")
            lines.append("")
        
        lines.append("## ⏱️ Upload Timing Analysis")
        lines.append("Review the publishing hours of your top videos to find the best upload window.")
        for v in top_videos:
            pub_time = datetime.strptime(v["published_at"], "%Y-%m-%dT%H:%M:%SZ")
            lines.append(f"- {v['title'][:30]}... was published at **{pub_time.strftime('%H:%M UTC')}**")

        report_content = "\n".join(lines)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
            
        log.info(f"✅ Daily insights generated at: {report_path}")
        return str(report_path)

    except Exception as e:
        log.error(f"Failed to generate insights: {e}")
        return None

if __name__ == "__main__":
    generate_daily_insights()
