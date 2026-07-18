"""
re_seo.py — Re-generate SEO for already-uploaded YouTube videos.
Fetches transcript + generates SEO with minimax-m3 → updates metadata (no re-upload).

Usage:
    python re_seo.py <URL_or_ID> [<URL_or_ID> ...]
"""
import _fix_encoding  # noqa: F401 — force UTF-8 on Windows cp1252

import json
import os
import re
import sys
import time
from pathlib import Path

from utils.ai_client import AIClient
from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("re_seo", cfg["logging"]["log_file"], cfg["logging"]["level"])

SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]


def extract_video_id(url: str) -> str:
    patterns = [
        r"(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/watch\?v=)([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"^([a-zA-Z0-9_-]{11})$",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    raise ValueError(f"Could not extract video ID from: {url}")


def _token_has_required_scope(token_data: dict) -> bool:
    actual_scopes = token_data.get("scopes", [])
    if isinstance(actual_scopes, str):
        actual_scopes = [actual_scopes]
    return SCOPES[0] in actual_scopes


def get_youtube_service():
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
        from google_auth_oauthlib.flow import InstalledAppFlow
        import google_auth_httplib2
        import httplib2
    except ImportError:
        print("Missing deps: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
        sys.exit(1)

    tokens_path = Path("yt_channel_token.json")

    def _do_oauth():
        print()
        print("Opening browser for YouTube authorization...")
        print("(requires youtube.force-ssl scope to update metadata)")
        client_secrets = Path("client_secrets.json")
        if not client_secrets.exists():
            print("No client_secrets.json found — cannot authenticate")
            print("Copy client_secrets.json from Google Cloud Console to this directory.")
            return None
        flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets), SCOPES)
        new_creds = flow.run_local_server(open_browser=True, open_browser_timeout_seconds=120)
        with open("yt_channel_token.json", "w", encoding="utf-8") as f:
            f.write(new_creds.to_json())
        print("Token saved to yt_channel_token.json")
        return new_creds

    if tokens_path.exists():
        with open(tokens_path) as f:
            token_data = json.load(f)
        creds = Credentials.from_authorized_user_info(token_data, SCOPES)

        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                with open("yt_channel_token.json", "w", encoding="utf-8") as f:
                    f.write(creds.to_json())
            except Exception as e:
                print(f"Token refresh failed: {e}")
                creds = None

        if creds and creds.valid and _token_has_required_scope(token_data):
            base_http = httplib2.Http(timeout=60)
            auth_http = google_auth_httplib2.AuthorizedHttp(creds, http=base_http)
            return build("youtube", "v3", http=auth_http)

    creds = _do_oauth()
    if not creds:
        return None

    base_http = httplib2.Http(timeout=60)
    auth_http = google_auth_httplib2.AuthorizedHttp(creds, http=base_http)
    return build("youtube", "v3", http=auth_http)


def get_video_snippet(youtube, video_id: str) -> dict:
    response = youtube.videos().list(part="snippet", id=video_id).execute()
    items = response.get("items", [])
    if not items:
        print(f"  Video {video_id} not found")
        return {}
    return items[0]["snippet"]


def get_transcript(video_id: str) -> str:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)
        data = transcript.to_raw_data()
        return " ".join(segment["text"] for segment in data)
    except Exception as e:
        log.warning("Transcript not available for %s: %s", video_id, e)
        return ""


def generate_seo(title: str, transcript: str) -> dict:
    ai = AIClient()

    system = (
        "You are a YouTube SEO expert for cricket content. "
        "Generate viral-optimized metadata for YouTube Shorts. "
        "Only use player names, teams, and events from the transcript. "
        "NEVER invent or hallucinate. "
        "Return ONLY valid JSON — no markdown, no explanation."
    )

    prompt = f"""CONTEXT:
  Video Title: {title}
  Transcript: {transcript or "(no transcript available, infer from title only)"}

TASK: Generate YouTube SEO metadata for this Shorts video.

You MUST return valid JSON (no markdown, no other text):
{{{{
  "title": "<max 80 chars, engaging English title with emojis>",
  "description": "<full SEO-rich English description following the format below>",
  "tags": ["<max 15 comma-separated search tags>"],
  "hashtags": ["<max 5 hashtags>"]
}}}}

TITLE (max 80 chars, English with emojis):
- Start with the MOST EXCITING moment of this clip
- Example: "Kohli SMASHES 120m Six! RCB vs MI IPL 2026 🔥"
- Must describe THIS specific video content
- End with 1-2 relevant emojis

DESCRIPTION FORMAT (full English, SEO-rich, with line breaks):
First line: Brief summary of the key moment
Then section with match/event context and key players
Then what makes this moment exciting
Call to action asking viewers to subscribe
Relevant emojis throughout
Line breaks between sections for readability

TAGS (max 15):
Player names, team names, tournament, keywords from the video

HASHTAGS (max 5):
Include #Shorts and event/cricket related hashtags

Generate the absolute best SEO to maximize reach."""

    print(f"  Generating SEO with minimax-m3...")
    start = time.time()
    response = ai.generate_text(prompt, system_instruction=system, prefer_model="minimax-m3")
    elapsed = time.time() - start
    print(f"  AI response in {elapsed:.1f}s ({len(response)} chars)")

    json_match = re.search(r'\{[\s\S]*\}', response)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    print(f"  Could not parse AI response as JSON. Raw: {response[:300]}")
    return {"title": title, "description": "", "tags": [], "hashtags": []}


def _ensure_list(val):
    if isinstance(val, str):
        return [val]
    if isinstance(val, list):
        return val
    return []


def update_metadata(youtube, video_id: str, seo: dict, current_snippet: dict):
    title = seo.get("title", current_snippet.get("title", ""))[:100]
    description = seo.get("description", current_snippet.get("description", ""))
    tags = _ensure_list(seo.get("tags", []))
    hashtags = _ensure_list(seo.get("hashtags", []))
    all_tags = list(dict.fromkeys(tags + hashtags))

    body = {
        "id": video_id,
        "snippet": {
            "title": title,
            "description": description[:5000],
            "tags": all_tags[:500],
            "categoryId": current_snippet.get("categoryId", "22"),
        }
    }

    print(f"  Updating YouTube metadata...")
    print(f"    Title ({len(title)} chars): {title[:80]}")
    print(f"    Description: {len(description)} chars")
    print(f"    Tags: {len(all_tags)}")

    youtube.videos().update(part="snippet", body=body).execute()
    print(f"  ✅ Metadata updated for {video_id}")


QUOTA_SLEEP_SECONDS = 86400  # 24h


def main():
    urls = sys.argv[1:]
    if not urls:
        print(__doc__.strip())
        sys.exit(1)

    video_ids = [extract_video_id(u) for u in urls]
    print(f"Processing {len(video_ids)} video(s): {video_ids}")

    youtube = get_youtube_service()
    if not youtube:
        print("Failed to authenticate.")
        sys.exit(1)

    api_quota_exceeded = False

    for i, vid in enumerate(video_ids):
        if api_quota_exceeded:
            print(f"\n  ⏭️  Skipping {vid} (quota exceeded)")
            continue

        print(f"\n{'='*50}")
        print(f"[{i+1}/{len(video_ids)}] {vid}")
        print(f"{'='*50}")

        try:
            snippet = get_video_snippet(youtube, vid)
            if not snippet:
                continue
            title = snippet.get("title", "")
            print(f"  Current title: {title[:80]}")

            transcript = get_transcript(vid)
            if transcript:
                print(f"  Transcript: {len(transcript)} chars")
            else:
                print(f"  No transcript available")

            seo = generate_seo(title, transcript)
            update_metadata(youtube, vid, seo, snippet)

            if i < len(video_ids) - 1:
                print("  Waiting 3s before next video...")
                time.sleep(3)

        except Exception as e:
            emsg = str(e).lower()
            if "quota" in emsg or "quotaExceeded" in emsg:
                print(f"  ❌ YouTube API quota exceeded. Remaining videos skipped.")
                print(f"  ⏳ Quota resets ~24h after your last API call.")
                api_quota_exceeded = True
            else:
                print(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
