"""
upload.py — Phase 7: Automated YouTube Shorts Uploader.

Uploads videos to YouTube with metadata generated in Phase 6.

NOTE: This uses YouTube Data API (not Drive API), so it has its own
auth via yt_token.json + client_secrets.json. This is intentionally
separate from the Drive auth in utils/drive_auth.py.
"""
import os
import json
import sys
import httplib2
import google_auth_httplib2
from pathlib import Path
from typing import Optional

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("upload", cfg["logging"]["log_file"], cfg["logging"]["level"])

try:
    import google.auth
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
except ImportError:
    log.error(
        "Missing YouTube API libraries. Run:\n"
        "  pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
    )
    sys.exit(1)

# YouTube API scopes (intentionally different from Drive scope)
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def get_authenticated_service():
    """
    Authenticate with YouTube Data API.
    Uses yt_token.json for saved credentials, client_secrets.json for OAuth flow.
    """
    creds = None

    # Load saved token
    if os.path.exists("yt_token.json"):
        try:
            creds = Credentials.from_authorized_user_file("yt_token.json", SCOPES)
        except Exception as e:
            log.warning(f"Failed to load yt_token.json: {e}")
            creds = None

    # Refresh or create new credentials
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                log.warning(f"Token refresh failed: {e}. Starting new auth flow.")
                creds = None

        if not creds:
            if not os.path.exists("client_secrets.json"):
                log.error(
                    "❌ client_secrets.json not found!\n"
                    "   Create a project in Google Cloud Console and download the OAuth secrets."
                )
                return None
            flow = InstalledAppFlow.from_client_secrets_file("client_secrets.json", SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open("yt_token.json", "w") as token:
            token.write(creds.to_json())

    # Build service with robust httplib2 transport
    base_http = httplib2.Http(timeout=60)
    auth_http = google_auth_httplib2.AuthorizedHttp(creds, http=base_http)
    return build("youtube", "v3", http=auth_http)


def upload_video(
    video_path: str,
    metadata_path: str,
    privacy: str = "private",
    publish_at: Optional[str] = None,
) -> Optional[str]:
    """
    Upload a single video to YouTube.

    Args:
        video_path: Path to the MP4 file
        metadata_path: Path to the SEO metadata JSON
        privacy: YouTube privacy status (private/unlisted/public)
        publish_at: ISO 8601 datetime for scheduled publishing

    Returns:
        YouTube video ID on success, None on failure
    """
    youtube = get_authenticated_service()
    if not youtube:
        return None

    # Load and validate SEO metadata
    with open(metadata_path, "r") as f:
        meta = json.load(f)

    title = meta.get("title", "Cricket Highlights #Shorts")
    description = meta.get("description", "")
    tags = meta.get("tags", [])

    log.info(f"🚀 Uploading to YouTube: {title}...")
    if publish_at:
        log.info(f"⏰ Scheduled for: {publish_at}")
        privacy = "private"  # Must be private to be scheduled

    body = {
        "snippet": {
            "title": title[:100],  # YouTube title limit
            "description": description[:5000],  # YouTube description limit
            "tags": tags[:500],  # YouTube tags limit
            "categoryId": cfg["youtube"]["category_id"],
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": cfg["youtube"]["self_declared_made_for_kids"],
        },
    }

    if publish_at:
        body["status"]["publishAt"] = publish_at

    insert_request = youtube.videos().insert(
        part=",".join(body.keys()),
        body=body,
        media_body=MediaFileUpload(video_path, chunksize=-1, resumable=True),
    )

    response = None
    while response is None:
        status, response = insert_request.next_chunk()
        if status:
            log.info(f"   Upload Progress: {int(status.progress() * 100)}%")

    video_id = response["id"]
    log.info(f"✅ Upload successful! Video ID: {video_id}")
    log.info(f"🔗 URL: https://youtu.be/{video_id}")
    return video_id


if __name__ == "__main__":
    # Example usage
    # upload_video("shorts/clip1.mp4", "shorts/clip1_metadata.json", "private")
    pass
