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
import time
import subprocess
import httplib2
import google_auth_httplib2
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("upload", cfg["logging"]["log_file"], cfg["logging"]["level"])

try:
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
SHORTS_MAX_SECONDS = 180.0


def _probe_video(video_path: Path) -> Optional[Dict[str, float]]:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height:format=duration",
        "-of", "json", str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            log.error("ffprobe failed for %s: %s", video_path, result.stderr[-300:])
            return None
        data = json.loads(result.stdout or "{}")
        stream = (data.get("streams") or [{}])[0]
        return {
            "width": int(stream.get("width") or 0),
            "height": int(stream.get("height") or 0),
            "duration": float(data.get("format", {}).get("duration") or 0),
        }
    except Exception as e:
        log.error("Could not inspect video %s: %s", video_path, e)
        return None


def _validate_shorts_video(video_path: Path) -> bool:
    """
    YouTube Data API uploads Shorts through the normal videos.insert endpoint.
    YouTube classifies them as Shorts when the file is square/vertical and short.
    """
    if not cfg["youtube"].get("enforce_shorts_eligibility", True):
        return True

    info = _probe_video(video_path)
    if not info:
        return False

    width = int(info["width"])
    height = int(info["height"])
    duration = float(info["duration"])
    max_seconds = float(cfg["youtube"].get("shorts_max_seconds", SHORTS_MAX_SECONDS))

    if width <= 0 or height <= 0 or duration <= 0:
        log.error("Invalid video metadata for Shorts upload: %s", info)
        return False
    if width > height:
        log.error(
            "Refusing to upload %s as a Short: landscape video %dx%d. Expected 9:16/square.",
            video_path.name, width, height,
        )
        return False
    if duration > max_seconds:
        log.error(
            "Refusing to upload %s as a Short: duration %.1fs exceeds %.1fs.",
            video_path.name, duration, max_seconds,
        )
        return False

    log.info("Shorts validation passed: %dx%d, %.1fs", width, height, duration)
    return True


def _has_shorts_marker(text: str) -> bool:
    return "#shorts" in (text or "").lower()


def _limit_youtube_tags(tags: List[str], max_chars: int = 450) -> List[str]:
    limited: List[str] = []
    total = 0
    for tag in tags:
        cleaned = str(tag).strip()
        if not cleaned:
            continue
        extra = len(cleaned) + (1 if limited else 0)
        if total + extra > max_chars:
            break
        limited.append(cleaned)
        total += extra
    return limited


def _ensure_shorts_metadata(title: str, description: str, tags: List[str]) -> Tuple[str, str, List[str]]:
    title = (title or "Cricket Highlights #Shorts")[:100]
    description = description or ""
    tags = [str(t).strip() for t in (tags or []) if str(t).strip()]

    if not any(t.lower().lstrip("#") == "shorts" for t in tags):
        tags.insert(0, "shorts")

    if not (_has_shorts_marker(title) or _has_shorts_marker(description)):
        marker = "\n\n#Shorts"
        description = (description[:5000 - len(marker)] + marker) if description else "#Shorts"

    return title[:100], description[:5000], _limit_youtube_tags(tags)


def _is_colab() -> bool:
    try:
        import google.colab
        return True
    except ImportError:
        return False


def get_authenticated_service(token_index=0):
    """
    Authenticate with YouTube Data API using token array rotation.
    Reads from yt_tokens.json which contains an array of token objects.
    """
    creds = None
    tokens_path = Path("yt_tokens.json")
    
    # Fallback to standard yt_token.json if array doesn't exist
    if not tokens_path.exists():
        tokens_path = Path("yt_token.json")
        if not tokens_path.exists():
            log.error("No yt_tokens.json or yt_token.json found.")
            return None
            
    try:
        with open(tokens_path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                tokens_array = data
            else:
                tokens_array = [data] # Wrap single token in list
    except Exception as e:
        log.error("Failed to parse %s: %s", tokens_path, e)
        return None
        
    if token_index >= len(tokens_array):
        log.error("Token index %d is out of bounds (array length %d)", token_index, len(tokens_array))
        return None
        
    token_data = tokens_array[token_index]
    
    try:
        creds = Credentials.from_authorized_user_info(token_data, SCOPES)
        log.info("Loaded token %d (expired=%s, has_refresh=%s)",
                 token_index,
                 not creds.valid if creds else "N/A",
                 bool(creds.refresh_token) if creds else "N/A")
    except Exception as e:
        log.warning("Failed to construct credentials for token %d: %s", token_index, e)
        return None

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            log.info("YouTube token %d refreshed successfully", token_index)
            tokens_array[token_index] = json.loads(creds.to_json())
            
            # If we were using the single token fallback, overwrite it as an array to migrate seamlessly
            with open("yt_tokens.json", "w") as f:
                json.dump(tokens_array, f, indent=2)
                
        except Exception as e:
            log.warning("Token %d refresh failed: %s", token_index, e)
            return None

    if not creds or not creds.valid:
        log.error("Token %d is invalid and could not be refreshed.", token_index)
        return None

    base_http = httplib2.Http(timeout=60)
    auth_http = google_auth_httplib2.AuthorizedHttp(creds, http=base_http)
    return build("youtube", "v3", http=auth_http)


def upload_video(
    video_path: str,
    metadata_path: str,
    privacy: str = "public",
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
    # ─── Pre-flight Checks ───────────────────────────────────────────────────
    v_path = Path(video_path)
    m_path = Path(metadata_path)
    if not v_path.exists():
        log.error(f"Video file not found: {video_path}")
        return None
    if not m_path.exists():
        log.error(f"Metadata file not found: {metadata_path}")
        return None
    if not _validate_shorts_video(v_path):
        return None

    youtube = get_authenticated_service()
    if not youtube:
        log.error("[EXIT] upload_video failed: no YouTube service (check yt_token.json)")
        return None

    # Load and validate SEO metadata
    with open(metadata_path, "r") as f:
        meta = json.load(f)

    title = meta.get("title", "Cricket Highlights #Shorts")
    description = meta.get("description", "")
    # Use dedicated SEO tags + search_terms for YouTube API tags field
    seo_tags = meta.get("tags", [])
    search_terms = meta.get("search_terms", [])
    # Merge both: SEO tags (dedicated discoverability tags) + search terms (search phrases)
    all_tags = list(dict.fromkeys([str(t).strip() for t in seo_tags + search_terms if str(t).strip()]))
    title, description, all_tags = _ensure_shorts_metadata(title, description, all_tags)

    if publish_at or privacy == "scheduled":
        if not publish_at:
            from scheduler import get_next_upload_time
            publish_at = get_next_upload_time()
            log.info(f"⏰ Privacy set to 'scheduled'. Auto-scheduling for: {publish_at}")
        else:
            log.info(f"⏰ Scheduled for: {publish_at}")
        privacy = "private"  # YouTube API requires "private" for scheduled videos

    body = {
        "snippet": {
            "title": title[:100],
            "description": description[:5000],
            "tags": all_tags,
            "categoryId": cfg["youtube"]["category_id"],
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": cfg.get("youtube", {}).get("self_declared_made_for_kids", False),
            "containsSyntheticMedia": False,
        },
    }

    if publish_at:
        body["status"]["publishAt"] = publish_at

    thumb_path = Path(video_path).with_name(f"{Path(video_path).stem}_thumb.jpg")
    
    # ── Token Rotation Loop ──────────────────────────────────────────────────────
    tokens_path = Path("yt_tokens.json")
    max_tokens = 1
    if tokens_path.exists():
        try:
            data = json.load(open(tokens_path))
            if isinstance(data, list):
                max_tokens = len(data)
        except Exception:
            pass

    for token_idx in range(max_tokens):
        youtube = get_authenticated_service(token_index=token_idx)
        if not youtube:
            log.warning("Could not get YouTube service for token %d. Trying next...", token_idx)
            continue
            
        log.info(f"🚀 Uploading to YouTube: {title[:80]}... (using token {token_idx})")
        
        media_body = MediaFileUpload(video_path, chunksize=-1, resumable=True)
        insert_request = youtube.videos().insert(
            part=",".join(body.keys()),
            body=body,
            media_body=media_body,
            notifySubscribers=True,
        )

        response = None
        last_progress = -10
        last_progress_time = time.monotonic()
        
        try:
            while response is None:
                try:
                    status, response = insert_request.next_chunk()
                    if status:
                        progress = int(status.progress() * 100)
                        now = time.monotonic()
                        if progress >= last_progress + 10 or now - last_progress_time >= 15 or progress >= 100:
                            log.info(f"   Upload Progress: {progress}%")
                            last_progress = progress
                            last_progress_time = now
                except Exception as e:
                    import http.client
                    from googleapiclient.errors import HttpError
                    if isinstance(e, HttpError) and e.resp.status in [500, 502, 503, 504]:
                        log.warning(f"Transient HTTP {e.resp.status} error, retrying in 5 seconds...")
                        time.sleep(5)
                        continue
                    elif isinstance(e, (http.client.IncompleteRead, http.client.CannotSendRequest, ConnectionError)):
                        log.warning(f"Network error: {e}, retrying in 5 seconds...")
                        time.sleep(5)
                        continue
                    else:
                        raise e

            video_id = response["id"]
            log.info(f"✅ Upload successful! Video ID: {video_id}")

            # Save the YouTube video ID to the metadata file
            try:
                if Path(metadata_path).exists():
                    with open(metadata_path, "r") as f:
                        meta_data = json.load(f)
                    meta_data["youtube_video_id"] = video_id
                    with open(metadata_path, "w") as f:
                        json.dump(meta_data, f, indent=2)
                    log.info(f"📝 Saved youtube_video_id: {video_id} to {Path(metadata_path).name}")
            except Exception as e:
                log.warning(f"Failed to write video_id to metadata: {e}")

            # ── Upload Thumbnail ──────────────────────────────────────────────────────
            if thumb_path.exists():
                try:
                    log.info(f"🖼️ Uploading thumbnail: {thumb_path.name}...")
                    youtube.thumbnails().set(
                        videoId=video_id,
                        media_body=MediaFileUpload(str(thumb_path))
                    ).execute()
                    log.info("✅ Thumbnail uploaded!")
                except Exception as e:
                    log.warning(f"Failed to upload thumbnail: {e}")
                    
            log.info(f"🔗 URL: https://youtu.be/{video_id}")
            return video_id
            
        except Exception as e:
            err_str = str(e)
            if "quotaExceeded" in err_str or "rateLimitExceeded" in err_str or "429" in err_str:
                log.warning("⚠️ Quota exceeded on token %d. Seamlessly rotating to next token...", token_idx)
                continue  # Try the next token in the loop
            else:
                log.error("❌ Error uploading %s: %s", video_path, e)
                return None
                
    log.error("[EXIT] All available tokens exhausted their quotas.")
    return None


if __name__ == "__main__":
    # Example usage
    # upload_video("shorts/clip1.mp4", "shorts/clip1_metadata.json", "private")
    pass
