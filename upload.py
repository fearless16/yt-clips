"""
upload.py — Phase 7: Automated YouTube Shorts Uploader.

Uploads videos to YouTube with metadata generated in Phase 6.

NOTE: This uses YouTube Data API (not Drive API), so it has its own
auth via yt_channel_token.json + client_secrets.json. This is intentionally
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

# YouTube API scopes — force-ssl covers upload + metadata update + caption read
SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]
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


def _truncate_bytes(text: str, max_bytes: int = 5000) -> str:
    """Truncate *text* so its UTF-8 encoding fits in *max_bytes*.

    YouTube's description limit is 5000 **bytes**, not characters — emoji and
    Hindi/Devanagari text use 3-4 bytes each, so char-based slicing could exceed
    the limit and get the upload rejected.
    """
    if text is None:
        return ""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    # Cut on a valid UTF-8 boundary.
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _limit_youtube_tags(tags: List[str], max_chars: int = 480) -> List[str]:
    """Limit tags to YouTube's ~500-char budget, accounting for quote overhead.

    YouTube counts multi-word (space-containing) tags as quoted, adding 2 chars
    per such tag. We budget conservatively (480) and include the +2 overhead so
    we never silently blow the real 500-char server-side cap.
    """
    limited: List[str] = []
    total = 0
    for tag in tags:
        cleaned = str(tag).strip()
        if not cleaned:
            continue
        quote_overhead = 2 if " " in cleaned else 0
        extra = len(cleaned) + quote_overhead + (1 if limited else 0)
        if total + extra > max_chars:
            continue  # skip this one but keep trying shorter later tags
        limited.append(cleaned)
        total += extra
    return limited


def _assignable_category_id(youtube, desired: str, region: str = "IN") -> str:
    """Return *desired* if it's an assignable YouTube category, else a safe fallback.

    An invalid/non-assignable categoryId causes a hard ``invalidCategoryId``
    upload failure. We validate against ``videoCategories.list`` (cached) and
    fall back to Sports(17)->Entertainment(24)->People(22) if needed. On any API
    error we return *desired* unchanged (don't block the upload).
    """
    desired = str(desired)
    cache_key = "yt_categories_%s" % region
    cached = _CATEGORY_CACHE.get(cache_key)
    try:
        if cached is None:
            resp = youtube.videoCategories().list(part="snippet", regionCode=region).execute()
            cached = {
                item["id"]: item["snippet"].get("assignable", False)
                for item in resp.get("items", [])
            }
            _CATEGORY_CACHE[cache_key] = cached
        if cached.get(desired):
            return desired
        for fallback in ("17", "24", "22"):  # Sports, Entertainment, People & Blogs
            if cached.get(fallback):
                log.warning("categoryId %s not assignable in %s — falling back to %s",
                            desired, region, fallback)
                return fallback
    except Exception as e:
        log.warning("Category validation skipped (%s) — using configured id %s", e, desired)
    return desired


_CATEGORY_CACHE: Dict[str, Dict[str, bool]] = {}


def _ensure_shorts_metadata(title: str, description: str, tags: List[str]) -> Tuple[str, str, List[str]]:
    title = (title or "Cricket Highlights #Shorts")[:100]
    description = description or ""
    tags = [str(t).strip() for t in (tags or []) if str(t).strip()]

    if not any(t.lower().lstrip("#") == "shorts" for t in tags):
        tags.insert(0, "shorts")

    if not (_has_shorts_marker(title) or _has_shorts_marker(description)):
        marker = "\n\n#Shorts"
        description = (_truncate_bytes(description, 5000 - len(marker)) + marker) if description else "#Shorts"

    return title[:100], _truncate_bytes(description, 5000), _limit_youtube_tags(tags)


def _is_colab() -> bool:
    try:
        import google.colab
        return True
    except ImportError:
        return False


def get_authenticated_service(token_index=0):
    """
    Authenticate with YouTube Data API using token array rotation.
    Reads from yt_channel_token.json which contains YouTube OAuth token.
    """
    creds = None
    tokens_path = Path("yt_channel_token.json")
    if not tokens_path.exists():
        log.error("No yt_channel_token.json found.")
        return None

    try:
        with open(tokens_path, "r", encoding="utf-8-sig") as f:
            token_data = json.load(f)
    except Exception as e:
        log.error("Failed to parse %s: %s", tokens_path, e)
        return None

    try:
        creds = Credentials.from_authorized_user_info(token_data, SCOPES)
        log.info("Loaded token (expired=%s, has_refresh=%s)",
                 not creds.valid if creds else "N/A",
                 bool(creds.refresh_token) if creds else "N/A")
    except Exception as e:
        log.warning("Failed to construct credentials: %s", e)
        return None

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            log.info("YouTube token refreshed successfully")
            with open("yt_channel_token.json", "w", encoding="utf-8") as f:
                f.write(creds.to_json())
        except Exception as e:
            log.warning("Token refresh failed: %s", e)
            return None

    if not creds or not creds.valid:
        log.error("Token is invalid and could not be refreshed.")
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
        log.error("[EXIT] upload_video failed: no YouTube service (check yt_channel_token.json)")
        return None

    # Load and validate SEO metadata
    with open(metadata_path, "r", encoding="utf-8") as f:
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

    # Validate category against assignable categories (avoids invalidCategoryId).
    category_id = _assignable_category_id(
        youtube, cfg["youtube"]["category_id"],
        region=cfg.get("youtube", {}).get("region_code", "IN"),
    )
    # Synthetic-media disclosure: default false, but allow per-clip metadata or
    # config override (e.g. AI-generated thumbnails/voiceover/clips).
    contains_synthetic = bool(
        meta.get("contains_synthetic_media",
                 cfg.get("youtube", {}).get("contains_synthetic_media", False))
    )

    body = {
        "snippet": {
            "title": title[:100],
            "description": _truncate_bytes(description, 5000),
            "tags": all_tags,
            "categoryId": category_id,
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": cfg.get("youtube", {}).get("self_declared_made_for_kids", False),
            "containsSyntheticMedia": contains_synthetic,
        },
    }

    if publish_at:
        body["status"]["publishAt"] = publish_at

    thumb_path = Path(video_path).with_name(f"{Path(video_path).stem}_thumb.jpg")

    log.info(f"🚀 Uploading to YouTube: {title[:80]}...")

    chunk_mb = int(cfg.get("youtube", {}).get("upload_chunk_size_mb", 8))
    media_body = MediaFileUpload(video_path, chunksize=chunk_mb * 1024 * 1024, resumable=False)
    insert_request = youtube.videos().insert(
        part=",".join(body.keys()),
        body=body,
        media_body=media_body,
        notifySubscribers=cfg.get("youtube", {}).get("notify_subscribers", True),
    )

    try:
        response = insert_request.execute()
        video_id = response["id"]
        log.info(f"✅ Upload successful! Video ID: {video_id}")

        try:
            if Path(metadata_path).exists():
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
                meta_data["youtube_video_id"] = video_id
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(meta_data, f, indent=2, ensure_ascii=False)
                log.info(f"📝 Saved youtube_video_id: {video_id} to {Path(metadata_path).name}")
        except Exception as e:
            log.warning(f"Failed to write video_id to metadata: {e}")

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
        log.error("❌ Error uploading %s: %s", video_path, e)
        return None


if __name__ == "__main__":
    # Example usage
    # upload_video("shorts/clip1.mp4", "shorts/clip1_metadata.json", "private")
    pass
