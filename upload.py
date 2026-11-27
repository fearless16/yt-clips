"""
upload.py — Phase 7: Automated YouTube Shorts Uploader.

Uploads videos to YouTube with metadata generated in Phase 6.

NOTE: This uses YouTube Data API (not Drive API), so it has its own
auth via yt_channel_token.json + client_secrets.json. This is intentionally
separate from the Drive auth in utils/drive_auth.py.
"""
import _fix_encoding  # noqa: F401 — force UTF-8 on Windows cp1252

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
from automation.youtube_delivery import (
    DeliveryIncompleteError, artifact_hashes, delivery_lock, finish_thumbnail,
    load_receipt, receipt_path, save_receipt, thumbnail_path, validate_thumbnail, RECEIPT_VERSION,
    delivery_intent, finish_publication, read_publication_status,
    require_private_staging, require_future_schedule, wait_for_processing,
)

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


def _limit_youtube_tags(tags: List[str], max_chars: int = 500) -> List[str]:
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


def _prepare_upload_metadata(meta, video):
    """Validate the exact outgoing credit after all transport transformations."""
    import re

    title = meta.get("title", "Cricket Highlights #Shorts")
    description = meta.get("description", "")
    if not isinstance(title, str) or not isinstance(description, str):
        raise ValueError("Upload title and description must be text")
    tags = []
    for key in ("tags", "search_terms"):
        values = meta.get(key) or []
        if not isinstance(values, list):
            raise ValueError(f"Upload {key} must be a list")
        tags.extend(str(t).strip() for t in values if str(t).strip())
    title, description, tags = _ensure_shorts_metadata(title, description, list(dict.fromkeys(tags)))
    description = re.sub(r'(?m)^#+\s+', '', description)
    description = re.sub(r'(\*\*|__)(.*?)\1', r'\2', description)
    description = re.sub(r'(?m)^\s*[-*]\s+', '• ', description)
    description = _truncate_bytes(re.sub(r'\n{3,}', '\n\n', description).strip(), 5000)
    sidecar = Path(video).with_name(f"{Path(video).stem}_gameplay.json")
    if sidecar.exists():
        from automation.cricket_gameplay import credit_text
        try:
            evidence = json.loads(sidecar.read_text(encoding="utf-8"))
            selection = evidence["selection"]
            if selection["asset"].get("status") == "REVIEW_ONLY":
                raise ValueError("Local-review gameplay cannot be uploaded; publication review is still required")
            if selection["asset"].get("status") == "AVAILABLE":
                from automation.cricket_gameplay import load_delivery_library, sha256
                library = load_delivery_library(selection["manifest"])
                if (library["manifest_sha256"] != selection["manifest_sha256"]
                        or selection["asset"] not in library["assets"]):
                    raise ValueError("Gameplay stock changed after rendering; repackage and review")
                if evidence.get("media_sha256") != sha256(video):
                    raise ValueError("Gameplay media changed after rendering; repackage and review")
            credit = credit_text(selection)
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError("Invalid gameplay attribution evidence") from exc
        if credit not in description:
            raise ValueError("Gameplay attribution would be missing, rewritten or truncated during upload; repackage and review")
    return title, description, tags


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
        refreshed = False
        for attempt in range(1, 3):
            try:
                creds.refresh(Request())
                log.info("YouTube token refreshed successfully (attempt %d)", attempt)
                with open("yt_channel_token.json", "w", encoding="utf-8") as f:
                    f.write(creds.to_json())
                refreshed = True
                break
            except Exception as e:
                err_msg = str(e).lower()
                if "invalid_grant" in err_msg or "token has been expired" in err_msg or "revoked" in err_msg:
                    log.error(
                        "YouTube token refresh FAILED (invalid_grant) — refresh token "
                        "is revoked or expired. Run: python setup_auth.py  to re-authenticate."
                    )
                    return None
                if attempt < 2:
                    log.warning("Token refresh attempt %d failed: %s — retrying", attempt, e)
                    time.sleep(2)
                else:
                    log.warning("Token refresh failed after %d attempts: %s", attempt, e)
                    return None
        if not refreshed:
            return None

    if not creds or not creds.valid:
        log.error("Token is invalid and could not be refreshed.")
        return None

    base_http = httplib2.Http(timeout=60)
    auth_http = google_auth_httplib2.AuthorizedHttp(creds, http=base_http)
    return build("youtube", "v3", http=auth_http)


class UploadRejected(RuntimeError):
    """Definitive refusal, carrying the real reason (never a bare None).

    stop_batch=True means every further upload this run will fail the same
    way (quota, upload limit, auth), so callers must stop calling the API.
    """

    def __init__(self, reason: str, *, stop_batch: bool = False):
        super().__init__(reason)
        self.stop_batch = stop_batch


_STOP_BATCH_REASONS = frozenset({
    "quotaExceeded", "dailyLimitExceeded", "uploadLimitExceeded", "rateLimitExceeded",
    "userRateLimitExceeded", "forbidden", "insufficientPermissions", "youtubeSignupRequired",
    "authError", "unauthorized",
})


def _http_error(exc) -> tuple[Optional[int], str]:
    """(HTTP status, API reason) for googleapiclient HttpError, else (None, "")."""
    resp = getattr(exc, "resp", None)
    status = getattr(resp, "status", None)
    if status is None:
        return None, ""
    reason = ""
    try:
        content = exc.content.decode("utf-8") if isinstance(exc.content, bytes) else str(exc.content)
        reason = json.loads(content)["error"]["errors"][0]["reason"]
    except (AttributeError, ValueError, KeyError, IndexError, TypeError):
        reason = str(getattr(exc, "reason", "") or "")
    return int(status), reason


def upload_video(
    video_path: str,
    metadata_path: str,
    privacy: str = "scheduled",
    publish_at: Optional[str] = None,
) -> Optional[str]:
    """Deliver once, then resume only missing thumbnail work on the same video.

    Returns an ID only for complete delivery. A known uploaded video with an
    incomplete cover raises DeliveryIncompleteError carrying its video_id.
    """
    if not Path(video_path).is_file():
        raise UploadRejected(f"Video file not found: {video_path}")
    with delivery_lock(Path(video_path)):
        return _upload_video(video_path, metadata_path, privacy, publish_at)


def _complete_delivery(youtube, video, metadata, cover, artifacts, receipt):
    video_id = receipt["video_id"]
    try:
        if receipt["status"] == "THUMBNAIL_PENDING":
            require_private_staging(read_publication_status(youtube, video_id, receipt["channel_id"]))
            if cover is not None:
                receipt["thumbnail"] = finish_thumbnail(
                    youtube, video_id, cover, cfg.get("youtube", {}), media_factory=MediaFileUpload)
            else:
                wait_for_processing(youtube, video_id, cfg.get("youtube", {}))
                receipt["thumbnail"] = {"status": "DISABLED"}
            receipt["status"] = "PUBLICATION_PENDING"
            save_receipt(video, receipt)
        if artifact_hashes(video, metadata, cover) != artifacts:
            raise ValueError("Reviewed artifacts changed during thumbnail delivery")
        if cfg.get("release_review", {}).get("required", False):
            from automation.release_review import require_ready
            require_ready(video)
        receipt["publication"] = finish_publication(youtube, video, receipt)
        if artifact_hashes(video, metadata, cover) != artifacts:
            raise ValueError("Reviewed artifacts changed during publication; remote visibility may already be applied")
        completed = {**receipt, "status": "COMPLETE"}
        completed.pop("error", None)
        save_receipt(video, completed)
        log.info("YouTube delivery complete (thumbnail %s): https://youtu.be/%s",
                 receipt["thumbnail"]["status"], video_id)
        return video_id
    except Exception as exc:
        receipt["error"] = str(exc)
        save_receipt(video, receipt)
        phase = "thumbnail/processing" if receipt["status"] == "THUMBNAIL_PENDING" else "publication"
        raise DeliveryIncompleteError(video_id, f"{phase} incomplete: {exc}") from exc


def _upload_video(video_path, metadata_path, privacy, publish_at):
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
    if not m_path.exists():
        raise UploadRejected(f"Metadata file not found: {metadata_path}")
    cover = thumbnail_path(v_path) if cfg.get("thumbnail", {}).get("enabled", True) else None
    if cover is not None:
        try:
            validate_thumbnail(cover)
        except (OSError, ValueError) as exc:
            # The cover is best-effort; it never blocks the video (C8).
            log.warning("Custom cover unusable for %s, uploading without it: %s", v_path.name, exc)
            cover = None
    try:
        with m_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        if not isinstance(meta, dict):
            raise ValueError("Metadata must be a JSON object")
        title, clean_desc, all_tags = _prepare_upload_metadata(meta, v_path)
    except (OSError, ValueError) as exc:
        raise UploadRejected(f"Delivery preflight failed: {exc}") from exc
    if cfg.get("release_review", {}).get("required", False):
        from automation.release_review import require_ready
        # Approval binds the canonical metadata file, not an arbitrary override.
        if m_path.resolve() != v_path.with_name(f"{v_path.stem}_metadata.json").resolve():
            raise UploadRejected("Release held: metadata path differs from reviewed package")
        try:
            require_ready(v_path)
        except ValueError as exc:
            raise UploadRejected(str(exc)) from exc
    if not _validate_shorts_video(v_path):
        raise UploadRejected("Not Shorts-eligible (orientation/duration/probe); see log")

    artifacts = artifact_hashes(v_path, m_path, cover)
    channel_id = cfg.get("youtube", {}).get("channel_id")
    receipt = load_receipt(v_path, artifacts, channel_id=channel_id)
    if receipt is None and meta.get("youtube_video_id"):
        raise DeliveryIncompleteError(meta["youtube_video_id"],
                                      "Legacy upload ID exists without a bound receipt; reconcile, do not re-upload")
    try:
        intent = delivery_intent(
            privacy, publish_at, notify_subscribers=cfg.get("youtube", {}).get("notify_subscribers", True),
            previous=receipt["intent"] if receipt is not None else None)
        if receipt is not None and intent != receipt["intent"]:
            raise ValueError("Delivery intent/settings changed (privacy, schedule or notifications); explicit reconciliation required")
        if receipt is None:
            require_future_schedule(intent)
    except (ValueError, TypeError) as exc:
        if receipt is not None:
            raise DeliveryIncompleteError(receipt["video_id"], str(exc)) from exc
        raise UploadRejected(f"Delivery intent preflight failed: {exc}") from exc
    if receipt is not None and receipt.get("status") == "COMPLETE":
        return receipt["video_id"]

    youtube = get_authenticated_service()
    if not youtube:
        raise UploadRejected("No YouTube service: token missing/invalid (yt_channel_token.json)",
                             stop_batch=True)
    if receipt is not None:
        log.info("Resuming %s for existing video %s; no video insert", receipt["status"], receipt["video_id"])
        return _complete_delivery(youtube, v_path, m_path, cover, artifacts, receipt)

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
            "title": title[:100] if title else "Shorts",
            "description": _truncate_bytes(clean_desc, 5000),
            "tags": all_tags,
            "categoryId": category_id,
        },
        "status": {
            # Never expose or schedule a video before thumbnail/release checks.
            "privacyStatus": "private",
            "selfDeclaredMadeForKids": cfg.get("youtube", {}).get("self_declared_made_for_kids", False),
            "containsSyntheticMedia": contains_synthetic,
        },
    }

    log.info(f"🚀 Uploading to YouTube: {title[:80]}...")

    chunk_mb = int(cfg.get("youtube", {}).get("upload_chunk_size_mb", 8))
    media_body = MediaFileUpload(video_path, chunksize=chunk_mb * 1024 * 1024, resumable=False)
    try:
        # Persist intent BEFORE insert: an ambiguous timeout must never cause a
        # second insert on the next invocation. Do not mutate reviewed metadata.
        receipt = {"schema_version": RECEIPT_VERSION, "channel_id": channel_id, "artifacts": artifacts,
                   "intent": intent,
                   "status": "VIDEO_UPLOAD_UNKNOWN", "video_id": None,
                   "thumbnail": {"status": "PENDING" if cover is not None else "DISABLED"}}
        save_receipt(v_path, receipt)
        insert_request = youtube.videos().insert(
            part=",".join(body.keys()),
            body=body,
            media_body=media_body,
            notifySubscribers=intent["notify_subscribers"],
        )
        response = insert_request.execute()
        video_id = response.get("id") if isinstance(response, dict) else None
        if not isinstance(video_id, str) or not video_id.strip():
            raise ValueError("Insert returned no valid video ID; outcome unknown, reconcile before retry")
        receipt.update(video_id=video_id, status="THUMBNAIL_PENDING")
        save_receipt(v_path, receipt)
        log.info("Video received by YouTube: %s; thumbnail delivery still pending", video_id)
        return _complete_delivery(youtube, v_path, m_path, cover, artifacts, receipt)
    except DeliveryIncompleteError:
        raise
    except Exception as e:
        status, reason = _http_error(e)
        detail = f"{type(e).__name__}: {str(e) or repr(e)}"
        if status is not None and 400 <= status < 500 and receipt and receipt.get("video_id") is None:
            # A 4xx on insert is a definitive refusal: no video exists, so the
            # pre-insert receipt must not hold this clip forever.
            try:
                receipt_path(v_path).unlink(missing_ok=True)
            except OSError as unlink_error:
                log.error("Could not clear refused-insert receipt: %s", unlink_error)
            stop = reason in _STOP_BATCH_REASONS or status in (401, 403, 429)
            raise UploadRejected(f"YouTube refused insert ({status} {reason}): {detail}",
                                 stop_batch=stop) from e
        # Timeouts/5xx are ambiguous: keep VIDEO_UPLOAD_UNKNOWN, never re-insert blindly.
        raise UploadRejected(f"Insert outcome unknown, receipt kept for reconciliation: {detail}") from e
    finally:
        media_body.stream().close()


if __name__ == "__main__":
    # Example usage
    # upload_video("shorts/clip1.mp4", "shorts/clip1_metadata.json", "private")
    pass
