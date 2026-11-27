"""Hash-bound YouTube delivery evidence, not analytics or release approval.

No authentication or API calls at import time. Network helpers are called only
by an explicitly requested upload. Reviewed media/metadata remain immutable.
"""
import hashlib
import io
import json
import math
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
from PIL import Image

from automation.artifacts import atomic_json

MAX_IMAGE_BYTES = 50 * 1024 * 1024
THUMBNAIL_SIZES = ("shortsAspectRatio", "maxres", "standard", "high", "medium", "default")
RECEIPT_VERSION = 2
DELIVERY_STATES = {"THUMBNAIL_PENDING", "PUBLICATION_PENDING", "PUBLICATION_UNKNOWN", "COMPLETE"}


class DeliveryIncompleteError(RuntimeError):
    """A video may exist: callers must not treat this as publication success."""

    def __init__(self, video_id, reason):
        self.video_id = video_id
        super().__init__(f"Delivery incomplete (video_id={video_id or 'unknown'}): {reason}")


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def thumbnail_path(video):
    video = Path(video)
    return video.with_name(f"{video.stem}_thumb.jpg")


def receipt_path(video):
    video = Path(video)
    return video.with_name(f"{video.stem}_delivery.json")


@contextmanager
def delivery_lock(video):
    """Conservative cross-process exclusion. Stale locks require reconciliation."""
    video = Path(video)
    lock = video.with_name(f"{video.stem}_delivery.lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise DeliveryIncompleteError(None, f"Delivery is locked: {lock}; reconcile before retry") from exc
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        lock.unlink()


def validate_thumbnail(path):
    """This pipeline deliberately delivers upright RGB JPEG portrait covers."""
    path = Path(path)
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_IMAGE_BYTES:
        raise ValueError(f"Missing, empty or oversized thumbnail: {path}")
    with Image.open(path) as image:
        if image.format != "JPEG" or image.mode != "RGB":
            raise ValueError("Thumbnail must be an RGB JPEG, not just named .jpg")
        if image.height < 640 or abs(image.width / image.height - 9 / 16) > .005:
            raise ValueError("Thumbnail must be 9:16 portrait, at least 640 pixels high")
        if image.getexif().get(274, 1) != 1:
            raise ValueError("Thumbnail orientation must be baked into pixels")
        image.load()
        return {"path": str(path.resolve()), "sha256": sha256(path),
                "width": image.width, "height": image.height,
                "bytes": path.stat().st_size, "mime_type": "image/jpeg"}


def stage_thumbnail(video, source, *, expected_sha256):
    """Explicit exact-byte binding. Never overwrite a different existing cover."""
    video, source = Path(video), Path(source)
    if not video.is_file():
        raise ValueError("Delivery video does not exist")
    info = validate_thumbnail(source)
    if info["sha256"] != expected_sha256:
        raise ValueError("Thumbnail approval hash differs from source")
    content = source.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("Thumbnail changed after approval hash check")
    target = thumbnail_path(video)
    with delivery_lock(video):
        if target.exists():
            if target.is_file() and sha256(target) == expected_sha256:
                return target
            raise FileExistsError(f"Preserve existing cover; explicit archive/re-review required: {target}")
        # The shared lock prevents the uploader seeing a partially staged JPEG.
        created = False
        try:
            with target.open("xb") as handle:
                created = True
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            if created:
                target.unlink(missing_ok=True)
            raise
    return target


def artifact_hashes(video, metadata, cover=None):
    return {key: {"path": str(Path(path).resolve()), "sha256": sha256(path)}
            for key, path in (("media", video), ("metadata", metadata), ("cover", cover))
            if path is not None}


def _utc_now():
    return datetime.now(timezone.utc)


def _publication_time(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Schedule requires a timezone-aware publication timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Schedule requires an explicit timezone")
    return parsed.astimezone(timezone.utc)


def validate_delivery_intent(intent):
    if (not isinstance(intent, dict)
            or set(intent) != {"privacy", "publish_at", "notify_subscribers"}
            or intent["privacy"] not in {"private", "public", "unlisted", "scheduled"}
            or not isinstance(intent["notify_subscribers"], bool)):
        raise ValueError("Invalid or missing delivery intent; reconcile before retry")
    if intent["privacy"] == "scheduled":
        normalized = _publication_time(intent["publish_at"]).isoformat().replace("+00:00", "Z")
        if normalized != intent["publish_at"]:
            raise ValueError("Receipt schedule is not normalized")
    elif intent["publish_at"] is not None:
        raise ValueError("Unexpected schedule for nonscheduled delivery intent")
    return intent


def delivery_intent(privacy, publish_at, *, notify_subscribers, previous=None):
    """Normalize intent once. An implicit retry reuses its already assigned slot."""
    if privacy not in {"private", "public", "unlisted", "scheduled"}:
        raise ValueError("Invalid requested privacy")
    effective_privacy = "scheduled" if publish_at is not None or privacy == "scheduled" else privacy
    if previous and effective_privacy != previous["privacy"]:
        # Reject before the scheduler can reserve/write a brand-new slot.
        raise ValueError("Delivery privacy intent changed; explicit reconciliation required")
    if publish_at is not None or privacy == "scheduled":
        privacy = "scheduled"
        if publish_at is None:
            if previous and previous["privacy"] == "scheduled":
                publish_at = previous["publish_at"]
            else:
                from scheduler import get_next_upload_time
                publish_at = get_next_upload_time()
        publish_at = _publication_time(publish_at).isoformat().replace("+00:00", "Z")
    return validate_delivery_intent({"privacy": privacy, "publish_at": publish_at,
                                     "notify_subscribers": notify_subscribers})


def require_future_schedule(intent):
    if intent["privacy"] == "scheduled" and _publication_time(intent["publish_at"]) <= _utc_now():
        raise ValueError("Requested schedule is past/expired; explicit reconciliation is required")


def load_receipt(video, artifacts, *, channel_id):
    path = receipt_path(video)
    if not path.exists():
        return None
    video_id = None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            video_id = payload.get("video_id")
        if not isinstance(payload, dict) or payload.get("schema_version") != RECEIPT_VERSION:
            raise ValueError("Legacy/invalid receipt lacks bound publication intent; reconcile, never re-upload")
        if (payload.get("artifacts") != artifacts
                or payload.get("channel_id") != channel_id):
            raise ValueError("Artifacts or channel differ/changed since delivery")
        if not isinstance(video_id, str) or not video_id.strip():
            raise ValueError("Video upload outcome unknown; reconcile the receipt, never blindly re-upload")
        if payload.get("status") not in DELIVERY_STATES:
            raise ValueError("Invalid delivery state; reconcile before retry")
        validate_delivery_intent(payload.get("intent"))
        if payload["status"] != "THUMBNAIL_PENDING":
            # A failed custom cover is recorded, never a publication blocker (C8).
            expected = {"VERIFIED", "FAILED"} if "cover" in artifacts else {"DISABLED"}
            if not isinstance(payload.get("thumbnail"), dict) or payload["thumbnail"].get("status") not in expected:
                raise ValueError("Publication receipt is missing completed thumbnail evidence")
        if payload["status"] == "COMPLETE":
            publication = payload.get("publication")
            if (not isinstance(publication, dict) or publication.get("status") != "VERIFIED"
                    or not publication_matches(publication.get("readback"), payload["intent"], allow_elapsed=True)):
                raise ValueError("Complete receipt lacks matching publication readback")
        return payload
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise DeliveryIncompleteError(video_id if isinstance(video_id, str) else None, str(exc)) from exc


def save_receipt(video, payload):
    try:
        atomic_json(receipt_path(video), payload)
    except Exception as exc:
        raise DeliveryIncompleteError(payload.get("video_id"),
                                      f"Cannot persist delivery receipt; reconcile before retry: {exc}") from exc


def _poll_settings(config, phase):
    timeout = float(config.get(f"thumbnail_{phase}_timeout_seconds", 180 if phase == "processing" else 90))
    interval = float(config.get("thumbnail_poll_interval_seconds", 3))
    attempts = int(config.get("thumbnail_max_poll_attempts", 60))
    if not all(math.isfinite(v) and v > 0 for v in (timeout, interval, attempts)):
        raise ValueError("Thumbnail deadlines, interval and attempt cap must be positive finite values")
    return timeout, interval, attempts


def _one_video(response, video_id):
    if not isinstance(response, dict) or not isinstance(response.get("items"), list):
        raise ValueError("Malformed video readback")
    items = response["items"]
    if len(items) != 1 or not isinstance(items[0], dict) or items[0].get("id") != video_id:
        raise ValueError("Missing or mismatched video readback")
    return items[0]


def read_publication_status(youtube, video_id, channel_id):
    item = _one_video(youtube.videos().list(part="status,snippet", id=video_id).execute(), video_id)
    if channel_id and item.get("snippet", {}).get("channelId") != channel_id:
        raise ValueError("Publication readback belongs to a different/missing channel")
    status = item.get("status")
    if not isinstance(status, dict) or status.get("privacyStatus") not in {"private", "public", "unlisted"}:
        raise ValueError("Missing or invalid publication status readback")
    return status


def require_private_staging(status):
    if status["privacyStatus"] != "private" or status.get("publishAt"):
        raise ValueError("Video is no longer privately staged without a schedule; reconcile before proceeding")


def publication_matches(status, intent, *, allow_elapsed=False):
    if not isinstance(status, dict):
        return False
    if intent["privacy"] != "scheduled":
        return status.get("privacyStatus") == intent["privacy"] and not status.get("publishAt")
    if status.get("privacyStatus") == "private" and status.get("publishAt"):
        return _publication_time(status["publishAt"]) == _publication_time(intent["publish_at"])
    # An acknowledged-or-ambiguous schedule may have executed before a retry.
    return (allow_elapsed and status.get("privacyStatus") == "public"
            and _utc_now() >= _publication_time(intent["publish_at"]))


def finish_publication(youtube, video, receipt):
    """Read before writing; persist uncertainty before a potentially public update."""
    video_id, intent = receipt["video_id"], receipt["intent"]
    observed = read_publication_status(youtube, video_id, receipt["channel_id"])
    if publication_matches(observed, intent, allow_elapsed=receipt["status"] == "PUBLICATION_UNKNOWN"):
        return {"status": "VERIFIED", "readback": observed}
    require_private_staging(observed)
    require_future_schedule(intent)
    # videos.update replaces mutable status fields. Do not reset unrelated
    # embedding, license, audience or disclosure settings while publishing.
    writable = ("embeddable", "license", "publicStatsViewable",
                "selfDeclaredMadeForKids", "containsSyntheticMedia")
    status = {key: observed[key] for key in writable if key in observed}
    status["privacyStatus"] = "private" if intent["privacy"] == "scheduled" else intent["privacy"]
    if intent["publish_at"] is not None:
        status["publishAt"] = intent["publish_at"]
    receipt["status"] = "PUBLICATION_UNKNOWN"
    save_receipt(video, receipt)
    response = youtube.videos().update(part="status", body={"id": video_id, "status": status}).execute()
    if not isinstance(response, dict) or response.get("id") != video_id:
        raise ValueError("Publication update returned a missing/mismatched video ID; reconcile by readback")
    observed = read_publication_status(youtube, video_id, receipt["channel_id"])
    if not publication_matches(observed, intent, allow_elapsed=True):
        raise ValueError("Publication readback does not yet match the authorized intent")
    return {"status": "VERIFIED", "readback": observed}


def wait_for_processing(youtube, video_id, config):
    timeout, interval, attempts = _poll_settings(config, "processing")
    deadline = time.monotonic() + timeout
    for attempt in range(attempts):
        if time.monotonic() >= deadline:
            break
        item = _one_video(youtube.videos().list(part="processingDetails", id=video_id).execute(), video_id)
        status = item.get("processingDetails", {}).get("processingStatus")
        if status == "succeeded":
            return
        if status != "processing":
            raise ValueError(f"Shorts processing failed or invalid: {status!r}")
        if attempt+1 < attempts:
            remaining = deadline-time.monotonic()
            if remaining > 0:
                time.sleep(min(interval, remaining))
    raise TimeoutError("Shorts processing deadline/attempt limit reached")


def _safe_thumbnail_url(url):
    if not isinstance(url, str):
        raise ValueError("Missing thumbnail URL")
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if (parsed.scheme != "https" or parsed.username or parsed.password or parsed.port not in (None, 443)
            or not (host.endswith(".ytimg.com") or host == "yt3.ggpht.com")):
        raise ValueError("Thumbnail readback URL is not a supported YouTube image host")
    return url


def validate_set_response(response):
    if not isinstance(response, dict) or not isinstance(response.get("items"), list) or not response["items"]:
        raise ValueError("Malformed thumbnail-set response")
    for item in response["items"]:
        if isinstance(item, dict):
            for size in THUMBNAIL_SIZES:
                row = item.get(size)
                if isinstance(row, dict) and row.get("url"):
                    return {"accepted_url": _safe_thumbnail_url(row["url"])}
    raise ValueError("Thumbnail-set response has no valid thumbnail resource")


def download_thumbnail(url, timeout):
    """Fetch only the first-party returned asset, bounded and without redirects."""
    url = _safe_thumbnail_url(url)
    with requests.get(url, params={"cwp_verify": uuid.uuid4().hex},
                      headers={"Cache-Control": "no-cache"}, stream=True,
                      allow_redirects=False, timeout=min(15, timeout)) as response:
        response.raise_for_status()
        # requests doesn't raise for redirects; never follow a returned location.
        if isinstance(response.status_code, int) and response.status_code != 200:
            raise ValueError(f"Thumbnail CDN returned HTTP {response.status_code}")
        chunks, size = [], 0
        for chunk in response.iter_content(chunk_size=65536):
            size += len(chunk)
            if size > MAX_IMAGE_BYTES:
                raise ValueError("Served thumbnail exceeds image size limit")
            chunks.append(chunk)
        return b"".join(chunks)


def compare_thumbnail_pixels(expected, served_bytes):
    """Perceptual evidence, not byte equality (YouTube resizes/recompresses)."""
    with Image.open(expected) as source, Image.open(io.BytesIO(served_bytes)) as served:
        if served.height < 192 or abs(served.width/served.height-9/16) > .015:
            raise ValueError("Served thumbnail is not a usable 9:16 Shorts cover")
        size = (180, 320)
        a = np.asarray(source.convert("RGB").resize(size, Image.Resampling.LANCZOS), dtype=np.float32)
        b = np.asarray(served.convert("RGB").resize(size, Image.Resampling.LANCZOS), dtype=np.float32)
    diff = (a-b)**2
    rmse = float(np.sqrt(diff.mean()))
    # A large unchanged dark field must not conceal a wrong face, brand or hook.
    worst_tile = max(float(np.sqrt(tile.mean()))
                     for row in np.array_split(diff, 4, axis=0)
                     for tile in np.array_split(row, 3, axis=1))
    if rmse > 12 or worst_tile > 24:
        raise ValueError(f"Served thumbnail pixels do not match: RMSE={rmse:.2f}, worst tile={worst_tile:.2f}")
    return {"method": "9:16 CDN pixel comparison; tolerant of resizing/recompression",
            "pixel_rmse": rmse, "worst_tile_rmse": worst_tile,
            "served_sha256": hashlib.sha256(served_bytes).hexdigest()}


def verify_thumbnail(youtube, video_id, cover, config):
    timeout, interval, attempts = _poll_settings(config, "verify")
    deadline = time.monotonic()+timeout
    last_error = "No Shorts thumbnail readback"
    for attempt in range(attempts):
        if time.monotonic() >= deadline:
            break
        try:
            item = _one_video(youtube.videos().list(part="snippet", id=video_id).execute(), video_id)
            row = item.get("snippet", {}).get("thumbnails", {}).get("shortsAspectRatio")
            if not isinstance(row, dict):
                raise ValueError("Custom Shorts thumbnail not visible in API readback yet")
            url = _safe_thumbnail_url(row.get("url"))
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                break
            result = compare_thumbnail_pixels(cover, download_thumbnail(url, remaining))
            return {**result, "url": url}
        except (ValueError, OSError, requests.RequestException) as exc:
            last_error = str(exc)
        if attempt+1 < attempts:
            remaining = deadline-time.monotonic()
            if remaining > 0:
                time.sleep(min(interval, remaining))
    raise TimeoutError(f"Thumbnail verification deadline/attempt limit reached: {last_error}")


def finish_thumbnail(youtube, video_id, cover, config, *, media_factory):
    """Processing must succeed; the custom cover is best-effort.

    A cover failure returns status FAILED with the real error so publication
    proceeds. Whether the Shorts feed even shows custom covers is unverified.
    """
    wait_for_processing(youtube, video_id, config)
    try:
        body = media_factory(str(cover), mimetype="image/jpeg")
        try:
            accepted = validate_set_response(youtube.thumbnails().set(videoId=video_id, media_body=body).execute())
        finally:
            body.stream().close()
        return {"status": "VERIFIED", "sha256": sha256(cover), "set_response": accepted,
                "verification": verify_thumbnail(youtube, video_id, cover, config)}
    except Exception as exc:  # noqa: BLE001 - any cover failure is non-fatal by design
        return {"status": "FAILED", "sha256": sha256(cover) if Path(cover).is_file() else None,
                "error": f"{type(exc).__name__}: {str(exc) or repr(exc)}"}
