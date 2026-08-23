"""automation/instagram/uploader.py

Container → rupload bytes → poll FINISHED → publish → verify permalink.
Implements the PLAN.md frozen GraphClient Protocol flow with:
- ERROR treated as transient up to 3 consecutive polls
- unknown statuses = not-ready (keep polling)
- publish ambiguity recovery: publish_container() None ⇒ re-query
  container_status(); PUBLISHED ⇒ recover media_id from status "id" field
- hard timeout raising InstagramUploadError listing stage reached
"""

import time
from pathlib import Path

STAGE_CREATE_CONTAINER = "CREATE_CONTAINER"
STAGE_UPLOAD_VIDEO = "UPLOAD_VIDEO"
STAGE_POLL_CONTAINER = "POLL_CONTAINER"
STAGE_PUBLISH = "PUBLISH"
STAGE_VERIFY_PERMALINK = "VERIFY_PERMALINK"

_MAX_CONSECUTIVE_ERRORS = 3


class InstagramUploadError(Exception):
    def __init__(self, message: str, stage: str = "") -> None:
        super().__init__(f"{message} [stage={stage}]" if stage else message)
        self.stage = stage


def _normalize_status(payload: dict) -> tuple[str, str | None]:
    raw = payload.get("status_code") or ""
    if isinstance(raw, dict):
        raw = raw.get("code") or ""
    if not raw:
        alt = payload.get("status")
        if isinstance(alt, dict):
            alt = alt.get("code") or ""
        elif isinstance(alt, str) and alt.strip().upper() in {
                "EXPIRED", "ERROR", "FINISHED", "IN_PROGRESS", "PUBLISHED"}:
            raw = alt
    status = str(raw).upper()
    media_id = payload.get("id")
    return status, (str(media_id) if media_id else None)


def _check_deadline(deadline: float, timeout_s: float, stage: str) -> None:
    if time.monotonic() > deadline:
        raise InstagramUploadError(
            f"hard timeout after {timeout_s}s", stage=stage)


def publish_reel(client, clip_path: Path, caption: str, *,
                 audio_name: str | None = None,
                 poll_interval_s: float = 10,
                 timeout_s: float = 900,
                 resume_creation_id: str | None = None,
                 on_creation=None) -> dict:
    """Publish one Reel; returns {"media_id": ..., "permalink": ...}.

    resume_creation_id: skip create+upload and poll an existing container
    (crash-after-create resume; orphan reconciliation).
    on_creation: callback fired immediately after a NEW container id exists
    so the caller can persist it before any byte upload begins.
    """
    clip_path = Path(clip_path)
    deadline = time.monotonic() + timeout_s

    if resume_creation_id:
        stage = STAGE_POLL_CONTAINER
        creation_id = str(resume_creation_id)
        already_published_id = None
    else:
        stage = STAGE_CREATE_CONTAINER
        creation_id = client.create_reels_container(
            caption=caption, share_to_feed=True, audio_name=audio_name)
        if on_creation is not None:
            try:
                on_creation(creation_id)
            except Exception:
                pass

        stage = STAGE_UPLOAD_VIDEO
        _check_deadline(deadline, timeout_s, stage)
        client.upload_video_bytes(creation_id, clip_path)

        stage = STAGE_POLL_CONTAINER
    consecutive_errors = 0
    already_published_id = None
    while True:
        _check_deadline(deadline, timeout_s, stage)
        payload = client.container_status(creation_id)
        status, recovered_id = _normalize_status(payload)
        if status == "FINISHED":
            break
        if status == "PUBLISHED":
            if not recovered_id:
                raise InstagramUploadError(
                    f"container reports PUBLISHED without an id field; "
                    f"cannot recover media id for creation {creation_id}",
                    stage=stage)
            already_published_id = recovered_id
            break
        if status == "EXPIRED":
            raise InstagramUploadError(
                "container expired before publishing; "
                "a NEW container is required on retry",
                stage=stage)
        if status == "ERROR":
            consecutive_errors += 1
            if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                raise InstagramUploadError(
                    f"container stuck in ERROR for "
                    f"{_MAX_CONSECUTIVE_ERRORS} consecutive polls",
                    stage=stage)
        else:
            consecutive_errors = 0
        if poll_interval_s > 0:
            time.sleep(poll_interval_s)

    if already_published_id:
        media_id = already_published_id
    else:
        stage = STAGE_PUBLISH
        _check_deadline(deadline, timeout_s, stage)
        media_id = client.publish_container(creation_id)
        if not media_id:
            payload = client.container_status(creation_id)
            status, recovered_id = _normalize_status(payload)
            if status == "PUBLISHED" and recovered_id:
                media_id = recovered_id
            else:
                raise InstagramUploadError(
                    f"publish outcome ambiguous and container status={status!r}; "
                    f"a NEW container is required on retry",
                    stage=stage)

    stage = STAGE_VERIFY_PERMALINK
    _check_deadline(deadline, timeout_s, stage)
    permalink = client.media_permalink(media_id)
    return {"media_id": str(media_id), "permalink": permalink}
