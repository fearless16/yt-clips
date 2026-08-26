"""automation/instagram/upload_post_client.py

Third-party publisher adapter (upload-post.com) — ONE REST call publishes
the Reel; they own Meta OAuth/tokens/hosting/retries. This replaces the
Graph container flow entirely when configured.

Contract: POST https://api.upload-post.com/api/upload (multipart)
Auth: Authorization: Apikey <key>
Sync response: {"success": true, "results": {"instagram": {...}}}
Async fallback: {"success": true, "request_id": ...} -> poll
GET /api/uploadposts/status?request_id=...

Kill-switch: same YT_CLIPS_INSTA_LIVE env gate as the Graph client.
"""

import hashlib
import os
import time
from pathlib import Path

import requests

BASE_URL = "https://api.upload-post.com"
KILL_SWITCH_ENV = "YT_CLIPS_INSTA_LIVE"

_STATUS_TERMINAL_DONE = {"finished", "success", "completed", "done"}
_STATUS_TERMINAL_FAILED = {"error", "failed"}


class UploadPostError(RuntimeError):
    pass


def _require_live() -> None:
    if os.environ.get(KILL_SWITCH_ENV) != "1":
        raise RuntimeError(
            f"{KILL_SWITCH_ENV}!=1: Upload-Post calls are disabled "
            "(set it to 1 for live mode)")


def _idempotency_key(video_path: Path) -> str:
    stat = Path(video_path).stat()
    raw = f"{Path(video_path).resolve()}|{stat.st_size}|{int(stat.st_mtime)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _extract_instagram_result(payload: dict) -> dict | None:
    """Handle BOTH response shapes.

    Sync /api/upload: results is a DICT keyed by platform
    ({"instagram": {...}}). Async /api/uploadposts/status: results is a
    LIST of per-platform entries with a "platform" field.
    """
    results = payload.get("results")
    if isinstance(results, dict):
        ig = results.get("instagram")
        if isinstance(ig, dict):
            return ig
    if isinstance(results, list):
        for entry in results:
            if isinstance(entry, dict) \
                    and str(entry.get("platform") or "").casefold() \
                    == "instagram":
                return entry
    return None


def _entry_is_terminal(entry: dict) -> bool:
    """True when a per-platform entry has settled (done or failed).

    An idempotent_replay / async snapshot can carry ``success: false`` with
    ``status: "processing"`` — that is NOT a failure, the request_id must
    be polled to completion instead.
    """
    status = str(entry.get("status") or "").casefold()
    if status in _STATUS_TERMINAL_DONE or status in _STATUS_TERMINAL_FAILED:
        return True
    return entry.get("success") is True


class UploadPostClient:
    def __init__(self, api_key: str, profile: str, *,
                 base_url: str = BASE_URL,
                 timeout: float = 60.0,
                 session: requests.Session | None = None) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        if not profile:
            raise ValueError("profile is required")
        self.api_key = api_key
        self.profile = profile
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def _headers(self, extra: dict | None = None) -> dict:
        headers = {"Authorization": f"Apikey {self.api_key}"}
        if extra:
            headers.update(extra)
        return headers

    def verify_credentials(self) -> dict:
        _require_live()
        resp = self.session.get(
            f"{self.base_url}/api/uploadposts/me",
            headers=self._headers(), timeout=self.timeout)
        payload = resp.json() if resp.headers.get("content-type",
                                                 "").startswith(
            "application/json") else {}
        if resp.status_code != 200 or payload.get("success") is not True:
            raise UploadPostError(
                f"credential check failed ({resp.status_code}): "
                f"{payload.get('message') or payload.get('error') or 'unknown'}")
        return payload

    def publish_reel(self, video_path: Path, caption: str, *,
                     audio_name: str | None = None,
                     share_to_feed: bool = True,
                     poll_interval_s: float = 10,
                     timeout_s: float = 900) -> dict:
        """Publish one Reel; returns {"media_id", "permalink"}."""
        _require_live()
        video_path = Path(video_path)
        deadline = time.monotonic() + timeout_s
        idem = _idempotency_key(video_path)

        data = {
            "user": self.profile,
            "platform[]": "instagram",
            "title": caption,
            "instagram_title": caption,
            "media_type": "REELS",
            "share_to_feed": "true" if share_to_feed else "false",
        }
        if audio_name:
            data["audio_name"] = audio_name

        with video_path.open("rb") as fh:
            resp = self.session.post(
                f"{self.base_url}/api/upload",
                headers=self._headers({
                    "Idempotency-Key": idem,
                    "X-Request-Id": idem,
                }),
                files={"video": (video_path.name, fh, "video/mp4")},
                data=data,
                timeout=max(self.timeout, 300),
            )

        payload = {}
        try:
            payload = resp.json()
        except ValueError:
            pass

        if resp.status_code == 401:
            raise UploadPostError("invalid/expired API key")
        if resp.status_code == 429:
            usage = payload.get("usage") or {}
            raise UploadPostError(
                f"monthly upload limit reached "
                f"(used {usage.get('count')}/{usage.get('limit')})")

        ig = _extract_instagram_result(payload)
        request_id = payload.get("request_id")
        if ig is not None and _entry_is_terminal(ig):
            return self._result_from(ig)

        if resp.status_code == 200 and payload.get("success") \
                and request_id:
            return self._poll_status(request_id,
                                     poll_interval_s=poll_interval_s,
                                     timeout_s=max(
                                         1.0, deadline - time.monotonic()))

        if ig is not None:
            return self._result_from(ig)
        raise UploadPostError(
            f"upload rejected ({resp.status_code}): "
            f"{payload.get('message') or payload.get('error') or resp.text[:200]}")

    def _poll_status(self, request_id: str, *,
                     poll_interval_s: float,
                     timeout_s: float) -> dict:
        deadline = time.monotonic() + max(timeout_s, 1.0)
        while True:
            resp = self.session.get(
                f"{self.base_url}/api/uploadposts/status",
                params={"request_id": request_id},
                headers=self._headers(), timeout=self.timeout)
            try:
                payload = resp.json()
            except ValueError:
                payload = {}
            ig = _extract_instagram_result(payload)
            if ig is not None and _entry_is_terminal(ig):
                return self._result_from(ig)
            status = str(payload.get("status") or "").casefold()
            if status in _STATUS_TERMINAL_FAILED:
                raise UploadPostError(
                    f"async upload failed: "
                    f"{payload.get('error') or payload.get('message')}")
            if time.monotonic() > deadline:
                raise UploadPostError(
                    f"status poll timed out after {timeout_s:.0f}s "
                    f"(request_id={request_id}, last status={status!r})")
            if poll_interval_s > 0:
                time.sleep(poll_interval_s)

    @staticmethod
    def _result_from(ig: dict) -> dict:
        if ig.get("success") is False:
            raise UploadPostError(f"instagram publish failed: {ig.get('error')}")
        permalink = (str(ig.get("url") or "").strip()
                     or str(ig.get("post_url") or "").strip())
        media_id = (str(ig.get("post_id") or "").strip()
                    or str(ig.get("container_id") or "").strip()
                    or str(ig.get("publish_id") or "").strip())
        message = str(ig.get("message") or "").strip()
        if not permalink and not media_id:
            if ig.get("success") is True or \
                    message.casefold() in {"published", "completed"}:
                # Platform-level success without an id (async list shape):
                # publish verified, permalink unknown.
                return {"media_id": f"up_{id(ig) & 0xffffffffff:x}",
                        "permalink": ""}
            raise UploadPostError(
                f"instagram result missing url/id fields: {list(ig)}")
        return {"media_id": media_id or permalink,
                "permalink": permalink}
