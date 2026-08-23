"""Thin Instagram Graph API client — Facebook-Login flavor ONLY.

Implements the frozen ``GraphClient`` protocol from PLAN.md:
container create (resumable, no video_url) -> rupload byte POST ->
status polling -> media_publish -> permalink + REAL hashtag signals.

Kill-switch: every live call raises RuntimeError unless env
``YT_CLIPS_INSTA_LIVE=1`` (test suites and dry runs run with it unset).
Publish failures are ALWAYS treated as ambiguous (Meta may have succeeded
server-side before the 5xx) — returns None so callers re-query
``container_status``; only ``PUBLISHED`` means success.
"""
from __future__ import annotations

import json
import logging
import os
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import requests

log = logging.getLogger("insta_graph")

GRAPH_BASE = "https://graph.facebook.com"
RUPLOAD_BASE = "https://rupload.facebook.com"
DEFAULT_VERSION = "v21.0"
KILL_SWITCH_ENV = "YT_CLIPS_INSTA_LIVE"
RECENT_WINDOW_S = 24 * 3600


def _live_enabled() -> bool:
    return os.environ.get(KILL_SWITCH_ENV, "").strip() == "1"


class GraphAPIError(Exception):
    """Graph API error parsed from the standard {"error": {...}} envelope."""

    def __init__(self, message: str, code: Optional[int] = None,
                 subcode: Optional[int] = None,
                 http_status: Optional[int] = None,
                 fbtrace_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.subcode = subcode
        self.http_status = http_status
        self.fbtrace_id = fbtrace_id

    def __repr__(self) -> str:
        return (f"GraphAPIError(code={self.code}, subcode={self.subcode}, "
                f"http={self.http_status}, message={self.message!r})")


@runtime_checkable
class GraphClient(Protocol):
    """Frozen contract — see PLAN.md §2. Tests inject fakes of this shape."""

    def create_reels_container(self, *, caption: str, share_to_feed: bool = True,
                               audio_name: Optional[str] = None,
                               trial_params: Optional[dict] = None) -> str: ...

    def upload_video_bytes(self, creation_id: str, video_path: Path) -> dict: ...

    def container_status(self, creation_id: str) -> dict: ...

    def publish_container(self, creation_id: str) -> Optional[str]: ...

    def media_permalink(self, media_id: str) -> str: ...

    def hashtag_search(self, q: str) -> dict: ...

    def hashtag_recent_velocity(self, hashtag_id: str) -> int: ...

    def hashtag_top_engagement(self, hashtag_id: str) -> float: ...


class FacebookGraphClient(GraphClient):
    """Real client against graph.facebook.com v21.0 + rupload byte upload."""

    def __init__(self, token: str, ig_user_id: str, *,
                 version: str = DEFAULT_VERSION,
                 timeout: float = 30.0,
                 graph_base: str = GRAPH_BASE,
                 rupload_base: str = RUPLOAD_BASE,
                 max_hashtag_pages: int = 4) -> None:
        self.token = str(token or "")
        self.ig_user_id = str(ig_user_id or "")
        self.version = version
        self.timeout = float(timeout)
        self.graph_base = graph_base.rstrip("/")
        self.rupload_base = rupload_base.rstrip("/")
        self.max_hashtag_pages = int(max_hashtag_pages)
        self.session = requests.Session()

    # -- plumbing -----------------------------------------------------------

    def _url(self, endpoint: str) -> str:
        return f"{self.graph_base}/{self.version}/{endpoint.lstrip('/')}"

    def _parse(self, resp: requests.Response) -> dict:
        try:
            payload = resp.json()
        except ValueError:
            payload = {}
        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            err = payload["error"]
            raise GraphAPIError(
                message=str(err.get("message") or "Unknown Graph API error"),
                code=err.get("code"),
                subcode=err.get("error_subcode"),
                http_status=getattr(resp, "status_code", None),
                fbtrace_id=err.get("fbtrace_id"),
            )
        if resp.status_code >= 400:
            raise GraphAPIError(
                message=f"HTTP {resp.status_code} from Graph API",
                http_status=resp.status_code,
            )
        return payload if isinstance(payload, dict) else {}

    def _get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        resp = self.session.get(self._url(endpoint), params=params,
                                timeout=self.timeout)
        return self._parse(resp)

    def _post(self, endpoint: str, data: Optional[dict] = None) -> dict:
        resp = self.session.post(self._url(endpoint), data=data,
                                 timeout=self.timeout)
        return self._parse(resp)

    @staticmethod
    def _require_live() -> None:
        if not _live_enabled():
            raise RuntimeError(
                f"{KILL_SWITCH_ENV}!=1: Instagram Graph calls are disabled "
                f"(set {KILL_SWITCH_ENV}=1 for live mode)")

    # -- publishing pipeline -------------------------------------------------

    def create_reels_container(self, *, caption: str, share_to_feed: bool = True,
                               audio_name: Optional[str] = None,
                               trial_params: Optional[dict] = None) -> str:
        self._require_live()
        data: Dict[str, Any] = {
            "media_type": "REELS",
            "upload_type": "resumable",
            "caption": caption,
            "share_to_feed": "true" if share_to_feed else "false",
        }
        if audio_name:
            data["audio_name"] = audio_name
        if trial_params:
            data["trial_params"] = json.dumps(trial_params)
        payload = self._post(f"{self.ig_user_id}/media", data=data)
        creation_id = payload.get("id")
        if not creation_id:
            raise ValueError(
                f"Container create returned no id: {payload!r}")
        log.info("Created REELS container %s", creation_id)
        return str(creation_id)

    def upload_video_bytes(self, creation_id: str, video_path: Path) -> dict:
        self._require_live()
        path = Path(video_path)
        size = path.stat().st_size
        url = f"{self.rupload_base}/ig-api-upload/{self.version}/{creation_id}"
        headers = {
            "Authorization": f"OAuth {self.token}",
            "offset": "0",
            "file_size": str(size),
            "Content-Type": "video/mp4",
        }
        resp = self.session.post(url, headers=headers,
                                 data=path.read_bytes(),
                                 timeout=max(self.timeout, 300.0))
        payload = self._parse(resp)
        log.info("rupload %s (%d bytes): %s", creation_id, size,
                 payload.get("success"))
        return payload

    def container_status(self, creation_id: str) -> dict:
        self._require_live()
        payload = self._get(str(creation_id),
                            params={"fields": "status_code,status,id"})
        return {"status_code": payload.get("status_code"),
                "status": payload.get("status"),
                "id": payload.get("id")}

    def publish_container(self, creation_id: str) -> Optional[str]:
        self._require_live()
        try:
            payload = self._post(f"{self.ig_user_id}/media_publish",
                                 data={"creation_id": str(creation_id)})
            return str(payload["id"])
        except (requests.RequestException, GraphAPIError, KeyError,
                ValueError) as exc:
            log.warning(
                "media_publish %s outcome AMBIGUOUS (%s) — caller MUST "
                "re-query container_status; PUBLISHED means success",
                creation_id, exc)
            return None

    def media_permalink(self, media_id: str) -> str:
        self._require_live()
        payload = self._get(str(media_id), params={"fields": "permalink"})
        permalink = payload.get("permalink")
        if not permalink:
            raise ValueError(f"No permalink for media {media_id}: "
                             f"{payload!r}")
        return str(permalink)

    # -- REAL hashtag signals (need Public Content Access approval) ----------

    def hashtag_search(self, q: str) -> dict:
        self._require_live()
        return self._get("ig_hashtag_search",
                         params={"user_id": self.ig_user_id, "q": str(q)})

    def hashtag_recent_velocity(self, hashtag_id: str) -> int:
        self._require_live()
        cutoff = time.time() - RECENT_WINDOW_S
        count = 0
        after: Optional[str] = None
        for _ in range(max(1, self.max_hashtag_pages)):
            params = {"user_id": self.ig_user_id,
                      "fields": "id,timestamp", "limit": 50}
            if after:
                params["after"] = after
            page = self._get(f"{hashtag_id}/recent_media", params=params)
            items = [i for i in page.get("data") or [] if isinstance(i, dict)]
            in_window = 0
            older_seen = False
            for item in items:
                ts = _parse_timestamp(item.get("timestamp"))
                if ts is None:
                    continue
                if ts >= cutoff:
                    in_window += 1
                else:
                    older_seen = True
            count += in_window
            paging = page.get("paging") or {}
            cursors = paging.get("cursors") or {}
            after = cursors.get("after")
            if not after or older_seen or in_window == 0:
                break
        return count

    def hashtag_top_engagement(self, hashtag_id: str) -> float:
        self._require_live()
        page = self._get(
            f"{hashtag_id}/top_media",
            params={"user_id": self.ig_user_id,
                    "fields": "like_count,comment_count", "limit": 50})
        values = []
        for item in page.get("data") or []:
            if not isinstance(item, dict):
                continue
            likes = item.get("like_count") or 0
            comments = item.get("comment_count") or 0
            values.append(float(likes) + float(comments))
        if not values:
            return 0.0
        return float(statistics.median(values))


def _parse_timestamp(raw: Any) -> Optional[float]:
    text = str(raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None
