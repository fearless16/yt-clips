"""Instagram token persistence — insta_token.json.

Mirrors upload.py's graceful token handling (missing/corrupt -> None,
never crash the pipeline). Fields: access_token, expires_at (epoch
seconds), ig_user_id. Refresh is intentionally a stub until setup_auth.py
gains the FB long-lived Page-token exchange (PLAN.md §6).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

log = logging.getLogger("insta_credential")

TOKEN_FILENAME = "insta_token.json"
REQUIRED_FIELDS = ("access_token", "expires_at", "ig_user_id")
EXPIRY_SKEW_S = 60.0

PathLike = Union[str, Path]


def load_token(path: Optional[PathLike] = None) -> Optional[Dict[str, Any]]:
    """Load insta_token.json; None when missing or unparsable."""
    target = Path(path) if path else Path(TOKEN_FILENAME)
    if not target.exists():
        log.warning("No %s found.", target)
        return None
    try:
        with open(target, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.error("Failed to parse %s: %s", target, exc)
        return None
    return data if isinstance(data, dict) else None


def save_token(data: Mapping[str, Any],
               path: Optional[PathLike] = None) -> Path:
    """Persist token fields; missing keys are written as empty defaults."""
    target = Path(path) if path else Path(TOKEN_FILENAME)
    payload: Dict[str, Any] = {field: "" for field in REQUIRED_FIELDS}
    payload.update(dict(data))
    with open(target, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return target


def is_valid(token: Any) -> bool:
    """True when an access_token exists and expires_at is in the future."""
    if not isinstance(token, Mapping):
        return False
    if not str(token.get("access_token") or "").strip():
        return False
    try:
        expires_at = float(token.get("expires_at"))
    except (TypeError, ValueError):
        return False
    return time.time() < expires_at - EXPIRY_SKEW_S


def refresh_if_needed(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(
        "Instagram token refresh is not wired yet — extend setup_auth.py "
        "with the FB long-lived Page-token exchange first (PLAN.md §6).")
