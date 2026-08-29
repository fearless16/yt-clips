"""Small fail-soft client for vidIQ's Streamable HTTP MCP endpoint."""

import json
import os
import threading
import time
from typing import Callable, Dict, Optional

import requests

from utils.logger import get_logger
from utils.config import load_config


_cfg = load_config()
_log = get_logger("vidiq", _cfg["logging"]["log_file"], _cfg["logging"]["level"])


_CACHE_LOCK = threading.Lock()
_TOOL_CACHE: Dict[str, tuple[float, object]] = {}


def _cache_key(name: str, arguments: Dict) -> str:
    try:
        args = json.dumps(arguments or {}, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    except (TypeError, ValueError):
        args = repr(arguments)
    return f"{name}|{args}"


def _cache_get(key: str) -> Optional[object]:
    now = time.monotonic()
    with _CACHE_LOCK:
        item = _TOOL_CACHE.get(key)
        if not item:
            return None
        expires_at, value = item
        if expires_at <= now:
            _TOOL_CACHE.pop(key, None)
            return None
        return value


def _cache_set(key: str, value: object, ttl_seconds: float) -> None:
    if ttl_seconds <= 0:
        return
    with _CACHE_LOCK:
        _TOOL_CACHE[key] = (time.monotonic() + ttl_seconds, value)
        # Hard cap: this worker may run for days. Do not build a tiny in-memory landfill.
        if len(_TOOL_CACHE) > 512:
            oldest = sorted(_TOOL_CACHE.items(), key=lambda kv: kv[1][0])[:128]
            for old_key, _ in oldest:
                _TOOL_CACHE.pop(old_key, None)


class _RetryableVidiqError(RuntimeError):
    pass


class VidiqClient:
    ENDPOINT = "https://mcp.vidiq.com/mcp"

    def __init__(
        self,
        *,
        enabled: bool = False,
        timeout_seconds: float = 8.0,
        endpoint: str = ENDPOINT,
        post: Callable = requests.post,
        max_calls: int = 4,
        default_cache_ttl_seconds: float = 1800.0,
    ):
        self.enabled = bool(enabled)
        try:
            timeout = float(timeout_seconds)
        except (TypeError, ValueError):
            timeout = 8.0
        self.timeout_seconds = max(1.0, min(timeout, 30.0))
        self.endpoint = endpoint
        self._post = post
        self._calls = 0
        self._used = False
        self._key_slot: Optional[str] = None
        try:
            self.max_calls = max(1, min(int(max_calls), 12))
        except (TypeError, ValueError):
            self.max_calls = 4
        try:
            self.default_cache_ttl_seconds = max(0.0, float(default_cache_ttl_seconds))
        except (TypeError, ValueError):
            self.default_cache_ttl_seconds = 1800.0
        self._cache_hits = 0
        self._budget_exhausted = False

    @property
    def audit(self) -> Dict:
        return {
            "enabled": self.enabled,
            "used": self._used,
            "calls": self._calls,
            "cache_hits": self._cache_hits,
            "max_calls": self.max_calls,
            "budget_exhausted": self._budget_exhausted,
            "key_slot": self._key_slot,
        }

    def call_tool(
        self,
        name: str,
        arguments: Dict,
        *,
        cache_ttl_seconds: Optional[float] = None,
    ) -> Optional[object]:
        """Call one MCP tool with a bounded budget and process-level TTL cache.

        ``calls`` counts logical network calls, not cache hits or credential retries.
        This keeps per-clip vidIQ usage predictable while still allowing a backup
        credential on transient/quota failures.
        """
        if not self.enabled:
            return None

        ttl = self.default_cache_ttl_seconds if cache_ttl_seconds is None else cache_ttl_seconds
        try:
            ttl = max(0.0, float(ttl))
        except (TypeError, ValueError):
            ttl = self.default_cache_ttl_seconds
        key = _cache_key(str(name), dict(arguments or {}))
        cached = _cache_get(key) if ttl > 0 else None
        if cached is not None:
            self._cache_hits += 1
            self._used = True
            return cached

        if self._calls >= self.max_calls:
            self._budget_exhausted = True
            _log.warning("vidIQ call budget exhausted for this SEO request (%d)", self.max_calls)
            return None

        keys = [
            ("primary", os.getenv("VIDIQ_API_KEY", "").strip()),
            ("backup", os.getenv("BACKUP_VIDIQ_API_KEY", "").strip()),
        ]
        keys = [(slot, key_value) for slot, key_value in keys if key_value]
        if not keys:
            _log.info("vidIQ disabled for this call: no API credential configured")
            return None

        self._calls += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._calls,
            "method": "tools/call",
            "params": {"name": str(name), "arguments": dict(arguments or {})},
        }
        for slot, api_key in keys:
            try:
                response = self._post(
                    self.endpoint,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Accept": "text/event-stream, application/json",
                        "Content-Type": "application/json",
                        "MCP-Protocol-Version": "2025-03-26",
                    },
                    timeout=self.timeout_seconds,
                )
                # Only explicit credential/quota/billing failures are safe reasons
                # to try the backup key. Timeout/5xx outcomes are ambiguous: the
                # paid tool may already have executed upstream, so retrying can
                # double-charge credits for the same logical SEO request.
                if response.status_code in (401, 402, 403, 429):
                    raise _RetryableVidiqError()
                if response.status_code == 408 or response.status_code >= 500:
                    _log.warning(
                        "vidIQ MCP transient HTTP failure (%s); not retrying paid call",
                        response.status_code,
                    )
                    return None
                response.raise_for_status()
                message = self._parse_stream(response.text)
                if isinstance(message, dict) and message.get("error"):
                    if self._is_retryable_error(message["error"]):
                        raise _RetryableVidiqError()
                    _log.warning("vidIQ MCP rejected the tool request")
                    return None
                try:
                    value = self._content_value(message)
                except RuntimeError:
                    if self._is_retryable_error(message.get("result", {})):
                        raise _RetryableVidiqError()
                    _log.warning("vidIQ tool returned a non-retryable error")
                    return None
                self._used = True
                self._key_slot = slot
                _cache_set(key, value, ttl)
                return value
            except _RetryableVidiqError as exc:
                _log.warning("vidIQ MCP %s credential/quota failure (%s)", slot, type(exc).__name__)
                continue
            except (requests.Timeout, requests.ConnectionError) as exc:
                _log.warning(
                    "vidIQ MCP %s transport failure (%s); not retrying paid call",
                    slot, type(exc).__name__,
                )
                return None
            except requests.HTTPError:
                _log.warning("vidIQ MCP rejected the HTTP request")
                return None
            except Exception as exc:  # fail-soft boundary for malformed optional responses
                _log.warning("vidIQ MCP response failed (%s)", type(exc).__name__)
                return None
        return None

    @staticmethod
    def _is_retryable_error(value: object) -> bool:
        text = json.dumps(value, ensure_ascii=True).casefold()
        return any(token in text for token in (
            "credit", "quota", "rate limit", "rate_limit", "unauthorized",
            "forbidden", "authentication", "billing",
        ))

    @staticmethod
    def _parse_stream(body: str) -> Dict:
        stripped = (body or "").replace("\r\n", "\n").strip()
        if not stripped:
            raise ValueError("empty MCP response")
        if stripped.startswith("{"):
            return json.loads(stripped)
        for event in stripped.split("\n\n"):
            data_lines = [line[5:].lstrip() for line in event.splitlines() if line.startswith("data:")]
            if not data_lines:
                continue
            data = "\n".join(data_lines)
            if data == "[DONE]":
                continue
            parsed = json.loads(data)
            if isinstance(parsed, dict) and ("result" in parsed or "error" in parsed):
                return parsed
        raise ValueError("no JSON-RPC event in MCP response")

    @staticmethod
    def _content_value(message: Dict) -> object:
        result = message.get("result") or {}
        if result.get("isError"):
            raise RuntimeError("vidIQ tool reported an error")
        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            return structured
        widget_data = (result.get("_meta") or {}).get("widgetData")
        if isinstance(widget_data, dict):
            return widget_data
        content = result.get("content") or []
        text = next(
            (item.get("text") for item in content if isinstance(item, dict) and item.get("type") == "text"),
            None,
        )
        if text is None:
            return result
        try:
            return json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return text
