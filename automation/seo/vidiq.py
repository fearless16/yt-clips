"""Small fail-soft client for vidIQ's Streamable HTTP MCP endpoint."""

import json
import os
from typing import Callable, Dict, Optional

import requests

from utils.logger import get_logger
from utils.config import load_config


_cfg = load_config()
_log = get_logger("vidiq", _cfg["logging"]["log_file"], _cfg["logging"]["level"])


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

    @property
    def audit(self) -> Dict:
        return {
            "enabled": self.enabled,
            "used": self._used,
            "calls": self._calls,
            "key_slot": self._key_slot,
        }

    def call_tool(self, name: str, arguments: Dict) -> Optional[object]:
        """Call one MCP tool, trying the backup credential after retryable failures."""
        if not self.enabled:
            return None
        keys = [
            ("primary", os.getenv("VIDIQ_API_KEY", "").strip()),
            ("backup", os.getenv("BACKUP_VIDIQ_API_KEY", "").strip()),
        ]
        keys = [(slot, key) for slot, key in keys if key]
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
        for slot, key in keys:
            try:
                response = self._post(
                    self.endpoint,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Accept": "text/event-stream, application/json",
                        "Content-Type": "application/json",
                        "MCP-Protocol-Version": "2025-03-26",
                    },
                    timeout=self.timeout_seconds,
                )
                if response.status_code in (401, 402, 403, 408, 429) or response.status_code >= 500:
                    raise _RetryableVidiqError()
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
                return value
            except (_RetryableVidiqError, requests.Timeout, requests.ConnectionError) as exc:
                # Never interpolate response bodies or credentials into logs.
                _log.warning("vidIQ MCP %s request failed (%s)", slot, type(exc).__name__)
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
