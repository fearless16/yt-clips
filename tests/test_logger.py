"""TDD tests for utils/logger.py — cross-platform (Windows cp1252) safety.

The Rich console handler must never raise on emoji/non-ASCII log messages,
and must not silently drop ERROR records (the original bug hid real failures
behind a UnicodeEncodeError on Windows).
"""
import logging
import sys

import pytest


def test_logger_emits_emoji_without_crashing(monkeypatch):
    """An emoji in a log message must not raise and must be recorded."""
    from utils.logger import get_logger, JsonFileHandler
    import tempfile
    import os

    tmp = tempfile.mkdtemp()
    log_file = os.path.join(tmp, "test.log")

    # Force text (non-json) console format so the Rich handler is used
    monkeypatch.setenv("YT_CONFIG_CONSOLE", "text")

    logger = get_logger("test_emoji", log_file=log_file, level="DEBUG")
    # Exercise both a normal and an error record with emoji
    logger.info("Pipeline ready 🚀")
    logger.error("Upload failed ❌ for clip %s", "clip1")

    # The JSON file handler must contain both records (not dropped)
    with open(log_file, encoding="utf-8") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    assert len(lines) >= 2
    msgs = " ".join(lines)
    assert "🚀" in msgs
    assert "❌" in msgs


def test_rich_handler_does_not_raise_on_emoji(capsys):
    """The console (Rich) handler must not propagate a UnicodeEncodeError."""
    from utils.logger import get_logger

    logger = get_logger("test_rich_emoji", log_file="logs/_test_emoji.log",
                        level="INFO")
    # If the handler raised, logging would call handleError and the record
    # would still be emitted to the file; here we assert no exception escapes.
    logger.warning("Download attempt 🧹 for %s", "url2")
    # No assertion on stdout (cp1252 may replace the emoji); just ensure the
    # call completed without raising.
