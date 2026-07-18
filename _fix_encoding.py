"""_fix_encoding.py — Enable full Unicode/emoji support on Windows.

Uses colorama.just_fix_windows_console() (v0.4.6+) to enable Windows' built-in
VT processing and Unicode support. Falls back to UTF-8 reconfigure if colorama
is unavailable.

Import this at the TOP of any CLI entry point:
    import _fix_encoding  # noqa: F401
"""
import io
import sys


def _enable_unicode() -> None:
    # 1. colorama: enables VT processing on Windows 10+ → full Unicode support
    try:
        from colorama import just_fix_windows_console
        just_fix_windows_console()
    except (ImportError, AttributeError):
        pass

    # 2. Fallback: reconfigure streams to UTF-8 if still on cp1252
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if stream is None:
            continue
        enc = getattr(stream, "encoding", "") or ""
        if enc.lower() == "utf-8":
            continue
        try:
            setattr(
                sys,
                name,
                io.TextIOWrapper(
                    stream.buffer,
                    encoding="utf-8",
                    errors="replace",
                    line_buffering=stream.line_buffering,
                ),
            )
        except Exception:
            pass


_enable_unicode()
