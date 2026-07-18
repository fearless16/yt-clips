"""_fix_encoding.py — Force UTF-8 on stdout/stderr for Windows cp1252 compatibility.

Import this at the TOP of any CLI entry point:
    import _fix_encoding  # noqa: F401
"""
import io
import sys

def _patch_stream(name: str) -> None:
    stream = getattr(sys, name)
    if stream is None:
        return
    if getattr(stream, "encoding", "").lower() == "utf-8":
        return
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

_patch_stream("stdout")
_patch_stream("stderr")
