"""Serial CLI config scope: new sources never replace the canonical input.

The existing workers share utils.config's cached dictionary. Preserve that
object while temporarily routing ALL stages to the same per-source paths.
This is not a concurrent-run API or a persistent settings change.
"""

import copy
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import parse_qs, urlparse

_SCOPE_LOCK = threading.Lock()


def youtube_source_id(url):
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    parts = parsed.path.strip("/").split("/")
    video_id = None
    if parsed.scheme in ("http", "https"):
        if host == "youtu.be" and len(parts) == 1:
            video_id = parts[0]
        elif host in ("youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"):
            if parts == ["watch"]:
                video_id = parse_qs(parsed.query).get("v", [None])[0]
            elif len(parts) == 2 and parts[0] in ("shorts", "live", "embed"):
                video_id = parts[1]
    if not isinstance(video_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("--source-scoped requires a real YouTube video URL with an 11-character ID")
    return video_id


@contextmanager
def source_runtime(url):
    from utils.config import load_config
    from automation import orchestrator
    source_id = youtube_source_id(url)
    shared = load_config()
    if not _SCOPE_LOCK.acquire(blocking=False):
        raise RuntimeError("Source-scoped CLI runs must be serialized")
    snapshot, previous_runner_config = dict(shared), orchestrator._CONFIG
    try:
        runtime = copy.deepcopy(shared)
        paths = runtime.setdefault("paths", {})
        for name in ("input", "transcripts", "highlights", "shorts"):
            base = Path(paths.get(name, name))
            paths[name] = str(base if base.name == source_id else base / source_id)
        shared.clear()
        shared.update(runtime)
        orchestrator._CONFIG = shared
        yield shared
    finally:
        shared.clear()
        shared.update(snapshot)
        orchestrator._CONFIG = previous_runner_config
        _SCOPE_LOCK.release()
