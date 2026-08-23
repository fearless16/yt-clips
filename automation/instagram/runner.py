"""automation/instagram/runner.py

Per-clip Instagram stage machine: EVIDENCE → CAPTION → UPLOAD → DONE.

- Checkpoint file: clip_insta_state.json {stage, attempts, last_error,
  updated_at} — resume skips completed stages.
- Failure at any stage → clip_insta_failed.json marker (same protocol as
  clip_seo_failed.json) consumed by retry_failed_insta(); NEVER propagates.
- Skip paths (env YT_CLIPS_SKIP_INSTAGRAM=1, config instagram.enabled false)
  return None with zero network and no client construction.
- Stage callables are overridable via register_stage_hook(); defaults lazy-
  import automation.instagram.seo (I3) guarded by try/except ImportError →
  None so the runner stays testable standalone.
"""

import importlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

STATE_FILE = "clip_insta_state.json"
FAILED_MARKER = "clip_insta_failed.json"

HOOK_RUN_EVIDENCE = "run_evidence_stage"
HOOK_RUN_CAPTION = "run_caption_stage"
HOOK_MAKE_CLIENT = "make_graph_client"

_HOOKS: dict = {}

_SKIP_ENV_VAR = "YT_CLIPS_SKIP_INSTAGRAM"


def register_stage_hook(name: str, fn) -> None:
    """Register/override a stage callable (e.g. I3's seo modules)."""
    _HOOKS[name] = fn


def resolve_stage_hook(name: str):
    if name in _HOOKS:
        return _HOOKS[name]
    try:
        mod = importlib.import_module("automation.instagram.seo")
    except ImportError:
        return None
    return getattr(mod, name, None)


def _skip_requested() -> bool:
    if os.environ.get(_SKIP_ENV_VAR) == "1":
        return True
    try:
        from utils.config import load_config
        cfg = load_config()
    except Exception:
        return True
    insta_cfg = cfg.get("instagram") or {}
    return not bool(insta_cfg.get("enabled", False))


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _atomic_json(path: Path, data: dict, ensure_ascii: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(data, ensure_ascii=ensure_ascii, indent=2),
        encoding="utf-8")
    os.replace(tmp_path, path)


def _write_state(clip_dir: Path, stage: str, attempts: int, *,
                 last_error=None, **extra) -> None:
    data = _read_json(clip_dir / STATE_FILE) or {}
    data.update(extra)
    data["stage"] = stage
    data["attempts"] = attempts
    data["last_error"] = last_error
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(clip_dir / STATE_FILE, data, ensure_ascii=False)


def _write_failed_marker(clip_dir: Path, transcript: str, video_title: str,
                         video_description: str, stage: str, error: str) -> None:
    marker = {
        "clip_id": clip_dir.name,
        "transcript": transcript,
        "video_title": video_title,
        "video_description": video_description,
        "stage": stage,
        "error": error,
    }
    _atomic_json(clip_dir / FAILED_MARKER, marker, ensure_ascii=True)


def _find_clip_video(clip_dir: Path) -> Path:
    videos = sorted(clip_dir.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"no .mp4 in {clip_dir}")
    return videos[0]


def _resolve_audio_name(clip_dir: Path):
    try:
        from automation.instagram.seo import METADATA_FILE
    except ImportError:
        METADATA_FILE = "insta_metadata.json"
    payload = _read_json(Path(clip_dir) / METADATA_FILE)
    if isinstance(payload, dict):
        audio_name = str(payload.get("audio_name") or "").strip()
        if audio_name:
            return audio_name
    return None


def process_instagram_for_clip(clip_dir: Path, transcript: str,
                               video_title: str, video_description: str,
                               *, force: bool = False):
    """Run the IG stage machine for one clip. Returns media_id or None."""
    clip_dir = Path(clip_dir)
    if not force and _skip_requested():
        return None

    state = _read_json(clip_dir / STATE_FILE) or {}
    if state.get("stage") == "DONE" and state.get("media_id"):
        return state["media_id"]

    attempts = int(state.get("attempts", 0)) + 1
    caption = state.get("caption")
    checkpoint = state.get("stage") or ""
    active = checkpoint or "EVIDENCE"

    def _save(stage_name: str, **extra) -> None:
        nonlocal checkpoint
        checkpoint = stage_name
        _write_state(clip_dir, stage_name, attempts, **extra)

    try:
        if checkpoint not in ("EVIDENCE", "CAPTION", "UPLOAD", "DONE"):
            active = "EVIDENCE"
            evidence_fn = resolve_stage_hook(HOOK_RUN_EVIDENCE)
            if evidence_fn is not None:
                evidence_fn(str(clip_dir), transcript, video_title,
                            video_description)
            _save("EVIDENCE")

        if checkpoint not in ("CAPTION", "UPLOAD", "DONE"):
            active = "CAPTION"
            caption_fn = resolve_stage_hook(HOOK_RUN_CAPTION)
            if caption_fn is not None:
                produced = caption_fn(str(clip_dir), transcript,
                                      video_title, video_description)
                if produced:
                    caption = produced
            if not caption:
                raise RuntimeError(
                    "no caption available "
                    f"(hook {HOOK_RUN_CAPTION} unavailable or empty)")
            _save("CAPTION", caption=caption)

        active = "UPLOAD"
        _save("UPLOAD", caption=caption)

        video_path = _find_clip_video(clip_dir)
        client_fn = resolve_stage_hook(HOOK_MAKE_CLIENT)
        client = client_fn() if client_fn is not None else \
            _default_graph_client()
        if client is None:
            raise RuntimeError(
                "Instagram GraphClient unavailable (graph_client module "
                "missing or misconfigured)")

        from automation.instagram.uploader import publish_reel
        result = publish_reel(client, video_path, caption,
                              audio_name=_resolve_audio_name(clip_dir))

        _save("DONE", last_error=None, caption=caption,
              media_id=result["media_id"], permalink=result["permalink"])
        (clip_dir / FAILED_MARKER).unlink(missing_ok=True)
        return result["media_id"]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        _write_failed_marker(clip_dir, transcript, video_title,
                             video_description, stage=active, error=error)
        _write_state(clip_dir, checkpoint, attempts, last_error=error)
        return None


def _default_graph_client():
    try:
        from automation.instagram.graph_client import FacebookGraphClient
        from automation.instagram.credential import load_page_credentials
    except ImportError:
        return None
    try:
        creds = load_page_credentials()
    except Exception:
        return None
    if not creds:
        return None
    token, ig_user_id = creds
    try:
        return FacebookGraphClient(token, ig_user_id)
    except Exception:
        return None


def retry_failed_insta(shorts_root="shorts") -> dict:
    """Consume *_insta_failed markers under shorts_root.

    Explicit user invocation ⇒ skip flags are bypassed (force=True).
    Marker deleted on recovery; kept on continued failure.
    """
    root = Path(shorts_root)
    if not root.is_dir():
        return {"retried": 0, "recovered": 0, "still_failed": 0}

    markers = sorted(root.rglob(FAILED_MARKER))
    retried = 0
    recovered = 0
    for marker in markers:
        retried += 1
        try:
            payload = _read_json(marker) or {}
            media_id = process_instagram_for_clip(
                marker.parent,
                payload.get("transcript", ""),
                payload.get("video_title", ""),
                payload.get("video_description", ""),
                force=True,
            )
            if media_id:
                recovered += 1
        except Exception:
            pass
    return {"retried": retried, "recovered": recovered,
            "still_failed": retried - recovered}
