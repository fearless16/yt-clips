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
HOOK_PUBLISH = "publish"

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


def _upload_timing() -> tuple[float, float]:
    """(poll_interval_s, timeout_s) from config instagram block, defaults kept."""
    defaults = (10.0, 900.0)
    try:
        from utils.config import load_config
        cfg = (load_config() or {}).get("instagram") or {}
    except Exception:
        return defaults
    try:
        interval = float(cfg.get("container_poll_interval_s", defaults[0]))
        timeout = float(cfg.get("upload_timeout_s", defaults[1]))
        return (interval, timeout)
    except (TypeError, ValueError):
        return defaults


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
        publish_fn = resolve_stage_hook(HOOK_PUBLISH)
        result = publish_fn(
            video_path=video_path, caption=caption,
            audio_name=_resolve_audio_name(clip_dir), clip_dir=clip_dir,
            state=dict(state)) if publish_fn is not None else None
        if result is not None:
            _save("DONE", last_error=None, caption=caption,
                  media_id=result["media_id"],
                  permalink=result.get("permalink", ""),
                  provider=str(result.get("provider") or "publish_hook"))
            (clip_dir / FAILED_MARKER).unlink(missing_ok=True)
            return result["media_id"]

        client_fn = resolve_stage_hook(HOOK_MAKE_CLIENT)
        client = client_fn() if client_fn is not None else \
            _default_graph_client()
        if client is None:
            raise RuntimeError(
                "Instagram GraphClient unavailable (graph_client module "
                "missing or misconfigured)")

        poll_interval_s, upload_timeout_s = _upload_timing()
        resume_creation_id = str(state.get("creation_id") or "").strip() or None

        from automation.instagram.uploader import (
            InstagramUploadError,
            publish_reel,
        )

        def _on_creation(creation_id: str) -> None:
            _write_state(clip_dir, "UPLOAD", attempts,
                         creation_id=creation_id, caption=caption)

        result = publish_reel(
            client, video_path, caption,
            audio_name=_resolve_audio_name(clip_dir),
            poll_interval_s=poll_interval_s,
            timeout_s=upload_timeout_s,
            resume_creation_id=resume_creation_id,
            on_creation=_on_creation,
        )

        _save("DONE", last_error=None, caption=caption,
              media_id=result["media_id"], permalink=result["permalink"])
        (clip_dir / FAILED_MARKER).unlink(missing_ok=True)
        return result["media_id"]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        stage_label = getattr(exc, "stage", "") or active
        _write_failed_marker(clip_dir, transcript, video_title,
                             video_description, stage=stage_label, error=error)
        _write_state(clip_dir, checkpoint, attempts, last_error=error)
        return None


def _default_graph_client():
    try:
        from automation.instagram.graph_client import FacebookGraphClient
        from automation.instagram.credential import load_token
        token = load_token()
    except ImportError:
        return None
    if not token:
        return None
    ig_user_id = str(token.get("ig_user_id") or "").strip()
    access = str(token.get("access_token") or "").strip()
    if not ig_user_id or not access:
        return None
    try:
        return FacebookGraphClient(access, ig_user_id)
    except Exception:
        return None


_RETRY_ATTEMPT_CAP = 3


def resolve_insta_transcript(clip_dir, info=None) -> str:
    """Best-effort clip transcript: export-time slice first."""
    text = ""
    if isinstance(info, dict):
        text = str(info.get("text") or "").strip()
    if not text:
        state = _read_json(Path(clip_dir) / STATE_FILE) or {}
        text = str(state.get("transcript") or "").strip()
    return text


def retry_failed_insta(shorts_root="shorts", *, force: bool = False) -> dict:
    """Consume *_insta_failed markers under shorts_root.

    Explicit user invocation ⇒ skip flags are bypassed (force=True).
    Markers whose state.attempts hit _RETRY_ATTEMPT_CAP stay parked unless
    force=True. Marker deleted on recovery; kept on continued failure.
    """
    root = Path(shorts_root)
    if not root.is_dir():
        return {"retried": 0, "recovered": 0, "still_failed": 0}

    markers = sorted(root.rglob(FAILED_MARKER))
    retried = 0
    recovered = 0
    for marker in markers:
        payload = _read_json(marker) or {}
        state = _read_json(marker.parent / STATE_FILE) or {}
        if not force and int(state.get("attempts", 0)) >= _RETRY_ATTEMPT_CAP:
            continue
        retried += 1
        try:
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
