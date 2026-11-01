"""Small, shared contracts for publication artifacts (not an outcome store)."""

import json
import os
import re
import tempfile
from pathlib import Path


def validate_clip_id(clip_id: str) -> str:
    if not isinstance(clip_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", clip_id):
        raise ValueError("Unsafe clip ID")
    return clip_id


def atomic_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tmp",
                                         dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def valid_metadata(payload: object) -> bool:
    """Structural validity only; grounding and editorial review are separate gates."""
    return (
        isinstance(payload, dict)
        and not payload.get("_seo_failed")
        and payload.get("metadata_valid") is not False
        and all(isinstance(payload.get(key), str) and payload[key].strip()
                for key in ("title", "description"))
    )


def read_metadata(clip: Path) -> dict | None:
    clip = Path(clip)
    if clip.with_name(f"{clip.stem}_seo_failed.json").exists():
        return None
    try:
        payload = json.loads(clip.with_name(f"{clip.stem}_metadata.json").read_text(encoding="utf-8"))
        return payload if valid_metadata(payload) else None
    except (OSError, ValueError, TypeError):
        return None
