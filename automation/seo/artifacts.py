import json
from pathlib import Path
from typing import Any


def is_valid_metadata(payload: Any) -> bool:
    """Return whether a persisted payload satisfies the upload schema."""
    if not isinstance(payload, dict) or "_seo_failed" in payload:
        return False
    title = payload.get("title")
    description = payload.get("description")
    return (
        isinstance(title, str)
        and bool(title.strip())
        and isinstance(description, str)
        and bool(description.strip())
    )


def load_valid_metadata(
    metadata_path: Path,
    failure_marker_path: Path | None = None,
) -> dict[str, Any] | None:
    """Load valid metadata, treating a failure marker as authoritative."""
    if failure_marker_path is not None and failure_marker_path.is_file():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return payload if is_valid_metadata(payload) else None
