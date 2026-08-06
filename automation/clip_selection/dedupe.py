"""Dedupe engine — cross-run time-window duplicate detection.

Prevents re-uploading near-identical clips when the same match is re-processed.
The primary signal is TIME-WINDOW OVERLAP with previously selected clips
(same match → same highlight windows get re-selected). Token-similarity text
matching is intentionally avoided: repeated phrases like "SIX! SIX!" are
legitimately distinct moments.

The signal used is the fraction of the candidate window that was already
published: ``shared / candidate_duration``. A candidate fully contained inside
a previously uploaded window scores 1.0 (reject — it is a re-upload), while a
much larger candidate that merely contains a small previously uploaded window
scores low (accept — mostly new content).

History is persisted as an append-only sidecar file so dedup guards against
every prior run, not just the immediately-previous one.

Key API:
    load_previous_windows(yaml_path)  → parse previous highlights/history YAML
    fraction_republished(a, b)        → how much of the candidate was published
    overlaps_any_previous(...)        → bool check against prior selections
    append_windows(history_path, ...) → grow the cross-run history atomically
"""

from pathlib import Path
from typing import Any

import yaml

from utils.logger import get_logger

log = get_logger("dedupe")


def fraction_republished(
    cand_start: float,
    cand_end: float,
    prev_start: float,
    prev_end: float,
) -> float:
    """Return what fraction of the candidate window was already published.

    ``shared / candidate_duration``, clamped to [0.0, 1.0]. A candidate fully
    inside a previous window scores 1.0; a candidate containing a small
    previously-published sub-window scores low. Zero/negative candidate
    duration scores 0.0.
    """
    cand_duration = cand_end - cand_start
    if cand_duration <= 0:
        return 0.0
    shared = max(0.0, min(cand_end, prev_end) - max(cand_start, prev_start))
    return min(1.0, shared / cand_duration)


def _parse_timestamp(value: Any) -> float | None:
    """Parse a timestamp that may be a float or an "HH:MM:SS" string."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        parts = value.strip().split(":")
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            return None
        if len(nums) == 3:
            return nums[0] * 3600 + nums[1] * 60 + nums[2]
        if len(nums) == 2:
            return nums[0] * 60 + nums[1]
        if len(nums) == 1:
            return nums[0]
    return None


def _entry_coords(entry: dict) -> tuple[float, float] | None:
    """Extract (start, end) seconds from a yaml clip entry, or None."""
    if "start_sec" in entry:
        start = _parse_timestamp(entry.get("start_sec"))
    else:
        start = _parse_timestamp(entry.get("start"))
    if "end_sec" in entry:
        end = _parse_timestamp(entry.get("end_sec"))
    else:
        end = _parse_timestamp(entry.get("end"))
    if start is None or end is None or end <= start:
        return None
    return float(start), float(end)


def load_previous_windows(yaml_path: str | Path) -> list[dict[str, float]]:
    """Load previously selected clip windows from a highlights/history YAML.

    Accepts dict-keyed (``clip1: ...``) or list-of-dict layouts. Reads
    ``start_sec``/``end_sec`` when present, falling back to ``start``/``end``
    timestamp strings. Entries without usable coordinates are skipped.
    Returns [] for missing or corrupt files.

    Args:
        yaml_path: Path to a previous highlights/history YAML.

    Returns:
        List of ``{"start": float, "end": float}`` dicts, sorted by start.
    """
    path = Path(yaml_path)
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception:
        log.warning("Failed to parse dedup history %s — skipping", path)
        return []
    if isinstance(data, dict):
        entries = list(data.values())
    elif isinstance(data, list):
        entries = data
    else:
        return []

    windows = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        coords = _entry_coords(entry)
        if coords is None:
            continue
        start, end = coords
        windows.append({"start": start, "end": end})

    windows.sort(key=lambda w: w["start"])
    return windows


def _valid_interval(entry: dict[str, float]) -> tuple[float, float] | None:
    """Extract a numeric (start, end) interval from a history entry, or None.

    Rejects bools (a subclass of int that must never be treated as a
    timestamp), non-numeric values, and empty/inverted windows.
    """
    prev_start = entry.get("start")
    prev_end = entry.get("end")
    if isinstance(prev_start, bool) or isinstance(prev_end, bool):
        return None
    if not isinstance(prev_start, (int, float)) or not isinstance(prev_end, (int, float)):
        return None
    if prev_end <= prev_start:
        return None
    return float(prev_start), float(prev_end)


def _merge_intervals(windows: list[dict[str, float]]) -> list[tuple[float, float]]:
    """Merge overlapping/adjacent previous windows into disjoint intervals.

    Previous windows are sorted by start. Unioning them first closes the hole
    where a candidate spanning several individually-below-threshold windows
    (e.g. 0-100s vs [10-50, 60-90]) would otherwise slip through dedup.
    """
    intervals = [_valid_interval(w) for w in windows]
    intervals = [iv for iv in intervals if iv is not None]
    intervals.sort(key=lambda iv: iv[0])
    merged: list[tuple[float, float]] = []
    for s, e in intervals:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def overlaps_any_previous(
    start: float,
    end: float,
    previous_windows: list[dict[str, float]],
    threshold: float = 0.6,
) -> bool:
    """Return True if enough of the candidate was already published.

    Previous windows are unioned first (overlapping/adjacent windows merge),
    so a candidate spanning multiple previously published windows is measured
    against their combined coverage, not window-by-window.

    Args:
        start: Candidate start (seconds).
        end: Candidate end (seconds).
        previous_windows: List of {"start", "end"} from prior runs.
        threshold: Min fraction of the candidate that must have been published
            to count as a duplicate. Clamped to (0, 1].
    """
    intervals = _merge_intervals(previous_windows)
    if not intervals:
        return False
    threshold = max(0.0, min(1.0, threshold))
    cand_duration = end - start
    if cand_duration <= 0:
        return False
    shared = 0.0
    for prev_start, prev_end in intervals:
        shared += max(0.0, min(end, prev_end) - max(start, prev_start))
    ratio = shared / cand_duration
    # Strict positive overlap required: touching windows (ratio 0.0) never
    # count as duplicates even when the threshold is configured as 0.
    return ratio > 0.0 and ratio >= threshold


def append_windows(history_path: str | Path, windows: list[dict[str, float]]) -> None:
    """Atomically append selected windows to the cross-run dedup history.

    Loads the existing history, merges new windows (dropping exact
    (start, end) duplicates), and writes back via a temp file + atomic rename
    so a killed run never corrupts the history into a silent dedup-disable.

    Args:
        history_path: Path to the append-only history YAML.
        windows: List of {"start", "end"} windows from the current selection.
    """
    path = Path(history_path)
    existing = load_previous_windows(path)
    existing_keys = {(w["start"], w["end"]) for w in existing}
    merged = list(existing)
    for w in windows:
        key = (w.get("start", 0.0), w.get("end", 0.0))
        if key not in existing_keys:
            existing_keys.add(key)
            merged.append({"start": key[0], "end": key[1]})
    merged.sort(key=lambda w: w["start"])

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, default_flow_style=False, allow_unicode=True)
    tmp_path.replace(path)
