"""
scheduler.py — Smart scheduling with jittered hourly slots + prime-time prioritization.

Generates deterministic-jittered upload schedules so every hour has a different
minute offset (e.g. 8:05, 9:12, 10:37). Best-performing clips are assigned to
prime-time windows (evening 19-22 IST, lunch 12-14 IST).

Design:
  - Jitter is deterministic per date+hour (MD5 hash → minute offset)
  - Same date+hour always produces the same offset (stable scheduling)
  - Different hours get different offsets (natural-looking jitter)
  - Prime-time slots are ranked: evening > lunch > off-peak
  - Best clips mapped to highest-ranked slots
"""

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

STATE_FILE = "scheduler_state.json"
IST = timezone(timedelta(hours=5, minutes=30))

# Prime-time windows (IST) in priority order (index, start_hour, end_hour)
# Lower index = higher priority for best-clip assignment
PRIME_WINDOWS: List[Tuple[int, int, int]] = [
    (0, 19, 22),   # Evening prime: 7 PM – 10 PM — highest engagement
    (1, 12, 14),   # Lunch prime:   12 PM – 2 PM — lunch scroll
]


def _now_ist() -> datetime:
    return datetime.now(IST)


def _jitter_minutes(dt: datetime, max_minutes: int = 55) -> int:
    """Deterministic minute jitter for a given datetime slot.

    Uses MD5 hash of YYYY-MM-DD-HH so every hour gets a unique, stable offset.
    Capped at 55 so jitter never spills into the next hour.
    """
    seed = dt.strftime("%Y-%m-%d-%H")
    h = int(hashlib.md5(seed.encode()).hexdigest()[:8], 16)
    return h % (max_minutes + 1)


def is_prime_time(dt: datetime) -> bool:
    """Check whether a datetime falls inside a prime-time window."""
    hour = dt.hour
    for _, start, end in PRIME_WINDOWS:
        if start <= hour < end:
            return True
    return False


def _slot_priority(dt: datetime) -> Tuple[int, int]:
    """Return a sortable priority tuple (lower = better slot).

    Evening prime → (0, -hour)  — later evening beats earlier
    Lunch prime   → (1, -hour)  — later lunch beats earlier
    Off-peak      → (2, hour)   — earlier off-peak first
    """
    hour = dt.hour
    for idx, start, end in PRIME_WINDOWS:
        if start <= hour < end:
            return (idx, -hour)
    return (len(PRIME_WINDOWS), hour)


def generate_schedule(
    num_slots: int,
    interval_hours: int = 1,
    start_from: Optional[datetime] = None,
) -> List[datetime]:
    """Generate *num_slots* jittered timestamps, each *interval_hours* apart.

    Each slot's minute offset is deterministic (date+hour hash), so the same
    day-hour always gets the same jitter.  Slots begin on the next clean hour
    boundary (not sub-minute).
    """
    if start_from is None:
        start_from = _now_ist()

    # Round up to the next hour
    start = start_from.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

    slots: List[datetime] = []
    for i in range(num_slots):
        base = start + timedelta(hours=i * interval_hours)
        jitter = _jitter_minutes(base)
        slots.append(base.replace(minute=jitter))
    return slots


def assign_clips_to_slots(
    clips: List[str],
    interval_hours: int = 1,
    clip_scores: Optional[Dict[str, float]] = None,
) -> List[Tuple[str, datetime]]:
    """Map clips → jittered slots, putting the best clip(s) in prime time.

    Args:
        clips: List of clip paths or identifiers (first = highest priority if no scores).
        interval_hours: Hours between consecutive slots.
        clip_scores: Optional dict of clip_id → quality score (higher = better).

    Returns:
        List of (clip_identifier, scheduled_datetime) sorted chronologically.
    """
    slots = generate_schedule(len(clips), interval_hours)

    # Rank slots best → worst
    ranked: List[Tuple[Tuple[int, int], int, datetime]] = []
    for i, slot in enumerate(slots):
        ranked.append((_slot_priority(slot), i, slot))
    ranked.sort(key=lambda x: x[0])  # lower priority tuple = better slot

    # Rank clips best → worst
    if clip_scores:
        scored = [(clip_scores.get(c, 0.0), c) for c in clips]
        scored.sort(reverse=True, key=lambda x: x[0])
        ranked_clips = [c for _, c in scored]
    else:
        ranked_clips = list(clips)  # first = best (as provided by pipeline)

    # Assign: best clip → best slot, 2nd best → 2nd best slot, etc.
    assignments: List[Tuple[str, datetime]] = []
    for rank_idx, (_, slot_idx, slot_dt) in enumerate(ranked):
        if rank_idx < len(ranked_clips):
            assignments.append((ranked_clips[rank_idx], slot_dt))

    # Return in chronological order
    assignments.sort(key=lambda x: x[1])
    return assignments


# ─── Legacy / sequential helpers (used by pipeline.py loop) ──────────────

def get_next_slot(interval_hours: int = 1) -> datetime:
    """Return the next available jittered slot.

    Tracks state via scheduler_state.json so consecutive calls yield different
    slots rather than the same slot repeatedly.

    This is used by pipeline.py's sequential upload loop.
    """
    now = _now_ist()

    state: Dict = {}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                state = json.load(f)
        except (json.JSONDecodeError, OSError):
            state = {}

    last_scheduled_str = state.get("last_scheduled")
    if last_scheduled_str:
        try:
            last_scheduled = datetime.fromisoformat(last_scheduled_str)
        except (ValueError, TypeError):
            last_scheduled = now
    else:
        last_scheduled = now

    # Next slot = last + interval
    next_slot = last_scheduled + timedelta(hours=interval_hours)

    # If we fell behind (e.g. pipeline restarted), start from next clean hour
    if next_slot < now:
        next_slot = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        jitter = _jitter_minutes(next_slot)
        next_slot = next_slot.replace(minute=jitter)
    else:
        jitter = _jitter_minutes(next_slot)
        next_slot = next_slot.replace(minute=jitter)

    # Persist
    state["last_scheduled"] = next_slot.isoformat()
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    return next_slot


def get_next_upload_time(interval_hours: int = 1) -> str:
    """Convenience: returns ISO 8601 string for the next jittered slot."""
    return format_for_youtube(get_next_slot(interval_hours))


def format_for_youtube(dt: datetime) -> str:
    """Format datetime for YouTube API (ISO 8601 with timezone)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def reset_state() -> None:
    """Clear scheduler state (useful for testing / manual override)."""
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)


# ─── Diagnostic ──────────────────────────────────────────────────────────

def preview_schedule(
    num_clips: int,
    interval_hours: int = 1,
) -> List[Dict]:
    """Return a human-readable schedule preview (no side effects)."""
    slots = generate_schedule(num_clips, interval_hours)
    preview = []
    for i, slot in enumerate(slots):
        preview.append({
            "clip_index": i,
            "datetime_ist": slot.strftime("%Y-%m-%d %H:%M"),
            "is_prime": is_prime_time(slot),
            "jitter_minutes": slot.minute,
        })
    return preview


if __name__ == "__main__":
    import json as _json
    print("=== Schedule Preview (next 8 slots, 1h interval) ===")
    for entry in preview_schedule(8, 1):
        mark = " ★ PRIME" if entry["is_prime"] else ""
        print(f"  Clip #{entry['clip_index']}: {entry['datetime_ist']}{mark}")
