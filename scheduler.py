"""
scheduler.py — Smart scheduling with day-window clamping + jittered hourly slots + prime-time prioritization.

Generates deterministic-jittered upload schedules so every hour has a different
minute offset (e.g. 9:05, 10:12, 11:37). All slots are clamped to 9 AM – 9 PM IST.
Best-performing clips are assigned to prime-time windows (evening 19-21 IST, lunch 12-14 IST).

Design:
  - Jitter is deterministic per date+hour (MD5 hash → minute offset)
  - Same date+hour always produces the same offset (stable scheduling)
  - Different hours get different offsets (natural-looking jitter)
  - Slots before 9 AM pushed to 9 AM; slots at/after 9 PM pushed to next day 9 AM
  - Prime-time slots are ranked: evening > lunch > off-peak
  - Best clips mapped to highest-ranked slots
"""

import hashlib
import json
import os
from datetime import date, datetime, time as dt_time, timedelta, timezone
from typing import Dict, List, Optional, Tuple

STATE_FILE = "scheduler_state.json"
IST = timezone(timedelta(hours=5, minutes=30))

# Day window (IST) — uploads only between 9 AM and 9 PM
DAY_START_HOUR = 9
DAY_END_HOUR = 21

# Prime-time windows (IST) in priority order (index, start_hour, end_hour)
# Lower index = higher priority for best-clip assignment
PRIME_WINDOWS: List[Tuple[int, int, int]] = [
    (0, 19, 21),   # Evening prime: 7 PM – 9 PM — highest engagement
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


def _clamp_to_day_window(dt: datetime) -> datetime:
    """Clamp dt to DAY_START_HOUR–DAY_END_HOUR window (IST).
    Before 9 AM → same day 9 AM (preserves minute).
    9 PM or later → next day 9 AM (preserves minute).
    """
    if dt.hour < DAY_START_HOUR:
        dt = dt.replace(hour=DAY_START_HOUR, second=0, microsecond=0)
    elif dt.hour >= DAY_END_HOUR:
        dt = (dt.replace(hour=DAY_START_HOUR, second=0, microsecond=0)
              + timedelta(days=1))
    return dt


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


def _as_ist(value: datetime) -> datetime:
    """Interpret naive scheduling inputs as IST and normalize aware inputs."""
    if value.tzinfo is None:
        return value.replace(tzinfo=IST)
    return value.astimezone(IST)


def _at_minute(day: date, minute_of_day: int) -> datetime:
    return datetime.combine(day, dt_time(), tzinfo=IST) + timedelta(
        minutes=minute_of_day
    )


def generate_schedule(
    num_slots: int,
    interval_hours: int = 1,
    start_from: Optional[datetime] = None,
    *,
    day_start_hour: int = DAY_START_HOUR,
    day_end_hour: int = DAY_END_HOUR,
    jitter_max_minutes: int = 55,
    spread_across_window: bool = True,
) -> List[datetime]:
    """Generate future IST slots inside a configurable daily window.

    ``interval_hours`` controls the maximum number of uploads per day. When
    ``spread_across_window`` is enabled, that day's clips are distributed from
    morning to evening instead of being packed into consecutive early hours.
    Overflow rolls to the next day without producing duplicate timestamps.
    """
    if num_slots <= 0:
        return []
    if not 0 <= day_start_hour < day_end_hour <= 24:
        raise ValueError("upload window must satisfy 0 <= start < end <= 24")
    if interval_hours <= 0:
        raise ValueError("interval_hours must be positive")

    current = _as_ist(start_from or _now_ist())
    interval_minutes = max(1, int(round(interval_hours * 60)))
    jitter_max_minutes = max(0, min(int(jitter_max_minutes), 59))
    window_start = day_start_hour * 60
    window_end = day_end_hour * 60 - 1  # end hour is exclusive

    next_hour = current.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    first_day = current.date()
    first_minute = window_start
    next_hour_minute = next_hour.hour * 60
    if current.hour >= day_end_hour or next_hour.date() != first_day:
        first_day += timedelta(days=1)
    elif next_hour_minute < window_start:
        first_minute = window_start
    elif next_hour_minute > window_end:
        first_day += timedelta(days=1)
    else:
        first_minute = next_hour_minute

    slots: List[datetime] = []
    remaining = num_slots
    day = first_day
    while remaining:
        available_start = first_minute if day == first_day else window_start
        if available_start > window_end:
            day += timedelta(days=1)
            continue

        capacity = ((window_end - available_start) // interval_minutes) + 1
        take = min(remaining, capacity)

        if take == 1:
            # A lone clip gets the strongest general evening slot, when future.
            preferred = 19 * 60
            base_minutes = [min(max(preferred, available_start), window_end)]
        elif spread_across_window:
            span = window_end - available_start
            base_minutes = [
                available_start + round(span * index / (take - 1))
                for index in range(take)
            ]
        else:
            base_minutes = [
                available_start + index * interval_minutes for index in range(take)
            ]

        if len(base_minutes) > 1:
            smallest_gap = min(
                later - earlier
                for earlier, later in zip(base_minutes, base_minutes[1:])
            )
            safe_jitter = min(
                jitter_max_minutes,
                max(0, smallest_gap - interval_minutes),
            )
        else:
            safe_jitter = min(jitter_max_minutes, window_end - base_minutes[0])

        for index, minute in enumerate(base_minutes):
            base = _at_minute(day, minute)
            # Keep the evening endpoint fixed inside the window.
            jitter = 0 if index == len(base_minutes) - 1 else _jitter_minutes(
                base, safe_jitter
            )
            slots.append(base + timedelta(minutes=jitter))

        remaining -= take
        day += timedelta(days=1)

    return slots


def assign_clips_to_slots(
    clips: List[str],
    interval_hours: int = 1,
    clip_scores: Optional[Dict[str, float]] = None,
    schedule_config: Optional[Dict] = None,
) -> List[Tuple[str, datetime]]:
    """Map clips → jittered slots, putting the best clip(s) in prime time.

    Args:
        clips: List of clip paths or identifiers (first = highest priority if no scores).
        interval_hours: Hours between consecutive slots.
        clip_scores: Optional dict of clip_id → quality score (higher = better).

    Returns:
        List of (clip_identifier, scheduled_datetime) sorted chronologically.
    """
    schedule_config = schedule_config or {}
    slots = generate_schedule(
        len(clips),
        interval_hours,
        day_start_hour=int(schedule_config.get("day_start_hour", DAY_START_HOUR)),
        day_end_hour=int(schedule_config.get("day_end_hour", DAY_END_HOUR)),
        jitter_max_minutes=int(schedule_config.get("jitter_max_minutes", 55)),
        spread_across_window=bool(
            schedule_config.get("spread_across_window", True)
        ),
    )

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

    # Clamp to day window
    next_slot = _clamp_to_day_window(next_slot)

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
