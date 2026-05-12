"""
scheduler.py — IST-aware smart scheduler.
Picks optimal peak-hour slots for Indian audience.

Peak hours (IST): 19:00-22:00 (evening prime), 12:00-14:00 (lunch)
"""

import os
import json
from datetime import datetime, timedelta, timezone, date

STATE_FILE = "scheduler_state.json"
IST = timezone(timedelta(hours=5, minutes=30))

PEAK_SLOTS = [
    (19, 22),  # Evening prime: 7 PM - 10 PM IST
    (12, 14),  # Lunch: 12 PM - 2 PM IST
]


def _now_ist() -> datetime:
    return datetime.now(IST)


def get_optimal_slot() -> datetime:
    """
    Return the next available IST peak-hour slot.
    First checks evening (19-22), then lunch (12-14).
    """
    now = _now_ist()

    for start_h, end_h in PEAK_SLOTS:
        slot = now.replace(hour=start_h, minute=0, second=0, microsecond=0)
        if slot < now:
            slot += timedelta(days=1)
        end = slot.replace(hour=end_h)
        if slot <= now + timedelta(hours=2):
            continue
        return slot

    return now + timedelta(hours=2)


def get_next_slot(interval_hours: int = 2):
    """
    Legacy: 2-hour interval scheduling. Falls back to optimal slot.
    """
    now = datetime.now(timezone.utc)

    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            state = json.load(f)
            last_scheduled = datetime.fromisoformat(state["last_scheduled"])
    else:
        last_scheduled = now.replace(minute=0, second=0, microsecond=0)

    next_slot = last_scheduled + timedelta(hours=interval_hours)
    if next_slot < now:
        next_slot = get_optimal_slot()

    with open(STATE_FILE, "w") as f:
        json.dump({"last_scheduled": next_slot.isoformat()}, f)

    return next_slot


def format_for_youtube(dt: datetime) -> str:
    """Format datetime for YouTube API (ISO 8601)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()
