"""Upload scheduling utilities — day gating and priority day routing.

Uses learner data to avoid dead days (e.g. Thursday) and prefer
high-performing days (Friday, Sunday, Wednesday).
"""
from datetime import datetime, timedelta
from typing import Optional

from utils.config import load_config
from utils.logger import get_logger

_cfg = load_config()
log = get_logger("scheduling", _cfg.get("logging", {}).get("log_file", "logs/pipeline.log"))


def is_dead_day(day_name: str, cfg: Optional[dict] = None) -> bool:
    """Check if a day of the week is marked as a dead day.

    Args:
        day_name: Lowercase day name (e.g. "thursday")
        cfg: Config dict. Uses global config if None.

    Returns:
        True if uploads should be avoided on this day.
    """
    if cfg is None:
        cfg = _cfg
    avoid = cfg.get("upload_schedule", {}).get("avoid_days", [])
    return day_name.lower() in [d.lower() for d in avoid]


def next_upload_day(from_dt: datetime, cfg: Optional[dict] = None) -> datetime:
    """Find the next suitable upload day, skipping dead days.

    If from_dt is on a dead day, advances to the next day that is
    either a priority day or at least not a dead day.

    Args:
        from_dt: The starting datetime
        cfg: Config dict. Uses global config if None.

    Returns:
        A datetime on an acceptable upload day (same time, shifted date).
    """
    if cfg is None:
        cfg = _cfg

    avoid = [d.lower() for d in cfg.get("upload_schedule", {}).get("avoid_days", [])]
    if not avoid:
        return from_dt

    priority = [d.lower() for d in cfg.get("upload_schedule", {}).get("priority_days", [])]

    current = from_dt
    for _ in range(7):  # Max 7 days lookahead
        day_name = current.strftime("%A").lower()
        if day_name not in avoid:
            return current
        # Advance to next day
        current = current + timedelta(days=1)
        # If next day is a priority day AND not dead, prefer it
        next_day = current.strftime("%A").lower()
        if next_day in priority and next_day not in avoid:
            return current

    # Fallback: return tomorrow
    return from_dt + timedelta(days=1)
