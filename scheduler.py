"""
scheduler.py — Version 4: Smart Scheduling Manager.
Calculates the next available 2-hour slot for YouTube uploads.
"""
import os
import json
from datetime import datetime, timedelta, timezone

STATE_FILE = "scheduler_state.json"

def get_next_slot(interval_hours: int = 2):
    """
    Determines the next available time slot for a scheduled upload.
    """
    now = datetime.now(timezone.utc)
    
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
            last_scheduled = datetime.fromisoformat(state['last_scheduled'])
    else:
        # If no state, start scheduling from the next hour
        last_scheduled = now.replace(minute=0, second=0, microsecond=0)

    # Calculate next slot
    next_slot = last_scheduled + timedelta(hours=interval_hours)
    
    # If the calculated slot is in the past, move it to the future
    if next_slot < now:
        next_slot = now + timedelta(hours=1)
        next_slot = next_slot.replace(minute=0, second=0, microsecond=0)

    # Save the new state
    with open(STATE_FILE, 'w') as f:
        json.dump({'last_scheduled': next_slot.isoformat()}, f)

    return next_slot

def format_for_youtube(dt: datetime):
    """Format datetime for YouTube API (ISO 8601)."""
    return dt.isoformat()
