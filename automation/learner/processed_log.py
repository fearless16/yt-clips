"""ProcessedEventLog — idempotency guard for event processing.

Tracks which events have been processed by which learners.
Prevents double-processing and replay inflation.
"""

import sqlite3
import threading
from datetime import datetime, timezone


class ProcessedEventLog:
    """Tracks which events have been processed by which learners."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._lock = threading.RLock()
        self._ensure_table()

    def _ensure_table(self):
        """Create table if it doesn't exist."""
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_events (
                    event_id    TEXT NOT NULL,
                    learner     TEXT NOT NULL,
                    processed_at TEXT NOT NULL,
                    PRIMARY KEY (event_id, learner)
                )
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_proc_learner
                ON processed_events(learner)
            """)
            self._conn.commit()

    def is_processed(self, event_id: str, learner: str) -> bool:
        """Check if an event has been processed by a specific learner.

        Args:
            event_id: The event ID to check
            learner: The learner name (e.g., "format", "entity", "timing")

        Returns:
            True if already processed, False otherwise
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM processed_events WHERE event_id=? AND learner=?",
                (event_id, learner)
            ).fetchone()
            return row is not None

    def mark_processed(self, event_id: str, learner: str) -> None:
        """Mark an event as processed by a specific learner.

        Args:
            event_id: The event ID to mark
            learner: The learner name
        """
        with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            self._conn.execute(
                """INSERT OR IGNORE INTO processed_events
                   (event_id, learner, processed_at)
                   VALUES (?, ?, ?)""",
                (event_id, learner, now)
            )
            self._conn.commit()

    def clear_learner(self, learner: str) -> int:
        """Clear all processed events for a specific learner.

        Used for replay scenarios where you want to reprocess events.

        Args:
            learner: The learner name to clear

        Returns:
            Number of records cleared
        """
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM processed_events WHERE learner=?",
                (learner,)
            )
            self._conn.commit()
            return cursor.rowcount

    def clear_all(self) -> int:
        """Clear all processed events.

        Returns:
            Number of records cleared
        """
        with self._lock:
            cursor = self._conn.execute("DELETE FROM processed_events")
            self._conn.commit()
            return cursor.rowcount

    def count(self, learner: str | None = None) -> int:
        """Count processed events, optionally filtered by learner.

        Args:
            learner: Optional learner name to filter by

        Returns:
            Count of processed events
        """
        with self._lock:
            if learner is None:
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM processed_events"
                ).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM processed_events WHERE learner=?",
                    (learner,)
                ).fetchone()
            return row[0]

    def get_learners(self) -> list[str]:
        """Get list of all learners that have processed events.

        Returns:
            List of learner names
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT learner FROM processed_events"
            ).fetchall()
            return [row[0] for row in rows]
