"""PersistentStateStore — SQLite-backed learned state store.

Replaces in-memory LearnedStateStore with persistent storage.
Stores learned state as JSON, supports entity-specific helpers.
"""

import json
import sqlite3
import threading
from datetime import datetime, timezone, timedelta
from typing import Any


class PersistentStateStore:
    """SQLite-backed learned state store. Replaces in-memory LearnedStateStore."""

    def __init__(self, db_path: str = "self_learner.db"):
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._ensure_tables()

    def _ensure_tables(self):
        """Create tables if they don't exist."""
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS learned_state (
                    state_key       TEXT PRIMARY KEY,
                    value_json      TEXT NOT NULL,
                    derived_from    TEXT,
                    updated_at      TEXT NOT NULL,
                    version         INTEGER DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS processed_events (
                    event_id    TEXT NOT NULL,
                    learner     TEXT NOT NULL,
                    processed_at TEXT NOT NULL,
                    PRIMARY KEY (event_id, learner)
                );
                CREATE INDEX IF NOT EXISTS idx_proc_learner ON processed_events(learner);

                CREATE TABLE IF NOT EXISTS trends (
                    trend_id        TEXT PRIMARY KEY,
                    query           TEXT NOT NULL,
                    source          TEXT NOT NULL,
                    category        TEXT NOT NULL,
                    initial_score   REAL NOT NULL,
                    half_life_hours REAL NOT NULL DEFAULT 72.0,
                    created_at      TEXT NOT NULL,
                    expires_at      TEXT,
                    entities_json   TEXT DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_trend_category ON trends(category);
                CREATE INDEX IF NOT EXISTS idx_trend_expires ON trends(expires_at);
            """)
            self._conn.commit()

    def get(self, key: str) -> dict | None:
        """Get learned state by key. Returns dict or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT value_json FROM learned_state WHERE state_key=?",
                (key,)
            ).fetchone()
            if row is None:
                return None
            return json.loads(row[0])

    def set(self, key: str, value: dict, derived_from: str | None = None) -> None:
        """Set learned state. Overwrites existing."""
        with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            self._conn.execute(
                """INSERT INTO learned_state (state_key, value_json, derived_from, updated_at, version)
                   VALUES (?, ?, ?, ?, 1)
                   ON CONFLICT(state_key) DO UPDATE SET
                       value_json = excluded.value_json,
                       derived_from = excluded.derived_from,
                       updated_at = excluded.updated_at,
                       version = version + 1""",
                (key, json.dumps(value), derived_from, now)
            )
            self._conn.commit()

    def get_all(self) -> dict[str, dict]:
        """Get all learned state entries."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT state_key, value_json FROM learned_state"
            ).fetchall()
            return {row[0]: json.loads(row[1]) for row in rows}

    def delete(self, key: str) -> None:
        """Delete learned state by key."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM learned_state WHERE state_key=?",
                (key,)
            )
            self._conn.commit()

    def clear(self) -> None:
        """Clear all learned state."""
        with self._lock:
            self._conn.execute("DELETE FROM learned_state")
            self._conn.commit()

    def get_entity(self, entity_type: str, name: str) -> dict:
        """Get entity data (player, team, series, format).

        Args:
            entity_type: "players", "teams", "series", or "formats"
            name: entity name (e.g., "bumrah", "mi", "ipl")

        Returns:
            Entity dict with score, n, avg_views, trend. Returns defaults if not found.
        """
        state = self.get("entity_scores")
        if state is None:
            return {"score": 0.5, "n": 0, "avg_views": 0, "trend": "stable", "recent_views_avg": 0}

        entities = state.get(entity_type, {})
        return entities.get(name, {
            "score": 0.5,
            "n": 0,
            "avg_views": 0,
            "trend": "stable",
            "recent_views_avg": 0
        })

    def set_entity(self, entity_type: str, name: str, data: dict) -> None:
        """Set entity data (player, team, series, format).

        Args:
            entity_type: "players", "teams", "series", or "formats"
            name: entity name
            data: entity dict with score, n, avg_views, trend
        """
        state = self.get("entity_scores")
        if state is None:
            state = {"players": {}, "teams": {}, "series": {}, "formats": {}}

        if entity_type not in state:
            state[entity_type] = {}

        state[entity_type][name] = data
        self.set("entity_scores", state)

    def get_all_entities(self, entity_type: str) -> dict[str, dict]:
        """Get all entities of a type.

        Args:
            entity_type: "players", "teams", "series", or "formats"

        Returns:
            Dict of entity_name -> entity_data
        """
        state = self.get("entity_scores")
        if state is None:
            return {}
        return state.get(entity_type, {})

    def count_recent(self, entity_type: str, name: str, window_days: int) -> int:
        """Count how many times an entity was used in recent uploads.

        Args:
            entity_type: "players", "teams", etc.
            name: entity name
            window_days: lookback window in days

        Returns:
            Count of recent uploads featuring this entity
        """
        state = self.get("entity_scores")
        if state is None:
            return 0

        entities = state.get(entity_type, {})
        entity = entities.get(name, {})

        # Track recent uploads as a list of timestamps
        recent = entity.get("recent_uploads", [])
        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
        cutoff_ts = cutoff.timestamp()

        return sum(1 for ts in recent if ts > cutoff_ts)

    def add_recent_upload(self, entity_type: str, name: str) -> None:
        """Record that an entity was used in a new upload.

        Args:
            entity_type: "players", "teams", etc.
            name: entity name
        """
        state = self.get("entity_scores")
        if state is None:
            state = {"players": {}, "teams": {}, "series": {}, "formats": {}}

        if entity_type not in state:
            state[entity_type] = {}

        if name not in state[entity_type]:
            state[entity_type][name] = {
                "score": 0.5,
                "n": 0,
                "avg_views": 0,
                "trend": "stable",
                "recent_views_avg": 0
            }

        entity = state[entity_type][name]
        if "recent_uploads" not in entity:
            entity["recent_uploads"] = []

        entity["recent_uploads"].append(datetime.now(timezone.utc).timestamp())
        state[entity_type][name] = entity
        self.set("entity_scores", state)

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
