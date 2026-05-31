"""memory.py — SQLite-backed persistent memory store.

Stores key-value pairs with metadata (timestamp, confidence, source, TTL).
Thread-safe via SQLite's default isolation. Survives restarts.
"""

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional


_MEMORY_DB = os.environ.get("SELF_LEARNER_DB", "self_learner.db")
_MEMORY_TABLE = """
CREATE TABLE IF NOT EXISTS memories (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    timestamp REAL NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    source TEXT DEFAULT 'manual',
    ttl REAL DEFAULT NULL
)
"""


class PersistentMemory:
    """SQLite-backed KV store with metadata.

    Args:
        db_path: Path to the SQLite database file. Defaults to
                 ``self_learner.db`` in the current directory.

    Thread-safe: each operation acquires a per-instance reentrant lock.
    """

    def __init__(self, db_path: Optional[str] = None):
        self._lock = threading.RLock()
        self._db_path = db_path or _MEMORY_DB
        self._conn: Optional[sqlite3.Connection] = None
        self._connect()
        self._ensure_table()

    # -- lifecycle -----------------------------------------------------------

    def _connect(self):
        if self._conn is not None:
            return
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def close(self):
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- internal helpers ----------------------------------------------------

    def _ensure_table(self):
        with self._lock:
            self._conn.execute(_MEMORY_TABLE)
            self._conn.commit()

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        return {
            "key": row["key"],
            "value": json.loads(row["value"]),
            "timestamp": row["timestamp"],
            "confidence": row["confidence"],
            "source": row["source"],
            "ttl": row["ttl"],
        }

    # -- public API ----------------------------------------------------------

    def remember(self, key: str, value: Any, *,
                 confidence: float = 1.0,
                 source: str = "manual",
                 ttl: Optional[float] = None) -> None:
        """Store a value under *key* with metadata.

        Args:
            key: Unique identifier for the memory.
            value: Any JSON-serializable value.
            confidence: Confidence score (0-1) for this memory.
            source: Origin of the memory (e.g., ``"observer"``).
            ttl: Time-to-live in seconds. ``None`` means forever.
        """
        serialized = json.dumps(value, default=str)
        now = time.time()
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO memories
                   (key, value, timestamp, confidence, source, ttl)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (key, serialized, now, confidence, source, ttl),
            )
            self._conn.commit()

    def recall(self, key: str) -> Optional[dict]:
        """Retrieve a memory by *key*.

        Returns:
            A dict with keys ``value``, ``timestamp``, ``confidence``,
            ``source``, ``ttl``, or ``None`` if missing or expired.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        mem = self._row_to_dict(row)
        if mem["ttl"] is not None and time.time() > mem["timestamp"] + mem["ttl"]:
            self.forget(key)
            return None
        return mem

    def recall_value(self, key: str) -> Any:
        """Convenience: return just the value for *key*, or ``None``."""
        mem = self.recall(key)
        return mem["value"] if mem is not None else None

    def forget(self, key: str) -> bool:
        """Remove a memory by *key*.

        Returns:
            ``True`` if a row was deleted.
        """
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM memories WHERE key = ?", (key,)
            )
            self._conn.commit()
            return cur.rowcount > 0

    def recall_all(self, *, source: Optional[str] = None) -> list[dict]:
        """Return all non-expired memories, optionally filtered by *source*.

        Returns:
            List of dicts with keys ``key``, ``value``, ``timestamp``,
            ``confidence``, ``source``, ``ttl``.
        """
        with self._lock:
            if source:
                rows = self._conn.execute(
                    "SELECT * FROM memories WHERE source = ?", (source,)
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM memories"
                ).fetchall()
        now = time.time()
        results = []
        for row in rows:
            mem = self._row_to_dict(row)
            if mem["ttl"] is not None and now > mem["timestamp"] + mem["ttl"]:
                self.forget(mem["key"])
                continue
            results.append(mem)
        return results

    def clear(self) -> None:
        """Remove all memories."""
        with self._lock:
            self._conn.execute("DELETE FROM memories")
            self._conn.commit()

    def size(self) -> int:
        """Return the number of non-expired memories."""
        return len(self.recall_all())

    def prune(self) -> int:
        """Remove all expired memories.

        Returns:
            Number of rows pruned.
        """
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM memories WHERE ttl IS NOT NULL "
                "AND (? - timestamp) > ttl",
                (now,),
            )
            self._conn.commit()
            return cur.rowcount
