"""ClipLearner — Phase 4: persists clip selection data for pattern learning.

After each upload, stores:
    hook_score, agent_scores, selected_rank, views, retention, completion_rate

This builds a dataset for identifying what makes winning clips on this channel.
"""

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.logger import get_logger

log = get_logger("clip_learner")

DB_PATH = Path("clip_learner.db")
_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS clip_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    clip_id TEXT NOT NULL,
    video_title TEXT,
    match_context TEXT,
    
    -- Selection data
    selected_rank INTEGER,
    final_score REAL,
    hook_score REAL,
    agent_scores_json TEXT,
    rejection_reasons TEXT,
    
    -- YouTube performance (populated after upload)
    youtube_video_id TEXT,
    views INTEGER DEFAULT 0,
    estimated_retention REAL DEFAULT 0.0,
    completion_rate REAL DEFAULT 0.0,
    avg_view_duration_seconds REAL DEFAULT 0.0,
    
    -- Metadata
    uploaded_at TEXT,
    last_updated TEXT,
    
    UNIQUE(clip_id)
);
"""

_CREATE_HOOK_ANALYSIS_TABLE = """
CREATE TABLE IF NOT EXISTS hook_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    clip_id TEXT NOT NULL,
    hook_score REAL,
    hook_type TEXT,
    swipe_risk TEXT,
    swipe_risks_json TEXT,
    first_20_words TEXT,
    agent_hook_score REAL,
    created_at TEXT,
    UNIQUE(clip_id)
);
"""


class ClipLearner:
    """Persists clip selection + hook analysis + performance data."""

    def __init__(self, db_path: str | Path = DB_PATH):
        self._db_path = Path(db_path)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.execute(_CREATE_TABLE)
            self._conn.execute(_CREATE_HOOK_ANALYSIS_TABLE)
            self._conn.commit()
        return self._conn

    def save_clip_selection(
        self,
        clip_id: str,
        selected_rank: int,
        final_score: float,
        agent_scores: dict[str, Any],
        rejection_reasons: list[str] | None = None,
        video_title: str = "",
        match_context: str = "",
    ) -> None:
        """Save clip selection data after agent scoring."""
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn.execute(
                """INSERT OR REPLACE INTO clip_performance
                   (clip_id, video_title, match_context, selected_rank,
                    final_score, agent_scores_json, rejection_reasons, uploaded_at, last_updated)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    clip_id, video_title, match_context, selected_rank,
                    final_score, json.dumps(agent_scores, default=str),
                    json.dumps(rejection_reasons or []), now, now,
                ),
            )
            conn.commit()

    def save_hook_analysis(
        self,
        clip_id: str,
        hook_score: float,
        hook_type: str,
        swipe_risk: str,
        swipe_risks: list[str],
        first_20_words: str,
        agent_hook_score: float | None = None,
    ) -> None:
        """Save hook analysis results."""
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn.execute(
                """INSERT OR REPLACE INTO hook_analysis
                   (clip_id, hook_score, hook_type, swipe_risk,
                    swipe_risks_json, first_20_words, agent_hook_score, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    clip_id, hook_score, hook_type, swipe_risk,
                    json.dumps(swipe_risks), first_20_words, agent_hook_score, now,
                ),
            )
            conn.commit()

    def update_performance(
        self,
        clip_id: str,
        youtube_video_id: str,
        views: int,
        estimated_retention: float = 0.0,
        completion_rate: float = 0.0,
        avg_view_duration_seconds: float = 0.0,
    ) -> None:
        """Update YouTube performance data after upload."""
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn.execute(
                """UPDATE clip_performance SET
                   youtube_video_id = ?,
                   views = ?,
                   estimated_retention = ?,
                   completion_rate = ?,
                   avg_view_duration_seconds = ?,
                   last_updated = ?
                   WHERE clip_id = ?""",
                (youtube_video_id, views, estimated_retention,
                 completion_rate, avg_view_duration_seconds, now, clip_id),
            )
            conn.commit()

    def _columns_of(self, cursor) -> list[str]:
        return [desc[0] for desc in cursor.description]

    def get_all_clips(self) -> list[dict]:
        """Get all clip performance records for pattern analysis."""
        conn = self._get_conn()
        with self._lock:
            cur = conn.execute("SELECT * FROM clip_performance ORDER BY views DESC")
            rows = cur.fetchall()
            columns = self._columns_of(cur)
            return [dict(zip(columns, row)) for row in rows]

    def get_top_performers(self, min_views: int = 100, limit: int = 20) -> list[dict]:
        """Get top-performing clips (views) for pattern discovery."""
        conn = self._get_conn()
        with self._lock:
            cur = conn.execute(
                "SELECT * FROM clip_performance WHERE views >= ? ORDER BY views DESC LIMIT ?",
                (min_views, limit),
            )
            rows = cur.fetchall()
            columns = self._columns_of(cur)
            return [dict(zip(columns, row)) for row in rows]

    def get_performance_summary(self) -> dict[str, Any]:
        """Get summary stats for analysis dashboard."""
        conn = self._get_conn()
        with self._lock:
            total = conn.execute("SELECT COUNT(*) FROM clip_performance").fetchone()[0]
            total_with_perf = conn.execute(
                "SELECT COUNT(*) FROM clip_performance WHERE views > 0"
            ).fetchone()[0]
            avg_views = conn.execute(
                "SELECT AVG(views) FROM clip_performance WHERE views > 0"
            ).fetchone()[0] or 0
            top_agent = conn.execute(
                """SELECT agent_scores_json FROM clip_performance
                   WHERE views > 0 ORDER BY views DESC LIMIT 1"""
            ).fetchone()
            return {
                "total_clips": total,
                "clips_with_performance": total_with_perf,
                "avg_views": round(avg_views, 0),
                "top_clip_agent_scores": top_agent[0] if top_agent else None,
            }

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
