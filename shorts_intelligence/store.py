"""Transactional canonical store for Shorts intelligence."""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shorts_intelligence.domain import CricketDomainGate
from shorts_intelligence.models import PerformanceSnapshot, ShortRecord


SCHEMA_VERSION = 4


_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS shorts (
    video_id TEXT PRIMARY KEY,
    channel_id TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    published_at TEXT NOT NULL,
    duration_seconds INTEGER NOT NULL CHECK(duration_seconds >= 0),
    is_short INTEGER NOT NULL CHECK(is_short = 1),
    domain_status TEXT NOT NULL CHECK(domain_status IN ('cricket','non_cricket','unknown')),
    domain_reason TEXT NOT NULL,
    category_id TEXT NOT NULL,
    tags_json TEXT NOT NULL,
    source_url TEXT NOT NULL,
    local_metadata_json TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_shorts_domain ON shorts(domain_status, published_at);

CREATE TABLE IF NOT EXISTS performance_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL REFERENCES shorts(video_id) ON DELETE CASCADE,
    captured_at TEXT NOT NULL,
    engaged_views INTEGER NOT NULL,
    views INTEGER NOT NULL,
    estimated_minutes_watched REAL NOT NULL,
    average_view_duration_seconds REAL NOT NULL,
    average_view_percentage REAL NOT NULL,
    likes INTEGER NOT NULL,
    comments INTEGER NOT NULL,
    shares INTEGER NOT NULL,
    subscribers_gained INTEGER NOT NULL,
    subscribers_lost INTEGER NOT NULL,
    source TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    UNIQUE(video_id, captured_at)
);

CREATE INDEX IF NOT EXISTS idx_snapshot_video_time
ON performance_snapshots(video_id, captured_at DESC);

CREATE TABLE IF NOT EXISTS production_features (
    clip_id TEXT PRIMARY KEY,
    video_id TEXT REFERENCES shorts(video_id) ON DELETE SET NULL,
    recorded_at TEXT NOT NULL,
    transcript TEXT NOT NULL,
    features_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pending_uploads (
    clip_id TEXT PRIMARY KEY REFERENCES production_features(clip_id) ON DELETE CASCADE,
    video_id TEXT NOT NULL UNIQUE,
    uploaded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_segments (
    model_version TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    feature_value TEXT NOT NULL,
    statistics_json TEXT NOT NULL,
    fitted_at TEXT NOT NULL,
    PRIMARY KEY(model_version, feature_name, feature_value)
);

CREATE TABLE IF NOT EXISTS models (
    model_version TEXT PRIMARY KEY,
    fitted_at TEXT NOT NULL,
    observations INTEGER NOT NULL,
    baseline_mean REAL NOT NULL,
    baseline_variance REAL NOT NULL,
    config_json TEXT NOT NULL,
    active INTEGER NOT NULL CHECK(active IN (0,1))
);

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    dedupe_key TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    counters_json TEXT NOT NULL,
    error TEXT
);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


class ShortsStore:
    """One durable, thread-safe source of truth for catalog and outcomes."""

    def __init__(self, db_path: str | Path, *, channel_id: str) -> None:
        self.path = Path(db_path)
        self.channel_id = channel_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA journal_mode=WAL")
        with self._conn:
            self._conn.executescript(_SCHEMA)
            columns = {
                row[1] for row in self._conn.execute("PRAGMA table_info(shorts)")
            }
            if "tags_json" not in columns:
                self._conn.execute(
                    "ALTER TABLE shorts ADD COLUMN tags_json TEXT NOT NULL DEFAULT '[]'"
                )
            self._conn.execute(
                "INSERT OR IGNORE INTO schema_version(version, applied_at) VALUES (?, ?)",
                (SCHEMA_VERSION, _utcnow()),
            )

    @property
    def schema_version(self) -> int:
        """Return the newest applied schema version."""
        row = self._conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        return int(row[0] or 0)

    def journal_mode(self) -> str:
        """Return SQLite's active journal mode."""
        return str(self._conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()

    def upsert_short(
        self,
        record: ShortRecord,
        *,
        domain_status: str | None = None,
        domain_reason: str = "",
    ) -> None:
        """Insert or refresh one shelf-proven Short without losing history."""
        params = self._prepare_short(record, domain_status, domain_reason)
        with self._lock, self._conn:
            self._execute_short(params)

    def add_snapshot(self, snapshot: PerformanceSnapshot) -> bool:
        """Append one immutable metric snapshot; return false for a duplicate."""
        with self._lock, self._conn:
            return self._insert_snapshot(snapshot)

    def add_snapshots_atomic(self, snapshots: list[PerformanceSnapshot]) -> dict[str, int]:
        """Append snapshots in one transaction; all video IDs must exist."""
        added = 0
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                for snapshot in snapshots:
                    exists = self._conn.execute(
                        "SELECT 1 FROM shorts WHERE video_id=?", (snapshot.video_id,)
                    ).fetchone()
                    if exists is None:
                        raise ValueError(f"snapshot video is not in catalog: {snapshot.video_id}")
                    added += int(self._insert_snapshot(snapshot))
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return {"added": added, "duplicates": len(snapshots) - added}

    def catalog_ids(self) -> set[str]:
        """Return exact shelf-proven video IDs."""
        return {str(row[0]) for row in self._conn.execute("SELECT video_id FROM shorts")}

    def ingest_batch(
        self,
        records: list[ShortRecord],
        snapshots: list[PerformanceSnapshot],
    ) -> dict[str, int]:
        """Atomically ingest one catalog/analytics pull."""
        prepared = [self._prepare_short(record, None, "") for record in records]
        record_ids = {record.video_id for record in records}
        unknown = [snapshot.video_id for snapshot in snapshots if snapshot.video_id not in record_ids]
        if unknown:
            raise ValueError(f"snapshot has no batch catalog record: {unknown[0]}")

        added = 0
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                for params in prepared:
                    self._execute_short(params)
                for snapshot in snapshots:
                    added += int(self._insert_snapshot(snapshot))
                for record in records:
                    self._conn.execute(
                        """UPDATE production_features SET video_id=?
                           WHERE clip_id=(
                               SELECT clip_id FROM pending_uploads WHERE video_id=?
                           )""",
                        (record.video_id, record.video_id),
                    )
                    self._conn.execute(
                        "DELETE FROM pending_uploads WHERE video_id=?",
                        (record.video_id,),
                    )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return {"records": len(records), "snapshots_added": added}

    def record_production(
        self,
        *,
        clip_id: str,
        transcript: str,
        features: dict[str, Any],
        recorded_at: str | None = None,
    ) -> None:
        """Upsert pre-publication features without claiming an outcome."""
        if not clip_id:
            raise ValueError("clip_id is required")
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO production_features (
                       clip_id, video_id, recorded_at, transcript, features_json
                   ) VALUES (?,NULL,?,?,?)
                   ON CONFLICT(clip_id) DO UPDATE SET
                       recorded_at=excluded.recorded_at,
                       transcript=excluded.transcript,
                       features_json=excluded.features_json""",
                (clip_id, recorded_at or _utcnow(), transcript, _dumps(features)),
            )

    def record_upload(self, clip_id: str, video_id: str) -> None:
        """Keep an upload link pending until it appears on the Shorts shelf."""
        if not clip_id or not video_id:
            raise ValueError("clip_id and video_id are required")
        with self._lock, self._conn:
            exists = self._conn.execute(
                "SELECT 1 FROM production_features WHERE clip_id=?", (clip_id,)
            ).fetchone()
            if exists is None:
                raise ValueError("production record must exist before upload")
            catalog = self._conn.execute(
                "SELECT 1 FROM shorts WHERE video_id=?", (video_id,)
            ).fetchone()
            if catalog:
                self._conn.execute(
                    "UPDATE production_features SET video_id=? WHERE clip_id=?",
                    (video_id, clip_id),
                )
            else:
                self._conn.execute(
                    """INSERT INTO pending_uploads(clip_id, video_id, uploaded_at)
                       VALUES (?,?,?)
                       ON CONFLICT(clip_id) DO UPDATE SET
                           video_id=excluded.video_id,
                           uploaded_at=excluded.uploaded_at""",
                    (clip_id, video_id, _utcnow()),
                )

    def get_production(self, clip_id: str) -> dict[str, Any] | None:
        """Return one production record."""
        row = self._conn.execute(
            """SELECT clip_id, video_id, recorded_at, transcript, features_json
               FROM production_features WHERE clip_id=?""",
            (clip_id,),
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["features"] = json.loads(result.pop("features_json"))
        return result

    def _prepare_short(
        self,
        record: ShortRecord,
        domain_status: str | None,
        domain_reason: str,
    ) -> tuple[Any, ...]:
        if not record.is_short:
            raise ValueError("record must come from the Shorts shelf")
        if record.channel_id != self.channel_id:
            raise ValueError("record belongs to a different channel")
        if not record.video_id:
            raise ValueError("video_id is required")
        if domain_status is None:
            decision = CricketDomainGate().classify(
                record.title,
                record.description,
                trusted_cricket_origin=bool(record.local_metadata),
            )
            domain_status = decision.status
            domain_reason = ",".join(decision.reasons)
        if domain_status not in {"cricket", "non_cricket", "unknown"}:
            raise ValueError("invalid domain status")
        now = _utcnow()
        return (
            record.video_id, record.channel_id, record.title, record.description,
            record.published_at, record.duration_seconds, 1, domain_status,
            domain_reason, record.category_id, _dumps(record.tags), record.source_url,
            _dumps(record.local_metadata), _dumps(record.raw), now, now,
        )

    def _execute_short(self, params: tuple[Any, ...]) -> sqlite3.Cursor:
        return self._conn.execute(
            """INSERT INTO shorts (
                   video_id, channel_id, title, description, published_at,
                   duration_seconds, is_short, domain_status, domain_reason,
                   category_id, tags_json, source_url, local_metadata_json, raw_json,
                   first_seen_at, last_seen_at
               ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(video_id) DO UPDATE SET
                   title=excluded.title, description=excluded.description,
                   published_at=excluded.published_at,
                   duration_seconds=excluded.duration_seconds,
                   domain_status=excluded.domain_status,
                   domain_reason=excluded.domain_reason,
                   category_id=excluded.category_id,
                   tags_json=excluded.tags_json,
                   source_url=excluded.source_url,
                   local_metadata_json=excluded.local_metadata_json,
                   raw_json=excluded.raw_json,
                   last_seen_at=excluded.last_seen_at""",
            params,
        )

    def _insert_snapshot(self, snapshot: PerformanceSnapshot) -> bool:
        signature_columns = (
            "engaged_views", "views", "estimated_minutes_watched",
            "average_view_duration_seconds", "average_view_percentage",
            "likes", "comments", "shares", "subscribers_gained",
            "subscribers_lost",
        )
        latest = self._conn.execute(
            f"""SELECT {','.join(signature_columns)}
                FROM performance_snapshots
                WHERE video_id=? AND source=?
                ORDER BY captured_at DESC, id DESC LIMIT 1""",
            (snapshot.video_id, snapshot.source),
        ).fetchone()
        signature = (
            snapshot.engaged_views, snapshot.views,
            snapshot.estimated_minutes_watched,
            snapshot.average_view_duration_seconds,
            snapshot.average_view_percentage, snapshot.likes,
            snapshot.comments, snapshot.shares,
            snapshot.subscribers_gained, snapshot.subscribers_lost,
        )
        if latest is not None and tuple(latest) == signature:
            return False
        cursor = self._conn.execute(
            """INSERT INTO performance_snapshots (
                   video_id, captured_at, engaged_views, views,
                   estimated_minutes_watched, average_view_duration_seconds,
                   average_view_percentage, likes, comments, shares,
                   subscribers_gained, subscribers_lost, source, raw_json
               ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(video_id, captured_at) DO NOTHING""",
            (
                snapshot.video_id, snapshot.captured_at, snapshot.engaged_views,
                snapshot.views, snapshot.estimated_minutes_watched,
                snapshot.average_view_duration_seconds,
                snapshot.average_view_percentage, snapshot.likes,
                snapshot.comments, snapshot.shares,
                snapshot.subscribers_gained, snapshot.subscribers_lost,
                snapshot.source, _dumps(snapshot.raw),
            ),
        )
        return cursor.rowcount == 1

    def summary(self) -> dict[str, int]:
        """Return compact row and quality counts."""
        row = self._conn.execute(
            """SELECT
                   COUNT(*) AS shorts,
                   SUM(domain_status='cricket') AS cricket,
                   SUM(domain_status='non_cricket') AS non_cricket,
                   SUM(domain_status='unknown') AS unknown
               FROM shorts"""
        ).fetchone()
        snapshots = self._conn.execute("SELECT COUNT(*) FROM performance_snapshots").fetchone()[0]
        return {
            "shorts": int(row["shorts"] or 0),
            "cricket": int(row["cricket"] or 0),
            "non_cricket": int(row["non_cricket"] or 0),
            "unknown": int(row["unknown"] or 0),
            "snapshots": int(snapshots),
        }

    def training_rows(self) -> list[dict[str, Any]]:
        """Return latest cricket-only outcomes with deterministic features."""
        rows = self._conn.execute(
            """SELECT s.*, p.*, pf.features_json AS production_features_json
               FROM shorts s
               JOIN performance_snapshots p ON p.video_id=s.video_id
               LEFT JOIN production_features pf ON pf.video_id=s.video_id
               WHERE s.domain_status='cricket'
                 AND p.id=(
                     SELECT p2.id FROM performance_snapshots p2
                     WHERE p2.video_id=s.video_id
                     ORDER BY p2.captured_at DESC, p2.id DESC LIMIT 1
                 )
               ORDER BY s.video_id"""
        ).fetchall()
        return [self._training_row(dict(row)) for row in rows]

    def save_model(self, model: Any, config: Any) -> str:
        """Persist a fitted model atomically and mark it active."""
        version = model.fitted_at
        config_dict = {
            name: getattr(config, name)
            for name in ("prior_strength", "min_segment_samples", "half_life_days", "confidence_z")
        }
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                self._conn.execute("UPDATE models SET active=0")
                self._conn.execute(
                    """INSERT OR REPLACE INTO models (
                           model_version, fitted_at, observations, baseline_mean,
                           baseline_variance, config_json, active
                       ) VALUES (?,?,?,?,?,?,1)""",
                    (
                        version, model.fitted_at, model.observations,
                        model.baseline_mean, model.baseline_variance,
                        _dumps(config_dict),
                    ),
                )
                self._conn.execute("DELETE FROM model_segments WHERE model_version=?", (version,))
                for (name, value), stats in model.segments.items():
                    self._conn.execute(
                        """INSERT INTO model_segments (
                               model_version, feature_name, feature_value,
                               statistics_json, fitted_at
                           ) VALUES (?,?,?,?,?)""",
                        (version, name, value, _dumps(stats.to_dict()), model.fitted_at),
                    )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return version

    def latest_model(self) -> dict[str, Any] | None:
        """Return compact metadata for the active fitted model."""
        row = self._conn.execute(
            """SELECT model_version, fitted_at, observations, baseline_mean,
                      baseline_variance, config_json
               FROM models WHERE active=1 LIMIT 1"""
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["config"] = json.loads(result.pop("config_json"))
        return result

    def active_recommendations(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return only statistically supported positive active segments."""
        rows = self._conn.execute(
            """SELECT ms.statistics_json
               FROM model_segments ms
               JOIN models m ON m.model_version=ms.model_version
               WHERE m.active=1"""
        ).fetchall()
        supported = []
        for row in rows:
            stats = json.loads(row[0])
            if stats.get("actionable") and float(stats.get("effect_lower_bound", 0)) > 0:
                supported.append(stats)
        return sorted(
            supported,
            key=lambda item: (
                float(item.get("effect_lower_bound", 0)),
                float(item.get("posterior_mean", 0)),
            ),
            reverse=True,
        )[:limit]

    def production_count(self) -> int:
        """Return captured pre-publication records."""
        return int(self._conn.execute("SELECT COUNT(*) FROM production_features").fetchone()[0])

    def _training_row(self, row: dict[str, Any]) -> dict[str, Any]:
        views = max(0, int(row["views"] or 0))
        engaged = max(0, int(row["engaged_views"] or 0))
        continued = min(1.0, engaged / views) if views else 0.0
        retention = min(2.0, float(row["average_view_percentage"] or 0) / 100.0) / 2.0
        interactions = (
            int(row["likes"] or 0)
            + 2 * int(row["comments"] or 0)
            + 3 * int(row["shares"] or 0)
            + 4 * max(0, int(row["subscribers_gained"] or 0) - int(row["subscribers_lost"] or 0))
        )
        interaction_rate = min(1.0, interactions / max(1, engaged) / 0.2)
        reach = min(1.0, math.log1p(engaged) / math.log1p(10_000))
        outcome = 0.4 * continued + 0.3 * retention + 0.2 * interaction_rate + 0.1 * reach
        duration = int(row["duration_seconds"] or 0)
        published = _parse_datetime(str(row["published_at"]))
        local = json.loads(row["local_metadata_json"] or "{}")
        features: dict[str, str] = {
            "duration_bucket": _duration_bucket(duration),
            "publish_day": published.strftime("%A").lower(),
            "publish_hour_block": f"{(published.hour // 4) * 4:02d}-{(published.hour // 4) * 4 + 3:02d}",
            "title_question": str("?" in row["title"]).lower(),
            "title_number": str(bool(any(char.isdigit() for char in row["title"]))).lower(),
            "title_length_bucket": _length_bucket(len(row["title"]), (40, 70, 100)),
            "description_length_bucket": _length_bucket(
                len(row["description"]), (500, 1500, 3000)
            ),
        }
        tags = json.loads(row.get("tags_json") or "[]")
        features["tag_count_bucket"] = _count_bucket(len(tags), (5, 15, 30))
        hashtag_count = row["title"].count("#") + row["description"].count("#")
        features["hashtag_count_bucket"] = _count_bucket(hashtag_count, (3, 8, 15))
        for key in ("hook_type", "format", "topic_cluster"):
            value = local.get(key)
            if value:
                features[key] = str(value).casefold()
        production = json.loads(row.get("production_features_json") or "{}")
        for key, value in production.items():
            if isinstance(value, (str, int, float, bool)):
                features[str(key)] = str(value).casefold()
        return {
            "video_id": row["video_id"],
            "title": row["title"],
            "duration_seconds": duration,
            "captured_at": row["captured_at"],
            "evidence_at": row["published_at"],
            "outcome_score": max(0.0, min(1.0, outcome)),
            "features": features,
            "views": views,
            "engaged_views": engaged,
            "average_view_percentage": float(row["average_view_percentage"] or 0),
        }

    def close(self) -> None:
        """Close the SQLite connection."""
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "ShortsStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _duration_bucket(seconds: int) -> str:
    if seconds < 15:
        return "under_15"
    if seconds < 25:
        return "15_24"
    if seconds < 40:
        return "25_39"
    if seconds < 60:
        return "40_59"
    return "60_plus"


def _length_bucket(value: int, limits: tuple[int, int, int]) -> str:
    return _count_bucket(value, limits)


def _count_bucket(value: int, limits: tuple[int, int, int]) -> str:
    low, medium, high = limits
    if value < low:
        return f"under_{low}"
    if value < medium:
        return f"{low}_{medium - 1}"
    if value < high:
        return f"{medium}_{high - 1}"
    return f"{high}_plus"
