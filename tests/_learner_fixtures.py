"""Shared test fixtures for the cricket learner CLI tests."""

import json
import sqlite3
import shutil
from pathlib import Path


def seed_with_facts(tmp_dir: str, db_path: str) -> None:
    """Create a self_learner.db-compatible file with sample facts and run migrate.

    Args:
        tmp_dir: A temp directory (used for the seed DB before copy)
        db_path: Final destination path for the test DB
    """
    seed_path = Path(tmp_dir) / "seed.db"
    conn = sqlite3.connect(str(seed_path))
    conn.executescript("""
        CREATE TABLE memories (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            timestamp REAL NOT NULL,
            confidence REAL DEFAULT 1.0,
            source TEXT DEFAULT 'manual',
            ttl REAL DEFAULT NULL
        );
    """)
    facts = [
        ("fact:upload:1", {
            "subject": "upload", "relation": "was_observed",
            "object": {
                "clip_id": "vidAAA", "title": "Bumrah magic spell",
                "duration": 25, "view_count": 1500, "engagement_rate": 3.5,
                "tag_count": 3, "hashtag_count": 1, "upload_date": "20260601",
            },
        }, 100.0),
        ("fact:upload:2", {
            "subject": "upload", "relation": "was_observed",
            "object": {
                "clip_id": "vidBBB", "title": "RCB vs MI Final Match",
                "duration": 35, "view_count": 200, "engagement_rate": 1.0,
                "tag_count": 5, "hashtag_count": 1, "upload_date": "20260602",
            },
        }, 200.0),
        ("fact:upload:3", {
            "subject": "upload", "relation": "was_observed",
            "object": {
                "clip_id": "vidCCC", "title": "Stunned by this catch 😱",
                "duration": 20, "view_count": 800, "engagement_rate": 3.75,
                "tag_count": 4, "hashtag_count": 1, "upload_date": "20260602",
            },
        }, 300.0),
        ("fact:upload:4", {
            "subject": "upload", "relation": "was_observed",
            "object": {
                "clip_id": "vidDDD", "title": "Long cricket match discussion analysis?",
                "duration": 55, "view_count": 400, "engagement_rate": 2.0,
                "tag_count": 6, "hashtag_count": 1, "upload_date": "20260603",
            },
        }, 400.0),
        ("fact:upload:5", {
            "subject": "upload", "relation": "was_observed",
            "object": {
                "clip_id": "vidEEE", "title": "Question of the day for cricket fans",
                "duration": 30, "view_count": 100, "engagement_rate": 0.5,
                "tag_count": 4, "hashtag_count": 1, "upload_date": "20260603",
            },
        }, 500.0),
        ("fact:seo_outcome:1", {
            "subject": "seo_outcome", "relation": "has_performance",
            "object": {
                "clip_id": "vidAAA", "title": "Bumrah magic spell",
                "description": "Bumrah bowled a great spell for MI",
                "tags": ["bumrah", "mi", "ipl"],
                "is_shorts": True,
            },
        }, 110.0),
        ("fact:seo_outcome:2", {
            "subject": "seo_outcome", "relation": "has_performance",
            "object": {
                "clip_id": "vidBBB", "title": "RCB vs MI Final Match",
                "description": "MI vs RCB rivalry",
                "tags": ["rcb", "mi", "ipl"],
                "is_shorts": True,
            },
        }, 210.0),
        ("fact:seo_outcome:3", {
            "subject": "seo_outcome", "relation": "has_performance",
            "object": {
                "clip_id": "vidCCC", "title": "Stunned by this catch",
                "description": "Amazing catch by kohli",
                "tags": ["kohli", "rcb"],
                "is_shorts": True,
            },
        }, 310.0),
        ("fact:seo_outcome:4", {
            "subject": "seo_outcome", "relation": "has_performance",
            "object": {
                "clip_id": "vidDDD", "title": "Long cricket match discussion analysis",
                "description": "Detailed analysis of match",
                "tags": ["cricket", "analysis"],
                "is_shorts": True,
            },
        }, 410.0),
        ("fact:seo_outcome:5", {
            "subject": "seo_outcome", "relation": "has_performance",
            "object": {
                "clip_id": "vidEEE", "title": "Question of the day for cricket fans",
                "description": "Daily question for fans",
                "tags": ["cricket", "question"],
                "is_shorts": True,
            },
        }, 510.0),
    ]
    for key, fact, ts in facts:
        conn.execute(
            "INSERT INTO memories (key, value, timestamp, source) VALUES (?, ?, ?, ?)",
            (key, json.dumps(fact), ts, "test")
        )
    conn.commit()
    conn.close()

    shutil.copy(str(seed_path), str(db_path))

    from automation.learner.migrate import migrate
    migrate(db_path=str(db_path), dry_run=False)
