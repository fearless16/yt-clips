from dataclasses import replace
from datetime import datetime, timezone

import pytest

from shorts_intelligence.models import PerformanceSnapshot, ShortRecord
from shorts_intelligence.store import ShortsStore


CHANNEL_ID = "UCQtCMKPc41MHd7hujVuGm5g"


def _short(video_id: str = "abc123") -> ShortRecord:
    return ShortRecord(
        video_id=video_id,
        channel_id=CHANNEL_ID,
        title="India win a Test match thriller",
        description="Cricket reaction",
        published_at="2026-08-01T10:00:00Z",
        duration_seconds=23,
        is_short=True,
    )


def test_store_is_wal_backed_and_idempotent(tmp_path):
    db = tmp_path / "shorts_intelligence.db"
    with ShortsStore(db, channel_id=CHANNEL_ID) as store:
        store.upsert_short(_short())
        store.upsert_short(_short())

        assert store.journal_mode() == "wal"
        assert store.summary()["shorts"] == 1
        assert store.schema_version == 4


def test_store_rejects_non_shelf_and_wrong_channel_rows(tmp_path):
    with ShortsStore(tmp_path / "db.sqlite", channel_id=CHANNEL_ID) as store:
        not_short = _short("not-short")
        not_short = ShortRecord(**{**not_short.to_dict(), "is_short": False})
        with pytest.raises(ValueError, match="Shorts shelf"):
            store.upsert_short(not_short)

        wrong = _short("wrong-channel")
        wrong = ShortRecord(**{**wrong.to_dict(), "channel_id": "other"})
        with pytest.raises(ValueError, match="channel"):
            store.upsert_short(wrong)


def test_snapshots_are_append_only_and_capture_looping_retention(tmp_path):
    captured = datetime(2026, 8, 20, 8, 0, tzinfo=timezone.utc)
    with ShortsStore(tmp_path / "db.sqlite", channel_id=CHANNEL_ID) as store:
        store.upsert_short(_short())
        snapshot = PerformanceSnapshot(
            video_id="abc123",
            captured_at=captured.isoformat(),
            engaged_views=500,
            views=1_500,
            average_view_duration_seconds=18,
            average_view_percentage=261.04,
            likes=18,
            comments=2,
            shares=1,
            subscribers_gained=2,
        )
        assert store.add_snapshot(snapshot) is True
        assert store.add_snapshot(snapshot) is False
        unchanged_later = replace(snapshot, captured_at="2026-08-20T09:00:00+00:00")
        assert store.add_snapshot(unchanged_later) is False
        changed_later = replace(
            snapshot,
            captured_at="2026-08-20T10:00:00+00:00",
            views=1_501,
            engaged_views=501,
        )
        assert store.add_snapshot(changed_later) is True

        rows = store.training_rows()
        assert len(rows) == 1
        assert rows[0]["average_view_percentage"] == 261.04
        assert rows[0]["evidence_at"] == "2026-08-01T10:00:00Z"


def test_training_rows_exclude_non_cricket_and_unknown(tmp_path):
    with ShortsStore(tmp_path / "db.sqlite", channel_id=CHANNEL_ID) as store:
        for video_id, status in (("c", "cricket"), ("n", "non_cricket"), ("u", "unknown")):
            record = _short(video_id)
            store.upsert_short(record, domain_status=status, domain_reason="test")
            store.add_snapshot(PerformanceSnapshot(
                video_id=video_id,
                captured_at="2026-08-20T08:00:00+00:00",
                views=100,
                engaged_views=50,
            ))

        assert [row["video_id"] for row in store.training_rows()] == ["c"]


def test_batch_ingestion_is_atomic(tmp_path):
    with ShortsStore(tmp_path / "db.sqlite", channel_id=CHANNEL_ID) as store:
        wrong = _short("wrong")
        wrong = ShortRecord(**{**wrong.to_dict(), "channel_id": "another-channel"})

        with pytest.raises(ValueError, match="channel"):
            store.ingest_batch([_short("valid"), wrong], [])

        assert store.summary()["shorts"] == 0


def test_pending_upload_links_production_features_on_catalog_sync(tmp_path):
    with ShortsStore(tmp_path / "db.sqlite", channel_id=CHANNEL_ID) as store:
        store.record_production(
            clip_id="batch/clip1",
            transcript="Complete cricket thought",
            features={"hook_type": "debate", "complete_thought": "true"},
        )
        store.record_upload("batch/clip1", "abc123")
        store.ingest_batch(
            [_short("abc123")],
            [PerformanceSnapshot(
                video_id="abc123",
                captured_at="2026-08-20T08:00:00+00:00",
                views=100,
                engaged_views=50,
            )],
        )

        row = store.training_rows()[0]
        assert row["features"]["hook_type"] == "debate"
        assert row["features"]["complete_thought"] == "true"
