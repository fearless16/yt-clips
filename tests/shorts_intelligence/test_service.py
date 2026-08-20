from shorts_intelligence.models import PerformanceSnapshot, ShortRecord
from shorts_intelligence.service import ShortsIntelligence
from shorts_intelligence.store import ShortsStore
from shorts_intelligence.youtube_source import IngestionBatch


CHANNEL_ID = "UCQtCMKPc41MHd7hujVuGm5g"


class _Source:
    def fetch(self, *, captured_at=None):
        records = []
        snapshots = []
        for index in range(40):
            video_id = f"v{index}"
            records.append(ShortRecord(
                video_id=video_id,
                channel_id=CHANNEL_ID,
                title=f"Cricket T20 debate {index}",
                description="Test match reaction",
                published_at=f"2026-08-{(index % 20) + 1:02d}T10:00:00Z",
                duration_seconds=20 if index < 30 else 32,
                is_short=True,
            ))
            snapshots.append(PerformanceSnapshot(
                video_id=video_id,
                captured_at="2026-08-20T08:00:00+00:00",
                views=100,
                engaged_views=80 if index < 30 else 10,
                average_view_percentage=90 if index < 30 else 20,
                likes=10 if index < 30 else 0,
            ))
        return IngestionBatch(records, snapshots, shelf_count=40)


def test_service_syncs_fits_and_returns_compact_report(tmp_path):
    with ShortsStore(tmp_path / "db.sqlite", channel_id=CHANNEL_ID) as store:
        service = ShortsIntelligence(store=store, source=_Source())

        report = service.sync_and_fit(captured_at="2026-08-20T08:00:00+00:00")

        assert report["catalog"] == 40
        assert report["cricket"] == 40
        assert report["snapshots_added"] == 40
        assert report["model"]["observations"] == 40
        assert report["model"]["recommendations"][0]["feature_value"] == "15_24"
        assert set(report["model"]["recommendations"][0]) == {
            "feature_name", "feature_value", "effect", "effect_low", "n", "confidence"
        }
        assert store.latest_model()["observations"] == 40
        assert store.active_recommendations()[0]["feature_value"] == "15_24"
