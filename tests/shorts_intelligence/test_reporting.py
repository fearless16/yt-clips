from shorts_intelligence.learner import BayesianSegmentLearner
from shorts_intelligence.models import PerformanceSnapshot, ShortRecord
from shorts_intelligence.reporting import build_report, render_html
from shorts_intelligence.store import ShortsStore


CHANNEL_ID = "UCQtCMKPc41MHd7hujVuGm5g"


def test_report_uses_only_canonical_store_and_real_outcomes(tmp_path):
    with ShortsStore(tmp_path / "shorts.db", channel_id=CHANNEL_ID) as store:
        for index in range(12):
            video_id = f"short-{index}"
            store.upsert_short(ShortRecord(
                video_id=video_id,
                channel_id=CHANNEL_ID,
                title=f"Cricket reaction {index}",
                description="India match analysis",
                published_at=f"2026-08-{index + 1:02d}T10:00:00Z",
                duration_seconds=12 if index < 8 else 32,
                is_short=True,
            ))
            store.add_snapshot(PerformanceSnapshot(
                video_id=video_id,
                captured_at="2026-08-20T08:00:00+00:00",
                views=1000,
                engaged_views=900 if index < 8 else 100,
                average_view_percentage=95 if index < 8 else 20,
                likes=80 if index < 8 else 2,
            ))
        model = BayesianSegmentLearner().fit(store.training_rows())
        store.save_model(model, BayesianSegmentLearner().config)

        report = build_report(store)

    assert report["store"]["cricket"] == 12
    assert report["model"]["observations"] == 12
    assert report["top_shorts"][0]["video_id"].startswith("short-")
    assert report["top_shorts"][0]["title"].startswith("Cricket reaction")
    assert "seo_performance.json" not in str(report)


def test_report_html_is_compact_and_escapes_channel_data():
    report = {
        "store": {"shorts": 1, "cricket": 1, "non_cricket": 0, "unknown": 0, "snapshots": 1},
        "model": {"observations": 1, "baseline_mean": 0.5, "fitted_at": "now"},
        "recommendations": [],
        "evaluation": {"status": "insufficient_data", "observations": 1},
        "top_shorts": [{"video_id": "x", "title": "<script>alert(1)</script>", "outcome_score": 0.5}],
    }

    page = render_html(report)

    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert len(page) < 20_000
