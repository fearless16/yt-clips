"""YouTube Analytics feedback sync tests."""

from unittest.mock import MagicMock


class _FakeQuery:
    def __init__(self, response):
        self._response = response

    def execute(self):
        return self._response


class _FakeReports:
    def __init__(self, response):
        self._response = response
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeQuery(self._response)


class _FakeService:
    def __init__(self, response):
        self.reports_api = _FakeReports(response)

    def reports(self):
        return self.reports_api


def test_sync_updates_clip_learner_with_real_short_metrics():
    from automation.seo.analytics import sync_clip_performance_from_youtube

    learner = MagicMock()
    learner.get_all_clips.return_value = [{
        "clip_id": "batch/clip1",
        "youtube_video_id": "AbC123xyz98",
    }]
    service = _FakeService({
        "columnHeaders": [
            {"name": "video"},
            {"name": "views"},
            {"name": "engagedViews"},
            {"name": "averageViewDuration"},
            {"name": "averageViewPercentage"},
        ],
        "rows": [["AbC123xyz98", 250, 75, 18.4, 62.5]],
    })

    updated = sync_clip_performance_from_youtube(learner=learner, service=service)

    assert updated == 1
    learner.update_performance.assert_called_once_with(
        clip_id="batch/clip1",
        youtube_video_id="AbC123xyz98",
        views=250,
        estimated_retention=0.625,
        completion_rate=0.3,
        avg_view_duration_seconds=18.4,
    )
    call = service.reports_api.calls[0]
    assert call["dimensions"] == "video"
    assert call["filters"] == "video==AbC123xyz98"


def test_sync_skips_api_when_no_uploaded_clips():
    from automation.seo.analytics import sync_clip_performance_from_youtube

    learner = MagicMock()
    learner.get_all_clips.return_value = [{
        "clip_id": "batch/clip1",
        "youtube_video_id": None,
    }]
    service = _FakeService({"rows": []})

    assert sync_clip_performance_from_youtube(learner=learner, service=service) == 0
    assert service.reports_api.calls == []
    learner.update_performance.assert_not_called()


def test_sync_clamps_looping_retention_and_zero_view_completion():
    from automation.seo.analytics import sync_clip_performance_from_youtube

    learner = MagicMock()
    learner.get_all_clips.return_value = [{
        "clip_id": "batch/clip1",
        "youtube_video_id": "LpQ123xyz98",
    }]
    service = _FakeService({
        "columnHeaders": [
            {"name": "video"}, {"name": "views"},
            {"name": "engagedViews"}, {"name": "averageViewDuration"},
            {"name": "averageViewPercentage"},
        ],
        "rows": [["LpQ123xyz98", 0, 9, 25, 140]],
    })

    sync_clip_performance_from_youtube(learner=learner, service=service)

    kwargs = learner.update_performance.call_args.kwargs
    assert kwargs["estimated_retention"] == 1.0
    assert kwargs["completion_rate"] == 0.0


def test_sync_ignores_invalid_stale_video_ids():
    from automation.seo.analytics import sync_clip_performance_from_youtube

    learner = MagicMock()
    learner.get_all_clips.return_value = [{
        "clip_id": "batch/clip1",
        "youtube_video_id": "uploaded_id_789",
    }]
    service = _FakeService({"rows": []})

    assert sync_clip_performance_from_youtube(learner=learner, service=service) == 0
    assert service.reports_api.calls == []
