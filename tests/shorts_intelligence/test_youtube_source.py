from shorts_intelligence.youtube_source import SourceConfig, YouTubeShortsSource


CHANNEL_ID = "UCQtCMKPc41MHd7hujVuGm5g"


def test_source_joins_only_shelf_proven_shorts_and_metric_groups():
    requested: list[list[str]] = []
    metric_calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def shelf_loader():
        return [
            {"id": "short-1", "title": "Shelf title"},
            {"id": "short-2", "title": "Shelf title 2"},
        ]

    def video_loader(ids):
        requested.append(list(ids))
        return [
            {
                "id": video_id,
                "snippet": {
                    "channelId": CHANNEL_ID,
                    "title": f"Cricket Test match {video_id}",
                    "description": "T20 cricket reaction",
                    "publishedAt": "2026-08-16T04:42:38Z",
                    "categoryId": "17",
                    "tags": ["cricket", "test match"],
                },
                "contentDetails": {"duration": "PT23S"},
                "statistics": {"viewCount": "143", "likeCount": "8", "commentCount": "1"},
            }
            for video_id in ids
        ]

    def analytics_loader(ids, metrics):
        metric_calls.append((tuple(ids), tuple(metrics)))
        if "engagedViews" in metrics:
            return [{
                "video": video_id,
                "engagedViews": 20,
                "views": 141,
                "estimatedMinutesWatched": 7,
                "averageViewDuration": 13,
                "averageViewPercentage": 57.69,
            } for video_id in ids]
        if "shares" in metrics:
            return [{"video": video_id, "likes": 7, "comments": 1, "shares": 2} for video_id in ids]
        return [{"video": video_id, "subscribersGained": 1, "subscribersLost": 0} for video_id in ids]

    source = YouTubeShortsSource(
        SourceConfig(channel_id=CHANNEL_ID, handle="@CricketWithPrajjwal2.0"),
        shelf_loader=shelf_loader,
        video_loader=video_loader,
        analytics_loader=analytics_loader,
        local_metadata_loader=lambda: {"short-1": {"hook_type": "debate"}},
    )

    batch = source.fetch(captured_at="2026-08-20T08:00:00+00:00")

    assert requested == [["short-1", "short-2"]]
    assert len(metric_calls) == 3
    assert [record.video_id for record in batch.records] == ["short-1", "short-2"]
    assert all(record.is_short for record in batch.records)
    assert batch.records[0].local_metadata["hook_type"] == "debate"
    assert batch.records[0].tags == ("cricket", "test match")
    assert batch.snapshots[0].average_view_percentage == 57.69
    assert batch.snapshots[0].shares == 2
    assert batch.snapshots[0].subscribers_gained == 1


def test_source_drops_cross_channel_video_even_if_loader_returns_it():
    source = YouTubeShortsSource(
        SourceConfig(channel_id=CHANNEL_ID, handle="@channel"),
        shelf_loader=lambda: [{"id": "bad"}],
        video_loader=lambda _ids: [{
            "id": "bad",
            "snippet": {
                "channelId": "other",
                "title": "Cricket",
                "description": "",
                "publishedAt": "2026-01-01T00:00:00Z",
            },
            "contentDetails": {"duration": "PT10S"},
            "statistics": {},
        }],
        analytics_loader=lambda _ids, _metrics: [],
    )

    batch = source.fetch(captured_at="2026-08-20T08:00:00+00:00")

    assert batch.records == []
    assert batch.snapshots == []
    assert "cross_channel:bad" in batch.issues


def test_iso_duration_parser_supports_hours_minutes_and_seconds():
    assert YouTubeShortsSource.parse_duration("PT1H2M3S") == 3723
