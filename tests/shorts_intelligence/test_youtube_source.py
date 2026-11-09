import json

import pytest

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


@pytest.fixture
def receipt_package(tmp_path):
    from automation.youtube_delivery import artifact_hashes
    clip = tmp_path / "clip.mp4"
    metadata = tmp_path / "clip_metadata.json"
    cover = tmp_path / "clip_thumb.jpg"
    clip.write_bytes(b"fixture video")
    metadata.write_text(json.dumps({"title": "Cricket analysis", "hook_type": "debate"}))
    cover.write_bytes(b"fixture cover; hash validation is separate from image decoding")
    payload = {"schema_version": 2, "channel_id": CHANNEL_ID, "video_id": "receipt-short",
               "status": "COMPLETE", "artifacts": artifact_hashes(clip, metadata, cover),
               "intent": {"privacy": "public", "publish_at": None, "notify_subscribers": False}}
    proof = tmp_path / "clip_delivery.json"
    proof.write_text(json.dumps(payload))
    source = YouTubeShortsSource(SourceConfig(channel_id=CHANNEL_ID, handle="@fixture", local_metadata_root=str(tmp_path)))
    return source, clip, metadata, cover, proof, payload


@pytest.mark.parametrize("version", [1, 2])
def test_local_metadata_joins_hash_bound_receipts_without_mutation(receipt_package, version):
    source, _, metadata, _, proof, payload = receipt_package
    payload["schema_version"] = version
    proof.write_text(json.dumps(payload))
    before = metadata.read_bytes()
    assert source._load_local_metadata()["receipt-short"]["hook_type"] == "debate"
    assert metadata.read_bytes() == before


@pytest.mark.parametrize("damage", ["media", "metadata", "cover", "channel", "path", "schema", "unknown", "corrupt"])
def test_invalid_receipt_cannot_supply_trusted_metadata(receipt_package, damage):
    source, clip, metadata, cover, proof, payload = receipt_package
    if damage in {"media", "metadata", "cover"}:
        {"media": clip, "metadata": metadata, "cover": cover}[damage].write_bytes(b"changed")
    elif damage == "channel":
        payload["channel_id"] = "other-channel"
    elif damage == "path":
        payload["artifacts"]["metadata"]["path"] = str(metadata.parent / "outside_metadata.json")
    elif damage == "schema":
        payload["schema_version"] = 999
    elif damage == "unknown":
        payload["status"] = "VIDEO_UPLOAD_UNKNOWN"
    if damage == "corrupt":
        proof.write_text("{")
    else:
        proof.write_text(json.dumps(payload))
    assert source._load_local_metadata() == {}


def test_legacy_metadata_id_remains_supported_without_a_receipt(tmp_path):
    metadata = {"youtube_video_id": "legacy-short", "hook_type": "debate"}
    (tmp_path / "legacy_metadata.json").write_text(json.dumps(metadata))
    source = YouTubeShortsSource(SourceConfig(channel_id=CHANNEL_ID, handle="@fixture", local_metadata_root=str(tmp_path)))
    assert source._load_local_metadata() == {"legacy-short": metadata}


def test_receipt_does_not_bypass_shelf_or_owner_gates(receipt_package):
    source, _, _, _, _, _ = receipt_package
    source._shelf_loader = lambda: [{"id": "different-shelf-short"}]
    source._video_loader = lambda _: []
    source._analytics_loader = lambda *_: []
    assert source.fetch().records == []
