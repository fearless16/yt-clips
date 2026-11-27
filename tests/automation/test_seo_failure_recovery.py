import json
from concurrent.futures import Future
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def existing_export(tmp_path, monkeypatch):
    from automation import orchestrator

    export_dir = tmp_path / "shorts" / "batch"
    export_dir.mkdir(parents=True)
    clip_path = export_dir / "clip1.mp4"
    clip_path.write_bytes(b"video")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(orchestrator, "_CONFIG", {
        "paths": {
            "input": str(tmp_path / "input"),
            "transcripts": str(tmp_path / "transcripts"),
            "highlights": str(tmp_path / "highlights"),
            "shorts": str(tmp_path / "shorts"),
        },
        "download": {"output_filename": "video.mp4"},
        "youtube": {"privacy_status": "private"},
        "upload_schedule": {"interval_hours": 1},
    })
    monkeypatch.setattr("thumbnail.process_all_thumbnails", lambda *_: None)
    monkeypatch.setattr(
        "automation.seo.analytics.Analytics.get_summary", lambda *_: {}
    )
    monkeypatch.setattr("shorts_intelligence.bridge.shadow_status", lambda *_: {})
    monkeypatch.setattr("sync.sync_db_to_drive", lambda: None)
    return clip_path


def _run_existing_export(**kwargs):
    from automation import orchestrator

    skip_sync = kwargs.pop("skip_sync", True)
    return orchestrator.run(
        "https://youtu.be/AbC123xyz98",
        skip_download=True,
        skip_transcribe=True,
        skip_highlight=True,
        skip_export=True,
        skip_sync=skip_sync,
        skip_enhancement=True,
        **kwargs,
    )


def test_seo_generated_counts_only_actual_metadata(existing_export):
    existing_export.with_name("clip1_metadata.json").write_text(
        json.dumps({"title": "grounded", "description": "specific clip context"}),
        encoding="utf-8",
    )

    result = _run_existing_export()

    assert result.seo_generated == 1


def test_stage8b_retries_marker_once_before_scheduling_and_upload(
    existing_export, monkeypatch
):
    export_dir = existing_export.parent
    marker = export_dir / "clip1_seo_failed.json"
    marker.write_text(
        json.dumps({"clip_id": "clip1", "error": "provider exhausted"}),
        encoding="utf-8",
    )
    (existing_export.parents[2] / "yt_channel_token.json").write_text("{}")
    calls = []

    def retry(output_dir, **_):
        calls.append(("retry", output_dir))
        existing_export.with_name("clip1_metadata.json").write_text(
            json.dumps({
                "title": "recovered",
                "description": "specific recovered clip context",
                "quality_score": 0.9,
            }),
            encoding="utf-8",
        )
        marker.unlink()
        return {"recovered": 1, "total": 1}

    def assign(**kwargs):
        calls.append(("schedule", kwargs["clips"]))
        return [("clip1", datetime(2026, 8, 31, tzinfo=timezone.utc))]

    def upload(*args, **kwargs):
        calls.append(("upload", args[0]))
        return "video-id"

    monkeypatch.setattr("automation.seo.seo.retry_failed_seo", retry)
    monkeypatch.setattr("scheduler.assign_clips_to_slots", assign)
    monkeypatch.setattr("automation.scheduling.is_dead_day", lambda *_: False)
    monkeypatch.setattr("upload.upload_video", upload)
    monkeypatch.setattr("shorts_intelligence.bridge.record_upload", lambda *_: None)

    result = _run_existing_export(auto_upload=True, auto_schedule=True)

    assert [call[0] for call in calls] == ["retry", "schedule", "upload"]
    assert calls[0][1] == str(export_dir)
    assert result.seo_generated == 1
    assert result.uploaded_count == 1




def test_stage8b_does_not_retry_seo_when_skip_seo_is_set(
    existing_export, monkeypatch
):
    existing_export.with_name("clip1_seo_failed.json").write_text(
        json.dumps({"clip_id": "clip1", "error": "provider exhausted"}),
        encoding="utf-8",
    )
    existing_export.with_name("clip1_metadata.json").write_text(
        json.dumps({"title": "stale title", "description": "stale description"}),
        encoding="utf-8",
    )
    (existing_export.parents[2] / "yt_channel_token.json").write_text("{}")
    retry = MagicMock()
    monkeypatch.setattr("automation.seo.seo.retry_failed_seo", retry)
    monkeypatch.setattr("upload.upload_video", MagicMock())

    result = _run_existing_export(
        auto_upload=True,
        skip_seo=True,
    )

    retry.assert_not_called()
    assert result.uploaded_count == 0
    assert result.seo_generated == 0


def test_stage8b_falsey_upload_is_failed_not_published(
    existing_export, monkeypatch
):
    existing_export.with_name("clip1_metadata.json").write_text(
        json.dumps({"title": "grounded", "description": "specific clip context"}),
        encoding="utf-8",
    )
    (existing_export.parents[2] / "yt_channel_token.json").write_text("{}")
    emitted = []
    infra_failures = []
    monkeypatch.setattr("upload.upload_video", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "automation.orchestrator._emit_event",
        lambda *args: emitted.append(args),
    )
    monkeypatch.setattr(
        "automation.orchestrator._emit_infra_failed",
        lambda *args, **kwargs: infra_failures.append((args, kwargs)),
    )

    result = _run_existing_export(auto_upload=True)

    assert result.uploaded_count == 0
    assert any("upload returned no video ID" in failure for failure in result.failures)
    assert len(infra_failures) == 1
    assert all(event[1].value != "published" for event in emitted)


def test_stage8b_incomplete_thumbnail_is_not_counted_or_published(existing_export, monkeypatch):
    from upload import DeliveryIncompleteError

    existing_export.with_name("clip1_metadata.json").write_text(
        json.dumps({"title": "grounded", "description": "specific clip context"}), encoding="utf-8")
    (existing_export.parents[2] / "yt_channel_token.json").write_text("{}")
    emitted = []

    def incomplete(*_args, **_kwargs):
        raise DeliveryIncompleteError("already-uploaded-id", "thumbnail verification pending")

    monkeypatch.setattr("upload.upload_video", incomplete)
    monkeypatch.setattr("automation.orchestrator._emit_event", lambda *args: emitted.append(args))
    result = _run_existing_export(auto_upload=True)
    assert result.uploaded_count == 0
    assert any("already-uploaded-id" in failure and "thumbnail" in failure for failure in result.failures)
    assert all(event[1].value != "published" for event in emitted)


def test_stage8b_schedules_only_clips_with_metadata_after_retry(
    existing_export, monkeypatch
):
    valid_clip = existing_export.with_name("clip2.mp4")
    valid_clip.write_bytes(b"video")
    valid_clip.with_name("clip2_metadata.json").write_text(
        json.dumps({
            "title": "grounded",
            "description": "specific clip context",
            "quality_score": 0.8,
        }),
        encoding="utf-8",
    )
    existing_export.with_name("clip1_seo_failed.json").write_text(
        json.dumps({"clip_id": "clip1", "error": "still unavailable"}),
        encoding="utf-8",
    )
    (existing_export.parents[2] / "yt_channel_token.json").write_text("{}")
    scheduled = []
    uploads = []
    monkeypatch.setattr(
        "automation.seo.seo.retry_failed_seo",
        lambda *_: {"recovered": 0, "total": 1},
    )

    def assign(**kwargs):
        scheduled.extend(kwargs["clips"])
        return [("clip2", datetime(2026, 8, 31, tzinfo=timezone.utc))]

    monkeypatch.setattr("scheduler.assign_clips_to_slots", assign)
    monkeypatch.setattr("automation.scheduling.is_dead_day", lambda *_: False)
    monkeypatch.setattr(
        "upload.upload_video",
        lambda path, *_args, **_kwargs: uploads.append(path) or "video-id",
    )
    monkeypatch.setattr("shorts_intelligence.bridge.record_upload", lambda *_: None)

    result = _run_existing_export(auto_upload=True, auto_schedule=True)

    assert scheduled == ["clip2"]
    assert uploads == [str(valid_clip)]
    assert result.uploaded_count == 1


def test_stage8b_does_not_assign_a_new_slot_to_an_existing_delivery(existing_export, monkeypatch):
    from thumbnail import ThumbnailBatchResult
    monkeypatch.setattr("thumbnail.process_all_thumbnails", lambda *_: ThumbnailBatchResult())
    existing_export.with_name("clip1_metadata.json").write_text(json.dumps({
        "title": "grounded", "description": "specific clip context"}))
    # The uploader, not the scheduler, must validate/reconcile this receipt.
    existing_export.with_name("clip1_delivery.json").write_text('{"status":"THUMBNAIL_PENDING"}')
    (existing_export.parents[2] / "yt_channel_token.json").write_text("{}")
    calls = []
    monkeypatch.setattr("scheduler.assign_clips_to_slots",
                        lambda **kwargs: calls.append(("scheduled", kwargs["clips"])) or [])
    def uploaded(*args, **kwargs):
        calls.append(("upload", kwargs))
        return "existing-video"
    monkeypatch.setattr("upload.upload_video", uploaded)
    monkeypatch.setattr("shorts_intelligence.bridge.record_upload", lambda *_: None)
    result = _run_existing_export(auto_upload=True, auto_schedule=True)
    assert result.uploaded_count == 1
    assert all("clip1" not in value for kind, value in calls if kind == "scheduled")
    kwargs = next(value for kind, value in calls if kind == "upload")
    assert kwargs["privacy"] == "scheduled"
    assert kwargs["publish_at"] is None, "Uploader must reuse and validate its bound slot"












def _stub_export_all(tmp_path, monkeypatch, export):
    monkeypatch.setenv("YT_CLIPS_SKIP_INSTAGRAM", "1")
    shorts = tmp_path / "shorts"
    monkeypatch.setitem(export.cfg, "paths", {
        **export.cfg["paths"],
        "shorts": str(shorts),
        "input": str(tmp_path),
        "transcripts": str(tmp_path),
    })
    monkeypatch.setitem(export.cfg, "export", {
        **export.cfg.get("export", {}),
        "composition": "portrait-caption-v1",
        "max_workers": 1,
        "super_resolution": False,
        "captions": {"enabled": False, "require_word_timing": False},
    })
    monkeypatch.setattr(export, "_new_export_batch_id", lambda: "batch")
    monkeypatch.setattr(export, "_load_transcript_segments", lambda *_: [])
    monkeypatch.setattr(export, "_get_best_encoder", lambda: "libx264")
    monkeypatch.setattr(export, "_validate_av_sync", lambda *_: True)
    monkeypatch.setattr(
        export,
        "analyze_clip",
        lambda *_args, **_kwargs: {
            "export_strategy": {"should_drop": False},
            "layout": {"layout_type": "solo"},
        },
    )

    def export_clip(*args, **kwargs):
        path = args[3]
        with open(path, "wb") as handle:
            handle.write(b"video")
        return path

    monkeypatch.setattr(export, "export_clip", export_clip)
    return shorts










def test_orchestrator_rejects_malformed_and_incomplete_metadata(
    existing_export, monkeypatch
):
    invalid_payloads = [
        "",
        "{malformed",
        json.dumps({}),
        json.dumps({"_seo_failed": True, "title": "x", "description": "y"}),
        json.dumps({"title": "title only"}),
        json.dumps({"description": "description only"}),
        json.dumps({"title": "  ", "description": "  "}),
    ]
    clips = []
    for index, payload in enumerate(invalid_payloads, start=1):
        clip = existing_export.with_name(f"clip{index}.mp4")
        clip.write_bytes(b"video")
        clip.with_name(f"clip{index}_metadata.json").write_text(
            payload, encoding="utf-8"
        )
        clips.append(clip)
    valid_clip = existing_export.with_name("clip8.mp4")
    valid_clip.write_bytes(b"video")
    valid_clip.with_name("clip8_metadata.json").write_text(
        json.dumps({
            "title": "specific title",
            "description": "specific clip description",
        }),
        encoding="utf-8",
    )
    (existing_export.parents[2] / "yt_channel_token.json").write_text("{}")
    uploads = []
    monkeypatch.setattr(
        "upload.upload_video",
        lambda path, *_args, **_kwargs: uploads.append(path) or "video-id",
    )
    monkeypatch.setattr("shorts_intelligence.bridge.record_upload", lambda *_: None)

    result = _run_existing_export(auto_upload=True)

    assert result.seo_generated == 1
    assert uploads == [str(valid_clip)]
    assert result.uploaded_count == 1


def test_recovery_runs_before_drive_sync_without_upload(
    existing_export, monkeypatch
):
    marker = existing_export.with_name("clip1_seo_failed.json")
    marker.write_text(
        json.dumps({"clip_id": "clip1", "error": "provider exhausted"}),
        encoding="utf-8",
    )
    metadata = existing_export.with_name("clip1_metadata.json")
    metadata.write_text(
        json.dumps({"title": "stale title", "description": "stale description"}),
        encoding="utf-8",
    )
    calls = []

    def retry(output_dir, **_):
        calls.append("retry")
        metadata.write_text(
            json.dumps({
                "title": "recovered title",
                "description": "recovered description",
            }),
            encoding="utf-8",
        )
        marker.unlink()
        return {"recovered": 1, "total": 1}

    def sync_to_drive(folder_path):
        calls.append("sync")
        assert folder_path == str(existing_export.parent)
        assert not marker.exists()
        assert json.loads(metadata.read_text(encoding="utf-8"))["title"] == "recovered title"

    monkeypatch.setattr("automation.seo.seo.retry_failed_seo", retry)
    monkeypatch.setattr("sync.sync_to_drive", sync_to_drive)

    result = _run_existing_export(auto_sync=True, skip_sync=False)

    assert calls == ["retry", "sync"]
    assert result.seo_generated == 1




