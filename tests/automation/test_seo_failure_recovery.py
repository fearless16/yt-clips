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
    (existing_export.parents[2] / "cookies.txt").write_text("auth")
    calls = []

    def retry(output_dir):
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


def test_stage8b_reports_persisted_seo_error_when_retry_does_not_recover(
    existing_export, monkeypatch
):
    marker = existing_export.with_name("clip1_seo_failed.json")
    marker.write_text(
        json.dumps({"clip_id": "clip1", "error": "quality gate rejected output"}),
        encoding="utf-8",
    )
    (existing_export.parents[2] / "cookies.txt").write_text("auth")
    uploads = []
    retries = []
    monkeypatch.setattr(
        "automation.seo.seo.retry_failed_seo",
        lambda output_dir: retries.append(output_dir) or {"recovered": 0, "total": 1},
    )
    monkeypatch.setattr(
        "upload.upload_video", lambda *args, **kwargs: uploads.append(args)
    )

    result = _run_existing_export(auto_upload=True)

    assert retries == [str(existing_export.parent)]
    assert uploads == []
    assert result.seo_generated == 0
    assert any(
        "stage8b clip1.mp4" in failure
        and "quality gate rejected output" in failure
        for failure in result.failures
    )


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
    (existing_export.parents[2] / "cookies.txt").write_text("auth")
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
    (existing_export.parents[2] / "cookies.txt").write_text("auth")
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
    (existing_export.parents[2] / "cookies.txt").write_text("auth")
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


def test_seo_failure_marker_persists_error_without_forwarding_it_on_retry(
    tmp_path, monkeypatch
):
    from automation.seo import seo

    monkeypatch.setattr(
        seo,
        "generate_clip_seo",
        lambda **_: (_ for _ in ()).throw(seo.SEOGenerationError("provider exhausted")),
    )
    metadata_path = tmp_path / "clip1_metadata.json"
    metadata_path.write_text(json.dumps({"title": "stale"}), encoding="utf-8")

    result = seo.generate_seo_for_exported_clip(
        "clip1",
        "Kohli hits a six",
        str(tmp_path),
        video_title="India match",
        approved_search_queries=[],
    )
    marker_path = tmp_path / "clip1_seo_failed.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))

    assert result == {"_seo_failed": True, "error": "provider exhausted"}
    assert marker["error"] == "provider exhausted"
    assert not metadata_path.exists()

    captured = {}
    monkeypatch.setattr(
        seo,
        "generate_clip_seo",
        lambda **kwargs: captured.update(kwargs) or {
            "title": "ok",
            "description": "specific recovered clip context",
        },
    )
    assert seo.retry_failed_seo(str(tmp_path)) == {"recovered": 1, "total": 1}
    assert "error" not in captured


def test_seo_worker_does_not_log_success_for_failure_result(monkeypatch):
    import export

    logger = MagicMock()
    monkeypatch.setattr(export, "log", logger)
    monkeypatch.setattr(
        "automation.seo.seo.generate_seo_for_exported_clip",
        lambda **_: {"_seo_failed": True, "error": "provider exhausted"},
    )

    export._seo_worker("clip1", {"text": "six"}, pytest.TempPathFactory, {})

    logger.info.assert_not_called()
    logger.error.assert_called_once_with(
        "SEO failed for %s: %s", "clip1", "provider exhausted"
    )


def test_retry_uses_marker_filename_clip_id_not_untrusted_payload(
    tmp_path, monkeypatch
):
    from automation.seo import seo

    marker = tmp_path / "clip1_seo_failed.json"
    marker.write_text(
        json.dumps({"clip_id": "../escaped", "transcript": "six"}),
        encoding="utf-8",
    )
    captured = {}
    monkeypatch.setattr(
        seo,
        "generate_clip_seo",
        lambda **kwargs: captured.update(kwargs) or {
            "title": "recovered",
            "description": "specific recovered clip context",
        },
    )

    result = seo.retry_failed_seo(str(tmp_path))

    assert result == {"recovered": 1, "total": 1}
    assert captured["clip_id"] == "clip1"
    assert (tmp_path / "clip1_metadata.json").is_file()
    assert not (tmp_path.parent / "escaped_metadata.json").exists()


def test_retry_marker_forces_regeneration_when_stale_metadata_exists(
    tmp_path, monkeypatch
):
    from automation.seo import seo

    marker = tmp_path / "clip1_seo_failed.json"
    marker.write_text(json.dumps({"clip_id": "clip1"}), encoding="utf-8")
    (tmp_path / "clip1_metadata.json").write_text(
        json.dumps({"title": "stale", "description": "stale description"}),
        encoding="utf-8",
    )
    generate = MagicMock(return_value={
        "title": "recovered",
        "description": "specific recovered clip context",
    })
    monkeypatch.setattr(seo, "generate_clip_seo", generate)

    result = seo.retry_failed_seo(str(tmp_path))

    assert result == {"recovered": 1, "total": 1}
    generate.assert_called_once()
    metadata = json.loads(
        (tmp_path / "clip1_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["title"] == "recovered"
    assert not marker.exists()


def test_retry_atomically_persists_latest_error(tmp_path, monkeypatch):
    from automation.seo import seo

    marker = tmp_path / "clip1_seo_failed.json"
    marker.write_text(
        json.dumps({"clip_id": "clip1", "error": "old error"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        seo,
        "generate_clip_seo",
        lambda **_: (_ for _ in ()).throw(seo.SEOGenerationError("latest error")),
    )
    replacements = []
    real_replace = seo.os.replace

    def replace(src, dst):
        replacements.append((str(src), str(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(seo.os, "replace", replace)

    result = seo.retry_failed_seo(str(tmp_path))

    assert result == {"recovered": 0, "total": 1}
    assert json.loads(marker.read_text(encoding="utf-8"))["error"] == "latest error"
    assert replacements[-1][1] == str(marker)
    assert not marker.with_suffix(".tmp").exists()


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
        "max_workers": 1,
        "super_resolution": False,
    })
    monkeypatch.setattr(export, "_new_export_batch_id", lambda: "batch")
    monkeypatch.setattr(export, "_load_transcript_segments", lambda *_: [])
    monkeypatch.setattr(export, "_load_export_cricket_context", lambda: "cricket")
    monkeypatch.setattr(export, "_is_exportable_cricket_highlight", lambda *_: True)
    monkeypatch.setattr(export, "_get_best_encoder", lambda: "libx264")
    monkeypatch.setattr(export, "_validate_av_sync", lambda *_: None)
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


def test_export_all_aborts_when_seo_artifact_persistence_fails(
    tmp_path, monkeypatch
):
    import export

    _stub_export_all(tmp_path, monkeypatch, export)

    def seo_worker(*_args):
        raise OSError("metadata disk full")

    monkeypatch.setattr(export, "_seo_worker", seo_worker)

    with pytest.raises(OSError, match="metadata disk full"):
        export.export_all(
            {"clip1": {"start": 0, "end": 10, "text": "Kohli six"}},
            str(tmp_path / "source.mp4"),
        )


def test_export_all_aborts_at_finite_seo_deadline_without_waiting_on_shutdown(
    tmp_path, monkeypatch
):
    import export

    _stub_export_all(tmp_path, monkeypatch, export)
    monkeypatch.setitem(export.cfg, "seo", {
        **export.cfg.get("seo", {}),
        "export_deadline_seconds": 0.01,
    })
    observed_timeouts = []
    shutdowns = []
    seo_pending = []

    def seo_worker(clip_id, _info, output_dir, _ctx):
        (output_dir / f"{clip_id}_metadata.json").write_text("{}")

    monkeypatch.setattr(export, "_seo_worker", seo_worker)

    class DeferredSEOFuture:
        def __init__(self, fn, args):
            self.fn = fn
            self.args = args
            self.cancelled = False

        def result(self, timeout=None):
            observed_timeouts.append(timeout)
            if timeout is not None:
                raise TimeoutError("synthetic timeout")

        def cancel(self):
            self.cancelled = True
            return True

    class Executor:
        def __init__(self, max_workers, **_kwargs):
            self.is_seo = max_workers == 2
            self.pending = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.shutdown(wait=True)

        def submit(self, fn, *args):
            if self.is_seo:
                future = DeferredSEOFuture(fn, args)
                self.pending.append(future)
                seo_pending.append(future)
                return future
            future = Future()
            future.set_result(fn(*args))
            return future

        def shutdown(self, wait=True, cancel_futures=False):
            shutdowns.append((self.is_seo, wait, cancel_futures))

    monkeypatch.setattr(export, "ThreadPoolExecutor", Executor)

    with pytest.raises(TimeoutError, match="SEO generation exceeded"):
        export.export_all(
            {"clip1": {"start": 0, "end": 10, "text": "Kohli six"}},
            str(tmp_path / "source.mp4"),
        )

    assert observed_timeouts and observed_timeouts[0] is not None
    assert all(future.cancelled for future in seo_pending)
    assert (True, False, True) in shutdowns


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
    (existing_export.parents[2] / "cookies.txt").write_text("auth")
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

    def retry(output_dir):
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


def test_initial_metadata_persistence_failure_propagates_and_removes_stale_file(
    tmp_path, monkeypatch
):
    from automation.seo import seo

    metadata = tmp_path / "clip1_metadata.json"
    metadata.write_text(
        json.dumps({"title": "stale", "description": "stale description"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        seo,
        "generate_clip_seo",
        lambda **_: {"title": "fresh", "description": "fresh description"},
    )
    real_replace = seo.os.replace

    def replace(src, dst):
        if Path(dst) == metadata:
            raise OSError("metadata disk full")
        return real_replace(src, dst)

    monkeypatch.setattr(seo.os, "replace", replace)

    with pytest.raises(OSError, match="metadata disk full"):
        seo.generate_seo_for_exported_clip(
            "clip1", "Kohli six", str(tmp_path), approved_search_queries=[]
        )

    assert not metadata.exists()


def test_retry_marker_rewrite_failure_does_not_block_other_markers(
    tmp_path, monkeypatch
):
    from automation.seo import seo

    first = tmp_path / "clip1_seo_failed.json"
    second = tmp_path / "clip2_seo_failed.json"
    for marker in (first, second):
        marker.write_text(
            json.dumps({"clip_id": marker.stem, "error": "old error"}),
            encoding="utf-8",
        )

    def generate(clip_id, **_kwargs):
        if clip_id == "clip1":
            raise seo.SEOGenerationError("latest error")
        return {"title": "recovered", "description": "recovered description"}

    monkeypatch.setattr(seo, "generate_clip_seo", generate)
    real_replace = seo.os.replace

    def replace(src, dst):
        if Path(dst) == first:
            raise OSError("marker disk full")
        return real_replace(src, dst)

    monkeypatch.setattr(seo.os, "replace", replace)

    result = seo.retry_failed_seo(str(tmp_path))

    assert result == {"recovered": 1, "total": 2}
    assert first.exists()
    assert not second.exists()
    assert (tmp_path / "clip2_metadata.json").is_file()
