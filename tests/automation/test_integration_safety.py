"""Integration boundaries must not reuse stale work or publish/sync implicitly."""
import json
from unittest.mock import Mock

import pytest


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    from automation import orchestrator

    monkeypatch.chdir(tmp_path)
    config = {
        "paths": {name: str(tmp_path / name) for name in
                  ("input", "transcripts", "highlights", "shorts")},
        "download": {"output_filename": "video.mp4"},
        "youtube": {"privacy_status": "private"},
        "thumbnail": {"enabled": False},
    }
    monkeypatch.setattr(orchestrator, "_CONFIG", config)
    monkeypatch.setattr("automation.seo.analytics.Analytics.get_summary", lambda *_: {})
    monkeypatch.setattr("shorts_intelligence.bridge.shadow_status", lambda *_: {})
    monkeypatch.setattr("shorts_intelligence.bridge.record_exported", lambda *_: 0)
    sync = Mock()
    monkeypatch.setattr("sync.sync_db_to_drive", sync)
    return orchestrator, config, sync


def run_local(orchestrator, **kwargs):
    return orchestrator.run("https://youtu.be/AbC123xyz98", skip_download=True,
                            skip_transcribe=True, skip_seo=True, **kwargs)


def test_selection_failure_cannot_export_stale_highlights(runtime, tmp_path, monkeypatch):
    orchestrator, _, _ = runtime
    stale = tmp_path / "highlights/video.yaml"
    stale.parent.mkdir()
    stale.write_text("old_clip: {start: 1, end: 20}")
    monkeypatch.setattr("automation.clip_selection.pipeline.detect_highlights",
                        Mock(side_effect=RuntimeError("selection unavailable")))
    export = Mock(return_value=[])
    monkeypatch.setattr("export.export_all", export)
    result = run_local(orchestrator)
    export.assert_not_called()
    assert result.failures == ["stage3-5: selection unavailable"]
    assert "old_clip" in stale.read_text()


def test_zero_selection_cannot_export_an_old_file(runtime, monkeypatch):
    orchestrator, _, _ = runtime
    monkeypatch.setattr("automation.clip_selection.pipeline.detect_highlights", Mock(return_value=[]))
    export = Mock(return_value=[])
    monkeypatch.setattr("export.export_all", export)
    result = run_local(orchestrator)
    export.assert_not_called()
    assert result.failures == [] and result.exported == []


def test_review_block_is_reported_instead_of_successful_empty_run(runtime, monkeypatch):
    orchestrator, _, _ = runtime
    def blocked(*args, diagnostics, **kwargs):
        diagnostics.update(status="REVIEW_REQUIRED", candidates=29, hard_rejected=8,
                           below_quality=8, review_candidates=13, editorial_rejected=0)
        return []
    monkeypatch.setattr("automation.clip_selection.pipeline.detect_highlights", blocked)
    export = Mock()
    monkeypatch.setattr("export.export_all", export)
    result = run_local(orchestrator)
    assert result.selection_status == "REVIEW_REQUIRED"
    assert result.candidate_count == 29
    assert result.rejected_clips == 16
    assert result.review_candidates == 13
    assert result.failures == ["stage3-5: editorial review required; no approved drafts"]
    assert next(stage for stage in result.stage_results if stage["stage"] == "highlight")["status"] == "review"
    export.assert_not_called()


def test_early_empty_run_has_one_structured_non_success_summary(runtime, monkeypatch):
    import io
    from utils.logger import JsonStreamHandler

    orchestrator, _, _ = runtime
    monkeypatch.setattr("automation.clip_selection.pipeline.detect_highlights", Mock(return_value=[]))
    stream = io.StringIO()
    handler = JsonStreamHandler(stream)
    orchestrator.log.addHandler(handler)
    try:
        result = run_local(orchestrator)
    finally:
        orchestrator.log.removeHandler(handler)
        handler.close()
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    endings = [record for record in records if record.get("stage") == "pipeline" and record["status"] != "start"]
    assert len(endings) == 1
    assert endings[0]["status"] == "empty"
    assert endings[0]["metadata"]["stages"] == result.stage_results
    assert next(stage for stage in result.stage_results if stage["stage"] == "highlight")["status"] == "empty"


def test_export_completion_counts_are_reported_by_the_export_phase(runtime, tmp_path, monkeypatch):
    orchestrator, _, _ = runtime
    batch = tmp_path / "shorts/batch"
    batch.mkdir(parents=True)
    clip = batch / "clip1.mp4"
    clip.write_bytes(b"fixture")
    monkeypatch.setattr("export.export_all", Mock(return_value=[clip]))
    result = run_local(orchestrator, skip_highlight=True)
    stages = {stage["stage"]: stage for stage in result.stage_results}
    assert stages["export"]["metadata"]["exported"] == 1
    assert "hook_audit" not in stages  # theatre scorer removed (A5)


def test_partial_batch_keeps_successful_outputs_and_reports_non_success_phase(runtime, tmp_path, monkeypatch):
    from export import ExportBatchError
    orchestrator, _, _ = runtime
    clip = tmp_path / "shorts/batch/clip1.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"successful current-run export")
    monkeypatch.setattr("export.export_all", Mock(side_effect=ExportBatchError(["clip2: alignment held"], [clip])))
    result = run_local(orchestrator, skip_highlight=True)
    assert result.exported == [clip]
    assert result.failures == ["stage6: Export batch incomplete: clip2: alignment held"]
    export_stage = next(stage for stage in result.stage_results if stage["stage"] == "export")
    assert export_stage["status"] == "partial"
    assert export_stage["metadata"]["exported"] == 1


def test_editorial_target_shortfall_still_ships_the_good_clip(runtime, tmp_path, monkeypatch):
    orchestrator, config, _ = runtime
    config["clip_selection"] = {"max_selected": 3, "editorial": {"enabled": True, "target_min": 2}}
    clip = tmp_path / "shorts/batch/clip1.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"one acceptable export")

    def one(*args, diagnostics, **kwargs):
        diagnostics.update(status="DRAFTS_SELECTED", candidates=10)
        return [{"id": "clip1", "start": 1, "end": 20, "text": "India cricket"}]

    monkeypatch.setattr("automation.clip_selection.pipeline.detect_highlights", one)
    monkeypatch.setattr("export.export_all", Mock(return_value=[clip]))
    result = run_local(orchestrator)
    assert result.exported == [clip]
    # One good clip is a shortfall to log, not a failed run (cancer register C5).
    assert not any("editorial_target" in failure for failure in result.failures)


def test_failed_drive_pull_aborts_before_using_stale_local_source(runtime, monkeypatch):
    orchestrator, _, _ = runtime
    monkeypatch.setattr("sync.download_from_drive", Mock(side_effect=RuntimeError("drive unavailable")))
    detector = Mock(return_value=[])
    monkeypatch.setattr("automation.clip_selection.pipeline.detect_highlights", detector)
    result = run_local(orchestrator, sync_from_drive=True)
    detector.assert_not_called()
    assert result.failures == ["stage1b: drive unavailable"]


@pytest.mark.parametrize("auto_sync,skip_sync,expected", [(False, False, 0), (True, True, 0), (True, False, 1)])
def test_database_sync_requires_explicit_sync_intent(runtime, tmp_path, monkeypatch,
                                                    auto_sync, skip_sync, expected):
    orchestrator, _, sync = runtime
    batch = tmp_path / "shorts/batch"
    batch.mkdir(parents=True)
    (batch / "clip1.mp4").write_bytes(b"fixture")
    monkeypatch.setattr("sync.sync_to_drive", Mock())
    run_local(orchestrator, skip_highlight=True, skip_export=True,
              auto_sync=auto_sync, skip_sync=skip_sync)
    assert sync.call_count == expected


def test_nested_review_video_is_not_a_resume_batch(runtime, tmp_path):
    orchestrator, _, _ = runtime
    nested = tmp_path / "shorts/pilot/review/packaging-v4"
    nested.mkdir(parents=True)
    (nested / "packaging-review.mp4").write_bytes(b"review")
    result = run_local(orchestrator, skip_highlight=True, skip_export=True)
    assert result.exported == []


def test_multiple_resume_batches_require_explicit_choice(runtime, tmp_path):
    orchestrator, _, _ = runtime
    for name in ("batch-a", "batch-z"):
        batch = tmp_path / "shorts" / name
        batch.mkdir(parents=True)
        (batch / "clip1.mp4").write_bytes(b"fixture")
    result = run_local(orchestrator, skip_highlight=True, skip_export=True)
    assert result.exported == []
    assert any("export-dir" in reason for reason in result.failures)


def test_explicit_resume_directory_scopes_work(runtime, tmp_path):
    orchestrator, _, _ = runtime
    chosen = tmp_path / "shorts/batch-a"
    for name in ("batch-a", "batch-z"):
        batch = tmp_path / "shorts" / name
        batch.mkdir(parents=True)
        (batch / "clip1.mp4").write_bytes(b"fixture")
    result = run_local(orchestrator, skip_highlight=True, skip_export=True,
                       existing_exports_dir=str(chosen))
    assert result.exported == [chosen / "clip1.mp4"]


def test_disabled_thumbnails_are_not_regenerated(runtime, tmp_path, monkeypatch):
    orchestrator, _, _ = runtime
    batch = tmp_path / "shorts/batch"
    batch.mkdir(parents=True)
    (batch / "clip1.mp4").write_bytes(b"fixture")
    thumbnails = Mock()
    monkeypatch.setattr("thumbnail.process_all_thumbnails", thumbnails)
    run_local(orchestrator, skip_highlight=True, skip_export=True)
    thumbnails.assert_not_called()


def test_thumbnail_batch_respects_failed_metadata_and_existing_cover(tmp_path, monkeypatch):
    import thumbnail

    monkeypatch.setitem(thumbnail.cfg, "thumbnail", {"enabled": True})
    generate = Mock()
    monkeypatch.setattr(thumbnail.ThumbnailGenerator, "generate_for_clip", generate)
    for stem in ("failed", "approved", "new"):
        (tmp_path / f"{stem}.mp4").write_bytes(b"video")
        (tmp_path / f"{stem}_metadata.json").write_text(json.dumps({"title": "ground", "description": "context"}))
    (tmp_path / "failed_seo_failed.json").write_text("{}")
    cover = tmp_path / "approved_thumb.jpg"
    cover.write_bytes(b"owner-selected cover")
    thumbnail.process_all_thumbnails(str(tmp_path))
    assert generate.call_count == 1
    assert generate.call_args.args[0] == str(tmp_path / "new.mp4")
    assert cover.read_bytes() == b"owner-selected cover"




def test_thumbnail_failure_reaches_pipeline_result(runtime, tmp_path, monkeypatch):
    import thumbnail

    orchestrator, config, _ = runtime
    batch = tmp_path / "shorts/batch"
    batch.mkdir(parents=True)
    (batch / "clip1.mp4").write_bytes(b"video")
    (batch / "clip1_metadata.json").write_text(json.dumps({
        "title": "Cricket discussion", "description": "Grounded context"}))
    config["thumbnail"] = {"enabled": True}
    monkeypatch.setitem(thumbnail.cfg, "thumbnail", {"enabled": True})
    monkeypatch.setattr("thumbnail.ThumbnailGenerator.generate_for_clip",
                        Mock(side_effect=ValueError("Cover text needs review")))
    result = run_local(orchestrator, skip_highlight=True, skip_export=True)
    assert result.failures == ["stage7b: clip1.mp4: Cover text needs review"]
    assert result.exported == [batch / "clip1.mp4"]
    assert result.thumbnails_generated == 0
    assert next(stage for stage in result.stage_results if stage["stage"] == "thumbnails")["status"] == "partial"


def test_ranker_prompt_cannot_excuse_missing_context():
    import prompts

    for prompt in (prompts.HIGHLIGHT_RANKER_USER_TEMPLATE, prompts.CANDIDATE_EVAL_SYSTEM):
        assert "unless they are extremely strong" not in prompt
        assert "necessary setup" in prompt.lower()


def test_resume_dir_rejects_escape_and_nested_review(runtime, tmp_path):
    orchestrator, _, _ = runtime
    for chosen in (tmp_path, tmp_path / "shorts/pilot/review"):
        with pytest.raises(ValueError, match="export-dir"):
            orchestrator._existing_exports(str(tmp_path / "shorts"), str(chosen))


def test_export_dir_requires_skip_export_before_any_work(runtime):
    orchestrator, _, _ = runtime
    result = run_local(orchestrator, existing_exports_dir="shorts/batch")
    assert result.failures == ["--export-dir requires --skip-export"]


def test_cli_passes_explicit_resume_directory(monkeypatch):
    from automation import cli, orchestrator

    run = Mock(return_value=orchestrator.PipelineResult())
    monkeypatch.setattr(orchestrator, "run", run)
    assert cli.main(["https://youtu.be/test", "--skip-export", "--export-dir", "shorts/batch"]) == 0
    assert run.call_args.kwargs["existing_exports_dir"] == "shorts/batch"






def test_missing_cover_copy_is_held_before_extracting_source(tmp_path):
    from thumbnail import ThumbnailGenerator

    clip = tmp_path / "clip.mp4"
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    edit = tmp_path / "clip_edit.json"
    edit.write_text(json.dumps({"source_path": str(source), "source_start": 0, "source_end": 10}))
    with pytest.raises(ValueError, match="Cover text"):
        ThumbnailGenerator()._generate_source_cover(clip, {}, edit, tmp_path / "cover.jpg")


def test_delivery_failure_stops_remote_side_effects(runtime, tmp_path, monkeypatch):
    orchestrator, config, sync = runtime
    config["delivery_qc"] = {"enabled": True}
    batch = tmp_path / "shorts/batch"
    batch.mkdir(parents=True)
    clip = batch / "clip1.mp4"
    clip.write_bytes(b"corrupt video")
    (batch / "clip1_metadata.json").write_text(json.dumps({"title": "Ground size", "description": "Context"}))
    (tmp_path / "cookies.txt").write_text("fixture")
    monkeypatch.setattr("automation.delivery_qc.check_delivery", lambda *_: {"status": "FAIL", "errors": ["decode failed"]})
    folder_sync, upload = Mock(), Mock(return_value="id")
    monkeypatch.setattr("sync.sync_to_drive", folder_sync)
    monkeypatch.setattr("upload.upload_video", upload)
    result = run_local(orchestrator, skip_highlight=True, skip_export=True,
                       auto_sync=True, auto_upload=True)
    assert result.exported == []  # quarantined; the reason stays in failures
    assert any("decode failed" in reason for reason in result.failures)
    assert result.release_held == 1
    for call in (sync, folder_sync, upload):
        call.assert_not_called()


@pytest.mark.parametrize("failure", [False, True])
def test_legacy_cover_extraction_owns_its_scratch(tmp_path, monkeypatch, failure):
    import subprocess
    import thumbnail
    from PIL import Image

    monkeypatch.setitem(thumbnail.cfg, "thumbnail", {"use_ai": False, "require_source_frame": False})
    scratch = tmp_path / "scratch"
    monkeypatch.setitem(thumbnail.cfg["paths"], "temp", str(scratch))
    clip = tmp_path / "clip.mp4"
    meta = tmp_path / "clip_metadata.json"
    meta.write_text(json.dumps({"title": "GROUND BHI DEKHO"}))
    original = clip.with_suffix(".jpg")
    original.write_bytes(b"existing source-side photo")
    seen = []

    def extract(cmd, **kwargs):
        from pathlib import Path
        # The final path is explicitly owned scratch, never clip.jpg.
        target = Path(cmd[cmd.index("-vframes") + 2])
        seen.append(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (1080, 1920)).save(target)
        if failure:
            raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(thumbnail.subprocess, "run", extract)
    monkeypatch.setattr(thumbnail.ThumbnailGenerator, "_add_text", lambda self, image, text: image)
    result = thumbnail.ThumbnailGenerator().generate_for_clip(str(clip), str(meta), str(tmp_path / "cover.jpg"))
    assert result is not failure
    assert original.read_bytes() == b"existing source-side photo"
    assert seen and all(path.is_relative_to(scratch) for path in seen)
    assert list(scratch.iterdir()) == []
