"""Readiness regressions: real clip identity, early holds and browser bootstrap."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from automation.cricket_gameplay import STADIUM_COMPOSITION, load_library
from tests.test_stadium_pipeline import fixture_library


@pytest.mark.parametrize("evidence", [
    {"composition": STADIUM_COMPOSITION},
    {"export_config": {"composition": "cricket-gameplay-split-v1"}},
    {"composition": "portrait-caption-v1", "gameplay_layout": {}},
])
def test_enhancement_checks_actual_edit_evidence(tmp_path, evidence):
    from automation.orchestrator import _validate_enhancement_inputs
    clip = tmp_path / "clip.mp4"
    clip.with_name("clip_edit.json").write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="gameplay export"):
        _validate_enhancement_inputs([clip])


@pytest.mark.parametrize("contents", ["invalid json", "[]", '{"export_config": null}', "{}",
                                      '{"composition": "unknown"}'])
def test_enhancement_holds_malformed_evidence(tmp_path, contents):
    from automation.orchestrator import _validate_enhancement_inputs
    clip = tmp_path / "clip.mp4"
    clip.with_name("clip_edit.json").write_text(contents)
    with pytest.raises(ValueError, match="enhancement held"):
        _validate_enhancement_inputs([clip])


def test_missing_evidence_cannot_silently_enable_enhancement(tmp_path):
    from automation.orchestrator import _validate_enhancement_inputs
    with pytest.raises(ValueError, match="missing edit evidence"):
        _validate_enhancement_inputs([tmp_path / "clip.mp4"])


@pytest.mark.parametrize("override", [None, "portrait-caption-v1"])
@pytest.mark.parametrize("edit_present", [True, False])
def test_resumed_stadium_cannot_be_graded_or_synced(tmp_path, monkeypatch, override, edit_present):
    from automation import orchestrator
    batch = tmp_path / "shorts/batch"
    batch.mkdir(parents=True)
    clip = batch / "clip.mp4"
    clip.write_bytes(b"SYNTHETIC accepted media - must not be replaced")
    clip.with_name("clip_gameplay.json").write_text(json.dumps({"composition": STADIUM_COMPOSITION}))
    if edit_present:
        clip.with_name("clip_edit.json").write_text(json.dumps({"composition": STADIUM_COMPOSITION}))
    for suffix in ("_review.json", "_qc.json"):
        clip.with_name("clip" + suffix).write_text('{"synthetic": "preserve"}')
    before = {p.name: p.read_bytes() for p in batch.iterdir()}
    config = {"paths": {key: str(tmp_path / key) for key in ("shorts", "input", "transcripts", "highlights")},
              "export": {"composition": STADIUM_COMPOSITION}}
    monkeypatch.setattr(orchestrator, "_CONFIG", config)
    grade = Mock(side_effect=AssertionError("Must not reach grading"))
    sync = Mock(side_effect=AssertionError("Must not reach sync"))
    monkeypatch.setitem(sys.modules, "ref_grade", SimpleNamespace(grade_video=grade))
    monkeypatch.setattr("sync.sync_to_drive", sync)
    result = orchestrator.run("https://youtu.be/test", skip_download=True, skip_transcribe=True,
        skip_highlight=True, skip_export=True, skip_seo=True, auto_sync=True, mode="ref_grade",
        existing_exports_dir=str(batch), composition=override)
    assert any("stage6b" in error and "gameplay export" in error for error in result.failures)
    grade.assert_not_called()
    sync.assert_not_called()
    assert before == {p.name: p.read_bytes() for p in batch.iterdir()}


def test_real_portrait_resume_is_not_misclassified_by_stadium_default(tmp_path, monkeypatch):
    from automation import orchestrator
    batch = tmp_path / "shorts/batch"
    batch.mkdir(parents=True)
    clip = batch / "clip.mp4"
    clip.write_bytes(b"SYNTHETIC portrait")
    clip.with_name("clip_edit.json").write_text(json.dumps({
        "composition": "portrait-caption-v1", "gameplay_layout": None,
        "export_config": {"composition": STADIUM_COMPOSITION}}))
    config = {"paths": {key: str(tmp_path / key) for key in ("shorts", "input", "transcripts", "highlights")},
              "export": {"composition": STADIUM_COMPOSITION}, "thumbnail": {"enabled": False}}
    monkeypatch.setattr(orchestrator, "_CONFIG", config)
    grade = Mock(return_value=None)  # No real processing/move in this routing test.
    monkeypatch.setitem(sys.modules, "ref_grade", SimpleNamespace(grade_video=grade))
    monkeypatch.setattr("automation.seo.analytics.Analytics.get_summary", lambda *_: {})
    monkeypatch.setattr("shorts_intelligence.bridge.shadow_status", lambda *_: {})
    result = orchestrator.run("https://youtu.be/test", skip_download=True, skip_transcribe=True,
        skip_highlight=True, skip_export=True, skip_seo=True, mode="ref_grade", existing_exports_dir=str(batch))
    assert not result.failures
    grade.assert_called_once()
    assert clip.read_bytes() == b"SYNTHETIC portrait"


def test_missing_playwright_has_actionable_preflight_error(monkeypatch):
    from automation.stadium_layout import _check_caption_browser
    monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
    with pytest.raises(ValueError, match="pip install -r requirements.txt"):
        _check_caption_browser()


@pytest.mark.parametrize("failure", ["missing_executable", "launch", "page", None])
def test_browser_probe_checks_launch_and_closes_browser(tmp_path, monkeypatch, failure):
    from automation.stadium_layout import _check_caption_browser
    executable = tmp_path / "TEST-ONLY-browser.exe"
    if failure != "missing_executable":
        executable.write_bytes(b"not executed; mocked browser")
    runtime = MagicMock()
    runtime.chromium.executable_path = str(executable)
    browser = runtime.chromium.launch.return_value
    browser.new_page.return_value.locator.return_value.inner_text.return_value = "caption preflight"
    if failure == "launch":
        runtime.chromium.launch.side_effect = RuntimeError("test launch failure")
    elif failure == "page":
        browser.new_page.side_effect = RuntimeError("test page failure")
    context = MagicMock()
    context.__enter__.return_value = runtime
    monkeypatch.setattr("playwright.sync_api.sync_playwright", lambda: context)
    if failure:
        with pytest.raises(ValueError, match="playwright install chromium"):
            _check_caption_browser()
    else:
        _check_caption_browser()
    if failure in ("page", None):
        browser.close.assert_called_once()
    if failure == "missing_executable":
        runtime.chromium.launch.assert_not_called()


@pytest.mark.parametrize("config,expected", [
    ({}, 38),
    ({"highlight": {"max_duration": 30}}, 30),
    ({"clip_selection": {"output_max_duration": 12}, "highlight": {"max_duration": 30}}, 12),
])
def test_pipeline_coverage_uses_configured_ceiling(config, expected):
    from automation.gameplay_export import pipeline_gameplay_seconds
    assert pipeline_gameplay_seconds(config) == expected


@pytest.mark.parametrize("seconds", [True, "38", 0, -1, float("nan"), float("inf")])
def test_bad_coverage_ceiling_is_not_silently_coerced(seconds):
    from automation.gameplay_export import pipeline_gameplay_seconds
    with pytest.raises(ValueError, match="finite and positive"):
        pipeline_gameplay_seconds({"clip_selection": {"output_max_duration": seconds}})


def test_range_too_short_holds_before_browser_or_download(tmp_path, monkeypatch):
    from automation import orchestrator
    manifest, _ = fixture_library(tmp_path)
    monkeypatch.setattr(orchestrator, "_CONFIG", {
        "export": {"composition": STADIUM_COMPOSITION, "gameplay_manifest": str(manifest)},
        "clip_selection": {"output_max_duration": 38}})
    monkeypatch.setattr("automation.gameplay_export._duration", lambda *_: 8)
    monkeypatch.setattr("automation.stadium_export.validate_gameplay_crop", lambda *_: [240,128,1440,810])
    browser = Mock(side_effect=AssertionError("Duration must be checked first"))
    download = Mock(side_effect=AssertionError("No network work before preflight"))
    monkeypatch.setattr("automation.stadium_layout._check_caption_browser", browser)
    monkeypatch.setattr("download.download", download)
    result = orchestrator.run("https://youtu.be/test")
    assert any("gameplay preflight" in e and "38.000s" in e for e in result.failures)
    browser.assert_not_called()
    download.assert_not_called()


def test_exact_short_duration_can_pass_direct_library_preflight(tmp_path, monkeypatch):
    from automation.gameplay_export import preflight_library
    manifest, _ = fixture_library(tmp_path)
    monkeypatch.setattr("automation.gameplay_export._duration", lambda *_: 8)
    monkeypatch.setattr("automation.stadium_export.validate_gameplay_crop", lambda *_: [240,128,1440,810])
    monkeypatch.setattr("automation.stadium_layout._check_caption_browser", lambda: None)
    # This fixture is byte-only: isolate its duration/ordering contract from
    # the real-FFmpeg visual scan covered in test_live_batch_defects.py.
    monkeypatch.setattr("automation.gameplay_visual_qc.scan_dark_intervals", lambda *a, **k: [])
    library = load_library(manifest)
    assert preflight_library(library, STADIUM_COMPOSITION, required_seconds=8) is library
    with pytest.raises(ValueError, match="No approved continuous"):
        preflight_library(library, STADIUM_COMPOSITION, required_seconds=8.001)


def test_direct_batch_insufficient_coverage_stops_before_analysis_or_asr(tmp_path, monkeypatch):
    import export
    manifest, _ = fixture_library(tmp_path)
    monkeypatch.setattr("automation.gameplay_export._duration", lambda *_: 8)
    monkeypatch.setattr("automation.stadium_export.validate_gameplay_crop", lambda *_: [240,128,1440,810])
    monkeypatch.setattr("automation.stadium_layout._check_caption_browser", lambda: None)
    monkeypatch.setitem(export.cfg, "paths", {**export.cfg["paths"], "shorts": str(tmp_path / "shorts")})
    analyze = Mock(side_effect=AssertionError("Must check coverage first"))
    align = Mock(side_effect=AssertionError("Must not spend on ASR"))
    monkeypatch.setattr(export, "analyze_clip", analyze)
    monkeypatch.setattr("automation.speech_alignment.transcribe_range", align)
    highlights = {"clip1": {"start": 0, "end": 10, "text": "India cricket"}}
    with pytest.raises(export.ExportBatchError, match="coverage preflight") as held:
        export.export_all(highlights, str(tmp_path / "source.mp4"), generate_seo=False,
                          composition=STADIUM_COMPOSITION, gameplay_manifest=str(manifest))
    assert held.value.exported == []
    analyze.assert_not_called()
    align.assert_not_called()
    reports = list((tmp_path / "shorts").glob("*/batch_failures.json"))
    assert len(reports) == 1 and json.loads(reports[0].read_text())["status"] == "PARTIAL"


def test_coverage_hold_preserves_sibling_and_reviewed_reserve_recovery(tmp_path, monkeypatch):
    """Routing-only test: rendering is a stub, permissions are synthetic."""
    import export
    manifest, _ = fixture_library(tmp_path)
    monkeypatch.setattr("automation.gameplay_export._duration", lambda *_: 8)
    monkeypatch.setattr("automation.stadium_export.validate_gameplay_crop", lambda *_: [240,128,1440,810])
    monkeypatch.setattr("automation.stadium_layout._check_caption_browser", lambda: None)
    monkeypatch.setitem(export.cfg, "paths", {**export.cfg["paths"], "shorts": str(tmp_path / "shorts")})
    monkeypatch.setitem(export.cfg["export"], "max_workers", 1)
    analyzed = []
    rendered = []

    def analyze(source, start, end, **kwargs):
        analyzed.append(start)
        return {"layout": {}, "export_strategy": {}}

    def render(source, start, end, output, *args, **kwargs):
        rendered.append(start)
        Path(output).write_bytes(b"SYNTHETIC MOCK RENDER; not a real playable video")
        return output

    monkeypatch.setattr(export, "analyze_clip", analyze)
    monkeypatch.setattr(export, "export_clip", render)
    monkeypatch.setattr(export, "_get_best_encoder", lambda: "libx264")
    monkeypatch.setattr(export, "_caption_options", lambda *_: ({}, False, False))
    monkeypatch.setattr(export, "_validate_av_sync", lambda *_: True)
    monkeypatch.setattr(export, "_is_exportable_cricket_highlight", lambda *_: True)
    highlights = {
        "too_long": {"start": 0, "end": 10, "text": "India cricket"},
        "sibling": {"start": 20, "end": 22, "text": "India cricket"},
        "reserve": {"start": 30, "end": 32, "text": "India cricket", "role": "reserve",
                    "complete_thought": True, "semantic_review": "text_only"},
    }
    clips = export.export_all(highlights, str(tmp_path / "source.mp4"), generate_seo=False,
                              composition=STADIUM_COMPOSITION, gameplay_manifest=str(manifest))
    assert {clip.stem for clip in clips} == {"sibling", "reserve"}
    assert analyzed == rendered == [20, 30]
    report = json.loads((clips[0].parent / "batch_failures.json").read_text())
    assert report["status"] == "RECOVERED"
    assert any("too_long" in error and "coverage preflight" in error for error in report["errors"])
