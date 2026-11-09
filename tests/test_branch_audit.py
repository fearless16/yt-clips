"""Branch audit regressions. Synthetic media only; no live service calls."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from tests.automation.test_youtube_thumbnail_delivery import delivery, run_upload, VIDEO_ID


def test_gameplay_marker_cannot_downgrade_release_checks(tmp_path):
    from automation.release_review import required_checks, _artifacts
    clip = tmp_path / "clip.mp4"
    clip.with_name("clip_edit.json").write_text('{"composition":"portrait-caption-v1"}')
    clip.with_name("clip_gameplay.json").write_text("{}")
    assert "gameplay_rights_and_attribution" in required_checks(clip)
    with pytest.raises(ValueError, match="[Gg]ameplay"):
        _artifacts(clip, tmp_path / "source.mp4")


def test_nested_gameplay_evidence_cannot_downgrade_release_checks(tmp_path):
    from automation.release_review import required_checks
    clip = tmp_path / "clip.mp4"
    clip.with_name("clip_edit.json").write_text(json.dumps({
        "composition": "portrait-caption-v1", "gameplay_layout": {}}))
    assert "gameplay_rights_and_attribution" in required_checks(clip)


@pytest.mark.parametrize("contents", ["[]", "null", "not json"])
def test_bad_finalization_manifest_is_a_controlled_resume_hold(tmp_path, contents):
    from automation.orchestrator import _existing_exports
    batch = tmp_path / "batch"
    batch.mkdir()
    (batch / "clip.mp4").write_bytes(b"synthetic private media")
    (batch / "finalization.json").write_text(contents)
    with pytest.raises(ValueError, match="[Ff]inal"):
        _existing_exports(str(tmp_path), str(batch))


@pytest.mark.parametrize("already_credited", [False, True])
def test_multibyte_credit_overflow_is_held_even_when_idempotent(tmp_path, already_credited):
    from automation.cricket_gameplay import append_gameplay_credit, credit_text
    selection = {"asset": {"attribution": "Gameplay: fixture owner"}, "disclaimer": "EA fixture disclaimer"}
    sidecar = tmp_path / "clip_gameplay.json"
    sidecar.write_text(json.dumps({"selection": selection}))
    description = "क" * 1700
    if already_credited:
        description += "\n\n" + credit_text(selection)
    with pytest.raises(ValueError, match="attribution exceeds"):
        append_gameplay_credit({"description": description}, sidecar)


@pytest.mark.parametrize("description", ["क" * 1700, "Cricket opinion"], ids=["multibyte", "markdown"])
def test_upload_never_drops_or_rewrites_gameplay_credit_before_auth(tmp_path, monkeypatch, description):
    import upload
    from automation.cricket_gameplay import credit_text
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"synthetic media")
    # Markdown stripping also used to alter the literal reviewed credit.
    selection = {"asset": {"attribution": "Recording: **Fixture Owner**"}, "disclaimer": "Fixture disclaimer"}
    metadata = tmp_path / "clip_metadata.json"
    metadata.write_text(json.dumps({"title": "Cricket", "description": description + "\n\n" + credit_text(selection)}))
    video.with_name("clip_gameplay.json").write_text(json.dumps({"selection": selection}))
    monkeypatch.setitem(upload.cfg, "release_review", {"required": False})
    monkeypatch.setitem(upload.cfg, "thumbnail", {"enabled": False})
    monkeypatch.setattr(upload, "_validate_shorts_video", lambda *_: True)
    auth = Mock(return_value=None)
    monkeypatch.setattr(upload, "get_authenticated_service", auth)
    before = metadata.read_bytes()
    assert upload.upload_video(str(video), str(metadata), privacy="private") is None
    auth.assert_not_called()
    assert metadata.read_bytes() == before


def test_upload_delivers_complete_plain_gameplay_credit(delivery):
    from automation.cricket_gameplay import credit_text
    selection = {"asset": {"attribution": "Recording: Fixture Owner"}, "disclaimer": "Fixture disclaimer"}
    delivery[0].with_name("portrait_gameplay.json").write_text(json.dumps({"selection": selection}))
    payload = json.loads(delivery[1].read_text())
    payload["description"] += "\n\n" + credit_text(selection)
    delivery[1].write_text(json.dumps(payload))
    assert run_upload(delivery) == VIDEO_ID
    description = delivery[3].videos().insert.call_args.kwargs["body"]["snippet"]["description"]
    assert credit_text(selection) in description


def _mock_export(tmp_path, monkeypatch):
    import export
    monkeypatch.setitem(export.cfg, "paths", {**export.cfg["paths"], "shorts": str(tmp_path / "shorts")})
    monkeypatch.setitem(export.cfg["export"], "max_workers", 1)
    monkeypatch.setattr(export, "_get_best_encoder", lambda: "libx264")
    monkeypatch.setattr(export, "_caption_options", lambda *_: ({}, False, False))
    monkeypatch.setattr(export, "_validate_av_sync", lambda *_: True)
    monkeypatch.setattr(export, "_is_exportable_cricket_highlight", lambda *_: True)
    rendered = []

    def render(source, start, end, output, *args, **kwargs):
        rendered.append(start)
        Path(output).write_bytes(b"synthetic render; not playable")
        return output

    monkeypatch.setattr(export, "export_clip", render)
    return export, rendered


def test_analysis_failure_preserves_sibling_and_reviewed_reserve(tmp_path, monkeypatch):
    export, rendered = _mock_export(tmp_path, monkeypatch)

    def analyze(source, start, end, **kwargs):
        if start == 0:
            raise RuntimeError("synthetic detector failure")
        return {"layout": {}, "export_strategy": {}}

    monkeypatch.setattr(export, "analyze_clip", analyze)
    clips = export.export_all({
        "bad": {"start": 0, "end": 2},
        "sibling": {"start": 20, "end": 22},
        "reserve": {"start": 30, "end": 32, "role": "reserve",
                    "complete_thought": True, "semantic_review": "text_only"},
    }, "source.mp4", generate_seo=False, composition="portrait-caption-v1")
    assert {clip.stem for clip in clips} == {"sibling", "reserve"}
    assert rendered == [20, 30]
    report = json.loads((clips[0].parent / "batch_failures.json").read_text())
    assert report["status"] == "RECOVERED"
    assert any("bad" in error and "analysis" in error for error in report["errors"])


@pytest.mark.parametrize("start,end", [(-1, 1), (0, float("nan")), (0, float("inf")),
                                      ("not a timestamp", 2), (None, 2), (True, 2)])
def test_invalid_source_range_never_reaches_analysis(tmp_path, monkeypatch, start, end):
    export, _ = _mock_export(tmp_path, monkeypatch)
    analyze = Mock(return_value={"layout": {}, "export_strategy": {}})
    monkeypatch.setattr(export, "analyze_clip", analyze)
    with pytest.raises(export.ExportBatchError, match="source range"):
        export.export_all({"bad": {"start": start, "end": end}}, "source.mp4",
                          generate_seo=False, composition="portrait-caption-v1")
    analyze.assert_not_called()


@pytest.mark.parametrize("clip_id", ["../escape", "..\\escape", "C:\\escape", "clip:alias", "NUL", "clip."])
def test_unsafe_clip_id_cannot_write_outside_its_batch(tmp_path, monkeypatch, clip_id):
    export, _ = _mock_export(tmp_path, monkeypatch)
    analyze = Mock(side_effect=AssertionError("unsafe IDs must be rejected first"))
    monkeypatch.setattr(export, "analyze_clip", analyze)
    with pytest.raises(ValueError, match="clip ID"):
        export.export_all({clip_id: {"start": 0, "end": 2}}, "source.mp4",
                          generate_seo=False, composition="portrait-caption-v1")
    analyze.assert_not_called()


@pytest.mark.parametrize("segments", [[None], [{"words": [None]}], [{"words": "bad"}]])
def test_malformed_speech_is_a_review_issue_not_an_attribute_crash(segments):
    from automation.speech_alignment import alignment_issues
    assert alignment_issues(segments, 0, 1)
