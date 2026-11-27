"""Enabled stock can enter delivery; availability never invents rights approval."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from automation.cricket_gameplay import (credit_text, load_library, load_render_library,
                                        select_gameplay, sha256)


def review_library(tmp_path, media=None):
    media = media or tmp_path / "stock.mp4"
    if not media.exists():
        media.write_bytes(b"SYNTHETIC stock media for loader tests only")
    evidence = tmp_path / "recording.json"
    evidence.write_text(json.dumps({"id": "fixture", "uploader": "fixture owner",
        "license": "Creative Commons Attribution license (reuse allowed)"}), encoding="utf-8")
    payload = {"schema_version": 1, "usage": "LOCAL_REVIEW_ONLY", "assets": [{
        "id": "stock", "path": media.name, "sha256": sha256(media),
        "game": "EA Cricket 07", "publisher": "EA", "status": "REVIEW_ONLY",
        "rights_status": "UNRESOLVED", "reviewer": "Synthetic fixture", "reviewed_at": "2026-11-08",
        "source_video_id": "fixture", "owner": "fixture owner", "attribution": "Synthetic recording credit",
        "license_evidence": {"path": evidence.name, "sha256": sha256(evidence)},
        "approved_ranges": [[100, 180], [300, 400]], "crop": [240, 128, 1440, 810],
    }]}
    manifest = tmp_path / "review-library.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest, payload


def stock_library(tmp_path):
    """Enabled stock retains its real provenance, not invented rights approval."""
    manifest, payload = review_library(tmp_path)
    payload["usage"] = "STOCK"
    payload["assets"][0]["status"] = "AVAILABLE"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest, payload


def test_dry_run_exercises_enabled_stock_with_synthetic_provenance():
    import yaml
    from dry_run import _isolated_run
    from automation.cricket_gameplay import load_delivery_library

    @_isolated_run
    def inspect_fixture():
        config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
        assert json.loads(Path("drive_token.json").read_text(encoding="utf-8")) == {"token": "fake"}
        library = load_delivery_library(config["export"]["gameplay_manifest"])
        assert library["usage"] == "STOCK"
        assert library["assets"][0]["status"] == "AVAILABLE"
        assert library["assets"][0]["owner"] == "SYNTHETIC TEST ONLY"
        return {}

    inspect_fixture()


def test_enabled_stock_dry_run_really_exercises_drive_and_scheduled_upload(monkeypatch):
    import dry_run
    import sync
    import upload
    uploaded_files = []
    upload_file = sync._upload_file

    def live_builder(*args, **kwargs):
        pytest.fail("Dry run leaked the warm uploader's live Google API binding")

    monkeypatch.setattr(upload, "build", live_builder)

    def record_upload(service, path, parent):
        result = upload_file(service, path, parent)
        uploaded_files.append(path.name)
        return result

    monkeypatch.setattr(sync, "_upload_file", record_upload)
    result = dry_run.dry_run("https://youtu.be/wRP_KO9vP5g", skip_download=True,
        skip_transcribe=True, auto_sync=True, auto_upload=True, auto_schedule=True)
    assert not result["failures"]
    assert result["uploaded_count"] == len(result["exported"]) == 3
    assert {name for name in uploaded_files if name.endswith(".mp4")} == {
        "clip1.mp4", "clip2.mp4", "clip3.mp4"}
    assert {name for name in uploaded_files if name.endswith("_gameplay.json")} == {
        "clip1_gameplay.json", "clip2_gameplay.json", "clip3_gameplay.json"}


def test_enabled_stock_loads_for_delivery_without_claiming_cleared_rights(tmp_path):
    from automation.cricket_gameplay import load_delivery_library
    manifest, _ = stock_library(tmp_path)
    library = load_delivery_library(manifest)
    assert library["usage"] == "STOCK"
    assert library["assets"][0]["status"] == "AVAILABLE"
    assert library["assets"][0]["rights_status"] == "UNRESOLVED"
    assert library == load_render_library(manifest)


@pytest.mark.parametrize("change", [
    {"status": "WITHDRAWN"}, {"status": "APPROVED"}, {"rights_status": "CLEARED"},
    {"sha256": "0" * 64}, {"owner": "different owner"}, {"license_evidence": {}},
])
def test_enabled_stock_keeps_provenance_and_withdrawal_checks(tmp_path, change):
    from automation.cricket_gameplay import load_delivery_library
    manifest, payload = stock_library(tmp_path)
    payload["assets"][0].update(change)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_delivery_library(manifest)


@pytest.mark.parametrize("delivery", ["auto_upload", "auto_sync"])
def test_enabled_stock_reaches_pipeline_after_preflight(tmp_path, monkeypatch, delivery):
    from automation import orchestrator
    from unittest.mock import Mock
    manifest, _ = stock_library(tmp_path)
    monkeypatch.setattr(orchestrator, "_CONFIG", {"export": {
        "composition": "cricket-gameplay-stadium-v1", "gameplay_manifest": str(manifest)}})
    preflight = Mock()
    download = Mock(side_effect=ValueError("TEST STOP: no real download"))
    monkeypatch.setattr("automation.gameplay_export.preflight_library", preflight)
    monkeypatch.setattr("download.download", download)
    result = orchestrator.run("https://youtu.be/test", **{delivery: True})
    preflight.assert_called_once()
    assert preflight.call_args.args[0]["usage"] == "STOCK"
    download.assert_called_once()
    assert result.failures == ["stage1c: TEST STOP: no real download"]


def test_enabled_stock_upload_metadata_keeps_credit_and_checks_current_manifest(tmp_path, monkeypatch):
    import upload
    manifest, payload = stock_library(tmp_path)
    selected = select_gameplay(load_render_library(manifest), 15, "source:1")
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"SYNTHETIC stock export")
    clip.with_name("clip_gameplay.json").write_text(json.dumps({
        "selection": selected, "media_sha256": sha256(clip)}))
    monkeypatch.setitem(upload.cfg, "release_review", {"required": False})
    meta = {"title": "Cricket", "description": credit_text(selected)}
    assert credit_text(selected) in upload._prepare_upload_metadata(meta, clip)[1]
    payload["assets"][0]["status"] = "WITHDRAWN"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        upload._prepare_upload_metadata(meta, clip)


@pytest.mark.parametrize("changed", ["media", "manifest", "credit"])
def test_enabled_stock_holds_changed_delivery_evidence(tmp_path, changed):
    import upload
    manifest, payload = stock_library(tmp_path)
    selected = select_gameplay(load_render_library(manifest), 15, "source:1")
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"SYNTHETIC stock export")
    clip.with_name("clip_gameplay.json").write_text(json.dumps({
        "selection": selected, "media_sha256": sha256(clip)}))
    meta = {"title": "Cricket", "description": credit_text(selected)}
    if changed == "media":
        clip.write_bytes(b"Changed encoded file")
    elif changed == "manifest":
        payload["assets"][0]["reviewed_at"] = "2026-11-09"
        manifest.write_text(json.dumps(payload), encoding="utf-8")
    else:
        meta["description"] = "Credit removed"
    with pytest.raises(ValueError, match="[Gg]ameplay"):
        upload._prepare_upload_metadata(meta, clip)


def test_enabled_stock_rights_are_machine_verified_then_owner_approved(tmp_path):
    from PIL import Image
    from automation.release_review import create_review, release_status, _artifacts, approve, review_path
    manifest, _ = stock_library(tmp_path)
    selected = select_gameplay(load_render_library(manifest), 15, "source:1")
    clip, source = tmp_path / "clip.mp4", tmp_path / "source.mp4"
    clip.write_bytes(b"SYNTHETIC stock export")
    source.write_bytes(b"SYNTHETIC source voice")
    Image.new("RGB", (1080, 1920)).save(tmp_path / "clip_thumb.jpg")
    clip.with_name("clip_metadata.json").write_text(json.dumps({
        "title": "Cricket", "description": credit_text(selected)}))
    clip.with_name("clip_qc.json").write_text(json.dumps({"status": "PASS", "media_sha256": sha256(clip)}))
    clip.with_name("clip_gameplay.json").write_text(json.dumps({
        "selection": selected, "media_sha256": sha256(clip)}))
    clip.with_name("clip_edit.json").write_text(json.dumps({
        "source_path": str(source), "source_start": 0, "source_end": 15,
        "completeness": {"status": "PASS", "method": "text_only", "reason": "synthetic"}}))
    draft = create_review(clip, source)
    assert draft["artifacts"] == _artifacts(clip, source)
    assert "gameplay_license_evidence" in draft["artifacts"]
    # Licence, manifest hash and on-description credit are verified, not hand-typed.
    assert draft["checks"]["gameplay_rights_and_attribution"]["status"] == "PASS"
    assert release_status(clip)[0] == "REVIEW"
    approve(clip, "SYNTHETIC TEST REVIEWER")
    assert release_status(clip) == ("READY", [])
    clip.write_bytes(b"Changed export after approval")
    assert release_status(clip)[0] != "READY"


def test_stock_can_render_but_cannot_claim_production_clearance(tmp_path):
    manifest, _ = review_library(tmp_path)
    library = load_render_library(manifest)
    assert library["usage"] == "LOCAL_REVIEW_ONLY"
    assert library["assets"][0]["rights_status"] == "UNRESOLVED"
    with pytest.raises(ValueError, match="APPROVED"):
        load_library(manifest)


@pytest.mark.parametrize("change", [
    {"status": "APPROVED"}, {"status": "WITHDRAWN"}, {"rights_status": "CLEARED"},
    {"sha256": "0" * 64}, {"owner": "different owner"}, {"source_video_id": "different"},
    {"license_evidence": {}}, {"crop": [0, 0, 99, 99]}, {"approved_ranges": [[10, 9]]},
])
def test_review_mode_retains_identity_hash_geometry_and_withdrawal_checks(tmp_path, change):
    manifest, payload = review_library(tmp_path)
    payload["assets"][0].update(change)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_render_library(manifest)


def test_changed_license_label_never_becomes_reusable_stock(tmp_path):
    manifest, payload = review_library(tmp_path)
    evidence = tmp_path / "recording.json"
    evidence.write_text(json.dumps({"id": "fixture", "uploader": "fixture owner", "license": "Standard YouTube License"}))
    payload["assets"][0]["license_evidence"]["sha256"] = sha256(evidence)
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Creative Commons"):
        load_render_library(manifest)


def test_stock_random_ranges_vary_by_moment_but_are_reproducible(tmp_path):
    manifest, _ = review_library(tmp_path)
    library = load_render_library(manifest)
    selected = [select_gameplay(library, 38, f"source:{start}") for start in range(12)]
    assert len({row["start"] for row in selected}) == len(selected)
    for start, row in enumerate(selected):
        assert row == select_gameplay(library, 38, f"source:{start}")
        assert row["approved_range"][0] <= row["start"] < row["end"] <= row["approved_range"][1]
        assert row["end"] - row["start"] == pytest.approx(38)


def test_two_pass_audio_measures_only_the_retained_source_voice(monkeypatch):
    from automation.gameplay_export import source_audio_filter
    from unittest.mock import Mock
    stats = {"input_i": "-20.2", "input_tp": "-3.1", "input_lra": "8.0",
             "input_thresh": "-30.7", "target_offset": "0.6"}
    run = Mock(return_value=subprocess.CompletedProcess([], 0, "", "ffmpeg log\n" + json.dumps(stats)))
    monkeypatch.setattr("automation.gameplay_export.subprocess.run", run)
    af, evidence = source_audio_filter("voice.mp4", 10.25, 22.54)
    cmd = run.call_args.args[0]
    assert cmd[cmd.index("-ss")+1] == "10.250000"
    assert cmd[cmd.index("-t")+1] == "22.540000"
    assert cmd[cmd.index("-map")+1] == "0:a:0"
    assert "measured_I=-20.2" in af and "offset=0.6:linear=true" in af
    assert "speed" not in af and "atempo" not in af
    assert evidence["source_range"] == [10.25, 32.79]
    assert "apad=pad_dur=0.25" in af
    assert af.endswith("aresample=48000,atrim=duration=22.540000")


@pytest.mark.parametrize("stderr", ["no measurement", json.dumps({"input_i": "-inf",
    "input_tp": "-inf", "input_lra": "0", "input_thresh": "-70", "target_offset": "inf"})])
def test_two_pass_audio_holds_missing_or_silent_measurements(monkeypatch, stderr):
    from automation.gameplay_export import source_audio_filter
    monkeypatch.setattr("automation.gameplay_export.subprocess.run",
        lambda *a, **k: subprocess.CompletedProcess([], 0, "", stderr))
    with pytest.raises(ValueError, match="loudness measurement"):
        source_audio_filter("voice.mp4", 0, 10)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="Real FFmpeg required")
def test_dynamic_loudness_keeps_the_nonround_last_source_audio_block(tmp_path):
    import wave
    import numpy as np
    from automation.gameplay_export import source_audio_filter
    rate, duration = 48000, 3.137
    count = round(rate * duration)
    samples = .02 * np.sin(2 * np.pi * 440 * np.arange(count) / rate)
    samples[rate:rate+48] += .85  # high crest factor forces dynamic normalization
    source, encoded = tmp_path / "voice.wav", tmp_path / "voice.m4a"
    with wave.open(str(source), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((samples * 32767).astype("int16").tobytes())
    af, _ = source_audio_filter(source, 0, duration)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(source), "-af", af,
        "-c:a", "aac", "-b:a", "192k", "-t", str(duration), str(encoded)],
        check=True, capture_output=True, timeout=30)
    result = subprocess.run(["ffmpeg", "-v", "error", "-i", str(encoded), "-ac", "1", "-ar", str(rate),
        "-f", "f32le", "-"], check=True, capture_output=True, timeout=30)
    decoded = np.frombuffer(result.stdout, dtype=np.float32)
    # MP4/AAC end trimming may round by 1 ms; it must not lose a ~55 ms block.
    assert len(decoded) >= count - rate // 1000
    # Avoid AAC's final transform-window taper; the 40–20 ms voice window must
    # still be present, not replaced with silence to hide the old ~55 ms loss.
    assert np.corrcoef(decoded[count-1920:count-960], samples[count-1920:count-960])[0, 1] > .95


@pytest.mark.parametrize("delivery", ["auto_upload", "auto_sync"])
def test_local_stock_blocks_delivery_before_paid_processing(tmp_path, monkeypatch, delivery):
    from automation import orchestrator
    manifest, _ = review_library(tmp_path)
    monkeypatch.setattr(orchestrator, "_CONFIG", {"export": {"composition": "cricket-gameplay-stadium-v1"}})
    monkeypatch.setattr("download.download", lambda *a, **k: pytest.fail("No media work before delivery hold"))
    result = orchestrator.run("https://youtu.be/test", gameplay_manifest=str(manifest), **{delivery: True})
    assert not result.exported
    assert any("draft rendering only" in message for message in result.failures)


def test_upload_cannot_bypass_stock_hold_by_disabling_release_checks(tmp_path, monkeypatch):
    import upload
    manifest, _ = review_library(tmp_path)
    selected = select_gameplay(load_render_library(manifest), 15, "source:1")
    clip = tmp_path / "clip.mp4"
    clip.with_name("clip_gameplay.json").write_text(json.dumps({"selection": selected}))
    monkeypatch.setitem(upload.cfg, "release_review", {"required": False})
    with pytest.raises(ValueError, match="Local-review gameplay cannot be uploaded"):
        upload._prepare_upload_metadata({"title": "Test", "description": credit_text(selected)}, clip)


def test_approved_libraries_keep_the_existing_route(tmp_path):
    from tests.test_cricket_gameplay import library
    manifest, _ = library(tmp_path)
    assert load_render_library(manifest) == load_library(manifest)


def test_local_review_draft_can_be_created_but_not_released(tmp_path):
    from PIL import Image
    from automation.release_review import create_review, release_status, _artifacts
    manifest, _ = review_library(tmp_path)
    selected = select_gameplay(load_render_library(manifest), 15, "source:1")
    clip, source = tmp_path / "clip.mp4", tmp_path / "source.mp4"
    clip.write_bytes(b"SYNTHETIC local export")
    source.write_bytes(b"SYNTHETIC voice source")
    Image.new("RGB", (1080, 1920)).save(tmp_path / "clip_thumb.jpg")
    clip.with_name("clip_metadata.json").write_text(json.dumps({"title": "Test", "description": credit_text(selected)}))
    clip.with_name("clip_qc.json").write_text(json.dumps({"status": "PASS", "media_sha256": sha256(clip)}))
    clip.with_name("clip_gameplay.json").write_text(json.dumps({"selection": selected, "media_sha256": sha256(clip)}))
    draft = create_review(clip, source)
    assert "gameplay_license_evidence" in draft["artifacts"]
    assert "gameplay_publisher_permission" not in draft["artifacts"]
    # A review-only library can be drafted but its rights check FAILs for delivery.
    check = draft["checks"]["gameplay_rights_and_attribution"]
    assert check["status"] == "FAIL" and "draft rendering only" in check["evidence"]
    assert release_status(clip)[0] != "READY"
    with pytest.raises(ValueError, match="draft rendering only"):
        _artifacts(clip, source)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="Real FFmpeg required")
def test_real_local_stock_stadium_pair_keeps_only_source_audio(tmp_path):
    import numpy as np
    from automation.gameplay_export import render_gameplay_clip
    source, stock = tmp_path / "source.mp4", tmp_path / "stock.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
        "color=c=red:s=640x360:r=30:d=8", "-f", "lavfi", "-i", "sine=frequency=440:duration=8",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", str(source)], check=True, capture_output=True, timeout=30)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
        "color=c=blue:s=640x360:r=30:d=8", "-an", "-c:v", "libx264", str(stock)],
        check=True, capture_output=True, timeout=30)
    manifest, payload = review_library(tmp_path, stock)
    payload["assets"][0].update(approved_ranges=[[0, 8]], crop=[0, 0, 640, 360])
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    config = {"paths": {"temp": str(tmp_path / "scratch")}, "export": {
        "composition": "cricket-gameplay-stadium-v1", "gameplay_manifest": str(manifest),
        "encoder_preset": "ultrafast", "captions": {"enabled": True, "require_word_timing": True}}}
    starts = []
    for number, start in enumerate((0, 4), 1):
        words = [{"word": word, "start": start + i * .7 + .1, "end": start + i * .7 + .65,
                  "probability": .99} for i, word in enumerate(("India", "cricket", "batting", "wins"))]
        transcript = [{"start": start, "end": start + 3, "text": "India cricket batting wins", "words": words}]
        analysis = {"source_dimensions": {"width": 640, "height": 360}}
        output = tmp_path / f"clip{number}.mp4"
        render_gameplay_clip(str(source), start, start + 3, output, analysis, transcript, config)
        evidence = json.loads(output.with_name(output.stem + "_gameplay.json").read_text())
        assert evidence["selection"]["asset"]["rights_status"] == "UNRESOLVED"
        assert evidence["media_sha256"] == sha256(output)
        assert evidence["top"][2:] == evidence["bottom"][2:] == [992, 558]
        assert analysis["edit_timeline"][0]["speed"] == 1
        assert json.loads(output.with_name(output.stem + "_qc.json").read_text())["status"] == "PASS"
        starts.append(evidence["selection"]["start"])
        audio = subprocess.run(["ffmpeg", "-v", "error", "-i", str(output), "-f", "f32le",
            "-ac", "1", "-ar", "8000", "-"], check=True, capture_output=True, timeout=20)
        samples = np.frombuffer(audio.stdout, dtype=np.float32)
        spectrum = abs(np.fft.rfft(samples[4000:20000]))
        assert np.fft.rfftfreq(16000, 1 / 8000)[spectrum.argmax()] == pytest.approx(440, abs=2)
    assert starts[0] != starts[1]
