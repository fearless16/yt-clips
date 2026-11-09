"""Rights, real encoded pixels/audio, and isolated opt-in gameplay routing."""

import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def library(tmp_path, media=None):
    media = media or tmp_path / "game.mp4"
    if not media.exists():
        media.write_bytes(b"synthetic recording, not licensed production media")
    policy = tmp_path / "policy.txt"
    policy.write_text("Synthetic rights evidence, TEST ONLY", encoding="utf-8")
    permission = tmp_path / "permission.txt"
    permission.write_text("Synthetic recording permission, TEST ONLY", encoding="utf-8")
    payload = {"schema_version": 1, "assets": [{
        "id": "fixture", "path": media.name, "sha256": digest(media),
        "game": "EA Cricket 07", "publisher": "EA", "modified": False,
        "status": "APPROVED", "reviewer": "test fixture", "reviewed_at": "2026-11-07",
        "publisher_permission": {"kind": "publisher_policy", "commercial_youtube": True,
            "editing": True, "split_screen_background": True, "no_third_party_promotion": True,
            "url": "https://help.ea.com/en/articles/security-and-rules/ea-content-policy/",
            "evidence": policy.name, "sha256": digest(policy)},
        "recording_permission": {"kind": "self_recorded", "owner": "fixture owner",
            "commercial_youtube": True, "editing": True, "split_screen_background": True,
            "evidence": permission.name, "sha256": digest(permission)},
        "third_party_visuals_cleared": True, "attribution": "Gameplay: fixture owner",
        "approved_ranges": [[0, 20]],
    }]}
    manifest = tmp_path / "library.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest, payload


def test_library_requires_both_layers_and_exact_evidence(tmp_path):
    from automation.cricket_gameplay import load_library, select_gameplay
    manifest, payload = library(tmp_path)
    approved = load_library(manifest)
    a = select_gameplay(approved, 3, "source:10:13")
    assert a == select_gameplay(approved, 3, "source:10:13")
    assert 0 <= a["start"] < a["end"] <= 20
    assert a["end"] - a["start"] == pytest.approx(3)
    assert a["asset"]["sha256"] == payload["assets"][0]["sha256"]
    (tmp_path / "permission.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_library(manifest)


@pytest.mark.parametrize("change", [
    {"status": "WITHDRAWN"}, {"modified": True}, {"sha256": "0" * 64},
    {"third_party_visuals_cleared": False}, {"reviewer": ""},
    {"approved_ranges": [[0, 0]]}, {"approved_ranges": [[0, float("nan")]]},
    {"approved_ranges": [[0, 10], [9, 20]]},
    {"publisher_permission": {}}, {"recording_permission": {}},
    {"publisher": "Nautilus", "game": "Real Cricket 24"},
])
def test_uncleared_or_changed_assets_fail_closed(tmp_path, change):
    from automation.cricket_gameplay import load_library
    manifest, payload = library(tmp_path)
    payload["assets"][0].update(change)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_library(manifest)


def test_rc24_needs_written_publisher_permission_and_exact_disclaimer(tmp_path):
    from automation.cricket_gameplay import load_library, select_gameplay, RC24_DISCLAIMER
    manifest, payload = library(tmp_path)
    asset = payload["assets"][0]
    asset.update(publisher="Nautilus", game="Real Cricket 24")
    asset["publisher_permission"].update(kind="written_authorization")
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    selected = select_gameplay(load_library(manifest), 3, "moment")
    assert selected["disclaimer"] == RC24_DISCLAIMER


def test_ea_policy_requires_review_of_background_use_and_brand_restrictions(tmp_path):
    from automation.cricket_gameplay import load_library, select_gameplay, EA_DISCLAIMER
    manifest, payload = library(tmp_path)
    assert select_gameplay(load_library(manifest), 3, "moment")["disclaimer"] == EA_DISCLAIMER
    for field in ("split_screen_background", "no_third_party_promotion"):
        changed = copy.deepcopy(payload)
        changed["assets"][0]["publisher_permission"][field] = False
        manifest.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError):
            load_library(manifest)


def test_missing_or_short_library_never_downloads_or_loops(tmp_path):
    from automation.cricket_gameplay import load_library, select_gameplay
    manifest, _ = library(tmp_path)
    with pytest.raises(ValueError, match="continuous"):
        select_gameplay(load_library(manifest), 21, "moment")
    with pytest.raises(ValueError, match="manifest"):
        load_library(tmp_path / "missing.json")


def test_source_crop_falls_back_to_full_frame_if_a_face_would_be_cut():
    from automation.cricket_gameplay import source_crop
    analysis = {"source_dimensions": {"width": 1920, "height": 1080},
        "export_strategy": {"active_crop": {"x": 200, "y": 0, "width": 600, "height": 1080,
            "face_x": 400, "face_y": 300, "face_w": 200, "face_h": 400}}}
    assert source_crop(analysis) == [200, 0, 600, 1080]
    analysis["face_samples"] = [{"face_x": 700, "face_y": 300, "face_w": 200, "face_h": 400}]
    assert source_crop(analysis) == [0, 0, 1920, 1080]
    analysis["export_strategy"]["guest_cam_on"] = True
    assert source_crop(analysis) == [0, 0, 1920, 1080]


def test_gameplay_caption_raster_stays_in_center_band(tmp_path):
    from automation.caption_engine import CaptionCue, render_transparent_cards_batch
    from PIL import Image
    cards = render_transparent_cards_batch([CaptionCue(0, 1, "Rohit ki batting", "", [])],
        tmp_path, style="gameplay-hud-v1")
    design = json.loads((tmp_path / "caption_design.json").read_text())
    assert design["design"] == "gameplay-hud-v1"
    assert all(40 <= row["y"] and row["bottom"] <= 280 for row in design["cards"][0]["lines"])
    assert "Rohit" in " ".join(row["text"] for row in design["cards"][0]["lines"])
    with Image.open(cards[0].path) as image:
        assert image.size == (1080, 320)
        box = image.getchannel("A").getbbox()
        assert box[1] >= 40 and box[3] <= 280


def test_opt_in_does_not_mutate_default_export_config(monkeypatch, tmp_path):
    import export
    from automation.cricket_gameplay import COMPOSITION
    before = copy.deepcopy(export.cfg)
    monkeypatch.setattr(export, "_get_video_info", lambda *_: {"width": 640, "height": 360, "fps": 30})
    monkeypatch.setattr(export, "_has_audio_stream", lambda *_: True)
    monkeypatch.setattr("automation.gameplay_export.render_gameplay_clip", lambda *a, **k: "gameplay.mp4")
    result = export.export_clip("source.mp4", 1, 4, str(tmp_path / "out.mp4"),
        analysis={"composition": COMPOSITION, "export_strategy": {}})
    assert result == "gameplay.mp4"
    assert export.cfg == before


def test_variant_forces_word_evidence_even_if_global_captions_are_disabled(monkeypatch):
    import export
    from automation.cricket_gameplay import COMPOSITION
    monkeypatch.setitem(export.cfg["export"], "captions", {"enabled": False})
    options, enabled, required = export._caption_options({"composition": COMPOSITION})
    assert enabled and required
    assert options["style"] == "gameplay-hud-v1"


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="Real FFmpeg required")
def test_real_encoded_panels_keep_source_audio_and_full_ending(tmp_path):
    import numpy as np
    from PIL import Image
    from automation.caption_engine import CaptionCue, render_transparent_cards_batch
    from automation.cricket_gameplay import load_library, select_gameplay
    from automation.gameplay_export import build_command

    source, gameplay = tmp_path / "source.mp4", tmp_path / "game.mp4"
    for path, color, frequency, seconds in ((source, "red", 440, 3), (gameplay, "blue", 1200, 1)):
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
            f"color=c={color}:s=640x360:r=30:d={seconds}", "-f", "lavfi", "-i",
            f"sine=frequency={frequency}:duration={seconds}", "-c:v", "libx264", "-c:a", "aac",
            "-shortest", str(path)], check=True, capture_output=True, timeout=30)
    # The file contains only 1s although the manifest declares 20s. The renderer
    # must reject that before encoding, not freeze/loop silently or shorten audio.
    manifest, payload = library(tmp_path, gameplay)
    selection = select_gameplay(load_library(manifest), 3, "moment")
    with pytest.raises(ValueError, match="duration"):
        build_command(str(source), 0, 3, str(tmp_path/"bad.mp4"), selection,
                      {"source_dimensions": {"width": 640, "height": 360}}, [], tmp_path, {})

    gameplay.unlink()
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
        "color=c=blue:s=640x360:r=30:d=5", "-f", "lavfi", "-i", "sine=frequency=1200:duration=5",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", str(gameplay)],
        check=True, capture_output=True, timeout=30)
    payload["assets"][0].update(sha256=digest(gameplay), approved_ranges=[[0, 5]])
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    selection = select_gameplay(load_library(manifest), 3, "moment")
    cards = render_transparent_cards_batch([CaptionCue(0, 3, "Rohit ki batting", "", [])],
        tmp_path/"cards", style="gameplay-hud-v1")
    target = tmp_path/"actual.mp4"
    cmd, evidence = build_command(str(source), 0, 3, str(target), selection,
        {"source_dimensions": {"width": 640, "height": 360}}, cards, tmp_path,
        {"fps": 30, "encoder_preset": "ultrafast"})
    assert "-shortest" not in cmd
    assert "fade=" not in " ".join(cmd) and "afade=" not in " ".join(cmd)
    subprocess.run(cmd, capture_output=True, check=True, timeout=90)
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(target), "-f", "null", "-"],
        capture_output=True, check=True, timeout=30)
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(target)],
        capture_output=True, text=True, check=True, timeout=10)
    streams = json.loads(probe.stdout)["streams"]
    assert len([s for s in streams if s["codec_type"] == "audio"]) == 1
    assert all(abs(float(s["duration"]) - 3) <= .08 for s in streams)
    frame = tmp_path/"frame.png"
    subprocess.run(["ffmpeg", "-v", "error", "-ss", "2.9", "-i", str(target), "-frames:v", "1", str(frame)],
        capture_output=True, check=True, timeout=15)
    with Image.open(frame) as image:
        pixels = np.asarray(image.convert("RGB"))
    assert pixels[420, 540, 0] > 180 and pixels[420, 540, 2] < 60
    assert pixels[1500, 540, 2] > 180 and pixels[1500, 540, 0] < 60
    assert (pixels[850:1070, :, :] > 220).any()
    audio = subprocess.run(["ffmpeg", "-v", "error", "-i", str(target), "-f", "f32le", "-ac", "1",
        "-ar", "8000", "-"], capture_output=True, check=True, timeout=15)
    samples = np.frombuffer(audio.stdout, dtype=np.float32)
    spectrum = abs(np.fft.rfft(samples[4000:20000]))
    frequencies = np.fft.rfftfreq(16000, 1/8000)
    assert frequencies[spectrum.argmax()] == pytest.approx(440, abs=2)
    assert np.sqrt(np.mean(samples[-1600:]**2)) > .02  # final speech not faded away
    assert evidence["audio"] == "source input 0 only; no fades; natural speed"


def test_distinct_count_does_not_count_visual_variants_twice(tmp_path):
    from automation.cricket_gameplay import count_distinct_exports
    clips = []
    for name, start, end in (("one", 10, 30), ("variant", 10, 30), ("two", 100, 120)):
        path = tmp_path / (name+".mp4")
        path.write_bytes(b"fixture")
        path.with_name(name+"_edit.json").write_text(json.dumps({
            "source_path": "source.mp4", "source_start": start, "source_end": end}))
        clips.append(path)
    assert count_distinct_exports(clips) == 2


def test_cli_exposes_explicit_profile_and_manifest():
    from automation.cli import setup_argparse
    args = setup_argparse().parse_args(["https://youtu.be/H8OjNZ8XLbI", "--composition",
        "cricket-gameplay-split-v1", "--gameplay-manifest", "input/gameplay/library.json"])
    assert args.composition == "cricket-gameplay-split-v1"
    assert args.gameplay_manifest == "input/gameplay/library.json"


def test_missing_recording_stops_before_paid_work(monkeypatch):
    from automation import orchestrator
    monkeypatch.setattr(orchestrator, "_cfg", lambda: {"export": {}, "clip_selection": {}})
    monkeypatch.setattr("download.download", lambda *a, **k: pytest.fail("must not download before rights preflight"))
    result = orchestrator.run("https://youtu.be/H8OjNZ8XLbI", composition="cricket-gameplay-split-v1")
    assert not result.exported
    assert any("gameplay preflight" in error for error in result.failures)


def test_gameplay_review_cannot_reuse_portrait_checks_or_ignore_withdrawal(tmp_path):
    from automation.cricket_gameplay import load_library, select_gameplay, credit_text
    from automation.release_review import _artifacts, required_checks
    from PIL import Image
    manifest, payload = library(tmp_path)
    selected = select_gameplay(load_library(manifest), 3, "moment")
    clip, source = tmp_path/"clip.mp4", tmp_path/"source.mp4"
    clip.write_bytes(b"synthetic export")
    source.write_bytes(b"synthetic source")
    Image.new("RGB", (360, 640)).save(tmp_path/"clip_thumb.jpg")
    (tmp_path/"clip_edit.json").write_text(json.dumps({"composition": "cricket-gameplay-split-v1"}))
    (tmp_path/"clip_qc.json").write_text("{}")
    (tmp_path/"clip_metadata.json").write_text(json.dumps({"title": "fixture", "description": credit_text(selected)}))
    (tmp_path/"clip_gameplay.json").write_text(json.dumps({"selection": selected, "media_sha256": digest(clip)}))
    assert "gameplay_rights_and_attribution" in required_checks(clip)
    artifacts = _artifacts(clip, source)
    assert {"gameplay", "gameplay_manifest", "gameplay_recording", "gameplay_publisher_permission",
            "gameplay_recording_permission"} <= artifacts.keys()
    payload["assets"][0]["status"] = "WITHDRAWN"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="APPROVED"):
        _artifacts(clip, source)


def test_generated_metadata_includes_gameplay_credit_before_review(monkeypatch, tmp_path):
    import export
    from automation.cricket_gameplay import load_library, select_gameplay, credit_text
    manifest, _ = library(tmp_path)
    selected = select_gameplay(load_library(manifest), 3, "moment")
    (tmp_path/"clip1_gameplay.json").write_text(json.dumps({"selection": selected}))
    metadata = {"title": "Rohit batting opinion", "description": "Original source-grounded opinion."}
    monkeypatch.setattr("automation.seo.seo.generate_seo_for_exported_clip", lambda **k: dict(metadata))
    result = export._seo_worker("clip1", {}, tmp_path, {})
    assert result["description"].endswith(credit_text(selected))
    assert json.loads((tmp_path/"clip1_metadata.json").read_text())["description"] == result["description"]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="Real FFmpeg required")
def test_real_two_moment_batch_routes_through_renderer_qc_and_sidecars(tmp_path, monkeypatch):
    import export
    from automation.cricket_gameplay import count_distinct_exports
    from automation.release_review import required_checks
    source, gameplay = tmp_path/"source.mp4", tmp_path/"game.mp4"
    for path, pattern, frequency in ((source, "testsrc2=s=640x360:r=30:d=10", 440),
                                     (gameplay, "testsrc2=s=640x360:r=30:d=10", 1200)):
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", pattern,
            "-f", "lavfi", "-i", f"sine=frequency={frequency}:duration=10", "-c:v", "libx264",
            "-c:a", "aac", "-shortest", str(path)], check=True, capture_output=True, timeout=30)
    manifest, payload = library(tmp_path, gameplay)
    payload["assets"][0]["approved_ranges"] = [[0, 10]]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setitem(export.cfg, "paths", {**export.cfg["paths"], "shorts": str(tmp_path/"shorts"),
                                             "temp": str(tmp_path/"scratch")})
    monkeypatch.setitem(export.cfg, "export", {**export.cfg["export"], "max_workers": 1,
        "encoder_preset": "ultrafast", "captions": {"enabled": True, "require_word_timing": True}})
    monkeypatch.setattr(export, "analyze_clip", lambda *a, **k: {"export_strategy": {}, "layout": {}})
    monkeypatch.setattr(export, "_get_best_encoder", lambda: "libx264")
    # Only frame detection and encoder hardware discovery are replaced. Actual
    # caption rasterization, media encode, decode, audio QC, timing, and sidecars run.
    highlights = {}
    for index, start in enumerate((0, 6), 1):
        words = [{"word": word, "start": start+i*.7+.1, "end": start+i*.7+.65, "probability": .99}
                 for i, word in enumerate(("India", "cricket", "batting", "wins"))]
        highlights[f"clip{index}"] = {"start": start, "end": start+3, "text": "India cricket batting wins",
            "complete_thought": True, "semantic_review": "text_only", "speech_segments": [
                {"start": start, "end": start+3, "text": "India cricket batting wins", "words": words}]}
    clips = export.export_all(highlights, str(source), generate_seo=False,
        composition="cricket-gameplay-split-v1", gameplay_manifest=str(manifest))
    assert len(clips) == count_distinct_exports(clips) == 2
    for clip in clips:
        assert clip.is_file()
        evidence = json.loads(clip.with_name(clip.stem+"_gameplay.json").read_text())
        assert evidence["media_sha256"] == digest(clip)
        assert json.loads(clip.with_name(clip.stem+"_qc.json").read_text())["status"] == "PASS"
        edit = json.loads(clip.with_name(clip.stem+"_edit.json").read_text())
        assert edit["composition"] == "cricket-gameplay-split-v1"
        assert edit["timeline"][0]["speed"] == 1
        assert "gameplay_rights_and_attribution" in required_checks(clip)
        assert not clip.with_name(clip.stem+"_review.json").exists()  # never forge owner approval
