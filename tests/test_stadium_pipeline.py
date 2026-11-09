"""Approved design in the REAL pipeline; synthetic permission/media fixtures only."""

import copy
import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from automation.cricket_gameplay import (STADIUM_COMPOSITION, load_library, select_gameplay,
    sha256, credit_text, append_gameplay_credit)


def fixture_library(tmp_path, media=None, crop=None):
    media = media or tmp_path / "game.mp4"
    if not media.exists():
        media.write_bytes(b"SYNTHETIC TEST ONLY - no production rights")
    permission = tmp_path / "test-only-permission.txt"
    permission.write_text("Synthetic fixture only, no rights or release granted",encoding="utf-8")
    grant = {"kind":"written_authorization","commercial_youtube":True,"editing":True,
        "split_screen_background":True,"evidence":permission.name,"sha256":sha256(permission)}
    asset = {"id":"test-only","path":media.name,"sha256":sha256(media),"game":"EA Cricket 07",
        "publisher":"EA","modified":False,"status":"APPROVED","reviewer":"SYNTHETIC TEST ONLY",
        "reviewed_at":"2026-11-08","publisher_permission":grant,
        "recording_permission":{**grant,"owner":"Synthetic fixture"},"third_party_visuals_cleared":True,
        "attribution":"SYNTHETIC recording credit - not production gameplay","approved_ranges":[[0,8]],
        "crop":crop or [240,128,1440,810]}
    path = tmp_path / "library.json"
    payload = {"schema_version":1,"assets":[asset]}
    path.write_text(json.dumps(payload),encoding="utf-8")
    return path,payload


def test_default_and_cli_select_approved_profile():
    from automation.config import load
    from automation.cli import setup_argparse
    config = load()
    assert config["export"]["composition"] == STADIUM_COMPOSITION
    assert config["export"]["gameplay_manifest"] == "input/gameplay/library-clean-v2.json"
    assert config["release_review"]["required"] is True
    parsed = setup_argparse().parse_args(["https://youtu.be/test","--composition",STADIUM_COMPOSITION])
    assert parsed.composition == STADIUM_COMPOSITION and not parsed.upload


def test_production_overlay_masks_and_font_are_locked():
    from automation.stadium_layout import preflight_assets, OVERLAY, HOST, GAMEPLAY
    design = preflight_assets()
    assert design["overlay"]["sha256"] == sha256(OVERLAY)
    alpha = np.asarray(Image.open(OVERLAY))[:,:,3]
    x,y,w,h = HOST
    gx,gy,gw,gh = GAMEPLAY
    assert (w,h) == (gw,gh) == (992,558) and w*9 == h*16
    assert np.array_equal(alpha[y:y+h,x:x+w],alpha[gy:gy+h,gx:gx+w])


def test_changed_overlay_stops_before_render(monkeypatch,tmp_path):
    from automation import stadium_layout
    fake = tmp_path / "overlay.png"
    fake.write_bytes(b"changed")
    monkeypatch.setattr(stadium_layout,"OVERLAY",fake)
    with pytest.raises(ValueError,match="overlay"):
        stadium_layout.preflight_assets()


def test_missing_runtime_binary_is_preflight_hold(monkeypatch):
    from automation.stadium_layout import preflight_runtime
    monkeypatch.setattr("automation.stadium_layout.shutil.which",lambda *_: None)
    with pytest.raises(ValueError,match="ffmpeg"):
        preflight_runtime()


@pytest.mark.parametrize("crop",[None,[0,0,1952,1098]])
def test_missing_or_out_of_bounds_crop_holds_preflight(tmp_path,monkeypatch,crop):
    from automation.gameplay_export import preflight_library
    manifest,payload = fixture_library(tmp_path)
    if crop is None:
        payload["assets"][0].pop("crop")
    else:
        payload["assets"][0]["crop"] = crop
    manifest.write_text(json.dumps(payload))
    monkeypatch.setattr("automation.gameplay_export._duration",lambda *_:8)
    probe = Mock(return_value=subprocess.CompletedProcess([],0,json.dumps({"streams":[
        {"width":1920,"height":1080,"sample_aspect_ratio":"1:1"}]}),""))
    monkeypatch.setattr("automation.stadium_export.subprocess.run",probe)
    with pytest.raises(ValueError,match="crop"):
        preflight_library(load_library(manifest),STADIUM_COMPOSITION)


@pytest.mark.parametrize("crop",[[1,0,1440,810],[0,0,1440,800],[-2,0,1440,810],[0,0,0,0],"crop"])
def test_manifest_rejects_malformed_or_distorting_crop(tmp_path,crop):
    manifest,payload = fixture_library(tmp_path)
    payload["assets"][0]["crop"] = crop
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match="crop"):
        load_library(manifest)


def test_ea_preference_is_deterministic_and_does_not_override_rights_or_duration(tmp_path):
    manifest,payload = fixture_library(tmp_path)
    other = copy.deepcopy(payload["assets"][0])
    other.update(id="other",game="Other Cricket Game",publisher="Other")
    other["approved_ranges"] = [[0,20]]
    payload["assets"].append(other)
    manifest.write_text(json.dumps(payload))
    library = load_library(manifest)
    first = select_gameplay(library,2,"one",preferred_game="EA Cricket 07")
    assert first == select_gameplay(library,2,"one",preferred_game="EA Cricket 07")
    assert first["asset"]["game"] == "EA Cricket 07"
    assert first != select_gameplay(library,2,"two",preferred_game="EA Cricket 07")
    assert select_gameplay(library,10,"one",preferred_game="EA Cricket 07")["asset"]["game"] == "Other Cricket Game"


def test_stadium_forces_its_caption_style_without_mutating_config(monkeypatch):
    import export
    before = copy.deepcopy(export.cfg)
    options,enabled,required = export._caption_options({"composition":STADIUM_COMPOSITION})
    assert enabled and required and options["style"] == "stadium-reference-v1"
    assert before == export.cfg


def test_explicit_portrait_filter_uses_analysis_profile_not_default(monkeypatch):
    import export
    monkeypatch.setitem(export.cfg["export"],"composition",STADIUM_COMPOSITION)
    analysis = {"source_dimensions":{"width":2560,"height":1440},"export_strategy":{
        "is_screen_share":True,"use_solo_frame":False,"active_crop":{
            "x":852,"y":0,"width":810,"height":1440,
            "face_x":967,"face_y":454,"face_w":580,"face_h":829}}}
    analysis["composition"] = "portrait-caption-v1"
    assert "[portrait_sharp]scale=1080:" in export._build_enhance_stack(analysis,use_logo=False)


def test_real_caption_font_bounds_and_timing_grouping(tmp_path):
    from automation.caption_engine import prepare_clip_captions
    words = [{"word":w,"roman":w,"start":i*.5,"end":i*.5+.4}
             for i,w in enumerate(["Partnership","batting","India","cricket"])]
    cards = prepare_clip_captions([{"start":0,"end":2,"words":words}],0,2,tmp_path,
                                  require_word_timing=True,caption_style="stadium-reference-v1")
    assert " ".join(c.text for c in cards) == " ".join(w["word"] for w in words)
    assert all(c.start in [w["start"] for w in words] and c.end in [w["end"] for w in words] for c in cards)
    evidence = json.loads((tmp_path/"caption_design.json").read_text())
    assert evidence["design"] == "stadium-reference-v1"
    for card in evidence["cards"]:
        assert card["font_size"] == 96
        for row in card["lines"]:
            assert 120 <= row["x"] and row["right"] <= 960
            assert 53 <= row["y"] and row["bottom"] <= 267
            assert abs(row["center_x"]-540) < .1


def test_oversized_word_is_held_not_dropped_or_shrunk(tmp_path):
    from automation.caption_engine import prepare_clip_captions
    word = "W"*29
    with pytest.raises(ValueError,match="word exceeds"):
        prepare_clip_captions([{"start":0,"end":1,"words":[{"word":word,"roman":word,"start":0,"end":1}]}],
            0,1,tmp_path,require_word_timing=True,caption_style="stadium-reference-v1")


def test_missing_production_manifest_fails_before_any_download(monkeypatch,tmp_path):
    from automation import orchestrator
    monkeypatch.setattr(orchestrator,"_CONFIG",{"export":{"composition":STADIUM_COMPOSITION,
                         "gameplay_manifest":str(tmp_path/"missing.json")}})
    download = Mock(side_effect=AssertionError("No paid/network work allowed"))
    monkeypatch.setattr("download.download",download)
    result = orchestrator.run("https://youtu.be/test")
    assert result.failures and "gameplay preflight" in result.failures[0]
    download.assert_not_called()


def test_post_render_enhancement_is_rejected_before_mutation(monkeypatch):
    from automation import orchestrator
    monkeypatch.setattr(orchestrator,"_CONFIG",{"export":{"composition":STADIUM_COMPOSITION}})
    result = orchestrator.run("https://youtu.be/test",mode="ref_grade")
    assert any("post-render enhancement" in e for e in result.failures)


def test_unknown_profile_cannot_silently_fall_back(monkeypatch):
    from automation import orchestrator
    monkeypatch.setattr(orchestrator,"_CONFIG",{"export":{"composition":"typo-layout"}})
    result = orchestrator.run("https://youtu.be/test")
    assert result.failures == ["Unknown composition: typo-layout"]


def test_missing_edit_cannot_hide_gameplay_release_checks(tmp_path):
    from automation.release_review import required_checks
    clip = tmp_path/"clip.mp4"
    clip.with_name("clip_gameplay.json").write_text(json.dumps({"composition":STADIUM_COMPOSITION}))
    assert "gameplay_rights_and_attribution" in required_checks(clip)


def test_explicit_portrait_override_is_forwarded(monkeypatch,tmp_path):
    from automation import orchestrator
    monkeypatch.setattr(orchestrator,"_CONFIG",{"export":{"composition":STADIUM_COMPOSITION},
        "paths":{"shorts":str(tmp_path)},"thumbnail":{"enabled":False}})
    render = Mock(return_value=[])
    monkeypatch.setattr("export.export_all",render)
    monkeypatch.setattr("automation.seo.analytics.Analytics.get_summary",lambda *_:{})
    monkeypatch.setattr("shorts_intelligence.bridge.shadow_status",lambda *_:{})
    result = orchestrator.run("https://youtu.be/test",skip_download=True,skip_transcribe=True,
        skip_highlight=True,skip_seo=True,composition="portrait-caption-v1")
    assert not result.failures
    assert render.call_args.kwargs["composition"] == "portrait-caption-v1"


def test_private_finalization_package_is_not_a_resumable_production_batch(tmp_path):
    from automation.orchestrator import _existing_exports
    batch = tmp_path/"final"
    batch.mkdir()
    (batch/"clip.mp4").write_bytes(b"private final layout")
    (batch/"finalization.json").write_text('{"publication_authorized":false}')
    assert _existing_exports(str(tmp_path)) == []
    with pytest.raises(ValueError,match="not a production"):
        _existing_exports(str(tmp_path),str(batch))


def test_attribution_is_idempotent_and_overflow_is_not_truncated(tmp_path):
    manifest,_ = fixture_library(tmp_path)
    selected = select_gameplay(load_library(manifest),2,"moment")
    sidecar = tmp_path/"clip_gameplay.json"
    sidecar.write_text(json.dumps({"selection":selected}))
    result = append_gameplay_credit({"description":"Source-grounded cricket opinion"},sidecar)
    assert result == append_gameplay_credit(result,sidecar)
    assert result["description"].endswith(credit_text(selected))
    with pytest.raises(ValueError,match="attribution exceeds"):
        append_gameplay_credit({"description":"x"*4800},sidecar)


def test_seo_retry_retains_gameplay_attribution(tmp_path,monkeypatch):
    from automation.seo import seo
    manifest,_ = fixture_library(tmp_path)
    selected = select_gameplay(load_library(manifest),2,"moment")
    (tmp_path/"clip_gameplay.json").write_text(json.dumps({"selection":selected}))
    (tmp_path/"clip_seo_failed.json").write_text('{"transcript":"India cricket"}')
    monkeypatch.setattr(seo,"generate_clip_seo",lambda **_: {"title":"India cricket","description":"Retained opinion","cover_text":"INDIA CRICKET"})
    assert seo.retry_failed_seo(str(tmp_path)) == {"recovered":1,"total":1}
    assert json.loads((tmp_path/"clip_metadata.json").read_text())["description"].endswith(credit_text(selected))


def test_default_dry_run_really_exercises_stadium_evidence():
    import dry_run
    result = dry_run.dry_run("https://youtu.be/test",skip_download=True,skip_transcribe=True)
    assert not result["failures"],result["failures"]
    assert result["validation"]["stadium_exports"] == len(result["exported"]) == 3
    assert result["uploaded_count"] == 0


@pytest.mark.skipif(not shutil.which("ffmpeg"),reason="Real media integration requires FFmpeg")
def test_real_two_moment_production_batch_crops_panels_audio_qc_and_release(tmp_path,monkeypatch):
    import export
    from automation.cricket_gameplay import count_distinct_exports
    from automation.release_review import required_checks, _artifacts
    from automation.stadium_layout import preflight_assets
    source,game = tmp_path/"source.mp4",tmp_path/"game.mp4"
    patterns = [(source,"color=c=red:s=640x360:r=30:d=8",440),
        (game,"color=c=magenta:s=1920x1080:r=30:d=8,drawbox=x=240:y=128:w=1440:h=810:color=blue:t=fill",1200)]
    for path,pattern,freq in patterns:
        subprocess.run(["ffmpeg","-nostdin","-v","error","-f","lavfi","-i",pattern,
            "-f","lavfi","-i",f"sine=frequency={freq}:duration=8","-c:v","libx264","-c:a","aac",
            "-shortest",str(path)],check=True,capture_output=True,timeout=60)
    manifest,_ = fixture_library(tmp_path,game)
    monkeypatch.setitem(export.cfg,"paths",{**export.cfg["paths"],"shorts":str(tmp_path/"shorts"),"temp":str(tmp_path/"scratch")})
    monkeypatch.setitem(export.cfg,"export",{**export.cfg["export"],"composition":STADIUM_COMPOSITION,
        "max_workers":1,"encoder_preset":"ultrafast","captions":{"enabled":True,"require_word_timing":True}})
    monkeypatch.setattr(export,"analyze_clip",lambda *a,**k:{"export_strategy":{},"layout":{}})
    monkeypatch.setattr(export,"_get_best_encoder",lambda:"libx264")
    highlights = {}
    for i,start in enumerate((0,4),1):
        words = [{"word":w,"roman":w,"start":start+j*.5+.05,"end":start+j*.5+.45,"probability":.99}
                 for j,w in enumerate(("India","cricket","batting","wins"))]
        highlights[f"clip{i}"] = {"start":start,"end":start+2,"text":"India cricket batting wins",
            "complete_thought":True,"semantic_review":"text_only","speech_segments":[
                {"start":start,"end":start+2,"text":"India cricket batting wins","words":words}]}
    before = copy.deepcopy(export.cfg)
    clips = export.export_all(highlights,str(source),generate_seo=False,composition=STADIUM_COMPOSITION,
                              gameplay_manifest=str(manifest))
    assert len(clips) == count_distinct_exports(clips) == 2 and export.cfg == before
    for clip in clips:
        evidence = json.loads(clip.with_name(clip.stem+"_gameplay.json").read_text())
        assert evidence["media_sha256"] == sha256(clip) and evidence["composition"] == STADIUM_COMPOSITION
        assert evidence["gameplay_crop"] == [240,128,1440,810]
        assert evidence["design_assets"] == preflight_assets()
        assert evidence["top"][2:] == evidence["bottom"][2:] == [992,558]
        assert json.loads(clip.with_name(clip.stem+"_qc.json").read_text())["status"] == "PASS"
        edit = json.loads(clip.with_name(clip.stem+"_edit.json").read_text())
        assert edit["composition"] == STADIUM_COMPOSITION and edit["timeline"][0]["speed"] == 1
        assert "gameplay_rights_and_attribution" in required_checks(clip)
        assert not clip.with_name(clip.stem+"_review.json").exists()
        frame = subprocess.run(["ffmpeg","-v","error","-ss","1","-i",str(clip),"-frames:v","1",
            "-f","rawvideo","-pix_fmt","rgb24","-"],capture_output=True,check=True,timeout=20)
        pixels = np.frombuffer(frame.stdout,dtype=np.uint8).reshape(1920,1080,3)
        assert pixels[500,540,0] > 200 and pixels[500,540,2] < 50  # whole host, not portrait zoom
        for px in (80,540,1000):  # magenta crop exterior/pillarboxes excluded, panel filled blue
            assert pixels[1500,px,2] > 200 and pixels[1500,px,0] < 50
        assert (pixels[950:1145] > 220).any()  # actual accepted caption ink
        audio = subprocess.run(["ffmpeg","-v","error","-i",str(clip),"-f","f32le","-ac","1","-ar","8000","-"],
                                capture_output=True,check=True,timeout=20)
        samples = np.frombuffer(audio.stdout,dtype=np.float32)
        segment = samples[2000:10000]
        freq = np.fft.rfftfreq(len(segment),1/8000)[abs(np.fft.rfft(segment)).argmax()]
        assert freq == pytest.approx(440,abs=2)  # no 1200Hz gameplay audio
        Image.fromarray(pixels).save(clip.with_suffix(".png"))
        # Prepare only fixture artifacts, never owner approval. Verify artwork/font
        # become additional hash-bound release dependencies and tampering holds.
        clip.with_name(clip.stem+"_metadata.json").write_text(json.dumps({"title":"India cricket",
            "description":credit_text(evidence["selection"]),"cover_text":"INDIA CRICKET"}))
        Image.new("RGB",(360,640)).save(clip.with_name(clip.stem+"_thumb.jpg"))
        bound = _artifacts(clip,source)
        assert {"stadium_overlay","stadium_font","gameplay_manifest"} <= bound.keys()
        changed = copy.deepcopy(evidence)
        changed["design_assets"]["overlay"]["sha256"] = "0"*64
        clip.with_name(clip.stem+"_gameplay.json").write_text(json.dumps(changed))
        with pytest.raises(ValueError,match="artwork/font/crop"):
            _artifacts(clip,source)
