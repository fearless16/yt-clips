"""Reference composition: real portrait, not a landscape panel inside blur."""

import json
import shutil
import subprocess

import pytest

import export


def recorded_analysis():
    return {"source_dimensions": {"width": 2560, "height": 1440}, "export_strategy": {
        "is_screen_share": True, "use_solo_frame": False,
        "active_crop": {"x": 852, "y": 0, "width": 810, "height": 1440,
                        "face_x": 967, "face_y": 454, "face_w": 580, "face_h": 829}}}


def test_recorded_dominant_host_cannot_be_rendered_as_screen_share(monkeypatch):
    monkeypatch.setitem(export.cfg["export"], "composition", "portrait-caption-v1")
    graph = export._build_enhance_stack(recorded_analysis(), use_logo=False)
    assert "screen_fit" not in graph
    assert "portrait_fit" not in graph  # no tiny floating camera panel
    assert "[portrait_sharp]scale=1080:" in graph
    assert "[portrait_canvas][portrait_picture]overlay=0:" in graph


def test_plan_keeps_face_below_hair_and_above_caption_and_excludes_chat():
    from automation.portrait_layout import plan_portrait
    plan = plan_portrait(recorded_analysis(), 1080, 1920)
    x, y, w, h = plan["crop"]
    assert x <= 967 and x+w >= 1547
    assert x+w < 1890  # actual right-side chat begins here in this source
    assert y == 0  # hair already touches source top; never crop more off
    assert y+h < 1400  # bottom stream controls must not survive
    assert plan["face_bottom"] + 60 <= plan["caption_top"]
    assert 0.5 <= 580 / w <= 0.65
    assert plan["picture_width"] == 1080
    assert plan["caption_top"] == 1484


def test_small_facecam_does_not_override_real_screen_share():
    from automation.portrait_layout import plan_portrait
    analysis = recorded_analysis()
    analysis["export_strategy"]["active_crop"].update(face_h=100, face_w=80)
    assert plan_portrait(analysis, 1080, 1920) is None


def test_worst_sampled_chin_is_preserved_not_just_best_confidence_frame():
    from automation.portrait_layout import plan_portrait
    analysis = recorded_analysis()
    analysis["face_samples"] = [
        {"face_x": 989, "face_y": 553, "face_w": 608, "face_h": 855},
        {"face_x": 936, "face_y": 436, "face_w": 583, "face_h": 806}]
    plan = plan_portrait(analysis)
    assert plan["crop"][3] >= 1408  # previous trim literally cut the beard off
    assert plan["face_bottom"] + 60 <= plan["caption_top"]


def test_reviewed_camera_bounds_exclude_baked_header_and_chat_without_cutting_face():
    from automation.portrait_layout import plan_portrait
    analysis = recorded_analysis()
    analysis["portrait_source_bounds"] = {"x": 764, "y": 0, "width": 1120, "height": 1440}
    plan = plan_portrait(analysis)
    x, y, w, h = plan["crop"]
    assert x >= 764 and x+w <= 1884
    assert x <= 967 and x+w >= 1547
    analysis["portrait_source_bounds"]["width"] = 300
    with pytest.raises(ValueError, match="bounds"):
        plan_portrait(analysis)


def test_real_reference_caption_raster_is_white_rounded_and_within_lower_safe_area(tmp_path):
    from PIL import Image
    from automation.caption_engine import CaptionCue, render_transparent_cards_batch
    cards = render_transparent_cards_batch(
        [CaptionCue(0, 1, "JEETNE KI KOSHISH", "", [])], tmp_path,
        style="portrait-glass-v1")
    evidence = json.loads((tmp_path / "caption_design.json").read_text())
    assert evidence["design"] == "portrait-glass-v1"
    bounds = evidence["cards"][0]
    assert bounds["x"] == 98 and bounds["y"] == 32
    assert bounds["width"] == 884 and bounds["height"] == 224
    assert [r["text"] for r in bounds["lines"]] == ["JEETNE KI", "KOSHISH"]
    assert all(120 <= r["x"] < r["right"] <= 960 for r in bounds["lines"])
    with Image.open(cards[0].path) as image:
        assert image.size == (1080, 320)
        assert image.getpixel((0, 0))[3] == 0
        assert image.getpixel((100, 34))[3] == 0  # genuinely rounded corner


def test_reference_caption_position_is_shared_with_crop_planner(monkeypatch):
    monkeypatch.setitem(export.cfg["export"]["captions"], "style", "portrait-glass-v1")
    assert export._caption_card_y() == 1452  # visible panel starts at 1484


def test_offline_review_preserves_cached_words_and_times_and_rejects_invalid_spans():
    from tools.portrait_review import cached_cues
    original = {"cards": [{"start": 0, "end": 1, "text": "actual retained words"}]}
    cues = cached_cues(original, 2)
    assert (cues[0].text, cues[0].start, cues[0].end) == ("actual retained words", 0, 1)
    with pytest.raises(ValueError, match="cached"):
        cached_cues(original, .5)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="Real FFmpeg required")
def test_actual_ffmpeg_pixels_keep_portrait_large_and_caption_off_face(tmp_path, monkeypatch):
    import numpy as np
    from PIL import Image
    from automation.caption_engine import CaptionCue, render_transparent_cards_batch
    monkeypatch.setitem(export.cfg["export"], "composition", "portrait-caption-v1")
    monkeypatch.setitem(export.cfg["export"]["captions"], "style", "portrait-glass-v1")
    analysis = recorded_analysis()
    graph = export._build_enhance_stack(analysis, use_logo=False)
    card = render_transparent_cards_batch([CaptionCue(0, 1, "JEETNE KI KOSHISH", "", [])],
                                          tmp_path/"cards", style="portrait-glass-v1")[0]
    graph = "[0:v]null[v_src];"+graph+f";[v_out][1:v]overlay=0:{export._caption_card_y()}[final]"
    target = tmp_path/"frame.png"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
        "color=c=blue:s=2560x1440,drawbox=x=967:y=454:w=580:h=829:color=red:t=fill",
        "-i", card.path, "-filter_complex", graph, "-map", "[final]", "-frames:v", "1", str(target)],
        capture_output=True, check=True, timeout=30)
    with Image.open(target) as image:
        assert image.size == (1080, 1920)
        pixels = np.asarray(image.convert("RGB"))
    # Pixel-level assertions detect the old landscape screen-share/inset route,
    # even if its graph labels or counters are renamed to look correct.
    ys, xs = np.where((pixels[:,:,0] > 180) & (pixels[:,:,1] < 80) & (pixels[:,:,2] < 80))
    assert 500 <= xs.max()-xs.min() <= 700
    assert ys.max()+60 < analysis["portrait_layout"]["caption_top"]
    assert (pixels[1520:1680,:,0] > 220).any()  # actual caption ink was composited


def test_dry_run_exercises_portrait_and_cannot_pass_if_old_layout_is_restored(monkeypatch):
    import dry_run
    result = dry_run.dry_run("https://youtu.be/test", skip_download=True, skip_transcribe=True,
                             composition="portrait-caption-v1")
    assert not result["failures"], result["failures"]
    assert result["validation"]["portrait_exports"] == 3
    monkeypatch.setattr("automation.portrait_layout.plan_portrait", lambda *a, **k: None)
    broken = dry_run.dry_run("https://youtu.be/test", skip_download=True, skip_transcribe=True,
                             composition="portrait-caption-v1")
    assert any("portrait" in error for error in broken["failures"])
