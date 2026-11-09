import math

import pytest

from automation.caption_engine import word_timed_cues
from automation.edit_timeline import RetainedPart
from automation.speech_alignment import alignment_issues


def test_word_captions_preserve_negation_and_do_not_overlap():
    segments = [{"start": 10, "end": 12, "words": [
        {"start": 10, "end": 10.5, "word": "Kohli"},
        {"start": 10.5, "end": 11, "word": "did"},
        {"start": 11, "end": 11.4, "word": "not"},
        {"start": 11.4, "end": 12, "word": "agree"},
    ]}]
    cues = word_timed_cues(segments, 10, 12, max_words=2)
    assert [cue.text for cue in cues] == ["Kohli did", "not agree"]
    assert [(cue.start, cue.end) for cue in cues] == [(0, 1), (1, 2)]


@pytest.mark.parametrize("start,end", [(10.2, 12), (10, 11.8)])
def test_caption_cut_cannot_straddle_word(start, end):
    segments = [{"start": 10, "end": 12, "words": [{"start": 10, "end": 12, "word": "boundary"}]}]
    with pytest.raises(ValueError, match="straddles"):
        word_timed_cues(segments, start, end)


def test_cue_only_transcript_is_not_word_alignment():
    with pytest.raises(ValueError, match="Missing word"):
        word_timed_cues([{"start": 10, "end": 12, "text": "Kohli did not agree"}], 10, 12)


def test_asr_gaps_and_low_confidence_are_held_not_silently_accepted():
    issues = alignment_issues([{"words": [{"start": 110, "end": 111, "probability": 0.4}]}], 80, 124)
    assert any("opening" in issue for issue in issues)
    assert any("ending" in issue for issue in issues)
    assert any("Low-confidence" in issue for issue in issues)


def test_edit_map_preserves_fractional_times_and_speed():
    part = RetainedPart(10.25, 20.75, 1.25)
    assert part.output_time(11.5) == 1
    assert part.output_duration == 8.4
    assert part.to_dict()["source_start"] == 10.25
    with pytest.raises(ValueError):
        part.output_time(10)


@pytest.mark.parametrize("start,end,speed", [(math.nan, 10, 1), (0, math.inf, 1), (2, 1, 1), (0, 2, 0)])
def test_invalid_edit_map_is_rejected(start, end, speed):
    with pytest.raises(ValueError):
        RetainedPart(start, end, speed)


def test_reserved_band_fits_complete_speaker_region(monkeypatch):
    import export
    monkeypatch.setitem(export.cfg["export"]["captions"], "style", "neon-reference-v2")
    monkeypatch.setitem(export.cfg["export"], "reserve_caption_band", True)
    graph = export._build_enhance_stack({"export_strategy": {
        "use_solo_frame": True, "active_crop": {"x": 400, "y": 0, "width": 1000, "height": 1080}}}, use_logo=False)
    assert "crop=1000:1080:400:0" in graph
    assert "[portrait_fg]scale=972:998:flags=lanczos:force_original_aspect_ratio=decrease" in graph
    assert "[portrait_blur][portrait_fit]overlay=(W-w)/2:210" in graph
    assert "pad=" not in graph
    assert export._caption_card_y() == 1216


def test_screen_share_fits_readable_foreground():
    import export
    graph = export._build_enhance_stack({"export_strategy": {"is_screen_share": True}}, use_logo=False)
    assert "[screen_fg]scale=1080:1920:flags=lanczos:force_original_aspect_ratio=decrease" in graph
    assert "[screen_blur][screen_fit]overlay=" in graph


def test_unknown_geometry_does_not_silently_center_crop(monkeypatch):
    import export
    monkeypatch.setitem(export.cfg["export"], "reserve_caption_band", True)
    graph = export._build_enhance_stack({}, use_logo=False)
    # Unknown camera geometry keeps the full source foreground; only background
    # is cover-cropped to fill portrait, never silently face/center-cropped.
    assert "[portrait_fg]scale=" in graph
    assert graph.count("crop=") == 1
