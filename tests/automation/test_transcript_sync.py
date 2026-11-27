"""Regressions for source word evidence, boundary context and encoded timing."""

import json
import math
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import export
from automation.caption_engine import CaptionCard, prepare_clip_captions, word_timed_cues
from automation.delivery_qc import av_timing_issues
from automation.speech_alignment import alignment_issues, transcribe_range


@pytest.fixture(autouse=True)
def standard_renderer_profile(monkeypatch):
    """These regressions target the standard trim/retime renderer, not gameplay."""
    monkeypatch.setitem(export.cfg["export"], "composition", "portrait-caption-v1")


def word(start, end, text="retained", probability=0.99):
    return {"start": start, "end": end, "word": text, "probability": probability}


def test_loader_preserves_word_text_timing_and_provenance(tmp_path):
    segment = {"start": 10.25, "end": 20.75, "text": "corrected words",
               "words": [word(10.25, 20.75)], "timing_method": "reviewed alignment"}
    path = tmp_path / "speech.json"
    path.write_text(json.dumps({"segments": [segment]}))
    assert export._load_transcript_segments(str(path)) == [segment]


@pytest.fixture
def batch(tmp_path, monkeypatch):
    monkeypatch.setitem(export.cfg, "paths", {**export.cfg["paths"], "shorts": str(tmp_path / "shorts")})
    monkeypatch.setitem(export.cfg, "export", {**export.cfg["export"], "max_workers": 1,
        "super_resolution": False, "captions": {"enabled": True, "require_word_timing": True}})
    monkeypatch.setattr(export, "analyze_clip", lambda *a, **k: {
        "export_strategy": {}, "layout": {"layout_type": "solo"}})
    monkeypatch.setattr(export, "_get_best_encoder", lambda: "libx264")
    monkeypatch.setattr(export, "_validate_av_sync", lambda *a: True)
    captured = {}

    def render(source, start, end, output, *args, **kwargs):
        from pathlib import Path
        captured["segments"] = kwargs["transcript_segments"]
        Path(output).write_bytes(b"encoded test fixture")
        return output

    monkeypatch.setattr(export, "export_clip", render)
    path = tmp_path / "speech.json"

    def run():
        return export.export_all({"clip1": {"start_sec": 10.25, "end_sec": 20.75,
            "text": "Kohli hits a six and India win the cricket match."}},
            "source.mp4", transcript_path=str(path), generate_seo=False)

    return path, captured, run


@pytest.mark.parametrize("contents", [None, "[]", '{"segments": []}', "broken JSON"])
def test_missing_or_empty_transcript_runs_contextual_asr(batch, monkeypatch, contents):
    path, captured, run = batch
    if contents is not None:
        path.write_text(contents)
    segments = [{"start": 8.25, "end": 22.75, "text": "outside retained outside",
        "words": [word(8.25, 9, "before"), word(10.25, 20.75), word(21, 22.75, "after")]}]
    asr = Mock(return_value=segments)
    monkeypatch.setattr("automation.speech_alignment.transcribe_range", asr)
    outputs = run()
    assert len(outputs) == 1
    assert captured["segments"] == segments
    assert asr.call_args.kwargs["context_seconds"] == 2.0
    edit = json.loads(outputs[0].with_name("clip1_edit.json").read_text())
    assert edit["text"] == "retained"  # no context leakage into packaging


def test_reviewed_words_are_reused_without_retranscription(batch, monkeypatch):
    path, captured, run = batch
    segments = [{"start": 10.25, "end": 20.75, "text": "corrected retained",
                 "words": [word(10.25, 20.75, "corrected retained")]}]
    path.write_text(json.dumps({"segments": segments}))
    asr = Mock(side_effect=AssertionError("Do not replace existing word evidence"))
    monkeypatch.setattr("automation.speech_alignment.transcribe_range", asr)
    assert len(run()) == 1
    assert captured["segments"] == segments
    asr.assert_not_called()


def test_per_highlight_caption_opt_in_uses_same_asr_contract(batch, monkeypatch):
    _, _, run = batch
    monkeypatch.setitem(export.cfg["export"], "captions", {"enabled": False, "require_word_timing": True})
    monkeypatch.setattr(export, "analyze_clip", lambda *a, **k: {
        "export_strategy": {}, "layout": {"layout_type": "solo"}, "caption_overlay": {"enabled": True}})
    asr = Mock(return_value=[{"start": 10.25, "end": 20.75, "words": [word(10.25, 20.75)]}])
    monkeypatch.setattr("automation.speech_alignment.transcribe_range", asr)
    assert len(run()) == 1
    asr.assert_called_once()


def test_missing_confidence_is_flagged_not_held_or_retranscribed(batch, monkeypatch):
    # Confidence is a caption-review flag (cancer register C2), never a cut-killer.
    path, _, run = batch
    path.write_text(json.dumps([{"start": 10.25, "end": 20.75, "words": [
        {"start": 10.25, "end": 20.75, "word": "corrected"}]}]))
    asr = Mock()
    monkeypatch.setattr("automation.speech_alignment.transcribe_range", asr)
    assert len(run()) == 1
    asr.assert_not_called()


def test_partial_word_evidence_is_held_without_overwriting_corrections(batch, monkeypatch):
    path, _, run = batch
    path.write_text(json.dumps([
        {"start": 10.25, "end": 15, "words": [word(10.25, 15, "corrected")]},
        {"start": 15, "end": 20.75, "text": "unreviewed cue-only text"},
    ]))
    asr = Mock()
    monkeypatch.setattr("automation.speech_alignment.transcribe_range", asr)
    with pytest.raises(RuntimeError, match="Missing word timestamps"):
        run()
    asr.assert_not_called()


@pytest.mark.parametrize("sync_result", [False, None])
def test_failed_or_unknown_av_timing_holds_batch(batch, monkeypatch, sync_result):
    path, _, run = batch
    path.write_text(json.dumps([{"start": 10.25, "end": 20.75, "words": [word(10.25, 20.75)]}]))
    monkeypatch.setattr(export, "_validate_av_sync", lambda *a: sync_result)
    with pytest.raises(RuntimeError, match="encoded A/V timing failed or unavailable"):
        run()


@pytest.mark.parametrize("bounds", [(0, 100), (math.nan, 11), (11, math.inf), (12, 11)])
def test_invalid_or_boundary_crossing_words_cannot_pass(bounds):
    issues = alignment_issues([{"words": [word(*bounds)]}], 10, 20)
    assert issues


def test_only_selected_words_affect_gaps_and_confidence():
    segments = [{"words": [word(0, 1, probability=0), word(10, 20), word(99, 100, probability=0)]}]
    assert alignment_issues(segments, 10, 20) == []


@pytest.mark.parametrize("probability", [None, "unknown", math.nan, -0.1, 1.1])
def test_invalid_confidence_is_an_explicit_flag_not_a_hold(probability):
    from automation.speech_alignment import alignment_flags
    segments = [{"words": [word(10, 20, probability=probability)]}]
    assert alignment_issues(segments, 10, 20) == []
    assert any("confidence unavailable" in flag for flag in alignment_flags(segments, 10, 20))


def test_reaction_pause_is_flagged_not_held():
    from automation.speech_alignment import alignment_flags
    segments = [{"words": [word(10, 11), word(15, 20)]},
                {"start": 11, "end": 15, "text": "", "words": []}]  # crowd roar
    assert alignment_issues(segments, 10, 20) == []
    assert any("pause" in flag for flag in alignment_flags(segments, 10, 20))


def test_overlapping_words_are_held_before_render():
    assert any("Overlapping" in issue for issue in alignment_issues(
        [{"words": [word(10, 12), word(11, 13)]}], 10, 13))


def _stub_whisper_cpp(monkeypatch, alignment, segments):
    from utils import whisper_cpp
    seen = {}

    def run(wav, **kwargs):
        seen.update(kwargs, wav=wav)
        return {"segments": segments, "language": "hi", "backend": "rocm", "model": "ggml-x.bin"}

    monkeypatch.setattr(whisper_cpp, "run_whisper_cli", run)
    ffmpeg = Mock(return_value=subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(alignment.subprocess, "run", ffmpeg)
    return seen, ffmpeg


@pytest.mark.parametrize("start", [0.5, 10.25])
def test_asr_adds_context_and_offsets_words_from_context_start(tmp_path, monkeypatch, start):
    import automation.speech_alignment as alignment
    import numpy as np
    context_start = max(0, start - 2)
    local_word = {"start": np.float64(start-context_start-0.1), "end": np.float64(start-context_start+0.3),
                  "word": "crossing", "probability": np.float64(0.99)}
    local_segment = {"start": local_word["start"], "end": local_word["end"],
                     "text": "crossing", "words": [local_word]}
    seen, ffmpeg = _stub_whisper_cpp(monkeypatch, alignment, [local_segment])
    segments = transcribe_range("source.mp4", start, start+10, model_name="fixture", temp_dir=tmp_path)
    cmd = ffmpeg.call_args.args[0]
    assert float(cmd[cmd.index("-ss")+1]) == context_start
    assert float(cmd[cmd.index("-t")+1]) == start+12-context_start
    assert segments[0]["words"][0]["start"] == pytest.approx(start-0.1)
    assert segments[0]["requested_range"] == [start, start+10]
    assert seen["model_name"] == "fixture" and seen["language"] == "hi"
    assert seen["max_context"] == 0  # independent windows: no previous-text conditioning
    assert type(segments[0]["start"]) is float
    assert all(type(segments[0]["words"][0][key]) is float for key in ("start", "end", "probability"))
    assert any("straddles" in issue for issue in alignment_issues(segments, start, start+10))


def test_asr_without_word_timestamps_holds_for_review(tmp_path, monkeypatch):
    import automation.speech_alignment as alignment
    _stub_whisper_cpp(monkeypatch, alignment, [{"start": 0, "end": 1, "text": "crowd", "words": []}])
    with pytest.raises(ValueError, match="no word timestamps"):
        transcribe_range("source.mp4", 5, 15, temp_dir=tmp_path)


def test_asr_never_drops_a_word_that_lacks_confidence(tmp_path, monkeypatch):
    import automation.speech_alignment as alignment
    words = [{"start": 1.0, "end": 1.4, "word": "a", "probability": 0.9},
             {"start": 1.4, "end": 1.8, "word": "b"}]
    _stub_whisper_cpp(monkeypatch, alignment, [{"start": 1.0, "end": 1.8, "text": "a b", "words": words}])
    segments = transcribe_range("source.mp4", 5, 15, temp_dir=tmp_path)
    assert [w["word"] for w in segments[0]["words"]] == ["a", "b"]
    assert "probability" not in segments[0]["words"][1]  # flagged later as 'confidence unavailable'


def test_asr_engine_is_whisper_cpp_and_cache_identity_says_so():
    """Old faster-whisper cache files must never be reused for whisper.cpp timings."""
    import automation.speech_alignment as alignment
    assert alignment.ASR_DECODING["engine"] == "whisper.cpp"
    assert not hasattr(alignment, "_MODELS")


def test_required_caption_preparation_rejects_empty_evidence(tmp_path):
    with pytest.raises(ValueError, match="no speech evidence"):
        prepare_clip_captions([], 10, 20, tmp_path, require_word_timing=True)


@pytest.mark.parametrize("alternate", ["premium_enabled", "super_res_enabled"])
def test_required_captions_force_standard_renderer(tmp_path, monkeypatch, alternate):
    monkeypatch.setattr(export, "premium_enabled", alternate == "premium_enabled")
    monkeypatch.setattr(export, "super_res_enabled", alternate == "super_res_enabled")
    monkeypatch.setattr(export, "_get_video_info", lambda *a: {"fps": 30, "width": 1920, "height": 1080})
    monkeypatch.setattr(export, "_get_best_encoder", lambda: "libx264")
    monkeypatch.setattr(export, "_has_audio_stream", lambda *a: True)
    monkeypatch.setattr(export, "_check_free_space", lambda *a: True)
    monkeypatch.setattr(export, "_build_enhance_stack", lambda *a, **k: "[v_src]null[v_out]")
    monkeypatch.setitem(export.cfg, "thumbnail", {"template_path": str(tmp_path / "absent.png")})
    monkeypatch.setitem(export.cfg, "export", {**export.cfg["export"],
        "captions": {"enabled": True, "require_word_timing": True}})
    build = Mock(return_value=[CaptionCard(str(tmp_path / "card.png"), 0, 1, "word")])
    monkeypatch.setattr("automation.caption_engine.prepare_clip_captions", build)
    render = Mock(return_value=(True, ""))
    monkeypatch.setattr(export, "_run_ffmpeg_with_retry", render)
    premium, sr = Mock(), Mock()
    monkeypatch.setitem(sys.modules, "premium_render", SimpleNamespace(PremiumRender=premium))
    monkeypatch.setitem(sys.modules, "utils.super_res", SimpleNamespace(SuperResEnhancer=sr))
    analysis = {"export_strategy": {}}
    output = str(tmp_path / "out.mp4")
    assert export.export_clip("source.mp4", 10, 20, output,
                              transcript_segments=[{"start": 10, "end": 20}], analysis=analysis) == output
    build.assert_called_once()
    premium.assert_not_called()
    sr.assert_not_called()
    assert "v_cap_final" in " ".join(render.call_args.args[0])
    assert analysis["edit_timeline"][0]["source_start"] == 10


@pytest.mark.parametrize("audio_start,audio_duration,valid", [(0, 10, True), (0.3, 10, False),
    (0, 9.5, False), ("N/A", 10, False), (0, math.nan, False), (0.02, 9.98, True)])
def test_stream_timing_checks_offsets_and_unknown_data(audio_start, audio_duration, valid):
    streams = [{"codec_type": "video", "start_time": 0, "duration": 10},
               {"codec_type": "audio", "start_time": audio_start, "duration": audio_duration}]
    assert (not av_timing_issues(streams)) == valid


@pytest.mark.parametrize("result", ["malformed", "missing_audio", "probe_failed"])
def test_av_probe_failures_never_return_success(monkeypatch, result):
    if result == "probe_failed":
        run = Mock(side_effect=subprocess.CalledProcessError(1, "ffprobe"))
    else:
        stdout = "broken JSON" if result == "malformed" else json.dumps({"streams": []})
        run = Mock(return_value=subprocess.CompletedProcess([], 0, stdout, ""))
    monkeypatch.setattr(export.subprocess, "run", run)
    assert export._validate_av_sync("clip.mp4", "clip") is False


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required")
@pytest.mark.parametrize("speed", [1.0, 1.25])
def test_real_encoded_caption_audio_timing_has_no_growing_drift(tmp_path, monkeypatch, speed):
    """Isolate production trim/speed/overlay timing from ASR, layout and fonts."""
    import numpy as np
    from PIL import Image

    source, output, card = tmp_path / "source.mp4", tmp_path / "output.mp4", tmp_path / "card.png"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
        "color=c=black:s=320x180:r=30:d=40", "-f", "lavfi", "-i",
        r"aevalsrc=0.7*sin(2*PI*1000*t)*(between(t\,5.22\,5.32)+between(t\,36.22\,36.32)):s=48000:d=40",
        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(source)],
        check=True, capture_output=True, timeout=60)
    Image.new("RGBA", (1080, 320), (255, 255, 255, 255)).save(card)
    monkeypatch.setattr(export, "premium_enabled", False)
    monkeypatch.setattr(export, "super_res_enabled", False)
    monkeypatch.setattr(export, "_get_best_encoder", lambda: "libx264")
    monkeypatch.setattr(export, "_build_enhance_stack", lambda *a, **k: "[v_src]scale=1080:1920,setsar=1[v_out]")
    # Execute real source-word -> relative-cue conversion; substitute only PNG
    # styling so this regression does not depend on fonts or browser installs.
    monkeypatch.setattr("automation.caption_engine.render_transparent_cards_batch",
        lambda cues, output_dir: [CaptionCard(str(card), cue.start, cue.end, cue.text) for cue in cues])
    monkeypatch.setitem(export.cfg, "thumbnail", {"template_path": str(tmp_path / "absent.png")})
    monkeypatch.setitem(export.cfg, "export", {**export.cfg["export"], "enable_variable_speed": True,
        "pre_seek": 4, "encoder_preset": "ultrafast", "min_output_bytes": 1000, "transitions": {},
        "captions": {"enabled": True, "require_word_timing": True, "from_bottom": 0.2}})
    monkeypatch.setitem(export.cfg, "highlight", {**export.cfg["highlight"], "min_duration": 1})
    assert export.export_clip(str(source), 4.22, 38.22, str(output),
        transcript_segments=[{"start": 5.22, "end": 36.32,
                              "words": [word(5.22, 5.32, "first"), word(36.22, 36.32, "last")]}],
        analysis={"export_strategy": {"speed_factor": speed}})
    assert export._validate_av_sync(str(output), "fixture")
    audio = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(output), "-vn", "-ac", "1",
        "-ar", "16000", "-f", "f32le", "-"], check=True, capture_output=True, timeout=60)
    samples = np.frombuffer(audio.stdout, dtype="<f4")
    count = len(samples) // 160
    rms = np.sqrt(np.mean(samples[:count*160].reshape(count, 160)**2, axis=1))
    active = np.flatnonzero(rms > rms.max()*0.2)
    audio_onsets = active[np.r_[True, np.diff(active) > 1]] / 100
    frames = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(output),
        "-vf", "crop=2:2:100:1300,format=gray", "-f", "rawvideo", "-"],
        check=True, capture_output=True, timeout=60)
    pixels = np.frombuffer(frames.stdout, dtype=np.uint8).reshape(-1, 4).mean(axis=1)
    bright = np.flatnonzero(pixels > 150)
    caption_onsets = bright[np.r_[True, np.diff(bright) > 1]] / 30
    assert len(audio_onsets) == len(caption_onsets) == 2
    assert max(abs(caption_onsets - np.array([1, 32])/speed)) < 0.04
    separation = caption_onsets - audio_onsets
    assert max(abs(separation)) < 0.08
    assert abs(separation[-1] - separation[0]) < 0.03
