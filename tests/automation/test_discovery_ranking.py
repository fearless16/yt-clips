"""Contract for clip discovery (automation/clip_selection/pipeline.py)."""
import copy
import json
from pathlib import Path

import yaml

from automation.clip_selection import pipeline

PREF = (10.0, 35.0)


def window(text, start=0.0, end=20.0):
    return {"start": start, "end": end, "text": text}


def test_loudness_is_scored_against_the_streams_own_baseline():
    quiet = [(float(t), 0.1) for t in range(100)]
    z = pipeline._energy_z(quiet + [(100.0, 0.9)])
    assert z[100] > 3 and abs(z[5]) < 0.5


def test_reaction_spike_with_hook_and_payoff_outranks_flat_chatter():
    z = {t: 0.0 for t in range(60)}
    z.update({0: 3.0, 1: 3.0, 18: 3.5, 19: 3.5})
    hype = pipeline.score_window(window("Arre kya six! Kohli ne wicket ke baad chhakka maara, out of the ground!"),
                                 z, {"kohli"}, PREF)
    flat = pipeline.score_window(window("Toh guys aaj ka match India ka hai aur score dekhte hain over ke baad",
                                        30, 50), z, set(), PREF)
    assert hype["final_score"] > flat["final_score"] + 30
    assert flat["dimension_scores"]["lead_in_penalty"] == 1.0
    assert set(hype["dimension_scores"]) >= {"energy_peak", "hook", "payoff", "reaction", "cricket_density"}


def test_duration_fit_is_a_prior_not_a_hold():
    assert pipeline._duration_fit(20, *PREF) == 1.0
    assert 0 < pipeline._duration_fit(38, *PREF) < 1.0


def test_banter_is_a_window_but_only_cricket_talk_is_labelled_cricket():
    thoughts = [{"start": 0, "end": 6, "text": "Bumrah ka yorker, middle stump ud gaya, wicket!", "source_cue_ids": [0]},
                {"start": 6.5, "end": 13, "text": "India ab match mein wapas aa gaya bhai", "source_cue_ids": [1]},
                {"start": 30, "end": 42, "text": "Aaj main shopping karne gaya tha aur movie dekhi", "source_cue_ids": [2]}]
    windows = pipeline._build_livestream_windows(thoughts, 10, 38)
    assert {w["cricket"] for w in windows if w["start"] < 20} == {True}
    assert [w["cricket"] for w in windows if w["start"] >= 30] == [False]


def test_exported_moments_are_remembered_and_skipped_on_rerun(tmp_path, monkeypatch):
    config = copy.deepcopy(pipeline.cfg)
    config["paths"] = {k: str(tmp_path / k) for k in ("input", "transcripts", "highlights", "shorts", "temp")}
    config["release_review"] = {"required": False}
    monkeypatch.setattr(pipeline, "cfg", config)
    monkeypatch.setattr(pipeline, "_extract_audio_rms", lambda _: [])
    seen = []

    def curate(segments, candidates, video_path, cfg, context, evidence_path):
        seen.append([(c["start"], c["end"]) for c in candidates])
        context["selection_summary"] = {"status": "NO_PUBLISHABLE_CANDIDATES"}
        return []

    monkeypatch.setattr("automation.editorial_planner.curate_clips", curate)
    transcript = tmp_path / "t.json"
    transcript.write_text(json.dumps({"segments": [
        {"start": 0, "end": 12, "text": "Bumrah ka yorker aur wicket, India ka match palat gaya bhai"},
        {"start": 60, "end": 72, "text": "Kohli ka cover drive four, boundary pe chauka, kya shot tha"}]}))
    out = tmp_path / "h.yaml"
    diagnostics = {}
    pipeline.detect_highlights(str(transcript), "v.mp4", str(out), match_key="m1", diagnostics=diagnostics)
    assert any(s < 30 for s, _ in seen[0]) and any(s >= 60 for s, _ in seen[0])
    pipeline.record_exported_windows(diagnostics["dedupe_history"], [{"start": 0.0, "end": 12.0}])
    pipeline.detect_highlights(str(transcript), "v.mp4", str(out), match_key="m1", diagnostics={})
    assert seen[1] and all(s >= 60 for s, _ in seen[1])
    assert yaml.safe_load(out.read_text(encoding="utf-8")) == {}
    assert Path(diagnostics["dedupe_history"]).name == "m1.dedupe_history.yaml"


def test_signals_read_the_romanizer_spelling_of_devanagari_speech():
    # Whisper/YouTube Hindi arrives as Devanagari; to_roman keeps the inherent
    # vowel (kyaa, aauta, chhakkaa). Measured: casual-only lexicons hit ~0 words.
    from utils.devanagari import to_roman
    text = to_roman("अरे यार क्या बात है, धोनी ने छक्का मारा, आउट हो गया!")
    assert pipeline._reaction_hits(text) >= 4
    assert pipeline.is_cricket_window(text)
    assert pipeline._LEAD_IN.match(to_roman("और मतलब")) and not pipeline._STRONG_OPEN.search("cha?aa")


def test_editor_hints_are_cue_ranges_of_its_own_section_with_audio_signals():
    from automation.editorial_planner import _suggestions
    section = [{"cue_id": i} for i in range(5, 10)]
    inside = {"source_cue_ids": [6, 8], "final_score": 70, "dimension_scores": {"energy_peak": 0.9, "duration_fit": 1}}
    outside = {"source_cue_ids": [9, 12], "final_score": 90, "dimension_scores": {}}
    assert _suggestions([outside, inside], section) == [
        {"start_cue_id": 6, "end_cue_id": 8, "score": 70, "signals": {"energy_peak": 0.9}}]
