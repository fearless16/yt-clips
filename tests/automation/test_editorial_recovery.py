"""Failure-first contracts for transcript -> aligned draft -> export recovery.

Only provider/ASR boundaries are substituted. No selector, alignment validator,
completeness gate, batch recovery, or acceptance assertion is mocked green.
"""

import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from automation.speech_alignment import alignment_issues
from shorts_intelligence.completeness import completeness_value, set_completeness


def words(start=10, count=20):
    return [{"start": start+i*.5, "end": start+i*.5+.4,
             "word": f"word{i}", "probability": .99} for i in range(count)]


def candidate(start=10.2):
    item = {"start": start, "end": 19.7, "text": "India cricket", "ai_reason": "old review"}
    set_completeness(item, "PASS", reason="old retained script", method="text_only")
    return item


def settings():
    return {"transcription": {"alignment_model": "medium", "alignment_fallback_model": "large-v3-turbo",
                              "language": "hi", "alignment_context_seconds": 2},
            "clip_selection": {"output_min_duration": 1, "output_max_duration": 38,
                               "alignment_max_boundary_shift": 3}}


def test_crossing_cut_is_repaired_before_render_and_old_review_is_invalidated(tmp_path, monkeypatch):
    from automation.speech_alignment import align_candidate
    evidence = [{"words": words()}]
    assert "straddles" in alignment_issues(evidence, 10.2, 19.7)[0]
    original = candidate()
    before = copy.deepcopy(original)
    monkeypatch.setattr("automation.speech_alignment.transcribe_range", Mock(return_value=evidence))
    result = align_candidate("source.mp4", original, settings(), tmp_path)
    assert [result["start"], result["end"]] == [10, 19.9]
    assert alignment_issues(result["speech_segments"], result["start"], result["end"]) == []
    assert completeness_value(result) is None  # never carry stale PASS into a repaired cut
    assert original == before


def test_invalid_asr_retries_without_fabricating_word_intervals(tmp_path, monkeypatch):
    from automation.speech_alignment import align_candidate
    invalid = [{"words": words()+[{"start": 15, "end": 15, "word": "bad", "probability": .1}]}]
    valid = [{"words": words()}]
    asr = Mock(side_effect=[invalid, valid])
    monkeypatch.setattr("automation.speech_alignment.transcribe_range", asr)
    result = align_candidate("source.mp4", candidate(), settings(), tmp_path)
    assert asr.call_count == 2
    assert [call.kwargs["model_name"] for call in asr.call_args_list] == ["medium", "large-v3-turbo"]
    assert result["speech_segments"][0]["words"] == words()
    assert result["alignment_attempts"][0]["issues"]
    assert result["alignment_attempts"][1]["issues"] == []
    assert list(tmp_path.glob("*.json"))  # raw failure evidence survives successful recovery
    for path in tmp_path.glob("speech-*.json"):
        identity = json.loads(path.read_text(encoding="utf-8"))["identity"]
        assert identity["version"] == 2
        assert identity["decoding"] == {"chunk_length": 8, "condition_on_previous_text": False}


def test_bad_timestamps_and_low_confidence_cannot_be_repaired_by_discarding_words(tmp_path, monkeypatch):
    from automation.speech_alignment import align_candidate
    bad = [{"words": [{**word, "probability": .1} for word in words()]}]
    asr = Mock(return_value=bad)
    monkeypatch.setattr("automation.speech_alignment.transcribe_range", asr)
    with pytest.raises(ValueError, match="alignment"):
        align_candidate("source.mp4", candidate(), settings(), tmp_path)
    assert asr.call_count == 2


def test_reviewed_invalid_evidence_is_never_overwritten_with_fresh_asr(tmp_path, monkeypatch):
    from automation.speech_alignment import align_candidate
    item = {**candidate(), "speech_segments": [{"words": words()+[{"start": 15, "end": 15}]}]}
    asr = Mock(side_effect=AssertionError("must not overwrite supplied evidence"))
    monkeypatch.setattr("automation.speech_alignment.transcribe_range", asr)
    with pytest.raises(ValueError):
        align_candidate("source.mp4", item, settings(), tmp_path)
    asr.assert_not_called()


def test_source_anchors_remove_next_topic_without_guessing_times(tmp_path, monkeypatch):
    from automation.speech_alignment import align_candidate
    item = {**candidate(), "opening_phrase": "word1 word2", "closing_phrase": "word16 word17"}
    monkeypatch.setattr("automation.speech_alignment.transcribe_range", Mock(return_value=[{"words": words()}]))
    result = align_candidate("source.mp4", item, settings(), tmp_path)
    assert [result["start"], result["end"]] == [10.5, 18.9]
    assert result["text"].startswith("word1 word2")
    assert result["text"].endswith("word16 word17")
    assert "word18" not in result["text"]


def test_missing_anchor_is_not_replaced_by_nearest_timestamp(tmp_path, monkeypatch):
    from automation.speech_alignment import align_candidate
    item = {**candidate(), "opening_phrase": "invented player", "closing_phrase": "word17"}
    monkeypatch.setattr("automation.speech_alignment.transcribe_range", Mock(return_value=[{"words": words()}]))
    with pytest.raises(ValueError, match="anchor"):
        align_candidate("source.mp4", item, settings(), tmp_path)


def test_malformed_word_strictly_outside_cut_does_not_poison_retained_speech():
    evidence = [{"words": words()+[{"start": 22, "end": 22, "word": "outside", "probability": .1}]}]
    assert alignment_issues(evidence, 10, 19.9) == []
    assert alignment_issues(evidence, 10, 23)  # inside the cut remains invalid


def test_unmatched_anchor_can_only_produce_unapproved_boundary_review_evidence(tmp_path, monkeypatch):
    from automation.speech_alignment import align_candidate
    item = {**candidate(), "opening_phrase": "source wording differs", "closing_phrase": "word17"}
    monkeypatch.setattr("automation.speech_alignment.transcribe_range", Mock(return_value=[{"words": words()}]))
    result = align_candidate("source.mp4", item, settings(), tmp_path, allow_boundary_review=True)
    assert result["boundary_review_required"] is True
    assert completeness_value(result) is None
    assert result["boundary_words"]


def test_invalid_cue_edge_can_reach_word_review_without_deleting_raw_evidence(tmp_path, monkeypatch):
    from automation.speech_alignment import align_candidate
    bank = [{"start": 10, "end": 10, "word": "bad edge", "probability": .1}] + words(start=10.5)
    evidence = [{"words": bank}]
    item = {**candidate(start=10), "opening_phrase": "caption wording differs", "closing_phrase": "word17"}
    monkeypatch.setattr("automation.speech_alignment.transcribe_range", Mock(return_value=evidence))
    result = align_candidate("source.mp4", item, settings(), tmp_path, allow_boundary_review=True)
    assert result["boundary_review_required"] is True
    assert result["speech_segments"] == evidence
    assert result["boundary_words"][0] == bank[0]
    assert completeness_value(result) is None
    assert result["alignment_attempts"][-1]["issues"]
    assert alignment_issues(result["speech_segments"], 10, 19.9)


@pytest.mark.parametrize("keep_bad_word", [False, True])
def test_word_review_may_trim_bad_edge_but_cannot_keep_or_skip_it(tmp_path, keep_bad_word):
    from automation.editorial_planner import _repair_boundaries
    from types import SimpleNamespace
    bank = [{"start": 10, "end": 10, "word": "bad edge", "probability": .1}] + words(start=10.5)
    item = {**candidate(), "speech_segments": [{"words": bank}], "boundary_words": bank,
            "boundary_review_required": True, "proposed_source_range": [10, 20.5],
            "opening_phrase": "caption", "closing_phrase": "differs"}
    request = SimpleNamespace(ask=lambda *a: {"cuts": [{"candidate_id": 1,
        "first_word_id": 0 if keep_bad_word else 1, "last_word_id": 20, "reason": "contiguous actual words"}]})
    decision = {"alignment_holds": []}
    result = _repair_boundaries([item], request, settings(), decision)
    if keep_bad_word:
        assert not result and decision["alignment_holds"]
    else:
        assert len(result) == 1
        assert result[0]["speech_segments"][0]["words"] == bank
        assert result[0]["start"] == 10.5
        assert result[0]["boundary_method"] == "source_words"
        assert alignment_issues(result[0]["speech_segments"], result[0]["start"], result[0]["end"]) == []
        assert completeness_value(result[0]) is None


@pytest.mark.parametrize("case_index", [0, 1])
def test_recorded_failed_run_remains_held_after_retry_not_silently_sanitized(tmp_path, monkeypatch, case_index):
    from automation.speech_alignment import align_candidate
    fixture = Path(__file__).parents[1] / "fixtures/pvA10GoCS9U_alignment_failures.json"
    case = json.loads(fixture.read_text(encoding="utf-8"))["cases"][case_index]
    before = copy.deepcopy(case["segments"])
    start, end = case["source_range"]
    assert alignment_issues(case["segments"], start, end)
    asr = Mock(return_value=case["segments"])
    monkeypatch.setattr("automation.speech_alignment.transcribe_range", asr)
    with pytest.raises(ValueError, match="alignment unresolved"):
        align_candidate("recorded-source.mp4", {"start": start, "end": end}, settings(), tmp_path)
    assert asr.call_count == 2
    assert case["segments"] == before
    artifacts = [json.loads(path.read_text(encoding="utf-8")) for path in tmp_path.glob("speech-*.json")]
    assert len(artifacts) == 2
    assert all(artifact["segments"] == before for artifact in artifacts)


def test_dry_run_exercises_required_word_timing_instead_of_disabling_it(monkeypatch):
    import dry_run
    import export
    observed = []
    original = export.export_all

    def checked(*args, **kwargs):
        observed.append(export.cfg["export"]["captions"]["require_word_timing"])
        return original(*args, **kwargs)

    monkeypatch.setattr(export, "export_all", checked)
    result = dry_run.dry_run("https://youtu.be/test", skip_download=True, skip_transcribe=True)
    assert observed == [True], "The dry run used to mask the exact production failure"
    assert not result["failures"], result["failures"]
    assert 2 <= len(result["exported"]) <= 3
    assert result["validation"]["word_aligned_exports"] == len(result["exported"])
    assert result["validation"]["boundary_review_requests"] >= 1


def test_successful_exports_survive_a_sibling_failure(tmp_path, monkeypatch):
    import export
    monkeypatch.setitem(export.cfg, "paths", {**export.cfg["paths"], "shorts": str(tmp_path / "shorts")})
    monkeypatch.setitem(export.cfg, "export", {**export.cfg["export"], "super_resolution": False,
        "composition": "portrait-caption-v1",
        "captions": {"enabled": False, "require_word_timing": False}, "max_workers": 1})
    monkeypatch.setattr(export, "analyze_clip", lambda *a, **k: {"export_strategy": {}})
    monkeypatch.setattr(export, "_get_best_encoder", lambda: "libx264")
    monkeypatch.setattr(export, "_validate_av_sync", lambda *a: True)

    def render(source, start, end, output, *args, **kwargs):
        if start == 30:
            raise ValueError("fixture failed sibling")
        Path(output).write_bytes(b"successful rendered media")
        return output

    monkeypatch.setattr(export, "export_clip", render)
    with pytest.raises(RuntimeError) as error:
        export.export_all({"clip1": {"start": 10, "end": 20, "text": "India cricket"},
                           "clip2": {"start": 30, "end": 40, "text": "India cricket"}},
                          "source.mp4", generate_seo=False)
    assert len(error.value.exported) == 1
    assert error.value.exported[0].read_bytes() == b"successful rendered media"
    assert "clip2" in str(error.value)


def test_dry_run_detects_unrecoverable_asr_instead_of_passing_zero_exports(monkeypatch):
    import dry_run
    bad = [{"words": [{"start": 10, "end": 10, "word": "invalid", "probability": .99}]}]
    monkeypatch.setattr("automation.speech_alignment.transcribe_range", Mock(return_value=bad))
    result = dry_run.dry_run("https://youtu.be/test", skip_download=True, skip_transcribe=True)
    assert not result["exported"]
    assert result["failures"]
    assert not dry_run._print_validation_report(result)


def test_dry_run_catches_missing_speech_artifacts_even_if_renderer_claims_success(monkeypatch):
    import dry_run
    import export
    original = export.export_all

    def missing(*args, **kwargs):
        paths = original(*args, **kwargs)
        for path in paths:
            path.with_name(f"{path.stem}_speech.json").unlink()
        return paths

    monkeypatch.setattr(export, "export_all", missing)
    result = dry_run.dry_run("https://youtu.be/test", skip_download=True, skip_transcribe=True)
    assert len(result["exported"]) == 3
    assert result["validation"]["word_aligned_exports"] == 0
    assert any("acceptance: word alignment" in error for error in result["failures"])
    assert not dry_run._print_validation_report(result)


@pytest.mark.parametrize("reviewed", [True, False])
def test_reviewed_reserve_fills_failed_export_without_rerendering_success(tmp_path, monkeypatch, reviewed):
    import export
    monkeypatch.setitem(export.cfg, "paths", {**export.cfg["paths"], "shorts": str(tmp_path / "shorts")})
    monkeypatch.setitem(export.cfg, "export", {**export.cfg["export"], "super_resolution": False,
        "composition": "portrait-caption-v1", "captions": {"enabled": False, "require_word_timing": False}})
    monkeypatch.setattr(export, "analyze_clip", lambda *a, **k: {"export_strategy": {}})
    monkeypatch.setattr(export, "_get_best_encoder", lambda: "libx264")
    monkeypatch.setattr(export, "_validate_av_sync", lambda *a: True)
    calls = []

    def render(source, start, end, output, clip_id, **kwargs):
        calls.append(clip_id)
        if clip_id == "clip2":
            raise ValueError("original failed")
        Path(output).write_bytes(b"encoded")
        return output

    monkeypatch.setattr(export, "export_clip", render)
    highlights = {clip_id: {"start": start, "end": start+10, "text": "India cricket",
                            "role": "reserve" if "reserve" in clip_id else "primary"}
                  for clip_id, start in (("clip1", 10), ("clip2", 30), ("reserve1", 60), ("reserve2", 90))}
    if reviewed:
        for clip_id in ("reserve1", "reserve2"):
            set_completeness(highlights[clip_id], "PASS", method="text_only", reason="Retained setup and payoff")
            highlights[clip_id]["semantic_review"] = "text_only"
    else:
        with pytest.raises(export.ExportBatchError):
            export.export_all(highlights, "source.mp4", generate_seo=False)
        assert calls == ["clip1", "clip2"]
        return
    outputs = export.export_all(highlights, "source.mp4", generate_seo=False)
    assert {p.stem for p in outputs} == {"clip1", "reserve1"}
    assert calls.count("clip1") == 1
    assert "reserve2" not in calls
    artifact = json.loads(outputs[0].with_name("batch_failures.json").read_text())
    assert artifact["status"] == "RECOVERED"
    assert any("clip2" in error for error in artifact["errors"])
