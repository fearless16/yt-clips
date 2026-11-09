"""Exercise the real curator, alignment, final review and cache contracts."""

import copy
import json
from unittest.mock import Mock

import pytest

from shorts_intelligence.completeness import completeness_value


def material():
    segments = []
    for i in range(4):
        text = f"India cricket idea{i} setup explains why the batter waits and wins"
        tokens = text.split()
        start = 10+i*30
        words = [{"start": start+j, "end": start+j+.8, "word": token, "probability": .99}
                 for j, token in enumerate(tokens)]
        segments.append({"start": start, "end": words[-1]["end"], "text": text, "words": words})
    return segments


def proposal(i):
    return {"start_cue_id": i, "end_cue_id": i, "opening_phrase": "India cricket",
            "closing_phrase": "and wins", "score": 90-i, "reason": "Specific cricket setup and payoff"}


def final(ids=(1, 2, 3), reserves=(4,)):
    def item(i):
        return {"candidate_id": i, "score": 95-i, "complete_thought": True,
                "reason": "The retained setup explains waiting and the retained payoff is winning."}
    return {"selected": [item(i) for i in ids], "reserves": [item(i) for i in reserves], "rejected": []}


def config():
    return {"ai": {"provider": "deepseek", "model": "deepseek-v4-pro"},
            "transcription": {"alignment_model": "medium"},
            "clip_selection": {"max_selected": 3, "output_min_duration": 5, "output_max_duration": 38,
                "editorial": {"enabled": True, "shortlist_size": 6, "target_min": 2,
                              "max_calls": 6, "max_transcript_chars": 10000}}}


def run(tmp_path, monkeypatch, responses, segments=None, candidates=None, context=None):
    from automation import editorial_planner as planner
    ai = Mock()
    ai.generate_json.side_effect = [json.dumps(response) for response in responses]
    monkeypatch.setattr(planner, "_get_ai", lambda: ai)
    source = tmp_path / "source.mp4"
    if not source.exists():
        source.write_bytes(b"stable source identity")
    context = {} if context is None else context
    result = planner.curate_clips(segments or material(), candidates or [], str(source), config(),
                                 context, tmp_path / "curation.json")
    return result, ai, context


def test_whole_transcript_can_rescue_quiet_candidates_before_heuristic_filter(tmp_path, monkeypatch):
    result, ai, context = run(tmp_path, monkeypatch,
        [{"proposals": [proposal(i) for i in range(4)]}, final()],
        candidates=[{"start": 10, "end": 21.8, "text": "India cricket", "final_score": 1}])
    assert len(result) == 3
    assert len(context["editorial_reserves"]) == 1
    assert all(completeness_value(item) is True for item in result+context["editorial_reserves"])
    for call in ai.generate_json.call_args_list:
        assert call.kwargs["reasoning_effort"] == "high"
    prompt = ai.generate_json.call_args_list[0].args[0]
    assert "idea3" in prompt  # source-wide coverage, not the old eight survivors
    assert context["selection_summary"]["selected"] == 3


def test_final_review_sees_actual_aligned_words_not_extra_cue_text(tmp_path, monkeypatch):
    segments = material()
    segments[0]["text"] += " unrelated next topic"
    result, ai, _ = run(tmp_path, monkeypatch,
        [{"proposals": [proposal(i) for i in range(4)]}, final()], segments)
    assert "unrelated next topic" in ai.generate_json.call_args_list[0].args[0]
    final_prompt = ai.generate_json.call_args_list[1].args[0]
    assert "unrelated next topic" not in final_prompt
    assert "setup explains" in result[0]["text"]


def test_failed_primary_alignment_uses_another_shortlisted_moment(tmp_path, monkeypatch):
    segments = material()
    segments[0]["words"][0]["end"] = segments[0]["words"][0]["start"]
    result, _, context = run(tmp_path, monkeypatch,
        [{"proposals": [proposal(i) for i in range(4)]}, final((1, 2, 3), ())], segments)
    assert len(result) == 3
    assert all(item["start"] >= 40 for item in result)
    assert len(context["editorial_decision"]["alignment_holds"]) == 1


@pytest.mark.parametrize("bad", [
    {**proposal(0), "start_cue_id": True},
    {**proposal(0), "end_cue_id": 999},
    {**proposal(0), "opening_phrase": "invented quotation"},
    {**proposal(0), "score": float("nan")},
])
def test_invented_or_invalid_source_evidence_never_reaches_final_selection(tmp_path, monkeypatch, bad):
    result, ai, context = run(tmp_path, monkeypatch, [{"proposals": [bad]}])
    assert result == []
    assert context["selection_summary"]["status"] == "REVIEW_REQUIRED"
    assert ai.generate_json.call_count <= 2


def test_same_source_and_plan_resume_without_new_llm_calls(tmp_path, monkeypatch):
    responses = [{"proposals": [proposal(i) for i in range(4)]}, final()]
    first, _, _ = run(tmp_path, monkeypatch, responses)
    second, ai, _ = run(tmp_path, monkeypatch, [])
    assert first == second
    ai.generate_json.assert_not_called()


def test_changed_source_cannot_reuse_cached_decisions(tmp_path, monkeypatch):
    responses = [{"proposals": [proposal(i) for i in range(4)]}, final()]
    run(tmp_path, monkeypatch, responses)
    (tmp_path / "source.mp4").write_bytes(b"different source")
    _, ai, _ = run(tmp_path, monkeypatch, responses)
    assert ai.generate_json.call_count == 2


def test_final_empty_attestation_and_duplicate_reserve_cannot_be_approved(tmp_path, monkeypatch):
    answer = final()
    answer["selected"][0]["complete_thought"] = "true"
    answer["reserves"] = [answer["selected"][1]]
    result, _, context = run(tmp_path, monkeypatch,
        [{"proposals": [proposal(i) for i in range(4)]}, answer])
    assert len(result) == 2
    assert not context["editorial_reserves"]
    assert context["editorial_decision"]["held_selections"]


def test_confirmed_failed_ranges_and_previously_published_ranges_are_not_rescued(tmp_path, monkeypatch):
    context = {"previous_windows": [{"start": 40, "end": 52}]}
    known_failure = {"start": 10, "end": 22, "complete_thought": False, "text": "India cricket"}
    result, _, context = run(tmp_path, monkeypatch,
        [{"proposals": [proposal(i) for i in range(4)]}, final((1, 2), ())],
        candidates=[known_failure], context=context)
    assert len(result) == 2
    assert all(item["start"] >= 70 for item in result)


def test_sectioning_covers_every_cue_or_fails_explicitly_without_api_call(tmp_path, monkeypatch):
    from automation import editorial_planner as planner
    segments = material()
    sections = planner._sections(segments, 300, 10)
    assert {cue["cue_id"] for section in sections for cue in section} == set(range(4))
    with pytest.raises(ValueError, match="No partial coverage"):
        planner._sections(segments, 300, 1)


def test_provider_exception_is_retried_with_feedback_not_unbounded_or_downgraded(tmp_path, monkeypatch):
    from automation import editorial_planner as planner
    class ProviderError(Exception):
        pass
    ai = Mock()
    ai.generate_json.side_effect = [ProviderError("transient provider failure"),
        json.dumps({"proposals": [proposal(i) for i in range(4)]}), json.dumps(final())]
    monkeypatch.setattr(planner, "_get_ai", lambda: ai)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    result = planner.curate_clips(material(), [], str(source), config(), {}, tmp_path / "curation.json")
    assert len(result) == 3
    assert ai.generate_json.call_count == 3
    assert "transient provider failure" in ai.generate_json.call_args_list[1].args[0]
    assert all(call.kwargs["reasoning_effort"] == "high" for call in ai.generate_json.call_args_list)


@pytest.mark.parametrize("script", [
    "मजा कल आएगा जब रोहित शर्मा न्यूजीलैंड में शतक लगाएगा।",
    "केएल राहुल ने मैच में अच्छी बल्लेबाजी नहीं की।",
    "ऋषभ पंत का फिटनेस टेस्ट क्यों हो रहा है?",
    "अगर वर्ल्ड कप जितना हो तो नो बॉल फेंक दो।",
])
def test_real_hindi_cricket_proposals_are_not_lost_to_roman_only_gate(script):
    from automation.editorial_planner import _proposal
    tokens = script.split()
    raw = {"start_cue_id": 0, "end_cue_id": 0, "opening_phrase": " ".join(tokens[:3]),
           "closing_phrase": " ".join(tokens[-3:]), "score": 80, "reason": "retained cricket argument"}
    item = _proposal(raw, [{"start": 0, "end": 20, "text": script}], {0}, config(),
                     source_context="India vs New Zealand cricket match")
    assert item["text"] == script  # relevance normalization must not rewrite speech


def test_different_asr_wording_requires_explicit_word_id_edit_then_fresh_final_review(tmp_path, monkeypatch):
    segments = material()
    for segment in segments:
        segment["words"][1]["word"] = "kriket"
    edits = {"cuts": [{"candidate_id": i+1, "first_word_id": 0,
                       "last_word_id": len(segment["words"])-1, "reason": "Complete actual retained sentence"}
                      for i, segment in enumerate(segments)]}
    result, ai, context = run(tmp_path, monkeypatch,
        [{"proposals": [proposal(i) for i in range(4)]}, edits, final()], segments)
    assert len(result) == 3
    assert ai.generate_json.call_count == 3
    assert "kriket" in ai.generate_json.call_args_list[-1].args[0]
    assert all(not item.get("boundary_review_required") for item in result)
    assert all(completeness_value(item) is True for item in result)


@pytest.mark.parametrize("first,last", [(True, 4), (0, 99), (5, 2), (0.5, 8)])
def test_invalid_word_id_repairs_cannot_inherit_or_create_approval(first, last):
    from automation.editorial_planner import _repair_boundaries
    segment = material()[0]
    item = {"start": segment["start"], "end": segment["end"], "speech_segments": [segment],
            "boundary_review_required": True, "boundary_words": segment["words"],
            "opening_phrase": "India cricket", "closing_phrase": "and wins",
            "proposed_source_range": [segment["start"], segment["end"]]}
    requests = Mock()
    requests.ask.return_value = {"cuts": [{"candidate_id": 1, "first_word_id": first,
        "last_word_id": last, "reason": "claimed complete"}]}
    decision = {"alignment_holds": []}
    assert _repair_boundaries([item], requests, config(), decision) == []
    assert decision["alignment_holds"]


def test_final_model_cannot_approve_an_unresolved_word_edit():
    from automation.editorial_planner import _finalize
    item = {"start": 10, "end": 25, "text": "India cricket", "boundary_review_required": True}
    decision = {}
    selected, reserves = _finalize(final((1,), ()), [item], 3, decision)
    assert selected == reserves == []
    assert decision["held_selections"][0]["reason"] == "Unresolved source word boundaries"
