from datetime import datetime

from automation.clip_selection.agents import ALL_AGENTS
from automation.clip_selection.cricket_heuristics import event_intensity
from automation.clip_selection.hook_auditor import analyze_clip_hook
from automation.clip_selection.selector import ClipSelector
from automation.clip_selection.topic_segmenter import TopicSegmenter
from automation.scheduling import next_upload_day


def test_registered_agents_score_one_cricket_candidate():
    candidate = {
        "start": 10.0,
        "end": 22.0,
        "text": "What a six by Kohli! India wins this unbelievable cricket moment.",
    }
    result = ClipSelector(use_llm_arbiter=False).score_candidates(
        [candidate],
        {"rms_map": {}, "avg_rms": 0.0, "max_rms": 0.0, "topics": []},
    )[0]

    assert set(result["agent_scores"]) == {agent.name for agent in ALL_AGENTS}
    assert 0 <= result["final_score"] <= 100


def test_selector_honors_hard_rejection():
    result = ClipSelector(use_llm_arbiter=False).score_candidates(
        [{"start": 0.0, "end": 2.0, "text": "single"}],
        {"rms_map": {}, "avg_rms": 0.0, "max_rms": 0.0, "topics": []},
    )[0]
    assert result["should_reject"] is True


def test_hook_and_cricket_heuristics_detect_instant_payoff():
    hook = analyze_clip_hook(
        clip_path="missing.mp4",
        transcript_text="SIX! Kohli smashes it out of the ground!",
        start_sec=0,
        clip_duration=12,
    )
    assert hook["hook_score"] > 0
    assert event_intensity("Kohli hits a six and India wins!")["total_events"] > 0


def test_topic_segmenter_empty_input_is_offline_safe():
    assert TopicSegmenter().segment([]) == []


def test_schedule_never_returns_configured_dead_priority_day():
    config = {
        "upload_schedule": {
            "avoid_days": ["thursday", "friday"],
            "priority_days": ["friday"],
        }
    }
    result = next_upload_day(datetime(2026, 6, 4, 12), config)
    assert result.strftime("%A").lower() == "saturday"


def test_llm_arbiter_does_not_refill_candidates_it_rejected(monkeypatch):
    import automation.clip_selection.arbiter as arbiter

    class FakeAI:
        def generate_text(self, *args, **kwargs):
            return '{"selected":[{"candidate_id":2,"score":91,"reason":"complete"}],"rejected":[]}'

    monkeypatch.setattr(arbiter, "_get_ai", lambda: FakeAI())
    candidates = [
        {"start": 0.0, "end": 8.0, "text": "one", "final_score": 80},
        {"start": 10.0, "end": 22.0, "text": "two", "final_score": 79},
        {"start": 30.0, "end": 40.0, "text": "three", "final_score": 78},
    ]

    result = arbiter.llm_arbiter_refine(
        candidates,
        {"transcript_segments": []},
        max_selected=3,
    )

    assert [item["text"] for item in result] == ["two"]


def test_llm_arbiter_receives_source_video_title(monkeypatch):
    import automation.clip_selection.arbiter as arbiter

    captured = {}

    class FakeAI:
        def generate_text(self, prompt, system_instruction=None):
            captured["prompt"] = prompt
            captured["system"] = system_instruction
            return '{"selected":[{"candidate_id":1,"score":90,"reason":"on-match"}]}'

    monkeypatch.setattr(arbiter, "_get_ai", lambda: FakeAI())
    candidates = [
        {"start": 1.0, "end": 12.0, "text": "Pakistan need wickets against England", "final_score": 50, "agent_scores": {}},
        {"start": 20.0, "end": 32.0, "text": "Ireland discussion", "final_score": 49, "agent_scores": {}},
    ]

    arbiter.llm_arbiter_refine(
        candidates,
        {"transcript_segments": [], "source_title": "England vs Pakistan 1st Test"},
        max_selected=2,
    )

    assert "England vs Pakistan 1st Test" in captured["prompt"]
    assert "directly match the source event" in captured["system"]


LONG_SCRIPT_TAIL = "so the plan stayed alive till the very last over of this complete thought"


def test_llm_arbiter_receives_full_candidate_script(monkeypatch):
    import automation.clip_selection.arbiter as arbiter

    captured = {}

    class FakeAI:
        def generate_text(self, prompt, system_instruction=None):
            captured["prompt"] = prompt
            return '{"selected":[{"candidate_id":1,"score":90,"reason":"full"}]}'

    monkeypatch.setattr(arbiter, "_get_ai", lambda: FakeAI())
    long_script = (
        "Rishabh Pant walks in at number six and the crowd is on its feet, "
        "he charges the second ball through extra cover for four, next ball "
        "is short and pulled behind square, the bowler changes his field, "
        f"{LONG_SCRIPT_TAIL}."
    )
    assert len(long_script) > 150
    candidates = [
        {"start": 1.0, "end": 25.0, "text": long_script, "final_score": 60, "agent_scores": {}},
        {"start": 40.0, "end": 48.0, "text": "short filler line", "final_score": 59, "agent_scores": {}},
    ]

    arbiter.llm_arbiter_refine(
        candidates,
        {"transcript_segments": []},
        max_selected=2,
    )

    assert LONG_SCRIPT_TAIL in captured["prompt"]


def test_llm_arbiter_prompt_states_candidate_id_contract(monkeypatch):
    import automation.clip_selection.arbiter as arbiter

    captured = {}

    class FakeAI:
        def generate_text(self, prompt, system_instruction=None):
            captured["prompt"] = prompt
            captured["system"] = system_instruction
            return '{"selected":[{"candidate_id":1,"score":90,"reason":"ok"}]}'

    monkeypatch.setattr(arbiter, "_get_ai", lambda: FakeAI())
    candidates = [
        {"start": 1.0, "end": 9.0, "text": "Bumrah bowls a maiden over", "final_score": 55, "agent_scores": {}},
        {"start": 15.0, "end": 22.0, "text": "crowd waits quietly", "final_score": 54, "agent_scores": {}},
    ]

    arbiter.llm_arbiter_refine(candidates, {"transcript_segments": []}, max_selected=1)

    combined = captured["prompt"] + "\n" + (captured["system"] or "")
    assert "integer" in combined.lower()
    assert "candidate_id" in combined


def test_llm_arbiter_receives_match_context(monkeypatch):
    import automation.clip_selection.arbiter as arbiter

    captured = {}

    class FakeAI:
        def generate_text(self, prompt, system_instruction=None):
            captured["prompt"] = prompt
            return '{"selected":[{"candidate_id":1,"score":90,"reason":"ok"}]}'

    monkeypatch.setattr(arbiter, "_get_ai", lambda: FakeAI())
    candidates = [
        {"start": 1.0, "end": 11.0, "text": "Pant reverse sweeps for six", "final_score": 58, "agent_scores": {}},
        {"start": 20.0, "end": 28.0, "text": "pitch report chatter", "final_score": 57, "agent_scores": {}},
    ]
    context = {
        "transcript_segments": [],
        "match_context": {
            "teams": ["India", "Australia"],
            "event": "Border Gavaskar Trophy 1st Test",
        },
    }

    arbiter.llm_arbiter_refine(candidates, context, max_selected=1)

    assert "Australia" in captured["prompt"]
    assert "Border Gavaskar Trophy 1st Test" in captured["prompt"]


def test_llm_arbiter_retries_once_on_unparseable_json(monkeypatch):
    import automation.clip_selection.arbiter as arbiter

    class FlakyAI:
        def __init__(self):
            self.calls = 0

        def generate_text(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return "I think clip one is nice but let me explain without JSON"
            return '{"selected":[{"candidate_id":1,"score":88,"reason":"second try"}]}'

    ai = FlakyAI()
    monkeypatch.setattr(arbiter, "_get_ai", lambda: ai)
    candidates = [
        {"start": 1.0, "end": 12.0, "text": "Siraj cleans up the tail", "final_score": 52, "agent_scores": {}},
        {"start": 20.0, "end": 30.0, "text": "drinks break chatter", "final_score": 51, "agent_scores": {}},
    ]

    result = arbiter.llm_arbiter_refine(candidates, {"transcript_segments": []}, max_selected=2)

    assert ai.calls == 2
    assert [item["text"] for item in result] == ["Siraj cleans up the tail"]
    assert result[0]["ai_score"] == 88


def test_selector_select_returns_empty_when_nothing_passes_quality():
    selector = ClipSelector(use_llm_arbiter=False)
    candidates = [{"start": 0.0, "end": 6.0, "text": "quiet pitch talk today"}]
    scored = selector.score_candidates(
        candidates, {"rms_map": {}, "avg_rms": 0.0, "max_rms": 0.0, "topics": []}
    )
    assert selector.select(scored, {}, max_selected=3, min_quality=99.0) == []


# ── Adversarial: LLM returns garbage shapes ─────────────────────────────────


def _two_candidates():
    return [
        {"start": 1.0, "end": 12.0, "text": "Siraj cleans up the tail", "final_score": 52, "agent_scores": {}},
        {"start": 20.0, "end": 30.0, "text": "Kohli reviews the LBW", "final_score": 51, "agent_scores": {}},
    ]


def test_llm_arbiter_handles_string_candidate_id(monkeypatch):
    import automation.clip_selection.arbiter as arbiter

    class FakeAI:
        def generate_text(self, *args, **kwargs):
            return '{"selected":[{"candidate_id":"2","score":77,"reason":"string id"}]}'

    monkeypatch.setattr(arbiter, "_get_ai", lambda: FakeAI())

    result = arbiter.llm_arbiter_refine(_two_candidates(), {"transcript_segments": []}, max_selected=2)

    assert [item["text"] for item in result] == ["Kohli reviews the LBW"]


def test_llm_arbiter_skips_out_of_range_and_nondict_selections(monkeypatch):
    import automation.clip_selection.arbiter as arbiter

    class FakeAI:
        def generate_text(self, *args, **kwargs):
            return (
                '{"selected":['
                '{"candidate_id":99,"score":99,"reason":"out of range"},'
                '"just a string entry",'
                '{"candidate_id":1,"score":70,"reason":"valid"}'
                "]}"
            )

    monkeypatch.setattr(arbiter, "_get_ai", lambda: FakeAI())

    result = arbiter.llm_arbiter_refine(_two_candidates(), {"transcript_segments": []}, max_selected=2)

    assert [item["text"] for item in result] == ["Siraj cleans up the tail"]
    assert result[0]["ai_reason"] == "valid"


def test_llm_arbiter_dedupes_duplicate_candidate_ids(monkeypatch):
    import automation.clip_selection.arbiter as arbiter

    class FakeAI:
        def generate_text(self, *args, **kwargs):
            return (
                '{"selected":['
                '{"candidate_id":1,"score":80,"reason":"first"},'
                '{"candidate_id":1,"score":81,"reason":"dup"}'
                "]}"
            )

    monkeypatch.setattr(arbiter, "_get_ai", lambda: FakeAI())

    result = arbiter.llm_arbiter_refine(_two_candidates(), {"transcript_segments": []}, max_selected=2)

    assert [item["text"] for item in result] == ["Siraj cleans up the tail"]


def test_llm_arbiter_falls_back_when_both_attempts_malformed(monkeypatch):
    import automation.clip_selection.arbiter as arbiter

    class BadJSONAI:
        def __init__(self):
            self.calls = 0

        def generate_text(self, *args, **kwargs):
            self.calls += 1
            return '{"selected": [{"candidate_id": '  # braces present, never valid

    ai = BadJSONAI()
    monkeypatch.setattr(arbiter, "_get_ai", lambda: ai)
    candidates = _two_candidates()

    result = arbiter.llm_arbiter_refine(candidates, {"transcript_segments": []}, max_selected=2)

    assert ai.calls == 2
    assert [item["text"] for item in result] == [
        "Siraj cleans up the tail",
        "Kohli reviews the LBW",
    ]
    assert "ai_score" not in result[0] or result[0]["ai_score"] == candidates[0]["final_score"]


def test_llm_arbiter_survives_exception_then_succeeds(monkeypatch):
    import automation.clip_selection.arbiter as arbiter

    class CrashThenOkAI:
        def __init__(self):
            self.calls = 0

        def generate_text(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("provider socket exploded")
            return '{"selected":[{"candidate_id":2,"score":66,"reason":"after crash"}]}'

    ai = CrashThenOkAI()
    monkeypatch.setattr(arbiter, "_get_ai", lambda: ai)

    result = arbiter.llm_arbiter_refine(_two_candidates(), {"transcript_segments": []}, max_selected=2)

    assert ai.calls == 2
    assert [item["text"] for item in result] == ["Kohli reviews the LBW"]


def test_llm_arbiter_survives_unserializable_match_context(monkeypatch):
    import automation.clip_selection.arbiter as arbiter

    captured = {}

    class FakeAI:
        def generate_text(self, prompt, system_instruction=None):
            captured["prompt"] = prompt
            return '{"selected":[{"candidate_id":1,"score":60,"reason":"ok"}]}'

    monkeypatch.setattr(arbiter, "_get_ai", lambda: FakeAI())
    context = {
        "transcript_segments": [],
        "match_context": {"teams": {"India", "Australia"}, "when": datetime(2026, 8, 23)},
    }

    result = arbiter.llm_arbiter_refine(_two_candidates(), context, max_selected=2)

    assert [item["text"] for item in result]
    assert "Here are 2 scored candidates" in captured["prompt"]


def test_llm_arbiter_caps_pathological_script_with_marker(monkeypatch):
    import automation.clip_selection.arbiter as arbiter

    captured = {}

    class FakeAI:
        def generate_text(self, prompt, system_instruction=None):
            captured["prompt"] = prompt
            return '{"selected":[{"candidate_id":1,"score":60,"reason":"ok"}]}'

    monkeypatch.setattr(arbiter, "_get_ai", lambda: FakeAI())
    monster_script = "bla " * 1000  # ~4000 chars pathological outlier
    candidates = [
        {"start": 1.0, "end": 20.0, "text": monster_script, "final_score": 50, "agent_scores": {}},
        {"start": 30.0, "end": 40.0, "text": "normal short line", "final_score": 49, "agent_scores": {}},
    ]

    arbiter.llm_arbiter_refine(candidates, {"transcript_segments": []}, max_selected=2)

    assert "[script truncated]" in captured["prompt"]
