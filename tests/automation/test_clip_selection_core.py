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
