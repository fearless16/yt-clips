# -*- coding: utf-8 -*-
"""LLM-assisted entity grounding: contract tests."""

import json
from unittest.mock import patch

import pytest


@pytest.fixture()
def grounding(monkeypatch):
    import automation.seo.entity_grounding as eg

    class FakeAI:
        def __init__(self, response):
            self.response = response

        def generate_text(self, prompt, system_instruction=None,
                          prefer_model=None, prefer_provider=None):
            return self.response

    def install(response):
        monkeypatch.setattr(eg, "AIClient", lambda: FakeAI(response))
        return eg

    return install


def test_extract_parses_vouched_entities(grounding, monkeypatch):
    eg = grounding(json.dumps({
        "players": ["Bumrah"],
        "teams": ["India", "Sri Lanka"],
        "topic_phrases": ["legal shot", "bat grip"],
    }))
    monkeypatch.setenv("YT_CLIPS_LLM_GROUNDING", "1")
    out = eg.extract_grounded_entities_llm(
        "c1", "bumaraaha kee genda par legal shot kaa bahas")
    assert out["players"] == ["Bumrah"]
    assert "India" in out["teams"]
    assert "legal shot" in out["topic_phrases"]


def test_extract_fail_soft_on_garbage(grounding):
    eg = grounding("I cannot answer in JSON today sorry")
    assert eg.extract_grounded_entities_llm("c2", "kuch bhi") == {}


def test_extract_fail_soft_on_exception(grounding):
    import automation.seo.entity_grounding as eg_mod

    class BoomAI:
        def generate_text(self, *a, **k):
            raise RuntimeError("provider down")

    monkey = eg_mod
    orig = monkey.AIClient
    try:
        monkey.AIClient = lambda: BoomAI()
        assert eg_mod.extract_grounded_entities_llm("c3", "x") == {}
    finally:
        monkey.AIClient = orig


def test_title_person_vouched_when_topic_phrase_covers_it():
    from automation.seo.entity_grounding import name_vouched_by_topics

    assert name_vouched_by_topics("Legal Shot", ["legal shot debate", "grip"])
    assert name_vouched_by_topics("Bat Grip", ["the grip discussion"])
    assert not name_vouched_by_topics("Saud Shakeel", ["legal shot", "grip"])


def test_generate_clip_seo_passes_with_llm_vouched_team(tmp_path, monkeypatch):
    """Regression: 'ungrounded entities India' when static catalog missed it."""
    import automation.seo.seo as seo
    import automation.clip_selection.content_type  # noqa: F401

    queries = [f"ind vs sl test day four clip {i}" for i in range(8)]
    messy_transcript = (
        "bhaaee yahaan spina viketa kee baata ho rahee hai aur inhonne "
        "hama log ko daboch liyaa thaa agara ye jaaari rahaa to"
    )
    ai_json = json.dumps({
        "title": "India spin trap exposed on day four 🏏",
        "description": (
            "India's spinners tightened the noose on day four of this test. "
            "Sri Lanka's lower order had no answer to the turning ball and "
            "the required runs kept climbing with every maiden over."
        ),
        "hashtags": ["#Shorts", "#INDvSL"],
        "search_terms": queries,
        "primary_search_terms": queries[:3],
        "tags": ["india", "sri lanka", "test cricket"],
    })

    monkeypatch.setattr(
        seo, "_attempt_seo_generation",
        lambda *a, **k: json.loads(ai_json),
    )
    monkeypatch.setattr(
        seo, "extract_grounded_entities_llm",
        lambda clip_id, transcript, video_title="", video_description="": {
            "players": [],
            "teams": ["India", "Sri Lanka"],
            "topic_phrases": ["spin trap", "day four"],
        },
    )

    result = seo.generate_clip_seo(
        "clip_llm_ground",
        messy_transcript,
        video_title="IND vs SL Live match today Day 4",
        approved_search_queries=queries,
    )
    assert result["packaging_version"] == "promise_v3_english"
    assert "India" in result["description"]
