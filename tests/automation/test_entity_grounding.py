# -*- coding: utf-8 -*-
"""LLM-assisted entity grounding: contract tests."""

import json
import re

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




def test_audit_written_copy_llm_parses_and_fails_soft(grounding, monkeypatch):
    eg = grounding(json.dumps({
        "unsupported_entities": ["Ravindra Jadeja"],
        "supported_topics": ["Massive Target"],
    }))
    monkeypatch.setenv("YT_CLIPS_LLM_GROUNDING", "1")
    out = eg.audit_written_copy_llm(
        "c9", "india ne bada score banaya", title="India Huge Total",
        description="Ravindra Jadeja praised the innings.")
    assert out["unsupported_entities"] == ["Ravindra Jadeja"]
    assert out["supported_topics"] == ["massive target"]

    bad = grounding("no json here {{{")
    assert bad.audit_written_copy_llm(
        "c10", "t", title="T", description="D") == {
        "unsupported_entities": [], "supported_topics": []}


def test_kill_switch_disables_both_calls(grounding, monkeypatch):
    eg = grounding(json.dumps({"players": ["X"]}))
    monkeypatch.setenv("YT_CLIPS_LLM_GROUNDING", "0")
    assert eg.extract_grounded_entities_llm("c11", "t") == {}
    assert eg.audit_written_copy_llm(
        "c11", "t", title="T", description="D")["unsupported_entities"] == []








def test_builder_prefers_canonical_subjects_over_phrase_noise():
    """Deterministic query subjects must be canonical teams even when a
    polluted runtime roster carries capitalized phrase noise."""
    from automation.seo.context_engine import build_grounded_search_queries

    out = build_grounded_search_queries(
        "IND vs SL Live Day 4 | Massive Target Chased",
        "Sri Lanka need 288 more runs.",
        "spine atak gaya bhaaee",
        suggestions=[],
        player_names=["Massive Target", "Ravindra Jadeja"],
    )
    assert out, "builder returned nothing"
    assert all("massive target" not in q.lower() for q in out)
    assert any("sri lanka" in q.lower() or "india" in q.lower() for q in out)






