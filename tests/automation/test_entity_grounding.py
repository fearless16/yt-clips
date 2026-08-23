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
    assert result["packaging_version"] == "promise_v4_longtail"
    assert "India" in result["description"]


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


def test_copy_audit_scrubs_hallucinated_player(monkeypatch):
    """Names the copy-audit marks unsupported are scrubbed before validation."""
    import automation.seo.seo as seo

    monkeypatch.setattr(seo, "extract_grounded_entities_llm", lambda *a, **k: {
        "players": [], "teams": ["India"], "topic_phrases": ["grip debate"],
    })
    monkeypatch.setattr(seo, "audit_written_copy_llm", lambda *a, **k: {
        "unsupported_entities": ["Ravindra Jadeja"],
        "supported_topics": ["massive target"],
    })
    queries = [f"india batting grip debate {i}" for i in range(8)]
    monkeypatch.setattr(seo, "extract_ocr_entities", lambda *_: {})
    monkeypatch.setattr(seo, "_attempt_seo_generation", lambda *a, **k: {
        "title": "India Batting Grip Debate 🏏",
        "description": (
            f"{queries[0]} aur {queries[1]}. Massive Target discussed. "
            "Ravindra Jadeja opinion on the batting grip with context. " * 12
        ),
        "hashtags": ["#Shorts", "#Cricket"],
        "search_terms": queries,
        "primary_search_terms": queries[:2],
    })

    result = seo.generate_clip_seo(
        "clip-audit",
        "batting grip ke baare mein baat",
        video_title="India cricket discussion",
        approved_search_queries=queries,
    )
    copy = (result["title"] + " " + result["description"]).casefold()
    assert "jadeja" not in copy


def test_audit_scrubs_contaminated_approved_queries(monkeypatch):
    """Upstream research can bake a hallucinated name into approved queries.

    Scrubbed queries must never reach search_terms or the rebuilt fallback
    copy; clean ones survive.
    """
    import automation.seo.seo as seo

    monkeypatch.setattr(seo, "extract_grounded_entities_llm", lambda *a, **k: {
        "players": [], "teams": ["India"], "topic_phrases": ["grip debate"],
    })
    monkeypatch.setattr(seo, "audit_written_copy_llm", lambda *a, **k: {
        "unsupported_entities": ["Ravindra Jadeja"],
        "supported_topics": [],
    })
    clean = [f"india batting grip debate {i}" for i in range(8)]
    dirty = ["ravindra jadeja six reaction", "ravindra jadeja fielding"]
    queries = dirty[:1] + clean[:4] + dirty[1:] + clean[4:]
    monkeypatch.setattr(seo, "extract_ocr_entities", lambda *_: {})
    monkeypatch.setattr(seo, "_attempt_seo_generation", lambda *a, **k: {
        "title": "India Batting Grip Debate 🏏",
        "description": (
            f"{queries[0]} aur {queries[1]}. "
            "Ravindra Jadeja opinion on the batting grip with context. " * 12
        ),
        "hashtags": ["#Shorts", "#Cricket"],
        "search_terms": queries,
        "primary_search_terms": queries[:2],
    })

    result = seo.generate_clip_seo(
        "clip-audit-queries",
        "batting grip ke baare mein baat",
        video_title="India cricket discussion",
        approved_search_queries=queries,
    )
    everything = json.dumps(result, ensure_ascii=False).casefold()
    assert "jadeja" not in everything
    assert len(result["search_terms"]) >= 8
    assert all("jadeja" not in term for term in result["search_terms"])


def test_all_dirty_queries_rebuilt_from_evidence(monkeypatch):
    """When every approved query carries the hallucinated name, rebuild a
    minimum grounded set from the source title and vouched teams."""
    import automation.seo.seo as seo

    monkeypatch.setattr(seo, "extract_grounded_entities_llm", lambda *a, **k: {
        "players": [], "teams": ["India", "Sri Lanka"],
        "topic_phrases": ["grip debate"],
    })
    monkeypatch.setattr(seo, "audit_written_copy_llm", lambda *a, **k: {
        "unsupported_entities": ["Ravindra Jadeja"],
        "supported_topics": [],
    })
    dirty = [f"ravindra jadeja six {i}" for i in range(10)]
    monkeypatch.setattr(seo, "extract_ocr_entities", lambda *_: {})
    monkeypatch.setattr(seo, "_attempt_seo_generation", lambda *a, **k: {
        "title": "India Batting Grip Debate 🏏",
        "description": (
            f"{dirty[0]} aur {dirty[1]}. "
            "Ravindra Jadeja opinion on the batting grip with context. " * 12
        ),
        "hashtags": ["#Shorts", "#Cricket"],
        "search_terms": list(dirty),
        "primary_search_terms": dirty[:2],
    })

    result = seo.generate_clip_seo(
        "clip-audit-rebuild",
        "batting grip ke baare mein baat",
        video_title="IND vs SL Live Day 4 Sri Lanka 84/4 Target 372 India Innings",
        approved_search_queries=list(dirty),
    )
    everything = json.dumps(result, ensure_ascii=False).casefold()
    assert "jadeja" not in everything
    assert len(result["search_terms"]) >= 8
    joined = " ".join(result["search_terms"]).casefold()
    assert "sri lanka" in joined or "india" in joined


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


def test_scrub_gutted_title_falls_back_to_evidence_promise(monkeypatch):
    """When scrubbing the hallucinated name empties the title's promise,
    an evidence-based fallback title replaces it instead of failing."""
    import automation.seo.seo as seo

    monkeypatch.setattr(seo, "extract_grounded_entities_llm", lambda *a, **k: {
        "players": [], "teams": ["India", "Sri Lanka"], "topic_phrases": [],
    })
    monkeypatch.setattr(seo, "audit_written_copy_llm", lambda *a, **k: {
        "unsupported_entities": ["Ravindra Jadeja"],
        "supported_topics": ["jadeja six magic"],
    })
    queries = [f"sri lanka six reaction {i}" for i in range(8)]
    monkeypatch.setattr(seo, "extract_ocr_entities", lambda *_: {})
    monkeypatch.setattr(seo, "_attempt_seo_generation", lambda *a, **k: {
        # Entire promise rides on the hallucinated name
        "title": "Ravindra Jadeja Six Magic",
        "description": (
            f"{queries[0]} aur {queries[1]}. "
            "Ravindra Jadeja hit a huge six and the crowd erupted. " * 10
        ),
        "hashtags": ["#Shorts", "#Cricket"],
        "search_terms": queries,
        "primary_search_terms": queries[:2],
    })

    result = seo.generate_clip_seo(
        "clip-title-fix",
        "chhakka gaya sab shock mein",
        video_title="IND vs SL Live Day 4",
        approved_search_queries=queries,
    )
    everything = json.dumps(result, ensure_ascii=False).casefold()
    assert "jadeja" not in everything
    assert result["title"].strip()


def test_ungrounded_title_triggers_llm_repair_not_template(monkeypatch):
    """v4 policy: no deterministic template fallback. An ungrounded name in
    the title triggers one corrective LLM pass; failure raises loudly."""
    import automation.seo.seo as seo

    monkeypatch.setattr(seo, "extract_ocr_entities", lambda *_: {})
    monkeypatch.setattr(seo, "extract_grounded_entities_llm", lambda *a, **k: {
        "players": [], "teams": ["India"], "topic_phrases": ["grip debate"],
    })
    monkeypatch.setattr(seo, "audit_written_copy_llm", lambda *a, **k: {
        "unsupported_entities": [], "supported_topics": [],
    })
    queries = [f"india batting grip debate {i}" for i in range(8)]

    def writer(clip_id, user_prompt, transcript, video_title, is_shorts,
               provider_override=None, model_override=None,
               sys_instruction=None, salvage_tmpl=None):
        if "CORRECTION REQUIRED" in user_prompt:
            # Repair pass: clean title, long keyword-rich body
            return {
                "title": "India Batting Grip Debate 🏏",
                "description": (
                    f"{queries[0]} aur {queries[1]}. "
                    "India batting grip debate explained. " * 45
                ),
                "hashtags": ["#Shorts", "#Cricket"],
                "search_terms": queries,
                "primary_search_terms": queries[:4],
            }
        # First pass: hallucinated celebrity name in title
        return {
            "title": "Ravindra Jadeja Grip Verdict",
            "description": (
                f"{queries[0]} aur {queries[1]}. "
                "India batting grip debate explained. " * 45
            ),
            "hashtags": ["#Shorts", "#Cricket"],
            "search_terms": queries,
            "primary_search_terms": queries[:4],
        }

    calls = {"n": 0}

    def counting_writer(*a, **k):
        calls["n"] += 1
        return writer(*a, **k)

    monkeypatch.setattr(seo, "_attempt_seo_generation", counting_writer)

    result = seo.generate_clip_seo(
        "clip-repair",
        "batting grip ke baare mein baat",
        video_title="India cricket discussion",
        approved_search_queries=queries,
    )
    assert calls["n"] == 2, "repair must issue exactly one corrective LLM call"
    assert "jadeja" not in json.dumps(result).casefold()
    assert "promise_v4_longtail" == result["packaging_version"]


def test_repair_failure_raises_instead_of_template_junk(monkeypatch):
    """If the corrective LLM pass also violates policy, generation fails
    loudly — no silent template fallback exists anymore."""
    import automation.seo.seo as seo

    monkeypatch.setattr(seo, "extract_ocr_entities", lambda *_: {})
    monkeypatch.setattr(seo, "extract_grounded_entities_llm", lambda *a, **k: {
        "players": [], "teams": ["India"], "topic_phrases": ["grip debate"],
    })
    monkeypatch.setattr(seo, "audit_written_copy_llm", lambda *a, **k: {
        "unsupported_entities": [], "supported_topics": [],
    })
    queries = [f"india batting grip debate {i}" for i in range(8)]

    def always_bad(*a, **k):
        return {
            "title": "Ravindra Jadeja Grip Verdict",
            "description": (
                f"{queries[0]} aur {queries[1]}. "
                "Ravindra Jadeja verdict on the India batting grip. " * 40
            ),
            "hashtags": ["#Shorts", "#Cricket"],
            "search_terms": queries,
            "primary_search_terms": queries[:4],
        }

    monkeypatch.setattr(seo, "_attempt_seo_generation", always_bad)

    with pytest.raises(seo.SEOGenerationError):
        seo.generate_clip_seo(
            "clip-repair-fail",
            "batting grip ke baare mein baat",
            video_title="India cricket discussion",
            approved_search_queries=queries,
        )
