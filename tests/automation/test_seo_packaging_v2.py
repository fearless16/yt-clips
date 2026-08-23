"""Packaging-v2 contracts for cricket Shorts metadata."""

import json


def _queries():
    return [f"yuvraj singh coach india {index}" for index in range(8)]


def test_title_cleanup_removes_shorts_fake_live_and_never_cuts_a_word(monkeypatch):
    import automation.seo.seo as seo

    monkeypatch.setitem(seo.cfg["seo"], "title_max_chars", 30)
    result = seo._enforce_limits({
        "title": "🔴 LIVE | Yuvraj Singh Coach Debate #Shorts Extra Words",
        "description": "Yuvraj Singh coach debate explained clearly.",
        "hashtags": ["#Shorts", "#YuvrajSingh"],
        "search_terms": _queries(),
    })

    assert "live" not in result["title"].casefold()
    assert "#shorts" not in result["title"].casefold()
    assert not result["title"].startswith(("🔴", "|", "-", ":"))
    # v3 packaging appends the channel-neutral emoji when the model omits one.
    assert result["title"] == "Yuvraj Singh Coach Debate 🏏"


def test_packaging_contract_uses_focused_queries_and_scores_promise_alignment(monkeypatch):
    import automation.seo.seo as seo

    queries = _queries()
    generated = {
        "title": "Yuvraj Singh India Coach Kyun Ban Sakte Hain?",
        "description": (
            f"{queries[0]} aur {queries[1]} par yeh clip ek clear opinion deta hai. "
            + "Yuvraj Singh ke cricket brain aur India coaching role ka grounded analysis. " * 30
        ),
        "hashtags": ["#Shorts", "#YuvrajSingh"],
        "search_terms": queries,
        "primary_search_terms": queries[:2],
    }
    monkeypatch.setattr(seo, "extract_ocr_entities", lambda *_: {})
    monkeypatch.setattr(seo, "_attempt_seo_generation", lambda *a, **k: generated)

    result = seo.generate_clip_seo(
        "clip-promise",
        "Yuvraj Singh ko India ka coach banana chahiye kyunki unka cricket brain kamaal hai.",
        video_title="India cricket coaching discussion",
        approved_search_queries=queries,
    )

    assert result["packaging_version"] == "promise_v3_english"
    assert result["primary_search_terms"] == queries[:2]
    assert result["promise_alignment_score"] >= 0.5


def test_packaging_contract_rejects_title_that_promises_another_topic(monkeypatch):
    import automation.seo.seo as seo

    queries = _queries()
    generated = {
        "title": "Stadium Pitch Ka Chhupa Secret",
        "description": (
            f"{queries[0]} aur {queries[1]}. "
            + "Yuvraj Singh coaching analysis with grounded cricket context. " * 30
        ),
        "hashtags": ["#Shorts", "#YuvrajSingh"],
        "search_terms": queries,
        "primary_search_terms": queries[:2],
    }
    monkeypatch.setattr(seo, "extract_ocr_entities", lambda *_: {})
    monkeypatch.setattr(seo, "_attempt_seo_generation", lambda *a, **k: generated)

    try:
        seo.generate_clip_seo(
            "clip-mismatch",
            "Yuvraj Singh ko India ka coach banana chahiye.",
            video_title="India coaching discussion",
            approved_search_queries=queries,
        )
    except seo.SEOGenerationError as exc:
        assert "promise" in str(exc).casefold()
    else:
        raise AssertionError("A title unrelated to the clip promise must be rejected")


def test_exported_clip_researches_each_transcript_and_passes_full_grounding(tmp_path, monkeypatch):
    import automation.seo.seo as seo

    captured = {}
    trend = {
        "topics": ["yuvraj singh coach debate"],
        "scorecard": "",
        "match_facts": ["Verified source fact"],
        "player_names": ["Yuvraj Singh"],
        "player_aliases": {"yuvi": "Yuvraj Singh"},
        "search_queries": _queries(),
        "sources": [{"kind": "youtube_search", "url": "https://youtube.com/results"}],
        "teams": ["India"],
    }

    def fake_research(**kwargs):
        captured["research"] = kwargs
        return trend

    def fake_generate(**kwargs):
        captured["generate"] = kwargs
        return {
            "title": "Yuvraj Singh Coach Debate",
            "description": "grounded " * 200,
            "hashtags": ["#Shorts", "#YuvrajSingh"],
            "search_terms": _queries(),
            "primary_search_terms": _queries()[:2],
            "ai_generated": True,
            "packaging_version": "promise_v2",
        }

    monkeypatch.setattr(seo, "get_trending_context", fake_research)
    monkeypatch.setattr(seo, "generate_clip_seo", fake_generate)
    seo.generate_seo_for_exported_clip(
        "clip1",
        "Yuvi ko India coach banana chahiye",
        str(tmp_path),
        video_title="India cricket discussion",
        video_description="A detailed source description",
    )

    assert captured["research"]["transcript"].startswith("Yuvi")
    assert captured["research"]["video_description"] == "A detailed source description"
    assert captured["research"]["include_live_stream_url"] is False
    assert captured["generate"]["approved_search_queries"] == _queries()
    assert captured["generate"]["grounded_players"] == ["Yuvraj Singh"]
    assert captured["generate"]["research_sources"] == trend["sources"]


def test_retry_preserves_the_complete_evidence_pack(tmp_path, monkeypatch):
    import automation.seo.seo as seo

    marker = {
        "clip_id": "clip1",
        "transcript": "Yuvi ko coach banana chahiye",
        "video_title": "India cricket discussion",
        "video_description": "Source context",
        "scorecard": "Verified scorecard",
        "trend_topics": ["coach debate"],
        "teams": ["India"],
        "approved_search_queries": _queries(),
        "match_facts": ["Verified fact"],
        "grounded_players": ["Yuvraj Singh"],
        "grounded_aliases": {"yuvi": "Yuvraj Singh"},
        "research_sources": [{"kind": "match", "url": "https://example.test"}],
        "is_shorts": True,
    }
    (tmp_path / "clip1_seo_failed.json").write_text(json.dumps(marker), encoding="utf-8")
    captured = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return {"title": "ok", "description": "ok"}

    monkeypatch.setattr(seo, "generate_clip_seo", fake_generate)
    assert seo.retry_failed_seo(str(tmp_path)) == {"recovered": 1, "total": 1}
    assert captured["video_description"] == "Source context"
    assert captured["approved_search_queries"] == _queries()
    assert captured["grounded_aliases"] == {"yuvi": "Yuvraj Singh"}
    assert captured["research_sources"] == marker["research_sources"]


def test_provider_override_is_forwarded_without_mutating_the_shared_client(monkeypatch):
    import automation.seo.seo as seo

    captured = {}

    class FakeAI:
        def generate_text(self, **kwargs):
            captured.update(kwargs)
            return json.dumps({
                "title": "Yuvraj Singh Coach Debate",
                "description": "Yuvraj Singh grounded coaching discussion. " * 40,
                "hashtags": ["#Shorts", "#YuvrajSingh"],
                "search_terms": _queries(),
            })

        def get_used_provider(self):
            return "openrouter"

        def get_used_model(self):
            return "model-x"

    monkeypatch.setattr(seo, "_get_ai", lambda: FakeAI())
    result = seo._generate_ai_seo(
        "clip-provider",
        "prompt",
        "Yuvraj Singh coach",
        True,
        provider_override="openrouter",
    )

    assert captured["prefer_provider"] == "openrouter"
    assert result["provider"] == "openrouter"
    assert result["model"] == "model-x"


def test_invalid_alignment_config_falls_back_instead_of_crashing(monkeypatch):
    import automation.seo.seo as seo

    queries = _queries()
    monkeypatch.setitem(seo.cfg["seo"], "min_promise_alignment_score", "broken")
    monkeypatch.setattr(seo, "extract_ocr_entities", lambda *_: {})
    monkeypatch.setattr(seo, "_attempt_seo_generation", lambda *a, **k: {
        "title": "Yuvraj Singh Coach Debate",
        "description": (
            f"{queries[0]} aur {queries[1]}. "
            + "Yuvraj Singh coach opinion explained with grounded context. " * 30
        ),
        "hashtags": ["#Shorts", "#YuvrajSingh"],
        "search_terms": queries,
        "primary_search_terms": queries[:2],
    })

    result = seo.generate_clip_seo(
        "clip-config",
        "Yuvraj Singh ko coach banana chahiye",
        video_title="India cricket discussion",
        approved_search_queries=queries,
    )
    assert result["packaging_version"] == "promise_v3_english"


def test_unknown_player_in_title_is_replaced_with_grounded_clip_topic(monkeypatch):
    import automation.seo.seo as seo

    queries = [f"new zealand captaincy debate {index}" for index in range(8)]
    monkeypatch.setattr(seo, "extract_ocr_entities", lambda *_: {})
    monkeypatch.setattr(seo, "_attempt_seo_generation", lambda *a, **k: {
        "title": "Saud Shakeel Captaincy Shock",
        "description": "New Zealand captaincy debate explained. " * 50,
        "hashtags": ["#Shorts", "#NewZealandCricket"],
        "search_terms": queries,
        "primary_search_terms": queries[:2],
    })

    result = seo.generate_clip_seo(
        "clip-grounded-title",
        "New Zealand cricket ki captaincy par bada sawaal hai",
        video_title="Cricket discussion",
        approved_search_queries=queries,
    )

    assert "saud shakeel" not in result["title"].casefold()
    assert "new zealand captaincy" in result["title"].casefold()


def test_match_roster_player_cannot_be_attributed_when_clip_never_says_name(monkeypatch):
    import automation.seo.seo as seo

    queries = [f"england pakistan test analysis {index}" for index in range(8)]
    monkeypatch.setattr(seo, "extract_ocr_entities", lambda *_: {})
    monkeypatch.setattr(seo, "_attempt_seo_generation", lambda *a, **k: {
        "title": "Joe Root Run Rate Debate",
        "description": "Joe Root is batting aggressively against Pakistan. " * 40,
        "hashtags": ["#Shorts", "#Cricket"],
        "search_terms": queries,
        "primary_search_terms": queries[:2],
    })

    result = seo.generate_clip_seo(
        "clip-no-player",
        "England ka run rate 41 chal raha hai",
        video_title="England vs Pakistan Test",
        approved_search_queries=queries,
        grounded_players=["Joe Root"],
    )

    rendered = f"{result['title']} {result['description']}".casefold()
    assert "joe root" not in rendered
    assert "england run rate 41" in result["title"].casefold()
