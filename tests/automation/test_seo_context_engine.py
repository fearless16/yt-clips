"""TDD for the cricket-only SEO evidence and research pipeline."""

from unittest.mock import Mock


def _approved_queries():
    return [
        "yuvraj singh coach",
        "yuvraj singh india coach",
        "yuvi coach india",
        "yuvraj singh coaching debate",
        "team india coach discussion",
        "yuvraj singh cricket brain",
        "yuvraj singh mentor india",
        "should yuvraj singh coach india",
    ]


def _long_description(queries=None):
    queries = queries or _approved_queries()
    opening = (
        "Yuvraj Singh aur Team India coaching debate par yeh complete Hinglish "
        "cricket discussion hai. "
    )
    embedded = ". ".join(
        f"Fans searching for {query} will find the exact opinion explained here"
        for query in queries
    )
    return (opening + embedded + ". " + opening * 20).strip()


def test_download_metadata_keeps_source_description_and_context():
    from download import _video_metadata_payload

    payload = _video_metadata_payload({
        "title": "India vs Sri Lanka Test Watch Along",
        "description": "Full Hindi watch-along and post-match discussion.",
        "channel": "Cricket With Prajjwal",
        "upload_date": "20260820",
        "duration": 14400,
        "tags": ["india vs sri lanka", "test cricket"],
    }, "https://youtu.be/example")

    assert payload["title"] == "India vs Sri Lanka Test Watch Along"
    assert payload["description"].startswith("Full Hindi watch-along")
    assert payload["channel"] == "Cricket With Prajjwal"
    assert payload["source_tags"] == ["india vs sri lanka", "test cricket"]


def test_evidence_pack_fuses_all_sources_and_resolves_yuvi():
    from automation.seo.context_engine import build_cricket_evidence_pack

    pack = build_cricket_evidence_pack(
        video_title="India cricket coach discussion",
        video_description="Why Yuvi understands pressure better than most players.",
        clip_transcript="Yuvi ko India ka coach bana do.",
        ocr_entities={"on_screen_text": ["TEAM INDIA"]},
        research_context={
            "match_facts": [
                "Shubman Gill captains India in the Sri Lanka Test series."
            ],
            "search_queries": _approved_queries(),
            "sources": [{"kind": "official", "url": "https://www.bcci.tv/"}],
        },
    )

    assert pack["source_video"]["title"] == "India cricket coach discussion"
    assert "Yuvraj Singh understands pressure" in pack["source_video"]["description"]
    assert pack["clip_transcript"] == "Yuvraj Singh ko India ka coach bana do."
    assert "Yuvraj Singh" in pack["grounded_entities"]["players"]
    assert pack["match_facts"] == [
        "Shubman Gill captains India in the Sri Lanka Test series."
    ]
    assert "Shubman Gill" in pack["grounded_entities"]["players"]
    assert 8 <= len(pack["approved_search_queries"]) <= 15


def test_research_uses_full_source_context_for_match_lookup(monkeypatch):
    from automation.seo import trends

    captured = {}
    monkeypatch.setattr(trends, "fetch_google_trends_in", lambda: [])
    monkeypatch.setattr(trends, "fetch_own_live_stream_url", lambda: "")
    monkeypatch.setattr(trends, "fetch_youtube_suggestions", lambda query: _approved_queries())
    monkeypatch.setattr(
        trends,
        "fetch_youtube_search_signals",
        lambda query: ["Yuvraj Singh as India Coach? Fan Debate"],
    )

    def fake_match(query):
        captured["query"] = query
        return {
            "facts": ["India vs Sri Lanka, 1st Test, Galle"],
            "source_url": "https://example.test/match",
        }

    monkeypatch.setattr(trends, "fetch_verified_match_context", fake_match)

    result = trends.get_trending_context(
        domain="cricket",
        region="IN",
        video_title="India vs Sri Lanka Test watch along",
        video_description="Shubman Gill discusses the Galle Test.",
        transcript="Yuvraj Singh should coach India after this Test series.",
    )

    assert "India" in captured["query"]
    assert "Sri Lanka" in captured["query"]
    assert "Test" in captured["query"]
    assert result["match_facts"] == ["India vs Sri Lanka, 1st Test, Galle"]
    assert result["sources"][0]["url"] == "https://example.test/match"
    assert 8 <= len(result["search_queries"]) <= 15


def test_research_network_failure_returns_grounded_local_queries(monkeypatch):
    from automation.seo import trends

    monkeypatch.setattr(trends, "fetch_google_trends_in", lambda: [])
    monkeypatch.setattr(trends, "fetch_own_live_stream_url", lambda: "")
    monkeypatch.setattr(trends, "fetch_youtube_suggestions", Mock(side_effect=OSError("offline")))
    monkeypatch.setattr(trends, "fetch_youtube_search_signals", Mock(side_effect=OSError("offline")))
    monkeypatch.setattr(trends, "fetch_verified_match_context", Mock(side_effect=OSError("offline")))

    result = trends.get_trending_context(
        video_title="India cricket coach debate",
        video_description="Yuvi as India mentor",
        transcript="Yuvi ko coach bana do",
    )

    assert result["match_facts"] == []
    assert result["sources"] == []
    assert 8 <= len(result["search_queries"]) <= 15
    assert all("yuvraj" in query.lower() or "india" in query.lower() for query in result["search_queries"])


def test_youtube_search_signal_parser_extracts_real_result_titles():
    from automation.seo.trends import parse_youtube_search_titles

    html = r'''{"videoRenderer":{"videoId":"one","title":{"runs":[{"text":"Yuvraj Singh Coach Debate"}]}}}
    {"videoRenderer":{"videoId":"two","title":{"runs":[{"text":"India vs Sri Lanka Test Analysis"}]}}}'''

    assert parse_youtube_search_titles(html) == [
        "Yuvraj Singh Coach Debate",
        "India vs Sri Lanka Test Analysis",
    ]


def test_offline_query_fallback_does_not_invent_team_india():
    from automation.seo.context_engine import build_grounded_search_queries

    queries = build_grounded_search_queries(
        "Cricket bowling technique",
        "A detailed fast-bowling discussion.",
        "What a yorker from the bowler!",
        [],
    )

    assert len(queries) >= 8
    assert all("india" not in query.casefold() for query in queries)


def test_generated_queries_must_be_from_approved_evidence(monkeypatch):
    import automation.seo.seo as seo

    monkeypatch.setattr(seo, "extract_ocr_entities", lambda *_: {})
    monkeypatch.setattr(
        seo,
        "_attempt_seo_generation",
        lambda *args, **kwargs: {
            "title": "Yuvraj Singh ko Coach Banao?",
            "description": _long_description(),
            "hashtags": ["#Shorts", "#YuvrajSingh"],
            "search_terms": _approved_queries()[:-1] + ["india cricket latest"],
        },
    )

    try:
        seo.generate_clip_seo(
            "clip1",
            "Yuvi ko India ka coach bana do",
            video_title="India cricket discussion",
            video_description="Yuvi coaching debate",
            approved_search_queries=_approved_queries(),
        )
    except seo.SEOGenerationError as exc:
        assert "unapproved search queries" in str(exc)
    else:
        raise AssertionError("unapproved AI query must be rejected")


def test_research_queries_can_stay_in_metadata_when_primary_queries_are_embedded(monkeypatch):
    import automation.seo.seo as seo

    monkeypatch.setattr(seo, "extract_ocr_entities", lambda *_: {})
    monkeypatch.setattr(
        seo,
        "_attempt_seo_generation",
        lambda *args, **kwargs: {
            "title": "Yuvraj Singh ko Coach Banao?",
            "description": _long_description(_approved_queries()[:2]),
            "hashtags": ["#Shorts", "#YuvrajSingh"],
            "search_terms": _approved_queries(),
        },
    )

    result = seo.generate_clip_seo(
        "clip1",
        "Yuvi ko India ka coach bana do",
        video_title="India cricket discussion",
        video_description="Yuvi coaching debate",
        approved_search_queries=_approved_queries(),
    )

    assert result["search_terms"] == _approved_queries()
    assert result["primary_search_terms"] == _approved_queries()[:2]


def test_hallucinated_player_in_api_tags_is_rejected(monkeypatch):
    import automation.seo.seo as seo

    monkeypatch.setattr(seo, "extract_ocr_entities", lambda *_: {})
    monkeypatch.setattr(
        seo,
        "_attempt_seo_generation",
        lambda *args, **kwargs: {
            "title": "Yuvraj Singh ko Coach Banao?",
            "description": _long_description(),
            "hashtags": ["#Shorts", "#YuvrajSingh"],
            "search_terms": _approved_queries(),
            "tags": ["Yuvraj Singh", "Virat Kohli"],
        },
    )

    try:
        seo.generate_clip_seo(
            "clip1",
            "Yuvi ko India ka coach bana do",
            video_title="India cricket discussion",
            video_description="Yuvi coaching debate",
            approved_search_queries=_approved_queries(),
        )
    except seo.SEOGenerationError as exc:
        assert "Virat Kohli" in str(exc)
    else:
        raise AssertionError("hallucinated API tag must be rejected")


def test_api_tags_respect_configured_character_budget():
    from automation.seo.seo import _enforce_limits
    from utils.config import load_config

    result = _enforce_limits({
        "title": "Yuvraj Singh Coach Debate",
        "description": "x" * 1400,
        "hashtags": ["#Shorts", "#YuvrajSingh"],
        "search_terms": _approved_queries(),
        "tags": [f"yuvraj-singh-grounded-tag-{index:02d}" for index in range(40)],
    })
    budget = load_config()["seo"]["max_tag_chars"]
    assert len(",".join(result["tags"])) <= budget


def test_active_system_prompt_has_no_fake_algorithm_date_or_compact_word_target():
    from automation.seo import seo

    assert "June 2026" not in seo._SYSTEM
    assert "30 words" not in seo._SYSTEM
    assert "source video title" in seo._SYSTEM.lower()
    assert "verified" in seo._SYSTEM.lower()


def test_escalation_keeps_full_evidence_and_approved_queries(monkeypatch):
    from automation.seo import seo

    ai = Mock()
    ai.generate_seo_text.return_value = ""
    monkeypatch.setattr(seo, "_get_ai", lambda: ai)

    seo._escalation_seo(
        "clip1",
        "SOURCE DESCRIPTION\nAPPROVED GROUNDED SEARCH QUERIES\n- yuvraj singh coach",
        "Yuvraj Singh should coach India",
        "India cricket discussion",
        True,
        salvage_tmpl=seo._CRICKET_ONLY_SALVAGE_TMPL
        .replace("{description_target_chars}", "3200")
        .replace("{description_max_chars}", "4000"),
    )

    sent_prompt = ai.generate_seo_text.call_args.kwargs["prompt"]
    assert "SOURCE DESCRIPTION" in sent_prompt
    assert "APPROVED GROUNDED SEARCH QUERIES" in sent_prompt
