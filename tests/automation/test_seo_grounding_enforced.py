import pytest


class _NoVidiq:
    audit = {"enabled": False, "used": False, "calls": 0, "key_slot": None}

    def call_tool(self, name, arguments):
        return None


def _install_grounded_writer(monkeypatch, seo, *, title, tags=None):
    queries = [f"yuvraj singh india coach debate {index}" for index in range(24)]
    monkeypatch.setattr(seo, "VidiqClient", lambda **kwargs: _NoVidiq())
    monkeypatch.setattr(seo, "extract_ocr_entities", lambda *_: {})
    monkeypatch.setattr(seo, "extract_grounded_entities_llm", lambda *a, **k: {
        "players": ["Yuvraj Singh"], "teams": ["India"], "topic_phrases": ["coach debate"],
    })
    monkeypatch.setattr(seo, "audit_written_copy_llm", lambda *a, **k: {
        "unsupported_entities": [], "supported_topics": [],
    })
    monkeypatch.setattr(seo, "_attempt_seo_generation", lambda *a, **k: {
        "title": title,
        "description": "Yuvraj Singh India coach debate explained with cricket context. " * 60,
        "hashtags": ["#Shorts", "#Cricket", "#YuvrajSingh"],
        "search_terms": queries,
        "primary_search_terms": queries[:4],
        "tags": tags or ["Yuvraj Singh", "India cricket"],
    })
    return queries


def test_unsupported_player_in_title_fails_closed(monkeypatch):
    import automation.seo.seo as seo

    queries = _install_grounded_writer(
        monkeypatch, seo, title="Virat Kohli Coach Debate",
    )
    monkeypatch.setattr(seo, "_llm_repair_seo", lambda *a, **k: None)

    with pytest.raises(seo.SEOGenerationError, match="ungrounded title name"):
        seo.generate_clip_seo(
            "title-grounding",
            "Yuvraj Singh ko India ka cricket coach bana do",
            video_title="India cricket coach discussion",
            approved_search_queries=queries,
        )


def test_unsupported_player_in_api_tags_fails_closed(monkeypatch):
    import automation.seo.seo as seo

    queries = _install_grounded_writer(
        monkeypatch,
        seo,
        title="Yuvraj Singh India Coach Debate",
        tags=["Yuvraj Singh", "Virat Kohli"],
    )

    with pytest.raises(seo.SEOGenerationError, match="Virat Kohli"):
        seo.generate_clip_seo(
            "tag-grounding",
            "Yuvraj Singh ko India ka cricket coach bana do",
            video_title="India cricket coach discussion",
            approved_search_queries=queries,
        )
