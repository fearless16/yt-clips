import json


class _Response:
    def __init__(self, status_code, events=""):
        self.status_code = status_code
        self.text = events

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(response=self)


def _sse(payload):
    return f"event: message\ndata: {json.dumps(payload)}\n\n"


def test_vidiq_calls_tools_call_and_fails_over_to_backup(monkeypatch, caplog):
    from automation.seo.vidiq import VidiqClient

    monkeypatch.setenv("VIDIQ_API_KEY", "primary-secret")
    monkeypatch.setenv("BACKUP_VIDIQ_API_KEY", "backup-secret")
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        if len(calls) == 1:
            return _Response(429)
        return _Response(200, _sse({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": [{"type": "text", "text": '{"keywords":["india cricket"]}'}]},
        }).replace("\n", "\r\n"))

    client = VidiqClient(enabled=True, timeout_seconds=4, post=post)
    result = client.call_tool("vidiq_keyword_research", {"keyword": "india cricket", "country": "IN"})

    assert result == {"keywords": ["india cricket"]}
    assert len(calls) == 2
    assert calls[0][1]["json"]["method"] == "tools/call"
    assert calls[0][1]["json"]["params"]["name"] == "vidiq_keyword_research"
    assert calls[0][1]["timeout"] == 4
    assert calls[0][1]["headers"]["Authorization"] == "Bearer primary-secret"
    assert calls[1][1]["headers"]["Authorization"] == "Bearer backup-secret"
    assert "primary-secret" not in caplog.text
    assert "backup-secret" not in caplog.text


def test_vidiq_is_fail_soft_without_keys(monkeypatch):
    from automation.seo.vidiq import VidiqClient

    monkeypatch.delenv("VIDIQ_API_KEY", raising=False)
    monkeypatch.delenv("BACKUP_VIDIQ_API_KEY", raising=False)

    client = VidiqClient(enabled=True, post=lambda *a, **k: (_ for _ in ()).throw(AssertionError()))

    assert client.call_tool("vidiq_keyword_research", {"keyword": "cricket"}) is None
    assert client.audit == {"enabled": True, "used": False, "calls": 0, "key_slot": None}


def test_vidiq_context_uses_grounded_seed_and_only_grounded_keyword_suggestions(monkeypatch):
    import automation.seo.seo as seo

    calls = []

    class Client:
        audit = {"enabled": True, "used": True, "calls": 2, "key_slot": "primary"}

        def call_tool(self, name, arguments):
            calls.append((name, arguments))
            if name == "vidiq_keyword_research":
                return {"relatedKeywords": [
                    {"keyword": "Yuvraj Singh India coach", "volume": 72.5},
                    {"keyword": "Rohit Sharma retirement", "volume": 90.0},
                ]}
            return {"titles": [
                {"title": "Rohit retires today", "score": 99},
                {
                    "title": "The hidden reason Yuvraj Singh should coach India #cricket #shorts",
                    "score": 95,
                },
            ]}

    monkeypatch.setattr(seo, "VidiqClient", lambda **kwargs: Client())
    context, audit, recommendations = seo._get_vidiq_context(
        ["Yuvraj Singh India coach"],
        grounding_text="Yuvraj Singh should coach India because his cricket brain is sharp",
    )

    assert calls[0][1]["keyword"] == "Yuvraj Singh India coach"
    assert calls[0][1]["country"] == "IN"
    assert calls[0][0] == "vidiq_keyword_research"
    assert calls[0][1]["mode"] == "research"
    assert calls[1][0] == "vidiq_generate_titles"
    assert calls[1][1]["type"] == "short"
    assert calls[1][1]["regionCode"] == "IN"
    assert "analysisSummary" in calls[1][1]
    assert len(calls) <= 2
    assert "Yuvraj Singh India coach" in context
    assert "Rohit" not in context
    assert recommendations == {
        "keywords": ["Yuvraj Singh India coach"],
        "titles": ["The hidden reason Yuvraj Singh should coach India"],
    }
    assert audit["used"] is True
    assert set(audit) == {"enabled", "used", "calls", "key_slot"}


def test_vidiq_prefers_structured_or_widget_data_over_prose(monkeypatch):
    from automation.seo.vidiq import VidiqClient

    monkeypatch.setenv("VIDIQ_API_KEY", "secret")
    monkeypatch.delenv("BACKUP_VIDIQ_API_KEY", raising=False)
    response = _Response(200, _sse({
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "_meta": {"widgetData": {"relatedKeywords": [{"keyword": "DPL 2026"}]}},
            "content": [{"type": "text", "text": "Research summary, not JSON"}],
        },
    }))
    client = VidiqClient(enabled=True, post=lambda *args, **kwargs: response)

    assert client.call_tool("vidiq_keyword_research", {}) == {
        "relatedKeywords": [{"keyword": "DPL 2026"}],
    }


def test_vidiq_context_is_fail_soft_for_malformed_optional_collections(monkeypatch):
    import automation.seo.seo as seo

    class Client:
        audit = {"enabled": True, "used": True, "calls": 2, "key_slot": "primary"}

        def call_tool(self, name, arguments):
            if name == "vidiq_keyword_research":
                return {"relatedKeywords": 42}
            return {"titles": {"title": "not-a-list"}}

    monkeypatch.setattr(seo, "VidiqClient", lambda **kwargs: Client())

    context, audit, recommendations = seo._get_vidiq_context(
        ["DPL 2026"], "DPL 2026 cricket"
    )

    assert context == ""
    assert recommendations == {"keywords": [], "titles": []}
    assert audit["used"] is True


def test_vidiq_does_not_retry_non_retryable_tool_errors_with_backup(monkeypatch):
    from automation.seo.vidiq import VidiqClient

    monkeypatch.setenv("VIDIQ_API_KEY", "primary-secret")
    monkeypatch.setenv("BACKUP_VIDIQ_API_KEY", "backup-secret")
    calls = []

    def post(*args, **kwargs):
        calls.append(kwargs)
        return _Response(200, _sse({
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32602, "message": "Invalid arguments"},
        }))

    client = VidiqClient(enabled=True, post=post)

    assert client.call_tool("vidiq_keyword_research", {}) is None
    assert len(calls) == 1


def test_seo_fails_closed_when_required_vidiq_has_no_output(monkeypatch):
    import automation.seo.seo as seo

    monkeypatch.setitem(seo.cfg["seo"]["vidiq"], "enabled", True)
    monkeypatch.setitem(seo.cfg["seo"]["vidiq"], "required", True)
    monkeypatch.setattr(seo, "_get_vidiq_context", lambda *a, **k: (
        "",
        {"enabled": True, "used": False, "calls": 1, "key_slot": None},
        {"keywords": [], "titles": []},
    ))
    monkeypatch.setattr(seo, "extract_ocr_entities", lambda *_: {})
    monkeypatch.setattr(seo, "extract_grounded_entities_llm", lambda *a, **k: {
        "players": ["Yuvraj Singh"], "teams": ["India"], "topic_phrases": [],
    })
    monkeypatch.setattr(
        seo,
        "_attempt_seo_generation",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM must not run")),
    )

    with __import__("pytest").raises(seo.SEOGenerationError, match="vidIQ required"):
        seo.generate_clip_seo(
            "vidiq-required",
            "Yuvraj Singh ko India ka cricket coach bana do",
            video_title="India cricket coach discussion",
            approved_search_queries=["Yuvraj Singh India coach"],
        )


def test_vidiq_exclusively_controls_final_title_terms_and_tags(monkeypatch):
    import automation.seo.seo as seo

    vidiq_terms = [f"yuvraj coach {index}" for index in range(24)]
    monkeypatch.setitem(seo.cfg["seo"]["vidiq"], "enabled", True)
    monkeypatch.setitem(seo.cfg["seo"]["vidiq"], "required", True)
    monkeypatch.setattr(seo, "_get_vidiq_context", lambda *a, **k: (
        "vidIQ output",
        {"enabled": True, "used": True, "calls": 2, "key_slot": "primary"},
        {"keywords": vidiq_terms, "titles": ["Yuvraj Singh India Coach Debate"]},
    ))
    monkeypatch.setattr(seo, "extract_ocr_entities", lambda *_: {})
    monkeypatch.setattr(seo, "extract_grounded_entities_llm", lambda *a, **k: {
        "players": ["Yuvraj Singh"], "teams": ["India"], "topic_phrases": [],
    })
    monkeypatch.setattr(seo, "audit_written_copy_llm", lambda *a, **k: {
        "unsupported_entities": [], "supported_topics": [],
    })
    monkeypatch.setattr(seo, "_attempt_seo_generation", lambda *a, **k: {
        "title": "Generic LLM Backup Title",
        "description": (
            "Yuvraj Singh India coach debate explained with cricket context. " * 70
        ),
        "hashtags": ["#Shorts", "#GenericBackup"],
        "search_terms": ["generic llm fallback"],
        "primary_search_terms": ["generic llm fallback"],
        "tags": ["generic llm fallback"],
    })

    result = seo.generate_clip_seo(
        "vidiq-exclusive",
        "Yuvraj Singh India coach debate explained with cricket context",
        video_title="Yuvraj Singh India coach debate",
        approved_search_queries=vidiq_terms,
    )

    assert result["title"] == "Yuvraj Singh India Coach Debate 🏏"
    assert result["search_terms"] == vidiq_terms
    assert result["primary_search_terms"] == vidiq_terms[:4]
    assert result["tags"] == vidiq_terms
    assert "generic llm fallback" not in str(result)


def test_required_vidiq_title_is_never_replaced_by_llm_repair(monkeypatch):
    import automation.seo.seo as seo

    vidiq_terms = [f"yuvraj coach {index}" for index in range(24)]
    monkeypatch.setitem(seo.cfg["seo"]["vidiq"], "enabled", True)
    monkeypatch.setitem(seo.cfg["seo"]["vidiq"], "required", True)
    monkeypatch.setattr(seo, "_get_vidiq_context", lambda *a, **k: (
        "vidIQ output",
        {"enabled": True, "used": True, "calls": 2, "key_slot": "primary"},
        {"keywords": vidiq_terms, "titles": ["Unrelated Wicket Story"]},
    ))
    monkeypatch.setattr(seo, "extract_ocr_entities", lambda *_: {})
    monkeypatch.setattr(seo, "extract_grounded_entities_llm", lambda *a, **k: {
        "players": ["Yuvraj Singh"], "teams": ["India"], "topic_phrases": [],
    })
    monkeypatch.setattr(seo, "audit_written_copy_llm", lambda *a, **k: {
        "unsupported_entities": [], "supported_topics": [],
    })
    monkeypatch.setattr(seo, "_attempt_seo_generation", lambda *a, **k: {
        "title": "ignored",
        "description": "Yuvraj Singh India coach debate cricket context. " * 70,
    })
    monkeypatch.setattr(
        seo,
        "_llm_repair_seo",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no SEO fallback allowed")),
    )

    with __import__("pytest").raises(seo.SEOGenerationError):
        seo.generate_clip_seo(
            "vidiq-no-repair",
            "Yuvraj Singh India coach debate cricket context",
            video_title="Yuvraj Singh India coach debate",
            approved_search_queries=vidiq_terms,
        )
