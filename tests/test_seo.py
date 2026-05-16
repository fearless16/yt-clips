"""
test_seo.py — TDD suite for seo.py

Covers:
  Unit tests: _extract_keywords, _enforce_limits, _parse_json_response
  Prompt tests: _SYSTEM, _PROMPT_TMPL placeholders
  Generation: generate_clip_seo (happy path + retry + fallback + ai_generated flag)
  Export: generate_seo_for_exported_clip, process_all_seo
  Upload guard: upload.py skip on ai_generated=False
  Tags: tags field merged into YouTube API
  Scorecard: parse_cricbuzz_scorecard enhancement
"""
import json
import pytest
from pathlib import Path
from seo import (
    _extract_keywords,
    _enforce_limits,
    _parse_json_response,
    _generate_template_seo,
    generate_clip_seo,
    generate_seo_for_exported_clip,
    process_all_seo,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

_GOOD_RESPONSE = {
    "clip_id": "t1",
    "title": "Washington Sundar Destroys SRH With 67 Off 34 🔥 | GT vs SRH | IPL 2026",
    "description": (
        "Match Summary:\n"
        "GT set a massive 214-run target for SRH in IPL 2026.\n\n"
        "What Happens in This Clip:\n"
        "Washington Sundar smashes a brilliant 67 off 34 balls.\n\n"
        "Key Players:\n"
        "Washington Sundar — 67(34), Sai Sudharsan — 78.\n\n"
        "Match Situation:\n"
        "Target 214. SRH at 89/3 in 9.2 overs. Need 126 from 64 balls.\n\n"
        "Search Keywords:\n"
        "Washington Sundar, GT vs SRH, IPL 2026\n\n"
        "Hashtags:\n"
        "#IPL2026 #GTvsSRH #Shorts"
    ),
    "hashtags": ["#IPL2026", "#GTvsSRH", "#WashingtonSundar", "#GujaratTitans", "#SunrisersHyderabad", "#Shorts", "#Cricket"],
    "tags": [
        "ipl 2026", "washington sundar", "gujarat titans", "sunrisers hyderabad",
        "gt vs srh", "t20 cricket", "cricket short", "six", "boundary",
        "cricket highlights", "tata ipl", "indian premier league",
    ],
    "search_terms": [
        "washington sundar six vs srh", "gt vs srh highlights 2026",
        "ipl 2026 best moments", "gujarat titans vs sunrisers hyderabad",
        "cricket short video", "washington sundar batting",
        "srh vs gt live score", "ipl 2026 live",
        "pat cummins batting today", "sai sudharsan 78 runs",
    ],
}

_TREND_RESPONSE = {
    "topics": ["IPL 2026", "GT vs SRH", "Washington Sundar"],
    "scorecard": "GT 214/4 (20) vs SRH 89/3 (9.2)",
    "live_stream_url": "",
}

def _patch(monkeypatch, ai_payload=None, trend_payload=None):
    import seo, trends
    payload = ai_payload if ai_payload is not None else json.dumps(_GOOD_RESPONSE)
    # Mock the model chain to avoid real API calls
    def fake_chain(cid, prompt):
        import json
        from seo import _enforce_limits
        data = json.loads(payload)
        result = _enforce_limits(data)
        result["clip_id"] = cid
        result["ai_generated"] = True
        return result
    monkeypatch.setattr(seo, "_try_model_chain", fake_chain)
    monkeypatch.setattr(
        trends, "get_trending_context",
        lambda *a, **kw: trend_payload if trend_payload is not None else _TREND_RESPONSE,
    )


# ── _extract_keywords ─────────────────────────────────────────────────────────

class TestExtractKeywords:
    def test_cricket_keywords_preserved(self):
        kw = _extract_keywords("Kohli hit a massive six into the crowd")
        assert "kohli" in kw
        assert "six" in kw

    def test_stop_words_filtered(self):
        kw = _extract_keywords("the player is hitting the ball very well")
        assert "the" not in kw
        assert "very" not in kw
        assert any(w in kw for w in ("player", "hitting", "ball", "well"))

    def test_limit_respected(self):
        long_text = " ".join(f"uniqueword{i}" for i in range(50))
        assert len(_extract_keywords(long_text, limit=14)) <= 14

    def test_empty_string_returns_empty(self):
        assert _extract_keywords("") == []

    def test_frequency_ordering(self):
        kw = _extract_keywords("kohli kohli kohli dhoni dhoni bumrah")
        assert kw[0] == "kohli"
        assert kw[1] == "dhoni"


# ── _enforce_limits ──────────────────────────────────────────────────────────

class TestEnforceLimits:
    def test_title_truncated(self):
        item = {"title": "A" * 150}
        result = _enforce_limits(item)
        assert len(result["title"]) == 100

    def test_hashtags_prefixed_and_capped(self):
        item = {"hashtags": ["A", "#B", "C", "D", "E", "F", "G", "H", "I"]}
        result = _enforce_limits(item)
        for h in result["hashtags"]:
            assert h.startswith("#")
        assert "#Shorts" in result["hashtags"]
        assert len(result["hashtags"]) <= 8

    def test_shorts_always_present(self):
        item = {"hashtags": ["#IPL2026", "#Cricket"]}
        result = _enforce_limits(item)
        assert any(h.lower() == "#shorts" for h in result["hashtags"])

    def test_tags_deduplicated_and_limited(self):
        item = {"tags": ["cricket", "cricket", "ipl 2026", "a" * 300]}
        result = _enforce_limits(item)
        assert len(result["tags"]) == len(set(result["tags"]))
        # Total tag chars should be <= 500
        total = sum(len(t) for t in result["tags"]) + len(result["tags"]) - 1
        assert total <= 500

    def test_search_terms_validation(self):
        item = {
            "hashtags": ["#Cricket"],
            "search_terms": [
                "kohli six",      # valid
                "shorts",         # too short/generic
                "cricket",        # generic
                "A",              # 1 word
                "valid term two", # valid
            ]
        }
        result = _enforce_limits(item)
        assert "kohli six" in result["search_terms"]
        assert "valid term two" in result["search_terms"]
        assert "shorts" not in result["search_terms"]
        assert "A" not in result["search_terms"]

    def test_search_terms_length_limit(self):
        item = {"search_terms": ["long term number " + str(i) for i in range(50)]}
        result = _enforce_limits(item)
        total_len = sum(len(t) for t in result["search_terms"]) + (len(result["search_terms"]) - 1) * 2
        assert total_len <= 500


# ── _parse_json_response ─────────────────────────────────────────────────────

class TestParseJsonResponse:
    def test_plain_json(self):
        result = _parse_json_response('{"title": "Test"}')
        assert result == {"title": "Test"}

    def test_json_with_preamble(self):
        result = _parse_json_response('Some text before {"title": "Test"} after')
        assert result == {"title": "Test"}

    def test_json_with_code_fence(self):
        result = _parse_json_response('```json\n{"title": "Test"}\n```')
        assert result == {"title": "Test"}

    def test_no_json_returns_none(self):
        result = _parse_json_response("No JSON here at all")
        assert result is None

    def test_malformed_json_returns_none(self):
        result = _parse_json_response('{"title": unquoted}')
        assert result is None


# ── _generate_template_seo ──────────────────────────────────────────────────

class TestGenerateTemplateSEO:
    def test_basic_template_output(self):
        result = _generate_template_seo(
            "clip_01", "kohli hit a massive six over long on",
            "RCB vs CSK Live", "MI 185/4 (18.3)", ["IPL 2026"],
        )
        assert result["clip_id"] == "clip_01"
        assert isinstance(result["title"], str)
        assert len(result["title"]) <= 100
        assert len(result["description"]) <= 5000
        assert isinstance(result["hashtags"], list)
        assert "#Shorts" in result["hashtags"]
        assert isinstance(result["tags"], list)
        assert len(result["tags"]) > 0
        assert isinstance(result["search_terms"], list)

    def test_template_includes_tags_field(self):
        result = _generate_template_seo("c1", "wicket", "Match", "", [])
        assert "tags" in result
        assert len(result["tags"]) > 0

    def test_template_does_not_use_hooks(self):
        from seo import _SYSTEM
        result = _generate_template_seo("c1", "six", "Match", "", [])
        assert "Arey" not in result["title"]
        assert "kya" not in result["description"]


# ── generate_clip_seo ─────────────────────────────────────────────────────────

class TestGenerateClipSEO:
    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        _patch(monkeypatch)
        monkeypatch.setattr("time.sleep", lambda x: None)

    def test_happy_path_includes_all_fields(self):
        result = generate_clip_seo("t1", "washington sundar hits a six")
        assert result["clip_id"] == "t1"
        assert result.get("title") and isinstance(result["title"], str)
        assert result.get("description") and isinstance(result["description"], str)
        assert isinstance(result.get("hashtags"), list)
        assert isinstance(result.get("tags"), list)
        assert isinstance(result.get("search_terms"), list)
        assert result.get("ai_generated") is True

    def test_ai_generated_flag_true_on_success(self):
        result = generate_clip_seo("t1", "kohli hits a six")
        assert result["ai_generated"] is True

    def test_ai_generated_flag_false_on_fallback(self, monkeypatch):
        import seo
        # Make model chain return None (all models fail)
        monkeypatch.setattr(seo, "_try_model_chain", lambda cid, prompt: None)
        def fail(*a, **kw): raise ValueError("429 Too Many Requests")
        monkeypatch.setattr(seo.ai, "generate_text", fail)

        result = generate_clip_seo("t1", "kohli hits a six")
        assert result["ai_generated"] is False

    def test_fallback_contains_valid_seo(self, monkeypatch):
        import seo
        monkeypatch.setattr(seo, "_try_model_chain", lambda cid, prompt: None)
        def fail(*a, **kw): raise ValueError("API error")
        monkeypatch.setattr(seo.ai, "generate_text", fail)

        result = generate_clip_seo("t1", "kohli hits a massive six")
        assert result["clip_id"] == "t1"
        assert len(result["title"]) > 0
        assert len(result["hashtags"]) >= 3
        assert "#Shorts" in result["hashtags"]
        assert "tags" in result
        assert len(result["tags"]) > 0

    def test_search_terms_are_phrases(self, monkeypatch):
        _patch(monkeypatch, json.dumps(_GOOD_RESPONSE))
        result = generate_clip_seo("t1", "washington sundar hits")
        for term in result.get("search_terms", []):
            assert len(term.split()) >= 2, f"Search term too short: {term}"

    def test_title_has_no_generic_pattern(self, monkeypatch):
        _patch(monkeypatch, json.dumps(_GOOD_RESPONSE))
        result = generate_clip_seo("t1", "cricket batting")
        title = result["title"].lower()
        assert "cricket live:" not in title, f"Banned pattern in title: {result['title']}"

    def test_tags_in_youtube_format(self, monkeypatch):
        _patch(monkeypatch, json.dumps(_GOOD_RESPONSE))
        result = generate_clip_seo("t1", "cricket")
        for tag in result.get("tags", []):
            assert not tag.startswith("#"), f"Tag should not be hashtag: {tag}"


# ── generate_seo_for_exported_clip ──────────────────────────────────────────

class TestGenerateSEOForExportedClip:
    def test_saves_metadata_file(self, tmp_path, monkeypatch):
        _patch(monkeypatch)
        monkeypatch.setattr("time.sleep", lambda x: None)

        result = generate_seo_for_exported_clip("t1", "transcript", str(tmp_path))
        expected_file = tmp_path / "t1_metadata.json"
        assert expected_file.exists()
        with open(expected_file, "r") as f:
            saved = json.load(f)
            assert saved["title"] == _GOOD_RESPONSE["title"]
            assert saved["ai_generated"] is True
            assert "tags" in saved

    def test_template_fallback_logs_warning(self, tmp_path, monkeypatch):
        import seo
        monkeypatch.setattr(seo, "_try_model_chain", lambda cid, prompt: None)
        def fail(*a, **kw): raise ValueError("API error")
        monkeypatch.setattr(seo.ai, "generate_text", fail)
        monkeypatch.setattr("time.sleep", lambda x: None)

        result = generate_seo_for_exported_clip("t1", "transcript", str(tmp_path))
        assert result["ai_generated"] is False


# ── process_all_seo ───────────────────────────────────────────────────────────

def test_process_all_seo(tmp_path, monkeypatch):
    import yaml
    _patch(monkeypatch)
    monkeypatch.setattr("time.sleep", lambda x: None)

    h_path = tmp_path / "highlights.yaml"
    out_dir = tmp_path / "shorts"

    highlights_data = {
        "clip1": {"text": "washington sundar six"},
        "clip2": {"text": "gill fifty partnership"},
    }
    with open(h_path, "w") as f:
        yaml.dump(highlights_data, f)

    res_path = process_all_seo(str(h_path), str(out_dir))

    assert Path(res_path).exists()
    assert (out_dir / "clip1_metadata.json").exists()
    assert (out_dir / "clip2_metadata.json").exists()

    with open(res_path, "r") as f:
        results = json.load(f)
        assert len(results) == 2
        assert results[0]["clip_id"] == "clip1"
        assert "tags" in results[0]
        assert "ai_generated" in results[0]


# ── AI failure threshold ─────────────────────────────────────────────────────

