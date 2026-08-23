"""TDD tests for SEO quality enforcement — no generic fallbacks allowed.

Tests verify:
- Generic tags/titles/search_terms are rejected
- Groq provider works with TPM rate limiting
- SEO quality validation catches low-effort output
- Removed generic SEOGenerator cannot return
"""
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


class TestNoGenericFallback:
    """Generic SEO kills channel performance. Every path must be clip-specific."""

    def test_enforce_limits_rejects_generic_search_terms(self):
        """Search terms like 'cricket video' or 'sports highlights' are poison."""
        from automation.seo.seo import _enforce_limits
        GENERIC_POISON = {
            "cricket highlights", "cricket live match", "ipl match video",
            "t20 cricket live", "best cricket moments", "cricket video",
            "sports video", "sports highlights",
        }
        item = {
            "title": "Kohli ka CHHAKKA! 🔥",
            "description": "Kohli hits massive six over long-on",
            "hashtags": ["#Shorts", "#Kohli"],
            "search_terms": ["kohli six RCB", "cricket highlights", "sports video"],
        }
        result = _enforce_limits(item)
        for term in result["search_terms"]:
            assert term.lower() not in GENERIC_POISON, \
                f"Generic poison term '{term}' leaked through!"

    def test_enforce_limits_rejects_live_framing_search_terms(self):
        """On-demand Shorts must not carry live-stream framing in tags.

        The channel uploads clips, not live streams. Search terms that frame
        the clip as live ('ipl 2026 live', 'live cricket score') leak the
        wrong intent into YouTube tags and confuse the algorithm/CTR.
        """
        from automation.seo.seo import _enforce_limits
        item = {
            "title": "Kohli ka CHHAKKA! 🔥",
            "description": "Kohli hits massive six over long-on",
            "hashtags": ["#Shorts", "#Kohli"],
            "search_terms": [
                "kohli six RCB",
                "ipl 2026 live",
                "world cup live",
                "live cricket score",
                "live match today",
                "how to watch ipl live",
                "rcb vs mi live score",
            ],
        }
        result = _enforce_limits(item)
        for term in result["search_terms"]:
            assert "live" not in term.lower().split(), \
                f"Live-framing search term '{term}' leaked through!"
        assert "kohli six RCB" in result["search_terms"], \
            "On-demand term must survive the live-framing strip"

    def test_enforce_limits_rejects_devanagari_live_terms(self):
        """'लाइव क्रिकेट स्कोर' (live cricket score in Hindi) is live framing too."""
        from automation.seo.seo import _enforce_limits
        item = {
            "title": "Kohli ka CHHAKKA! 🔥",
            "description": "Kohli hits massive six over long-on",
            "hashtags": ["#Shorts", "#Kohli"],
            "search_terms": ["लाइव क्रिकेट स्कोर", "kohli batting today"],
        }
        result = _enforce_limits(item)
        assert not any("लाइव" in t for t in result["search_terms"]), \
            "Devanagari live-framing term leaked through!"
        assert "kohli batting today" in result["search_terms"]

    def test_enforce_limits_rejects_live_framing_hashtags(self):
        """#LiveCricket / #CricketLive / #LiveScore are live-framing tags."""
        from automation.seo.seo import _enforce_limits
        item = {
            "title": "Kohli ka CHHAKKA! 🔥",
            "description": "Kohli hits massive six over long-on",
            "hashtags": ["#Shorts", "#LiveCricket", "#CricketLive", "#LiveScore",
                         "#LiveStream", "#Kohli", "#Live"],
            "search_terms": ["kohli six"],
        }
        result = _enforce_limits(item)
        tags = {t.lstrip("#").lower() for t in result["hashtags"]}
        assert "livecricket" not in tags
        assert "cricketlive" not in tags
        assert "livescore" not in tags
        assert "livestream" not in tags
        assert "live" not in tags
        assert "shorts" in tags and "kohli" in tags

    def test_enforce_limits_keeps_non_live_words_with_live_substring(self):
        """Words containing 'live' as a substring ('alive') must NOT be stripped."""
        from automation.seo.seo import _enforce_limits
        item = {
            "title": "Kohli ka CHHAKKA! 🔥",
            "description": "Kohli hits massive six over long-on",
            "hashtags": ["#Shorts", "#Kohli"],
            "search_terms": ["kohli six", "alive and kicking moment"],
        }
        result = _enforce_limits(item)
        assert any("alive" in t for t in result["search_terms"]), \
            "Non-live-framing term containing 'alive' must survive"

    def test_enforce_limits_keeps_on_demand_terms(self):
        """Highlights/analysis terms are the correct frame for on-demand Shorts."""
        from automation.seo.seo import _enforce_limits
        item = {
            "title": "Kohli ka CHHAKKA! 🔥",
            "description": "Kohli hits massive six over long-on",
            "hashtags": ["#Shorts", "#Kohli"],
            "search_terms": [
                "kohli six wankhede", "RCB vs CSK highlights",
                "aaj ka match", "ipl 2026 match 54", "virat kohli innings",
            ],
        }
        result = _enforce_limits(item)
        assert len(result["search_terms"]) == 5, \
            "On-demand search terms must all survive"

    def test_enforce_limits_rejects_live_shorts_and_multi_token_compounds(self):
        """#LiveShorts and multi-token compounds (#LiveCricketMatch,
        #CricketLiveMatch) are live-framing tags and must be stripped."""
        from automation.seo.seo import _enforce_limits
        item = {
            "title": "Kohli ka CHHAKKA! 🔥",
            "description": "Kohli hits massive six over long-on",
            "hashtags": ["#Shorts", "#LiveShorts", "#LiveCricketMatch",
                         "#CricketLiveMatch", "#Kohli"],
            "search_terms": ["kohli six"],
        }
        result = _enforce_limits(item)
        tags = {t.lstrip("#").lower() for t in result["hashtags"]}
        assert "liveshorts" not in tags, "#LiveShorts leaked through"
        assert "livecricketmatch" not in tags, "#LiveCricketMatch leaked through"
        assert "cricketlivematch" not in tags, "#CricketLiveMatch leaked through"
        assert "shorts" in tags and "kohli" in tags

    def test_enforce_limits_strips_compound_live_from_title(self):
        """Title compounds like 'LIVESTREAM' / 'LiveScore' must be stripped,
        not just standalone 'live' words."""
        from automation.seo.seo import _enforce_limits
        item = {
            "title": "LIVESTREAM FULL MATCH NOW!",
            "description": "Kohli hits massive six over long-on",
            "hashtags": ["#Shorts", "#Kohli"],
            "search_terms": ["kohli six"],
        }
        result = _enforce_limits(item)
        assert "livestream" not in result["title"].lower(), \
            f"Compound live-framing survived in title: {result['title']!r}"

    def test_enforce_limits_tags_non_list_guarded(self):
        """A malformed 'tags' key (dict/int) must not crash and must not leak
        its keys as tags. Only list (or single string) is a valid tag block."""
        from automation.seo.seo import _enforce_limits
        item = {
            "title": "Kohli ka CHHAKKA! 🔥",
            "description": "Kohli hits massive six over long-on",
            "hashtags": ["#Shorts"],
            "search_terms": ["kohli six"],
            "tags": {"live_score": "cricket live match today", "kohli": "kohli"},
        }
        result = _enforce_limits(item)
        assert isinstance(result["tags"], list), "tags must be normalized to a list"
        assert result["tags"] == [], \
            f"dict tags must not leak keys as tag values: {result['tags']!r}"

    def test_enforce_limits_tags_int_does_not_crash(self):
        """An int 'tags' key must be treated as empty, not crash the pipeline."""
        from automation.seo.seo import _enforce_limits
        item = {
            "title": "Kohli ka CHHAKKA! 🔥",
            "description": "Kohli hits massive six over long-on",
            "hashtags": ["#Shorts"],
            "search_terms": ["kohli six"],
            "tags": 42,
        }
        result = _enforce_limits(item)
        assert result["tags"] == []

    def test_enforce_limits_skips_non_string_items_inside_tags(self):
        from automation.seo.seo import _enforce_limits
        result = _enforce_limits({
            "title": "Kohli ka CHHAKKA! 🔥",
            "description": "Kohli hits a massive six over long-on",
            "hashtags": ["#Shorts"],
            "search_terms": ["kohli six"],
            "tags": [None, 42, "Virat Kohli"],
        })
        assert result["tags"] == ["Virat Kohli"]

    def test_enforce_limits_strips_empty_title(self):
        """Empty title must not pass validation."""
        from automation.seo.seo import _enforce_limits, _validate_seo_quality
        item = {
            "title": "",
            "description": "Some description",
            "hashtags": ["#Shorts"],
            "search_terms": ["term"],
        }
        result = _enforce_limits(item)
        assert not _validate_seo_quality(result)

    def test_enforce_limits_malformed_llm_output_does_not_crash(self):
        """_enforce_limits exists to sanitize LLM output; a model emitting a
        non-string title/description or None/int inside hashtags/search_terms
        must NOT crash the whole SEO batch — it must be coerced/skipped."""
        from automation.seo.seo import _enforce_limits
        items = [
            {"title": 123, "description": "d", "hashtags": ["#Shorts"], "search_terms": ["x"]},
            {"title": "X", "description": 123, "hashtags": ["#Shorts"], "search_terms": ["x"]},
            {"title": "X", "description": "d", "hashtags": ["#Shorts", None, 5], "search_terms": ["x"]},
            {"title": "X", "description": "d", "hashtags": ["#Shorts"], "search_terms": ["x", None, 42]},
        ]
        for item in items:
            result = _enforce_limits(item)
            assert isinstance(result["title"], str)
            assert isinstance(result["description"], str)
            assert all(isinstance(h, str) for h in result["hashtags"]), \
                f"non-str hashtag leaked: {result['hashtags']!r}"
            assert all(isinstance(s, str) for s in result["search_terms"]), \
                f"non-str search term leaked: {result['search_terms']!r}"

    def test_enforce_limits_strips_generic_title(self):
        """Titles like 'Cricket Highlights' are generic garbage."""
        from automation.seo.seo import _validate_seo_quality
        item = {
            "title": "Cricket Highlights",
            "description": "Watch cricket",
            "hashtags": ["#Shorts"],
            "search_terms": ["cricket"],
        }
        assert not _validate_seo_quality(item)


class TestGroqProvider:
    """Groq should be available as a provider with TPM-aware rate limiting."""

    def test_groq_in_provider_models(self):
        from utils.ai_client import AIClient
        assert "groq" in AIClient.PROVIDER_MODELS

    def test_groq_in_failover_chain(self):
        from utils.ai_client import AIClient
        ai = AIClient()
        chain = ai._get_failover_chain("opencode")
        assert "groq" in chain

    def test_groq_has_tpm_rate_limit(self):
        """Groq has stricter TPM limits — bucket should have lower capacity."""
        from utils.ai_client import AIClient
        ai = AIClient()
        # Groq should have a lower token bucket capacity
        groq_capacity = ai.PROVIDER_RATE_LIMITS.get("groq", {}).get("capacity")
        assert groq_capacity is not None
        assert groq_capacity <= 15  # Much lower than default 30

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    def test_groq_appears_in_all_models(self):
        from utils.ai_client import AIClient
        ai = AIClient()
        ai.groq_api_key = "test-key"
        models = ai._all_models()
        groq_models = [m for p, m in models if p == "groq"]
        assert len(groq_models) > 0


class TestSEOQualityGate:
    """No SEO passes to upload without quality validation."""

    def test_quality_gate_rejects_devanagari_title(self):
        """Devanagari script in title kills Shorts discoverability."""
        from automation.seo.seo import _validate_seo_quality
        item = {
            "title": "कोहली ने मारा सिक्स! 🔥",
            "description": "Kohli hits six",
            "hashtags": ["#Shorts"],
            "search_terms": ["kohli six"],
        }
        assert not _validate_seo_quality(item)

    def test_quality_gate_accepts_good_hinglish(self):
        from automation.seo.seo import _validate_seo_quality
        item = {
            "title": "Kohli ne maara CHHAKKA! 🔥",
            "description": (
                "Virat Kohli batting analysis explains the complete Hinglish "
                "cricket discussion without inventing an opponent or score. " * 45
            ),
            "hashtags": ["#Shorts", "#Kohli", "#RCBvsCSK"],
            "search_terms": ["kohli six wankhede", "RCB vs CSK highlights"],
        }
        assert _validate_seo_quality(item)

    def test_quality_gate_rejects_too_short_description(self):
        from automation.seo.seo import _validate_seo_quality
        item = {
            "title": "Kohli ne maara CHHAKKA! 🔥",
            "description": "Watch",
            "hashtags": ["#Shorts"],
            "search_terms": ["kohli"],
        }
        assert not _validate_seo_quality(item)

    def test_generate_clip_seo_drops_low_quality_ai_response(self):
        """If AI returns generic garbage, the system must raise not accept."""
        from automation.seo.seo import generate_clip_seo, SEOGenerationError
        generic_response = json.dumps({
            "title": "Cricket Highlights",
            "description": "Watch cricket highlights",
            "hashtags": ["#Shorts"],
            "search_terms": ["cricket video"],
        })
        with patch("utils.ai_client.AIClient.generate_fastest_first",
                   return_value=generic_response), \
             patch("utils.ai_client.AIClient.generate_text",
                   return_value=generic_response), \
             patch("utils.ai_client.AIClient.generate_seo_text",
                   return_value=generic_response):
            with pytest.raises(SEOGenerationError):
                generate_clip_seo("c1", "kohli hit six over long on", "RCB vs CSK")


class TestSEOGeneratorClassRemoved:
    """The old generic wrapper cannot be reintroduced."""

    def test_seo_generator_symbol_is_gone(self):
        import automation.seo.seo as seo

        assert not hasattr(seo, "SEOGenerator")
