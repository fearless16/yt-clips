"""TDD tests for SEO quality enforcement — no generic fallbacks allowed.

Tests verify:
- Generic tags/titles/search_terms are rejected
- Groq provider works with TPM rate limiting
- SEO quality validation catches low-effort output
- SEOGenerator class never produces generic fallback
- _inject_viral_elements doesn't randomly override good titles
"""
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


class TestNoGenericFallback:
    """Generic SEO kills channel performance. Every path must be clip-specific."""

    def test_rank_tags_never_returns_generic_defaults(self):
        """Empty tags must return [] not generic defaults."""
        from automation.seo.seo import _rank_and_optimize_tags
        result = _rank_and_optimize_tags([], "kohli six")
        # Must NOT produce ["#Shorts", "#Cricket", "#IPL2026"] on empty input
        assert result == []

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


class TestViralElementsBias:
    """_inject_viral_elements must NEVER randomly override a good title."""

    def test_inject_never_replaces_good_title(self):
        from automation.seo.seo import _inject_viral_elements
        original_title = "Kohli ne maara CHHAKKA! 🔥"
        # Run 50 times — title must NEVER be randomly replaced
        for _ in range(50):
            result = _inject_viral_elements(
                original_title,
                "Kohli hits massive six",
                ["#Shorts", "#Kohli"],
            )
            assert result["title"] == original_title, \
                f"Title was randomly overridden to: {result['title']}"


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
            "description": "📝 Virat Kohli smashes a massive six over long-on! The crowd at Chinnaswamy goes absolutely wild as King Kohli deposits the bowler into the stands. Subscribe for more!",
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
    """The old SEOGenerator class produces generic garbage. It must not be used."""

    def test_seo_generator_not_used_in_pipeline(self):
        """No code outside tests should use SEOGenerator.generate()."""
        import ast
        from pathlib import Path
        repo = Path(__file__).resolve().parent.parent
        for py in sorted(repo.rglob("*.py")):
            if ".venv" in str(py) or "test_" in py.name or py.name == "seo.py":
                continue
            text = py.read_text(encoding="utf-8", errors="ignore")
            if "SEOGenerator" in text and "from automation.seo" in text:
                # Only imports for backward compat should exist
                assert "SEOGenerator()" not in text or "generate(" not in text, \
                    f"SEOGenerator used in production code: {py}"
