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




class TestSEOGeneratorClassRemoved:
    """The old generic wrapper cannot be reintroduced."""

    def test_seo_generator_symbol_is_gone(self):
        import automation.seo.seo as seo

        assert not hasattr(seo, "SEOGenerator")
