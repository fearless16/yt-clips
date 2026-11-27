"""TDD tests for automation/seo module.

Covers:
- _parse_json_response  (all edge cases)
- process_all_seo failure-marker writes
- Inter-clip sleep is config-driven (not hardcoded)
- retry_failed_seo smoke test
"""
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# _parse_json_response
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# process_all_seo — failure marker writes
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Inter-clip sleep is configurable (P2 fix)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# retry_failed_seo
# ---------------------------------------------------------------------------

class TestRetryFailedSEO:


    def test_retry_returns_zero_when_no_markers(self, tmp_path):
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        from automation.seo.seo import retry_failed_seo
        result = retry_failed_seo(str(out_dir))
        assert result == {"recovered": 0, "total": 0}

    def test_retry_returns_zero_when_dir_not_found(self, tmp_path):
        from automation.seo.seo import retry_failed_seo
        result = retry_failed_seo(str(tmp_path / "nonexistent"))
        assert result == {"recovered": 0, "total": 0}


# ---------------------------------------------------------------------------
# _get_ai lazy init (P1 fix — no module-level AIClient)
# ---------------------------------------------------------------------------

class TestLazyAIClientInit:

    def test_seo_module_has_get_ai_function(self):
        """After P1 fix: seo.py must expose _get_ai() instead of module-level ai."""
        import automation.seo.seo as seo_mod
        assert callable(getattr(seo_mod, "_get_ai", None)), (
            "_get_ai() function missing — module-level ai = AIClient() not yet refactored"
        )

    def test_get_ai_returns_ai_client(self):
        import automation.seo.seo as seo_mod
        client = seo_mod._get_ai()
        assert client is not None
        # Same instance on second call
        assert seo_mod._get_ai() is client


# ---------------------------------------------------------------------------
# SEO Model Restrictions (P0 — block nvidia/groq from SEO)
# ---------------------------------------------------------------------------


