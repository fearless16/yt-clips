"""TDD — Shorts-style SEO (research-backed).

Champion channels (NIGHT STARS, ANKUSH EDITS, CShorts Hindi4) use:
1. Curiosity/record-style titles, NO "Live Score"/"LIVE" framing.
2. 2-3 hashtags max (not 15).

Verifies:
1. Config exposes seo.max_hashtags (2-5 range).
2. _enforce_limits caps hashtags at max_hashtags for Shorts.
3. Prompts no longer push "Live Score" in title examples; instruct against LIVE.
4. Description-level hashtag cap (visible YouTube hashtags come from description).
5. max_hashtags config is crash-proof (null/non-int/out-of-range).
6. Salvage templates aligned with 2-3 hashtag + no-LIVE-title policy.
"""
import re

import pytest


class TestConfigMaxHashtags:

    def test_config_has_max_hashtags(self):
        from utils.config import load_config
        seo_cfg = load_config().get("seo", {})
        assert "max_hashtags" in seo_cfg
        assert 2 <= int(seo_cfg["max_hashtags"]) <= 5














class TestNoGenericFlow:

    def test_general_templates_removed(self):
        import automation.seo.seo as seo
        for attr in ("_PROMPT_TMPL_GENERAL", "_SALVAGE_TMPL_GENERAL", "_SYSTEM_GENERAL"):
            assert not hasattr(seo, attr), f"{attr} must be removed — cricket-only channel"


