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


class TestEnforceLimitsHashtagCap:

    def test_enforce_limits_caps_hashtags_for_shorts(self):
        from utils.config import load_config
        from automation.seo.seo import _enforce_limits
        cap = int(load_config().get("seo", {}).get("max_hashtags", 3))
        item = {
            "title": "Kohli ka record-breaking SIX! 🏏",
            "description": "Kohli breaks fastest century record with a massive six over long-on, crowd erupts in Wankhede.",
            "hashtags": ["#Shorts", "#Kohli", "#RCB", "#IPL2026", "#ViratKohli", "#Cricket"],
            "search_terms": ["kohli six", "ipl 2026"],
        }
        result = _enforce_limits(item, is_shorts=True)
        assert len(result["hashtags"]) <= cap

    def test_enforce_limits_keeps_shorts_first(self):
        from automation.seo.seo import _enforce_limits
        item = {
            "title": "Bumrah ki deadly YORKER! 💥",
            "description": "Bumrah knocks over the stumps with an unplayable yorker in the death overs.",
            "hashtags": ["#Cricket", "#Shorts", "#Bumrah"],
            "search_terms": ["bumrah yorker"],
        }
        result = _enforce_limits(item, is_shorts=True)
        assert result["hashtags"][0].lstrip("#").lower() == "shorts"

    def test_enforce_limits_longform_keeps_15(self):
        from automation.seo.seo import _enforce_limits
        item = {
            "title": "Full match analysis video",
            "description": "Long form match analysis covering both innings with detailed player stats and strategies.",
            "hashtags": [f"#Tag{i}" for i in range(15)],
            "search_terms": ["match analysis"],
        }
        result = _enforce_limits(item, is_shorts=False)
        assert len(result["hashtags"]) == 15


class TestDescriptionHashtagCap:

    def _count_hashtags(self, text):
        return re.findall(r"#[\w]+", text or "")

    def test_description_hashtags_capped_for_shorts(self):
        from automation.seo.seo import _enforce_limits
        item = {
            "title": "Kohli ka record-breaking SIX! 🏏",
            "description": (
                "Kohli breaks the fastest century record! "
                "#️⃣ Hashtags: #Shorts #Kohli #RCB #IPL2026 #ViratKohli #Cricket"
            ),
            "hashtags": ["#Shorts", "#Kohli", "#RCB", "#IPL2026", "#ViratKohli", "#Cricket"],
            "search_terms": ["kohli six"],
        }
        result = _enforce_limits(item, is_shorts=True)
        assert len(self._count_hashtags(result["description"])) <= 3

    def test_description_keeps_shorts_first_when_capped(self):
        from automation.seo.seo import _enforce_limits
        item = {
            "title": "Bumrah ki deadly YORKER! 💥",
            "description": "Deadly yorker. #Shorts #Bumrah #MI #Cricket #IPL2026 #Bowling",
            "hashtags": ["#Shorts", "#Bumrah", "#MI", "#Cricket", "#IPL2026", "#Bowling"],
            "search_terms": ["bumrah yorker"],
        }
        result = _enforce_limits(item, is_shorts=True)
        tags = self._count_hashtags(result["description"])
        assert tags[0].lstrip("#").lower() == "shorts"

    def test_description_hashtags_not_capped_for_longform(self):
        from automation.seo.seo import _enforce_limits
        item = {
            "title": "Full match analysis",
            "description": "Analysis. #Shorts #Cricket #IPL #Highlights #Match #Wicket",
            "hashtags": ["#Shorts", "#Cricket", "#IPL", "#Highlights", "#Match", "#Wicket"],
            "search_terms": ["analysis"],
        }
        result = _enforce_limits(item, is_shorts=False)
        assert len(self._count_hashtags(result["description"])) == 6


class TestMaxHashtagsConfigRobustness:

    def test_null_max_hashtags_falls_back(self, monkeypatch):
        from automation.seo import seo
        from automation.seo.seo import _enforce_limits
        monkeypatch.setitem(seo.cfg.get("seo", {}), "max_hashtags", None)
        item = {
            "title": "Test clip",
            "description": "Test description.",
            "hashtags": ["#Shorts", "#Kohli", "#RCB", "#IPL2026", "#ViratKohli"],
            "search_terms": ["kohli"],
        }
        result = _enforce_limits(item, is_shorts=True)
        assert 1 <= len(result["hashtags"]) <= 15

    def test_non_int_max_hashtags_falls_back(self, monkeypatch):
        from automation.seo import seo
        from automation.seo.seo import _enforce_limits
        monkeypatch.setitem(seo.cfg.get("seo", {}), "max_hashtags", "abc")
        item = {
            "title": "Test clip",
            "description": "Test description.",
            "hashtags": ["#Shorts", "#Kohli", "#RCB", "#IPL2026"],
            "search_terms": ["kohli"],
        }
        result = _enforce_limits(item, is_shorts=True)
        assert 1 <= len(result["hashtags"]) <= 15

    def test_zero_max_hashtags_clamped(self, monkeypatch):
        from automation.seo import seo
        from automation.seo.seo import _enforce_limits
        monkeypatch.setitem(seo.cfg.get("seo", {}), "max_hashtags", 0)
        item = {
            "title": "Test clip",
            "description": "Test description.",
            "hashtags": ["#Shorts", "#Kohli", "#RCB"],
            "search_terms": ["kohli"],
        }
        result = _enforce_limits(item, is_shorts=True)
        assert len(result["hashtags"]) >= 1


class TestSalvageTemplatesAligned:

    def test_salvage_templates_no_10_15_hashtags(self):
        from automation.seo import seo
        for tmpl in (seo._SALVAGE_TMPL, seo._SALVAGE_TMPL_FOOTBALL, seo._SALVAGE_TMPL_GENERAL):
            assert "10-15" not in tmpl, "salvage template still asks for 10-15 hashtags"

    def test_salvage_templates_instruct_against_live_title(self):
        from automation.seo import seo
        for tmpl in (seo._SALVAGE_TMPL, seo._SALVAGE_TMPL_FOOTBALL, seo._SALVAGE_TMPL_GENERAL):
            assert "NEVER use" in tmpl and "Live Score" in tmpl, \
                "salvage template must instruct against LIVE title framing"


class TestPromptsNoLiveScoreTitle:

    def test_cricket_prompt_no_live_score_example(self):
        from automation.seo import seo
        assert "Live Score #Shorts" not in seo._PROMPT_TMPL

    def test_football_prompt_no_live_score_example(self):
        from automation.seo import seo
        assert "Live Score #Shorts" not in seo._PROMPT_TMPL_FOOTBALL

    def test_prompts_instruct_against_live_title(self):
        from automation.seo import seo
        for tmpl in (seo._PROMPT_TMPL, seo._PROMPT_TMPL_FOOTBALL, seo._PROMPT_TMPL_GENERAL):
            assert 'NEVER use "Live Score", "LIVE"' in tmpl, \
                "prompt should explicitly warn against LIVE framing in title"

    def test_prompts_instruct_against_shorts_in_title(self):
        from automation.seo import seo
        for tmpl in (seo._PROMPT_TMPL, seo._PROMPT_TMPL_FOOTBALL, seo._PROMPT_TMPL_GENERAL):
            assert 'or "#Shorts" in the title' in tmpl, \
                "prompt should warn against #Shorts in the title text"
