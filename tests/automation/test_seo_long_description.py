"""TDD — Long-form descriptions for Shorts SEO (research-backed).

Research across 12 cricket-only low-sub channels (SnapSnippets 925 subs/51K
views, Mr X Cricket 12.2K/25K, Sportify 250/1.2K) shows high-view Shorts use
18-40 words in their description while low-view Shorts average ~9 words.
The old quality gate only demanded >= 100 chars (~12 words), which let thin
descriptions through. This enforces a config-driven minimum word count and
makes both AI prompts demand explicit word counts.

Verifies:
1. seo.min_description_words exists and is an int in 20-40.
2. _validate_seo_quality rejects descriptions below the word floor.
3. _validate_seo_quality accepts rich (>= floor) descriptions.
4. _PROMPT_TMPL carries explicit word-count guidance.
5. _SALVAGE_TMPL carries the SAME word-count guidance.
6. Existing good descriptions still pass the quality gate.
"""
import pytest

from automation.seo import seo
from utils.config import load_config

MIN_WORDS = int(load_config().get("seo", {}).get("min_description_words", 20))

WORD_COUNT_GUIDANCE = (
    "Write at least 30 words of rich description (aim for 40-60 words) — "
    "thin one-line descriptions destroy Shorts discoverability."
)

RICH_DESCRIPTION = (
    "Kohli smashed a brutal six over long-on and the Chinnaswamy crowd went "
    "absolutely berserk. RCB needed forty-five off the last eighteen balls "
    "and King Kohli decided to take matters into his own hands with pure "
    "aggression. This was a masterclass in chase finishing from one of the "
    "greatest batters the game has ever seen. Subscribe for more cricket action!"
)


def _make_item(description, title="Kohli ne maara CHHAKKA! 🔥"):
    return {
        "title": title,
        "description": description,
        "hashtags": ["#Shorts", "#Kohli"],
        "search_terms": ["kohli six"],
    }


class TestMinDescriptionWordsConfig:

    def test_config_has_min_description_words(self):
        seo_cfg = load_config().get("seo", {})
        assert "min_description_words" in seo_cfg
        assert isinstance(seo_cfg["min_description_words"], int)
        assert 20 <= seo_cfg["min_description_words"] <= 40


class TestWordCountQualityGate:

    def test_quality_gate_rejects_below_min_words(self):
        below = ("wordy " * (MIN_WORDS - 1)).strip()
        assert len(below.split()) == MIN_WORDS - 1
        assert len(below) >= 100
        assert not seo._validate_seo_quality(_make_item(below))

    def test_quality_gate_accepts_rich_description(self):
        assert len(RICH_DESCRIPTION.split()) >= MIN_WORDS
        assert seo._validate_seo_quality(_make_item(RICH_DESCRIPTION))

    def test_quality_gate_accepts_existing_good_description(self):
        good = (
            "📝 Virat Kohli smashes a massive six over long-on! The crowd at "
            "Chinnaswamy goes absolutely wild as King Kohli deposits the "
            "bowler into the stands. Subscribe for more!"
        )
        assert len(good.split()) >= MIN_WORDS
        assert seo._validate_seo_quality(_make_item(good))


class TestPromptWordCountGuidance:

    def test_prompt_tmpl_has_word_count_guidance(self):
        assert WORD_COUNT_GUIDANCE in seo._PROMPT_TMPL

    def test_salvage_tmpl_has_word_count_guidance(self):
        assert WORD_COUNT_GUIDANCE in seo._SALVAGE_TMPL