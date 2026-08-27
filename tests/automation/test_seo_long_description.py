"""TDD contract for config-driven long-tail Shorts descriptions.

promise_v4_longtail: the description is algorithm-facing keyword surface,
not human reading material. Budgets live in config.yaml under seo
(3000 / 3500 / 4500 by default, hard-capped under YouTube's 5000).
"""
import pytest

from automation.seo import seo
from utils.config import load_config

SEO_CFG = load_config().get("seo", {})
MIN_CHARS = seo._description_min_chars()
TARGET_CHARS = seo._description_target_chars()
MAX_CHARS = seo._description_max_chars()

RICH_DESCRIPTION = (
    "Virat Kohli batting analysis explains the complete cricket discussion "
    "without inventing a score or opponent. "
).strip() * 2


def _make_item(description, title="Kohli ne maara CHHAKKA! 🔥"):
    return {
        "title": title,
        "description": description,
        "hashtags": ["#Shorts", "#Kohli"],
        "search_terms": ["kohli six"],
    }


class TestDescriptionCharacterBudgetConfig:

    def test_config_has_right_sized_description_budgets(self):
        assert 100 <= MIN_CHARS < TARGET_CHARS < MAX_CHARS <= 400


class TestDescriptionCharacterQualityGate:

    def test_quality_gate_rejects_below_min_chars(self):
        below = "Kohli cricket discussion. " * 3  # ~78 chars < 150
        assert len(below) < MIN_CHARS
        assert not seo._validate_seo_quality(_make_item(below))

    def test_quality_gate_accepts_rich_description(self):
        assert MIN_CHARS <= len(RICH_DESCRIPTION) <= MAX_CHARS
        assert seo._validate_seo_quality(_make_item(RICH_DESCRIPTION))

    def test_limit_uses_configured_maximum(self):
        item = _make_item("x " * (MAX_CHARS + 500))
        limited = seo._enforce_limits(item)
        assert len(limited["description"]) == MAX_CHARS - 1  # rstrip of trail space
