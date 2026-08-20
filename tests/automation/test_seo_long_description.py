"""TDD contract for config-driven, long, grounded Shorts descriptions."""
import pytest

from automation.seo import seo
from utils.config import load_config

SEO_CFG = load_config().get("seo", {})
MIN_CHARS = int(SEO_CFG.get("description_min_chars", 1200))
TARGET_CHARS = int(SEO_CFG.get("description_target_chars", 3200))
MAX_CHARS = int(SEO_CFG.get("description_max_chars", 4000))

RICH_DESCRIPTION = (
    "Virat Kohli batting analysis explains the complete cricket discussion "
    "without inventing a score or opponent. " * 30
).strip()


def _make_item(description, title="Kohli ne maara CHHAKKA! 🔥"):
    return {
        "title": title,
        "description": description,
        "hashtags": ["#Shorts", "#Kohli"],
        "search_terms": ["kohli six"],
    }


class TestDescriptionCharacterBudgetConfig:

    def test_config_has_long_description_budgets(self):
        assert 800 <= MIN_CHARS < TARGET_CHARS < MAX_CHARS
        assert MAX_CHARS == 4000


class TestDescriptionCharacterQualityGate:

    def test_quality_gate_rejects_below_min_chars(self):
        below = "Kohli cricket discussion. " * 20
        assert len(below) < MIN_CHARS
        assert not seo._validate_seo_quality(_make_item(below))

    def test_quality_gate_accepts_rich_description(self):
        assert MIN_CHARS <= len(RICH_DESCRIPTION) <= MAX_CHARS
        assert seo._validate_seo_quality(_make_item(RICH_DESCRIPTION))

    def test_limit_uses_configured_maximum(self):
        item = _make_item("x" * (MAX_CHARS + 500))
        limited = seo._enforce_limits(item)
        assert len(limited["description"]) == MAX_CHARS
