"""Tests for natural keyword embedding validation in SEO output.

Verifies that _validate_seo_quality catches:
- Keyword/tag/search_terms block appended at end of description
- Comma-separated keyword dumps in last 200 chars
"""

import pytest

from automation.seo.seo import _validate_seo_quality, _enforce_limits


def make_item(title, description, hashtags=None, search_terms=None):
    return {
        "title": title,
        "description": description,
        "hashtags": hashtags or ["#Shorts"],
        "search_terms": search_terms or [],
    }


class TestNaturalEmbedding:
    def test_natural_description_passes(self):
        item = make_item(
            "Kohli ka CHHAKKA! 🔥 | RCB vs MI IPL 2026",
            "Virat Kohli smashed a massive six over long-on! The crowd at Chinnaswamy went wild. "
            "RCB needed 45 off 18 and Kohli decided to take matters into his own hands. This was "
            "pure class from the King. RCB vs MI live score shows RCB 198/4. Subscribe for more "
            "cricket action! #Shorts #RCB #MI",
        )
        assert _validate_seo_quality(item)

    def test_keyword_block_at_end_rejected(self):
        item = make_item(
            "Kohli ka CHHAKKA! 🔥",
            "Great shot by Kohli. The crowd loved it. Subscribe for more.\n"
            "Tags: cricket, kohli, six, rcb, mi, ipl, 2026, live, score",
        )
        assert not _validate_seo_quality(item)

    def test_keywords_block_with_label_rejected(self):
        item = make_item(
            "Bumrah yorker!🔥",
            "Jasprit Bumrah bowled an incredible yorker.\n"
            "Keywords: bumrah, yorker, mi, rcb, ipl, cricket, live, score",
        )
        assert not _validate_seo_quality(item)

    def test_comma_separated_single_words_on_last_line_rejected(self):
        """Single words separated by commas on last line = keyword dump."""
        item = make_item(
            "Dhoni finish! 🔥",
            "MS Dhoni finished the game in style.\n"
            "dhoni, six, csk, mi, ipl, 2026, live, match, score, final",
        )
        assert not _validate_seo_quality(item)

    def test_short_description_rejected(self):
        item = make_item("Kohli six 🔥 RCB vs MI", "Short")
        assert not _validate_seo_quality(item)  # desc < 100

    def test_descriptions_with_hashtags_at_end_pass(self):
        """Hashtags at end are OK — they're #hashtag format, not keyword dumps."""
        item = make_item(
            "Sky high! 🔥 | Suryakumar Yadav 360°",
            "Suryakumar Yadav played an incredible 360 degree shot. "
            "Pure innovation. Subscribe for more cricket action!\n\n"
            "#Shorts #SuryakumarYadav #MI #IPL2026 #Cricket #Six",
        )
        assert _validate_seo_quality(item)
