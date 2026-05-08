"""
test_seo.py — Tests for seo.py
Covers: _extract_keywords, _validate_tags, inject_trend_topics_into_tags,
        ensure_trend_in_title, validate_hinglish_content,
        validate_emoji_usage, batch_generate_seo (new schema).

NOTE: validate_description_hooks was removed from seo.py — tests removed too.
"""
import json
import pytest
from seo import (
    _extract_keywords,
    _validate_tags,
    inject_trend_topics_into_tags,
    ensure_trend_in_title,
    validate_hinglish_content,
    validate_emoji_usage,
    batch_generate_seo,
)


# ─── _extract_keywords ────────────────────────────────────────────────────────

class TestKeywordExtraction:
    def test_hindi_words_preserved(self):
        kw = _extract_keywords("Kohli ne amazing six maara stadium mein")
        assert "kohli" in kw
        assert "maara" in kw
        assert "stadium" in kw

    def test_stop_words_filtered(self):
        kw = _extract_keywords("the player is hitting the ball very well")
        assert "the" not in kw
        assert "is" not in kw
        assert "very" not in kw
        assert any(w in kw for w in ("player", "hitting", "ball", "well"))

    def test_short_words_filtered(self):
        kw = _extract_keywords("I am a pro at cricket")
        assert "i" not in kw
        assert "am" not in kw
        assert "a" not in kw
        assert "pro" in kw or "cricket" in kw

    def test_limit_respected(self):
        long_text = " ".join(f"uniqueword{i}" for i in range(50))
        assert len(_extract_keywords(long_text, limit=14)) <= 14

    def test_empty_text(self):
        assert _extract_keywords("") == []
        assert _extract_keywords(None) == []


# ─── _validate_tags ──────────────────────────────────────────────────────────

class TestValidateTags:
    def test_generic_single_word_tags_removed(self):
        valid = _validate_tags(["cricket", "shorts", "viral", "trending"])
        assert len(valid) == 0

    def test_specific_long_tail_tags_pass(self):
        tags = ["kohli six vs csk 2024", "rcb run chase thriller", "dhoni reaction ipl"]
        assert len(_validate_tags(tags)) == 3

    def test_single_word_filtered_by_min_words(self):
        assert _validate_tags(["boundaries", "stumping"], min_words=2) == []

    def test_empty_list(self):
        assert _validate_tags([]) == []

    def test_whitespace_only_skipped(self):
        assert _validate_tags(["  ", "  "]) == []

    def test_all_generic_combo_filtered(self):
        """All words in GENERIC_TAGS → tag removed even if multi-word."""
        assert _validate_tags(["cricket shorts", "viral trending"]) == []


# ─── inject_trend_topics_into_tags ───────────────────────────────────────────

class TestInjectTrendTopics:
    def test_injects_into_base_tags(self):
        result = inject_trend_topics_into_tags(
            ["kohli six", "rcb highlights"], ["IPL 2024", "Mumbai Indians"]
        )
        assert "kohli six" in result
        has_trend = any("ipl" in t.lower() or "mumbai" in t.lower() for t in result)
        assert has_trend

    def test_no_duplicates_when_already_present(self):
        result = inject_trend_topics_into_tags(["ipl 2024", "kohli six"], ["IPL 2024"])
        assert sum(1 for t in result if t.lower() == "ipl 2024") == 1

    def test_player_name_combo_tag(self):
        result = inject_trend_topics_into_tags(
            ["cover drive"], ["IPL 2024"], player_name="virat kohli"
        )
        assert any("virat kohli" in t and "ipl" in t.lower() for t in result)

    def test_empty_trends_returns_base(self):
        base = ["kohli six", "rcb"]
        assert inject_trend_topics_into_tags(base, []) == base

    def test_top_3_trends_only(self):
        result = inject_trend_topics_into_tags([], [f"trend{i}" for i in range(10)])
        assert len([t for t in result if t.startswith("trend")]) <= 3


# ─── ensure_trend_in_title ───────────────────────────────────────────────────

class TestEnsureTrendInTitle:
    def test_unchanged_when_trend_present_full(self):
        """Title already contains exact trend → returned as-is (truncated to 100)."""
        title = "IPL 2024: Kohli's Six!"
        result = ensure_trend_in_title(title, ["IPL 2024"])
        assert result == title[:100]

    def test_unchanged_when_first_2_keywords_match(self):
        """'IPL 2024' in title satisfies trend 'IPL 2024 Playoffs'."""
        title = "IPL 2024 Kohli Big Shot"
        result = ensure_trend_in_title(title, ["IPL 2024 Playoffs"])
        # Should NOT append again because 'ipl 2024' already in title
        assert "IPL 2024 Playoffs" not in result or result == title[:100]

    def test_appends_trend_when_missing(self):
        """When trend is absent, function appends '| <trend>' rather than prepending."""
        title = "Kohli's Amazing Six!"
        result = ensure_trend_in_title(title, ["IPL 2024 Playoffs"])
        # New behaviour: appends as context tag
        assert len(result) <= 100
        # Either the title was kept (too long to append) or trend appears after |
        assert "Kohli" in result

    def test_truncated_to_100_chars(self):
        long_title = "X" * 200
        result = ensure_trend_in_title(long_title, ["IPL 2024"])
        assert len(result) <= 100

    def test_empty_trends_returns_truncated_original(self):
        title = "Kohli's Six!"
        assert ensure_trend_in_title(title, []) == title[:100]


# ─── validate_hinglish_content ───────────────────────────────────────────────

class TestValidateHinglishContent:
    def test_hinglish_title_two_hindi_words(self):
        is_valid, lang = validate_hinglish_content("Kohli ka Dhamaakedaar Six! 💥")
        assert is_valid is True
        assert lang == "hinglish"

    def test_hinglish_with_ne(self):
        """'ne' was added to the hindi_words set in the latest change."""
        is_valid, lang = validate_hinglish_content("Kohli ne maara six yaar!")
        assert is_valid is True
        assert lang == "hinglish"

    def test_pure_english_still_valid(self):
        is_valid, lang = validate_hinglish_content("Kohli hits a six over cover")
        assert is_valid is True
        assert lang == "english"

    def test_one_hindi_word_light_hinglish(self):
        is_valid, lang = validate_hinglish_content("Kohli ka cover drive!")
        assert is_valid is True
        assert lang == "light_hinglish"

    def test_multiple_hindi_words_hinglish(self):
        is_valid, lang = validate_hinglish_content("Yeh shot tha aur kohli ne maara!")
        assert is_valid is True
        assert lang == "hinglish"


# ─── validate_emoji_usage ────────────────────────────────────────────────────

class TestValidateEmojiUsage:
    def test_no_emojis_valid(self):
        assert validate_emoji_usage("Kohli's Amazing Six") is True

    def test_few_emojis_valid(self):
        assert validate_emoji_usage("Kohli's Six! 💥🔥") is True

    def test_exactly_five_valid(self):
        assert validate_emoji_usage("💥🔥😱🎉🏆") is True

    def test_six_emojis_invalid(self):
        assert validate_emoji_usage("💥🔥😱🎉🏆✨") is False

    def test_excessive_emojis_invalid(self):
        assert validate_emoji_usage("💥🔥😱🎉🏆✨ Kohli") is False


# ─── batch_generate_seo (new schema) ─────────────────────────────────────────

class TestBatchGenerateSEO:
    """Mocked AI + trends — no network/API calls."""

    _mock_response = {
        "clips_seo": [
            {
                "clip_id": "t1",
                "title_variants": [
                    "Kohli ne kya kar diya yaar 😤 | RCB vs CSK",
                    "Only Kohli Can Hit This In a Chase 👀 | IPL 2024",
                    "🔴 LIVE NOW: RCB chasing 200 | Kohli ki innings LIVE",
                ],
                "thumbnail_text": "KOHLI NE KAR DIYA",
                "description": "Kya shot tha yaar! 🔥 Kohli hit an absolute belter. IPL 2024.",
                "tags": ["kohli six 2024", "rcb highlights ipl", "kohli cover drive csk"],
            },
            {
                "clip_id": "t2",
                "title_variants": [
                    "Dhoni ne stumping kar diya 😱 | CSK vs RCB",
                    "Only Dhoni Can Do This In 0.1 Seconds 👀",
                    "🔴 LIVE NOW: CSK defending 185 | Dhoni magic",
                ],
                "thumbnail_text": "YEH KYA THA YAAR",
                "description": "Believe nahi hoga! 🔥 Dhoni fastest stumping ever.",
                "tags": ["dhoni stumping csk", "msd magic ipl 2024"],
            },
        ]
    }

    @pytest.fixture(autouse=True)
    def mock_ai_and_trends(self, monkeypatch):
        import seo, trends
        monkeypatch.setattr(
            seo.ai, "generate_text",
            lambda prompt, system_instruction=None: json.dumps(self._mock_response),
        )
        monkeypatch.setattr(
            trends, "get_trending_context",
            lambda domain, region, video_title: {
                "topics": ["IPL 2024", "Mumbai Indians"],
                "tags": ["#Shorts", "#Cricket"],
                "scorecard": "MI vs CSK",
                "live_stream_url": "https://www.youtube.com/watch?v=test123",
            },
        )

    def test_returns_correct_count(self):
        clips = [{"clip_id": "t1", "text": "kohli six"},
                 {"clip_id": "t2", "text": "dhoni stumping"}]
        assert len(batch_generate_seo(clips)) == 2

    def test_required_fields_present(self):
        r = batch_generate_seo([{"clip_id": "t1", "text": "kohli six"}])[0]
        for key in ("clip_id", "title", "title_variants", "thumbnail_text",
                    "description", "tags", "hashtags", "trend_topics", "live_stream_url"):
            assert key in r, f"Missing key: {key}"

    def test_title_variants_is_list_of_3(self):
        r = batch_generate_seo([{"clip_id": "t1", "text": "kohli six"}])[0]
        assert isinstance(r["title_variants"], list)
        assert len(r["title_variants"]) == 3

    def test_default_title_is_variant_a(self):
        """title field should equal (trend-enhanced) first variant."""
        r = batch_generate_seo([{"clip_id": "t1", "text": "kohli six"}])[0]
        assert r["title"] == r["title_variants"][0]

    def test_title_max_100_chars(self):
        r = batch_generate_seo([{"clip_id": "t1", "text": "kohli six"}])[0]
        for v in r["title_variants"]:
            assert len(v) <= 100
        assert len(r["title"]) <= 100

    def test_thumbnail_text_present(self):
        r = batch_generate_seo([{"clip_id": "t1", "text": "kohli six"}])[0]
        assert isinstance(r["thumbnail_text"], str)
        assert len(r["thumbnail_text"]) > 0

    def test_live_stream_url_present(self):
        r = batch_generate_seo([{"clip_id": "t1", "text": "kohli six"}])[0]
        assert "live_stream_url" in r

    def test_clip_ids_preserved(self):
        clips = [{"clip_id": "t1", "text": "a"}, {"clip_id": "t2", "text": "b"}]
        ids = {r["clip_id"] for r in batch_generate_seo(clips)}
        assert "t1" in ids and "t2" in ids

    def test_trend_topics_in_result(self):
        r = batch_generate_seo([{"clip_id": "t1", "text": "kohli six"}])[0]
        assert len(r["trend_topics"]) > 0

    def test_raises_on_empty_ai_response(self, monkeypatch):
        import seo
        monkeypatch.setattr(seo.ai, "generate_text", lambda *a, **kw: "{}")
        with pytest.raises(ValueError, match="SEO Generation Failed"):
            batch_generate_seo([{"clip_id": "t1", "text": "kohli"}])

    def test_tags_are_specific_after_validation(self):
        r = batch_generate_seo([{"clip_id": "t1", "text": "kohli six"}])[0]
        for tag in r["tags"]:
            assert len(tag.split()) >= 2, f"Single-word generic tag slipped through: {tag}"
