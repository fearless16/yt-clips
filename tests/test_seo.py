"""
test_seo.py — Tests for seo.py
Covers: _extract_keywords, _validate_tags, inject_trend_topics_into_tags,
        ensure_trend_in_title, validate_hinglish_content,
        validate_description_hooks, validate_emoji_usage, batch_generate_seo.
"""
import json
import pytest
from seo import (
    _extract_keywords,
    _validate_tags,
    inject_trend_topics_into_tags,
    ensure_trend_in_title,
    validate_hinglish_content,
    validate_description_hooks,
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
        kw = _extract_keywords(long_text, limit=14)
        assert len(kw) <= 14

    def test_empty_text(self):
        assert _extract_keywords("") == []
        assert _extract_keywords(None) == []


# ─── _validate_tags ──────────────────────────────────────────────────────────

class TestValidateTags:
    def test_generic_single_word_tags_removed(self):
        tags = ["cricket", "shorts", "viral", "trending"]
        valid = _validate_tags(tags)
        assert len(valid) == 0

    def test_specific_long_tail_tags_pass(self):
        tags = ["kohli six vs csk 2024", "rcb run chase thriller", "dhoni reaction ipl"]
        valid = _validate_tags(tags)
        assert len(valid) == 3

    def test_single_word_non_generic_still_filtered_by_min_words(self):
        """min_words=2 means even non-generic single words are removed."""
        tags = ["boundaries", "stumping"]
        valid = _validate_tags(tags, min_words=2)
        assert len(valid) == 0

    def test_empty_list(self):
        assert _validate_tags([]) == []

    def test_whitespace_only_tags_skipped(self):
        assert _validate_tags(["  ", "  "]) == []

    def test_all_generic_combo_still_filtered(self):
        """Tags whose words are all in GENERIC_TAGS set should be removed."""
        tags = ["cricket shorts", "viral trending"]
        valid = _validate_tags(tags)
        assert len(valid) == 0


# ─── inject_trend_topics_into_tags ───────────────────────────────────────────

class TestInjectTrendTopics:
    def test_injects_into_base_tags(self):
        result = inject_trend_topics_into_tags(
            ["kohli six", "rcb highlights"],
            ["IPL 2024", "Mumbai Indians"]
        )
        assert "kohli six" in result
        has_trend = any("ipl" in t.lower() or "mumbai" in t.lower() for t in result)
        assert has_trend

    def test_no_duplicates_when_already_present(self):
        result = inject_trend_topics_into_tags(
            ["ipl 2024", "kohli six"],
            ["IPL 2024"]
        )
        count = sum(1 for t in result if t.lower() == "ipl 2024")
        assert count == 1

    def test_player_name_combo_tag(self):
        result = inject_trend_topics_into_tags(
            ["cover drive"],
            ["IPL 2024"],
            player_name="virat kohli"
        )
        has_combo = any("virat kohli" in t and "ipl" in t.lower() for t in result)
        assert has_combo

    def test_empty_trends_returns_base(self):
        base = ["kohli six", "rcb"]
        result = inject_trend_topics_into_tags(base, [])
        assert result == base

    def test_top_3_trends_only(self):
        """Only top 3 trending topics should be injected."""
        trends = [f"trend{i}" for i in range(10)]
        result = inject_trend_topics_into_tags([], trends)
        injected = [t for t in result if t.startswith("trend")]
        assert len(injected) <= 3


# ─── ensure_trend_in_title ───────────────────────────────────────────────────

class TestEnsureTrendInTitle:
    def test_prepends_when_missing(self):
        title = "Kohli's Amazing Six!"
        result = ensure_trend_in_title(title, ["IPL 2024 Playoffs"])
        assert "IPL 2024" in result
        assert "Kohli" in result

    def test_unchanged_when_trend_present(self):
        title = "IPL 2024: Kohli's Six!"
        result = ensure_trend_in_title(title, ["IPL 2024"])
        assert result == title

    def test_truncated_to_100_chars(self):
        long = "X" * 200
        result = ensure_trend_in_title(long, ["IPL 2024"])
        assert len(result) <= 100

    def test_empty_trends_returns_truncated_original(self):
        title = "Kohli's Six!"
        result = ensure_trend_in_title(title, [])
        assert result == title[:100]

    def test_partial_trend_keyword_match(self):
        """'IPL 2024' in title should match trend 'IPL 2024 Playoffs'."""
        title = "IPL 2024 Kohli's Big Shot"
        result = ensure_trend_in_title(title, ["IPL 2024 Playoffs"])
        # Should not prepend again since 'IPL 2024' already in title
        assert not result.startswith("IPL 2024 Playoffs")


# ─── validate_hinglish_content ───────────────────────────────────────────────

class TestValidateHinglishContent:
    def test_hinglish_title_detected(self):
        is_valid, lang = validate_hinglish_content("Kohli ka Dhamaakedaar Six! 💥")
        assert is_valid is True
        assert "hinglish" in lang.lower()

    def test_pure_english_still_valid(self):
        is_valid, lang = validate_hinglish_content("Kohli hits a six over cover")
        assert is_valid is True
        assert lang == "english"

    def test_light_hinglish_one_hindi_word(self):
        is_valid, lang = validate_hinglish_content("Kohli ka cover drive!")
        assert is_valid is True
        assert lang in ("light_hinglish", "hinglish")

    def test_multiple_hindi_words_full_hinglish(self):
        is_valid, lang = validate_hinglish_content("Yeh shot tha aur kohli ne maara!")
        assert is_valid is True
        assert lang == "hinglish"


# ─── validate_description_hooks ──────────────────────────────────────────────

class TestValidateDescriptionHooks:
    def test_hindi_hook_detected(self):
        desc = "Kya shot tha yaar! 😱 Virat Kohli hits a massive six..."
        has_hook, hook_type = validate_description_hooks(desc)
        assert has_hook is True
        assert hook_type == "hindi"

    def test_emotional_hook_detected(self):
        desc = "insane shot by Kohli! Crowd goes wild!"
        has_hook, hook_type = validate_description_hooks(desc)
        assert has_hook is True
        assert hook_type == "emotional"

    def test_question_hook_detected(self):
        desc = "Did you see that? Kohli just hit a 100!"
        has_hook, hook_type = validate_description_hooks(desc)
        assert has_hook is True
        assert hook_type == "question_or_exclamation"

    def test_standard_hook_fallback(self):
        desc = "Virat Kohli hits a magnificent drive in the powerplay."
        has_hook, hook_type = validate_description_hooks(desc)
        assert has_hook is True
        assert hook_type == "standard"

    def test_empty_description(self):
        has_hook, hook_type = validate_description_hooks("")
        assert has_hook is False
        assert hook_type == "none"


# ─── validate_emoji_usage ────────────────────────────────────────────────────

class TestValidateEmojiUsage:
    def test_no_emojis_valid(self):
        assert validate_emoji_usage("Kohli's Amazing Six") is True

    def test_few_emojis_valid(self):
        assert validate_emoji_usage("Kohli's Six! 💥🔥") is True

    def test_too_many_emojis_invalid(self):
        assert validate_emoji_usage("💥🔥😱🎉🏆✨ Kohli") is False

    def test_exactly_five_emojis_valid(self):
        assert validate_emoji_usage("💥🔥😱🎉🏆") is True

    def test_six_emojis_invalid(self):
        assert validate_emoji_usage("💥🔥😱🎉🏆✨") is False


# ─── batch_generate_seo ──────────────────────────────────────────────────────

class TestBatchGenerateSEO:
    """Uses mocked AI + trends to avoid network/API calls."""

    _mock_response = {
        "clips_seo": [
            {"clip_id": "t1", "title": "Kohli's MASSIVE Six! 💥",
             "description": "Virat Kohli IPL 2024 hit. #Cricket", "tags": ["kohli six 2024", "rcb highlights ipl"]},
            {"clip_id": "t2", "title": "Dhoni Magic Stumping! 🔥",
             "description": "MS Dhoni quick stumping. #CSK", "tags": ["dhoni stumping csk", "msd magic 2024"]},
        ]
    }

    @pytest.fixture(autouse=True)
    def mock_ai_and_trends(self, monkeypatch):
        import seo, trends

        monkeypatch.setattr(seo.ai, "generate_text",
                            lambda prompt, system_instruction=None: json.dumps(self._mock_response))
        monkeypatch.setattr(trends, "get_trending_context",
                            lambda domain, region, video_title: {
                                "topics": ["IPL 2024", "Mumbai Indians"],
                                "tags": ["#Shorts", "#Cricket"],
                                "scorecard": "MI vs CSK"
                            })

    def test_returns_correct_count(self):
        clips = [{"clip_id": "t1", "text": "kohli six"},
                 {"clip_id": "t2", "text": "dhoni stumping"}]
        results = batch_generate_seo(clips)
        assert len(results) == 2

    def test_required_fields_present(self):
        clips = [{"clip_id": "t1", "text": "kohli six"}]
        r = batch_generate_seo(clips)[0]
        for key in ("clip_id", "title", "description", "tags", "hashtags", "trend_topics"):
            assert key in r, f"Missing: {key}"

    def test_title_max_100_chars(self):
        clips = [{"clip_id": "t1", "text": "kohli six"}]
        r = batch_generate_seo(clips)[0]
        assert len(r["title"]) <= 100

    def test_clip_ids_preserved(self):
        clips = [{"clip_id": "t1", "text": "a"}, {"clip_id": "t2", "text": "b"}]
        results = batch_generate_seo(clips)
        ids = {r["clip_id"] for r in results}
        assert "t1" in ids and "t2" in ids

    def test_trend_topics_in_result(self):
        clips = [{"clip_id": "t1", "text": "kohli six"}]
        r = batch_generate_seo(clips)[0]
        assert "trend_topics" in r
        assert len(r["trend_topics"]) > 0

    def test_raises_on_empty_ai_response(self, monkeypatch):
        import seo
        monkeypatch.setattr(seo.ai, "generate_text",
                            lambda *a, **kw: "{}")  # AI returns no clips_seo
        with pytest.raises(ValueError, match="SEO Generation Failed"):
            batch_generate_seo([{"clip_id": "t1", "text": "kohli"}])

    def test_tags_are_specific_after_validation(self):
        """Generic single-word tags should be stripped by _validate_tags."""
        clips = [{"clip_id": "t1", "text": "kohli six"}]
        r = batch_generate_seo(clips)[0]
        for tag in r["tags"]:
            assert len(tag.split()) >= 2 or any(
                specific in tag.lower() for specific in ("2024", "ipl", "kohli", "rcb", "mumbai")
            )
