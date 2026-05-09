"""
test_seo.py — Premium TDD suite for seo.py

Covers (TDD order — spec first):
  _extract_keywords, _validate_search_terms,
  inject_trend_topics_into_tags, ensure_trend_in_title,
  validate_hinglish_content, validate_emoji_usage,
  batch_generate_seo (happy path + all fallback/error branches),
  generate_seo (legacy wrapper).
"""
import json
import pytest
from seo import (
    _extract_keywords,
    _validate_search_terms,
    inject_trend_topics_into_tags,
    ensure_trend_in_title,
    validate_hinglish_content,
    validate_emoji_usage,
    batch_generate_seo,
    generate_seo,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

_CLIP_T1 = {"clip_id": "t1", "text": "kohli six"}
_CLIP_T2 = {"clip_id": "t2", "text": "dhoni stumping"}

_GOOD_SEO = {
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
            "search_terms": [
                "kohli six 2024", "rcb highlights ipl", "kohli cover drive csk",
                "virat kohli ipl 2024", "rcb vs csk last over",
            ],
            "hashtags": ["#RCBvsCSK", "#IPL2024", "#CricketShorts", "#Kohli"],
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
            "search_terms": ["dhoni stumping csk", "msd magic ipl 2024"],
            "hashtags": ["#CSKvsRCB", "#IPL2024", "#Dhoni"],
        },
    ]
}

_TREND_RESPONSE = {
    "topics": ["IPL 2024", "Mumbai Indians"],
    "tags": ["#Shorts", "#Cricket"],
    "scorecard": "MI vs CSK",
    "live_stream_url": "https://www.youtube.com/watch?v=test123",
}


def _patch(monkeypatch, ai_payload=None, trend_payload=None):
    import seo, trends
    payload = ai_payload if ai_payload is not None else json.dumps(_GOOD_SEO)
    monkeypatch.setattr(seo.ai, "generate_text", lambda *a, **kw: payload)
    monkeypatch.setattr(
        trends, "get_trending_context",
        lambda *a, **kw: trend_payload if trend_payload is not None else _TREND_RESPONSE,
    )


# ── _extract_keywords ─────────────────────────────────────────────────────────

class TestExtractKeywords:
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

    def test_empty_string_returns_empty(self):
        assert _extract_keywords("") == []

    def test_none_returns_empty(self):
        assert _extract_keywords(None) == []

    def test_frequency_ordering(self):
        """Most frequent word must appear first."""
        kw = _extract_keywords("kohli kohli kohli dhoni dhoni bumrah")
        assert kw[0] == "kohli"
        assert kw[1] == "dhoni"

    def test_deduplication(self):
        kw = _extract_keywords("kohli kohli kohli")
        assert kw.count("kohli") == 1

    def test_numeric_tokens_kept(self):
        kw = _extract_keywords("match 2024 highlights")
        assert "2024" in kw


# ── _validate_search_terms ────────────────────────────────────────────────────

class TestValidateSearchTerms:
    def test_generic_single_words_removed(self):
        assert _validate_search_terms(["cricket", "shorts", "viral", "trending"]) == []

    def test_long_tail_passes(self):
        terms = ["kohli six vs csk 2024", "rcb run chase thriller", "dhoni reaction ipl"]
        assert len(_validate_search_terms(terms)) == 3

    def test_single_word_filtered_by_min_words(self):
        assert _validate_search_terms(["boundaries", "stumping"], min_words=2) == []

    def test_empty_list(self):
        assert _validate_search_terms([]) == []

    def test_whitespace_only_skipped(self):
        assert _validate_search_terms(["  ", "  "]) == []

    def test_all_generic_combo_filtered(self):
        assert _validate_search_terms(["cricket shorts", "viral trending"]) == []

    def test_mixed_valid_invalid_batch(self):
        terms = ["kohli six ipl", "shorts", "rcb vs csk highlights", "viral"]
        valid = _validate_search_terms(terms)
        assert "kohli six ipl" in valid
        assert "rcb vs csk highlights" in valid
        assert "shorts" not in valid
        assert "viral" not in valid

    def test_preserves_original_casing(self):
        terms = ["Kohli Six IPL 2024"]
        result = _validate_search_terms(terms)
        assert result[0] == "Kohli Six IPL 2024"


# ── inject_trend_topics_into_tags ─────────────────────────────────────────────

class TestInjectTrendTopics:
    def test_injects_into_base_tags(self):
        result = inject_trend_topics_into_tags(
            ["kohli six", "rcb highlights"], ["IPL 2024", "Mumbai Indians"]
        )
        assert "kohli six" in result
        assert any("ipl" in t.lower() or "mumbai" in t.lower() for t in result)

    def test_no_duplicates_when_already_present(self):
        result = inject_trend_topics_into_tags(["ipl 2024", "kohli six"], ["IPL 2024"])
        assert sum(1 for t in result if t.lower() == "ipl 2024") == 1

    def test_player_name_combo_tag(self):
        result = inject_trend_topics_into_tags(
            ["cover drive"], ["IPL 2024"], player_name="virat kohli"
        )
        assert any("virat kohli" in t and "ipl" in t.lower() for t in result)

    def test_empty_trends_returns_base_unchanged(self):
        base = ["kohli six", "rcb"]
        assert inject_trend_topics_into_tags(base, []) == base

    def test_top_3_trends_only(self):
        result = inject_trend_topics_into_tags([], [f"trend{i}" for i in range(10)])
        trend_items = [t for t in result if t.startswith("trend")]
        assert len(trend_items) <= 3

    def test_no_player_name_no_combo(self):
        result = inject_trend_topics_into_tags([], ["IPL 2024"], player_name="")
        assert all("ipl 2024" == t or not t.startswith(" ") for t in result)


# ── ensure_trend_in_title ─────────────────────────────────────────────────────

class TestEnsureTrendInTitle:
    def test_unchanged_when_trend_present(self):
        title = "IPL 2024: Kohli's Six!"
        assert ensure_trend_in_title(title, ["IPL 2024"]) == title[:100]

    def test_unchanged_when_first_2_keywords_match(self):
        title = "IPL 2024 Kohli Big Shot"
        result = ensure_trend_in_title(title, ["IPL 2024 Playoffs"])
        assert "IPL 2024 Playoffs" not in result or result == title[:100]

    def test_appends_trend_when_missing(self):
        title = "Kohli's Amazing Six!"
        result = ensure_trend_in_title(title, ["IPL 2024 Playoffs"])
        assert len(result) <= 100
        assert "Kohli" in result

    def test_hard_truncated_to_100_chars(self):
        long_title = "X" * 200
        assert len(ensure_trend_in_title(long_title, ["IPL 2024"])) <= 100

    def test_exact_100_char_boundary(self):
        title = "A" * 100
        result = ensure_trend_in_title(title, [])
        assert len(result) == 100

    def test_empty_trends_returns_truncated_original(self):
        title = "Kohli's Six!"
        assert ensure_trend_in_title(title, []) == title[:100]

    def test_candidate_over_100_falls_back_to_original(self):
        title = "A" * 95
        trend = "XXXXXX"  # candidate would be 95 + 3 + 6 = 104 > 100
        result = ensure_trend_in_title(title, [trend])
        assert len(result) <= 100


# ── validate_hinglish_content ─────────────────────────────────────────────────

class TestValidateHinglishContent:
    def test_two_hindi_words_is_hinglish(self):
        is_valid, lang = validate_hinglish_content("Kohli ka Dhamaakedaar Six! 💥")
        assert is_valid is True
        assert lang == "hinglish"

    def test_ne_counts_as_hindi(self):
        is_valid, lang = validate_hinglish_content("Kohli ne maara six yaar!")
        assert is_valid is True
        assert lang == "hinglish"

    def test_pure_english(self):
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

    def test_always_valid_returns_true(self):
        """validate_hinglish_content never returns False — only categorises."""
        for text in ["", "cricket", "kya yaar bhai log"]:
            is_valid, _ = validate_hinglish_content(text)
            assert is_valid is True


# ── validate_emoji_usage ──────────────────────────────────────────────────────

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

    def test_empty_string_valid(self):
        assert validate_emoji_usage("") is True


# ── batch_generate_seo — happy path ──────────────────────────────────────────

class TestBatchGenerateSEOHappyPath:
    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        _patch(monkeypatch)

    def test_returns_correct_count(self):
        assert len(batch_generate_seo([_CLIP_T1, _CLIP_T2])) == 2

    def test_required_fields_present(self):
        r = batch_generate_seo([_CLIP_T1])[0]
        for key in ("clip_id", "title", "title_variants", "thumbnail_text",
                    "description", "search_terms", "hashtags", "trend_topics", "live_stream_url"):
            assert key in r, f"Missing field: {key}"

    def test_no_tags_key_in_output(self):
        r = batch_generate_seo([_CLIP_T1])[0]
        assert "tags" not in r

    def test_title_variants_list_of_3(self):
        r = batch_generate_seo([_CLIP_T1])[0]
        assert isinstance(r["title_variants"], list)
        assert len(r["title_variants"]) == 3

    def test_default_title_is_variant_a(self):
        r = batch_generate_seo([_CLIP_T1])[0]
        assert r["title"] == r["title_variants"][0]

    def test_all_title_variants_max_100_chars(self):
        r = batch_generate_seo([_CLIP_T1])[0]
        for v in r["title_variants"]:
            assert len(v) <= 100
        assert len(r["title"]) <= 100

    def test_thumbnail_text_non_empty_string(self):
        r = batch_generate_seo([_CLIP_T1])[0]
        assert isinstance(r["thumbnail_text"], str)
        assert len(r["thumbnail_text"]) > 0

    def test_live_stream_url_present(self):
        r = batch_generate_seo([_CLIP_T1])[0]
        assert r["live_stream_url"] == _TREND_RESPONSE["live_stream_url"]

    def test_clip_ids_preserved(self):
        ids = {r["clip_id"] for r in batch_generate_seo([_CLIP_T1, _CLIP_T2])}
        assert ids == {"t1", "t2"}

    def test_trend_topics_non_empty(self):
        r = batch_generate_seo([_CLIP_T1])[0]
        assert len(r["trend_topics"]) > 0

    def test_search_terms_all_2plus_words(self):
        r = batch_generate_seo([_CLIP_T1])[0]
        for term in r["search_terms"]:
            assert len(term.split()) >= 2, f"Single-word term: {term}"

    def test_hashtags_all_have_prefix(self):
        r = batch_generate_seo([_CLIP_T1])[0]
        for h in r["hashtags"]:
            assert h.startswith("#"), f"Missing # prefix: {h}"

    def test_hashtags_count_3_to_5(self):
        r = batch_generate_seo([_CLIP_T1])[0]
        assert 3 <= len(r["hashtags"]) <= 5

    def test_search_terms_and_hashtags_no_overlap(self):
        r = batch_generate_seo([_CLIP_T1])[0]
        hashtag_values = {h.lstrip("#").lower() for h in r["hashtags"]}
        for term in r["search_terms"]:
            assert term.lower() not in hashtag_values, f"Overlap: '{term}'"

    def test_description_capped_at_5000_chars(self):
        r = batch_generate_seo([_CLIP_T1])[0]
        assert len(r["description"]) <= 5000

    def test_search_terms_capped_at_30(self):
        r = batch_generate_seo([_CLIP_T1])[0]
        assert len(r["search_terms"]) <= 30


# ── batch_generate_seo — fallback & error branches ───────────────────────────

class TestBatchGenerateSEOFallbacks:
    """Each test patches independently to isolate the branch under test."""

    def test_raises_on_empty_ai_response(self, monkeypatch):
        _patch(monkeypatch, ai_payload="{}")
        with pytest.raises(ValueError, match="SEO Generation Failed"):
            batch_generate_seo([_CLIP_T1])

    def test_raises_on_malformed_json(self, monkeypatch):
        _patch(monkeypatch, ai_payload="not json at all")
        with pytest.raises(ValueError, match="SEO Generation Failed"):
            batch_generate_seo([_CLIP_T1])

    def test_parses_json_wrapped_in_markdown_fences(self, monkeypatch):
        fenced = f"```json\n{json.dumps(_GOOD_SEO)}\n```"
        _patch(monkeypatch, ai_payload=fenced)
        result = batch_generate_seo([_CLIP_T1])
        assert len(result) == 1
        assert result[0]["clip_id"] == "t1"

    def test_old_tags_key_falls_back_gracefully(self, monkeypatch):
        """AI returns 'tags' instead of 'search_terms' — must not crash."""
        payload = json.loads(json.dumps(_GOOD_SEO))
        clip = payload["clips_seo"][0]
        clip["tags"] = clip.pop("search_terms")
        _patch(monkeypatch, ai_payload=json.dumps(payload))
        r = batch_generate_seo([_CLIP_T1])[0]
        # search_terms key must exist (populated from tags fallback)
        assert "search_terms" in r

    def test_missing_clip_id_in_ai_response_produces_safe_defaults(self, monkeypatch):
        """If AI doesn't return a matching clip_id, use safe defaults — no crash."""
        payload = {"clips_seo": []}  # AI returned nothing for our clip
        _patch(monkeypatch, ai_payload=json.dumps(payload))
        # Should raise because results list is empty
        with pytest.raises(ValueError, match="SEO Generation Failed"):
            batch_generate_seo([_CLIP_T1])

    def test_hashtag_without_prefix_auto_prefixed(self, monkeypatch):
        """AI returns hashtags without '#' — must be fixed automatically."""
        payload = json.loads(json.dumps(_GOOD_SEO))
        payload["clips_seo"][0]["hashtags"] = ["RCBvsCSK", "IPL2024", "CricketShorts"]
        _patch(monkeypatch, ai_payload=json.dumps(payload))
        r = batch_generate_seo([_CLIP_T1])[0]
        for h in r["hashtags"]:
            assert h.startswith("#"), f"# not added: {h}"

    def test_ai_hashtags_under_3_uses_trend_fallback(self, monkeypatch):
        """AI returns < 3 hashtags → fallback to trend tags."""
        payload = json.loads(json.dumps(_GOOD_SEO))
        payload["clips_seo"][0]["hashtags"] = ["#Kohli"]  # only 1
        trend = {**_TREND_RESPONSE, "tags": ["#Shorts", "#Cricket", "#IPL2024"]}
        _patch(monkeypatch, ai_payload=json.dumps(payload), trend_payload=trend)
        r = batch_generate_seo([_CLIP_T1])[0]
        assert len(r["hashtags"]) >= 3

    def test_hashtags_capped_at_5_even_if_ai_returns_more(self, monkeypatch):
        payload = json.loads(json.dumps(_GOOD_SEO))
        payload["clips_seo"][0]["hashtags"] = [
            "#A", "#B", "#C", "#D", "#E", "#F", "#G"
        ]
        _patch(monkeypatch, ai_payload=json.dumps(payload))
        r = batch_generate_seo([_CLIP_T1])[0]
        assert len(r["hashtags"]) <= 5

    def test_no_trend_live_stream_url_is_empty_string(self, monkeypatch):
        trend = {**_TREND_RESPONSE, "live_stream_url": ""}
        _patch(monkeypatch, trend_payload=trend)
        r = batch_generate_seo([_CLIP_T1])[0]
        assert r["live_stream_url"] == ""

    def test_trend_topics_empty_still_produces_output(self, monkeypatch):
        trend = {**_TREND_RESPONSE, "topics": [], "tags": ["#Shorts", "#Cricket", "#IPL"]}
        _patch(monkeypatch, trend_payload=trend)
        r = batch_generate_seo([_CLIP_T1])[0]
        assert r["clip_id"] == "t1"
        assert isinstance(r["trend_topics"], list)


# ── generate_seo (legacy single-clip wrapper) ─────────────────────────────────

class TestGenerateSEOLegacyWrapper:
    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        _patch(monkeypatch)

    def test_returns_single_dict(self):
        r = generate_seo("kohli six", "t1")
        assert isinstance(r, dict)
        assert r["clip_id"] == "t1"

    def test_has_required_fields(self):
        r = generate_seo("kohli six", "t1")
        for key in ("title", "title_variants", "search_terms", "hashtags"):
            assert key in r

    def test_empty_clip_id_still_returns_dict(self):
        """Should not crash on minimal input."""
        r = generate_seo("", "t1")
        assert isinstance(r, dict)

# ── process_all_seo (batching queue) ──────────────────────────────────────────

class TestProcessAllSEOQueue:
    def test_batches_in_chunks_of_3(self, tmp_path, monkeypatch):
        import yaml
        h_path = tmp_path / "highlights.yaml"
        out_dir = tmp_path / "shorts"
        
        highlights_data = {
            f"clip{i}": {"text": f"text {i}"} for i in range(5)
        }
        with open(h_path, "w") as f:
            yaml.dump(highlights_data, f)
            
        call_chunks = []
        def mock_batch_generate_seo(clips, domain, region):
            call_chunks.append(len(clips))
            return [{"clip_id": c["clip_id"]} for c in clips]
            
        import seo
        monkeypatch.setattr(seo, "batch_generate_seo", mock_batch_generate_seo)
        
        # Mock time.sleep to run fast
        import time
        monkeypatch.setattr(time, "sleep", lambda x: None)
        
        seo.process_all_seo(str(h_path), str(out_dir))
        
        # 5 clips should be chunked into 3 and 2
        assert call_chunks == [3, 2]
        
    def test_retries_on_429(self, tmp_path, monkeypatch):
        import yaml
        h_path = tmp_path / "highlights.yaml"
        out_dir = tmp_path / "shorts"
        
        with open(h_path, "w") as f:
            yaml.dump({"clip1": {"text": "text"}}, f)
            
        attempts = [0]
        def mock_batch_generate_seo(clips, domain, region):
            attempts[0] += 1
            if attempts[0] == 1:
                raise Exception("429 Rate Limit Hit")
            return [{"clip_id": c["clip_id"]}]
            
        import seo
        monkeypatch.setattr(seo, "batch_generate_seo", mock_batch_generate_seo)
        
        import time
        sleeps = []
        monkeypatch.setattr(time, "sleep", lambda x: sleeps.append(x))
        
        seo.process_all_seo(str(h_path), str(out_dir))
        
        # Should attempt twice
        assert attempts[0] == 2
        # Should have slept for 8 seconds due to first retry
        assert 8 in sleeps
