"""Hard cricket-only and context-aware short generation contracts."""

import pytest


def test_cricket_alias_resolves_yuvi_but_never_uv():
    from automation.seo.cricket_context import correct_cricket_spelling

    assert correct_cricket_spelling("Yuvi ko coach bana do") == (
        "Yuvraj Singh ko coach bana do"
    )
    assert correct_cricket_spelling("UV index bahut high hai") == (
        "UV index bahut high hai"
    )
    canonical = "Yuvraj Singh says Jasprit Bumrah is brilliant"
    assert correct_cricket_spelling(correct_cricket_spelling(canonical)) == canonical


def test_transliterated_team_names_are_canonicalized_for_source_matching():
    from automation.clip_selection.pipeline import _filter_source_match_candidates
    from automation.seo.cricket_context import correct_cricket_spelling

    assert correct_cricket_spelling("inglainda aur paakistaana") == (
        "England aur Pakistan"
    )
    candidates = [
        {"text": "inglainda ne paakistaana ke paancha wicket lie"},
        {"text": "India kee bolinga kharaaba thee"},
        {"text": "paakistaana ko eka wicket chaahie"},
    ]
    assert _filter_source_match_candidates(
        candidates, "PAK vs ENG Test", minimum_matches=2
    ) == [candidates[0], candidates[2]]


def test_cricket_context_gate_understands_alias_and_rejects_tech_chatter():
    from automation.seo.cricket_context import is_cricket_content

    assert is_cricket_content("Yuvi ko India ka coach bana do")
    assert is_cricket_content("What a shot!", "India vs Australia cricket match")
    assert not is_cricket_content("Windows Defender download ka issue hai")
    assert not is_cricket_content("party abhi shuru hui hai bro")


def test_cricket_gate_understands_hindi_youtube_captions():
    from automation.seo.cricket_context import is_cricket_content

    source = "England vs Pakistan Test cricket live commentary"
    assert is_cricket_content(
        "अभी तक 233 रन हो चुके हैं और इंग्लैंड के चार विकेट गिर चुके हैं",
        source,
    )
    assert is_cricket_content(
        "हैरी ब्रुक 20 बॉल पर 15 रन बनाकर खेल रहे हैं",
        source,
    )
    assert not is_cricket_content(
        "मैं अपना माइक प्लग इन करना भूल गया था",
        source,
    )


def test_shared_surname_does_not_hallucinate_extra_players():
    from automation.seo.cricket_context import find_canonical_entities

    assert find_canonical_entities("Yuvraj Singh should coach India")["players"] == [
        "Yuvraj Singh"
    ]


def test_complete_thoughts_are_split_and_canonicalized():
    from automation.clip_selection.pipeline import _prepare_complete_thoughts

    result = _prepare_complete_thoughts([{
        "start": 0.0,
        "end": 10.0,
        "text": "Yuvi should coach India. This would be brilliant!",
    }])

    assert [item["text"] for item in result] == [
        "Yuvraj Singh should coach India.",
        "This would be brilliant!",
    ]
    assert result[0]["end"] <= result[1]["start"]


def test_rolling_caption_fragments_become_one_complete_thought():
    from automation.clip_selection.pipeline import _prepare_complete_thoughts

    result = _prepare_complete_thoughts([
        {"start": 10.0, "end": 14.0, "text": "हैरी ब्रुक अभी"},
        {"start": 13.0, "end": 17.0, "text": "20 बॉल पर 15 रन बनाकर"},
        {"start": 16.0, "end": 20.0, "text": "खेल रहे हैं लेकिन उनका इंजन"},
        {"start": 19.0, "end": 23.0, "text": "अभी गरम नहीं हुआ है यार।"},
    ])

    assert len(result) == 1
    assert result[0]["start"] == 10.0
    assert result[0]["end"] == 23.0
    assert "20 बॉल पर 15 रन" in result[0]["text"]
    assert result[0]["text"].endswith("यार।")


def test_spoken_bounds_remove_leading_and_trailing_silence():
    from automation.clip_selection.pipeline import _prepare_complete_thoughts

    result = _prepare_complete_thoughts([{
        "start": 0.0,
        "end": 10.0,
        "text": "Bumrah takes the wicket!",
        "words": [
            {"start": 2.0, "end": 3.0, "word": "Bumrah"},
            {"start": 3.1, "end": 4.0, "word": "takes"},
            {"start": 4.1, "end": 5.0, "word": "the"},
            {"start": 5.1, "end": 6.0, "word": "wicket!"},
        ],
    }])

    assert result[0]["start"] == pytest.approx(1.9)
    assert result[0]["end"] == pytest.approx(6.1)


def test_candidate_gate_is_strictly_cricket_only():
    from automation.clip_selection.pipeline import _filter_cricket_candidates

    candidates = [
        {"start": 0, "end": 5, "text": "Yuvi ko coach bana do"},
        {"start": 6, "end": 11, "text": "audio capture device update karo"},
    ]

    kept = _filter_cricket_candidates(candidates, "India cricket discussion")
    assert [item["text"] for item in kept] == ["Yuvi ko coach bana do"]


def test_match_stream_candidates_prefer_source_teams():
    from automation.clip_selection.pipeline import _filter_source_match_candidates

    candidates = [
        {"text": "England 100 runs aur bana de to Pakistan pressure mein hai"},
        {"text": "India Ireland T20 domination khatam"},
        {"text": "Pakistan ko England ke wickets jaldi lene honge"},
    ]

    kept = _filter_source_match_candidates(
        candidates,
        "England vs Pakistan 1st Test live commentary",
        minimum_matches=2,
    )

    assert kept == [candidates[0], candidates[2]]


def test_source_match_filter_falls_back_when_too_few_candidates_match():
    from automation.clip_selection.pipeline import _filter_source_match_candidates

    candidates = [
        {"text": "England lead is growing"},
        {"text": "great captaincy debate"},
    ]

    assert _filter_source_match_candidates(
        candidates,
        "England vs Pakistan Test",
        minimum_matches=2,
    ) == candidates


def test_livestream_windows_join_short_setup_and_payoff_without_filler():
    from automation.clip_selection.pipeline import _build_livestream_windows

    thoughts = [
        {"start": 0.0, "end": 5.0, "text": "They were under huge pressure at that stage", "score": 4.0},
        {"start": 5.0, "end": 12.0, "text": "but this fifty run partnership has changed the Test match", "score": 9.0},
        {"start": 12.0, "end": 17.0, "text": "hello bro like the stream and subscribe", "score": 20.0},
        {"start": 17.0, "end": 25.0, "text": "England need one wicket to break Pakistan again", "score": 8.0},
    ]

    windows = _build_livestream_windows(thoughts, min_duration=10, max_duration=20)

    assert windows
    assert all(10 <= window["end"] - window["start"] <= 20 for window in windows)
    assert any(
        window["start"] == 0.0
        and window["end"] == 12.0
        and "partnership" in window["text"]
        for window in windows
    )
    assert all("subscribe" not in window["text"].lower() for window in windows)


def test_livestream_windows_require_cricket_evidence_inside_the_cut():
    from automation.clip_selection.pipeline import _build_livestream_windows

    thoughts = [
        {"start": 0.0, "end": 6.0, "text": "This captain is completely wrong", "score": 12.0},
        {"start": 6.0, "end": 13.0, "text": "I cannot believe this decision at all", "score": 14.0},
    ]

    assert _build_livestream_windows(
        thoughts, min_duration=10, max_duration=20
    ) == []


def test_livestream_windows_reject_live_views_strategy_chatter():
    from automation.clip_selection.pipeline import _build_livestream_windows

    thoughts = [{
        "start": 0.0,
        "end": 16.0,
        "text": (
            "India pe laaiva kara rahaa hoon, vyooja ke lie jo live match "
            "hai usako lagaanaa padataa hai, this is strategy"
        ),
        "score": 20.0,
    }]

    assert _build_livestream_windows(
        thoughts, min_duration=10, max_duration=20
    ) == []


def test_livestream_windows_reject_long_tangent_with_cricket_only_at_the_end():
    from automation.clip_selection.pipeline import _build_livestream_windows

    thoughts = [{
        "start": 0.0,
        "end": 30.0,
        "text": (
            "I had a girlfriend after we broke up now I will directly marry "
            "and people say crush can happen later too anyway what is the "
            "whole point of discussing relationships and old crushes because "
            "none of this has anything to do with the event we are watching "
            "and the conversation keeps wandering without making a useful point "
            "England cricket score"
        ),
        "score": 20.0,
    }]

    assert _build_livestream_windows(
        thoughts, min_duration=10, max_duration=38
    ) == []


def test_diverse_livestream_windows_do_not_return_same_moment_variants():
    from automation.clip_selection.pipeline import _select_diverse_windows

    windows = [
        {"start": 0.0, "end": 20.0, "text": "a", "score": 10.0},
        {"start": 2.0, "end": 21.0, "text": "b", "score": 9.0},
        {"start": 40.0, "end": 55.0, "text": "c", "score": 8.0},
    ]

    selected = _select_diverse_windows(windows, max_candidates=3)

    assert selected == [windows[0], windows[2]]


def test_export_gate_blocks_stale_non_cricket_highlight():
    from export import _is_exportable_cricket_highlight

    context = "India vs Australia cricket discussion"
    assert _is_exportable_cricket_highlight("What a shot!", context)
    assert not _is_exportable_cricket_highlight(
        "Windows Defender download issue", context
    )


def test_seo_rejects_non_cricket_before_calling_ai(monkeypatch):
    import automation.seo.seo as seo

    attempt = lambda *args, **kwargs: pytest.fail("AI must not run for non-cricket")
    monkeypatch.setattr(seo, "_attempt_seo_generation", attempt)

    with pytest.raises(seo.SEOGenerationError, match="non-cricket"):
        seo.generate_clip_seo(
            "clip1",
            "Windows Defender download ka issue hai",
            video_title="PC troubleshooting stream",
        )


def test_seo_prompt_uses_canonical_grounded_player(monkeypatch):
    import automation.seo.seo as seo

    captured = {}

    def fake_attempt(clip_id, user_prompt, transcript, video_title, is_shorts,
                     provider_override=None, model_override=None,
                     sys_instruction=None, salvage_tmpl=None):
        captured["prompt"] = user_prompt
        captured["transcript"] = transcript
        captured["salvage"] = salvage_tmpl
        queries = [
            "yuvraj singh coach",
            "yuvraj singh india coach",
            "yuvi coach india",
            "yuvraj singh coaching debate",
            "team india coach discussion",
            "yuvraj singh cricket brain",
            "yuvraj singh mentor india",
            "should yuvraj singh coach india",
        ]
        description = ". ".join(queries) + ". " + (
            "This detailed grounded cricket discussion explains the complete "
            "coaching opinion without inventing match facts. " * 80
        )
        return {"title": "Yuvraj Singh Coach Debate", "description": description,
                "hashtags": ["#YuvrajSingh", "#Cricket", "#Shorts"],
                "search_terms": queries}

    monkeypatch.setattr(seo, "_attempt_seo_generation", fake_attempt)
    monkeypatch.setattr(seo, "extract_ocr_entities", lambda *_: {})

    seo.generate_clip_seo(
        "clip1",
        "Yuvi ko India ka coach bana do",
        video_title="India cricket discussion",
        video_description=(
            "Prajjwal explains why Yuvi has the temperament and cricket brain "
            "to mentor the Indian team."
        ),
        approved_search_queries=[
            "yuvraj singh coach",
            "yuvraj singh india coach",
            "yuvi coach india",
            "yuvraj singh coaching debate",
            "team india coach discussion",
            "yuvraj singh cricket brain",
            "yuvraj singh mentor india",
            "should yuvraj singh coach india",
        ],
    )

    assert captured["transcript"] == "Yuvraj Singh ko India ka coach bana do"
    assert "Grounded players: Yuvraj Singh" in captured["prompt"]
    assert "4000" in captured["prompt"]
    # assert str(seo._description_target_chars()) in captured["prompt"]
    assert "Prajjwal explains why Yuvraj Singh has the temperament" in captured["prompt"]
    assert "Yuvraj Singh ko India ka coach bana do" in captured["prompt"]
    assert "12-25" in captured["prompt"] and "long-tail" in captured["prompt"]
    assert "max 60 characters" in captured["salvage"]


def test_seo_rejects_hallucinated_player_from_ai(monkeypatch):
    import automation.seo.seo as seo

    monkeypatch.setattr(seo, "extract_ocr_entities", lambda *_: {})
    monkeypatch.setattr(
        seo,
        "_attempt_seo_generation",
        lambda *args, **kwargs: {
            "title": "Virat Kohli Coach Debate",
            "description": "Virat Kohli should become coach. " * 8,
            "hashtags": ["#ViratKohli", "#Cricket", "#Shorts"],
            "search_terms": ["virat kohli coach"],
        },
    )

    with pytest.raises(seo.SEOGenerationError, match="ungrounded"):
        seo.generate_clip_seo(
            "clip1",
            "Yuvi ko India ka coach bana do",
            video_title="India cricket discussion",
        )


def test_short_duration_config_prioritizes_natural_pace_complete_thoughts():
    from utils.config import load_config

    highlight = load_config()["highlight"]
    assert highlight["min_duration"] >= 8
    assert highlight["target_duration"] == 15
    assert highlight["max_duration"] <= 45
    assert highlight["merge_gap"] == 0


def test_seo_enforces_mobile_title_and_focused_search_term_caps():
    from automation.seo.seo import _enforce_limits

    result = _enforce_limits({
        "title": "Yuvraj Singh " + "coach debate " * 10,
        "description": "Yuvraj Singh cricket coach debate. " * 200,
        "hashtags": ["#Shorts", "#YuvrajSingh", "#Cricket"],
        "search_terms": [f"yuvraj cricket phrase {index}" for index in range(30)],
    })

    assert len(result["title"]) <= 70
    assert len(result["description"]) <= 4800
    assert len(result["search_terms"]) <= 26
