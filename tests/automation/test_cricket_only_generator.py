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


def test_cricket_context_gate_understands_alias_and_rejects_tech_chatter():
    from automation.seo.cricket_context import is_cricket_content

    assert is_cricket_content("Yuvi ko India ka coach bana do")
    assert is_cricket_content("What a shot!", "India vs Australia cricket match")
    assert not is_cricket_content("Windows Defender download ka issue hai")
    assert not is_cricket_content("party abhi shuru hui hai bro")


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
            "coaching opinion without inventing match facts. " * 20
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
    assert "TITLE: maximum 70 characters" in captured["prompt"]
    assert "DESCRIPTION: target 3200 characters; maximum 4000 characters" in captured["prompt"]
    assert "Prajjwal explains why Yuvraj Singh has the temperament" in captured["prompt"]
    assert "Yuvraj Singh ko India ka coach bana do" in captured["prompt"]
    assert "8-15" in captured["prompt"]
    assert "maximum 70 characters" in captured["salvage"]


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

    with pytest.raises(seo.SEOGenerationError, match="ungrounded entities"):
        seo.generate_clip_seo(
            "clip1",
            "Yuvi ko India ka coach bana do",
            video_title="India cricket discussion",
        )


def test_short_duration_config_prioritizes_minimal_complete_thoughts():
    from utils.config import load_config

    highlight = load_config()["highlight"]
    assert highlight["min_duration"] <= 4
    assert highlight["target_duration"] <= 15
    assert highlight["max_duration"] <= 20
    assert highlight["merge_gap"] == 0


def test_seo_enforces_mobile_title_and_focused_search_term_caps():
    from automation.seo.seo import _enforce_limits

    result = _enforce_limits({
        "title": "Yuvraj Singh " + "coach debate " * 10,
        "description": "Yuvraj Singh cricket coach debate. " * 10,
        "hashtags": ["#Shorts", "#YuvrajSingh", "#Cricket"],
        "search_terms": [f"yuvraj cricket phrase {index}" for index in range(25)],
    })

    assert len(result["title"]) <= 70
    assert len(result["description"]) <= 4000
    assert len(result["search_terms"]) <= 15
