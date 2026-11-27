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


def test_banter_inside_a_cricket_stream_is_a_window_labelled_not_cricket():
    """Owner policy 2026-11-27: banter and live moments are valid Shorts; only the
    source must be cricket. The window's own cricket talk is a ranking label."""
    from automation.clip_selection.pipeline import _build_livestream_windows

    banter = [
        {"start": 0.0, "end": 6.0, "text": "Bhai tu chat mein roz yahi bolta hai", "score": 12.0},
        {"start": 6.0, "end": 13.0, "text": "aaj teri shaadi ka card bhi aa gaya kya", "score": 14.0},
    ]
    cricket = [
        {"start": 0.0, "end": 6.0, "text": "Bumrah ki yorker pe wicket gaya", "score": 12.0},
        {"start": 6.0, "end": 13.0, "text": "India ka match ab pakka hai bhai", "score": 14.0},
    ]

    [window] = _build_livestream_windows(banter, min_duration=10, max_duration=20)
    assert window["cricket"] is False and window["end"] == 13.0
    assert _build_livestream_windows(cricket, min_duration=10, max_duration=20)[0]["cricket"] is True


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




def test_diverse_livestream_windows_do_not_return_same_moment_variants():
    from automation.clip_selection.pipeline import _select_diverse_windows

    windows = [
        {"start": 0.0, "end": 20.0, "text": "a", "score": 10.0},
        {"start": 2.0, "end": 21.0, "text": "b", "score": 9.0},
        {"start": 40.0, "end": 55.0, "text": "c", "score": 8.0},
    ]

    selected = _select_diverse_windows(windows, max_candidates=3)

    assert selected == [windows[0], windows[2]]












