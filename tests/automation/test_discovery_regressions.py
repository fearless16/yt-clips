"""Caption-based discovery regressions, not audio or release certification."""

import pytest

from automation.clip_selection import pipeline
from automation.seo.cricket_context import correct_cricket_spelling, is_cricket_content
from utils.devanagari import to_roman


@pytest.mark.parametrize("text", [
    "Hardik Pandya could be traded to Punjab Kings for Arshdeep Singh. Mumbai should build new players instead.",
    "हार्दिक पांड्या कुड बी ट्रेडेड टू पंजाब किंग्स विद अदीप सिंह एंड भाई मत लाओ अदीप को यार।",
    "टेस्ट मैच में क्या होगा वो मैं सोच रहा हूं। टेस्ट मैच में तो हमें नींद का सैक्रिफाइस करना पड़ेगा कि यार सोएंगे ही ना क्योंकि सुबह 3:30 बजे तो हम लोग उठ नहीं पाएंगे यार नामुमकिन डे पॉसिबल हो जाएगा हम लोग के लिए।",
    "भाई साहब हर्षित राणा और गौतम गंभीर के बीच में एक गाना चलता है यार हर्षित राणा हर जगह क्यों घुसेड़ देते हो तो गौतम गंभीर ने गाना गाया था दिल पे जख्म खाते हैं जान से गुजरते हैं जुर्म सिर्फ इतना हम उनको प्यार करते हैं।",
])
def test_cricket_discussion_and_comedy_reach_discovery(text):
    roman = to_roman(text)
    assert is_cricket_content(roman)
    assert pipeline._has_dense_cricket_evidence(roman)
    windows = pipeline._build_livestream_windows([
        {"start": 0, "end": 18, "text": roman}], 10, 38)
    assert windows
    assert windows[0]["complete_thought"] is None


@pytest.mark.parametrize("text", [
    "Mumbai and Punjab are nice places to travel and eat food.",
    "Trade this phone for a new laptop and build a team for our business.",
    "भाई यह किलर एनिमेशन है ऐसा इंटरनेट पर कहीं नहीं मिलेगा।",
    "I went shopping and watched a movie " * 20 + "Hardik Pandya",
    "I went shopping and watched a movie " * 20 + "Hardik Pandya Punjab Kings",
])
def test_unrelated_chatter_is_not_rescued_by_discovery(text):
    assert pipeline._build_livestream_windows([
        {"start": 0, "end": 18, "text": to_roman(text)}], 10, 38) == []


def test_phrase_aliases_are_script_consistent_and_do_not_guess_caption_names():
    text = "हार्दिक पांड्या पंजाब किंग्स गौतम गंभीर हर्षित राणा टेस्ट मैच अदीप सिंह"
    canonical = correct_cricket_spelling(text)
    assert canonical == correct_cricket_spelling(canonical)
    assert to_roman(canonical) == correct_cricket_spelling(to_roman(text))
    assert "Arshdeep" not in canonical  # Source says Adeep; listening review owns this.
    assert "Harshit Rana" in canonical
    assert "Gautam Gambhir" in canonical
    assert correct_cricket_spelling("Harshit Rana and Gautam Gambhir") == "Harshit Rana and Gautam Gambhir"


def test_long_caption_group_keeps_source_cue_entry_points():
    cues = [
        {"start": i * 3, "end": (i + 1) * 3,
         "text": "wrestling storyline keeps going" if i < 12 else text,
         "source_cue_ids": [i]}
        for i, text in enumerate([""] * 12 + [
            "हार्दिक पांड्या सब टीम में", "एक महीना खेलता है फिर वापस मुंबई आ",
            "जाता है।", "हार्दिक पांड्या कुड बी ट्रेडेड टू पंजाब किंग्स।"])
    ]
    for cue in cues:
        cue["text"] = to_roman(cue["text"])
    prepared = pipeline._prepare_complete_thoughts(pipeline._split_segments_on_pauses(cues))
    assert any(part["start"] == 36 and part["source_cue_ids"] == [12] for part in prepared)
    assert all(part["end"] - part["start"] <= 20 for part in prepared)
    windows = pipeline._build_livestream_windows(prepared, 10, 38)
    assert any(window["start"] == 36 for window in windows)
    assert all(window["complete_thought"] is None for window in windows)


def test_real_overlapping_caption_cues_keep_gambhir_setup_and_song_discoverable():
    # Saved hi auto-captions from pvA10GoCS9U; no guessed words or audio claims.
    source = [
        (1802.799, 1807.6, "भाई साहब हर्षित राणा और प्रसित कृष्णा"),
        (1805.36, 1810.48, "भाई हर्षित राणा और गौतम गंभीर के बीच में"),
        (1807.6, 1813.6, "एक गाना चलता है यार हर्षित राणा ने नहीं"),
        (1810.48, 1815.84, "प्रसित सॉरी गौतम गंभीर से पूछा गया कि आप"),
        (1813.6, 1818.48, "हर्षित राणा हर्षित राणा हर जगह हर्षित"),
        (1815.84, 1824.159, "राणा क्यों घुसेड़ देते हो तो गौतम गंभीर"),
        (1818.48, 1827.919, "ने गाना गाया था दिल पे जख्म खाते हैं जान"),
        (1824.159, 1831.6, "से गुजरते हैं"),
        (1827.919, 1836.559, "जुर्म सिर्फ इतना"),
        (1831.6, 1838.24, "हम उनको प्यार करते हैं। तो यह बात है भाई"),
    ]
    cues = [{"start": start, "end": end, "text": to_roman(text), "source_cue_ids": [i]}
            for i, (start, end, text) in enumerate(source)]
    thoughts = pipeline._prepare_complete_thoughts(pipeline._split_segments_on_pauses(cues))
    windows = pipeline._build_livestream_windows(thoughts, 10, 38)
    assert any("Gautam Gambhir" in window["text"] and "pyaara karate" in window["text"]
               and window["start"] == source[0][0] for window in windows)
    assert all(window["boundary_method"] == "estimated_cue_fraction" for window in windows)
    assert all(window["complete_thought"] is None for window in windows)
