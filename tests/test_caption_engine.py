import pytest
from pathlib import Path
from automation.caption_engine import (
    highlight_keywords_html,
    chunk_sentence_into_cues,
    process_transcript_segments,
    prepare_clip_captions,
    CaptionCue,
    _HINGLISH_CACHE,
)

def test_highlight_keywords_html():
    text = "Namashkar dosto main aapka bhai Prajjwal Cricket Wali Baat leke hazir hoon"
    html = highlight_keywords_html(text)
    assert '<span class="holo-gold">Prajjwal</span>' in html
    assert '<span class="holo-pill">Cricket Wali Baat</span>' in html
    assert '<span class="holo-pill">hazir hoon</span>' in html

def test_highlight_keywords_escapes_html():
    text = "Rohit & Kohli <score> 100 runs"
    html = highlight_keywords_html(text)
    assert "&amp;" in html
    assert "&lt;" in html
    assert "&gt;" in html
    assert '<span class="holo-pill">score</span>' in html
    assert '<span class="holo-gold">Rohit</span>' in html
    assert '<span class="holo-gold">Kohli</span>' in html

def test_chunk_sentence_into_cues_short():
    text = "Namashkar dosto!"
    cues = chunk_sentence_into_cues(text, 0.0, 1.5, max_words=4)
    assert len(cues) == 1
    assert cues[0].text == "Namashkar dosto!"
    assert cues[0].start == 0.0
    assert cues[0].end == 1.5

def test_chunk_sentence_into_cues_long():
    text = "Namashkar dosto main aapka bhai Prajjwal hazir hoon Cricket Wali Baat ka ek aur taza tarin episode leke"
    cues = chunk_sentence_into_cues(text, 0.0, 10.0, max_words=4)
    assert len(cues) > 1
    for c in cues:
        assert len(c.words) <= 4
        assert c.start < c.end
    assert cues[0].start == 0.0
    assert cues[-1].end == 10.0

def test_chunk_sentence_into_cues_no_drift():
    # Verify sequential partition with zero gaps
    text = "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9 Word10"
    cues = chunk_sentence_into_cues(text, 2.5, 7.5, max_words=3)
    assert len(cues) >= 3
    for i in range(len(cues) - 1):
        assert abs(cues[i].end - cues[i+1].start) <= 0.05
    assert cues[0].start == 2.5
    assert cues[-1].end == 7.5

def test_process_transcript_segments_clip_duration_clamped():
    segments = [
        {"start": 0.0, "end": 2.0, "text": "Segment 1"},
        {"start": 2.0, "end": 5.0, "text": "Segment 2"},
        {"start": 5.0, "end": 8.0, "text": "Segment 3 past clip end"},
    ]
    # Pass clip_duration=4.0
    cues = process_transcript_segments(segments, clip_duration=4.0)
    # No cues should start at or after 4.0
    for c in cues:
        assert c.start < 4.0
        assert c.end <= 4.0

def test_hinglish_cache():
    _HINGLISH_CACHE["परीक्षण पाठ"] = "test text"
    from automation.caption_engine import convert_hindi_to_hinglish_llm
    res = convert_hindi_to_hinglish_llm(["परीक्षण पाठ"])
    assert res == ["test text"]

def test_llm_failure_falls_back_to_rule_based():
    from automation.caption_engine import convert_hindi_to_hinglish_llm
    # Given an endpoint that fails immediately (connection refused or port 1)
    res = convert_hindi_to_hinglish_llm(["नमस्कार दोस्तों"], api_url="http://127.0.0.1:1/v1/chat/completions", timeout=1)
    assert res is not None
    assert len(res) == 1
    # Fallback should transliterate Devanagari to Roman script
    assert "namaskar" in res[0].lower()

def test_normalize_rolling_transcript_segments():
    from automation.caption_engine import normalize_rolling_transcript_segments
    # Simulating YouTube rolling subtitles with overlapping 2-line display
    raw_rolling = [
        {"start": 55.0, "end": 58.2, "text": "Line 1"},
        {"start": 56.8, "end": 60.0, "text": "Line 2"},
        {"start": 58.2, "end": 61.6, "text": "Line 3"},
    ]
    normalized = normalize_rolling_transcript_segments(raw_rolling)
    # Line 1 speech end should be clamped to near Line 2 start
    assert normalized[0]["start"] == 55.0
    assert normalized[0]["end"] <= 57.0
    # Line 2 speech end should be clamped to near Line 3 start
    assert normalized[1]["start"] == 56.8
    assert normalized[1]["end"] <= 58.4
    assert normalized[2]["start"] == 58.2
