"""Whisper emits long punctuation-free blocks; selection needs thought-sized
units. Splits must happen on word-timestamp silence gaps so the complete-thought
gate stops producing multi-minute blobs.
"""

from automation.clip_selection.pipeline import _split_segments_on_pauses


def _seg(start, end, text, words=None):
    seg = {"start": start, "end": end, "text": text}
    if words is not None:
        seg["words"] = words
    return seg


def test_short_segment_passes_through_untouched():
    segs = [_seg(0.0, 4.0, "chhota thought hai yeh")]
    out = _split_segments_on_pauses(segs)
    assert len(out) == 1
    assert out[0]["text"] == "chhota thought hai yeh"


def test_split_happens_at_silence_gap():
    # 8 words, 2s pause after 4th word -> two segments at the gap
    words = []
    for i in range(8):
        base = 0.0 + i * 0.5 if i < 4 else 4.0 + (i - 4) * 0.5
        words.append({"start": base, "end": base + 0.4, "word": f"w{i}"})
    segs = [_seg(0.0, 6.0, " ".join(f"w{i}" for i in range(8)), words)]
    out = _split_segments_on_pauses(segs, pause_seconds=1.0)
    assert len(out) == 2
    assert out[0]["end"] <= 4.0 + 0.01
    assert out[1]["start"] >= 4.0 - 0.01
    assert out[0]["text"].split() == [f"w{i}" for i in range(4)]
    assert out[1]["text"].split() == [f"w{i}" for i in range(4, 8)]


def test_oversized_block_is_hard_split_even_without_gaps():
    # 60s continuous speech, no words -> proportional hard split
    text = " ".join(f"word{i}" for i in range(60))
    segs = [_seg(0.0, 60.0, text)]
    out = _split_segments_on_pauses(segs, max_block_seconds=20.0)
    assert len(out) >= 3
    assert all((s["end"] - s["start"]) <= 21.0 for s in out)


def test_missing_words_falls_back_to_proportional_split():
    text = " ".join(f"word{i}" for i in range(40))
    segs = [_seg(100.0, 140.0, text)]
    out = _split_segments_on_pauses(segs, max_block_seconds=20.0)
    assert len(out) == 2
    assert abs(out[0]["start"] - 100.0) < 0.01
    assert abs(out[-1]["end"] - 140.0) < 0.01


def test_word_text_spacing_is_normalized():
    words = [
        {"start": 0.0, "end": 0.5, "word": " naam"},
        {"start": 2.0, "end": 2.5, "word": "askaara"},
    ]
    segs = [_seg(0.0, 3.0, "naam askaara", words)]
    out = _split_segments_on_pauses(segs, pause_seconds=1.0)
    assert out[0]["text"] == "naam"
    assert out[1]["text"] == "askaara"
