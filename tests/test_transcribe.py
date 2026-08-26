import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import json


def test_decode_whisper_output_accepts_valid_utf8():
    from transcribe import decode_whisper_output

    payload = json.dumps(
        {"transcription": [{"offsets": {"from": 0, "to": 500}, "text": "नमस्कार"}]},
        ensure_ascii=False,
    ).encode("utf-8")
    data, warnings = decode_whisper_output(payload)
    assert warnings == []
    assert data["transcription"][0]["text"] == "नमस्कार"


def test_decode_whisper_output_survives_truncated_multibyte():
    """whisper.cpp's JSON writer can cut a multibyte char mid-sequence when a
    token crosses an internal boundary; strict decoding must not fall back to
    cp1252/latin-1 (which silently destroys Devanagari into mojibake)."""
    from transcribe import decode_whisper_output

    # Truncate ONE trailing byte of the last multibyte char ("ो") while the
    # surrounding JSON structure stays intact.
    doc_prefix = (
        '{"transcription": [{"offsets": {"from": 0, "to": 500}, "text": "'
    ).encode("ascii")
    broken = doc_prefix + "नमस्कार तो".encode("utf-8")[:-1] + b'"}]}'
    data, warnings = decode_whisper_output(broken)
    assert any("utf-8" in w.lower() for w in warnings)
    assert data["transcription"][0]["text"].startswith("नमस्कार")


def test_decode_whisper_output_never_falls_back_to_single_byte_codecs():
    from transcribe import decode_whisper_output

    raw = b'{"transcription": [{"text": "\xe0\xa4\x95\xe0\xa4"}]}'  # cut char
    try:
        data, warnings = decode_whisper_output(raw)
    except ValueError:
        return  # hard failure is acceptable; silent mojibake is not
    assert "\xfffd" not in json.dumps(data, ensure_ascii=False)
    assert any("utf-8" in w.lower() for w in warnings)

from transcribe import correct_segments_with_llm, transcribe
from automation.transcript import fetch

def test_correct_segments_with_llm_success():
    mock_segments = [
        {"start": 0.0, "end": 2.0, "text": "coaly hit a six"},
        {"start": 2.0, "end": 4.0, "text": "bumra bowled a fast delivery"}
    ]
    with patch("utils.ai_client.AIClient") as MockAIClient:
        instance = MockAIClient.return_value
        instance.groq_api_key = "test_key"
        instance.generate_text.return_value = "0: Kohli hit a six\n1: Bumrah bowled a fast delivery"
        
        corrected = correct_segments_with_llm(mock_segments)
        assert corrected[0]["text"] == "Kohli hit a six"
        assert corrected[1]["text"] == "Bumrah bowled a fast delivery"

def test_correct_segments_with_llm_failure_recovery():
    mock_segments = [
        {"start": 0.0, "end": 2.0, "text": "coaly hit a six"}
    ]
    with patch("utils.ai_client.AIClient") as MockAIClient:
        instance = MockAIClient.return_value
        instance.groq_api_key = "test_key"
        instance.generate_text.side_effect = Exception("API error")
        
        corrected = correct_segments_with_llm(mock_segments)
        assert corrected[0]["text"] == "coaly hit a six"

def test_cricket_terms_substitution(tmp_path):
    output_file = tmp_path / "transcript.json"
    
    mock_segment = MagicMock()
    mock_segment.start = 0.0
    mock_segment.end = 2.0
    mock_segment.text = "coaly and bumra and sky did well"
    mock_segment.words = []
    
    mock_info = MagicMock()
    mock_info.duration = 2.0
    mock_info.language = "en"
    
    with patch("faster_whisper.WhisperModel") as MockWhisperModel, \
         patch("transcribe.correct_segments_with_llm", side_effect=lambda x: x):
        
        instance = MockWhisperModel.return_value
        instance.transcribe.return_value = ([mock_segment], mock_info)
        
        transcribe("mock_video.mp4", str(output_file))
        
        assert output_file.exists()
        with open(output_file, "r") as f:
            data = json.load(f)
            
        # NOTE: "sky" is intentionally NOT uppercased anymore — it's an ordinary
        # English word (false-positive guard). Player-nickname disambiguation is
        # delegated to the context-aware LLM correction pass.
        assert data["segments"][0]["text"] == "Kohli and Bumrah and sky did well"
        assert data["source"] == "whisper"

def test_fetch_unified_path_fallback(tmp_path):
    video_path = tmp_path / "test_video.mp4"
    video_path.write_bytes(b"dummy video content")
    
    output_path = tmp_path / "fetched_transcript.json"

    from automation.transcript import TRANSCRIPT_CACHE
    TRANSCRIPT_CACHE.clear()

    with patch("automation.transcript._fetch_via_youtube_data_api", return_value=None), \
         patch("automation.transcript._fetch_via_api", return_value=None), \
         patch("automation.transcript._fetch_via_ytdlp", return_value=None), \
         patch("faster_whisper.WhisperModel") as MockWhisperModel, \
         patch("transcribe.correct_segments_with_llm", side_effect=lambda x: x):
             
        mock_segment = MagicMock()
        mock_segment.start = 1.0
        mock_segment.end = 3.0
        mock_segment.text = "local transcription text"
        mock_segment.words = []
        
        mock_info = MagicMock()
        mock_info.duration = 3.0
        mock_info.language = "en"
        
        instance = MockWhisperModel.return_value
        instance.transcribe.return_value = ([mock_segment], mock_info)
        
        url = "https://www.youtube.com/watch?v=12345678901"
        res = fetch(url, output_path=str(output_path), video_path=str(video_path))
        
        assert res["source"] == "local_whisper"
        assert res["segments"][0]["text"] == "local transcription text"
        assert output_path.exists()
