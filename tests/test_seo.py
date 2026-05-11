"""
test_seo.py — Premium TDD suite for seo.py

Covers:
  _extract_keywords, _enforce_limits, _parse_json_response,
  generate_clip_seo (happy path + retry + fallback),
  generate_seo_for_exported_clip,
  process_all_seo (batching/sequential runner).
"""
import json
import pytest
from pathlib import Path
from seo import (
    _extract_keywords,
    _enforce_limits,
    _parse_json_response,
    generate_clip_seo,
    generate_seo_for_exported_clip,
    process_all_seo,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

_GOOD_RESPONSE = {
    "clip_id": "t1",
    "title": "Kohli ne kya kar diya yaar 😤 | RCB vs CSK | IPL 2026",
    "description": "Kya shot tha yaar! 🔥 Kohli hit an absolute belter. IPL 2026.",
    "search_terms": ["kohli six 2026", "rcb highlights ipl", "kohli cover drive csk"],
    "hashtags": ["#RCBvsCSK", "#IPL2026", "#CricketShorts", "#Kohli"],
}

_TREND_RESPONSE = {
    "topics": ["IPL 2026", "Mumbai Indians"],
    "scorecard": "MI vs CSK",
    "live_stream_url": "https://www.youtube.com/watch?v=test123",
}

def _patch(monkeypatch, ai_payload=None, trend_payload=None):
    import seo, trends
    payload = ai_payload if ai_payload is not None else json.dumps(_GOOD_RESPONSE)
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

    def test_limit_respected(self):
        long_text = " ".join(f"uniqueword{i}" for i in range(50))
        assert len(_extract_keywords(long_text, limit=14)) <= 14

    def test_empty_string_returns_empty(self):
        assert _extract_keywords("") == []

    def test_frequency_ordering(self):
        kw = _extract_keywords("kohli kohli kohli dhoni dhoni bumrah")
        assert kw[0] == "kohli"
        assert kw[1] == "dhoni"

# ── _enforce_limits ──────────────────────────────────────────────────────────

class TestEnforceLimits:
    def test_title_truncated(self):
        item = {"title": "A" * 150}
        result = _enforce_limits(item)
        assert len(result["title"]) == 100

    def test_hashtags_prefixed_and_capped(self):
        item = {"hashtags": ["A", "#B", "C", "D", "E", "F"]}
        result = _enforce_limits(item)
        assert result["hashtags"] == ["#A", "#B", "#C", "#D", "#Shorts"]

    def test_search_terms_validation(self):
        item = {
            "hashtags": ["#Cricket"],
            "search_terms": [
                "kohli six",      # valid
                "shorts",         # too short/generic
                "cricket",        # generic
                "cricket",        # duplicate
                "A",              # 1 word
                "valid term two", # valid
            ]
        }
        result = _enforce_limits(item)
        assert "kohli six" in result["search_terms"]
        assert "valid term two" in result["search_terms"]
        assert "shorts" not in result["search_terms"]
        assert "A" not in result["search_terms"]

    def test_search_terms_length_limit(self):
        item = {"search_terms": ["long term number " + str(i) for i in range(50)]}
        result = _enforce_limits(item)
        total_len = sum(len(t) for t in result["search_terms"]) + (len(result["search_terms"]) - 1) * 2
        assert total_len <= 500

# ── generate_clip_seo ─────────────────────────────────────────────────────────

class TestGenerateClipSEO:
    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        _patch(monkeypatch)
        # Fast retries for testing
        import seo
        monkeypatch.setattr(seo, "time", type('obj', (object,), {'sleep': lambda x: None, 'info': lambda *a: None}) )
        # Actually just monkeypatch time.sleep globally for the module
        monkeypatch.setattr("time.sleep", lambda x: None)

    def test_happy_path(self):
        result = generate_clip_seo("t1", "kohli hits a six")
        assert result["clip_id"] == "t1"
        assert result["title"] == _GOOD_RESPONSE["title"]
        assert len(result["hashtags"]) >= 3

    def test_fallback_on_ai_failure(self, monkeypatch):
        import seo
        def fail(*a, **kw): raise ValueError("429 Too Many Requests")
        monkeypatch.setattr(seo.ai, "generate_text", fail)
        
        result = generate_clip_seo("t1", "kohli hits a six")
        assert result["clip_id"] == "t1"
        assert "Cricket Live Highlights" in result["title"]
        assert len(result["hashtags"]) == 3

    def test_json_parsing_resilience(self, monkeypatch):
        import seo
        fenced = f"Some preamble ```json\n{json.dumps(_GOOD_RESPONSE)}\n``` footer"
        monkeypatch.setattr(seo.ai, "generate_text", lambda *a, **kw: fenced)
        
        result = generate_clip_seo("t1", "kohli hits a six")
        assert result["title"] == _GOOD_RESPONSE["title"]

# ── generate_seo_for_exported_clip ──────────────────────────────────────────

def test_generate_seo_for_exported_clip(tmp_path, monkeypatch):
    _patch(monkeypatch)
    monkeypatch.setattr("time.sleep", lambda x: None)
    
    out_dir = tmp_path / "shorts"
    result = generate_seo_for_exported_clip("t1", "transcript", str(out_dir))
    
    assert result["clip_id"] == "t1"
    expected_file = out_dir / "t1_metadata.json"
    assert expected_file.exists()
    with open(expected_file, "r") as f:
        saved = json.load(f)
        assert saved["title"] == _GOOD_RESPONSE["title"]

# ── process_all_seo ───────────────────────────────────────────────────────────

def test_process_all_seo(tmp_path, monkeypatch):
    import yaml
    _patch(monkeypatch)
    monkeypatch.setattr("time.sleep", lambda x: None)
    
    h_path = tmp_path / "highlights.yaml"
    out_dir = tmp_path / "shorts"
    
    highlights_data = {
        "clip1": {"text": "kohli"},
        "clip2": {"text": "dhoni"}
    }
    with open(h_path, "w") as f:
        yaml.dump(highlights_data, f)
        
    res_path = process_all_seo(str(h_path), str(out_dir))
    
    assert Path(res_path).exists()
    assert (out_dir / "clip1_metadata.json").exists()
    assert (out_dir / "clip2_metadata.json").exists()
    
    with open(res_path, "r") as f:
        results = json.load(f)
        assert len(results) == 2
        assert results[0]["clip_id"] == "clip1"


def test_upload_metadata_adds_shorts_marker():
    from upload import _ensure_shorts_metadata

    title, description, tags = _ensure_shorts_metadata(
        "RCB vs CSK Live | Big wicket",
        "Kohli wicket moment in IPL 2026.",
        ["kohli wicket", "rcb csk highlights"],
    )

    assert title == "RCB vs CSK Live | Big wicket"
    assert "#Shorts" in description
    assert "shorts" in tags


def test_upload_guard_rejects_landscape(monkeypatch):
    import upload
    from upload import _validate_shorts_video

    monkeypatch.setattr(upload, "_probe_video", lambda path: {
        "width": 1920,
        "height": 1080,
        "duration": 29.0,
    })

    assert _validate_shorts_video(Path("clip.mp4")) is False


def test_upload_guard_accepts_vertical(monkeypatch):
    import upload
    from upload import _validate_shorts_video

    monkeypatch.setattr(upload, "_probe_video", lambda path: {
        "width": 1080,
        "height": 1920,
        "duration": 29.0,
    })

    assert _validate_shorts_video(Path("clip.mp4")) is True
