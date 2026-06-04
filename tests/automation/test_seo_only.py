"""TDD tests for automation/seo_only.py — Mac-side SEO-only pipeline."""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestSEOOnlyDiscovery:

    def test_discovers_clips_from_directory(self, tmp_path):
        from automation.seo_only import discover_clips
        # Create fake exported clips
        (tmp_path / "clip1.mp4").write_bytes(b"\x00" * 1000)
        (tmp_path / "clip2.mp4").write_bytes(b"\x00" * 1000)
        (tmp_path / "clip1_metadata.json").write_text("{}")  # already has SEO
        (tmp_path / "not_a_clip.txt").write_text("ignore")

        clips = discover_clips(str(tmp_path), skip_existing=True)
        # clip1 already has metadata → skip; clip2 needs SEO
        assert len(clips) == 1
        assert clips[0]["clip_id"] == "clip2"

    def test_discovers_all_when_skip_false(self, tmp_path):
        from automation.seo_only import discover_clips
        (tmp_path / "clip1.mp4").write_bytes(b"\x00" * 1000)
        (tmp_path / "clip2.mp4").write_bytes(b"\x00" * 1000)
        (tmp_path / "clip1_metadata.json").write_text("{}")

        clips = discover_clips(str(tmp_path), skip_existing=False)
        assert len(clips) == 2

    def test_returns_empty_for_nonexistent_dir(self):
        from automation.seo_only import discover_clips
        clips = discover_clips("/nonexistent/path/12345")
        assert clips == []


class TestSEOOnlyTranscriptLoading:

    def test_loads_transcript_from_highlights_yaml(self, tmp_path):
        import yaml
        from automation.seo_only import _load_clip_transcript
        h_path = tmp_path / "highlights.yaml"
        h_path.write_text(yaml.dump({
            "clip1": {"text": "Kohli ne maara six!", "start": 0, "end": 15},
            "clip2": {"text": "Bumrah ki yorker!", "start": 20, "end": 35},
        }))
        text = _load_clip_transcript("clip1", str(tmp_path), str(h_path))
        assert "Kohli" in text

    def test_falls_back_to_transcript_json(self, tmp_path):
        from automation.seo_only import _load_clip_transcript
        tj = tmp_path / "video_transcript.json"
        tj.write_text(json.dumps({
            "segments": [
                {"start": 0, "end": 10, "text": "Welcome to match"},
                {"start": 10, "end": 20, "text": "Kohli batting"},
            ]
        }))
        text = _load_clip_transcript("clip1", str(tmp_path), transcript_json=str(tj))
        assert "Kohli" in text or "Welcome" in text

    def test_returns_empty_when_nothing_found(self, tmp_path):
        from automation.seo_only import _load_clip_transcript
        text = _load_clip_transcript("clip1", str(tmp_path))
        assert text == ""


class TestSEOOnlyRun:

    @patch("automation.seo_only._get_ai")
    def test_generates_seo_for_discovered_clips(self, mock_get_ai, tmp_path):
        from automation.seo_only import run_seo_only

        (tmp_path / "clip1.mp4").write_bytes(b"\x00" * 1000)
        (tmp_path / "clip2.mp4").write_bytes(b"\x00" * 1000)

        mock_ai = MagicMock()
        mock_ai.generate_fastest_first.return_value = json.dumps({
            "title": "Kohli ne maara CHHAKKA! 🔥 #Shorts",
            "description": "Virat Kohli smashes massive six over long-on, crowd goes crazy in Wankhede!",
            "hashtags": ["#Shorts", "#Kohli", "#RCBvsCSK"],
            "search_terms": ["kohli six wankhede", "RCB vs CSK highlights"],
        })
        mock_get_ai.return_value = mock_ai

        result = run_seo_only(str(tmp_path))
        assert result["processed"] == 2
        assert result["failed"] == 0
        # Metadata files should exist
        assert (tmp_path / "clip1_metadata.json").exists()
        assert (tmp_path / "clip2_metadata.json").exists()
