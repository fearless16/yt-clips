"""TDD tests for automation/seo_only.py — Mac-side SEO-only pipeline."""
import json
import os
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

    def test_discovers_uppercase_mp4_extensions(self, tmp_path):
        from automation.seo_only import discover_clips
        (tmp_path / "clip1.MP4").write_bytes(b"\x00" * 1000)
        (tmp_path / "clip2.Mp4").write_bytes(b"\x00" * 1000)
        (tmp_path / "clip3.mp4").write_bytes(b"\x00" * 1000)
        clips = discover_clips(str(tmp_path), skip_existing=True)
        ids = sorted(c["clip_id"] for c in clips)
        assert ids == ["clip1", "clip2", "clip3"]

    def test_corrupt_metadata_is_not_permanently_skipped(self, tmp_path):
        """A truncated/garbage metadata.json must NOT be treated as 'done'."""
        from automation.seo_only import discover_clips
        (tmp_path / "clip1.mp4").write_bytes(b"\x00" * 1000)
        (tmp_path / "clip1_metadata.json").write_text("{ truncated json")
        clips = discover_clips(str(tmp_path), skip_existing=True)
        assert len(clips) == 1
        assert clips[0]["clip_id"] == "clip1"


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

    def test_loads_utf8_transcript_with_emoji(self, tmp_path):
        from automation.seo_only import _load_clip_transcript
        tj = tmp_path / "video_transcript.json"
        tj.write_text(json.dumps({
            "segments": [
                {"start": 0, "end": 10, "text": "🔴 Kohli ne maara six!"},
            ]
        }, ensure_ascii=False), encoding="utf-8")
        text = _load_clip_transcript("clip1", str(tmp_path), transcript_json=str(tj))
        assert "Kohli" in text

    def test_handles_non_dict_segments_safely(self, tmp_path):
        from automation.seo_only import _load_clip_transcript
        tj = tmp_path / "video_transcript.json"
        tj.write_text(json.dumps({
            "segments": [
                {"start": 0, "end": 10, "text": "Kohli batting"},
                "bare string segment",
                {"start": 20, "end": 30},  # missing text key
            ]
        }))
        # Should not raise; returns joined text without crashing
        text = _load_clip_transcript("clip1", str(tmp_path), transcript_json=str(tj))
        assert "Kohli" in text


class TestSEOOnlyRun:

    @patch("automation.seo_only._generate_seo_for_clip")
    def test_generates_seo_for_discovered_clips(self, mock_generate, tmp_path):
        from automation.seo_only import run_seo_only

        (tmp_path / "clip1.mp4").write_bytes(b"\x00" * 1000)
        (tmp_path / "clip2.mp4").write_bytes(b"\x00" * 1000)

        mock_generate.return_value = {
            "title": "Kohli ka Batting Debate",
            "description": "Grounded long cricket description. " * 50,
            "hashtags": ["#Shorts", "#ViratKohli"],
            "search_terms": [f"virat kohli batting analysis {i}" for i in range(8)],
            "tags": ["Virat Kohli", "cricket analysis"],
        }

        result = run_seo_only(str(tmp_path))
        assert result["processed"] == 2
        assert result["failed"] == 0
        # Metadata files should exist
        assert (tmp_path / "clip1_metadata.json").exists()
        assert (tmp_path / "clip2_metadata.json").exists()
        assert mock_generate.call_count == 2
