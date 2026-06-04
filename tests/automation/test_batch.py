"""TDD tests for automation/batch.py — multi-URL batch pipeline runner.

Tests cover:
- URL validation and dedup
- Fail-forward: one bad URL doesn't kill the batch
- Checkpoint/resume: intermediate state persisted
- Stage isolation: download-all → transcribe-all → highlight-all → export-top-N
- Configurable top-N clip selection across all videos
"""
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


class TestBatchURLValidation:

    def test_dedup_urls(self):
        from automation.batch import _dedup_urls
        urls = [
            "https://youtu.be/abc123def45",
            "https://www.youtube.com/watch?v=abc123def45",
            "https://youtu.be/xyz789ghijk",
            "https://youtu.be/abc123def45",  # exact dup
        ]
        result = _dedup_urls(urls)
        assert len(result) == 2

    def test_rejects_non_youtube_urls(self):
        from automation.batch import _dedup_urls
        urls = [
            "https://youtu.be/abc123def45",
            "https://example.com/notyt",
            "",
            "not-a-url",
        ]
        result = _dedup_urls(urls)
        assert len(result) == 1
        assert "abc123def45" in result[0]

    def test_empty_list(self):
        from automation.batch import _dedup_urls
        assert _dedup_urls([]) == []


class TestBatchCheckpoint:

    def test_checkpoint_written_after_stage(self, tmp_path):
        from automation.batch import BatchCheckpoint
        cp = BatchCheckpoint(tmp_path / "checkpoint.json")
        cp.mark_done("download", "https://youtu.be/abc123def45")
        cp.save()

        # Reload from disk
        cp2 = BatchCheckpoint(tmp_path / "checkpoint.json")
        cp2.load()
        assert cp2.is_done("download", "https://youtu.be/abc123def45")
        assert not cp2.is_done("transcribe", "https://youtu.be/abc123def45")

    def test_checkpoint_survives_crash(self, tmp_path):
        from automation.batch import BatchCheckpoint
        cp = BatchCheckpoint(tmp_path / "checkpoint.json")
        cp.mark_done("download", "url1")
        cp.mark_done("download", "url2")
        cp.mark_done("transcribe", "url1")
        cp.save()

        cp2 = BatchCheckpoint(tmp_path / "checkpoint.json")
        cp2.load()
        assert cp2.is_done("download", "url1")
        assert cp2.is_done("download", "url2")
        assert cp2.is_done("transcribe", "url1")
        assert not cp2.is_done("transcribe", "url2")

    def test_fresh_checkpoint_has_no_completions(self, tmp_path):
        from automation.batch import BatchCheckpoint
        cp = BatchCheckpoint(tmp_path / "checkpoint.json")
        assert not cp.is_done("download", "anything")


class TestBatchTopNSelection:

    def test_selects_top_n_across_videos(self):
        from automation.batch import _select_top_clips
        highlights = {
            "video1": [
                {"id": "v1c1", "weighted_score": 9.0},
                {"id": "v1c2", "weighted_score": 3.0},
            ],
            "video2": [
                {"id": "v2c1", "weighted_score": 7.0},
                {"id": "v2c2", "weighted_score": 8.0},
            ],
        }
        top = _select_top_clips(highlights, top_n=3)
        assert len(top) == 3
        ids = [c["id"] for c in top]
        assert ids[0] == "v1c1"  # score 9
        assert "v2c2" in ids     # score 8
        assert "v2c1" in ids     # score 7
        assert "v1c2" not in ids # score 3, cut off

    def test_top_n_larger_than_total(self):
        from automation.batch import _select_top_clips
        highlights = {
            "video1": [{"id": "c1", "weighted_score": 5.0}],
        }
        top = _select_top_clips(highlights, top_n=100)
        assert len(top) == 1

    def test_empty_highlights(self):
        from automation.batch import _select_top_clips
        assert _select_top_clips({}, top_n=10) == []


class TestBatchRunFailForward:

    @patch("automation.batch._download_one")
    def test_one_download_fails_others_continue(self, mock_dl):
        from automation.batch import _download_all
        mock_dl.side_effect = [
            "/path/to/video1.mp4",
            RuntimeError("403 blocked"),
            "/path/to/video3.mp4",
        ]
        urls = ["url1", "url2", "url3"]
        results, failures = _download_all(urls, "input/")
        assert len(results) == 2
        assert len(failures) == 1
        assert "url2" in failures[0]
