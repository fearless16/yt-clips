"""Tests for utils/ocr.py — keyframe extraction, cricket entity filtering, degraded mode."""

import numpy as np
import pytest

from utils.ocr import (
    extract_ocr_entities,
    is_ocr_available,
    _scene_change_score,
    _dedup_texts,
    _extract_cricket_entities,
)


class TestSceneChange:
    def test_identical_frames_score_zero(self):
        a = np.ones((100, 100, 3), dtype=np.uint8) * 128
        b = a.copy()
        assert _scene_change_score(a, b) < 0.05

    def test_different_frames_score_high(self):
        a = np.ones((100, 100, 3), dtype=np.uint8) * 0
        b = np.ones((100, 100, 3), dtype=np.uint8) * 255
        assert _scene_change_score(a, b) > 0.3


class TestDedup:
    def test_removes_duplicates(self):
        texts = [("Kohli 67", 0.9), ("kohli 67", 0.8), ("RCB 198", 0.7)]
        result = _dedup_texts(texts)
        assert len(result) == 2
        assert result[0][0] == "Kohli 67"

    def test_handles_empty(self):
        assert _dedup_texts([]) == []


class TestCricketEntities:
    def test_scoreboard_detected(self):
        texts = [("RCB 198/4", 0.9), ("Need 8 off 3 balls", 0.8),
                 ("Kohli 67*", 0.7), ("Some ad text", 0.5)]
        entities = _extract_cricket_entities(texts)
        assert "RCB 198/4" in entities["scoreboard"]
        assert "Need 8 off 3 balls" in entities["scoreboard"]
        assert "Some ad text" not in entities["scoreboard"]

    def test_player_names_detected(self):
        texts = [("Kohli", 0.9), ("Bumrah bowling yorker", 0.8)]
        entities = _extract_cricket_entities(texts)
        assert entities["player_names"]
        assert any("kohli" in p.lower() for p in entities["player_names"])

    def test_returns_dict_with_expected_keys(self):
        result = _extract_cricket_entities([("test", 0.5)])
        assert "scoreboard" in result
        assert "player_names" in result
        assert "on_screen_text" in result


class TestOcrDegraded:
    def test_returns_dict_with_expected_keys(self):
        """Even on nonexistent file, returns dict with expected keys."""
        result = extract_ocr_entities("nonexistent.mp4")
        assert isinstance(result, dict)
        assert "scoreboard" in result
        assert "player_names" in result
        assert "on_screen_text" in result
