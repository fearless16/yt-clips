"""Shorts must be under 25 seconds — research-backed (vidIQ + competitor audit + 12-channel scan).

TDD: tests written first. Verifies:
1. Config hard caps highlight windows below 25s.
2. Target duration drives compression to a sub-25s result.
3. Speed factor math holds for a 24s window compressed to a 20s target.
"""
import pytest

from utils.config import load_config


def test_config_max_duration_below_25():
    hl = load_config()["highlight"]
    assert float(hl["max_duration"]) <= 24


def test_config_preferred_duration_max_below_25():
    hl = load_config()["highlight"]
    assert float(hl["preferred_duration_max"]) <= 24


def test_config_target_duration_below_25():
    hl = load_config()["highlight"]
    assert float(hl["target_duration"]) < 25


def test_speed_factor_for_24s_window_to_20s_target():
    from highlight import _compute_speed_factor
    assert _compute_speed_factor(24.0, 20.0, 1.6) == pytest.approx(1.2, abs=0.01)


def test_clip_selection_speed_factor_24s():
    from automation.clip_selection.pipeline import _compute_speed_factor
    assert _compute_speed_factor(24.0, 20.0, 1.6) == pytest.approx(1.2, abs=0.01)