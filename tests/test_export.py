import pytest
from unittest.mock import patch, MagicMock

from export import _build_enhance_stack, _sanitize_strategy, _normalize_speed, _sanitize_lighting_filter

def test_normalize_speed():
    assert _normalize_speed(1.5) == 1.5
    assert _normalize_speed(5.0) == 4.0   # Cap at 4.0
    assert _normalize_speed(0.1) == 0.25  # Floor at 0.25
    assert _normalize_speed("invalid") == 1.0
    assert _normalize_speed(None) == 1.0

def test_sanitize_lighting_filter():
    assert _sanitize_lighting_filter("eq=contrast=1.2") == "eq=contrast=1.2"
    assert _sanitize_lighting_filter("curves=vintage") == "curves=vintage"
    assert _sanitize_lighting_filter("invalid_filter=1.0") == "" # Not in safe filters
    assert _sanitize_lighting_filter("eq=;rm -rf /") == "" # Malicious character check
    assert _sanitize_lighting_filter("eq=[box]") == "" # Malicious bracket check

def test_sanitize_strategy():
    raw = {
        "use_solo_frame": True,
        "speed_factor": "1.5",
        "active_crop": {"x": 100, "y": "200", "width": 300, "height": 400},
        "lighting_filter": "eq=contrast=1.1"
    }
    sanitized = _sanitize_strategy(raw)
    assert sanitized["use_solo_frame"] is True
    assert sanitized["speed_factor"] == 1.5
    assert sanitized["active_crop"] == {"x": 100, "y": 200, "width": 300, "height": 400}
    assert sanitized["lighting_filter"] == "eq=contrast=1.1"

def test_build_enhance_stack_guest_cam():
    analysis = {
        "export_strategy": {
            "guest_cam_on": True
        }
    }
    # Test guest cam stack structure (contains split=2 and vstack)
    filter_chain = _build_enhance_stack(analysis)
    assert "split=2" in filter_chain
    assert "vstack=inputs=2" in filter_chain

def test_build_enhance_stack_screen_share():
    analysis = {
        "export_strategy": {
            "is_screen_share": True
        }
    }
    # Test screen share structure
    filter_chain = _build_enhance_stack(analysis)
    assert "scale=" in filter_chain
    assert "force_original_aspect_ratio=increase" in filter_chain

def test_build_enhance_stack_solo_crop():
    analysis = {
        "export_strategy": {
            "use_solo_frame": True,
            "active_crop": {"x": 150, "y": 250, "width": 320, "height": 480}
        }
    }
    filter_chain = _build_enhance_stack(analysis)
    # Checks that active crop coordinates are injected
    assert "crop=320:480:150:250" in filter_chain

def test_build_enhance_stack_guest_cam_off():
    analysis = {
        "export_strategy": {
            "guest_cam_off": True,
            "black_panel_side": "right"
        }
    }
    filter_chain = _build_enhance_stack(analysis)
    assert "crop=iw/2:ih:0:0" in filter_chain
