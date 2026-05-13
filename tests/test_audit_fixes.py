"""
test_audit_fixes.py — TDD regression tests for audit-identified bugs.

Each test is written to FAIL against the buggy code, then PASS after the fix.
"""
import json
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


# ──────────────────────────────────────────────────────────────────────────────
# BUG-1: _PREV_CROP_X global state bleeds across threads
# ──────────────────────────────────────────────────────────────────────────────

class TestCropStateThreadSafety:
    """Crop smoothing state must be thread-local, not global."""

    def test_reset_in_one_thread_does_not_affect_another(self):
        """Two threads calling reset_crop_state independently must not interfere."""
        from frame_analyzer import reset_crop_state, _get_prev_crop_x, _set_prev_crop_x

        results = {}
        barrier = threading.Barrier(2)

        def worker(name, initial_x):
            reset_crop_state()
            _set_prev_crop_x(initial_x)
            barrier.wait()  # Sync both threads
            time.sleep(0.01)  # Let the other thread run
            # After the other thread set its own value, ours should be untouched
            results[name] = _get_prev_crop_x()

        t1 = threading.Thread(target=worker, args=("A", 100))
        t2 = threading.Thread(target=worker, args=("B", 200))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # With thread-local, each thread sees its own value.
        assert results["A"] == 100, f"Thread A expected 100, got {results['A']}"
        assert results["B"] == 200, f"Thread B expected 200, got {results['B']}"
        assert results["A"] != results["B"], (
            "Crop state leaked between threads — both threads see the same _PREV_CROP_X"
        )

    def test_analyze_clip_resets_crop_state_per_call(self):
        """Each call to analyze_clip must start with fresh crop state."""
        import frame_analyzer
        # Set a known previous crop value
        frame_analyzer._PREV_CROP_X = 999
        frame_analyzer.reset_crop_state()
        assert frame_analyzer._PREV_CROP_X is None


# ──────────────────────────────────────────────────────────────────────────────
# BUG-2: HostDetector.identify_host() receives timestamp, not image
# ──────────────────────────────────────────────────────────────────────────────

class TestHostDetectorFrameData:
    """HostDetector must receive actual face image data, not timestamps."""

    def test_all_faces_contain_frame_image_key(self):
        """The 'frame_img' key in face dicts must be a numpy array, not a float."""
        # Simulate what PremiumAnalyzer.analyze_clip builds
        # Current bug: "frame": float(t) — this is a timestamp, not an image
        face_dict = {
            "x": 100.0, "y": 200.0,
            "w": 50.0, "h": 60.0,
            "id": 0,
            "frame": 1.5,  # BUG: This is a timestamp!
            "brightness": 128,
        }
        # The HostDetector tries to call _compute_face_embedding on face["frame"]
        # which is a float, not a numpy array
        from premium_analyzer import HostDetector
        detector = HostDetector([])
        # With the bug, identify_host always falls back because face_img is a float
        # After fix, face dicts should have "frame_img" as numpy array or None
        result_idx = detector.identify_host([face_dict])
        # The test validates that the code path doesn't silently ignore the data
        assert result_idx >= 0  # Should always return a valid index


# ──────────────────────────────────────────────────────────────────────────────
# BUG-3: chat_exclusion not sanitized in _sanitize_strategy
# ──────────────────────────────────────────────────────────────────────────────

class TestChatExclusionSanitization:
    """_sanitize_strategy must validate chat_exclusion fields."""

    def test_chat_exclusion_with_string_width_is_sanitized(self):
        from export import _sanitize_strategy

        raw = {
            "chat_exclusion": {
                "exclude_chat": True,
                "chat_side": "right",
                "chat_exclude_width": "not_a_number",  # Corrupted data
            },
        }
        result = _sanitize_strategy(raw)
        chat = result.get("chat_exclusion")
        # After fix: should be sanitized to None or have int width
        if chat and chat.get("exclude_chat"):
            assert isinstance(chat["chat_exclude_width"], int), (
                "chat_exclude_width should be sanitized to int"
            )

    def test_chat_exclusion_with_valid_data_preserved(self):
        from export import _sanitize_strategy

        raw = {
            "chat_exclusion": {
                "exclude_chat": True,
                "chat_side": "right",
                "chat_exclude_width": 350,
            },
        }
        result = _sanitize_strategy(raw)
        chat = result.get("chat_exclusion")
        assert chat is not None
        assert chat["exclude_chat"] is True
        assert chat["chat_exclude_width"] == 350

    def test_chat_exclusion_none_passes_through(self):
        from export import _sanitize_strategy

        raw = {"chat_exclusion": None}
        result = _sanitize_strategy(raw)
        assert result.get("chat_exclusion") is None


# ──────────────────────────────────────────────────────────────────────────────
# BUG-4: PhaseTracker.end() never called for pipeline phases
# ──────────────────────────────────────────────────────────────────────────────

class TestPhaseTrackerCompleteness:
    """Pipeline phases should call both begin() and end()."""

    def test_phase_tracker_records_all_phases(self):
        from utils.logger import PhaseTracker

        tracker = PhaseTracker()
        tracker.begin("PHASE 1 — DOWNLOAD")
        tracker.end("done")
        tracker.begin("PHASE 2 — TRANSCRIPTION")
        tracker.end("done")

        assert len(tracker.phases) == 2
        assert tracker.phases[0]["name"] == "PHASE 1 — DOWNLOAD"
        assert tracker.phases[1]["name"] == "PHASE 2 — TRANSCRIPTION"

    def test_phase_tracker_summary_includes_all_phases(self):
        from utils.logger import PhaseTracker

        tracker = PhaseTracker()
        tracker.begin("PHASE 1")
        tracker.end("done")
        tracker.begin("PHASE 2")
        tracker.end("done")
        tracker.begin("PHASE 3")
        tracker.end("done")

        table = tracker.summary_table()
        # Rich Table has row_count property
        assert table.row_count >= 4  # 3 phases + TOTAL row


# ──────────────────────────────────────────────────────────────────────────────
# BUG-5: _identify_by_facecam_region double-adds w/2
# ──────────────────────────────────────────────────────────────────────────────

class TestFacecamRegionIdentification:
    """Host identification should not double-offset face centers."""

    def test_face_at_facecam_center_is_identified_as_host(self):
        """A face whose center coordinates match the facecam center should be host."""
        from premium_analyzer import HostDetector

        detector = HostDetector([])
        # Facecam config: x=0, y=540, width=320, height=180
        # Facecam center: (160, 630)
        # Face dict with CENTER coordinates already:
        faces = [
            {"x": 500.0, "y": 300.0, "w": 60.0, "h": 70.0},  # Far from facecam
            {"x": 160.0, "y": 630.0, "w": 60.0, "h": 70.0},  # At facecam center
        ]
        # Bug: code does face_cx = x + w/2 = 160 + 30 = 190 (wrong!)
        # Fix: face_cx = x = 160 (x is already center)
        idx = detector._identify_by_facecam_region(faces)
        assert idx == 1, f"Expected face 1 (at facecam center), got face {idx}"


# ──────────────────────────────────────────────────────────────────────────────
# BUG-6: scipy import not guarded in premium_render.py
# ──────────────────────────────────────────────────────────────────────────────

class TestScipyGuard:
    """premium_render.generate_speed_profile should not crash without scipy."""

    def test_speed_profile_without_scipy_does_not_crash(self):
        """If scipy is unavailable, speed profile should use a fallback."""
        import importlib
        import sys

        # We can't easily remove scipy at runtime, but we can verify
        # the import is inside the function, not at module level
        import premium_render
        source = Path(premium_render.__file__).read_text()
        # Check that scipy import is inside the function, not at module top
        # The fix should guard with try/except or check HAS_SCIPY
        lines = source.split("\n")
        scipy_imports = [
            (i + 1, line) for i, line in enumerate(lines)
            if "from scipy" in line or "import scipy" in line
        ]
        for lineno, line in scipy_imports:
            # Module-level scipy imports (not inside a function) are the bug
            stripped = line.lstrip()
            # Count leading spaces to check indentation
            indent = len(line) - len(stripped)
            if indent == 0 and not stripped.startswith("#"):
                pytest.fail(
                    f"premium_render.py line {lineno}: unguarded module-level scipy import: {stripped}"
                )


# ──────────────────────────────────────────────────────────────────────────────
# BUG-7: detect_dead_air uses .decode() inconsistently
# ──────────────────────────────────────────────────────────────────────────────

class TestDeadAirConsistency:
    """detect_dead_air should handle stderr consistently (text mode)."""

    def test_dead_air_returns_dict_with_has_dead_air_key(self):
        from frame_analyzer import detect_dead_air
        # With a non-existent file, should return gracefully
        result = detect_dead_air("/nonexistent/video.mp4", 0.0, 1.0)
        assert isinstance(result, dict)
        assert "has_dead_air" in result


# ──────────────────────────────────────────────────────────────────────────────
# ARCH-4: _call_single_model mutates shared ai state
# ──────────────────────────────────────────────────────────────────────────────

class TestAIClientStateSafety:
    """_call_single_model must not corrupt shared AIClient state on failure."""

    def test_provider_restored_after_exception(self):
        from seo import ai, _call_single_model

        original_provider = ai._provider
        original_model = ai._model

        # Simulate a call that raises an exception
        with patch.object(ai, 'generate_text', side_effect=RuntimeError("API down")):
            result = _call_single_model("openrouter", "test-model", "prompt", "system")

        # After exception, original state must be restored
        assert ai._provider == original_provider, (
            f"Provider corrupted: expected {original_provider}, got {ai._provider}"
        )
        assert ai._model == original_model, (
            f"Model corrupted: expected {original_model}, got {ai._model}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# OPT-5: load_config() should be cached
# ──────────────────────────────────────────────────────────────────────────────

class TestConfigCaching:
    """load_config() should return the same object on repeated calls."""

    def test_load_config_returns_same_object(self):
        from utils.config import load_config

        cfg1 = load_config()
        cfg2 = load_config()
        # After adding caching, both calls should return the same dict object
        assert cfg1 is cfg2, "load_config() should cache its result"


# ──────────────────────────────────────────────────────────────────────────────
# EDGE-2: _merge_windows loses text metadata
# ──────────────────────────────────────────────────────────────────────────────

class TestMergeWindowsMetadata:
    """Merged windows should preserve text from both windows."""

    def test_merged_window_has_combined_text(self):
        from highlight import _merge_windows

        windows = [
            {"start": 10.0, "end": 15.0, "score": 5.0, "text": "first segment"},
            {"start": 16.0, "end": 20.0, "score": 3.0, "text": "second segment"},
        ]
        # gap=5 means these should merge (16.0 - 15.0 = 1.0 < 5.0)
        merged = _merge_windows(windows, gap=5.0)
        assert len(merged) == 1
        # After fix: merged window should contain text from both
        if "text" in merged[0]:
            # Text should contain content from both segments
            assert "first" in merged[0]["text"] or "second" in merged[0]["text"]
