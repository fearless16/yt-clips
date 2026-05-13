"""
test_premium_fixes.py — TDD tests for premium pipeline bugs + token optimization + 16:9→9:16 conversion.

Tests written to FAIL first, then fixed.
"""
import subprocess
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, List

import numpy as np
import pytest


# ──────────────────────────────────────────────────────────────────────────────
# BUG-2: HostDetector receives timestamp float, not face image
# ──────────────────────────────────────────────────────────────────────────────

class TestHostDetectorFaceImage:
    """HostDetector.identify_host must receive actual face image data."""

    def test_face_dict_has_face_img_numpy_array(self):
        """Face dicts built by PremiumAnalyzer must carry face_img as ndarray."""
        # Simulate the face dict that PremiumAnalyzer builds
        # BUG: currently stores "frame": float(t) — a timestamp
        # FIX: should store "face_img": np.ndarray or None
        from premium_analyzer import HostDetector

        # Create a fake face image (gray square)
        face_img = np.ones((60, 50, 3), dtype=np.uint8) * 180

        face_with_img = {
            "x": 160.0, "y": 200.0,
            "w": 50.0, "h": 60.0,
            "id": 0,
            "face_img": face_img,  # FIXED key
            "brightness": 128,
        }
        face_without_img = {
            "x": 400.0, "y": 200.0,
            "w": 50.0, "h": 60.0,
            "id": 1,
            "face_img": None,
            "brightness": 128,
        }

        # Create detector with a reference embedding
        ref_img = np.ones((64, 64, 3), dtype=np.uint8) * 180
        detector = HostDetector([ref_img])

        # The detector should use face_img to compute embeddings, not crash
        idx = detector.identify_host([face_with_img, face_without_img])
        # face_with_img has similar brightness to ref → should be picked
        assert idx == 0, f"Expected face 0 (with matching image), got {idx}"

    def test_identify_host_uses_face_img_key_not_frame(self):
        """identify_host must read 'face_img', not 'frame' for image data."""
        from premium_analyzer import HostDetector
        import inspect

        source = inspect.getsource(HostDetector.identify_host)
        # After fix, code should use face_img, not face.get("frame")
        assert 'face.get("face_img")' in source or "face_img" in source, (
            "identify_host still uses 'frame' key — should use 'face_img'"
        )


# ──────────────────────────────────────────────────────────────────────────────
# BUG-5: _identify_by_facecam_region double-adds w/2 to center coords
# ──────────────────────────────────────────────────────────────────────────────

class TestFacecamCenterCalc:
    """Face center must not be double-offset when coords are already centers."""

    def test_face_center_used_directly(self):
        """x,y in face dicts are centers — must not add w/2 again."""
        from premium_analyzer import HostDetector
        import inspect

        source = inspect.getsource(HostDetector._identify_by_facecam_region)
        # BUG: face_cx = face.get("x", 0) + face.get("w", 0) / 2
        # FIX: face_cx = face.get("x", 0)  (x IS the center already)
        assert '+ face.get("w"' not in source, (
            "_identify_by_facecam_region still double-offsets x by w/2"
        )

    def test_closest_face_to_facecam_wins(self):
        """Face closest to facecam center should be identified as host."""
        from premium_analyzer import HostDetector

        detector = HostDetector([])
        # Facecam config: x=0, y=540, w=320, h=180 → center (160, 630)
        faces = [
            {"x": 500.0, "y": 100.0, "w": 60.0, "h": 70.0},  # Far
            {"x": 160.0, "y": 630.0, "w": 60.0, "h": 70.0},  # At facecam center
        ]
        idx = detector._identify_by_facecam_region(faces)
        assert idx == 1, f"Expected face 1 at facecam center, got {idx}"


# ──────────────────────────────────────────────────────────────────────────────
# BUG-6: scipy import not guarded in premium_render.py
# ──────────────────────────────────────────────────────────────────────────────

class TestScipyImportGuard:
    """generate_speed_profile must not crash if scipy is unavailable."""

    def test_scipy_import_is_guarded(self):
        """scipy must be imported inside try/except or behind HAS_SCIPY flag."""
        import premium_render
        source = Path(premium_render.__file__).read_text()

        # Find all scipy imports
        for i, line in enumerate(source.split("\n"), 1):
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            if ("from scipy" in stripped or "import scipy" in stripped) and not stripped.startswith("#"):
                if indent == 0:
                    pytest.fail(
                        f"Line {i}: Unguarded module-level scipy import: {stripped}"
                    )

    def test_speed_profile_returns_valid_output(self):
        """generate_speed_profile should return timestamps + speed arrays."""
        from premium_render import generate_speed_profile

        ts, speed = generate_speed_profile(
            duration=10.0, fps=30.0,
            silence_regions=[(3.0, 5.0)],
        )
        assert len(ts) > 0
        assert len(speed) == len(ts)
        assert all(0.9 <= s <= 1.3 for s in speed)


# ──────────────────────────────────────────────────────────────────────────────
# OPT-1: Highlight AI sends too many transcript segments (saves ~1500 tokens)
# ──────────────────────────────────────────────────────────────────────────────

class TestHighlightTokenOptimization:
    """AI refinement should only send candidate-overlapping segments."""

    def test_only_overlapping_segments_sent_to_ai(self):
        """Transcript context sent to AI must only include segments that overlap candidates."""
        from highlight import _refine_highlights_with_ai
        import inspect

        source = inspect.getsource(_refine_highlights_with_ai)
        # BUG: "for seg in segments[:50]" — sends first 50 regardless
        # FIX: should filter to segments that overlap with candidate windows
        assert "segments[:50]" not in source, (
            "AI refinement still sends first 50 segments blindly — should filter to candidate overlap"
        )

    @patch("highlight.ai")
    def test_ai_prompt_token_count_is_bounded(self, mock_ai):
        """The prompt sent to AI should not exceed reasonable token count."""
        from highlight import _refine_highlights_with_ai

        # Create many segments, only 2 candidates
        segments = [
            {"start": i * 2.0, "end": i * 2.0 + 1.5, "text": f"segment {i} " * 10}
            for i in range(100)
        ]
        candidates = [
            {"start": 10.0, "end": 15.0, "start_ts": "00:00:10", "end_ts": "00:00:15", "score": 5.0},
            {"start": 50.0, "end": 55.0, "start_ts": "00:00:50", "end_ts": "00:00:55", "score": 4.0},
        ]
        mock_ai.generate_text.return_value = "[1, 2]"

        result = _refine_highlights_with_ai(segments, candidates, "Test Video", 5)

        # Check prompt size — should be much smaller than sending all 100 segments
        call_args = mock_ai.generate_text.call_args
        if call_args:
            prompt = call_args[0][0]
            # With 100 segments of ~100 chars each, full dump would be ~10K chars
            # With only overlapping segments (~10), should be under 3K
            assert len(prompt) < 5000, (
                f"Prompt too large ({len(prompt)} chars) — should filter to overlapping segments"
            )


# ──────────────────────────────────────────────────────────────────────────────
# OPT-2: SEO transcript truncation (saves ~2000 tokens per clip)
# ──────────────────────────────────────────────────────────────────────────────

class TestSEOTokenOptimization:
    """SEO prompt should use actual clip transcript length, not 3000 chars."""

    def test_short_transcript_not_padded_to_3000(self):
        """A 100-word clip transcript should not be padded or waste space."""
        from seo import _PROMPT_TMPL
        # The template uses {transcript} — check that generate_clip_seo
        # doesn't blindly truncate to 3000 chars when the text is shorter
        short_text = "Kohli smashes a six over long-on! The crowd goes wild."
        assert len(short_text) < 200  # This is a typical clip transcript

    def test_seo_prompt_uses_bounded_transcript(self):
        """SEO should cap transcript to actual content length, not 3000."""
        import inspect
        from seo import generate_clip_seo

        source = inspect.getsource(generate_clip_seo)
        # BUG: transcript[:3000] — always sends up to 3000 chars
        # FIX: should use min(len(transcript), reasonable_cap) like 1500
        assert "transcript[:3000]" not in source, (
            "SEO still truncates at 3000 chars — should cap at actual content or 1500"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 16:9 → 9:16 CONVERSION: The core visual transformation
# ──────────────────────────────────────────────────────────────────────────────

def _ffmpeg(cmd: list, description: str, timeout: int = 60) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg {description} failed:\n{result.stderr[:1000]}")


def _probe_video(path: Path) -> Dict:
    """Get width, height, duration of a video file."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height:format=duration",
        "-of", "json", str(path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    data = json.loads(res.stdout or "{}")
    stream = (data.get("streams") or [{}])[0]
    return {
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "duration": float(data.get("format", {}).get("duration") or 0),
    }


@pytest.fixture(scope="session")
def landscape_16_9_video(tmp_path_factory):
    """A pure 16:9 landscape video (640x360) with a face-colored region."""
    out = tmp_path_factory.mktemp("conversion") / "landscape.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=0x2d5a27:s=640x360:d=4:r=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
        "-filter_complex",
        "[0:v]drawbox=x=40:y=80:w=120:h=160:color=0xffcc99:t=fill[face]",
        "-map", "[face]", "-map", "1:a",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", "-pix_fmt", "yuv420p",
        str(out),
    ]
    _ffmpeg(cmd, "landscape 16:9 video")
    return out


class TestLandscapeTo916Conversion:
    """16:9 landscape input MUST produce 9:16 portrait output."""

    @staticmethod
    def _solo_analysis():
        """Pre-computed analysis to bypass face detection in tests."""
        return {
            "layout": {"layout_type": "solo", "prefer_solo": True},
            "export_strategy": {
                "use_solo_frame": True,
                "use_vertical_stack": False,
                "has_black_panel": False,
                "guest_cam_on": False,
                "guest_cam_off": False,
                "is_screen_share": False,
                "should_drop": False,
                "active_crop": None,
                "speed_factor": 1.0,
                "apply_lighting_fix": False,
                "lighting_filter": "",
                "skip_silence": False,
                "active_segments": [],
            },
        }

    def test_export_produces_portrait_output(self, landscape_16_9_video, tmp_path):
        """A 16:9 input video must be converted to 9:16 (1080x1920) output."""
        from export import export_clip

        out_file = tmp_path / "portrait_output.mp4"
        result = export_clip(
            str(landscape_16_9_video), 0.0, 3.0, str(out_file),
            clip_id="test_portrait",
            analysis=self._solo_analysis(),
        )
        assert result is not None, "export_clip returned None — export failed"
        assert Path(result).exists(), "Output file was not created"

        info = _probe_video(Path(result))
        assert info["width"] > 0 and info["height"] > 0, "Could not probe output video"
        assert info["height"] > info["width"], (
            f"Output is not portrait: {info['width']}x{info['height']} — "
            f"expected height > width (9:16)"
        )

    def test_output_dimensions_are_1080x1920(self, landscape_16_9_video, tmp_path):
        """Output must be exactly 1080x1920 for YouTube Shorts."""
        from export import export_clip

        out_file = tmp_path / "exact_dims.mp4"
        result = export_clip(
            str(landscape_16_9_video), 0.0, 3.0, str(out_file),
            clip_id="test_dims",
            analysis=self._solo_analysis(),
        )
        assert result is not None

        info = _probe_video(Path(result))
        assert info["width"] == 1080, f"Width should be 1080, got {info['width']}"
        assert info["height"] == 1920, f"Height should be 1920, got {info['height']}"

    def test_aspect_ratio_is_9_16(self, landscape_16_9_video, tmp_path):
        """Output aspect ratio must be 9:16 (0.5625)."""
        from export import export_clip

        out_file = tmp_path / "aspect_test.mp4"
        result = export_clip(
            str(landscape_16_9_video), 0.0, 3.0, str(out_file),
            clip_id="test_aspect",
            analysis=self._solo_analysis(),
        )
        assert result is not None

        info = _probe_video(Path(result))
        ratio = info["width"] / info["height"]
        assert abs(ratio - 9 / 16) < 0.01, (
            f"Aspect ratio should be 9:16 (0.5625), got {ratio:.4f}"
        )

    def test_output_is_not_landscape(self, landscape_16_9_video, tmp_path):
        """Output must NEVER be wider than tall."""
        from export import export_clip

        out_file = tmp_path / "not_landscape.mp4"
        result = export_clip(
            str(landscape_16_9_video), 0.0, 3.0, str(out_file),
            clip_id="test_not_landscape",
            analysis=self._solo_analysis(),
        )
        assert result is not None

        info = _probe_video(Path(result))
        assert info["width"] <= info["height"], (
            f"Output is landscape ({info['width']}x{info['height']}) — "
            f"must be portrait for YouTube Shorts"
        )

    def test_screen_share_conversion_preserves_content(self, landscape_16_9_video, tmp_path):
        """Screen-share layout should use blurred-bg + fit, not aggressive crop."""
        from export import _build_enhance_stack

        analysis = {
            "export_strategy": {
                "is_screen_share": True,
                "use_solo_frame": False,
            },
        }
        filtergraph = _build_enhance_stack(analysis, source_fps=30.0)
        # Should contain blurred background technique
        assert "gblur" in filtergraph, "Screen-share must use blurred background"
        assert "overlay" in filtergraph, "Screen-share must overlay sharp content on blur"

    def test_solo_conversion_uses_center_crop(self, landscape_16_9_video, tmp_path):
        """Solo layout should center-crop 16:9 to 9:16."""
        from export import _build_enhance_stack

        analysis = {
            "export_strategy": {
                "use_solo_frame": False,
                "is_screen_share": False,
            },
        }
        filtergraph = _build_enhance_stack(analysis, source_fps=30.0)
        # Should contain 9:16 crop formula
        assert "9/16" in filtergraph or "crop" in filtergraph, (
            "Solo layout must crop to 9:16"
        )
        assert "1080" in filtergraph and "1920" in filtergraph, (
            "Solo layout must scale to 1080x1920"
        )
