"""
test_premium_render.py — Speed profile, interpolation stubs, two-pass VBR.
"""

import numpy as np
import pytest

from premium_render import generate_speed_profile, FrameInterpolator, FaceEnhancer


# ─── Speed Profile ────────────────────────────────────────────────────────

class TestGenerateSpeedProfile:
    def test_basic_profile_length(self):
        t, s = generate_speed_profile(10.0, fps=30)
        assert len(t) == len(s) == 300
        assert np.all(s >= 1.0)

    def test_default_speed_is_one(self):
        t, s = generate_speed_profile(5.0, fps=30)
        assert np.allclose(s, 1.0, atol=0.05)

    def test_silence_regions_speed_up(self):
        t, s = generate_speed_profile(10.0, fps=30, silence_regions=[(2.0, 4.0)])
        idx = (t >= 2.0) & (t <= 4.0)
        assert s[idx].mean() > 1.1

    def test_speed_clamped(self):
        t, s = generate_speed_profile(10.0, fps=30, base_speed=1.0, max_speed=1.25)
        assert s.max() <= 1.26

    def test_gaussian_smooth_no_spikes(self):
        t, s = generate_speed_profile(10.0, fps=60, silence_regions=[(4.0, 6.0)])
        diffs = np.abs(np.diff(s))
        assert diffs.max() < 0.1  # No abrupt jumps


# ─── FrameInterpolator ───────────────────────────────────────────────────

class TestFrameInterpolator:
    def test_backend_detected(self):
        fi = FrameInterpolator()
        assert fi.backend in ("ffmpeg", "rife")

    def test_interpolate_with_ffmpeg_fallback(self, tmp_path):
        """Test FFmpeg-based interpolation with a real video."""
        import subprocess
        src = tmp_path / "src.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=2:size=640x360:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-c:v", "libx264", "-c:a", "aac", "-shortest", str(src),
        ], capture_output=True)
        out = tmp_path / "out.mp4"
        fi = FrameInterpolator()
        speed = np.ones(120)  # 2s at 60fps
        ok = fi.interpolate(str(src), 0.0, 2.0, str(out), speed)
        assert ok
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_speed_profile_shape_matches(self):
        t, s = generate_speed_profile(2.0, fps=60)
        assert len(s) == 120  # 2 * 60


# ─── FaceEnhancer ────────────────────────────────────────────────────────

class TestFaceEnhancer:
    def test_backend_detected(self):
        fe = FaceEnhancer()
        assert fe.backend in ("none", "gfpgan")
