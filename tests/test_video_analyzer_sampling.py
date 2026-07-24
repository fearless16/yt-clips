"""Tests for video_analyzer.py frame sampling — NV12 pipe + software decode."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import pytest

INPUT_VIDEO = Path("input/video.mp4")


def _requires_video(func):
    """Skip test if input video is missing."""
    return pytest.mark.skipif(not INPUT_VIDEO.exists(), reason="input/video.mp4 not found")(func)


class TestNv12ToBgrConversion:
    """NV12 → BGR conversion must produce correct 3-channel output."""

    def test_nv12_to_bgr_shape(self):
        w, h = 320, 180
        nv12 = np.zeros((h * 3 // 2, w), dtype=np.uint8)
        bgr = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
        assert bgr.shape == (h, w, 3), f"Expected ({h},{w},3), got {bgr.shape}"
        assert bgr.dtype == np.uint8

    def test_nv12_to_bgr_gray_preservation(self):
        """Gray NV12 frame → all output channels should match."""
        w, h = 64, 48
        gray_val = 100
        nv12 = np.full((h * 3 // 2, w), 128, dtype=np.uint8)
        nv12[:h, :] = gray_val  # Y plane = gray
        bgr = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
        assert np.allclose(bgr.mean(axis=2), gray_val, atol=5)

    def test_nv12_to_bgr_different_sizes(self):
        for w, h in [(320, 180), (640, 360), (192, 108)]:
            nv12 = np.zeros((h * 3 // 2, w), dtype=np.uint8)
            bgr = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
            assert bgr.shape == (h, w, 3), f"Failed for {w}x{h}"


class TestFfmpegNv12Pipe:
    """FFmpeg NV12 pipe produces valid BGR frames after conversion."""

    @_requires_video
    def test_software_nv12_pipe_produces_frames(self):
        import subprocess
        PROBE = ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height,r_frame_rate",
                 "-of", "json", str(INPUT_VIDEO)]
        r = subprocess.run(PROBE, capture_output=True, text=True, timeout=10)
        import json
        data = json.loads(r.stdout)
        stream = data.get("streams", [{}])[0]
        w, h = int(stream["width"]), int(stream["height"])

        step = 120  # every 120th frame at 60fps ≈ every 2s
        cmd = [
            "ffmpeg", "-hide_banner", "-i", str(INPUT_VIDEO),
            "-t", "5",
            "-vf", f"select=not(mod(n\\,{step}))",
            "-vsync", "0",
            "-f", "rawvideo", "-pix_fmt", "nv12", "pipe:1",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**7)
        frame_size_nv12 = w * h * 3 // 2
        frames = []
        for _ in range(5):
            raw = proc.stdout.read(frame_size_nv12)
            if not raw or len(raw) < frame_size_nv12:
                break
            nv12 = np.frombuffer(raw, np.uint8).reshape((h * 3 // 2, w))
            bgr = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
            frames.append(bgr)
        proc.kill()
        proc.wait()

        assert len(frames) > 0, "NV12 pipe produced zero frames"
        for bgr in frames:
            assert bgr.shape == (h, w, 3), f"Frame shape {bgr.shape} != {(h,w,3)}"
            assert bgr.dtype == np.uint8

    @_requires_video
    def test_software_nv12_is_faster_than_hw_bgr24(self):
        """Software + NV12 should complete faster than dxva2 + BGR24 for short segment."""
        import subprocess
        import time

        def measure(args):
            t0 = time.perf_counter()
            proc = subprocess.Popen(
                args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            proc.wait(timeout=30)
            return time.perf_counter() - t0

        nv12_cmd = [
            "ffmpeg", "-hide_banner", "-i", str(INPUT_VIDEO),
            "-t", "5",
            "-vf", "select=not(mod(n\\,120))",
            "-vsync", "0",
            "-f", "rawvideo", "-pix_fmt", "nv12", "pipe:1",
        ]
        hw_cmd = [
            "ffmpeg", "-hide_banner", "-hwaccel", "dxva2", "-i", str(INPUT_VIDEO),
            "-t", "5",
            "-vf", "select=not(mod(n\\,120))",
            "-vsync", "0",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
        ]
        t_nv12 = measure(nv12_cmd)
        t_hw = measure(hw_cmd)
        assert t_nv12 < t_hw, (
            f"Software+NV12 ({t_nv12:.3f}s) should be faster than "
            f"dxva2+BGR24 ({t_hw:.3f}s)"
        )


class TestVideoAnalyzerIntegration:
    """End-to-end test of the sampling function."""

    @_requires_video
    def test_sample_frames_returns_valid_frames(self):
        from video_analyzer import _sample_frames, _probe_video
        probe = _probe_video(str(INPUT_VIDEO))
        duration = probe.get("duration", 0)
        # Only sample first 30s of video to keep test fast
        import subprocess
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(INPUT_VIDEO),
             "-t", "10", "-c", "copy", "-y", tmp.name],
            capture_output=True, timeout=30,
        )
        try:
            frames = list(_sample_frames(tmp.name, interval_sec=2.0))
            assert len(frames) > 0, "Should produce at least one frame"
            for ts, frame in frames:
                assert ts >= 0, f"Timestamp {ts} should be >= 0"
                assert len(frame.shape) == 3, f"Frame should be 3-channel BGR, got shape {frame.shape}"
                h, w = frame.shape[:2]
                assert w > 0 and h > 0, f"Frame dimensions {w}x{h} invalid"
                assert frame.dtype == np.uint8
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    @_requires_video
    def test_sample_frames_interval(self):
        from video_analyzer import _sample_frames
        import subprocess
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(INPUT_VIDEO),
             "-t", "10", "-c", "copy", "-y", tmp.name],
            capture_output=True, timeout=30,
        )
        try:
            frames = list(_sample_frames(tmp.name, interval_sec=2.0))
            if len(frames) >= 3:
                ts_diff = frames[2][0] - frames[1][0]
                assert abs(ts_diff - 2.0) < 0.5, (
                    f"Frame interval {ts_diff:.2f}s should be ~2.0s"
                )
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    @_requires_video
    def test_sample_frames_face_detectable(self):
        """Frames from new pipe should work with face detection."""
        from video_analyzer import _sample_frames
        from utils.face_detect import detect_faces
        import subprocess
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(INPUT_VIDEO),
             "-t", "10", "-c", "copy", "-y", tmp.name],
            capture_output=True, timeout=30,
        )
        try:
            for ts, frame in _sample_frames(tmp.name, interval_sec=2.0):
                faces = detect_faces(frame, score_threshold=0.5)
                assert isinstance(faces, list)
                break
        finally:
            Path(tmp.name).unlink(missing_ok=True)
