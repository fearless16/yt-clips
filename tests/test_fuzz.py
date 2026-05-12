"""
test_fuzz.py — Fuzz testing: random/shit inputs at every public function.
If something crashes, we catch it here before it hits the pipeline.
"""

import numpy as np
import pytest
import random
import string
import subprocess
from pathlib import Path
from typing import Any


def rand_str():
    return ''.join(random.choices(string.printable, k=random.randint(0, 100)))

def rand_float():
    return random.uniform(-1000, 1000)

def rand_int():
    return random.randint(-10000, 10000)


# ─── frame_analyzer fuzz ────────────────────────────────────────────────

class TestFrameAnalyzerFuzz:
    def _get_mod(self):
        import frame_analyzer as m
        return m

    def test_smooth_int_fuzz(self):
        m = self._get_mod()
        for _ in range(30):
            prev = random.choice([None, rand_int()])
            curr = rand_int()
            alpha = rand_float()
            try:
                result = m._smooth_int(prev, curr, alpha)
                assert isinstance(result, int)
            except (TypeError, ValueError):
                pass  # Expected with garbage
            except Exception as e:
                pytest.fail(f"_smooth_int crashed: {e}")

    def test_frame_stats_fuzz(self):
        m = self._get_mod()
        for _ in range(30):
            val = random.choice([None, b'', b'\x00' * 100, rand_str().encode(), bytes(range(256))])
            try:
                result = m._frame_stats(val)
                assert isinstance(result, dict)
                assert 'avg' in result
                assert 'var' in result
            except (TypeError, ValueError):
                pass
            except Exception as e:
                pytest.fail(f"_frame_stats crashed: {e}")

    def test_detect_black_frames_fuzz(self):
        m = self._get_mod()
        for _ in range(30):
            samples = random.choice([
                [],
                [{}],
                [{"avg": rand_float(), "var": rand_float()}],
                None,
                "garbage",
            ])
            try:
                safe = samples if isinstance(samples, list) else []
                result = m.detect_black_frames(safe)
                assert isinstance(result, dict)
            except (TypeError, ValueError, KeyError, IndexError):
                pass
            except Exception as e:
                pytest.fail(f"detect_black_frames crashed: {e}")

    def test_analyze_lighting_fuzz(self):
        m = self._get_mod()
        for _ in range(30):
            samples = random.choice([[], [{"avg": rand_float()}], None, "x"])
            try:
                result = m.analyze_lighting(samples if isinstance(samples, list) else [])
                assert isinstance(result, dict)
            except (TypeError, ValueError):
                pass
            except Exception as e:
                pytest.fail(f"analyze_lighting crashed: {e}")

    def test_get_video_dimensions_fuzz(self):
        m = self._get_mod()
        for _ in range(10):
            path = random.choice(["", "/nonexistent/video.mp4", "/dev/null", rand_str()])
            try:
                result = m._get_video_dimensions(path)
                assert "width" in result
                assert "height" in result
            except Exception:
                pass  # FFmpeg errors expected

    def test_has_vertical_divider_fuzz(self):
        m = self._get_mod()
        for _ in range(20):
            shape = random.choice([(180, 320), (100, 100), (50, 200), (1, 1)])
            arr = np.random.randint(0, 256, size=shape, dtype=np.uint8)
            w = random.choice([320, 100, 200, 1, 0, -1])
            try:
                m._has_vertical_divider(arr, w)
            except Exception as e:
                pytest.fail(f"_has_vertical_divider crashed on shape {shape}, w={w}: {e}")


# ─── premium_analyzer fuzz ──────────────────────────────────────────────

class TestPremiumAnalyzerFuzz:
    def _get_mod(self):
        import premium_analyzer as m
        return m

    def test_bezier_interpolate_fuzz(self):
        m = self._get_mod()
        for _ in range(30):
            p0, p1, p2, p3 = [rand_float() for _ in range(4)]
            t = rand_float()
            try:
                result = m._bezier_interpolate(p0, p1, p2, p3, t)
                assert isinstance(result, float)
            except Exception as e:
                pytest.fail(f"_bezier_interpolate crashed: {e}")

    def test_iou_fuzz(self):
        m = self._get_mod()
        for _ in range(30):
            a = np.random.randn(4).astype(np.float32) * 1000
            b = np.random.randn(4).astype(np.float32) * 1000
            try:
                result = m._iou(a, b)
                assert isinstance(result, (float, np.floating))
                assert 0.0 <= float(result) <= 1.0 or np.isnan(float(result))
            except Exception as e:
                pytest.fail(f"_iou crashed: {e}")

    def test_bytetrack_fuzz(self):
        m = self._get_mod()
        for _ in range(10):
            try:
                bt = m.ByteTrack()
                num_dets = random.randint(0, 10)
                dets = np.random.randn(num_dets, 4).astype(np.float32) * 1000
                scores = np.random.rand(num_dets).astype(np.float32)
                result = bt.update(dets, scores)
                assert isinstance(result, list)
            except ImportError:
                pass
            except Exception as e:
                pytest.fail(f"ByteTrack crashed with {num_dets} dets: {e}")

    def test_smooth_crop_fuzz(self):
        m = self._get_mod()
        for _ in range(10):
            try:
                sc = m.SmoothCrop(random.randint(1, 3840), random.randint(1, 2160))
                cx = rand_float()
                cy = rand_float()
                result = sc.get_crop(cx, cy)
                assert "x" in result
                assert "y" in result
                assert "width" in result
                assert "height" in result
            except Exception as e:
                pytest.fail(f"SmoothCrop crashed: {e}")

    def test_classify_layout_fuzz(self):
        m = self._get_mod()
        for _ in range(5):
            h = random.choice([1, 10, 100, 360])
            w = random.choice([1, 10, 100, 640])
            frame = np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
            try:
                m._classify_layout(frame)
            except Exception as e:
                pytest.fail(f"_classify_layout crashed on {h}x{w}: {e}")


# ─── premium_render fuzz ────────────────────────────────────────────────

class TestPremiumRenderFuzz:
    def _get_mod(self):
        import premium_render as m
        return m

    def test_generate_speed_profile_fuzz(self):
        m = self._get_mod()
        for _ in range(10):
            duration = random.choice([0.0, 1.0, 10.0, 30.0])
            fps = random.choice([1, 30, 60])
            try:
                t, s = m.generate_speed_profile(max(0.01, duration), max(1, fps))
                assert len(t) == len(s) > 0
                assert np.all(s >= 0.5)
            except Exception as e:
                pytest.fail(f"generate_speed_profile(d={duration}, fps={fps}) crashed: {e}")

    def test_encode_two_pass_fuzz(self, tmp_path):
        m = self._get_mod()
        for _ in range(5):
            inp = tmp_path / f"nonexistent_{rand_int()}.mp4"
            out = tmp_path / f"out_{rand_int()}.mp4"
            br = random.choice(["15M", "1500K", "", "abc", rand_str()])
            try:
                m.encode_two_pass(str(inp), str(out), br)
            except (FileNotFoundError, subprocess.CalledProcessError, RuntimeError, ValueError):
                pass  # Expected with nonexistent input
            except Exception as e:
                pytest.fail(f"encode_two_pass crashed: {e}")


# ─── export.py fuzz ────────────────────────────────────────────────────

class TestExportFuzz:
    def _get_mod(self):
        from export import _normalize_speed, _sanitize_strategy, _parse_fps, _get_video_info
        return locals()

    def test_normalize_speed_fuzz(self):
        m = self._get_mod()
        for _ in range(30):
            val = random.choice([None, "abc", [], {}, rand_float(), rand_int(), float('nan')])
            try:
                result = m['_normalize_speed'](val)
                assert 0.25 <= result <= 4.0
            except Exception as e:
                pytest.fail(f"_normalize_speed crashed: {e}")

    def test_parse_fps_fuzz(self):
        m = self._get_mod()
        for _ in range(30):
            val = random.choice([None, "abc", "30/1", "0/0", "nan", "inf", rand_str(), rand_float()])
            try:
                result = m['_parse_fps'](val)
                assert isinstance(result, float)
            except Exception as e:
                pytest.fail(f"_parse_fps crashed: {e}")

    def test_sanitize_strategy_fuzz(self):
        m = self._get_mod()
        for _ in range(30):
            val = random.choice([
                None,
                {},
                {"active_crop": None},
                {"active_crop": {"x": "abc", "y": None, "width": -1, "height": []}},
                {"speed_factor": None},
                {"speed_factor": float('nan')},
                {"speed_factor": 999},
                rand_str(),
                [],
            ])
            try:
                result = m['_sanitize_strategy'](val)
                assert isinstance(result, dict)
            except Exception as e:
                pytest.fail(f"_sanitize_strategy crashed: {e}")

    def test_get_video_info_fuzz(self):
        m = self._get_mod()
        for _ in range(10):
            path = random.choice(["", "/dev/random", rand_str(), "/nonexistent"])
            try:
                result = m['_get_video_info'](path)
                assert "width" in result
                assert "height" in result
                assert "fps" in result
            except Exception:
                pass


# ─── highlight.py fuzz ──────────────────────────────────────────────────

class TestHighlightFuzz:
    def test_words_per_minute_fuzz(self):
        from highlight import _words_per_minute
        for _ in range(30):
            text = random.choice(["", rand_str(), " ".join([rand_str() for _ in range(10)])])
            dur = rand_float()
            try:
                result = _words_per_minute(text, dur)
                assert isinstance(result, float)
            except Exception as e:
                pytest.fail(f"_words_per_minute crashed: {e}")

    def test_silence_seconds_fuzz(self):
        from highlight import _silence_seconds
        for _ in range(30):
            text = random.choice(["", rand_str(), None])
            dur = rand_float()
            try:
                result = _silence_seconds(text, max(0.01, dur))
                assert isinstance(result, float)
            except (AttributeError, TypeError, ValueError):
                pass
            except Exception as e:
                pytest.fail(f"_silence_seconds crashed: {e}")

    def test_format_ts_fuzz(self):
        from highlight import _format_ts
        for _ in range(30):
            val = random.choice([0, -1, 999999, rand_float(), rand_int()])
            try:
                result = _format_ts(max(0, val))
                assert isinstance(result, str)
            except Exception:
                pass
