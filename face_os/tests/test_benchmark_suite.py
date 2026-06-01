"""Tests for benchmark suite module.

Validates synthetic clip generators, metric functions, BenchmarkSuite,
and benchmark execution.
"""

from collections import namedtuple
from copy import deepcopy
from typing import Any, Dict, List

import numpy as np
import pytest

from face_os.benchmark_suite import (
    BenchmarkClip,
    BenchmarkMetrics,
    BenchmarkSuite,
    ClipCategory,
    SyntheticClipGenerator,
    compute_drift_score,
    compute_flicker_score,
    compute_geometric_consistency,
    create_default_suite,
)

try:
    from face_os.benchmark_suite import run_benchmark
    _HAS_RUN_BENCHMARK = True
except ImportError:
    _HAS_RUN_BENCHMARK = False

# ── Helper / shared constants ────────────────────────────────────────

TransformStub = namedtuple("TransformStub", ["scale"])

_EXPECTED_SHAPE = (360, 640, 3)

GENERATORS: List[tuple] = [
    ("easy", "generate_easy_clip", ()),
    ("medium", "generate_medium_clip", ()),
    ("hard", "generate_hard_clip", ()),
    ("adversarial", "generate_adversarial_clip", ()),
    ("occlusion", "generate_occlusion_clip", ()),
    ("dropped_frames", "generate_dropped_frames_clip", ()),
    ("lighting_change", "generate_lighting_change_clip", ()),
    ("overexposure", "generate_overexposure_clip", ()),
    ("webcam_noise", "generate_webcam_noise_clip", ()),
    ("rolling_shutter", "generate_rolling_shutter_clip", ()),
    ("beard_shadow", "generate_beard_shadow_clip", ()),
    ("face_cutoff", "generate_face_cutoff_clip", ()),
]


def _get_gen() -> SyntheticClipGenerator:
    return SyntheticClipGenerator()


def _get_generator_method(name: str):
    gen = _get_gen()
    method = getattr(gen, name, None)
    if method is None:
        pytest.skip(f"Generator {name} not yet implemented")
    return method


# ═══════════════════════════════════════════════════════════════════════
# 1. TestSyntheticClipGenerator — Basic Output Validity
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("gen_name,gen_method,gen_args", GENERATORS)
class TestSyntheticClipGenerator:
    """All synthetic clips produce valid, non-trivial frame sequences."""

    def test_produces_frames(self, gen_name, gen_method, gen_args):
        method = _get_generator_method(gen_method)
        frames = method(*gen_args)
        assert isinstance(frames, list)
        assert len(frames) == 90

    def test_frames_valid(self, gen_name, gen_method, gen_args):
        method = _get_generator_method(gen_method)
        frames = method(*gen_args)
        for f in frames:
            assert isinstance(f, np.ndarray)
            assert f.dtype == np.uint8
            assert f.ndim == 3, f"Expected 3 dims, got {f.ndim}"
            assert f.shape[2] == 3, f"Expected 3 channels, got {f.shape[2]}"
            assert f.shape == _EXPECTED_SHAPE, f"Shape mismatch: {f.shape}"

    def test_not_blank(self, gen_name, gen_method, gen_args):
        method = _get_generator_method(gen_method)
        frames = method(*gen_args)
        any_variance = False
        for f in frames:
            if np.var(f.astype(np.float64)) > 1.0:
                any_variance = True
                break
        assert any_variance, f"All frames appear blank for {gen_name}"


# ═══════════════════════════════════════════════════════════════════════
# 2. TestSyntheticClipGenerator — Edge Cases
# ═══════════════════════════════════════════════════════════════════════

class TestSyntheticClipGeneratorEdgeCases:
    """Custom parameterisation and stress-edge behaviour."""

    def test_custom_frame_count(self):
        gen = _get_gen()
        for method_name in [n for _, n, _ in GENERATORS]:
            method = getattr(gen, method_name, None)
            if method is None:
                continue
            for nf in (30, 150):
                frames = method(num_frames=nf)
                assert len(frames) == nf, f"{method_name}(num_frames={nf}) returned {len(frames)} frames"

    def test_occlusion_timing(self):
        gen = _get_gen()
        frames = gen.generate_occlusion_clip(
            num_frames=50, occlusion_start=10, occlusion_duration=15
        )
        assert len(frames) == 50

    def test_dropped_frames_has_black(self):
        gen = _get_gen()
        frames = gen.generate_dropped_frames_clip(num_frames=30, drop_every=3)
        zero_count = sum(1 for f in frames if np.all(f == 0))
        assert zero_count > 0, f"Expected some zero-valued frames, got {zero_count}"


# ═══════════════════════════════════════════════════════════════════════
# 3. TestMetricFunctions
# ═══════════════════════════════════════════════════════════════════════

class TestMetricFunctions:
    """Mathematical correctness of compute_* metric functions."""

    # ── drift ────────────────────────────────────────────────────

    def test_drift_single_frame(self):
        frames = _get_gen().generate_easy_clip(num_frames=1)
        assert compute_drift_score(frames) == 0.0

    def test_drift_identical_frames(self):
        f = _get_gen().generate_easy_clip(num_frames=1)[0]
        identical = [f.copy() for _ in range(10)]
        assert compute_drift_score(identical) == 0.0

    def test_drift_different_frames(self):
        frames = _get_gen().generate_hard_clip(num_frames=30)
        score = compute_drift_score(frames)
        assert score > 0.0, f"Expected drift > 0 for varying frames, got {score}"

    # ── flicker ──────────────────────────────────────────────────

    def test_flicker_single_frame(self):
        frames = _get_gen().generate_easy_clip(num_frames=1)
        assert compute_flicker_score(frames) == 0.0

    def test_flicker_constant_frames(self):
        f = _get_gen().generate_easy_clip(num_frames=1)[0]
        constant = [f.copy() for _ in range(10)]
        score = compute_flicker_score(constant)
        assert abs(score) < 1e-6, f"Flicker should be ~0 for constant frames, got {score}"

    # ── geometric consistency ─────────────────────────────────────

    def test_geometric_consistency_single(self):
        assert compute_geometric_consistency([]) == 1.0

    def test_geometric_consistency_deterministic(self):
        t = [TransformStub(scale=1.5) for _ in range(10)]
        score = compute_geometric_consistency(t)
        assert score > 0.9, f"Identical transforms should score > 0.9, got {score}"

    def test_geometric_consistency_decays(self):
        t_same = [TransformStub(scale=1.0) for _ in range(10)]
        t_vary = [TransformStub(scale=s) for s in np.linspace(0.5, 2.0, 10)]
        score_same = compute_geometric_consistency(t_same)
        score_vary = compute_geometric_consistency(t_vary)
        assert score_vary < score_same, (
            f"Varying scales should yield lower score; same={score_same:.4f}, vary={score_vary:.4f}"
        )


# ═══════════════════════════════════════════════════════════════════════
# 4. TestBenchmarkSuite
# ═══════════════════════════════════════════════════════════════════════

class TestBenchmarkSuite:
    """Core suite operations: add, filter, summarise."""

    def test_add_clip(self):
        suite = BenchmarkSuite()
        suite.add_clip("test", ClipCategory.EASY, "desc", "easy", lambda: [])
        assert len(suite.clips) == 1
        assert suite.clips[0].path == "test"
        assert suite.clips[0].category == ClipCategory.EASY

    def test_get_by_category(self):
        suite = BenchmarkSuite()
        suite.add_clip("e1", ClipCategory.EASY, "", "", None)
        suite.add_clip("h1", ClipCategory.HARD, "", "", None)
        suite.add_clip("h2", ClipCategory.HARD, "", "", None)
        suite.add_clip("a1", ClipCategory.ADVERSARIAL, "", "", None)

        easy = suite.get_clips_by_category(ClipCategory.EASY)
        hard = suite.get_clips_by_category(ClipCategory.HARD)
        med  = suite.get_clips_by_category(ClipCategory.MEDIUM)

        assert len(easy) == 1
        assert easy[0].path == "e1"
        assert len(hard) == 2
        assert len(med) == 0

    def test_get_summary_empty(self):
        suite = BenchmarkSuite()
        summary = suite.get_summary()
        assert isinstance(summary, dict)
        assert len(summary) == 0

    def test_get_summary_with_metrics(self):
        suite = BenchmarkSuite()
        suite.add_clip("c1", ClipCategory.EASY, "", "", None)
        suite.add_clip("c2", ClipCategory.EASY, "", "", None)
        suite.add_clip("c3", ClipCategory.HARD, "", "", None)

        metrics_easy_1 = BenchmarkMetrics(
            physical_render_rate=0.9,
            drift_score=2.5,
            flicker_score=1.2,
            geometric_consistency_score=0.85,
        )
        metrics_easy_2 = BenchmarkMetrics(
            physical_render_rate=0.7,
            drift_score=3.1,
            flicker_score=1.8,
            geometric_consistency_score=0.75,
        )
        metrics_hard = BenchmarkMetrics(
            physical_render_rate=0.4,
            drift_score=8.0,
            flicker_score=4.5,
            geometric_consistency_score=0.55,
        )

        suite.clips[0].metrics = metrics_easy_1
        suite.clips[1].metrics = metrics_easy_2
        suite.clips[2].metrics = metrics_hard

        summary = suite.get_summary()

        assert "EASY" in summary
        assert summary["EASY"]["clip_count"] == 2
        assert summary["EASY"]["metrics_count"] == 2
        assert summary["EASY"]["avg_drift_score"] == pytest.approx((2.5 + 3.1) / 2)
        assert summary["EASY"]["avg_flicker_score"] == pytest.approx((1.2 + 1.8) / 2)
        assert summary["EASY"]["avg_geometric_consistency"] == pytest.approx((0.85 + 0.75) / 2)

        assert "HARD" in summary
        assert summary["HARD"]["clip_count"] == 1
        assert summary["HARD"]["metrics_count"] == 1

        assert "MEDIUM" not in summary


# ═══════════════════════════════════════════════════════════════════════
# 5. TestRunBenchmark
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _HAS_RUN_BENCHMARK, reason="run_benchmark not yet implemented")
class TestRunBenchmark:
    """Execution-time assertion tests (fast – synthetic generators only)."""

    def test_run_benchmark_populates_metrics(self):
        suite = create_default_suite()
        run_benchmark(suite)
        for clip in suite.clips:
            assert clip.metrics is not None, f"{clip.path}: metrics not populated"

    def test_run_benchmark_metrics_range(self):
        suite = create_default_suite()
        run_benchmark(suite)
        for clip in suite.clips:
            m = clip.metrics
            assert isinstance(m.drift_score, float), f"{clip.path}: drift not float"
            assert isinstance(m.flicker_score, float), f"{clip.path}: flicker not float"
            assert m.drift_score >= 0.0, f"{clip.path}: drift negative"
            assert m.flicker_score >= 0.0, f"{clip.path}: flicker negative"
            assert np.isfinite(m.drift_score), f"{clip.path}: drift non-finite"
            assert np.isfinite(m.flicker_score), f"{clip.path}: flicker non-finite"

    def test_run_benchmark_total_frames(self):
        suite = create_default_suite()
        run_benchmark(suite)
        for clip in suite.clips:
            assert clip.metrics.total_frames == 90, (
                f"{clip.path}: expected 90 frames, got {clip.metrics.total_frames}"
            )


# ═══════════════════════════════════════════════════════════════════════
# 6. TestCreateDefaultSuite
# ═══════════════════════════════════════════════════════════════════════

class TestCreateDefaultSuite:
    """Default suite factory guarantees coverage of all expected clips."""

    def test_default_suite_has_all_clips(self):
        suite = create_default_suite()
        assert len(suite.clips) == 12, f"Expected 12 clips, got {len(suite.clips)}"

    def test_default_suite_covers_all_categories(self):
        suite = create_default_suite()
        categories_found = {clip.category for clip in suite.clips}
        all_categories = set(ClipCategory)
        assert categories_found == all_categories, (
            f"Missing categories: {all_categories - categories_found}"
        )

    def test_default_suite_generator_invocation(self):
        suite = create_default_suite()
        for clip in suite.clips:
            frames = clip.generator()
            assert isinstance(frames, list), f"{clip.path}: generator did not return list"
            assert len(frames) > 0, f"{clip.path}: generator returned empty list"
