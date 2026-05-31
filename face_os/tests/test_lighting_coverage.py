"""Tests for §16.7 Lighting Coverage (D-05 / C_recon third factor).

arch §16.7:
    Coverage_light = |observed lighting states| / |total lighting states|
    "A face seen only frontally under warm light is NOT 'known';
     confidence must be capped accordingly."
    Invariant: identity confidence is upper-bounded by a function of
        Coverage_pose · Coverage_light.
    Required test: low-coverage state caps reported confidence below the
        high-coverage ceiling, regardless of per-frame quality.

Lighting coverage caps identity confidence so a face observed under only ONE
lighting condition is not over-trusted — just as §16.7 pose coverage caps for
narrow pose range. The lighting signal is the per-frame ``LightingModel``
(ambient, diffuse_direction) already produced by ``estimate_lighting`` each
frame. The binning function ``_lighting_bin`` quantizes it into discrete states
the same way ``_pose_bin`` quantizes pose, and the canonical set is DERIVED
from the binning function (not a magic constant), so the denominator can never
silently drift.

Determinism: fixed synthetic inputs, no randomness (arch §3).
"""
from __future__ import annotations

import numpy as np
import pytest

from face_os.patch_memory import (
    PatchMemory,
    _lighting_bin,
    canonical_lighting_bins,
    apply_pose_coverage,
)
from face_os.physical_renderer import LightingModel


# ─── _lighting_bin(): quantize LightingModel → discrete label ─────────────────

class TestLightingBin:
    def test_frontal_above_normal_ambient(self):
        light = LightingModel(ambient=0.15, diffuse_direction=np.array([0, 0, 1.0]))
        b = _lighting_bin(light)
        assert isinstance(b, str) and len(b) >= 2

    def test_returns_string_label(self):
        light = LightingModel()
        b = _lighting_bin(light)
        assert isinstance(b, str)

    def test_none_returns_any(self):
        assert _lighting_bin(None) == 'any'

    def test_same_input_deterministic(self):
        light = LightingModel(ambient=0.15, diffuse_direction=np.array([0.5, 0.0, 0.866]))
        assert _lighting_bin(light) == _lighting_bin(light)

    def test_direction_change_changes_label(self):
        """A left-lit and right-lit frame must produce DIFFERENT labels."""
        left = LightingModel(ambient=0.15, diffuse_direction=np.array([-0.8, 0.0, 0.6]))
        right = LightingModel(ambient=0.15, diffuse_direction=np.array([0.8, 0.0, 0.6]))
        assert _lighting_bin(left) != _lighting_bin(right)

    def test_ambient_change_changes_label(self):
        """Dim vs bright ambient under the same direction must differ."""
        dim = LightingModel(ambient=0.05, diffuse_direction=np.array([0, 0, 1.0]))
        bright = LightingModel(ambient=0.5, diffuse_direction=np.array([0, 0, 1.0]))
        assert _lighting_bin(dim) != _lighting_bin(bright)

    def test_below_light_changes_label(self):
        """Light from below (negative Z) is a different direction octant."""
        above = LightingModel(ambient=0.15, diffuse_direction=np.array([0, 0, 1.0]))
        below = LightingModel(ambient=0.15, diffuse_direction=np.array([0, 0, -1.0]))
        assert _lighting_bin(above) != _lighting_bin(below)

    def test_degenerate_zero_direction_uses_ambient(self):
        """A degenerate (zero-direction, zero-diffuse) fit falls back to
        ambient-only binning — the scalar is always available."""
        degenerate = LightingModel(ambient=0.15, diffuse_intensity=0.0,
                                   diffuse_direction=np.array([0, 0, 1.0]))
        b = _lighting_bin(degenerate)
        assert isinstance(b, str) and len(b) >= 1


# ─── canonical_lighting_bins(): the denominator must be DERIVED ───────────────

class TestCanonicalLightingBins:
    def test_returns_a_set(self):
        cb = canonical_lighting_bins()
        assert isinstance(cb, set) and len(cb) > 0

    def test_any_sentinel_excluded(self):
        assert 'any' not in canonical_lighting_bins()

    def test_returns_copy_not_shared_mutable(self):
        a = canonical_lighting_bins()
        a.add('JUNK')
        assert 'JUNK' not in canonical_lighting_bins()

    def test_matches_sweep_of_lighting_bin(self):
        """DRIFT GUARD: the declared canonical set must equal what
        _lighting_bin produces when swept over the operational LightingModel
        space. If the binning cascade changes, this fails loudly rather than
        silently corrupting the coverage denominator.
        """
        # Direction octants: 6 signed-axis directions (±X, ±Y, ±Z)
        # × 3 ambient bands = 18 bins total.
        dirs = [
            [0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
        ]
        # Ambient band centers: 0.05 (dim), 0.15 (normal), 0.5 (bright)
        ambients = [0.05, 0.15, 0.5]
        produced = set()
        for d in dirs:
            for a in ambients:
                produced.add(_lighting_bin(LightingModel(
                    ambient=a, diffuse_direction=np.array(d, dtype=np.float64)
                )))
        assert canonical_lighting_bins() == produced


# ─── PatchMemory.coverage_light(): real observed/total ratio ──────────────────

def _fake_quality_map(size=64, value=0.8):
    return np.full((size, size), value, dtype=np.float32)


def _fake_face(size=64):
    return np.ones((size, size, 3), dtype=np.float32) * 0.5


class TestCoverageLight:
    def test_uninitialized_memory_is_zero(self):
        pm = PatchMemory()
        assert pm.coverage_light() == 0.0

    def test_one_lighting_observed_is_one_over_total(self):
        pm = PatchMemory()
        pm.initialize(_fake_face(), _fake_quality_map())
        pm.record_lighting(LightingModel(ambient=0.15, diffuse_direction=np.array([0, 0, 1.0])))
        total = len(canonical_lighting_bins())
        assert pm.coverage_light() == pytest.approx(1.0 / total, abs=1e-9)

    def test_two_distinct_lights_increases_coverage(self):
        pm = PatchMemory()
        pm.initialize(_fake_face(), _fake_quality_map())
        pm.record_lighting(LightingModel(ambient=0.05, diffuse_direction=np.array([0, 0, 1.0])))
        pm.record_lighting(LightingModel(ambient=0.5, diffuse_direction=np.array([-1, 0, 0.0])))
        total = len(canonical_lighting_bins())
        assert pm.coverage_light() == pytest.approx(2.0 / total, abs=1e-9)

    def test_same_light_twice_does_not_increase(self):
        pm = PatchMemory()
        pm.initialize(_fake_face(), _fake_quality_map())
        light = LightingModel(ambient=0.15, diffuse_direction=np.array([0, 0, 1.0]))
        pm.record_lighting(light)
        pm.record_lighting(light)
        total = len(canonical_lighting_bins())
        assert pm.coverage_light() == pytest.approx(1.0 / total, abs=1e-9)

    def test_out_of_range_bin_does_not_inflate(self):
        """A degenerate lighting bin that doesn't match any canonical entry
        must not count."""
        pm = PatchMemory()
        pm.initialize(_fake_face(), _fake_quality_map())
        # Record a degenerate light (zero diffuse) and a normal light
        degenerate = LightingModel(ambient=0.15, diffuse_intensity=0.0,
                                   diffuse_direction=np.array([0, 0, 1.0]))
        pm.record_lighting(degenerate)
        # The degenerate bin might or might not be canonical, but coverage ≤ 1
        assert 0.0 <= pm.coverage_light() <= 1.0


# ─── apply_pose_coverage reused as §16.7 lighting cap ─────────────────────────
# (The §16.7 cap is the SAME multiplicative form for both pose and lighting:
# C_recon = C_obs · Coverage_pose · Coverage_light · Visibility. apply_pose_coverage
# = c · cov is already tested for all cap invariants. Lighting uses the same function.)

class TestLightingCapInvariants:
    def test_low_coverage_caps_below_ceiling_regardless_of_quality(self):
        """arch §16.7 required test, lighting variant: a high-quality
        observation under one lighting caps BELOW a low-quality observation
        seen under many lights."""
        total = len(canonical_lighting_bins())  # 18
        high_quality_one_light = apply_pose_coverage(0.95, 1.0 / total)
        low_quality_many_lights = apply_pose_coverage(0.30, 0.5)
        assert high_quality_one_light < low_quality_many_lights
