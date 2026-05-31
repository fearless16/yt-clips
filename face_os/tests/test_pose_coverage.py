"""Tests for §16.7 Pose Coverage (D-05 / C_recon prerequisite).

arch.md §16.7:
    Coverage_pose = |observed pose bins| / |total pose bins|
    Invariant: identity confidence is upper-bounded by a function of
        Coverage_pose (· Coverage_light).
    Required test: a low-coverage state caps reported confidence below the
        high-coverage ceiling, regardless of per-frame quality.

This is the FIRST real factor of the §16.8 composite
    C_recon = C_obs · Coverage_pose · Coverage_light · Visibility
so the cap here is the multiplicative pose factor of that product (the other
factors — §16.6 Visibility, lighting coverage — are still MISSING per §17 and
multiply in later). The cap is therefore NOT yet wired into the live Phase-2B
gate (§19 mandatory order); it is surfaced as an observable telemetry signal and
kept as a pure, tested function ready for §16.8 to compose.

Determinism: fixed synthetic inputs, no randomness (arch §3).
"""
from __future__ import annotations

import numpy as np
import pytest

from face_os.patch_memory import (
    PatchMemory,
    _pose_bin,
    canonical_pose_bins,
    apply_pose_coverage,
)


# ─── canonical_pose_bins(): the denominator must be DERIVED from _pose_bin ────
# A magic constant for |total| would silently drift if the binning cascade is
# edited. These tests pin the canonical set to what _pose_bin actually produces.

class TestCanonicalPoseBins:
    def test_total_is_37_directional_bins(self):
        """F + L10..L90 + R10..R90 + D10..D90 + U10..U90 = 1 + 9*4 = 37."""
        assert len(canonical_pose_bins()) == 37

    def test_any_sentinel_excluded(self):
        """'any' is the no-pose sentinel (pose is None), never a directional bin.

        _pose_bin only stores a bin when pose is not None (patch_memory.py:159),
        so 'any' must not count toward pose coverage.
        """
        assert 'any' not in canonical_pose_bins()

    def test_canonical_set_matches_pose_bin_generator(self):
        """DRIFT GUARD: the declared canonical set must equal what _pose_bin
        emits when swept over the operational ±90° head-pose range. If the
        binning cascade changes, this fails loudly instead of silently
        corrupting the coverage denominator.
        """
        produced = set()
        produced.add(_pose_bin((0.0, 0.0, 0.0)))  # frontal
        # band centers 15,25,...,95 land squarely inside bands 10,20,...,90
        for c in range(15, 100, 10):
            produced.add(_pose_bin((float(c), 0.0, 0.0)))    # R*
            produced.add(_pose_bin((float(-c), 0.0, 0.0)))   # L*
            produced.add(_pose_bin((0.0, float(c), 0.0)))    # U*
            produced.add(_pose_bin((0.0, float(-c), 0.0)))   # D*
        assert canonical_pose_bins() == produced

    def test_returns_a_set_copy_not_shared_mutable(self):
        """Callers must not be able to mutate the canonical denominator."""
        a = canonical_pose_bins()
        a.add('JUNK')
        assert 'JUNK' not in canonical_pose_bins()


# ─── PatchMemory.coverage_pose(): real observed/total ratio on the real type ──

def _quality_map(value: float, size: int = 64) -> np.ndarray:
    return np.full((size, size), value, dtype=np.float32)


def _canonical_face(size: int = 64) -> np.ndarray:
    return np.ones((size, size, 3), dtype=np.float32) * 0.5


class TestCoveragePose:
    def test_uninitialized_memory_is_zero(self):
        pm = PatchMemory()
        assert pm.coverage_pose() == 0.0

    def test_frontal_only_is_one_over_total(self):
        """A frontally-enrolled face has observed exactly {'F'} ⇒ 1/37."""
        pm = PatchMemory()
        face = _canonical_face()
        pm.initialize(face, _quality_map(0.5))
        # update with a frontal pose and higher quality so the 'F' bin registers
        pm.update(face, _quality_map(0.9), pose=(0.0, 0.0, 0.0))
        assert pm.coverage_pose() == pytest.approx(1.0 / 37.0, abs=1e-9)

    def test_more_distinct_poses_increase_coverage(self):
        """Observing F, a left bin, and a right bin ⇒ 3/37 (union of bins).

        Quality must strictly increase per pose so each new patch beats the
        per-region best_quality gate (patch_memory.py:153) and registers its
        bin — this mirrors real enrollment where better observations replace.
        """
        pm = PatchMemory()
        face = _canonical_face()
        pm.initialize(face, _quality_map(0.3))
        pm.update(face, _quality_map(0.5), pose=(0.0, 0.0, 0.0))    # F
        pm.update(face, _quality_map(0.7), pose=(-30.0, 0.0, 0.0))  # L30
        pm.update(face, _quality_map(0.9), pose=(40.0, 0.0, 0.0))   # R40
        assert pm.coverage_pose() == pytest.approx(3.0 / 37.0, abs=1e-9)

    def test_coverage_is_union_across_regions(self):
        """Coverage counts the UNION of observed bins across all regions, not a
        per-region count — different regions may capture different poses."""
        pm = PatchMemory()
        face = _canonical_face()
        pm.initialize(face, _quality_map(0.3))
        pm.update(face, _quality_map(0.6), pose=(0.0, 0.0, 0.0))   # F
        pm.update(face, _quality_map(0.8), pose=(20.0, 0.0, 0.0))  # R20
        observed = pm.observed_pose_bins()
        assert observed == {'F', 'R20'}
        assert pm.coverage_pose() == pytest.approx(2.0 / 37.0, abs=1e-9)

    def test_coverage_bounded_unit_interval(self):
        pm = PatchMemory()
        face = _canonical_face()
        pm.initialize(face, _quality_map(0.3))
        q = 0.4
        for yaw in range(-90, 100, 10):
            pm.update(face, _quality_map(q), pose=(float(yaw), 0.0, 0.0))
            q += 0.02
        cov = pm.coverage_pose()
        assert 0.0 <= cov <= 1.0

    def test_out_of_range_bins_do_not_inflate_coverage(self):
        """A bin _pose_bin can emit beyond the canonical range (e.g. R100 at
        yaw=105) must not count — coverage = |observed ∩ canonical| / |canonical|
        so the ratio can never exceed 1."""
        pm = PatchMemory()
        face = _canonical_face()
        pm.initialize(face, _quality_map(0.3))
        pm.update(face, _quality_map(0.6), pose=(0.0, 0.0, 0.0))    # F (in range)
        pm.update(face, _quality_map(0.9), pose=(105.0, 0.0, 0.0))  # R100 (out)
        assert 'R100' in pm.observed_pose_bins()  # it WAS observed
        # but only F counts toward coverage
        assert pm.coverage_pose() == pytest.approx(1.0 / 37.0, abs=1e-9)


# ─── apply_pose_coverage(): the §16.7 cap (== §16.8 pose factor) ──────────────

class TestApplyPoseCoverage:
    def test_full_coverage_leaves_confidence_unchanged(self):
        assert apply_pose_coverage(0.8, 1.0) == pytest.approx(0.8, abs=1e-9)

    def test_zero_coverage_zeros_confidence(self):
        assert apply_pose_coverage(0.8, 0.0) == 0.0

    def test_never_exceeds_input_confidence(self):
        """§16.8 invariant: C_recon ≤ C_obs (coverage can only REDUCE trust)."""
        for c in (0.0, 0.2, 0.5, 0.9, 1.0):
            for cov in (0.0, 0.1, 0.5, 0.999, 1.0):
                assert apply_pose_coverage(c, cov) <= c + 1e-12

    def test_monotonic_non_decreasing_in_coverage(self):
        c = 0.7
        prev = -1.0
        for cov in (0.0, 0.25, 0.5, 0.75, 1.0):
            val = apply_pose_coverage(c, cov)
            assert val >= prev
            prev = val

    def test_low_coverage_caps_below_high_coverage_ceiling_regardless_of_quality(self):
        """arch §16.7 REQUIRED test. A high-quality observation seen at low
        pose coverage must score BELOW a lower-quality observation seen at full
        coverage. Quality alone cannot buy trust the coverage hasn't earned.
        """
        high_quality_low_coverage = apply_pose_coverage(0.95, 1.0 / 37.0)
        low_quality_full_coverage = apply_pose_coverage(0.30, 1.0)
        assert high_quality_low_coverage < low_quality_full_coverage

    def test_clamps_inputs_to_unit_interval(self):
        """Defensive: out-of-range coverage/confidence clamp, never explode."""
        assert apply_pose_coverage(1.5, 1.0) == pytest.approx(1.0, abs=1e-9)
        assert apply_pose_coverage(0.5, 1.5) == pytest.approx(0.5, abs=1e-9)
        assert apply_pose_coverage(-0.2, 0.5) == 0.0
        assert apply_pose_coverage(0.5, -0.2) == 0.0
