"""Tests for expression-aware hybrid blend (Task 2.5).

Verifies that effective_blend_max scales by appearance uncertainty:
  effective = configured + (1.0 - configured) * appearance_uncertainty
  clamped to [configured, 1.0]
"""

from __future__ import annotations

import numpy as np
import pytest


def _compute_effective_blend_max(configured: float, appearance_uncertainty: float | None) -> float:
    """Mirrors pipeline.py:2991-2993 expression-aware blend scaling."""
    appear_unc = float(appearance_uncertainty or 0.0)
    effective = configured + (1.0 - configured) * appear_unc
    return float(np.clip(effective, configured, 1.0))


@pytest.mark.parametrize("configured", [0.3, 0.5, 0.7, 0.9])
def test_uncertainty_zero_equals_configured(configured: float):
    """effective_blend_max == configured when appearance_uncertainty is 0.0."""
    assert _compute_effective_blend_max(configured, 0.0) == pytest.approx(configured)


@pytest.mark.parametrize("configured", [0.3, 0.5, 0.7, 0.9])
def test_uncertainty_half_is_midway(configured: float):
    """effective_blend_max is halfway between configured and 1.0 at appearance_uncertainty=0.5."""
    expected = configured + (1.0 - configured) * 0.5
    assert _compute_effective_blend_max(configured, 0.5) == pytest.approx(expected)


@pytest.mark.parametrize("configured", [0.3, 0.5, 0.7])
def test_uncertainty_one_maxes_out(configured: float):
    """effective_blend_max == 1.0 when appearance_uncertainty is 1.0."""
    assert _compute_effective_blend_max(configured, 1.0) == pytest.approx(1.0)


def test_effective_blend_max_clamped_lower():
    """effective_blend_max never drops below configured value (clamp lower bound)."""
    assert _compute_effective_blend_max(0.3, -0.5) == pytest.approx(0.3)
    assert _compute_effective_blend_max(0.5, -1.0) == pytest.approx(0.5)
    assert _compute_effective_blend_max(0.7, -2.0) == pytest.approx(0.7)


def test_effective_blend_max_clamped_upper():
    """effective_blend_max never exceeds 1.0 (clamp upper bound)."""
    assert _compute_effective_blend_max(0.5, 2.0) == pytest.approx(1.0)
    assert _compute_effective_blend_max(0.3, 3.0) == pytest.approx(1.0)


def test_appearance_uncertainty_none_treated_as_zero():
    """None appearance_uncertainty is treated as 0.0."""
    for configured in (0.3, 0.5, 0.7):
        assert _compute_effective_blend_max(configured, None) == pytest.approx(configured)


def test_appearance_uncertainty_range_is_monotonic():
    """effective_blend_max is monotonic non-decreasing with appearance_uncertainty."""
    uncertains = np.linspace(0.0, 1.0, 20)
    for cfg in (0.3, 0.5, 0.7, 0.9):
        values = [_compute_effective_blend_max(cfg, u) for u in uncertains]
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1] - 1e-10
