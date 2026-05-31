"""TDD tests for D-05 Task 4.4: color-cast compensation with reject-on-failure.

Tests verify:
1. Color-cast compensation removes teal/green bias
2. Reject-on-failure: if compensation would increase LAB drift from anchor, skip it
3. Compensation is stable (EMA smoothing prevents flicker)
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from face_os.identity_state import IdentityState


def _make_state():
    """Create a minimal IdentityState for testing color-cast compensation."""
    state = IdentityState.__new__(IdentityState)
    state._wb_scale_ema = np.ones(3, dtype=np.float32)
    state._wb_ema_rate = 0.15
    state._anchor_lab = None
    state._anchor_albedo = None
    return state


class TestColorCastCompensation:
    def test_removes_teal_bias(self):
        """Teal-shifted albedo (high B/G, low R) should be corrected toward neutral."""
        state = _make_state()
        # Teal albedo: blue-green biased
        albedo = np.zeros((10, 10, 3), dtype=np.float32)
        albedo[:, :, 0] = 0.3  # R low
        albedo[:, :, 1] = 0.6  # G medium
        albedo[:, :, 2] = 0.7  # B high

        corrected = state._compensate_color_cast(albedo)
        mean_per_ch = np.mean(corrected, axis=(0, 1))
        # After correction, channels should be closer to equal
        ch_std = np.std(mean_per_ch)
        orig_std = np.std(np.mean(albedo, axis=(0, 1)))
        assert ch_std < orig_std

    def test_rejects_worse_correction(self):
        """If compensation increases LAB drift from anchor, reject it."""
        state = _make_state()
        state._wb_ema_rate = 0.01  # slow EMA so pre-set value persists

        # Anchor: neutral gray in LAB
        state._anchor_lab = np.full((10, 10, 3), 50.0, dtype=np.float32)
        state._anchor_lab[:, :, 1] = 0.0
        state._anchor_lab[:, :, 2] = 0.0

        # Pre-set EMA to a scale that would COOL the albedo (increase B, decrease R)
        # This would push albedo AWAY from neutral anchor
        state._wb_scale_ema = np.array([0.7, 1.0, 1.3], dtype=np.float32)

        # Albedo nearly neutral
        albedo = np.full((10, 10, 3), 0.5, dtype=np.float32)

        corrected = state._compensate_color_cast(albedo)
        # The pre-set EMA would shift albedo to (0.35, 0.5, 0.65) — MORE drift from neutral
        # So correction should be rejected → return original
        np.testing.assert_array_equal(corrected, albedo)

    def test_stable_under_repeated_calls(self):
        """EMA smoothing should prevent flicker between frames."""
        state = _make_state()
        albedo = np.full((10, 10, 3), 0.5, dtype=np.float32)
        albedo[:, :, 0] = 0.4  # slight red shift

        results = []
        for _ in range(10):
            results.append(state._compensate_color_cast(albedo.copy()))

        # Last few should converge (stable)
        diff = np.mean(np.abs(results[-1] - results[-2]))
        assert diff < 0.01

    def test_no_anchor_uses_gray_world(self):
        """Without anchor, use gray-world correction."""
        state = _make_state()
        state._anchor_lab = None

        albedo = np.zeros((10, 10, 3), dtype=np.float32)
        albedo[:, :, 0] = 0.3
        albedo[:, :, 1] = 0.5
        albedo[:, :, 2] = 0.7

        # Run enough iterations for EMA convergence
        for _ in range(100):
            corrected = state._compensate_color_cast(albedo.copy())

        mean_per_ch = np.mean(corrected, axis=(0, 1))
        # Gray-world: channels should converge toward equal
        assert np.std(mean_per_ch) < 0.1
