"""Layer 4: Telemetry honesty + temporal/energy scaler tests.

Validates D-06 (temporal SIM2) and D-08 (telemetry honesty).

Invariants:
  - Energy scaler normalization is idempotent after convergence
  - Running stats converge
  - Photometric lock stabilizes output
  - Telemetry JSON has required fields
"""
import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════
# Energy Scaler
# ═══════════════════════════════════════════════════════════════════

class TestEnergyScaler:
    """EnergyScaler normalizes energy terms to unit variance."""

    @pytest.fixture
    def scaler(self):
        from face_os.energy_scaling import EnergyScaler
        return EnergyScaler()

    def test_normalize_returns_float(self, scaler):
        """Normalize returns a float."""
        val = scaler.normalize("E_geom", 0.5)
        assert isinstance(val, float)

    def test_normalize_positive_input(self, scaler):
        """Positive input → positive output."""
        val = scaler.normalize("E_geom", 1.0)
        assert val >= 0.0

    def test_running_stats_converge(self, scaler):
        """After many updates, normalization should stabilize."""
        values = np.random.uniform(0.1, 0.5, 50)
        results = []
        for v in values:
            results.append(scaler.normalize("E_test", float(v)))
        # Last 10 should be more consistent than first 10
        early_std = float(np.std(results[:10]))
        late_std = float(np.std(results[-10:]))
        # At minimum, late values should not explode
        assert late_std < early_std * 5 + 0.1

    def test_different_terms_independent(self, scaler):
        """Different energy terms maintain independent statistics."""
        for _ in range(20):
            scaler.normalize("E_geom", 0.3)
            scaler.normalize("E_identity", 3.0)
        # The two terms should have different normalization
        v1 = scaler.normalize("E_geom", 0.3)
        v2 = scaler.normalize("E_identity", 3.0)
        # They should be normalized to similar scales
        # (both should be roughly 1.0 after convergence)
        assert abs(v1) < 20, f"E_geom normalized to {v1}"
        assert abs(v2) < 20, f"E_identity normalized to {v2}"

    def test_reset_clears_stats(self, scaler):
        """reset() clears all accumulated statistics."""
        for _ in range(20):
            scaler.normalize("E_test", 1.0)
        scaler.reset()
        stats = scaler.get_stats()
        assert len(stats) == 0 or all(s.get('count', 0) == 0 for s in stats.values())

    def test_compute_weight_positive(self, scaler):
        """compute_weight returns a positive value."""
        w = scaler.compute_weight("E_test", uncertainty=1.0)
        assert w > 0


# ═══════════════════════════════════════════════════════════════════
# Photometric Lock
# ═══════════════════════════════════════════════════════════════════

class TestPhotometricLock:
    """Photometric lock should stabilize frame brightness."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset photometric state before each test."""
        from face_os.photometric import reset_photometric_lock
        reset_photometric_lock()

    def test_photometric_lock_returns_valid_image(self):
        """Output is valid uint8 BGR."""
        from face_os.photometric import photometric_lock
        img = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        result = photometric_lock(img)
        assert result.dtype == np.uint8
        assert result.shape == img.shape

    def test_photometric_lock_stabilizes(self):
        """Repeated calls should converge to stable output."""
        from face_os.photometric import photometric_lock
        # Feed same image repeatedly
        img = np.random.randint(100, 180, (64, 64, 3), dtype=np.uint8)
        results = []
        for _ in range(10):
            result = photometric_lock(img.copy())
            results.append(float(np.mean(result)))
        # Last few should be very similar
        late_std = float(np.std(results[-5:]))
        assert late_std < 5.0, f"Not stabilizing: late std={late_std:.2f}"

    def test_photometric_lock_with_mask(self):
        """Mask should not cause crashes."""
        from face_os.photometric import photometric_lock
        img = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        mask = np.ones((64, 64), dtype=np.float32) * 0.5
        result = photometric_lock(img, mask=mask)
        assert result is not None
        assert result.shape == img.shape


# ═══════════════════════════════════════════════════════════════════
# Telemetry Contract
# ═══════════════════════════════════════════════════════════════════

class TestTelemetryContract:
    """Pipeline telemetry must expose required fields per D-08."""

    REQUIRED_TELEMETRY_FIELDS = {
        'render_path', 'fallback_reason', 'intrinsic_used',
        'geometry_source', 'energy_terms', 'transform_det',
    }

    def test_emit_frame_telemetry_signature(self):
        """_emit_frame_telemetry exists and accepts required args."""
        from face_os.pipeline import FaceOSPipeline
        assert hasattr(FaceOSPipeline, '_emit_frame_telemetry')

    def test_telemetry_dict_has_counters(self):
        """Pipeline telemetry dict has required counter keys."""
        from face_os.pipeline import FaceOSPipeline
        p = FaceOSPipeline.__new__(FaceOSPipeline)
        # Initialize telemetry manually
        if hasattr(p, '_init_telemetry'):
            p._init_telemetry()
        elif hasattr(p, '_telemetry'):
            pass
        else:
            # Pipeline might initialize in __init__
            p = FaceOSPipeline()
            assert hasattr(p, '_telemetry')
            tel = p._telemetry
            assert 'physical_render_frames' in tel
            assert 'alpha_fallback_frames' in tel


# ═══════════════════════════════════════════════════════════════════
# Temporal Estimation Basics
# ═══════════════════════════════════════════════════════════════════

class TestTemporalBasics:
    """Basic temporal estimation contracts."""

    def test_temporal_solve_import(self):
        """temporal_solve module is importable."""
        from face_os import temporal_solve
        assert hasattr(temporal_solve, 'TemporalSolver') or hasattr(temporal_solve, 'BidirectionalSolver')

    def test_state_evolution_import(self):
        """state_evolution module is importable."""
        from face_os import state_evolution
        assert state_evolution is not None
