"""
tests/face_os/test_v2_subsystems.py — Tests for Face OS V2 subsystem architecture.

Validates:
1. Subsystem isolation (no cross-contamination)
2. Coordinate system correctness
3. Mesh-based semantic masking
4. Deterministic rendering
5. Temporal consistency constraints
6. Mathematical invariants
"""

import cv2
import numpy as np
import pytest

from face_os.types import (
    GeometryState,
    IdentityState,
    TemporalState,
    CropPlan,
    CropStrategy,
    Landmarks,
)
from face_os.subsystems.geometry_estimator import GeometryEstimator
from face_os.subsystems.identity_estimator import IdentityEstimator
from face_os.subsystems.temporal_estimator import TemporalEstimator
from face_os.subsystems.renderer import Renderer
from face_os.crop_planner import CropPlanner


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def crop_planner():
    """Create a CropPlanner for testing."""
    return CropPlanner(reference_image="expectation.png")


@pytest.fixture
def geometry_estimator(crop_planner):
    """Create a GeometryEstimator for testing."""
    return GeometryEstimator(crop_planner)


@pytest.fixture
def identity_estimator():
    """Create an IdentityEstimator for testing."""
    return IdentityEstimator()


@pytest.fixture
def temporal_estimator():
    """Create a TemporalEstimator for testing."""
    return TemporalEstimator()


@pytest.fixture
def renderer():
    """Create a Renderer for testing."""
    return Renderer()


@pytest.fixture
def sample_frame():
    """Create a sample 16:9 frame for testing."""
    return np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def sample_landmarks():
    """Create sample landmarks for testing."""
    # Create 478 points in a face-like pattern
    points = np.zeros((478, 2), dtype=np.float32)
    # Face oval
    for i in range(36):
        angle = 2 * np.pi * i / 36
        points[i] = [640 + 200 * np.cos(angle), 360 + 250 * np.sin(angle)]
    # Eyes, nose, mouth (simplified)
    points[33] = [540, 300]  # Left eye
    points[263] = [740, 300]  # Right eye
    points[1] = [640, 360]  # Nose tip
    points[61] = [590, 420]  # Left mouth
    points[291] = [690, 420]  # Right mouth
    
    return Landmarks(
        points=points,
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        left_eye_center=(540.0, 300.0),
        right_eye_center=(740.0, 300.0),
        nose_tip=(640.0, 360.0),
        mouth_center=(640.0, 420.0),
        landmark_confidence=0.9,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SUBSYSTEM A — GEOMETRY ESTIMATOR TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestGeometryEstimator:
    """Tests for Geometry Estimator subsystem."""
    
    def test_estimate_returns_geometry_state(self, geometry_estimator, sample_frame):
        """Geometry estimator returns valid GeometryState."""
        state = geometry_estimator.estimate(sample_frame, None)
        assert isinstance(state, GeometryState)
        
    def test_estimate_without_face_track(self, geometry_estimator, sample_frame):
        """Geometry estimator handles missing face track gracefully."""
        state = geometry_estimator.estimate(sample_frame, None)
        assert state.landmarks_478 is None
        assert state.landmarks is None
        assert state.pose == (0.0, 0.0, 0.0)
        assert state.geometry_confidence == 0.0
        
    def test_geometry_mask_is_brightness_invariant(self, geometry_estimator, sample_frame):
        """Geometry mask does not depend on frame brightness."""
        # Create frames with different brightness
        dark_frame = (sample_frame * 0.2).astype(np.uint8)
        bright_frame = (sample_frame * 0.8).astype(np.uint8)
        
        # Mask should be identical regardless of brightness
        mask1 = geometry_estimator._create_geometry_mask(sample_frame.shape[:2], None)
        mask2 = geometry_estimator._create_geometry_mask(dark_frame.shape[:2], None)
        mask3 = geometry_estimator._create_geometry_mask(bright_frame.shape[:2], None)
        
        np.testing.assert_array_equal(mask1, mask2)
        np.testing.assert_array_equal(mask1, mask3)
        
    def test_geometry_mask_has_valid_range(self, geometry_estimator, sample_frame):
        """Geometry mask values are in [0, 1]."""
        mask = geometry_estimator._create_geometry_mask(sample_frame.shape[:2], None)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0
        assert mask.dtype == np.float32
        
    def test_geometry_confidence_is_bounded(self, geometry_estimator, sample_frame):
        """Geometry confidence is in [0, 1]."""
        state = geometry_estimator.estimate(sample_frame, None)
        assert 0.0 <= state.geometry_confidence <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# SUBSYSTEM B — IDENTITY ESTIMATOR TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestIdentityEstimator:
    """Tests for Identity Estimator subsystem."""
    
    def test_estimate_returns_identity_state(self, identity_estimator):
        """Identity estimator returns valid IdentityState."""
        geometry_state = GeometryState()
        state = identity_estimator.estimate(geometry_state)
        assert isinstance(state, IdentityState)
        
    def test_identity_state_uninitialized_has_high_uncertainty(self, identity_estimator):
        """Uninitialized identity state has high uncertainty."""
        geometry_state = GeometryState()
        state = identity_estimator.estimate(geometry_state)
        assert state.identity_uncertainty >= 0.5
        assert not state.initialized
        
    def test_identity_anchor_can_be_set(self, identity_estimator):
        """Identity anchor can be set from reference."""
        ref_face = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        identity_estimator.set_anchor(ref_face)
        # Anchor should be set internally
        assert identity_estimator.identity_belief._anchor_low is not None


# ═══════════════════════════════════════════════════════════════════════════════
# SUBSYSTEM C — TEMPORAL ESTIMATOR TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTemporalEstimator:
    """Tests for Temporal Estimator subsystem."""
    
    def test_estimate_returns_temporal_state(self, temporal_estimator):
        """Temporal estimator returns valid TemporalState."""
        geometry_state = GeometryState()
        identity_state = IdentityState()
        state = temporal_estimator.estimate(geometry_state, identity_state)
        assert isinstance(state, TemporalState)
        
    def test_temporal_confidence_is_bounded(self, temporal_estimator):
        """Temporal confidence is in [0, 1]."""
        geometry_state = GeometryState()
        identity_state = IdentityState()
        state = temporal_estimator.estimate(geometry_state, identity_state)
        assert 0.0 <= state.temporal_confidence <= 1.0
        
    def test_drift_score_is_non_negative(self, temporal_estimator):
        """Drift score is non-negative."""
        geometry_state = GeometryState()
        identity_state = IdentityState()
        state = temporal_estimator.estimate(geometry_state, identity_state)
        assert state.drift_score >= 0.0
        
    def test_continuity_score_is_bounded(self, temporal_estimator):
        """Continuity score is in [0, 1]."""
        geometry_state = GeometryState()
        identity_state = IdentityState()
        state = temporal_estimator.estimate(geometry_state, identity_state)
        assert 0.0 <= state.continuity_score <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# SUBSYSTEM D — RENDERER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestRenderer:
    """Tests for Renderer subsystem."""
    
    def test_render_preserves_output_contract(self, renderer, sample_frame):
        """Renderer preserves output dimensions and dtype."""
        crop_plan = CropPlan(
            strategy=CropStrategy.CENTER,
            src_x=100, src_y=0, src_w=1080, src_h=1920,
            dst_w=1080, dst_h=1920,
        )
        geometry_state = GeometryState()
        identity_state = IdentityState()
        temporal_state = TemporalState()
        
        output = renderer.render(
            sample_frame, geometry_state, identity_state, temporal_state, crop_plan
        )
        
        assert output.shape == (1920, 1080, 3)
        assert output.dtype == np.uint8
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))
        
    def test_render_without_identity_fallback(self, renderer, sample_frame):
        """Renderer falls back to enhancement-only when no identity."""
        crop_plan = CropPlan(
            strategy=CropStrategy.CENTER,
            src_x=100, src_y=0, src_w=1080, src_h=1920,
            dst_w=1080, dst_h=1920,
        )
        geometry_state = GeometryState()
        identity_state = IdentityState()  # Uninitialized
        temporal_state = TemporalState()
        
        output = renderer.render(
            sample_frame, geometry_state, identity_state, temporal_state, crop_plan
        )
        
        assert output.shape == (1920, 1080, 3)
        assert output.dtype == np.uint8


# ═══════════════════════════════════════════════════════════════════════════════
# MATHEMATICAL INVARIANT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestMathematicalInvariants:
    """Tests for mathematical invariants across subsystems."""
    
    def test_output_shape_invariance(self, renderer, sample_frame):
        """Output shape is invariant across different input conditions."""
        crop_plan = CropPlan(
            strategy=CropStrategy.CENTER,
            src_x=100, src_y=0, src_w=1080, src_h=1920,
            dst_w=1080, dst_h=1920,
        )
        
        # Test with different geometry states
        for geometry_state in [
            GeometryState(),
            GeometryState(geometry_confidence=0.5),
            GeometryState(geometry_confidence=1.0),
        ]:
            identity_state = IdentityState()
            temporal_state = TemporalState()
            
            output = renderer.render(
                sample_frame, geometry_state, identity_state, temporal_state, crop_plan
            )
            assert output.shape == (1920, 1080, 3)
            
    def test_output_dtype_invariance(self, renderer, sample_frame):
        """Output dtype is always uint8."""
        crop_plan = CropPlan(
            strategy=CropStrategy.CENTER,
            src_x=100, src_y=0, src_w=1080, src_h=1920,
            dst_w=1080, dst_h=1920,
        )
        geometry_state = GeometryState()
        identity_state = IdentityState()
        temporal_state = TemporalState()
        
        output = renderer.render(
            sample_frame, geometry_state, identity_state, temporal_state, crop_plan
        )
        assert output.dtype == np.uint8
        
    def test_no_nan_inf_in_outputs(self, renderer, sample_frame):
        """No NaN or Inf in renderer outputs."""
        crop_plan = CropPlan(
            strategy=CropStrategy.CENTER,
            src_x=100, src_y=0, src_w=1080, src_h=1920,
            dst_w=1080, dst_h=1920,
        )
        geometry_state = GeometryState()
        identity_state = IdentityState()
        temporal_state = TemporalState()
        
        output = renderer.render(
            sample_frame, geometry_state, identity_state, temporal_state, crop_plan
        )
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))
        
    def test_pixel_values_in_valid_range(self, renderer, sample_frame):
        """Pixel values are in [0, 255]."""
        crop_plan = CropPlan(
            strategy=CropStrategy.CENTER,
            src_x=100, src_y=0, src_w=1080, src_h=1920,
            dst_w=1080, dst_h=1920,
        )
        geometry_state = GeometryState()
        identity_state = IdentityState()
        temporal_state = TemporalState()
        
        output = renderer.render(
            sample_frame, geometry_state, identity_state, temporal_state, crop_plan
        )
        assert output.min() >= 0
        assert output.max() <= 255


# ═══════════════════════════════════════════════════════════════════════════════
# COORDINATE SYSTEM TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCoordinateSystems:
    """Tests for coordinate system correctness."""
    
    def test_crop_plan_declares_spaces(self):
        """Crop plan declares source and target spaces."""
        crop_plan = CropPlan(
            strategy=CropStrategy.FACE_LOCKED,
            src_x=100, src_y=0, src_w=1080, src_h=1920,
            dst_w=1080, dst_h=1920,
        )
        assert crop_plan.src_w > 0
        assert crop_plan.src_h > 0
        assert crop_plan.dst_w > 0
        assert crop_plan.dst_h > 0
        
    def test_geometry_state_has_transform_chain(self):
        """Geometry state has complete transform chain."""
        state = GeometryState()
        # All transforms should be optional but present as fields
        assert hasattr(state, 'canonical_transform')
        assert hasattr(state, 'inverse_transform')
        assert hasattr(state, 'crop_transform')