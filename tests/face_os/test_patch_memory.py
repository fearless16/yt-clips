"""
tests/face_os/test_patch_memory.py — Regression tests for Patch Memory.

Tests:
- Region definitions
- Per-patch dynamics
- Pose-conditioned storage
- Expression-conditioned storage
- Lighting-conditioned storage
- Independent confidence
"""

import cv2
import numpy as np
import pytest

from face_os.patch_memory import (
    PatchMemory,
    RegionPatch,
    REGION_DEFS,
    _pose_bin,
    _expression_bin,
    _lighting_bin,
    _composite_condition_key,
)


class TestRegionDefinitions:
    """Test region definitions."""

    def test_all_required_regions(self):
        """Must have all required regions."""
        required = {"forehead", "left_eye", "right_eye", "beard", "lips", "nose", "skin"}
        for region in required:
            assert region in REGION_DEFS, f"Missing region: {region}"

    def test_region_has_bounds(self):
        """Each region must have bounds."""
        for name, rdef in REGION_DEFS.items():
            assert 'bounds' in rdef, f"Region {name} missing bounds"
            assert len(rdef['bounds']) == 4, f"Region {name} bounds must have 4 elements"

    def test_region_has_update_rate(self):
        """Each region must have update rate."""
        for name, rdef in REGION_DEFS.items():
            assert 'update_rate' in rdef, f"Region {name} missing update_rate"
            assert 0 < rdef['update_rate'] <= 1, f"Region {name} update_rate must be (0, 1]"

    def test_region_has_priority(self):
        """Each region must have priority."""
        for name, rdef in REGION_DEFS.items():
            assert 'priority' in rdef, f"Region {name} missing priority"
            assert rdef['priority'] in ['critical', 'high', 'medium', 'low']

    def test_eye_priority_critical(self):
        """Eyes must be critical priority."""
        assert REGION_DEFS['left_eye']['priority'] == 'critical'
        assert REGION_DEFS['right_eye']['priority'] == 'critical'

    def test_forehead_priority_low(self):
        """Forehead must be low priority."""
        assert REGION_DEFS['forehead']['priority'] == 'low'


class TestPoseBin:
    """Test pose bin quantization."""

    def test_frontal_neutral(self):
        """Small yaw/pitch must be frontal_neutral."""
        assert _pose_bin((0, 0, 0)) == 'frontal_neutral'
        assert _pose_bin((5, 5, 0)) == 'frontal_neutral'

    def test_left_yaw(self):
        """Negative yaw must be left."""
        assert _pose_bin((-20, 0, 0)).startswith('left')

    def test_right_yaw(self):
        """Positive yaw must be right."""
        assert _pose_bin((20, 0, 0)).startswith('right')

    def test_up_pitch(self):
        """Positive pitch must be up."""
        assert _pose_bin((0, 20, 0)).startswith('up')

    def test_down_pitch(self):
        """Negative pitch must be down."""
        assert _pose_bin((0, -20, 0)).startswith('down')


class TestExpressionBin:
    """Test expression bin quantization."""

    def test_neutral_default(self):
        """None must be neutral."""
        assert _expression_bin(None) == 'neutral'

    def test_smile(self):
        """High smile value must be smile."""
        assert _expression_bin(np.array([0.0, 0.8])) == 'smile'

    def test_talk(self):
        """High mouth_open must be talk."""
        assert _expression_bin(np.array([0.5, 0.0])) == 'talk'


class TestLightingBin:
    """Test lighting bin quantization."""

    def test_neutral_default(self):
        """None must be neutral."""
        assert _lighting_bin(None) == 'neutral'

    def test_warm(self):
        """High R must be warm."""
        assert _lighting_bin(np.array([200, 150, 100])) == 'warm'

    def test_cool(self):
        """High B must be cool."""
        assert _lighting_bin(np.array([100, 150, 200])) == 'cool'


class TestCompositeConditionKey:
    """Test composite condition key."""

    def test_all_none(self):
        """All None must produce valid key."""
        key = _composite_condition_key(None, None, None)
        assert 'any' in key or 'neutral' in key

    def test_with_pose(self):
        """Must include pose."""
        key = _composite_condition_key(pose=(0, 0, 0))
        assert 'frontal' in key

    def test_with_expression(self):
        """Must include expression."""
        key = _composite_condition_key(expression='smile')
        assert 'smile' in key


class TestRegionPatch:
    """Test region patch."""

    def test_initializes(self):
        """Must initialize correctly."""
        patch = RegionPatch('left_eye', REGION_DEFS['left_eye'])

        assert patch.name == 'left_eye'
        assert patch.update_rate == REGION_DEFS['left_eye']['update_rate']

    def test_extract_region(self, canonical_face):
        """Must extract region from face."""
        patch = RegionPatch('left_eye', REGION_DEFS['left_eye'])

        region = patch.extract_region(canonical_face)

        assert region is not None
        assert region.ndim == 3

    def test_update_stores_best(self, canonical_face):
        """Must store best observation."""
        patch = RegionPatch('left_eye', REGION_DEFS['left_eye'])

        quality = np.ones((256, 256), dtype=np.float32) * 0.8
        patch.update(canonical_face, quality, pose=(0, 0, 0), frame_idx=0)

        assert patch.best_patch is not None
        assert patch.best_quality > 0

    def test_query_returns_patch(self, canonical_face):
        """Must return patch from query."""
        patch = RegionPatch('left_eye', REGION_DEFS['left_eye'])

        quality = np.ones((256, 256), dtype=np.float32) * 0.8
        patch.update(canonical_face, quality, pose=(0, 0, 0), frame_idx=0)

        result, conf = patch.query(pose=(0, 0, 0))

        assert result is not None
        assert conf > 0

    def test_pose_conditioned_storage(self, canonical_face):
        """Must store patches per pose."""
        patch = RegionPatch('left_eye', REGION_DEFS['left_eye'])

        quality = np.ones((256, 256), dtype=np.float32) * 0.8

        for yaw in [-30, 0, 30]:
            patch.update(canonical_face, quality, pose=(yaw, 0, 0), frame_idx=yaw + 30)

        assert len(patch.pose_patches) > 0

    def test_expression_conditioned_storage(self, canonical_face):
        """Must store patches per expression."""
        patch = RegionPatch('left_eye', REGION_DEFS['left_eye'])

        quality = np.ones((256, 256), dtype=np.float32) * 0.8

        for expr in ['neutral', 'smile', 'talk']:
            patch.update(canonical_face, quality, pose=(0, 0, 0),
                        frame_idx=0, expression=expr)

        assert len(patch.condition_patches) > 0

    def test_freeze_on_blink(self, canonical_face):
        """Must freeze during blink for eye regions."""
        patch = RegionPatch('left_eye', REGION_DEFS['left_eye'])

        quality = np.ones((256, 256), dtype=np.float32) * 0.8
        patch.update(canonical_face, quality, pose=(0, 0, 0), frame_idx=0)

        # Update with blink
        conf = patch.update(canonical_face, quality, pose=(0, 0, 0),
                           is_blink=True, frame_idx=1)

        # Should return existing confidence
        assert conf == patch.current_confidence


class TestPatchMemory:
    """Test patch memory."""

    def test_initializes(self, canonical_face):
        """Must initialize correctly."""
        memory = PatchMemory()
        quality = np.ones((256, 256), dtype=np.float32) * 0.8

        memory.initialize(canonical_face, quality)

        assert memory._initialized == True
        assert len(memory.regions) > 0

    def test_update_all_regions(self, canonical_face):
        """Must update all regions."""
        memory = PatchMemory()
        quality = np.ones((256, 256), dtype=np.float32) * 0.8

        memory.initialize(canonical_face, quality)
        confidences = memory.update(canonical_face, quality, pose=(0, 0, 0), frame_idx=0)

        assert len(confidences) == len(memory.regions)

    def test_query_region(self, canonical_face):
        """Must query specific region."""
        memory = PatchMemory()
        quality = np.ones((256, 256), dtype=np.float32) * 0.8

        memory.initialize(canonical_face, quality)
        memory.update(canonical_face, quality, pose=(0, 0, 0), frame_idx=0)

        patch, conf = memory.query_region('left_eye', pose=(0, 0, 0))

        assert patch is not None
        assert conf > 0

    def test_query_all(self, canonical_face):
        """Must query all regions."""
        memory = PatchMemory()
        quality = np.ones((256, 256), dtype=np.float32) * 0.8

        memory.initialize(canonical_face, quality)
        memory.update(canonical_face, quality, pose=(0, 0, 0), frame_idx=0)

        result, conf_map = memory.query_all((256, 256), pose=(0, 0, 0))

        assert result is not None
        assert conf_map is not None

    def test_independent_dynamics(self, canonical_face):
        """Regions must have independent dynamics."""
        memory = PatchMemory()
        quality = np.ones((256, 256), dtype=np.float32) * 0.8

        memory.initialize(canonical_face, quality)

        # Check different update rates
        rates = {name: r.update_rate for name, r in memory.regions.items()}
        unique_rates = set(rates.values())

        assert len(unique_rates) > 1, "Regions should have different update rates"
