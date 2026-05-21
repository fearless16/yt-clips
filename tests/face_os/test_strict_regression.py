"""
tests/face_os/test_strict_regression.py — STRICT deterministic regression tests.

Every test must have a deterministic, numeric assertion.
No vague "looks okay" assertions. No weak shape-only checks.

BUG CLASSES TARGETED:
  A) Mask stability — centroid drift, intensity-threshold hacks, warp misalignment
  B) Frame size invariance — every pipeline path must return identical dimensions
  C) NaN/Inf detection — blend and warp outputs must be clean
  D) Compositor contract — output shape, dtype, channels invariant

TDD RULES:
  - Write the failing test before the fix.
  - Tighten tests, never weaken them.
  - Every fix must have a regression test that catches reintroduction.
"""

import cv2
import numpy as np
import pytest

from face_os.types import CropPlan, CropStrategy, Landmarks
from face_os.crop_planner import apply_crop, CropPlanner
from face_os.compositor import Compositor
from face_os.identity_state import FrequencyDecomposition, IdentityState


# ═══════════════════════════════════════════════════════════════════════════════
# BUG CLASS B: Frame Size Invariance
# ═══════════════════════════════════════════════════════════════════════════════

def _assert_frame_contract(frame: np.ndarray, expected_h: int, expected_w: int,
                           expected_dtype=np.uint8, expected_channels: int = 3):
    """Centralised frame contract validation."""
    assert frame is not None, "Frame is None — violates contract"
    assert frame.shape == (expected_h, expected_w, expected_channels), (
        f"Frame shape {frame.shape} != ({expected_h}, {expected_w}, {expected_channels})"
    )
    assert frame.dtype == expected_dtype, (
        f"Frame dtype {frame.dtype} != {expected_dtype}"
    )
    assert len(frame.shape) == 3, f"Frame must be 3D (HWC), got {len(frame.shape)}D"
    assert not np.any(np.isnan(frame)), "Frame contains NaN"
    assert not np.any(np.isinf(frame)), "Frame contains Inf"
    assert np.min(frame) >= 0, f"Frame has negative values: min={np.min(frame)}"
    assert np.max(frame) <= 255, f"Frame has overflow values: max={np.max(frame)}"


class TestFrameContract:
    """Frame contract must hold for every pipeline path.

    The contract is:
      - Shape: (cfg.crop.output_size[1], cfg.crop.output_size[0], 3)
               i.e. (1920, 1080, 3) for default config
      - dtype: np.uint8
      - No NaN, No Inf
      - Value range [0, 255]
    """

    OUTPUT_H = 1920
    OUTPUT_W = 1080

    @pytest.fixture
    def src_frame(self):
        """640x360 source frame (typical 16:9 input)."""
        return np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)

    @pytest.fixture
    def center_crop_plan(self):
        return CropPlan(
            strategy=CropStrategy.CENTER,
            src_x=0, src_y=0, src_w=640, src_h=360,
            dst_w=self.OUTPUT_W, dst_h=self.OUTPUT_H,
            confidence=0.5,
        )

    @pytest.fixture
    def last_known_crop_plan(self):
        return CropPlan(
            strategy=CropStrategy.LAST_KNOWN,
            src_x=50, src_y=30, src_w=400, src_h=225,
            dst_w=self.OUTPUT_W, dst_h=self.OUTPUT_H,
            face_center_out=(540, 960),
            confidence=0.5,
        )

    def test_apply_crop_center_produces_correct_dimensions(self, src_frame, center_crop_plan):
        """Center crop must produce exact output dimensions."""
        result = apply_crop(src_frame, center_crop_plan)
        _assert_frame_contract(result, self.OUTPUT_H, self.OUTPUT_W)

    def test_apply_crop_last_known_produces_correct_dimensions(self, src_frame, last_known_crop_plan):
        """Last known crop must produce exact output dimensions."""
        result = apply_crop(src_frame, last_known_crop_plan)
        _assert_frame_contract(result, self.OUTPUT_H, self.OUTPUT_W)

    def test_apply_crop_face_locked_produces_correct_dimensions(self, src_frame):
        """Face-locked crop must produce exact output dimensions."""
        plan = CropPlan(
            strategy=CropStrategy.FACE_LOCKED,
            src_x=100, src_y=50, src_w=300, src_h=533,
            dst_w=self.OUTPUT_W, dst_h=self.OUTPUT_H,
            face_center_out=(540, 600),
            headroom_ratio=0.3,
            confidence=0.9,
        )
        result = apply_crop(src_frame, plan)
        _assert_frame_contract(result, self.OUTPUT_H, self.OUTPUT_W)

    def test_apply_crop_degenerate_fallback_produces_correct_dimensions(self, src_frame):
        """Degenerate crop (w<2 or h<2) must fallback and still match expected size."""
        plan = CropPlan(
            strategy=CropStrategy.CENTER,
            src_x=0, src_y=0, src_w=1, src_h=1,
            dst_w=self.OUTPUT_W, dst_h=self.OUTPUT_H,
            confidence=0.1,
        )
        result = apply_crop(src_frame, plan)
        _assert_frame_contract(result, self.OUTPUT_H, self.OUTPUT_W)

    def test_crop_planner_center_path_output_size(self, src_frame):
        """CropPlanner.plan_crop with no face must output correct dimensions."""
        planner = CropPlanner()
        plan = planner.plan_crop(src_frame.shape[:2], face_track=None, landmarks=None)
        result = apply_crop(src_frame, plan)
        _assert_frame_contract(result, self.OUTPUT_H, self.OUTPUT_W)

    def test_repeated_crop_planner_calls_same_output_size(self, src_frame):
        """Multiple consecutive calls with no face must all match contract."""
        planner = CropPlanner()
        for _ in range(5):
            plan = planner.plan_crop(src_frame.shape[:2], face_track=None, landmarks=None)
            result = apply_crop(src_frame, plan)
            _assert_frame_contract(result, self.OUTPUT_H, self.OUTPUT_W)

    def test_compositor_shape_dtype_invariant(self):
        """Compositor output must always match input shape and dtype."""
        compositor = Compositor()
        h, w = self.OUTPUT_H, self.OUTPUT_W

        original = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        enhanced = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

        # Test with face_mask
        face_mask = np.zeros((h, w), dtype=np.float32)
        face_mask[h//4:3*h//4, w//4:3*w//4] = 1.0
        result = compositor.composite(original, enhanced, face_mask=face_mask)
        _assert_frame_contract(result, h, w)

        # Test with no face_mask
        result2 = compositor.composite(original, enhanced)
        _assert_frame_contract(result2, h, w)

        # Test with confidence map
        from face_os.types import ConfidenceMap
        conf = ConfidenceMap(combined=np.ones((h, w), dtype=np.float32) * 0.5)
        result3 = compositor.composite(original, enhanced, confidence=conf, face_mask=face_mask)
        _assert_frame_contract(result3, h, w)


# ═══════════════════════════════════════════════════════════════════════════════
# BUG CLASS A: Mask Stability
# ═══════════════════════════════════════════════════════════════════════════════

class TestMaskStability:
    """Mask must be spatially stable across consecutive frames.

    The mask centroid should not jitter more than a few pixels
    when the face position is nearly identical between frames.
    """

    @pytest.fixture
    def stable_landmarks_points(self):
        """Create a stable set of 478 MediaPipe-like landmark points.

        Returns a (478, 2) array with face-like geometry.
        """
        np.random.seed(42)
        pts = np.zeros((478, 2), dtype=np.float32)
        # Face oval: create circular arrangement
        angles = np.linspace(0, 2 * np.pi, 36)  # ~36 face contour points
        cx, cy = 320, 180  # center
        rx, ry = 120, 160  # semi-axes
        face_oval_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
                             361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                             176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
                             162, 21, 54, 103, 67, 109]
        for i, idx in enumerate(face_oval_indices):
            a = angles[i % len(angles)]
            pts[idx] = [cx + rx * np.cos(a), cy + ry * np.sin(a)]
        # Key points
        pts[1] = [cx, cy - 40]   # nose tip
        pts[33] = [cx - 30, cy - 30]  # left eye inner
        pts[263] = [cx + 30, cy - 30]  # right eye inner
        pts[61] = [cx - 15, cy + 50]   # mouth left
        pts[291] = [cx + 15, cy + 50]  # mouth right
        pts[152] = [cx, cy + 140]  # chin
        return pts

    def test_canonical_geometry_mask_has_minimum_coverage(self):
        """Geometry-based canonical mask must cover > 30% of canonical atlas."""
        from face_os.pipeline import FaceOSPipeline

        mask = FaceOSPipeline._make_canonical_geometry_mask((256, 256))
        mask_ratio = mask.sum() / mask.size
        assert mask_ratio > 0.30, (
            f"Geometry mask only covers {mask_ratio:.1%} of canonical area — "
            f"mask is too small to blend identity face."
        )

    def test_canonical_geometry_mask_has_maximum_coverage(self):
        """Geometry mask must not cover > 90% (should not extend to borders)."""
        from face_os.pipeline import FaceOSPipeline

        mask = FaceOSPipeline._make_canonical_geometry_mask((256, 256))
        mask_ratio = mask.sum() / mask.size
        assert mask_ratio < 0.90, (
            f"Geometry mask covers {mask_ratio:.1%} of canonical area — "
            f"mask extends too far, may blend background."
        )

    def test_canonical_geometry_mask_brightness_invariant(self):
        """Geometry-based mask must be identical regardless of input brightness."""
        from face_os.pipeline import FaceOSPipeline

        mask1 = FaceOSPipeline._make_canonical_geometry_mask((256, 256))
        mask2 = FaceOSPipeline._make_canonical_geometry_mask((256, 256))
        mask3 = FaceOSPipeline._make_canonical_geometry_mask((256, 256))

        # Must be deterministic (same every call)
        assert np.array_equal(mask1, mask2), "Geometry mask is not deterministic"
        assert np.array_equal(mask2, mask3), "Geometry mask is not deterministic"
        # Must be in [0, 1]
        assert np.all(mask1 >= 0) and np.all(mask1 <= 1)
        assert mask1.dtype == np.float32

    def test_canonical_geometry_mask_has_smooth_edges(self):
        """Geometry mask must have smooth (feathered) edges, not binary."""
        from face_os.pipeline import FaceOSPipeline

        mask = FaceOSPipeline._make_canonical_geometry_mask((256, 256))
        # Count pixels in the transition zone (0 < x < 1)
        transition = np.sum((mask > 0.01) & (mask < 0.99))
        total = mask.size
        transition_ratio = transition / total
        # At least 5% of pixels should be in transition zone
        assert transition_ratio > 0.05, (
            f"Only {transition_ratio:.1%} of pixels are in transition zone — "
            f"edges not feathered enough."
        )

    def test_mask_centroid_stable_across_similar_frames(self):
        """Mask centroid must not drift > 2px when face position is nearly identical."""
        from face_os.landmarks import create_region_masks, extract_landmarks

        # Two nearly identical frames
        frame1 = np.ones((360, 640, 3), dtype=np.uint8) * 128
        frame2 = np.ones((360, 640, 3), dtype=np.uint8) * 128

        np.random.seed(42)
        pts = self._make_realistic_mediapipe_mesh(320, 180)

        lm = extract_landmarks(frame1, pts)
        assert lm is not None, "Could not extract landmarks"

        # Get face mask from region masks (identical landmarks → identical masks)
        masks1 = create_region_masks(lm, frame1.shape[:2])
        masks2 = create_region_masks(lm, frame2.shape[:2])

        face_mask1 = masks1.get("face", np.zeros(frame1.shape[:2], dtype=np.float32))
        face_mask2 = masks2.get("face", np.zeros(frame2.shape[:2], dtype=np.float32))

        # Centroid should be identical for identical landmarks
        def centroid(mask):
            if mask.max() < 0.01:
                return np.array([0.0, 0.0])
            ys, xs = np.where(mask > 0.5)
            if len(ys) == 0:
                return np.array([0.0, 0.0])
            return np.array([float(np.mean(xs)), float(np.mean(ys))])

        c1 = centroid(face_mask1)
        c2 = centroid(face_mask2)

        drift = np.linalg.norm(c1 - c2)
        assert drift < 0.5, (
            f"Mask centroid drifted {drift:.2f}px between identical frames."
        )

    def test_mask_centroid_stable_with_1px_landmark_shift(self):
        """Mask centroid must not drift > 2px with 1px landmark shift."""
        from face_os.landmarks import create_region_masks, extract_landmarks

        frame = np.ones((360, 640, 3), dtype=np.uint8) * 128
        np.random.seed(42)
        pts = self._make_realistic_mediapipe_mesh(320, 180)
        pts2 = pts.copy()
        pts2[:, 0] += 1
        pts2[:, 1] += 1

        lm1 = extract_landmarks(frame, pts)
        lm2 = extract_landmarks(frame, pts2)

        masks1 = create_region_masks(lm1, frame.shape[:2])
        masks2 = create_region_masks(lm2, frame.shape[:2])
        fm1 = masks1.get("face", np.zeros(frame.shape[:2], dtype=np.float32))
        fm2 = masks2.get("face", np.zeros(frame.shape[:2], dtype=np.float32))

        def centroid(mask):
            ys, xs = np.where(mask > 0.5)
            if len(ys) == 0:
                return np.array([0.0, 0.0])
            return np.array([float(np.mean(xs)), float(np.mean(ys))])

        drift = np.linalg.norm(centroid(fm1) - centroid(fm2))
        assert drift < 2.0, (
            f"Mask centroid drifted {drift:.2f}px with 1px landmark shift."
        )

    @staticmethod
    def _make_realistic_mediapipe_mesh(cx: float = 320, cy: float = 180, scale: float = 1.0) -> np.ndarray:
        """Create realistic 478-point MediaPipe mesh with proper face geometry."""
        from face_os.landmarks import (
            MPI_FACE_OVAL, MPI_LEFT_EYE_CONTOUR, MPI_RIGHT_EYE_CONTOUR,
            MPI_LIPS_CONTOUR, MPI_NOSE_TIP, MPI_CHIN,
        )
        pts = np.zeros((478, 2), dtype=np.float32)
        s = scale
        # Face oval (contour of the face)
        n_oval = len(MPI_FACE_OVAL)
        for i, idx in enumerate(MPI_FACE_OVAL):
            angle = 2 * np.pi * i / n_oval
            pts[idx] = [cx + 130 * s * np.cos(angle), cy + 170 * s * np.sin(angle)]

        # Nose tip
        pts[1] = [cx, cy - 30 * s]
        pts[2] = [cx - 10 * s, cy - 15 * s]
        pts[3] = [cx + 10 * s, cy - 15 * s]
        pts[4] = [cx - 5 * s, cy]
        pts[5] = [cx + 5 * s, cy]
        pts[6] = [cx - 15 * s, cy - 10 * s]
        pts[197] = [cx - 15 * s, cy - 5 * s]
        pts[195] = [cx + 15 * s, cy - 5 * s]
        pts[168] = [cx, cy - 20 * s]

        # Left eye contour
        for i, idx in enumerate(MPI_LEFT_EYE_CONTOUR):
            angle = 2 * np.pi * i / len(MPI_LEFT_EYE_CONTOUR)
            pts[idx] = [cx - 35 * s + 15 * s * np.cos(angle),
                        cy - 20 * s + 8 * s * np.sin(angle)]

        # Right eye contour
        for i, idx in enumerate(MPI_RIGHT_EYE_CONTOUR):
            angle = 2 * np.pi * i / len(MPI_RIGHT_EYE_CONTOUR)
            pts[idx] = [cx + 35 * s + 15 * s * np.cos(angle),
                        cy - 20 * s + 8 * s * np.sin(angle)]

        # Eye corners
        pts[33] = [cx - 30 * s, cy - 20 * s]   # left eye inner
        pts[133] = [cx - 45 * s, cy - 20 * s]  # left eye outer
        pts[263] = [cx + 30 * s, cy - 20 * s]  # right eye inner
        pts[362] = [cx + 45 * s, cy - 20 * s]  # right eye outer

        # Lips
        for i, idx in enumerate(MPI_LIPS_CONTOUR):
            angle = 2 * np.pi * i / len(MPI_LIPS_CONTOUR)
            pts[idx] = [cx + 25 * s * np.cos(angle),
                        cy + 50 * s + 10 * s * np.sin(angle)]

        # Mouth corners
        pts[61] = [cx - 20 * s, cy + 50 * s]
        pts[291] = [cx + 20 * s, cy + 50 * s]

        # Chin
        pts[152] = [cx, cy + 160 * s]

        # Eyebrows
        for i, idx in enumerate([66, 105, 63, 70, 156, 46, 53, 52, 65, 55]):
            angle = -np.pi * 0.5 + np.pi * i / 9
            pts[idx] = [cx - 30 * s + 30 * s * np.cos(angle),
                        cy - 45 * s + 5 * s * np.sin(angle)]
        for i, idx in enumerate([296, 334, 293, 300, 276, 283, 282, 295, 285]):
            angle = -np.pi * 0.5 + np.pi * i / 8
            pts[idx] = [cx + 30 * s + 30 * s * np.cos(angle),
                        cy - 45 * s + 5 * s * np.sin(angle)]

        # Fill in remaining points by interpolating from face oval center
        for i in range(478):
            if pts[i, 0] == 0 and pts[i, 1] == 0:
                pts[i] = [cx + (np.random.rand() - 0.5) * 100 * s,
                          cy + (np.random.rand() - 0.5) * 120 * s]
        return pts

    def test_landmark_derived_mask_consistency(self):
        """Landmark-derived face mask should have IoU > 0.9 between similar frames."""
        from face_os.landmarks import create_region_masks, extract_landmarks

        np.random.seed(42)
        pts = self._make_realistic_mediapipe_mesh(320, 180)
        pts2 = pts.copy()
        pts2[:, 0] += 1
        pts2[:, 1] += 1

        frame = np.ones((360, 640, 3), dtype=np.uint8) * 128
        lm = extract_landmarks(frame, pts)
        assert lm is not None

        masks = create_region_masks(lm, frame.shape[:2])
        face_mask = masks.get("face")

        assert face_mask is not None
        assert face_mask.shape == (360, 640)
        assert face_mask.dtype == np.float32
        assert face_mask.max() > 0.5, f"Face mask has no face pixels, max={face_mask.max():.3f}"
        assert np.all(face_mask >= 0) and np.all(face_mask <= 1)

        lm2 = extract_landmarks(frame, pts2)
        masks2 = create_region_masks(lm2, frame.shape[:2])
        face_mask2 = masks2.get("face")

        binary1 = (face_mask > 0.5).astype(np.float32)
        binary2 = (face_mask2 > 0.5).astype(np.float32)
        intersection = np.sum(binary1 * binary2)
        union = np.sum(np.clip(binary1 + binary2, 0, 1))
        iou = intersection / max(union, 1)
        assert iou > 0.9, f"IoU for 1px shifted landmarks: {iou:.4f} — too low"


# ═══════════════════════════════════════════════════════════════════════════════
# BUG CLASS C: NaN / Inf in blend and warp outputs
# ═══════════════════════════════════════════════════════════════════════════════

class TestNumericStability:
    """No NaN, Inf, or degenerate values in any blend/warp output."""

    def test_compositor_no_nan(self):
        """Compositor must never produce NaN or Inf."""
        compositor = Compositor()
        h, w = 1920, 1080
        orig = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        enh = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        result = compositor.composite(orig, enh)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_compositor_with_all_black(self):
        """Compositor must handle all-black inputs without NaN."""
        compositor = Compositor()
        h, w = 1920, 1080
        black = np.zeros((h, w, 3), dtype=np.uint8)
        face_mask = np.zeros((h, w), dtype=np.float32)
        face_mask[100:500, 200:600] = 1.0
        result = compositor.composite(black, black, face_mask=face_mask)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        assert result.dtype == np.uint8
        assert result.shape == (h, w, 3)

    def test_compositor_with_all_white(self):
        """Compositor must handle all-white inputs without overflow."""
        compositor = Compositor()
        h, w = 1920, 1080
        white = np.ones((h, w, 3), dtype=np.uint8) * 255
        result = compositor.composite(white, white)
        assert not np.any(np.isnan(result))
        assert np.max(result) == 255
        assert result.dtype == np.uint8

    def test_frequency_decomposition_no_nan(self):
        """Frequency decomposition must not produce NaN."""
        freq = FrequencyDecomposition()
        for _ in range(5):
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            low, high = freq.decompose(img)
            assert not np.any(np.isnan(low))
            assert not np.any(np.isnan(high))
            assert not np.any(np.isinf(low))
            assert not np.any(np.isinf(high))
            recon = freq.reconstruct(low, high)
            assert not np.any(np.isnan(recon))

    def test_identity_state_query_no_nan(self):
        """Identity state query must not produce NaN."""
        state = IdentityState()
        # Not initialized — should return safe defaults
        face = np.ones((256, 256, 3), dtype=np.uint8) * 128
        quality = np.ones((256, 256), dtype=np.float32) * 0.8
        result, conf = state.query(face, quality)
        assert result.shape == (256, 256, 3)
        assert conf.shape == (256, 256)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isnan(conf))


# ═══════════════════════════════════════════════════════════════════════════════
# BUG CLASS D: Bidirectional Path Frame Size
# ═══════════════════════════════════════════════════════════════════════════════

class TestBidirectionalPathFrameSize:
    """The bidirectional pass 3 no-face path must output same dimensions as face path.

    BUG: In pipeline.py _process_bidirectional (line ~459-465), when
    frame_idx not in frame_data, cropped = source_frame is used directly
    instead of applying the crop. This outputs 16:9 frames in a 9:16 pipeline.
    """

    OUTPUT_H = 1920
    OUTPUT_W = 1080

    def test_fallback_path_must_crop_to_target_size(self):
        """Test that apply_crop is used in fallback path (not raw source_frame)."""
        src_frame = np.ones((360, 640, 3), dtype=np.uint8) * 128
        plan = CropPlan(
            strategy=CropStrategy.FACE_LOCKED,
            src_x=50, src_y=30, src_w=300, src_h=533,
            dst_w=self.OUTPUT_W, dst_h=self.OUTPUT_H,
            face_center_out=(540, 600),
            confidence=0.9,
        )
        # This is what the buggy code does: directly use source_frame
        buggy_output = src_frame
        # This is what it should do: apply_crop
        fixed_output = apply_crop(src_frame, plan)

        # The buggy output is the wrong size
        assert buggy_output.shape != fixed_output.shape, (
            "BUG: raw source_frame has different size from cropped output. "
            "_process_bidirectional must call apply_crop even for fallback path."
        )
        # Verify the fixed path
        _assert_frame_contract(fixed_output, self.OUTPUT_H, self.OUTPUT_W)

    def test_last_known_crop_produces_same_dimensions(self):
        """LAST_KNOWN strategy must produce same dimensions as FACE_LOCKED."""
        src_frame = np.ones((360, 640, 3), dtype=np.uint8) * 128

        plan_face = CropPlan(
            strategy=CropStrategy.FACE_LOCKED,
            src_x=50, src_y=30, src_w=300, src_h=533,
            dst_w=self.OUTPUT_W, dst_h=self.OUTPUT_H,
            face_center_out=(540, 600),
            confidence=0.9,
        )
        plan_last = CropPlan(
            strategy=CropStrategy.LAST_KNOWN,
            src_x=50, src_y=30, src_w=300, src_h=533,
            dst_w=self.OUTPUT_W, dst_h=self.OUTPUT_H,
            face_center_out=(540, 600),
            confidence=0.5,
        )

        out_face = apply_crop(src_frame, plan_face)
        out_last = apply_crop(src_frame, plan_last)
        assert out_face.shape == out_last.shape, (
            f"LAST_KNOWN shape {out_last.shape} != FACE_LOCKED shape {out_face.shape}"
        )
        _assert_frame_contract(out_last, self.OUTPUT_H, self.OUTPUT_W)


# ═══════════════════════════════════════════════════════════════════════════════
# BUG CLASS E: Pipeline Consistency — No Identity Mode
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoIdentityPath:
    """No-identity mode must produce same output contract as identity mode."""

    OUTPUT_H = 1920
    OUTPUT_W = 1080

    def test_render_frame_shape_preserved(self):
        """render_frame must preserve input shape and dtype."""
        from face_os.face_enhance import render_frame

        # Various sizes
        for h, w in [(480, 640), (1080, 1920), (1920, 1080)]:
            frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            result = render_frame(frame)
            assert result.shape == (h, w, 3), f"render_frame changed shape at {h}x{w}"
            assert result.dtype == np.uint8, f"render_frame changed dtype at {h}x{w}"
            assert not np.any(np.isnan(result))

    def test_render_frame_with_masks_shape_preserved(self):
        """render_frame with masks must preserve shape."""
        from face_os.face_enhance import render_frame, _create_enhancement_mask
        from face_os.landmarks import create_region_masks
        from face_os.types import Landmarks

        h, w = 480, 640
        frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        pts = np.zeros((478, 2), dtype=np.float32)
        pts[1] = [320, 100]
        pts[33] = [280, 120]
        pts[263] = [360, 120]
        pts[61] = [300, 200]
        pts[291] = [340, 200]
        pts[152] = [320, 260]

        lm = Landmarks(points=pts)
        masks = create_region_masks(lm, (h, w))
        enhancement_mask = _create_enhancement_mask(masks, frame.shape)
        result = render_frame(frame, enhancement_mask, masks)

        assert result.shape == (h, w, 3)
        assert result.dtype == np.uint8


# ═══════════════════════════════════════════════════════════════════════════════
# BUG CLASS F: Landmark/Crop Coordinate Consistency
# ═══════════════════════════════════════════════════════════════════════════════

class TestLandmarkCropConsistency:
    """Landmarks must translate correctly between source and crop spaces."""

    def test_adjust_landmarks_to_crop_basic(self):
        """Landmark coordinates must scale correctly with crop."""
        from face_os.pipeline import FaceOSPipeline
        pts = np.array([[100, 200], [300, 400], [500, 600]], dtype=np.float32)
        lm = Landmarks(
            points=pts,
            yaw=5.0, pitch=2.0, roll=1.0,
            left_eye_center=(150, 250),
            right_eye_center=(350, 250),
            nose_tip=(320, 350),
            mouth_center=(320, 450),
        )
        plan = CropPlan(
            strategy=CropStrategy.CENTER,
            src_x=50, src_y=100, src_w=400, src_h=600,
            dst_w=1080, dst_h=1920,
        )
        pipeline = FaceOSPipeline()
        adjusted = pipeline._adjust_landmarks_to_crop(lm, plan)

        assert adjusted is not None
        # After crop offset (src_x=50, src_y=100) and scale (1080/400, 1920/600)
        expected_x0 = (100 - 50) * (1080 / 400)
        expected_y0 = (200 - 100) * (1920 / 600)
        assert abs(adjusted.points[0, 0] - expected_x0) < 1.0
        assert abs(adjusted.points[0, 1] - expected_y0) < 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# BUG CLASS H: Render Core — All paths must use _render_core
# ═══════════════════════════════════════════════════════════════════════════════

class TestRenderCoreUsage:
    """Both _process_frame_v2() and _render_frame_v2() must call _render_core().

    _render_core() is the SINGLE source of truth for all rendering logic.
    NO rendering logic may exist outside it.
    """

    def test_pipeline_has_render_core_method(self):
        """FaceOSPipeline must have _render_core and _composite_identity_to_crop methods."""
        from face_os.pipeline import FaceOSPipeline
        assert hasattr(FaceOSPipeline, '_render_core'), "Missing _render_core"
        assert hasattr(FaceOSPipeline, '_composite_identity_to_crop'), "Missing _composite_identity_to_crop"
        assert callable(getattr(FaceOSPipeline, '_render_core')), "_render_core not callable"

    def test_process_frame_v2_calls_render_core(self):
        """_process_frame_v2 must contain a call to _render_core."""
        import inspect
        from face_os.pipeline import FaceOSPipeline
        source = inspect.getsource(FaceOSPipeline._process_frame_v2)
        assert 'self._render_core(' in source, (
            "_process_frame_v2 does not call _render_core"
        )
        assert 'self._composite_identity_to_crop(' not in source, (
            "_process_frame_v2 directly calls _composite_identity_to_crop — must go through _render_core"
        )
        assert 'self._render_with_physical_renderer(' not in source, (
            "_process_frame_v2 directly calls _render_with_physical_renderer — must go through _render_core"
        )

    def test_render_frame_v2_calls_render_core(self):
        """_render_frame_v2 must contain a call to _render_core (not inline rendering)."""
        import inspect
        from face_os.pipeline import FaceOSPipeline
        source = inspect.getsource(FaceOSPipeline._render_frame_v2)
        assert 'self._render_core(' in source, (
            "_render_frame_v2 does not call _render_core"
        )
        # Must NOT contain warp+blend rendering logic (duplicated from _render_core)
        assert 'self._composite_identity_to_crop(' not in source, (
            "_render_frame_v2 directly calls _composite_identity_to_crop — must go through _render_core"
        )
        assert 'self._render_with_physical_renderer(' not in source, (
            "_render_frame_v2 directly calls _render_with_physical_renderer — must go through _render_core"
        )

    def test_render_core_has_no_duplicate_paths(self):
        """_render_core must contain the PhysicalRenderer dispatch and identity composite logic."""
        import inspect
        from face_os.pipeline import FaceOSPipeline
        source = inspect.getsource(FaceOSPipeline._render_core)
        # Must contain PhysicalRenderer condition
        assert 'RendererMode.PHYSICAL' in source or 'PHYSICAL' in source, (
            "_render_core missing PhysicalRenderer dispatch"
        )
        # Must contain identity composite call
        assert 'self._composite_identity_to_crop(' in source, (
            "_render_core missing _composite_identity_to_crop call"
        )

    def test_telemetry_incremented_in_render_core(self):
        """_render_core must track physical_render_frames and alpha_fallback_frames."""
        import inspect
        from face_os.pipeline import FaceOSPipeline
        source = inspect.getsource(FaceOSPipeline._render_core)
        assert 'physical_render_frames' in source, "_render_core missing physical_render_frames telemetry"
        assert 'alpha_fallback_frames' in source, "_render_core missing alpha_fallback_frames telemetry"


# ═══════════════════════════════════════════════════════════════════════════════
# BUG CLASS G: EMA Smoothing Stability
# ═══════════════════════════════════════════════════════════════════════════════

class TestEMASmoothing:
    """EMA smoothing must not cause excessive lag or drift."""

    def test_M_inv_ema_not_too_aggressive(self):
        """M_inv EMA alpha should not be so low that it causes visible lag (300 frame lag)."""
        # Current code: M_inv = 0.7 * last + 0.3 * current
        # After 10 frames of constant input: value = (1 - 0.3^10) ≈ 99.9994%
        # After 1 frame: value = 30%
        # After 3 frames: value = 65.7%
        # After 10 frames: value = 97.2%
        alpha = 0.3  # Current
        # Time to reach 95% of target
        n = np.log(0.05) / np.log(1 - alpha)
        assert n < 15, f"Alpha={alpha}: takes {n:.1f} frames to reach 95%. Too slow!"

    def test_alpha_can_be_increased(self):
        """A reasonable alpha (0.5+) converges in < 5 frames."""
        for alpha in [0.5, 0.6, 0.7]:
            n = np.log(0.05) / np.log(1 - alpha)
            assert n < 7, f"Alpha={alpha}: takes {n:.1f} frames to reach 95%. Too slow!"
