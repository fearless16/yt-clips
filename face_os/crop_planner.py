"""
crop_planner.py — Reference-Based Face-Aware Crop Planning.

OLD APPROACH: fixed headroom_ratio → face positioned at 30% from top
NEW APPROACH: analyze expectation reference → match composition

The expectation image tells us EXACTLY how the output should look:
  - Face top: 24.3% from top
  - Face height: 33.7% of output
  - Face center: 41.1% from top
  - Headroom: 24.3%

We use these as COMPOSITION TARGETS and adapt based on source constraints.

When source face is larger than expectation face:
  - Can't shrink face below source size (crop limitation)
  - Use expectation's RELATIVE positioning within face region
  - Accept slightly larger face, match headroom ratio

When source face is smaller than expectation face:
  - Zoom in to match expectation face size
  - Position using expectation's headroom ratio
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from face_os.config import get_config
from face_os.types import CropPlan, CropStrategy, FaceTrack, Landmarks


cfg = get_config()


# ─── Expectation Reference Analysis ─────────────────────────────────────────

class CompositionReference:
    """Composition metrics extracted from the expectation image.

    These are the TARGET values that the crop planner tries to match.
    """

    def __init__(
        self,
        face_top_pct: float = 0.243,      # Face top position (% from top)
        face_height_pct: float = 0.337,    # Face height (% of output)
        face_center_y_pct: float = 0.411,  # Face center Y (% from top)
        headroom_pct: float = 0.243,       # Space above face (% of output)
    ):
        self.face_top_pct = face_top_pct
        self.face_height_pct = face_height_pct
        self.face_center_y_pct = face_center_y_pct
        self.headroom_pct = headroom_pct

    @classmethod
    def from_image(cls, image_path: str) -> 'CompositionReference':
        """Extract composition metrics from a reference image."""
        img = cv2.imread(image_path)
        if img is None:
            return cls()  # Use defaults

        h, w = img.shape[:2]
        
        # Use MediaPipe instead of Haar cascade
        from face_os.detect_track import detect_faces
        detections = detect_faces(img)

        if not detections:
            return cls()

        track = detections[0]
        x, y, fw, fh = track.smooth_bbox

        face_top_pct = y / h
        face_height_pct = fh / h
        face_center_y_pct = (y + fh / 2) / h
        headroom_pct = y / h

        return cls(
            face_top_pct=face_top_pct,
            face_height_pct=face_height_pct,
            face_center_y_pct=face_center_y_pct,
            headroom_pct=headroom_pct,
        )

    def __repr__(self) -> str:
        return (
            f"CompositionReference("
            f"top={self.face_top_pct:.1%}, "
            f"height={self.face_height_pct:.1%}, "
            f"center={self.face_center_y_pct:.1%}, "
            f"headroom={self.headroom_pct:.1%})"
        )


# ─── Crop Planner ───────────────────────────────────────────────────────────

class CropPlanner:
    """Plans 9:16 crops from 16:9 source frames.

    Uses reference-based composition matching:
    1. Analyze expectation image at startup
    2. For each frame, plan crop that MATCHES expectation composition
    3. Adapt based on source face size vs expectation face size
    4. Smooth transitions with EMA
    """

    def __init__(self, reference_image: str = "expectation.png"):
        # Analyze expectation for composition targets
        self.reference = CompositionReference.from_image(reference_image)
        print(f"  Crop reference: {self.reference}")

        # Smoothing state
        self._prev_crop: Optional[CropPlan] = None
        self._smooth_x: Optional[float] = None
        self._smooth_y: Optional[float] = None
        self._smooth_w: Optional[float] = None
        self._smooth_h: Optional[float] = None
        self._frames_without_face = 0

    def plan_crop(
        self,
        source_shape: Tuple[int, int],
        face_track: Optional[FaceTrack] = None,
        landmarks: Optional[Landmarks] = None,
    ) -> CropPlan:
        """Plan the crop for this frame.

        Args:
            source_shape: (height, width) of source frame
            face_track: Current face tracking data
            landmarks: Detected landmarks

        Returns:
            CropPlan with source crop region and target dimensions
        """
        src_h, src_w = source_shape
        dst_w, dst_h = cfg.crop.output_size

        if face_track and face_track.smooth_bbox:
            self._frames_without_face = 0
            plan = self._plan_reference_based(
                src_w, src_h, dst_w, dst_h,
                face_track, landmarks,
            )
        else:
            self._frames_without_face += 1
            # FIX: Limit last_known to 5 frames, then fall back to center
            if self._prev_crop and self._frames_without_face < 5:
                plan = self._plan_last_known(src_w, src_h, dst_w, dst_h)
            else:
                plan = self._plan_center(src_w, src_h, dst_w, dst_h)

        # Smooth the crop
        plan = self._smooth(plan)
        self._prev_crop = plan
        return plan

    def _plan_reference_based(
        self,
        src_w: int, src_h: int,
        dst_w: int, dst_h: int,
        face_track: FaceTrack,
        landmarks: Optional[Landmarks],
    ) -> CropPlan:
        """Plan crop using reference-based composition matching.

        Algorithm:
        1. Get face bbox from source
        2. Compute ideal crop height to match expectation face ratio
        3. If crop exceeds source, use full source height
        4. Position face to match expectation's relative position
        5. Protect forehead if landmarks available

        CONSTRAINTS:
        - Source face may be HIGHER than expectation (less headroom)
        - Source face may be LARGER than expectation (can't shrink)
        - We match what we CAN, accept what we can't
        """
        fx, fy, fw, fh = face_track.smooth_bbox

        # Face center in source
        face_cx = fx + fw // 2
        face_cy = fy + fh // 2

        # ─── Step 1: Determine ideal crop dimensions ───

        # What face height ratio do we want in the output?
        target_face_ratio = self.reference.face_height_pct  # 0.337

        # What's the current face ratio if we use full source height?
        current_face_ratio = fh / src_h  # e.g., 135/360 = 0.375

        if current_face_ratio <= target_face_ratio:
            # Face is smaller than target — zoom in to match
            ideal_crop_h = int(fh / target_face_ratio)
            scale = dst_h / ideal_crop_h
            crop_w = int(dst_w / scale)
            crop_h = ideal_crop_h
        else:
            # Face is larger than target — use full source height
            # Can't shrink face below source size
            crop_h = src_h
            scale = dst_h / crop_h
            crop_w = int(dst_w / scale)

        # Clamp to source bounds
        crop_w = min(crop_w, src_w)
        crop_h = min(crop_h, src_h)

        # Maintain 9:16 aspect ratio
        target_aspect = dst_w / dst_h  # 0.5625
        current_aspect = crop_w / max(crop_h, 1)

        if current_aspect > target_aspect:
            crop_w = int(crop_h * target_aspect)
        else:
            crop_h = int(crop_w / target_aspect)

        # ─── Step 2: Position face — preserve source headroom ───

        # Architecture: source is ground truth for POSITION
        # We can't add headroom that doesn't exist in source
        # So we PRESERVE the source's headroom ratio

        source_headroom = fy / max(src_h, 1)  # e.g., 68/360 = 0.189

        # Minimum headroom from reference set (p7.png has 21.6%)
        min_headroom = 0.15  # Absolute minimum for hair preservation

        # Use the BETTER of source headroom or minimum
        target_headroom = max(source_headroom, min_headroom)

        # Position crop so face top is at target_headroom of crop height
        # face_top_in_crop = fy - crop_y
        # target_headroom = (fy - crop_y) / crop_h
        # crop_y = fy - target_headroom * crop_h
        crop_y = int(fy - target_headroom * crop_h)
        crop_y = max(0, crop_y)

        # Horizontal: center on face
        crop_x = face_cx - crop_w // 2

        # ─── Step 3: Clamp to source bounds ───

        crop_x = max(0, min(crop_x, src_w - crop_w))
        crop_y = max(0, min(crop_y, src_h - crop_h))

        # ─── Step 4: Protect forehead ───

        if landmarks and cfg.crop.protect_forehead:
            head_top = int(np.min(landmarks.points[:, 1]))
            min_crop_y = head_top - 10
            if crop_y > min_crop_y:
                crop_y = max(0, min_crop_y)
                # Re-clamp bottom
                if crop_y + crop_h > src_h:
                    crop_y = src_h - crop_h

        # ─── Step 5: Compute output face position ───

        face_out_x = int((face_cx - crop_x) * dst_w / max(crop_w, 1))
        face_out_y = int((face_cy - crop_y) * dst_h / max(crop_h, 1))

        return CropPlan(
            strategy=CropStrategy.FACE_LOCKED,
            src_x=crop_x,
            src_y=crop_y,
            src_w=crop_w,
            src_h=crop_h,
            dst_w=dst_w,
            dst_h=dst_h,
            face_center_out=(face_out_x, face_out_y),
            headroom_ratio=self.reference.headroom_pct,
            confidence=face_track.detection.confidence if face_track.detection else 0.5,
        )

    def _plan_center(
        self, src_w: int, src_h: int, dst_w: int, dst_h: int,
    ) -> CropPlan:
        """Fallback: center crop when no face detected."""
        target_aspect = dst_w / dst_h
        crop_w = min(src_w, int(src_h * target_aspect))
        crop_h = min(src_h, int(src_w / target_aspect))

        crop_x = (src_w - crop_w) // 2
        crop_y = (src_h - crop_h) // 2

        return CropPlan(
            strategy=CropStrategy.CENTER,
            src_x=crop_x,
            src_y=crop_y,
            src_w=crop_w,
            src_h=crop_h,
            dst_w=dst_w,
            dst_h=dst_h,
            confidence=0.1,
        )

    def _plan_last_known(
        self, src_w: int, src_h: int, dst_w: int, dst_h: int,
    ) -> CropPlan:
        """Use last known crop position (face temporarily lost)."""
        prev = self._prev_crop
        if prev is None:
            return self._plan_center(src_w, src_h, dst_w, dst_h)

        return CropPlan(
            strategy=CropStrategy.LAST_KNOWN,
            src_x=prev.src_x,
            src_y=prev.src_y,
            src_w=prev.src_w,
            src_h=prev.src_h,
            dst_w=dst_w,
            dst_h=dst_h,
            face_center_out=prev.face_center_out,
            confidence=max(0.1, prev.confidence * 0.9),
        )

    def _smooth(self, plan: CropPlan) -> CropPlan:
        """Apply EMA smoothing to crop position."""
        alpha = cfg.crop.smoothing_alpha
        max_vel = cfg.crop.max_crop_velocity

        if self._smooth_x is None:
            self._smooth_x = float(plan.src_x)
            self._smooth_y = float(plan.src_y)
            self._smooth_w = float(plan.src_w)
            self._smooth_h = float(plan.src_h)
            return plan

        # Clamp velocity
        dx = plan.src_x - self._smooth_x
        dy = plan.src_y - self._smooth_y
        dx = np.clip(dx, -max_vel, max_vel)
        dy = np.clip(dy, -max_vel, max_vel)

        # EMA smooth
        self._smooth_x += dx * alpha
        self._smooth_y += dy * alpha
        self._smooth_w = self._smooth_w * (1 - alpha) + plan.src_w * alpha
        self._smooth_h = self._smooth_h * (1 - alpha) + plan.src_h * alpha

        plan.src_x = int(self._smooth_x)
        plan.src_y = int(self._smooth_y)
        plan.src_w = int(self._smooth_w)
        plan.src_h = int(self._smooth_h)

        return plan

    def reset(self) -> None:
        """Reset crop state (call at start of each clip)."""
        self._prev_crop = None
        self._smooth_x = None
        self._smooth_y = None
        self._smooth_w = None
        self._smooth_h = None
        self._frames_without_face = 0


def apply_crop(
    frame: np.ndarray,
    plan: CropPlan,
) -> np.ndarray:
    """Apply a crop plan to a frame.

    Crops from source region and resizes to target dimensions.
    """
    src_h, src_w = frame.shape[:2]

    # Clamp crop region to frame bounds
    x = max(0, plan.src_x)
    y = max(0, plan.src_y)
    w = min(plan.src_w, src_w - x)
    h = min(plan.src_h, src_h - y)

    if w < 2 or h < 2:
        # Degenerate crop — return center crop as fallback
        target_aspect = plan.dst_w / plan.dst_h
        cw = min(src_w, int(src_h * target_aspect))
        ch = min(src_h, int(src_w / target_aspect))
        cx = (src_w - cw) // 2
        cy = (src_h - ch) // 2
        cropped = frame[cy:cy+ch, cx:cx+cw]
    else:
        cropped = frame[y:y+h, x:x+w]

    return cv2.resize(cropped, (plan.dst_w, plan.dst_h), interpolation=cv2.INTER_LANCZOS4)
