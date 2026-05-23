"""
crop_planner.py — Reference-Based Face-Aware Crop Planning.

BEAST MODE FIXES:
- Fixed the Snail-Cam velocity bug (EMA before velocity clamp).
- Fixed Alien-Face stretch bug in apply_crop (aspect ratio safeguard).
- Hardened edge-case math for zero-division and negative crops.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from face_os.config import get_config
from face_os.types import CropPlan, CropStrategy, FaceTrack, Landmarks

cfg = get_config()


class CompositionReference:
    """Composition metrics extracted from the expectation image."""

    def __init__(
        self,
        face_top_pct: float = 0.243,
        face_height_pct: float = 0.337,
        face_center_y_pct: float = 0.411,
        headroom_pct: float = 0.243,
    ):
        self.face_top_pct = face_top_pct
        self.face_height_pct = face_height_pct
        self.face_center_y_pct = face_center_y_pct
        self.headroom_pct = headroom_pct

    @classmethod
    def from_image(cls, image_path: str) -> 'CompositionReference':
        img = cv2.imread(image_path)
        if img is None:
            return cls()

        h, w = img.shape[:2]
        from face_os.detect_track import detect_faces
        detections = detect_faces(img)

        if not detections:
            return cls()

        track = detections[0]
        if track.smooth_bbox is None:
            return cls()
            
        x, y, fw, fh = track.smooth_bbox

        face_top_pct = y / max(h, 1)
        face_height_pct = fh / max(h, 1)
        face_center_y_pct = (y + fh / 2) / max(h, 1)
        headroom_pct = y / max(h, 1)

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


class CropPlanner:
    """Plans 9:16 crops from 16:9 source frames."""

    def __init__(self, reference_image: str = "expectation.png"):
        self.reference = CompositionReference.from_image(reference_image)
        print(f"  Crop reference: {self.reference}")

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
            if self._prev_crop:
                plan = self._plan_last_known(src_w, src_h, dst_w, dst_h)
            else:
                plan = self._plan_center(src_w, src_h, dst_w, dst_h)

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
        fx, fy, fw, fh = face_track.smooth_bbox
        face_cx = fx + fw // 2
        face_cy = fy + fh // 2

        target_face_ratio = max(self.reference.face_height_pct, 0.1)
        current_face_ratio = fh / max(src_h, 1)

        if current_face_ratio <= target_face_ratio:
            ideal_crop_h = int(fh / target_face_ratio)
            scale = dst_h / max(ideal_crop_h, 1)
            crop_w = int(dst_w / max(scale, 0.001))
            crop_h = ideal_crop_h
        else:
            crop_h = src_h
            scale = dst_h / max(crop_h, 1)
            crop_w = int(dst_w / max(scale, 0.001))

        crop_w = min(max(1, crop_w), src_w)
        crop_h = min(max(1, crop_h), src_h)

        target_aspect = dst_w / max(dst_h, 1)
        current_aspect = crop_w / max(crop_h, 1)

        if current_aspect > target_aspect:
            crop_w = int(crop_h * target_aspect)
        else:
            crop_h = int(crop_w / target_aspect)
            
        crop_w = max(1, min(crop_w, src_w))
        crop_h = max(1, min(crop_h, src_h))

        source_headroom = fy / max(src_h, 1)
        min_headroom = 0.15
        target_headroom = max(source_headroom, min_headroom)

        crop_y = int(fy - target_headroom * crop_h)
        crop_y = max(0, crop_y)
        crop_x = face_cx - crop_w // 2

        crop_x = max(0, min(crop_x, src_w - crop_w))
        crop_y = max(0, min(crop_y, src_h - crop_h))

        if landmarks and getattr(cfg.crop, 'protect_forehead', True):
            head_top = int(np.min(landmarks.points[:, 1]))
            min_crop_y = head_top - 10
            if crop_y > min_crop_y:
                crop_y = max(0, min_crop_y)
                if crop_y + crop_h > src_h:
                    crop_y = max(0, src_h - crop_h)

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

    def _plan_center(self, src_w: int, src_h: int, dst_w: int, dst_h: int) -> CropPlan:
        target_aspect = dst_w / max(dst_h, 1)
        crop_w = min(src_w, int(src_h * target_aspect))
        crop_h = min(src_h, int(src_w / target_aspect))
        crop_w = max(1, crop_w)
        crop_h = max(1, crop_h)

        crop_x = (src_w - crop_w) // 2
        crop_y = (src_h - crop_h) // 2

        return CropPlan(
            strategy=CropStrategy.CENTER,
            src_x=crop_x, src_y=crop_y,
            src_w=crop_w, src_h=crop_h,
            dst_w=dst_w, dst_h=dst_h,
            confidence=0.1,
        )

    def _plan_last_known(self, src_w: int, src_h: int, dst_w: int, dst_h: int) -> CropPlan:
        prev = self._prev_crop
        if prev is None:
            return self._plan_center(src_w, src_h, dst_w, dst_h)

        return CropPlan(
            strategy=CropStrategy.LAST_KNOWN,
            src_x=prev.src_x, src_y=prev.src_y,
            src_w=prev.src_w, src_h=prev.src_h,
            dst_w=dst_w, dst_h=dst_h,
            face_center_out=prev.face_center_out,
            confidence=max(0.1, prev.confidence * 0.9),
        )

    def _smooth(self, plan: CropPlan) -> CropPlan:
        alpha = getattr(cfg.crop, 'smoothing_alpha', 0.25)
        max_vel = getattr(cfg.crop, 'max_crop_velocity', 50)

        if self._smooth_x is None:
            self._smooth_x = float(plan.src_x)
            self._smooth_y = float(plan.src_y)
            self._smooth_w = float(plan.src_w)
            self._smooth_h = float(plan.src_h)
            return plan

        # BEAST MODE FIX: Apply EMA first, THEN clamp velocity.
        # Previous code clipped dx before alpha, making max speed = max_vel * alpha (Snail Cam).
        prev_x, prev_y = self._smooth_x, self._smooth_y
        
        self._smooth_x = self._smooth_x * (1 - alpha) + plan.src_x * alpha
        self._smooth_y = self._smooth_y * (1 - alpha) + plan.src_y * alpha
        self._smooth_w = self._smooth_w * (1 - alpha) + plan.src_w * alpha
        self._smooth_h = self._smooth_h * (1 - alpha) + plan.src_h * alpha

        dx = self._smooth_x - prev_x
        dy = self._smooth_y - prev_y
        
        dx = np.clip(dx, -max_vel, max_vel)
        dy = np.clip(dy, -max_vel, max_vel)
        
        self._smooth_x = prev_x + dx
        self._smooth_y = prev_y + dy

        plan.src_x = int(self._smooth_x)
        plan.src_y = int(self._smooth_y)
        plan.src_w = int(self._smooth_w)
        plan.src_h = int(self._smooth_h)

        return plan

    def reset(self) -> None:
        self._prev_crop = None
        self._smooth_x = None
        self._smooth_y = None
        self._smooth_w = None
        self._smooth_h = None
        self._frames_without_face = 0


def apply_crop(frame: np.ndarray, plan: CropPlan) -> np.ndarray:
    """Apply a crop plan to a frame."""
    src_h, src_w = frame.shape[:2]

    x = max(0, plan.src_x)
    y = max(0, plan.src_y)
    w = min(plan.src_w, src_w - x)
    h = min(plan.src_h, src_h - y)

    # BEAST MODE FIX: Prevent Alien-Face stretch.
    # If w or h got clamped by frame bounds, aspect ratio is broken.
    # Recalculate to maintain target aspect ratio.
    target_aspect = plan.dst_w / max(plan.dst_h, 1)
    current_aspect = w / max(h, 1)
    
    if w < 2 or h < 2 or abs(current_aspect - target_aspect) > 0.05:
        cw = min(src_w, int(src_h * target_aspect))
        ch = min(src_h, int(src_w / target_aspect))
        cw = max(2, cw)
        ch = max(2, ch)
        cx = (src_w - cw) // 2
        cy = (src_h - ch) // 2
        cropped = frame[cy:cy+ch, cx:cx+cw]
    else:
        cropped = frame[y:y+h, x:x+w]

    return cv2.resize(cropped, (plan.dst_w, plan.dst_h), interpolation=cv2.INTER_LANCZOS4)