"""
crop_objective.py - Deterministic objective-based crop planning.

Implements the crop objective described in face_os/arch.md:

    C* = argmin(E_crop)

where E_crop is composed of named alignment, cutoff, temporal, and composition
terms under hard geometry constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, gcd, isfinite
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from face_os.config import get_config
from face_os.types import CropPlan, CropStrategy, FaceTrack, Landmarks


_ENERGY_KEYS = (
    "face_alignment_error",
    "head_cutoff_penalty",
    "temporal_motion_penalty",
    "composition_error",
)


@dataclass
class CropObjectiveWeights:
    """Weights for the named crop objective terms."""

    face_alignment_error: float = 1.0
    head_cutoff_penalty: float = 5.0
    temporal_motion_penalty: float = 0.75
    composition_error: float = 1.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "face_alignment_error": float(self.face_alignment_error),
            "head_cutoff_penalty": float(self.head_cutoff_penalty),
            "temporal_motion_penalty": float(self.temporal_motion_penalty),
            "composition_error": float(self.composition_error),
        }


@dataclass
class CropObjectiveConfig:
    """Deterministic crop planner controls."""

    output_size: Tuple[int, int] = (1080, 1920)
    target_face_center: Tuple[float, float] = (0.50, 0.42)
    target_face_width_ratio: float = 0.25
    target_headroom_ratio: float = 0.24
    min_headroom_px: int = 12
    max_velocity_px: float = 50.0
    max_acceleration_px: float = 25.0
    weights: CropObjectiveWeights = field(default_factory=CropObjectiveWeights)

    @classmethod
    def from_runtime_config(cls) -> "CropObjectiveConfig":
        cfg = get_config()
        dst_w, dst_h = getattr(cfg.crop, "output_size", (1080, 1920))
        face_target_width = float(getattr(cfg.crop, "face_target_width", 270))
        target_face_width_ratio = face_target_width / max(float(dst_w), 1.0)
        return cls(
            output_size=(int(dst_w), int(dst_h)),
            target_face_width_ratio=float(np.clip(target_face_width_ratio, 0.05, 0.90)),
            target_headroom_ratio=float(getattr(cfg.crop, "headroom_ratio", 0.24)),
            max_velocity_px=float(getattr(cfg.crop, "max_crop_velocity", 50.0)),
        )


@dataclass
class CropObjectiveState:
    """Small temporal state for bounded velocity and acceleration."""

    previous_plan: Optional[CropPlan] = None
    previous_velocity: Tuple[float, float] = (0.0, 0.0)

    def update(self, plan: CropPlan) -> None:
        if self.previous_plan is None:
            self.previous_velocity = (0.0, 0.0)
        else:
            self.previous_velocity = (
                float(plan.src_x - self.previous_plan.src_x),
                float(plan.src_y - self.previous_plan.src_y),
            )
        self.previous_plan = plan

    def reset(self) -> None:
        self.previous_plan = None
        self.previous_velocity = (0.0, 0.0)


@dataclass(frozen=True)
class _Candidate:
    plan: CropPlan
    report: Dict[str, Any]
    score_tuple: Tuple[float, float, float, float, float, int, int, int]


class ObjectiveCropPlanner:
    """Plans crops by deterministic constrained energy minimization."""

    def __init__(
        self,
        config: Optional[CropObjectiveConfig] = None,
        state: Optional[CropObjectiveState] = None,
    ):
        self.config = config if config is not None else CropObjectiveConfig.from_runtime_config()
        self.state = state if state is not None else CropObjectiveState()

    def plan_crop(
        self,
        source_shape: Tuple[int, int],
        face_track: Optional[FaceTrack] = None,
        landmarks: Optional[Landmarks] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        update_state: bool = True,
    ) -> Tuple[CropPlan, Dict[str, Any]]:
        src_h, src_w = _source_hw(source_shape)
        dst_w, dst_h = self.config.output_size
        face_bbox = _resolve_bbox(face_track, bbox)

        if face_bbox is None:
            plan = self._center_plan(src_w, src_h, dst_w, dst_h)
            plan = self._bounded_plan(plan, src_w, src_h)
            report = self._empty_report(
                strategy=plan.strategy,
                constrained=True,
                face_observed=False,
            )
            if update_state:
                self.state.update(plan)
            return plan, report

        face_bbox = _clamp_bbox(face_bbox, src_w, src_h)
        candidates = list(self._candidates(src_w, src_h, dst_w, dst_h, face_bbox, landmarks))
        if not candidates:
            plan = self._center_plan(src_w, src_h, dst_w, dst_h)
            plan = self._bounded_plan(plan, src_w, src_h)
            report = self._empty_report(
                strategy=plan.strategy,
                constrained=False,
                face_observed=True,
            )
            report["constraint_violation"] = 1.0
            if update_state:
                self.state.update(plan)
            return plan, report

        best = min(candidates, key=lambda c: c.score_tuple)
        if update_state:
            self.state.update(best.plan)
        return best.plan, dict(best.report)

    def reset(self) -> None:
        self.state.reset()

    def _candidates(
        self,
        src_w: int,
        src_h: int,
        dst_w: int,
        dst_h: int,
        bbox: Tuple[int, int, int, int],
        landmarks: Optional[Landmarks],
    ) -> Iterable[_Candidate]:
        fx, fy, fw, fh = bbox
        face_right = fx + fw
        face_bottom = fy + fh
        face_cx = fx + fw / 2.0
        face_cy = fy + fh / 2.0
        head_top = _head_top(landmarks, fallback=fy)
        protected_top = max(0.0, head_top - float(self.config.min_headroom_px))

        unit_w, unit_h = _aspect_units(dst_w, dst_h)
        k_min = max(
            1,
            ceil(fw / unit_w),
            ceil((face_bottom - protected_top) / unit_h),
        )
        k_max = min(src_w // unit_w, src_h // unit_h)

        prev = self.state.previous_plan
        prev_vx, prev_vy = self.state.previous_velocity

        for k in range(k_min, k_max + 1):
            crop_w = unit_w * k
            crop_h = unit_h * k
            x_bounds = _containment_bounds(fx, face_right, crop_w, src_w)
            y_bounds = _containment_bounds(fy, face_bottom, crop_h, src_h)
            if x_bounds is None or y_bounds is None:
                continue

            y_bounds = (y_bounds[0], min(y_bounds[1], protected_top))
            if y_bounds[0] > y_bounds[1]:
                continue

            if prev is not None:
                x_bounds = _intersect_bounds(
                    x_bounds,
                    _temporal_position_bounds(
                        prev.src_x,
                        prev_vx,
                        self.config.max_velocity_px,
                        self.config.max_acceleration_px,
                    ),
                )
                y_bounds = _intersect_bounds(
                    y_bounds,
                    _temporal_position_bounds(
                        prev.src_y,
                        prev_vy,
                        self.config.max_velocity_px,
                        self.config.max_acceleration_px,
                    ),
                )
                if x_bounds is None or y_bounds is None:
                    continue

            ideal_x = face_cx - self.config.target_face_center[0] * crop_w
            ideal_y = face_cy - self.config.target_face_center[1] * crop_h
            crop_x = int(round(_clamp(ideal_x, x_bounds[0], x_bounds[1])))
            crop_y = int(round(_clamp(ideal_y, y_bounds[0], y_bounds[1])))

            plan = CropPlan(
                strategy=CropStrategy.FACE_LOCKED,
                src_x=crop_x,
                src_y=crop_y,
                src_w=crop_w,
                src_h=crop_h,
                dst_w=dst_w,
                dst_h=dst_h,
                face_center_out=(
                    int(round((face_cx - crop_x) * dst_w / max(crop_w, 1))),
                    int(round((face_cy - crop_y) * dst_h / max(crop_h, 1))),
                ),
                headroom_ratio=float((head_top - crop_y) / max(crop_h, 1)),
                confidence=1.0,
            )
            report = self._energy_report(plan, bbox, head_top, protected_top)
            score = report["total_energy"]
            score_tuple = (
                score,
                report["face_alignment_error"],
                report["head_cutoff_penalty"],
                report["temporal_motion_penalty"],
                report["composition_error"],
                plan.src_w,
                plan.src_y,
                plan.src_x,
            )
            yield _Candidate(plan=plan, report=report, score_tuple=score_tuple)

    def _energy_report(
        self,
        plan: CropPlan,
        bbox: Tuple[int, int, int, int],
        head_top: float,
        protected_top: float,
    ) -> Dict[str, Any]:
        fx, fy, fw, fh = bbox
        face_cx = fx + fw / 2.0
        face_cy = fy + fh / 2.0
        out_x = (face_cx - plan.src_x) / max(plan.src_w, 1)
        out_y = (face_cy - plan.src_y) / max(plan.src_h, 1)
        tx, ty = self.config.target_face_center
        face_alignment_error = (out_x - tx) ** 2 + (out_y - ty) ** 2

        cutoff_px = max(0.0, float(plan.src_y) - protected_top)
        head_cutoff_penalty = (cutoff_px / max(float(fh), 1.0)) ** 2

        temporal_motion_penalty = 0.0
        prev = self.state.previous_plan
        if prev is not None:
            dx = float(plan.src_x - prev.src_x)
            dy = float(plan.src_y - prev.src_y)
            prev_vx, prev_vy = self.state.previous_velocity
            vel = (dx / max(self.config.max_velocity_px, 1.0)) ** 2
            vel += (dy / max(self.config.max_velocity_px, 1.0)) ** 2
            acc = ((dx - prev_vx) / max(self.config.max_acceleration_px, 1.0)) ** 2
            acc += ((dy - prev_vy) / max(self.config.max_acceleration_px, 1.0)) ** 2
            temporal_motion_penalty = vel + acc

        face_width_ratio = fw / max(float(plan.src_w), 1.0)
        headroom_ratio = (head_top - plan.src_y) / max(float(plan.src_h), 1.0)
        composition_error = (face_width_ratio - self.config.target_face_width_ratio) ** 2
        composition_error += (headroom_ratio - self.config.target_headroom_ratio) ** 2

        terms = {
            "face_alignment_error": float(face_alignment_error),
            "head_cutoff_penalty": float(head_cutoff_penalty),
            "temporal_motion_penalty": float(temporal_motion_penalty),
            "composition_error": float(composition_error),
        }
        weights = self.config.weights.as_dict()
        total = sum(terms[key] * weights[key] for key in _ENERGY_KEYS)
        terms.update(
            {
                "total_energy": float(total),
                "weighted_total_energy": float(total),
                "strategy": plan.strategy.name,
                "face_observed": True,
                "constrained": True,
                "weights": weights,
            }
        )
        return terms

    def _center_plan(self, src_w: int, src_h: int, dst_w: int, dst_h: int) -> CropPlan:
        unit_w, unit_h = _aspect_units(dst_w, dst_h)
        k = max(1, min(src_w // unit_w, src_h // unit_h))
        crop_w = unit_w * k
        crop_h = unit_h * k
        return CropPlan(
            strategy=CropStrategy.CENTER,
            src_x=(src_w - crop_w) // 2,
            src_y=(src_h - crop_h) // 2,
            src_w=crop_w,
            src_h=crop_h,
            dst_w=dst_w,
            dst_h=dst_h,
            face_center_out=None,
            confidence=0.1,
        )

    def _bounded_plan(self, plan: CropPlan, src_w: int, src_h: int) -> CropPlan:
        prev = self.state.previous_plan
        if prev is None:
            return plan

        prev_vx, prev_vy = self.state.previous_velocity
        x_bounds = _intersect_bounds(
            (0.0, float(max(0, src_w - plan.src_w))),
            _temporal_position_bounds(
                prev.src_x,
                prev_vx,
                self.config.max_velocity_px,
                self.config.max_acceleration_px,
            ),
        )
        y_bounds = _intersect_bounds(
            (0.0, float(max(0, src_h - plan.src_h))),
            _temporal_position_bounds(
                prev.src_y,
                prev_vy,
                self.config.max_velocity_px,
                self.config.max_acceleration_px,
            ),
        )
        if x_bounds is not None:
            plan.src_x = int(round(_clamp(plan.src_x, x_bounds[0], x_bounds[1])))
        if y_bounds is not None:
            plan.src_y = int(round(_clamp(plan.src_y, y_bounds[0], y_bounds[1])))
        return plan

    def _empty_report(
        self,
        strategy: CropStrategy,
        constrained: bool,
        face_observed: bool,
    ) -> Dict[str, Any]:
        report = {key: 0.0 for key in _ENERGY_KEYS}
        report.update(
            {
                "total_energy": 0.0,
                "weighted_total_energy": 0.0,
                "strategy": strategy.name,
                "face_observed": face_observed,
                "constrained": constrained,
                "weights": self.config.weights.as_dict(),
            }
        )
        return report


def plan_objective_crop(
    source_shape: Tuple[int, int],
    face_track: Optional[FaceTrack] = None,
    landmarks: Optional[Landmarks] = None,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    state: Optional[CropObjectiveState] = None,
    config: Optional[CropObjectiveConfig] = None,
    update_state: bool = True,
) -> Tuple[CropPlan, Dict[str, Any]]:
    """Plan a deterministic objective crop and return its energy report."""

    planner = ObjectiveCropPlanner(config=config, state=state)
    return planner.plan_crop(
        source_shape=source_shape,
        face_track=face_track,
        landmarks=landmarks,
        bbox=bbox,
        update_state=update_state,
    )


def _source_hw(source_shape: Tuple[int, int]) -> Tuple[int, int]:
    if len(source_shape) < 2:
        raise ValueError("source_shape must contain at least height and width")
    src_h, src_w = int(source_shape[0]), int(source_shape[1])
    if src_h <= 0 or src_w <= 0:
        raise ValueError("source_shape dimensions must be positive")
    return src_h, src_w


def _resolve_bbox(
    face_track: Optional[FaceTrack],
    bbox: Optional[Tuple[int, int, int, int]],
) -> Optional[Tuple[int, int, int, int]]:
    if bbox is not None:
        return bbox
    if face_track is None:
        return None
    if face_track.smooth_bbox is not None:
        return face_track.smooth_bbox
    if face_track.detection is not None:
        return face_track.detection.bbox
    return None


def _clamp_bbox(
    bbox: Tuple[int, int, int, int],
    src_w: int,
    src_h: int,
) -> Tuple[int, int, int, int]:
    x, y, w, h = [int(round(v)) for v in bbox]
    x = int(_clamp(x, 0, src_w - 1))
    y = int(_clamp(y, 0, src_h - 1))
    right = int(_clamp(x + max(w, 1), x + 1, src_w))
    bottom = int(_clamp(y + max(h, 1), y + 1, src_h))
    return x, y, right - x, bottom - y


def _head_top(landmarks: Optional[Landmarks], fallback: float) -> float:
    if landmarks is None or landmarks.points is None:
        return float(fallback)
    points = np.asarray(landmarks.points, dtype=np.float32)
    if points.size == 0 or points.ndim < 2 or points.shape[1] < 2:
        return float(fallback)
    y_values = points[:, 1]
    finite = y_values[np.isfinite(y_values)]
    if finite.size == 0:
        return float(fallback)
    value = float(np.min(finite))
    return value if isfinite(value) else float(fallback)


def _aspect_units(dst_w: int, dst_h: int) -> Tuple[int, int]:
    dst_w = max(1, int(dst_w))
    dst_h = max(1, int(dst_h))
    divisor = gcd(dst_w, dst_h)
    return dst_w // divisor, dst_h // divisor


def _containment_bounds(
    face_start: int,
    face_end: int,
    crop_len: int,
    source_len: int,
) -> Optional[Tuple[float, float]]:
    lower = max(0, face_end - crop_len)
    upper = min(face_start, source_len - crop_len)
    if lower > upper:
        return None
    return float(lower), float(upper)


def _temporal_position_bounds(
    previous_pos: int,
    previous_velocity: float,
    max_velocity: float,
    max_acceleration: float,
) -> Tuple[float, float]:
    min_delta = max(-abs(max_velocity), previous_velocity - abs(max_acceleration))
    max_delta = min(abs(max_velocity), previous_velocity + abs(max_acceleration))
    return float(previous_pos) + min_delta, float(previous_pos) + max_delta


def _intersect_bounds(
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    lower = max(a[0], b[0])
    upper = min(a[1], b[1])
    if lower > upper:
        return None
    return lower, upper


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), float(lower)), float(upper))
