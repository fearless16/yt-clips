"""Benchmark Suite with Synthetic Clip Generators.

BEAST MODE FIXES:
- Added texture/noise to base frame so MediaPipe doesn't reject it as a blank wall.
- Fixed BORDER_REFLECT rotation artifacts (now uses BORDER_CONSTANT).
- Replaced O(N^3) matrix determinant with O(1) scale**2 math.
- Center-cropped Drift/Flicker metrics to ignore background rotation noise.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Callable

import cv2
import numpy as np


class ClipCategory(Enum):
    EASY = auto()
    MEDIUM = auto()
    HARD = auto()
    ADVERSARIAL = auto()


@dataclass
class BenchmarkMetrics:
    physical_render_rate: float = 0.0
    alpha_fallback_rate: float = 0.0
    intrinsic_success_rate: float = 0.0
    avg_intrinsic_confidence: float = 0.0
    avg_decomposition_error: float = 0.0
    renderer_mode_transitions: int = 0
    drift_score: float = 0.0
    flicker_score: float = 0.0
    geometric_consistency_score: float = 0.0
    failure_reason_distribution: Dict[str, int] = field(default_factory=dict)
    total_frames: int = 0
    processing_time_s: float = 0.0
    fps: float = 0.0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class BenchmarkClip:
    path: str
    category: ClipCategory
    description: str
    expected_difficulty: str
    generator: Optional[Callable] = None
    metrics: Optional[BenchmarkMetrics] = None


@dataclass
class BenchmarkSuite:
    clips: List[BenchmarkClip] = field(default_factory=list)

    def add_clip(self, path: str, category: ClipCategory, description: str,
                 expected_difficulty: str, generator: Optional[Callable] = None) -> None:
        self.clips.append(BenchmarkClip(
            path=path, category=category, description=description,
            expected_difficulty=expected_difficulty, generator=generator,
        ))

    def get_clips_by_category(self, category: ClipCategory) -> List[BenchmarkClip]:
        return [c for c in self.clips if c.category == category]

    def get_summary(self) -> dict:
        summary = {}
        for category in ClipCategory:
            cat_clips = self.get_clips_by_category(category)
            if not cat_clips: continue
            cat_metrics = [c.metrics for c in cat_clips if c.metrics is not None]
            if cat_metrics:
                summary[category.name] = {
                    "clip_count": len(cat_clips),
                    "metrics_count": len(cat_metrics),
                    "avg_physical_render_rate": float(np.mean([m.physical_render_rate for m in cat_metrics])),
                    "avg_drift_score": float(np.mean([m.drift_score for m in cat_metrics])),
                    "avg_flicker_score": float(np.mean([m.flicker_score for m in cat_metrics])),
                    "avg_geometric_consistency": float(np.mean([m.geometric_consistency_score for m in cat_metrics])),
                }
        return summary


class SyntheticClipGenerator:
    def __init__(self, width: int = 640, height: int = 360, fps: float = 30.0):
        self.width = width
        self.height = height
        self.fps = fps

    def generate_easy_clip(self, num_frames: int = 90) -> List[np.ndarray]:
        frames = []
        for i in range(num_frames):
            frame = self._create_base_frame()
            frame = self._add_slight_jitter(frame, i, amplitude=1)
            frames.append(frame)
        return frames

    def generate_medium_clip(self, num_frames: int = 90) -> List[np.ndarray]:
        frames = []
        for i in range(num_frames):
            frame = self._create_base_frame()
            yaw_angle = 15 * np.sin(2 * np.pi * i / num_frames)
            frame = self._apply_rotation(frame, yaw_angle)
            brightness = 0.8 + 0.4 * np.sin(2 * np.pi * i / num_frames)
            frame = self._apply_brightness(frame, brightness)
            if i % 5 == 0: frame = self._apply_blur(frame, kernel_size=3)
            frames.append(frame)
        return frames

    def generate_hard_clip(self, num_frames: int = 90) -> List[np.ndarray]:
        frames = []
        for i in range(num_frames):
            frame = self._create_base_frame()
            yaw_angle = 60 * np.sin(4 * np.pi * i / num_frames)
            frame = self._apply_rotation(frame, yaw_angle)
            brightness = 0.5 + 0.8 * np.sin(6 * np.pi * i / num_frames)
            frame = self._apply_brightness(frame, brightness)
            if abs(np.cos(4 * np.pi * i / num_frames)) > 0.7: frame = self._apply_blur(frame, kernel_size=7)
            if i % 3 == 0: frame = self._apply_compression(frame, quality=30)
            frames.append(frame)
        return frames

    def generate_adversarial_clip(self, num_frames: int = 90) -> List[np.ndarray]:
        frames = []
        for i in range(num_frames):
            frame = self._create_base_frame()
            brightness = 0.2 + 0.1 * np.sin(2 * np.pi * i / num_frames)
            frame = self._apply_brightness(frame, brightness)
            frame = self._add_noise(frame, sigma=25)
            if num_frames // 3 < i < 2 * num_frames // 3: frame = self._apply_crop_offset(frame, offset_x=50)
            if i % 4 == 0: frame = self._add_sunglasses(frame)
            frames.append(frame)
        return frames

    def generate_occlusion_clip(self, num_frames: int = 90, occlusion_start: int = 30, occlusion_duration: int = 20) -> List[np.ndarray]:
        frames = []
        for i in range(num_frames):
            frame = self._create_base_frame()
            yaw_angle = 10 * np.sin(2 * np.pi * i / num_frames)
            frame = self._apply_rotation(frame, yaw_angle)
            if occlusion_start <= i < occlusion_start + occlusion_duration: frame = self._apply_full_occlusion(frame)
            frames.append(frame)
        return frames

    def generate_dropped_frames_clip(self, num_frames: int = 90, drop_every: int = 3) -> List[np.ndarray]:
        frames = []
        for i in range(num_frames):
            frame = self._create_base_frame()
            yaw_angle = 15 * np.sin(2 * np.pi * i / num_frames)
            frame = self._apply_rotation(frame, yaw_angle)
            if i % drop_every == 0: frame = np.zeros_like(frame)
            frames.append(frame)
        return frames

    def generate_lighting_change_clip(self, num_frames: int = 90) -> List[np.ndarray]:
        frames = []
        for i in range(num_frames):
            frame = self._create_base_frame()
            brightness = 1.0 if (i // 15) % 2 == 0 else 0.3
            frame = self._apply_brightness(frame, brightness)
            frames.append(frame)
        return frames

    def _create_base_frame(self) -> np.ndarray:
        # BEAST MODE FIX: Added noise/texture so MediaPipe doesn't reject it as a flat poster.
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 180
        noise = np.random.normal(0, 8, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frame = cv2.GaussianBlur(frame, (3, 3), 0)

        center = (self.width // 2, self.height // 2)
        axes = (self.width // 4, self.height // 3)
        cv2.ellipse(frame, center, axes, 0, 0, 360, (200, 180, 160), -1)

        eye_y = self.height // 3
        left_eye = (self.width // 3, eye_y)
        right_eye = (2 * self.width // 3, eye_y)
        cv2.circle(frame, left_eye, 8, (50, 50, 50), -1)
        cv2.circle(frame, right_eye, 8, (50, 50, 50), -1)

        nose = (self.width // 2, self.height // 2)
        cv2.circle(frame, nose, 5, (150, 130, 120), -1)

        mouth = (self.width // 2, 2 * self.height // 3)
        cv2.ellipse(frame, mouth, (20, 10), 0, 0, 180, (100, 80, 80), 2)
        return frame

    def _apply_rotation(self, frame: np.ndarray, angle_deg: float) -> np.ndarray:
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        # BEAST MODE FIX: BORDER_CONSTANT prevents reflection artifacts that break MediaPipe
        return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    def _apply_brightness(self, frame: np.ndarray, factor: float) -> np.ndarray:
        return np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    def _apply_blur(self, frame: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

    def _apply_compression(self, frame: np.ndarray, quality: int = 30) -> np.ndarray:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', frame, encode_param)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    def _add_noise(self, frame: np.ndarray, sigma: float = 20) -> np.ndarray:
        noise = np.random.randn(*frame.shape) * sigma
        return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def _add_slight_jitter(self, frame: np.ndarray, frame_idx: int, amplitude: float = 2) -> np.ndarray:
        dx = amplitude * np.sin(frame_idx * 0.1)
        dy = amplitude * np.cos(frame_idx * 0.1)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_CONSTANT)

    def _apply_crop_offset(self, frame: np.ndarray, offset_x: int = 50) -> np.ndarray:
        h, w = frame.shape[:2]
        result = np.zeros_like(frame)
        src_x = max(0, offset_x)
        dst_x = max(0, -offset_x)
        copy_w = w - abs(offset_x)
        if copy_w > 0: result[:, dst_x:dst_x + copy_w] = frame[:, src_x:src_x + copy_w]
        return result

    def _add_sunglasses(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        y1, y2 = h // 3 - 10, h // 3 + 15
        x1, x2 = w // 4, 3 * w // 4
        result = frame.copy()
        result[y1:y2, x1:x2] = (30, 30, 30)
        return result

    def _apply_full_occlusion(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        result = frame.copy()
        center = (w // 2, h // 2)
        axes = (w // 3, h // 4)
        cv2.ellipse(result, center, axes, 0, 0, 360, (100, 80, 70), -1)
        return result


def _center_crop(frames: List[np.ndarray]) -> List[np.ndarray]:
    """BEAST MODE: Center crop 50% to ignore background rotation noise in metrics."""
    if not frames: return frames
    h, w = frames[0].shape[:2]
    cy, cx = h // 4, w // 4
    return [f[cy:h-cy, cx:w-cx] for f in frames]


def compute_drift_score(frames: List[np.ndarray]) -> float:
    if len(frames) < 2: return 0.0
    cropped = _center_crop(frames)
    drifts = []
    for i in range(1, len(cropped)):
        lab1 = cv2.cvtColor(cropped[i - 1], cv2.COLOR_BGR2LAB).astype(np.float32)
        lab2 = cv2.cvtColor(cropped[i], cv2.COLOR_BGR2LAB).astype(np.float32)
        drifts.append(float(np.mean(np.abs(lab1 - lab2))))
    return float(np.mean(drifts))


def compute_flicker_score(frames: List[np.ndarray]) -> float:
    if len(frames) < 2: return 0.0
    cropped = _center_crop(frames)
    lum_changes = []
    for i in range(1, len(cropped)):
        gray1 = cv2.cvtColor(cropped[i - 1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray2 = cv2.cvtColor(cropped[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
        lum_changes.append(float(np.mean(np.abs(gray2 - gray1))))
    return float(np.std(lum_changes))


def compute_geometric_consistency(transforms: list) -> float:
    if len(transforms) < 2: return 1.0
    dets = []
    for t in transforms:
        if hasattr(t, 'scale'):
            # BEAST MODE FIX: O(1) math instead of O(N^3) matrix det
            dets.append(t.scale ** 2)
        else:
            M = np.array(t)
            if M.shape in [(2, 3), (3, 3)]:
                dets.append(np.linalg.det(M[:2, :2]))
    
    if not dets: return 1.0
    dets = np.array(dets)
    mean_det, std_det = np.mean(dets), np.std(dets)
    if mean_det < 1e-8: return 0.0
    return float(max(0.0, 1.0 - min(std_det / mean_det, 1.0)))


def create_default_suite() -> BenchmarkSuite:
    suite = BenchmarkSuite()
    gen = SyntheticClipGenerator()

    suite.add_clip("synthetic_easy_frontal", ClipCategory.EASY, "Frontal face, stable lighting", "easy", gen.generate_easy_clip)
    suite.add_clip("synthetic_medium_yaw", ClipCategory.MEDIUM, "Moderate yaw, lighting shifts", "medium", gen.generate_medium_clip)
    suite.add_clip("synthetic_hard_motion", ClipCategory.HARD, "Fast motion, profile rotation", "hard", gen.generate_hard_clip)
    suite.add_clip("synthetic_hard_occlusion", ClipCategory.HARD, "Face occlusion", "hard", lambda: gen.generate_occlusion_clip())
    suite.add_clip("synthetic_hard_dropped", ClipCategory.HARD, "Dropped frames", "hard", lambda: gen.generate_dropped_frames_clip())
    suite.add_clip("synthetic_adversarial_lowlight", ClipCategory.ADVERSARIAL, "Low light, noise, sunglasses", "adversarial", gen.generate_adversarial_clip)
    suite.add_clip("synthetic_adversarial_lighting", ClipCategory.ADVERSARIAL, "Sudden lighting changes", "adversarial", gen.generate_lighting_change_clip)

    return suite