"""Benchmark Suite with Synthetic Clip Generators.

RULE 2: Categorized clips with per-clip metrics.
P0: Hard/adversarial clip generators for robustness testing.

Categories:
    EASY: frontal, stable lighting, low motion
    MEDIUM: moderate yaw, lighting shifts, slight blur
    HARD: fast motion, motion blur, profile rotation, occlusion, compression
    ADVERSARIAL: sunglasses, masks, webcam noise, low light, face partially offscreen
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Callable

import cv2
import numpy as np


class ClipCategory(Enum):
    """Benchmark clip categories."""
    EASY = auto()
    MEDIUM = auto()
    HARD = auto()
    ADVERSARIAL = auto()


@dataclass
class BenchmarkMetrics:
    """Per-clip benchmark metrics."""

    # Activation metrics
    physical_render_rate: float = 0.0
    alpha_fallback_rate: float = 0.0
    intrinsic_success_rate: float = 0.0
    avg_intrinsic_confidence: float = 0.0
    avg_decomposition_error: float = 0.0
    renderer_mode_transitions: int = 0

    # Quality metrics
    drift_score: float = 0.0
    flicker_score: float = 0.0
    geometric_consistency_score: float = 0.0

    # Failure analysis
    failure_reason_distribution: Dict[str, int] = field(default_factory=dict)

    # Timing
    total_frames: int = 0
    processing_time_s: float = 0.0
    fps: float = 0.0

    def to_dict(self) -> dict:
        return {
            "physical_render_rate": self.physical_render_rate,
            "alpha_fallback_rate": self.alpha_fallback_rate,
            "intrinsic_success_rate": self.intrinsic_success_rate,
            "avg_intrinsic_confidence": self.avg_intrinsic_confidence,
            "avg_decomposition_error": self.avg_decomposition_error,
            "renderer_mode_transitions": self.renderer_mode_transitions,
            "drift_score": self.drift_score,
            "flicker_score": self.flicker_score,
            "geometric_consistency_score": self.geometric_consistency_score,
            "failure_reason_distribution": self.failure_reason_distribution,
            "total_frames": self.total_frames,
            "processing_time_s": self.processing_time_s,
            "fps": self.fps,
        }


@dataclass
class BenchmarkClip:
    """A benchmark clip with metadata."""
    path: str
    category: ClipCategory
    description: str
    expected_difficulty: str
    generator: Optional[Callable] = None
    metrics: Optional[BenchmarkMetrics] = None


@dataclass
class BenchmarkSuite:
    """Full benchmark suite with categorized clips."""

    clips: List[BenchmarkClip] = field(default_factory=list)

    def add_clip(self, path: str, category: ClipCategory, description: str,
                 expected_difficulty: str, generator: Optional[Callable] = None) -> None:
        """Add a clip to the benchmark suite."""
        self.clips.append(BenchmarkClip(
            path=path,
            category=category,
            description=description,
            expected_difficulty=expected_difficulty,
            generator=generator,
        ))

    def get_clips_by_category(self, category: ClipCategory) -> List[BenchmarkClip]:
        """Get all clips in a category."""
        return [c for c in self.clips if c.category == category]

    def get_summary(self) -> dict:
        """Get summary of all benchmark results."""
        summary = {}
        for category in ClipCategory:
            cat_clips = self.get_clips_by_category(category)
            if not cat_clips:
                continue
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


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic Clip Generators
# ═══════════════════════════════════════════════════════════════════════════════

class SyntheticClipGenerator:
    """Generates synthetic test clips for benchmarking.

    Each generator creates a sequence of frames with specific conditions
    to test pipeline robustness.
    """

    def __init__(self, width: int = 640, height: int = 360, fps: float = 30.0):
        self.width = width
        self.height = height
        self.fps = fps

    def generate_easy_clip(self, num_frames: int = 90) -> List[np.ndarray]:
        """Generate easy clip: frontal face, stable lighting, low motion.

        Args:
            num_frames: Number of frames to generate

        Returns:
            List of frames (H, W, 3) uint8
        """
        frames = []
        for i in range(num_frames):
            frame = self._create_base_frame()
            # Minimal variation
            frame = self._add_slight_jitter(frame, i, amplitude=1)
            frames.append(frame)
        return frames

    def generate_medium_clip(self, num_frames: int = 90) -> List[np.ndarray]:
        """Generate medium clip: moderate yaw, lighting shifts, mild blur.

        Args:
            num_frames: Number of frames to generate

        Returns:
            List of frames (H, W, 3) uint8
        """
        frames = []
        for i in range(num_frames):
            frame = self._create_base_frame()
            # Moderate yaw rotation
            yaw_angle = 15 * np.sin(2 * np.pi * i / num_frames)
            frame = self._apply_rotation(frame, yaw_angle)
            # Lighting shift
            brightness = 0.8 + 0.4 * np.sin(2 * np.pi * i / num_frames)
            frame = self._apply_brightness(frame, brightness)
            # Mild blur on some frames
            if i % 5 == 0:
                frame = self._apply_blur(frame, kernel_size=3)
            frames.append(frame)
        return frames

    def generate_hard_clip(self, num_frames: int = 90) -> List[np.ndarray]:
        """Generate hard clip: fast motion, profile rotation, occlusion, compression.

        Args:
            num_frames: Number of frames to generate

        Returns:
            List of frames (H, W, 3) uint8
        """
        frames = []
        for i in range(num_frames):
            frame = self._create_base_frame()
            # Fast yaw rotation (up to 60 degrees)
            yaw_angle = 60 * np.sin(4 * np.pi * i / num_frames)
            frame = self._apply_rotation(frame, yaw_angle)
            # Aggressive lighting changes
            brightness = 0.5 + 0.8 * np.sin(6 * np.pi * i / num_frames)
            frame = self._apply_brightness(frame, brightness)
            # Motion blur on fast segments
            if abs(np.cos(4 * np.pi * i / num_frames)) > 0.7:
                frame = self._apply_blur(frame, kernel_size=7)
            # Compression artifacts
            if i % 3 == 0:
                frame = self._apply_compression(frame, quality=30)
            frames.append(frame)
        return frames

    def generate_adversarial_clip(self, num_frames: int = 90) -> List[np.ndarray]:
        """Generate adversarial clip: sunglasses, low light, face partially offscreen.

        Args:
            num_frames: Number of frames to generate

        Returns:
            List of frames (H, W, 3) uint8
        """
        frames = []
        for i in range(num_frames):
            frame = self._create_base_frame()
            # Low light conditions
            brightness = 0.2 + 0.1 * np.sin(2 * np.pi * i / num_frames)
            frame = self._apply_brightness(frame, brightness)
            # Heavy noise (webcam quality)
            frame = self._add_noise(frame, sigma=25)
            # Face partially offscreen in some frames
            if num_frames // 3 < i < 2 * num_frames // 3:
                frame = self._apply_crop_offset(frame, offset_x=50)
            # Sunglasses simulation (dark band over eyes)
            if i % 4 == 0:
                frame = self._add_sunglasses(frame)
            frames.append(frame)
        return frames

    def generate_occlusion_clip(self, num_frames: int = 90, occlusion_start: int = 30,
                                 occlusion_duration: int = 20) -> List[np.ndarray]:
        """Generate clip with face occlusion for testing prediction.

        Args:
            num_frames: Number of frames to generate
            occlusion_start: Frame where occlusion begins
            occlusion_duration: Number of frames face is occluded

        Returns:
            List of frames (H, W, 3) uint8
        """
        frames = []
        for i in range(num_frames):
            frame = self._create_base_frame()
            # Normal motion before/after occlusion
            yaw_angle = 10 * np.sin(2 * np.pi * i / num_frames)
            frame = self._apply_rotation(frame, yaw_angle)
            # Occlusion period
            if occlusion_start <= i < occlusion_start + occlusion_duration:
                frame = self._apply_full_occlusion(frame)
            frames.append(frame)
        return frames

    def generate_dropped_frames_clip(self, num_frames: int = 90,
                                      drop_every: int = 3) -> List[np.ndarray]:
        """Generate clip with simulated dropped frames.

        Args:
            num_frames: Number of frames to generate
            drop_every: Drop every Nth frame (replace with blank)

        Returns:
            List of frames (H, W, 3) uint8
        """
        frames = []
        for i in range(num_frames):
            frame = self._create_base_frame()
            yaw_angle = 15 * np.sin(2 * np.pi * i / num_frames)
            frame = self._apply_rotation(frame, yaw_angle)
            # Simulate dropped frame
            if i % drop_every == 0:
                frame = np.zeros_like(frame)
            frames.append(frame)
        return frames

    def generate_lighting_change_clip(self, num_frames: int = 90) -> List[np.ndarray]:
        """Generate clip with sudden lighting changes.

        Args:
            num_frames: Number of frames to generate

        Returns:
            List of frames (H, W, 3) uint8
        """
        frames = []
        for i in range(num_frames):
            frame = self._create_base_frame()
            # Sudden lighting changes every 15 frames
            if (i // 15) % 2 == 0:
                brightness = 1.0
            else:
                brightness = 0.3
            frame = self._apply_brightness(frame, brightness)
            frames.append(frame)
        return frames

    # ─── Helper methods ─────────────────────────────────────────────────────

    def _create_base_frame(self) -> np.ndarray:
        """Create a base frame with a simple face-like pattern."""
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 180

        # Draw a simple face oval
        center = (self.width // 2, self.height // 2)
        axes = (self.width // 4, self.height // 3)
        cv2.ellipse(frame, center, axes, 0, 0, 360, (200, 180, 160), -1)

        # Eyes
        eye_y = self.height // 3
        left_eye = (self.width // 3, eye_y)
        right_eye = (2 * self.width // 3, eye_y)
        cv2.circle(frame, left_eye, 8, (50, 50, 50), -1)
        cv2.circle(frame, right_eye, 8, (50, 50, 50), -1)

        # Nose
        nose = (self.width // 2, self.height // 2)
        cv2.circle(frame, nose, 5, (150, 130, 120), -1)

        # Mouth
        mouth = (self.width // 2, 2 * self.height // 3)
        cv2.ellipse(frame, mouth, (20, 10), 0, 0, 180, (100, 80, 80), 2)

        return frame

    def _apply_rotation(self, frame: np.ndarray, angle_deg: float) -> np.ndarray:
        """Apply rotation to simulate head yaw."""
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    def _apply_brightness(self, frame: np.ndarray, factor: float) -> np.ndarray:
        """Apply brightness change."""
        return np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    def _apply_blur(self, frame: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Apply Gaussian blur."""
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

    def _apply_compression(self, frame: np.ndarray, quality: int = 30) -> np.ndarray:
        """Apply JPEG compression artifacts."""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', frame, encode_param)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    def _add_noise(self, frame: np.ndarray, sigma: float = 20) -> np.ndarray:
        """Add Gaussian noise."""
        noise = np.random.randn(*frame.shape) * sigma
        return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def _add_slight_jitter(self, frame: np.ndarray, frame_idx: int,
                            amplitude: float = 2) -> np.ndarray:
        """Add slight positional jitter."""
        dx = amplitude * np.sin(frame_idx * 0.1)
        dy = amplitude * np.cos(frame_idx * 0.1)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]),
                              borderMode=cv2.BORDER_REFLECT)

    def _apply_crop_offset(self, frame: np.ndarray, offset_x: int = 50) -> np.ndarray:
        """Apply crop offset to simulate face partially offscreen."""
        h, w = frame.shape[:2]
        result = np.zeros_like(frame)
        src_x = max(0, offset_x)
        dst_x = max(0, -offset_x)
        copy_w = w - abs(offset_x)
        if copy_w > 0:
            result[:, dst_x:dst_x + copy_w] = frame[:, src_x:src_x + copy_w]
        return result

    def _add_sunglasses(self, frame: np.ndarray) -> np.ndarray:
        """Add sunglasses simulation (dark band over eye area)."""
        h, w = frame.shape[:2]
        y1 = h // 3 - 10
        y2 = h // 3 + 15
        x1 = w // 4
        x2 = 3 * w // 4
        result = frame.copy()
        result[y1:y2, x1:x2] = (30, 30, 30)
        return result

    def _apply_full_occlusion(self, frame: np.ndarray) -> np.ndarray:
        """Simulate full face occlusion (hand over face)."""
        h, w = frame.shape[:2]
        result = frame.copy()
        # Draw a "hand" over the face area
        center = (w // 2, h // 2)
        axes = (w // 3, h // 4)
        cv2.ellipse(result, center, axes, 0, 0, 360, (100, 80, 70), -1)
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Metric Computation Functions
# ═══════════════════════════════════════════════════════════════════════════════

def compute_drift_score(frames: List[np.ndarray]) -> float:
    """Compute identity drift score across frames.

    Drift = mean LAB distance between consecutive frames.

    Args:
        frames: List of output frames (H, W, 3) uint8

    Returns:
        Drift score (lower is better)
    """
    if len(frames) < 2:
        return 0.0

    drifts = []
    for i in range(1, len(frames)):
        lab1 = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2LAB).astype(np.float32)
        lab2 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2LAB).astype(np.float32)
        drift = float(np.mean(np.abs(lab1 - lab2)))
        drifts.append(drift)

    return float(np.mean(drifts))


def compute_flicker_score(frames: List[np.ndarray]) -> float:
    """Compute temporal flicker score.

    Flicker = std of frame-to-frame luminance changes.

    Args:
        frames: List of output frames (H, W, 3) uint8

    Returns:
        Flicker score (lower is better)
    """
    if len(frames) < 2:
        return 0.0

    lum_changes = []
    for i in range(1, len(frames)):
        gray1 = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray2 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
        change = float(np.mean(np.abs(gray2 - gray1)))
        lum_changes.append(change)

    return float(np.std(lum_changes))


def compute_geometric_consistency(transforms: list) -> float:
    """Compute geometric consistency score.

    Consistency = 1 - std(determinant) / mean(determinant)

    Args:
        transforms: List of SIM2Transform objects

    Returns:
        Consistency score [0, 1] (higher is better)
    """
    if len(transforms) < 2:
        return 1.0

    from face_os.lie_group import SIM2Transform
    dets = []
    for t in transforms:
        M = t.to_matrix()
        det = np.linalg.det(M[:2, :2])
        dets.append(det)

    dets = np.array(dets)
    mean_det = np.mean(dets)
    std_det = np.std(dets)

    if mean_det < 1e-8:
        return 0.0

    consistency = 1.0 - min(std_det / mean_det, 1.0)
    return float(max(0.0, consistency))


def create_default_suite() -> BenchmarkSuite:
    """Create default benchmark suite with all categories.

    Returns:
        BenchmarkSuite with clips for each category
    """
    suite = BenchmarkSuite()
    gen = SyntheticClipGenerator()

    # Easy clips
    suite.add_clip(
        path="synthetic_easy_frontal",
        category=ClipCategory.EASY,
        description="Frontal face, stable lighting, low motion",
        expected_difficulty="easy",
        generator=gen.generate_easy_clip,
    )

    # Medium clips
    suite.add_clip(
        path="synthetic_medium_yaw",
        category=ClipCategory.MEDIUM,
        description="Moderate yaw rotation, lighting shifts, mild blur",
        expected_difficulty="medium",
        generator=gen.generate_medium_clip,
    )

    # Hard clips
    suite.add_clip(
        path="synthetic_hard_motion",
        category=ClipCategory.HARD,
        description="Fast motion, profile rotation, occlusion, compression",
        expected_difficulty="hard",
        generator=gen.generate_hard_clip,
    )

    suite.add_clip(
        path="synthetic_hard_occlusion",
        category=ClipCategory.HARD,
        description="Face occlusion for 20 frames, then recovery",
        expected_difficulty="hard",
        generator=lambda: gen.generate_occlusion_clip(num_frames=90, occlusion_start=30, occlusion_duration=20),
    )

    suite.add_clip(
        path="synthetic_hard_dropped",
        category=ClipCategory.HARD,
        description="Dropped every 3rd frame",
        expected_difficulty="hard",
        generator=lambda: gen.generate_dropped_frames_clip(num_frames=90, drop_every=3),
    )

    # Adversarial clips
    suite.add_clip(
        path="synthetic_adversarial_lowlight",
        category=ClipCategory.ADVERSARIAL,
        description="Low light, webcam noise, sunglasses, face partially offscreen",
        expected_difficulty="adversarial",
        generator=gen.generate_adversarial_clip,
    )

    suite.add_clip(
        path="synthetic_adversarial_lighting",
        category=ClipCategory.ADVERSARIAL,
        description="Sudden lighting changes every 15 frames",
        expected_difficulty="adversarial",
        generator=gen.generate_lighting_change_clip,
    )

    return suite
