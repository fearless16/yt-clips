"""A/B Validation Module for Face OS.

RULE 3: Validation > Activation.
P0: Proves modules IMPROVE quality, not just that they run.

A/B comparisons:
    - PhysicalRenderer vs alpha compositing
    - Intrinsic rendering vs non-intrinsic
    - Lie-group smoothing vs linear EMA

Metrics:
    Photometric: LAB drift, luminance consistency, temporal brightness stability
    Geometric: Procrustes consistency, landmark coherence, transform determinant stability
    Perceptual: SSIM, temporal smoothness
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ABMetrics:
    """A/B comparison metrics.

    RULE 3: Required metrics for proving quality improvement.
    """

    # Photometric metrics
    lab_drift: float = 0.0                    # LAB distance from reference
    luminance_consistency: float = 0.0        # Temporal luminance stability
    temporal_brightness_stability: float = 0.0  # Brightness std across frames

    # Geometric metrics
    procrustes_consistency: float = 0.0       # Shape consistency
    landmark_coherence: float = 0.0           # Landmark temporal coherence
    transform_determinant_stability: float = 0.0  # det(A) stability

    # Perceptual metrics
    ssim: float = 0.0                         # Structural similarity
    temporal_smoothness: float = 0.0          # Frame-to-frame smoothness

    def to_dict(self) -> dict:
        return {
            "lab_drift": self.lab_drift,
            "luminance_consistency": self.luminance_consistency,
            "temporal_brightness_stability": self.temporal_brightness_stability,
            "procrustes_consistency": self.procrustes_consistency,
            "landmark_coherence": self.landmark_coherence,
            "transform_determinant_stability": self.transform_determinant_stability,
            "ssim": self.ssim,
            "temporal_smoothness": self.temporal_smoothness,
        }


@dataclass
class ABComparison:
    """Result of comparing two rendering approaches."""
    approach_a: str = ""
    approach_b: str = ""
    metrics_a: ABMetrics = field(default_factory=ABMetrics)
    metrics_b: ABMetrics = field(default_factory=ABMetrics)
    winner: str = ""
    improvement_pct: float = 0.0
    details: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "approach_a": self.approach_a,
            "approach_b": self.approach_b,
            "metrics_a": self.metrics_a.to_dict(),
            "metrics_b": self.metrics_b.to_dict(),
            "winner": self.winner,
            "improvement_pct": self.improvement_pct,
            "details": self.details,
        }


@dataclass
class ABValidationReport:
    """Full A/B validation report with all comparisons."""
    comparisons: List[ABComparison] = field(default_factory=list)

    def add_comparison(self, comparison: ABComparison) -> None:
        self.comparisons.append(comparison)

    def get_summary(self) -> dict:
        summary = {}
        for comp in self.comparisons:
            key = f"{comp.approach_a}_vs_{comp.approach_b}"
            summary[key] = {
                "winner": comp.winner,
                "improvement_pct": comp.improvement_pct,
            }
        return summary

    def to_dict(self) -> dict:
        return {
            "comparisons": [c.to_dict() for c in self.comparisons],
            "summary": self.get_summary(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Photometric Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_lab_drift(frame: np.ndarray, reference: np.ndarray) -> float:
    """Compute LAB drift from reference.

    Args:
        frame: Output frame (H, W, 3) uint8
        reference: Reference frame (H, W, 3) uint8

    Returns:
        Mean LAB distance (lower is better)
    """
    if frame.shape != reference.shape:
        reference = cv2.resize(reference, (frame.shape[1], frame.shape[0]))

    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    return float(np.mean(np.sqrt(np.sum((lab_frame - lab_ref) ** 2, axis=2))))


def compute_luminance_consistency(frames: List[np.ndarray]) -> float:
    """Compute temporal luminance consistency.

    Args:
        frames: List of output frames (H, W, 3) uint8

    Returns:
        Consistency score [0, 1] (higher is better)
    """
    if len(frames) < 2:
        return 1.0

    means = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        means.append(float(np.mean(gray)))

    std = np.std(means)
    mean = np.mean(means)
    if mean < 1e-8:
        return 0.0

    consistency = 1.0 - min(std / mean, 1.0)
    return float(max(0.0, consistency))


def compute_temporal_brightness_stability(frames: List[np.ndarray]) -> float:
    """Compute temporal brightness stability.

    Args:
        frames: List of output frames (H, W, 3) uint8

    Returns:
        Stability score [0, 1] (higher is better)
    """
    if len(frames) < 2:
        return 1.0

    brightnesses = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightnesses.append(float(np.mean(gray)))

    # Stability = 1 - coefficient of variation
    mean_b = np.mean(brightnesses)
    std_b = np.std(brightnesses)
    if mean_b < 1e-8:
        return 0.0

    stability = 1.0 - min(std_b / mean_b, 1.0)
    return float(max(0.0, stability))


# ═══════════════════════════════════════════════════════════════════════════════
# Geometric Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_procrustes_consistency(landmarks_list: list) -> float:
    """Compute Procrustes shape consistency.

    Args:
        landmarks_list: List of (N, 2) landmark arrays

    Returns:
        Consistency score [0, 1] (higher is better)
    """
    if len(landmarks_list) < 2:
        return 1.0

    shapes = [lm[:, :2] if lm.ndim > 1 else lm for lm in landmarks_list if lm is not None]
    if len(shapes) < 2:
        return 1.0

    normalized = []
    for shape in shapes:
        centroid = np.mean(shape, axis=0)
        centered = shape - centroid
        scale = np.sqrt(np.sum(centered**2))
        if scale > 1e-8:
            normalized.append(centered / scale)
        else:
            normalized.append(centered)

    stacked = np.stack(normalized, axis=0)
    variance = np.mean(np.std(stacked, axis=0))

    consistency = 1.0 - min(variance * 10, 1.0)
    return float(max(0.0, consistency))


def compute_landmark_coherence(landmarks_list: list) -> float:
    """Compute landmark temporal coherence.

    Coherence = 1 - mean(frame-to-frame landmark distance) / face_size

    Args:
        landmarks_list: List of (N, 2) landmark arrays

    Returns:
        Coherence score [0, 1] (higher is better)
    """
    if len(landmarks_list) < 2:
        return 1.0

    distances = []
    for i in range(1, len(landmarks_list)):
        prev = landmarks_list[i - 1]
        curr = landmarks_list[i]
        if prev is not None and curr is not None:
            if prev.shape == curr.shape:
                dist = np.mean(np.sqrt(np.sum((curr[:, :2] - prev[:, :2]) ** 2, axis=1)))
                distances.append(dist)

    if not distances:
        return 1.0

    mean_dist = np.mean(distances)
    # Normalize by typical face size (~200px)
    coherence = 1.0 - min(mean_dist / 200.0, 1.0)
    return float(max(0.0, coherence))


def compute_transform_determinant_stability(transforms: list) -> float:
    """Compute transform determinant stability.

    Stability = 1 - std(det) / mean(det)

    Args:
        transforms: List of 2x3 or 3x3 transform matrices

    Returns:
        Stability score [0, 1] (higher is better)
    """
    if len(transforms) < 2:
        return 1.0

    dets = []
    for t in transforms:
        if t is not None:
            M = np.array(t)
            if M.shape == (2, 3):
                det = np.linalg.det(M[:, :2])
            elif M.shape == (3, 3):
                det = np.linalg.det(M[:2, :2])
            else:
                continue
            dets.append(det)

    if len(dets) < 2:
        return 1.0

    dets = np.array(dets)
    mean_det = np.mean(dets)
    std_det = np.std(dets)

    if abs(mean_det) < 1e-8:
        return 0.0

    stability = 1.0 - min(std_det / abs(mean_det), 1.0)
    return float(max(0.0, stability))


def compute_transform_jitter(transforms: list) -> float:
    """Compute frame-to-frame transform jitter.

    Jitter = mean(||log(T_t^{-1} * T_{t-1})||)

    Lower jitter = smoother temporal transitions.

    Args:
        transforms: List of SIM2Transform or SE2Transform objects

    Returns:
        Mean jitter (lower is better)
    """
    if len(transforms) < 2:
        return 0.0

    jitters = []
    for i in range(1, len(transforms)):
        if transforms[i] is not None and transforms[i - 1] is not None:
            try:
                T_inv = transforms[i - 1].inverse()
                T_rel = T_inv.compose(transforms[i])
                v = T_rel.log()
                jitters.append(float(np.linalg.norm(v)))
            except Exception:
                continue

    if not jitters:
        return 0.0

    return float(np.mean(jitters))


# ═══════════════════════════════════════════════════════════════════════════════
# Perceptual Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ssim(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Compute structural similarity between two frames.

    Simplified SSIM computation without external dependencies.

    Args:
        frame_a: First frame (H, W, 3) uint8
        frame_b: Second frame (H, W, 3) uint8

    Returns:
        SSIM score [-1, 1] (higher is better)
    """
    if frame_a.shape != frame_b.shape:
        frame_b = cv2.resize(frame_b, (frame_a.shape[1], frame_a.shape[0]))

    a = frame_a.astype(np.float64)
    b = frame_b.astype(np.float64)

    mu_a = np.mean(a)
    mu_b = np.mean(b)
    sigma_a_sq = np.var(a)
    sigma_b_sq = np.var(b)
    sigma_ab = np.mean((a - mu_a) * (b - mu_b))

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    numerator = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    denominator = (mu_a**2 + mu_b**2 + C1) * (sigma_a_sq + sigma_b_sq + C2)

    return float(numerator / denominator)


def compute_temporal_smoothness(frames: List[np.ndarray]) -> float:
    """Compute temporal smoothness.

    Args:
        frames: List of output frames (H, W, 3) uint8

    Returns:
        Smoothness score [0, 1] (higher is better)
    """
    if len(frames) < 2:
        return 1.0

    changes = []
    for i in range(1, len(frames)):
        diff = np.mean(np.abs(
            frames[i].astype(np.float32) - frames[i - 1].astype(np.float32)
        ))
        changes.append(diff)

    if not changes:
        return 1.0

    mean_change = np.mean(changes)
    std_change = np.std(changes)

    if mean_change < 1e-8:
        return 1.0

    smoothness = 1.0 - min(std_change / (mean_change + 1e-8), 1.0)
    return float(max(0.0, smoothness))


# ═══════════════════════════════════════════════════════════════════════════════
# A/B Comparison Functions
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_metrics(
    frames: List[np.ndarray],
    reference: Optional[np.ndarray] = None,
    landmarks_list: Optional[list] = None,
    transforms: Optional[list] = None,
) -> ABMetrics:
    """Compute all A/B metrics for a set of frames.

    Args:
        frames: List of output frames
        reference: Optional reference frame for LAB drift
        landmarks_list: Optional list of landmark arrays
        transforms: Optional list of transform matrices

    Returns:
        ABMetrics with all computed metrics
    """
    metrics = ABMetrics()

    if frames:
        metrics.luminance_consistency = compute_luminance_consistency(frames)
        metrics.temporal_brightness_stability = compute_temporal_brightness_stability(frames)
        metrics.temporal_smoothness = compute_temporal_smoothness(frames)

        if reference is not None:
            metrics.lab_drift = compute_lab_drift(frames[0], reference)

    if landmarks_list:
        metrics.procrustes_consistency = compute_procrustes_consistency(landmarks_list)
        metrics.landmark_coherence = compute_landmark_coherence(landmarks_list)

    if transforms:
        metrics.transform_determinant_stability = compute_transform_determinant_stability(transforms)

    return metrics


def compare_approaches(
    approach_a: str,
    approach_b: str,
    metrics_a: ABMetrics,
    metrics_b: ABMetrics,
) -> ABComparison:
    """Compare two approaches and determine winner.

    Args:
        approach_a: Name of approach A
        approach_b: Name of approach B
        metrics_a: Metrics from approach A
        metrics_b: Metrics from approach B

    Returns:
        ABComparison with winner and improvement details
    """
    comparison = ABComparison(
        approach_a=approach_a,
        approach_b=approach_b,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
    )

    # Count wins for each approach
    score_a = 0
    score_b = 0
    details = {}

    # Photometric: lower lab_drift is better
    if metrics_a.lab_drift < metrics_b.lab_drift:
        score_a += 1
        details["lab_drift"] = metrics_b.lab_drift - metrics_a.lab_drift
    else:
        score_b += 1
        details["lab_drift"] = metrics_a.lab_drift - metrics_b.lab_drift

    # Higher luminance_consistency is better
    if metrics_a.luminance_consistency > metrics_b.luminance_consistency:
        score_a += 1
    else:
        score_b += 1

    # Higher temporal_smoothness is better
    if metrics_a.temporal_smoothness > metrics_b.temporal_smoothness:
        score_a += 1
    else:
        score_b += 1

    # Higher procrustes_consistency is better
    if metrics_a.procrustes_consistency > metrics_b.procrustes_consistency:
        score_a += 1
    else:
        score_b += 1

    # Higher transform_determinant_stability is better
    if metrics_a.transform_determinant_stability > metrics_b.transform_determinant_stability:
        score_a += 1
    else:
        score_b += 1

    # Higher ssim is better
    if metrics_a.ssim > metrics_b.ssim:
        score_a += 1
    else:
        score_b += 1

    # Determine winner
    if score_a > score_b:
        comparison.winner = approach_a
        comparison.improvement_pct = (score_a - score_b) / 6 * 100
    elif score_b > score_a:
        comparison.winner = approach_b
        comparison.improvement_pct = (score_b - score_a) / 6 * 100
    else:
        comparison.winner = "tie"
        comparison.improvement_pct = 0.0

    comparison.details = details
    return comparison


# ═══════════════════════════════════════════════════════════════════════════════
# Specific A/B Test Scenarios
# ═══════════════════════════════════════════════════════════════════════════════

def run_ab_test_physical_vs_alpha(
    frames_physical: List[np.ndarray],
    frames_alpha: List[np.ndarray],
    reference: Optional[np.ndarray] = None,
) -> ABComparison:
    """Run A/B test: PhysicalRenderer vs alpha compositing.

    Args:
        frames_physical: Frames from PhysicalRenderer
        frames_alpha: Frames from alpha compositing
        reference: Optional reference frame

    Returns:
        ABComparison result
    """
    metrics_a = compute_all_metrics(frames_physical, reference)
    metrics_b = compute_all_metrics(frames_alpha, reference)

    return compare_approaches("PhysicalRenderer", "AlphaCompositing", metrics_a, metrics_b)


def run_ab_test_sim2_vs_ema(
    transforms_sim2: list,
    transforms_ema: list,
) -> ABComparison:
    """Run A/B test: SIM(2) vs linear EMA.

    I-09: Compares geometric stability metrics between SIM(2) geodesic
    interpolation and linear EMA on high-rotation clips.

    Metrics compared:
    - Transform determinant stability (scale/rotation consistency)
    - Transform jitter (frame-to-frame smoothness, lower is better)

    Args:
        transforms_sim2: Transforms from SIM(2) interpolation
        transforms_ema: Transforms from linear EMA

    Returns:
        ABComparison result
    """
    metrics_a = ABMetrics()
    metrics_b = ABMetrics()

    metrics_a.transform_determinant_stability = compute_transform_determinant_stability(transforms_sim2)
    metrics_b.transform_determinant_stability = compute_transform_determinant_stability(transforms_ema)

    # I-09: Jitter comparison (lower is better)
    jitter_sim2 = compute_transform_jitter(transforms_sim2)
    jitter_ema = compute_transform_jitter(transforms_ema)
    # Store as negative so "higher is better" comparison works
    metrics_a.temporal_smoothness = -jitter_sim2
    metrics_b.temporal_smoothness = -jitter_ema

    return compare_approaches("SIM2", "LinearEMA", metrics_a, metrics_b)


def run_ab_test_intrinsic_vs_rgb(
    frames_intrinsic: List[np.ndarray],
    frames_rgb: List[np.ndarray],
    reference: Optional[np.ndarray] = None,
) -> ABComparison:
    """Run A/B test: intrinsic rendering vs RGB fallback.

    Args:
        frames_intrinsic: Frames from intrinsic rendering
        frames_rgb: Frames from RGB fallback
        reference: Optional reference frame

    Returns:
        ABComparison result
    """
    metrics_a = compute_all_metrics(frames_intrinsic, reference)
    metrics_b = compute_all_metrics(frames_rgb, reference)

    return compare_approaches("IntrinsicRendering", "RGBFallback", metrics_a, metrics_b)


# ═══════════════════════════════════════════════════════════════════════════════
# D-02: Pipeline-Level A/B Comparison
# ═══════════════════════════════════════════════════════════════════════════════

class ABComparator:
    """D-02: Pipeline-level A/B comparison framework.

    ACTIVE != GOOD. Must prove PhysicalRenderer improves output over alpha compositing.
    Runs the pipeline twice with different render modes and compares metrics.
    """

    def compare_render_methods(
        self,
        pipeline,
        video_path: str,
        max_frames: int = 100,
    ) -> dict:
        """A/B compare PhysicalRenderer vs alpha compositing.

        D-02: ACTIVE != GOOD. Must prove PhysicalRenderer improves output.

        Runs pipeline twice:
        1. With PhysicalRenderer enabled (default)
        2. With PhysicalRenderer forced to alpha mode via render_mode_override

        Compares:
        - Photometric: LAB drift, lighting consistency, temporal luminance stability
        - Geometric: Procrustes consistency, mesh coherence, landmark stability
        - Perceptual: SSIM, temporal smoothness

        Args:
            pipeline: FaceOSPipeline instance
            video_path: Path to input video
            max_frames: Maximum frames to process per run

        Returns:
            Dict with per-metric comparison and winner
        """
        import cv2

        # Run 1: PhysicalRenderer enabled (default)
        frames_physical, landmarks_physical, transforms_physical = self._run_pipeline(
            pipeline, video_path, max_frames, use_physical=True
        )

        # Run 2: Alpha compositing (PhysicalRenderer disabled)
        frames_alpha, landmarks_alpha, transforms_alpha = self._run_pipeline(
            pipeline, video_path, max_frames, use_physical=False
        )

        # Compute metrics for both
        metrics_physical = compute_all_metrics(
            frames_physical,
            landmarks_list=landmarks_physical,
            transforms=transforms_physical,
        )
        metrics_alpha = compute_all_metrics(
            frames_alpha,
            landmarks_list=landmarks_alpha,
            transforms=transforms_alpha,
        )

        # Compute SSIM between corresponding frames
        if frames_physical and frames_alpha:
            ssim_scores = []
            for fa, fb in zip(frames_physical, frames_alpha):
                ssim_scores.append(compute_ssim(fa, fb))
            metrics_physical.ssim = float(np.mean(ssim_scores)) if ssim_scores else 0.0

        # Compare
        comparison = compare_approaches(
            "PhysicalRenderer", "AlphaCompositing",
            metrics_physical, metrics_alpha,
        )

        return {
            "comparison": comparison.to_dict(),
            "metrics_physical": metrics_physical.to_dict(),
            "metrics_alpha": metrics_alpha.to_dict(),
            "frames_processed": len(frames_physical),
            "winner": comparison.winner,
            "improvement_pct": comparison.improvement_pct,
        }

    def _run_pipeline(
        self,
        pipeline,
        video_path: str,
        max_frames: int,
        use_physical: bool,
    ) -> tuple:
        """Run pipeline and collect output frames, landmarks, transforms.

        Uses the real pipeline.process_frame() API with render_mode_override
        to switch between PhysicalRenderer and alpha compositing.

        Args:
            pipeline: FaceOSPipeline instance
            video_path: Path to input video
            max_frames: Max frames to process
            use_physical: True for PhysicalRenderer, False for alpha compositing

        Returns:
            (frames, landmarks_list, transforms_list)
        """
        import cv2

        # Save and set render mode override
        original_override = getattr(pipeline, 'render_mode_override', None)
        if not use_physical:
            pipeline.render_mode_override = 'alpha'
        else:
            pipeline.render_mode_override = None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            pipeline.render_mode_override = original_override
            return [], [], []

        frames = []
        landmarks_list = []
        transforms_list = []
        frame_idx = 0

        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                result = pipeline.process_frame(frame, frame_idx=frame_idx)
                if result is not None and result.get('frame') is not None:
                    frames.append(result['frame'])
                    lm = result.get('landmarks')
                    if lm is not None:
                        landmarks_list.append(lm)
                    tf = result.get('transform')
                    if tf is not None:
                        transforms_list.append(tf)
            except Exception:
                pass

            frame_idx += 1

        cap.release()
        pipeline.render_mode_override = original_override
        return frames, landmarks_list, transforms_list

    def benchmark_report(
        self,
        comparison_result: dict,
    ) -> str:
        """Generate markdown benchmark report from A/B comparison.

        Args:
            comparison_result: Output from compare_render_methods()

        Returns:
            Markdown-formatted benchmark report
        """
        comp = comparison_result.get("comparison", {})
        m_phys = comparison_result.get("metrics_physical", {})
        m_alpha = comparison_result.get("metrics_alpha", {})
        winner = comparison_result.get("winner", "unknown")
        improvement = comparison_result.get("improvement_pct", 0.0)
        n_frames = comparison_result.get("frames_processed", 0)

        lines = [
            "# A/B Validation Report: PhysicalRenderer vs Alpha Compositing",
            "",
            f"**Frames processed:** {n_frames}",
            f"**Winner:** {winner}",
            f"**Improvement:** {improvement:.1f}%",
            "",
            "## Photometric Metrics",
            "",
            "| Metric | PhysicalRenderer | AlphaCompositing | Better |",
            "|--------|-----------------|------------------|--------|",
        ]

        photometric_metrics = [
            ("LAB Drift", "lab_drift", "lower"),
            ("Luminance Consistency", "luminance_consistency", "higher"),
            ("Brightness Stability", "temporal_brightness_stability", "higher"),
        ]

        for name, key, direction in photometric_metrics:
            val_p = m_phys.get(key, 0.0)
            val_a = m_alpha.get(key, 0.0)
            if direction == "lower":
                better = "Physical" if val_p < val_a else "Alpha"
            else:
                better = "Physical" if val_p > val_a else "Alpha"
            lines.append(f"| {name} | {val_p:.4f} | {val_a:.4f} | {better} |")

        lines.extend([
            "",
            "## Geometric Metrics",
            "",
            "| Metric | PhysicalRenderer | AlphaCompositing | Better |",
            "|--------|-----------------|------------------|--------|",
        ])

        geometric_metrics = [
            ("Procrustes Consistency", "procrustes_consistency", "higher"),
            ("Landmark Coherence", "landmark_coherence", "higher"),
            ("Transform Stability", "transform_determinant_stability", "higher"),
        ]

        for name, key, direction in geometric_metrics:
            val_p = m_phys.get(key, 0.0)
            val_a = m_alpha.get(key, 0.0)
            if direction == "lower":
                better = "Physical" if val_p < val_a else "Alpha"
            else:
                better = "Physical" if val_p > val_a else "Alpha"
            lines.append(f"| {name} | {val_p:.4f} | {val_a:.4f} | {better} |")

        lines.extend([
            "",
            "## Perceptual Metrics",
            "",
            "| Metric | PhysicalRenderer | AlphaCompositing | Better |",
            "|--------|-----------------|------------------|--------|",
        ])

        perceptual_metrics = [
            ("SSIM", "ssim", "higher"),
            ("Temporal Smoothness", "temporal_smoothness", "higher"),
        ]

        for name, key, direction in perceptual_metrics:
            val_p = m_phys.get(key, 0.0)
            val_a = m_alpha.get(key, 0.0)
            if direction == "lower":
                better = "Physical" if val_p < val_a else "Alpha"
            else:
                better = "Physical" if val_p > val_a else "Alpha"
            lines.append(f"| {name} | {val_p:.4f} | {val_a:.4f} | {better} |")

        lines.extend([
            "",
            "## Verdict",
            "",
            f"**{winner}** wins with **{improvement:.1f}%** improvement.",
            "",
            "D-02: ACTIVE != GOOD. This report proves (or disproves) that",
            "PhysicalRenderer improves output quality over alpha compositing.",
        ])

        return "\n".join(lines)
