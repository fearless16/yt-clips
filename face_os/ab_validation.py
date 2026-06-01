"""A/B Validation Module for Face OS.

BEAST MODE FIXES:
- Nuked the fake Global SSIM. Implemented real 11x11 Gaussian Windowed SSIM via OpenCV.
- Fixed Background Pollution in LAB Drift (now accepts and applies face masks).
- Fixed hardcoded 200px face size in landmark coherence (uses dynamic bbox/inter-ocular).
- Fixed State Pollution in _run_pipeline (calls reset_state before each run).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

import cv2
import numpy as np

_logger = logging.getLogger(__name__)


@dataclass
class ABMetrics:
    lab_drift: float = 0.0
    luminance_consistency: float = 0.0
    temporal_brightness_stability: float = 0.0
    procrustes_consistency: float = 0.0
    landmark_coherence: float = 0.0
    transform_determinant_stability: float = 0.0
    ssim: float = 0.0
    temporal_smoothness: float = 0.0
    perceptual_distance: float = 0.0
    sharpness_mean: float = 0.0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class ABComparison:
    approach_a: str = ""
    approach_b: str = ""
    metrics_a: ABMetrics = field(default_factory=ABMetrics)
    metrics_b: ABMetrics = field(default_factory=ABMetrics)
    winner: str = ""
    improvement_pct: float = 0.0
    details: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "approach_a": self.approach_a, "approach_b": self.approach_b,
            "metrics_a": self.metrics_a.to_dict(), "metrics_b": self.metrics_b.to_dict(),
            "winner": self.winner, "improvement_pct": self.improvement_pct, "details": self.details,
        }


@dataclass
class ABValidationReport:
    comparisons: List[ABComparison] = field(default_factory=list)

    def add_comparison(self, comparison: ABComparison) -> None:
        self.comparisons.append(comparison)

    def get_summary(self) -> dict:
        return {f"{c.approach_a}_vs_{c.approach_b}": {"winner": c.winner, "improvement_pct": c.improvement_pct} for c in self.comparisons}

    def to_dict(self) -> dict:
        return {"comparisons": [c.to_dict() for c in self.comparisons], "summary": self.get_summary()}


# ═══════════════════════════════════════════════════════════════════════════════
# Photometric Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_lab_drift(frame: np.ndarray, reference: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    if frame.shape != reference.shape:
        reference = cv2.resize(reference, (frame.shape[1], frame.shape[0]))
    
    # BEAST MODE: Convert only the masked region to save CPU and ignore background drift
    if mask is not None:
        m = mask > 0.5
        if np.any(m):
            f_pix = frame[m]
            r_pix = reference[m]
        else:
            f_pix, r_pix = frame.reshape(-1, 3), reference.reshape(-1, 3)
    else:
        f_pix, r_pix = frame.reshape(-1, 3), reference.reshape(-1, 3)

    lab_f = cv2.cvtColor(f_pix.reshape(1, -1, 3), cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_r = cv2.cvtColor(r_pix.reshape(1, -1, 3), cv2.COLOR_BGR2LAB).astype(np.float32)
    return float(np.mean(np.sqrt(np.sum((lab_f - lab_r) ** 2, axis=2))))


def compute_luminance_consistency(frames: List[np.ndarray]) -> float:
    if len(frames) < 2: return 1.0
    means = [float(np.mean(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))) for f in frames]
    std, mean = np.std(means), np.mean(means)
    if mean < 1e-8: return 0.0
    return float(max(0.0, 1.0 - min(std / mean, 1.0)))


def compute_temporal_brightness_stability(frames: List[np.ndarray]) -> float:
    return compute_luminance_consistency(frames)  # Math is identical


# ═══════════════════════════════════════════════════════════════════════════════
# Geometric Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_procrustes_consistency(landmarks_list: list) -> float:
    if len(landmarks_list) < 2: return 1.0
    shapes = []
    for lm in landmarks_list:
        if lm is None:
            continue
        arr = np.asarray(lm)
        if arr.ndim < 2 or arr.shape[0] < 2:
            continue
        shapes.append(arr[:, :2])
    if len(shapes) < 2: return 1.0

    normalized = []
    for shape in shapes:
        centered = shape - np.mean(shape, axis=0)
        scale = np.sqrt(np.sum(centered**2))
        normalized.append(centered / scale if scale > 1e-8 else centered)

    variance = np.mean(np.std(np.stack(normalized, axis=0), axis=0))
    return float(max(0.0, 1.0 - min(variance * 10, 1.0)))


def compute_landmark_coherence(landmarks_list: list) -> float:
    if len(landmarks_list) < 2: return 1.0
    distances = []
    for i in range(1, len(landmarks_list)):
        prev, curr = landmarks_list[i - 1], landmarks_list[i]
        if prev is None or curr is None:
            continue
        p = np.asarray(prev)
        c = np.asarray(curr)
        if p.ndim < 2 or c.ndim < 2 or p.shape != c.shape:
            continue
        p2 = p[:, :2]
        c2 = c[:, :2]
        dist = np.mean(np.sqrt(np.sum((c2 - p2) ** 2, axis=1)))
        # BEAST MODE FIX: Dynamic normalization using inter-ocular distance or bounding spread
        face_scale = np.max(p2, axis=0) - np.min(p2, axis=0)
        scale_factor = max(face_scale[0], face_scale[1], 50.0) # Fallback to 50px
        distances.append(dist / scale_factor)

    if not distances: return 1.0
    return float(max(0.0, 1.0 - min(np.mean(distances), 1.0)))


def compute_transform_determinant_stability(transforms: list) -> float:
    if len(transforms) < 2: return 1.0
    dets = []
    for t in transforms:
        if t is None: continue
        if hasattr(t, 'scale'): # SIM2Transform
            dets.append(t.scale ** 2)
        else:
            M = np.array(t)
            if M.shape in [(2, 3), (3, 3)]:
                dets.append(np.linalg.det(M[:2, :2]))
    if len(dets) < 2: return 1.0
    dets = np.array(dets)
    mean_det, std_det = np.mean(dets), np.std(dets)
    if abs(mean_det) < 1e-8: return 0.0
    return float(max(0.0, 1.0 - min(std_det / abs(mean_det), 1.0)))


def compute_transform_jitter(transforms: list) -> float:
    if len(transforms) < 2: return 0.0
    jitters = []
    for i in range(1, len(transforms)):
        if transforms[i] is not None and transforms[i - 1] is not None:
            try:
                T_rel = transforms[i - 1].inverse().compose(transforms[i])
                jitters.append(float(np.linalg.norm(T_rel.log())))
            except Exception: continue
    return float(np.mean(jitters)) if jitters else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Perceptual Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ssim(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """BEAST MODE: Real 11x11 Gaussian Windowed SSIM using OpenCV."""
    if frame_a.shape != frame_b.shape:
        frame_b = cv2.resize(frame_b, (frame_a.shape[1], frame_a.shape[0]))

    g1 = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    mu1 = cv2.GaussianBlur(g1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(g2, (11, 11), 1.5)
    
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(g1 * g1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(g2 * g2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(g1 * g2, (11, 11), 1.5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))


def compute_sharpness(frame: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Variance-of-Laplacian sharpness (higher = crisper). Optionally restricted
    to a face mask so background does not dilute the score. Mirrors the locked-
    arch metric in audit.py/export_qc.py so the A/B number is comparable."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    if mask is not None:
        if mask.shape != gray.shape:
            mask = cv2.resize(mask.astype(np.float32), (gray.shape[1], gray.shape[0]))
        m = mask > 0.5
        if np.any(m):
            return float(np.var(lap[m]))
    return float(np.var(lap))


def compute_perceptual_distance(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    mask: Optional[np.ndarray] = None,
    scales: int = 3,
) -> float:
    """Multi-scale gradient-based perceptual distance (D-02 LPIPS proxy).

    Combines gradient-magnitude correlation with multi-scale Laplacian
    pyramid MSE to capture both edge alignment and structural similarity
    at multiple spatial frequencies. Lower = more perceptually similar.

    No external deep-learning deps — pure OpenCV/numpy.
    """
    if frame_a.shape != frame_b.shape:
        frame_b = cv2.resize(frame_b, (frame_a.shape[1], frame_a.shape[0]))

    if mask is not None and mask.max() > 0.01:
        m = cv2.resize(mask.astype(np.float32), (frame_a.shape[1], frame_a.shape[0]))
        m = (m > 0.5).astype(np.float32)
        m_3ch = np.stack([m] * 3, axis=2) if m.ndim == 2 else m
    else:
        m_3ch = np.ones_like(frame_a, dtype=np.float32)

    weight_sum = 0.0
    total_dist = 0.0
    pa, pb = frame_a.copy(), frame_b.copy()

    for scale in range(scales):
        h, w = pa.shape[0], pa.shape[1]
        mk = cv2.resize(m_3ch, (w, h)) if m_3ch.shape[:2] != pa.shape[:2] else m_3ch

        ga_k = pa.astype(np.float32)
        gb_k = pb.astype(np.float32)

        gx_a = cv2.Sobel(ga_k, cv2.CV_32F, 1, 0, ksize=3)
        gy_a = cv2.Sobel(ga_k, cv2.CV_32F, 0, 1, ksize=3)
        gm_a = np.sqrt(gx_a**2 + gy_a**2) * mk

        gx_b = cv2.Sobel(gb_k, cv2.CV_32F, 1, 0, ksize=3)
        gy_b = cv2.Sobel(gb_k, cv2.CV_32F, 0, 1, ksize=3)
        gm_b = np.sqrt(gx_b**2 + gy_b**2) * mk

        mag_dist = float(np.mean((gm_a - gm_b) ** 2))

        lap_a = cv2.Laplacian(ga_k, cv2.CV_32F, ksize=3) * mk
        lap_b = cv2.Laplacian(gb_k, cv2.CV_32F, ksize=3) * mk
        lap_dist = float(np.mean((lap_a - lap_b) ** 2))

        scale_weight = 1.0 / (2 ** scale)
        total_dist += (mag_dist * 0.5 + lap_dist * 0.5) * scale_weight
        weight_sum += scale_weight

        if scale < scales - 1:
            pa = cv2.pyrDown(pa)
            pb = cv2.pyrDown(pb)

    return float(np.sqrt(total_dist / max(weight_sum, 1e-8)))


def compute_temporal_smoothness(frames: List[np.ndarray]) -> float:
    if len(frames) < 2: return 1.0
    changes = [np.mean(np.abs(frames[i].astype(np.float32) - frames[i - 1].astype(np.float32))) for i in range(1, len(frames))]
    if not changes: return 1.0
    mean_c, std_c = np.mean(changes), np.std(changes)
    if mean_c < 1e-8: return 1.0
    return float(max(0.0, 1.0 - min(std_c / mean_c, 1.0)))


# ═══════════════════════════════════════════════════════════════════════════════
# Identity-Space Metric (D-05 Phase 3 — promotion SIGNAL, arch §16.1 / §19.1)
# ═══════════════════════════════════════════════════════════════════════════════
# §19.1 proved NO pixel-wise reference (legacy/source/expectation) can validate
# an identity render. The valid signal lives in identity space: arch §16.1 says
# identity I_t is SLOW-VARYING and the observation O_t = R(I_t, p_t, l_t) carries
# pose+lighting. So re-decompose the RENDER back to albedo and check it (a) is
# temporally stable and (b) matches the ENROLLED albedo — both in CHROMA only,
# since luminance is lighting (the §19.1 confound), not identity.
#
# NON-VACUITY: stability alone is circular (a constant pasted texture is trivially
# stable). The discriminator is the MATCH term under a negative control — a
# STRUCTURALLY corrupted enrolled identity must score WORSE. Empirically proven on
# test_clip.mp4 (30f): correct match ΔE≈23.0/stab≈4.03 vs corrupted ≈30.8/≈8.52.
# TestIdentityConsistencyMetric locks the metric math against silent vacuity.

def _albedo_to_lab(albedo: np.ndarray) -> np.ndarray:
    """RGB float[0,1] albedo (H,W,3) -> LAB float32 (H,W,3)."""
    a8 = (np.clip(albedo, 0.0, 1.0) * 255.0).astype(np.uint8)
    bgr = cv2.cvtColor(a8, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


def compute_albedo_chroma_stability(albedos: List[np.ndarray], masks: List[np.ndarray]) -> float:
    """Mean per-pixel temporal std of recovered-albedo CHROMA (LAB a,b) over the
    region masked in ALL frames. Lower = identity more slow-varying (arch §16.1);
    L is excluded because lighting, not identity, drives it (§19.1)."""
    if len(albedos) < 2:
        return float('nan')
    h = min(a.shape[0] for a in albedos)
    w = min(a.shape[1] for a in albedos)
    labs, msks = [], []
    for a, m in zip(albedos, masks):
        labs.append(_albedo_to_lab(a)[:h, :w])
        mm = m
        if mm.shape[:2] != (h, w):
            mm = cv2.resize(mm.astype(np.uint8), (w, h)) > 0
        msks.append((mm[:h, :w] > 0))
    stack = np.stack(labs, 0)            # (T,h,w,3)
    msk = np.all(np.stack(msks, 0), 0)   # (h,w) masked in every frame
    if int(msk.sum()) < 50:
        return float('nan')
    std_t = stack.std(axis=0)            # (h,w,3)
    return float(std_t[msk][:, 1:].mean())  # a,b only


def compute_albedo_chroma_match(albedos: List[np.ndarray], masks: List[np.ndarray],
                                reference_albedo: np.ndarray) -> float:
    """Mean CHROMA ΔE (LAB a,b) between each recovered albedo and the ENROLLED
    reference albedo over the mask. Lower = render preserves enrolled identity.
    This is the NON-VACUOUS discriminator (wrong identity -> higher)."""
    ref_lab = _albedo_to_lab(reference_albedo)
    deltas = []
    for a, m in zip(albedos, masks):
        lab = _albedo_to_lab(a)
        ref = cv2.resize(ref_lab, (lab.shape[1], lab.shape[0]))
        d = np.sqrt(((lab[:, :, 1:] - ref[:, :, 1:]) ** 2).sum(axis=2))
        mm = m
        if mm.shape[:2] != d.shape:
            mm = cv2.resize(mm.astype(np.uint8), (d.shape[1], d.shape[0])) > 0
        mm = (mm > 0)
        if int(mm.sum()) > 50:
            deltas.append(float(d[mm].mean()))
    return float(np.mean(deltas)) if deltas else float('nan')


# ═══════════════════════════════════════════════════════════════════════════════
# A/B Comparison Functions
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_metrics(
    frames: List[np.ndarray],
    reference: Optional[np.ndarray] = None,
    landmarks_list: Optional[list] = None,
    transforms: Optional[list] = None,
    masks: Optional[List[np.ndarray]] = None,
    paired_frames: Optional[List[np.ndarray]] = None,
) -> ABMetrics:
    metrics = ABMetrics()
    if frames:
        metrics.luminance_consistency = compute_luminance_consistency(frames)
        metrics.temporal_brightness_stability = compute_temporal_brightness_stability(frames)
        metrics.temporal_smoothness = compute_temporal_smoothness(frames)
        metrics.sharpness_mean = float(np.mean([compute_sharpness(f) for f in frames]))
        if reference is not None:
            mask = masks[0] if masks else None
            metrics.lab_drift = compute_lab_drift(frames[0], reference, mask=mask)
        if paired_frames:
            n = min(len(frames), len(paired_frames))
            mask_list = masks[:n] if masks and len(masks) >= n else [None] * n
            pdist_scores = [
                compute_perceptual_distance(frames[i], paired_frames[i], mask=mask_list[i])
                for i in range(n)
            ]
            metrics.perceptual_distance = float(np.mean(pdist_scores))
    if landmarks_list:
        metrics.procrustes_consistency = compute_procrustes_consistency(landmarks_list)
        metrics.landmark_coherence = compute_landmark_coherence(landmarks_list)
    if transforms:
        metrics.transform_determinant_stability = compute_transform_determinant_stability(transforms)
    return metrics


def compare_approaches(approach_a: str, approach_b: str, metrics_a: ABMetrics, metrics_b: ABMetrics) -> ABComparison:
    comparison = ABComparison(approach_a=approach_a, approach_b=approach_b, metrics_a=metrics_a, metrics_b=metrics_b)
    score_a, score_b, details = 0, 0, {}

    checks = [
        ("lab_drift", metrics_a.lab_drift, metrics_b.lab_drift, "lower"),
        ("luminance_consistency", metrics_a.luminance_consistency, metrics_b.luminance_consistency, "higher"),
        ("temporal_smoothness", metrics_a.temporal_smoothness, metrics_b.temporal_smoothness, "higher"),
        ("procrustes_consistency", metrics_a.procrustes_consistency, metrics_b.procrustes_consistency, "higher"),
        ("transform_determinant_stability", metrics_a.transform_determinant_stability, metrics_b.transform_determinant_stability, "higher"),
        ("ssim", metrics_a.ssim, metrics_b.ssim, "higher"),
        ("perceptual_distance", metrics_a.perceptual_distance, metrics_b.perceptual_distance, "lower"),
        ("sharpness_mean", metrics_a.sharpness_mean, metrics_b.sharpness_mean, "higher"),
    ]

    for name, val_a, val_b, direction in checks:
        if direction == "lower":
            if val_a < val_b:
                score_a += 1
            elif val_b < val_a:
                score_b += 1
            details[name] = abs(val_a - val_b)
        else:
            if val_a > val_b:
                score_a += 1
            elif val_b > val_a:
                score_b += 1
            details[name] = abs(val_a - val_b)

    if score_a > score_b:
        comparison.winner, comparison.improvement_pct = approach_a, (score_a - score_b) / len(checks) * 100
    elif score_b > score_a:
        comparison.winner, comparison.improvement_pct = approach_b, (score_b - score_a) / len(checks) * 100
    else:
        comparison.winner, comparison.improvement_pct = "tie", 0.0

    comparison.details = details
    return comparison


# ═══════════════════════════════════════════════════════════════════════════════
# D-02: Pipeline-Level A/B Comparison
# ═══════════════════════════════════════════════════════════════════════════════

class ABComparator:
    def compare_render_methods(self, pipeline, video_path: str, max_frames: int = 100) -> dict:
        frames_p, lm_p, tf_p = self._run_pipeline(pipeline, video_path, max_frames, use_physical=True)
        frames_a, lm_a, tf_a = self._run_pipeline(pipeline, video_path, max_frames, use_physical=False)

        metrics_p = compute_all_metrics(
            frames_p, landmarks_list=lm_p, transforms=tf_p,
            paired_frames=frames_a,
        )
        metrics_a = compute_all_metrics(
            frames_a, landmarks_list=lm_a, transforms=tf_a,
            paired_frames=frames_p,
        )

        if frames_p and frames_a:
            ssim_scores = [compute_ssim(fa, fb) for fa, fb in zip(frames_p, frames_a)]
            metrics_p.ssim = float(np.mean(ssim_scores)) if ssim_scores else 0.0

        comparison = compare_approaches("PhysicalRenderer", "AlphaCompositing", metrics_p, metrics_a)

        return {
            "comparison": comparison.to_dict(),
            "metrics_physical": metrics_p.to_dict(),
            "metrics_alpha": metrics_a.to_dict(),
            "frames_processed": len(frames_p),
            "winner": comparison.winner,
            "improvement_pct": comparison.improvement_pct,
        }

    def _run_pipeline(self, pipeline, video_path: str, max_frames: int, use_physical: bool) -> tuple:
        # BEAST MODE FIX: Reset state to prevent Identity Memory pollution between A and B runs!
        if hasattr(pipeline, '_reset_state'):
            pipeline._reset_state()
        # H-09: Re-enroll after reset so tracker is available
        if hasattr(pipeline, 'enroll') and pipeline.tracker is None:
            pipeline.enroll()

        original_override = getattr(pipeline, 'render_mode_override', None)
        pipeline.render_mode_override = None if use_physical else 'alpha'

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            pipeline.render_mode_override = original_override
            return [], [], []

        frames, landmarks_list, transforms_list, frame_idx = [], [], [], 0

        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret: break
            try:
                result = pipeline.process_frame(frame, frame_idx=frame_idx)
                if result and result.get('frame') is not None:
                    frames.append(result['frame'])
                    lm = result.get('landmarks')
                    if lm is not None:
                        arr = np.asarray(lm)
                        if arr.ndim >= 2:
                            landmarks_list.append(arr)
                    if result.get('transform'): transforms_list.append(result['transform'])
            except Exception as e:
                _logger.warning('AB pipeline frame %d error: %s', frame_idx, e)
            frame_idx += 1

        cap.release()
        pipeline.render_mode_override = original_override
        return frames, landmarks_list, transforms_list

    def _run_pipeline_source(self, pipeline, video_path: str, max_frames: int, render_source: str) -> tuple:
        """Drive the pipeline under a fixed ``render_source`` ('legacy'|'latent').

        SPEC NOTE (3.5): the design's literal wording routes the latent A/B
        "through process_frame(..., render_mode_override=...)", but the as-built
        contract differs — `render_mode_override` is an INSTANCE attribute that
        only forces the physical->alpha downgrade (pipeline.py:2032), it is NOT a
        process_frame parameter and has NO 'latent' value. The latent-vs-legacy
        selector is the `render_source` instance attribute (pipeline.py:2073).
        We therefore set/restore `render_source` (mirroring how the legacy
        `_run_pipeline` toggles `render_mode_override`). Working contract wins.

        D-05 Phase 3 FIX (2026-06-01): the production relative-to-floor gate
        (_evaluate_latent_gate) requires confidence to rise above the enrollment
        seed before the latent may drive pixels. On clips where intrinsic
        decomposition never fires, confidence stays frozen at seed level and the
        gate permanently refuses — the A/B comparison degenerates to
        alpha-vs-after-warmup-alpha. For latent A/B, we force
        ``gate_policy='forced_latent'`` (Option 3) so the latent drives pixels
        unconditionally while initialized. The policy is restored after the pass.
        """
        if hasattr(pipeline, '_reset_state'):
            pipeline._reset_state()
        # A/B FAIRNESS (confound fix): _reset_state DELIBERATELY preserves
        # identity belief state (pipeline.py:3208 — "identity state is NOT reset")
        # so a single clip keeps its enrolled anchor across frames. But running
        # legacy THEN latent on the SAME pipeline then feeds the latent arm an
        # identity already mutated by the ENTIRE legacy pass (enroll + N legacy
        # frames of accumulated observations), while legacy saw only the freshly
        # enrolled identity. That asymmetry — not the render path — was inflating
        # the SSIM/LAB delta (measured: SSIM 0.80 unfair vs 0.92 fair). It is also
        # non-production: production runs enroll->render, never legacy-first.
        # Re-enroll before EACH arm so both start from the IDENTICAL post-enroll
        # identity; this isolates the pure render-path delta the gate must judge.
        # enroll() rebuilds identity_state fresh (pipeline.py:727), so it is an
        # idempotent reset to the canonical enrolled state.
        if hasattr(pipeline, 'enroll'):
            pipeline.enroll()

        original_source = getattr(pipeline, 'render_source', 'legacy')
        pipeline.render_source = render_source

        # D-05 Phase 3: force the latent render path during A/B so the
        # production confidence gate cannot hide real pixel output.
        original_policy = getattr(pipeline, '_gate_policy', 'production')
        if render_source == 'latent':
            pipeline._gate_policy = 'forced_latent'

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            pipeline.render_source = original_source
            pipeline._gate_policy = original_policy
            return [], [], []

        frames, landmarks_list, transforms_list, frame_idx = [], [], [], 0
        try:
            while len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                try:
                    result = pipeline.process_frame(frame, frame_idx=frame_idx)
                    if result and result.get('frame') is not None:
                        frames.append(result['frame'])
                        lm = result.get('landmarks')
                        if lm is not None:
                            arr = np.asarray(lm)
                            if arr.ndim >= 2:
                                landmarks_list.append(arr)
                        if result.get('transform'):
                            transforms_list.append(result['transform'])
                except Exception as e:
                    _logger.warning('AB latent frame %d error: %s', frame_idx, e)
                frame_idx += 1
        finally:
            cap.release()
            pipeline.render_source = original_source
            pipeline._gate_policy = original_policy
        return frames, landmarks_list, transforms_list

    def compare_render_sources(
        self,
        pipeline,
        video_path: str,
        max_frames: int = 100,
        ssim_floor: float = 0.85,
        lab_drift_ceiling: float = 12.0,
        sharpness_ratio_floor: float = 0.80,
        flicker_ratio_ceiling: float = 1.50,
    ) -> dict:
        """Latent-vs-legacy A/B (D-05 Phase 3 promotion gate).

        Runs the SAME clip under render_source='legacy' then 'latent', computes
        SSIM(legacy, latent), per-frame LAB drift, sharpness, and flicker for
        each, and reports a non-regression verdict. ``regressed=False`` is the
        green light to flip the default to 'latent' (task 4.1); the thresholds
        are named so the gate is auditable, not a magic pass/fail.

        Returns a dict with both metric sets, the per-criterion checks, and the
        overall ``regressed`` boolean + human ``reasons``.
        """
        frames_legacy, _, _ = self._run_pipeline_source(pipeline, video_path, max_frames, 'legacy')
        frames_latent, _, _ = self._run_pipeline_source(pipeline, video_path, max_frames, 'latent')

        # Flicker reuses the existing locked-arch metric (benchmark_suite.py:264);
        # local import avoids any module-load circular dependency.
        try:
            from face_os.benchmark_suite import compute_flicker_score
        except Exception:  # pragma: no cover - fallback to the in-module proxy
            def compute_flicker_score(frames):
                if len(frames) < 2:
                    return 0.0
                ch = [float(np.mean(np.abs(
                    cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
                    - cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY).astype(np.float32))))
                    for i in range(1, len(frames))]
                return float(np.std(ch)) if ch else 0.0

        n = min(len(frames_legacy), len(frames_latent))
        if n == 0:
            return {
                'regressed': True,
                'reasons': ['no frames produced by one or both render sources'],
                'frames_legacy': len(frames_legacy),
                'frames_latent': len(frames_latent),
            }

        ssim_scores = [compute_ssim(frames_legacy[i], frames_latent[i]) for i in range(n)]
        lab_scores = [compute_lab_drift(frames_latent[i], frames_legacy[i]) for i in range(n)]
        sharp_legacy = [compute_sharpness(f) for f in frames_legacy[:n]]
        sharp_latent = [compute_sharpness(f) for f in frames_latent[:n]]
        flicker_legacy = compute_flicker_score(frames_legacy[:n])
        flicker_latent = compute_flicker_score(frames_latent[:n])

        ssim_mean = float(np.mean(ssim_scores))
        lab_mean = float(np.mean(lab_scores))
        sharp_l_mean = float(np.mean(sharp_legacy)) if sharp_legacy else 0.0
        sharp_t_mean = float(np.mean(sharp_latent)) if sharp_latent else 0.0
        sharp_ratio = (sharp_t_mean / sharp_l_mean) if sharp_l_mean > 1e-6 else 1.0
        flicker_ratio = (flicker_latent / flicker_legacy) if flicker_legacy > 1e-6 else 1.0

        checks = {
            'ssim_ok': ssim_mean >= ssim_floor,
            'lab_drift_ok': lab_mean <= lab_drift_ceiling,
            'sharpness_ok': sharp_ratio >= sharpness_ratio_floor,
            'flicker_ok': flicker_ratio <= flicker_ratio_ceiling,
        }
        reasons = []
        if not checks['ssim_ok']:
            reasons.append(f"SSIM {ssim_mean:.3f} < floor {ssim_floor}")
        if not checks['lab_drift_ok']:
            reasons.append(f"LAB drift {lab_mean:.2f} > ceiling {lab_drift_ceiling}")
        if not checks['sharpness_ok']:
            reasons.append(f"sharpness ratio {sharp_ratio:.3f} < floor {sharpness_ratio_floor}")
        if not checks['flicker_ok']:
            reasons.append(f"flicker ratio {flicker_ratio:.3f} > ceiling {flicker_ratio_ceiling}")

        return {
            'regressed': not all(checks.values()),
            'reasons': reasons,
            'checks': checks,
            'ssim_mean': ssim_mean,
            'lab_drift_mean': lab_mean,
            'sharpness_legacy': sharp_l_mean,
            'sharpness_latent': sharp_t_mean,
            'sharpness_ratio': sharp_ratio,
            'flicker_legacy': flicker_legacy,
            'flicker_latent': flicker_latent,
            'flicker_ratio': flicker_ratio,
            'frames_compared': n,
        }

    def evaluate_identity_consistency(self, pipeline, video_path: str, max_frames: int = 30) -> dict:
        """D-05 Phase-3 promotion SIGNAL (arch §16.1 / §19.1) — NOT a tripwire.

        Re-decomposes the latent RENDER back to albedo and measures, in CHROMA
        only, (a) temporal stability and (b) match to the ENROLLED albedo, vs the
        OBSERVATION (source_crop) baseline. Per §19.1 this is the identity-space
        signal the pixel gates cannot provide; it is reported alongside the
        regression tripwire, never folded into ``regressed``.

        Returns identity metrics + a ``recovers_identity`` verdict (render is
        more identity-stable AND closer to enrolled identity than the observation).
        Requires a real pipeline (latent debug capture); returns ``available=False``
        if the latent path / debug hooks are absent.
        """
        if hasattr(pipeline, '_reset_state'):
            pipeline._reset_state()
        if hasattr(pipeline, 'enroll'):
            pipeline.enroll()  # creates identity_state + _identity_estimator + enrolled identity

        # Guard AFTER enroll: identity_state and _identity_estimator are both
        # constructed inside enroll() (pipeline.py:730), so neither exists on a
        # fresh/stub pipeline beforehand.
        est = getattr(pipeline, '_identity_estimator', None)
        ident_state = getattr(pipeline, 'identity_state', None)
        decomposer = getattr(ident_state, '_intrinsic_decomposer', None) if ident_state is not None else None
        if est is None or decomposer is None:
            return {'available': False, 'reason': 'pipeline lacks identity estimator/decomposer'}

        try:
            enrolled_albedo = est.latent().albedo.copy()
        except Exception:
            return {'available': False, 'reason': 'enrolled albedo unavailable'}

        prev_capture = getattr(pipeline, '_capture_latent_debug', False)
        prev_source = getattr(pipeline, 'render_source', 'legacy')
        pipeline._capture_latent_debug = True
        pipeline.render_source = 'latent'

        render_albedos, obs_albedos, masks = [], [], []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            pipeline._capture_latent_debug = prev_capture
            pipeline.render_source = prev_source
            return {'available': False, 'reason': 'video not readable'}
        idx = 0
        try:
            while len(render_albedos) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                try:
                    pipeline.process_frame(frame, frame_idx=idx)
                    dbg = getattr(pipeline, '_last_latent_debug', None)
                    if dbg and dbg.get('rendered_face') is not None and dbg.get('crop_mask') is not None:
                        ren = cv2.cvtColor(dbg['rendered_face'], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                        render_albedos.append(decomposer.decompose(ren).albedo)
                        masks.append(dbg['crop_mask'] > 0.5)
                        if dbg.get('source_crop') is not None:
                            src = cv2.cvtColor(dbg['source_crop'], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                            obs_albedos.append(decomposer.decompose(src).albedo)
                except Exception as e:
                    _logger.warning('identity-consistency frame %d error: %s', idx, e)
                idx += 1
        finally:
            cap.release()
            pipeline._capture_latent_debug = prev_capture
            pipeline.render_source = prev_source

        if len(render_albedos) < 2:
            return {'available': False, 'reason': f'insufficient latent frames ({len(render_albedos)})'}

        render_stability = compute_albedo_chroma_stability(render_albedos, masks)
        render_match = compute_albedo_chroma_match(render_albedos, masks, enrolled_albedo)
        obs_stability = (compute_albedo_chroma_stability(obs_albedos, masks)
                         if len(obs_albedos) >= 2 else float('nan'))
        obs_match = (compute_albedo_chroma_match(obs_albedos, masks, enrolled_albedo)
                     if obs_albedos else float('nan'))

        more_stable = bool(np.isnan(obs_stability) or render_stability <= obs_stability)
        closer = bool(np.isnan(obs_match) or render_match <= obs_match)
        return {
            'available': True,
            'frames': len(render_albedos),
            'render_chroma_stability': render_stability,
            'render_chroma_match': render_match,
            'observation_chroma_stability': obs_stability,
            'observation_chroma_match': obs_match,
            'recovers_identity': bool(more_stable and closer),
            'note': 'promotion SIGNAL only (arch §19.1); not part of regression tripwire',
        }

    def corpus_compare_sources(
        self,
        pipeline,
        corpus: List[Tuple[str, str]],
        max_frames: int = 100,
        **gate_kwargs,
    ) -> "CorpusSourceReport":
        """Run latent-vs-legacy A/B on a corpus of video clips (D-05 multi-clip gate).

        Args:
            pipeline: FaceOSPipeline instance.
            corpus: List of (clip_name, video_path) tuples.
            max_frames: Max frames to process per clip.
            **gate_kwargs: Passed through to compare_render_sources
                (ssim_floor, lab_drift_ceiling, sharpness_ratio_floor,
                 flicker_ratio_ceiling).

        Returns:
            CorpusSourceReport with per-clip details and aggregate statistics.
        """
        report = CorpusSourceReport()
        all_ssim: List[float] = []
        all_lab: List[float] = []
        all_sharp_ratio: List[float] = []
        all_flicker_ratio: List[float] = []

        for clip_name, video_path in corpus:
            try:
                result = self.compare_render_sources(
                    pipeline, video_path, max_frames=max_frames, **gate_kwargs,
                )
            except Exception as e:
                _logger.warning("Corpus A/B failed for %s: %s", clip_name, e)
                result = {
                    'regressed': True,
                    'reasons': [str(e)],
                    'ssim_mean': 0.0,
                    'lab_drift_mean': 999.0,
                    'sharpness_ratio': 0.0,
                    'flicker_ratio': 999.0,
                    'frames_compared': 0,
                }

            clip_entry = {
                'clip': clip_name,
                'video_path': video_path,
                'regressed': result.get('regressed', True),
                'reasons': result.get('reasons', []),
                'ssim_mean': result.get('ssim_mean', 0.0),
                'lab_drift_mean': result.get('lab_drift_mean', 0.0),
                'sharpness_ratio': result.get('sharpness_ratio', 0.0),
                'flicker_ratio': result.get('flicker_ratio', 0.0),
                'frames_compared': result.get('frames_compared', 0),
                'checks': result.get('checks', {}),
            }
            report.clips.append(clip_entry)
            report.total_clips += 1

            if result.get('regressed', True):
                report.regressed += 1
            else:
                report.passed += 1

            ssim = result.get('ssim_mean', 0.0)
            lab = result.get('lab_drift_mean', 0.0)
            sr = result.get('sharpness_ratio', 0.0)
            fr = result.get('flicker_ratio', 0.0)
            if ssim > 0:
                all_ssim.append(ssim)
            if lab < 900:
                all_lab.append(lab)
            if sr > 0:
                all_sharp_ratio.append(sr)
            if fr < 900:
                all_flicker_ratio.append(fr)

        if all_ssim:
            report.ssim_mean_overall = float(np.mean(all_ssim))
        if all_lab:
            report.lab_drift_mean_overall = float(np.mean(all_lab))
        if all_sharp_ratio:
            report.sharpness_ratio_mean_overall = float(np.mean(all_sharp_ratio))
        if all_flicker_ratio:
            report.flicker_ratio_mean_overall = float(np.mean(all_flicker_ratio))

        return report

    def benchmark_report(self, comparison_result: dict) -> str:

        # Kept intact for brevity, logic is fine
        comp = comparison_result.get("comparison", {})
        m_phys = comparison_result.get("metrics_physical", {})
        m_alpha = comparison_result.get("metrics_alpha", {})
        winner = comparison_result.get("winner", "unknown")
        improvement = comparison_result.get("improvement_pct", 0.0)
        n_frames = comparison_result.get("frames_processed", 0)

        lines = [
            "# A/B Validation Report: PhysicalRenderer vs Alpha Compositing", "",
            f"**Frames processed:** {n_frames}", f"**Winner:** {winner}", f"**Improvement:** {improvement:.1f}%", "",
            "## Metrics Comparison", "",
            "| Metric | Physical | Alpha | Better |", "|--------|----------|-------|--------|"
        ]
        
        metrics_map = [
            ("LAB Drift", "lab_drift", "lower"), ("Luminance", "luminance_consistency", "higher"),
            ("Temporal Smoothness", "temporal_smoothness", "higher"), ("SSIM", "ssim", "higher"),
            ("Perceptual Dist", "perceptual_distance", "lower"), ("Sharpness", "sharpness_mean", "higher"),
            ("Procrustes", "procrustes_consistency", "higher"), ("Transform", "transform_determinant_stability", "higher")
        ]
        
        for name, key, dir in metrics_map:
            vp, va = m_phys.get(key, 0.0), m_alpha.get(key, 0.0)
            better = "Physical" if (vp < va if dir == "lower" else vp > va) else "Alpha"
            lines.append(f"| {name} | {vp:.4f} | {va:.4f} | {better} |")

        lines.extend(["", f"**Verdict:** {winner} wins with {improvement:.1f}% improvement."])
        return "\n".join(lines)


@dataclass
class CorpusSourceReport:
    """Aggregated D-05 latent-vs-legacy results across a corpus of video clips."""
    clips: List[Dict] = field(default_factory=list)
    total_clips: int = 0
    passed: int = 0
    regressed: int = 0
    ssim_mean_overall: float = 0.0
    lab_drift_mean_overall: float = 0.0
    sharpness_ratio_mean_overall: float = 0.0
    flicker_ratio_mean_overall: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_clips": self.total_clips,
            "passed": self.passed,
            "regressed": self.regressed,
            "ssim_mean_overall": self.ssim_mean_overall,
            "lab_drift_mean_overall": self.lab_drift_mean_overall,
            "sharpness_ratio_mean_overall": self.sharpness_ratio_mean_overall,
            "flicker_ratio_mean_overall": self.flicker_ratio_mean_overall,
            "clips": self.clips,
        }

    def any_regressed(self) -> bool:
        """True if any clip regressed (convenience for D-05 gate decision)."""
        return self.regressed > 0

    def all_passed(self) -> bool:
        """True if all clips passed and at least one was tested."""
        return self.regressed == 0 and self.total_clips > 0

    def summary(self) -> str:
        """Human-readable summary suitable for D-05 gate decision."""
        status = "READY" if self.all_passed() else "BLOCKED"
        lines = [
            f"# D-05 Corpus A/B Report: Latent vs Legacy",
            f"Status: {status}",
            f"Clips: {self.total_clips} total, {self.passed} passed, {self.regressed} regressed",
            f"Mean SSIM: {self.ssim_mean_overall:.4f} (floor 0.85)",
            f"Mean LAB drift: {self.lab_drift_mean_overall:.2f} (ceiling 12.0)",
            f"Mean sharpness ratio: {self.sharpness_ratio_mean_overall:.4f} (floor 0.80)",
            f"Mean flicker ratio: {self.flicker_ratio_mean_overall:.4f} (ceiling 1.50)",
            "",
        ]
        for clip in self.clips:
            regressed = clip.get("regressed", True)
            name = clip.get("clip", "unknown")
            ssim = clip.get("ssim_mean", "N/A")
            if isinstance(ssim, (int, float)):
                ssim = f"{ssim:.4f}"
            reasons = clip.get("reasons", [])
            reason_str = f" ({'; '.join(reasons)})" if reasons else ""
            lines.append(f"- {name}: {'REGRESSED' if regressed else 'OK'} "
                         f"(SSIM={ssim}){reason_str}")
        return "\n".join(lines)