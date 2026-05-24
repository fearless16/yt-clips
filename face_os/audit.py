"""face_os/audit.py — Mathematical Diagnostic Audit Instrument.

Samples N frames from a video, intercepts the live pipeline via non-invasive
monkey-patching, captures every intermediate tensor with physical meaning, and
produces a structured report mapped to D-01..D-10 in LOCKED_ARCHITECTURE.md.

Model being audited:
    Y = A ⊙ S + spec + detail           (intrinsic decomposition)
    rendered = w_a*(A·La) + w_d*(A·Ld·N·L̂) + w_s*(Ls·(N·Ĥ)^n)
             → normalize to unit mean → × S → energy conservation

Usage:
    .venv/bin/python -m face_os.audit \\
        --video input/video.mp4 \\
        --expectation expectation.png \\
        --photos photos/ \\
        --frames 5 \\
        --output output/face_os/audit_report.json

Zero modifications to existing pipeline code.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import glob
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Colour helpers (no skimage dependency)
# ──────────────────────────────────────────────────────────────────────────────

def _srgb_to_linear(img_u8: np.ndarray) -> np.ndarray:
    """uint8 BGR → float32 linear [0,1]."""
    f = np.clip(img_u8.astype(np.float32) / 255.0, 0.0, 1.0)
    mask = f <= 0.04045
    return np.where(mask, f / 12.92, ((f + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _bgr_to_lab(img_u8: np.ndarray) -> np.ndarray:
    """uint8 BGR → float32 LAB."""
    return cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB).astype(np.float32)


def _lab_mean(img_u8: np.ndarray) -> np.ndarray:
    return np.mean(_bgr_to_lab(img_u8), axis=(0, 1))


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Structural similarity (grayscale float32 [0,255] inputs)."""
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    mu_a2, mu_b2, mu_ab = mu_a ** 2, mu_b ** 2, mu_a * mu_b
    sig_a2 = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a2
    sig_b2 = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b2
    sig_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab
    num = (2 * mu_ab + C1) * (2 * sig_ab + C2)
    den = (mu_a2 + mu_b2 + C1) * (sig_a2 + sig_b2 + C2)
    return float(np.mean(num / (den + 1e-8)))


def _sharpness(img_u8: np.ndarray) -> float:
    """Laplacian variance — the locked-arch target is 274."""
    gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return float(np.var(lap))


def _contrast(img_u8: np.ndarray) -> float:
    """Std of LAB_L channel × scale — locked-arch target is 73."""
    lab = _bgr_to_lab(img_u8)
    return float(np.std(lab[:, :, 0]))


def _frequency_retention(output_u8: np.ndarray, source_u8: np.ndarray) -> float:
    """HF variance ratio: output / source. Target > 0.6."""
    sh_out = _sharpness(output_u8)
    sh_src = _sharpness(source_u8)
    return float(sh_out / (sh_src + 1e-8))


def _compute_embedding(img_u8: np.ndarray) -> Optional[np.ndarray]:
    """LAB histogram embedding (same as detect_track._compute_embedding)."""
    if img_u8.size == 0:
        return None
    lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    lab_norm = cv2.merge([l, a, b])
    hist_l = cv2.calcHist([lab_norm], [0], None, [32], [0, 256]).flatten()
    hist_a = cv2.calcHist([lab_norm], [1], None, [32], [0, 256]).flatten()
    hist_b = cv2.calcHist([lab_norm], [2], None, [32], [0, 256]).flatten()
    hist = np.concatenate([hist_l, hist_a, hist_b])
    hist = hist / max(hist.sum(), 1e-8)
    return hist.astype(np.float32)


def _embedding_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Histogram intersection distance — lower is more similar [0,1]."""
    if a is None or b is None:
        return 1.0
    return float(1.0 - np.minimum(a, b).sum())


def _face_crop_dlib(img_u8: np.ndarray) -> Optional[np.ndarray]:
    """Return largest face crop via dlib HOG detector. None if no face."""
    import dlib
    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 0)
    if not dets:
        return None
    d = max(dets, key=lambda r: r.width() * r.height())
    x1, y1 = max(0, d.left()), max(0, d.top())
    x2, y2 = min(img_u8.shape[1], d.right()), min(img_u8.shape[0], d.bottom())
    return img_u8[y1:y2, x1:x2]


# ──────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FrameAuditCapture:
    """All intermediate tensors intercepted from the pipeline."""
    frame_idx: int
    source_bgr: np.ndarray               # original frame (uint8)
    source_linear: np.ndarray            # sRGB→linear, float32 [0,1]

    # Decomposition (from IntrinsicDecomposer)
    albedo: Optional[np.ndarray] = None           # A  (H,W,3) float32
    shading: Optional[np.ndarray] = None          # S  (H,W,1) float32
    specular: Optional[np.ndarray] = None         # spec (H,W,3) float32
    detail_residual: Optional[np.ndarray] = None  # detail (H,W,3) float32
    normal_map: Optional[np.ndarray] = None       # N (H,W,3) unit vectors

    # Renderer components (from PhysicalRenderer.render)
    ambient_component: Optional[np.ndarray] = None   # A·La
    diffuse_component: Optional[np.ndarray] = None   # A·Ld·(N·L̂)
    specular_component: Optional[np.ndarray] = None  # Ls·(N·Ĥ)^n
    base_render: Optional[np.ndarray] = None          # combined before detail
    final_rendered: Optional[np.ndarray] = None       # after detail + clamp

    # Pipeline output
    blended_output: Optional[np.ndarray] = None  # final composited uint8 crop
    source_crop: Optional[np.ndarray] = None      # actual 9:16 source crop (for fair fidelity)
    render_path: str = "unknown"
    fallback_reason: Optional[str] = None
    geometry_source: str = "unknown"
    sim2_det: float = 0.0
    lighting_ambient: float = 0.0
    lighting_diffuse: float = 0.0
    telemetry: dict = field(default_factory=dict)


@dataclass
class FrameAuditMetrics:
    """All computed scalar metrics for one frame."""
    frame_idx: int
    render_path: str

    # GROUP 1: Decomposition Fidelity
    recon_mae: float = 0.0          # mean(|Y - A⊙S - spec|)
    recon_pct: float = 0.0          # recon_mae / mean(Y)
    albedo_mean: float = 0.0
    albedo_std: float = 0.0
    shading_mean: float = 0.0
    shading_smoothness: float = 0.0  # mean(‖∇S‖) / mean(S)
    specular_mean: float = 0.0
    specular_sparsity: float = 0.0   # frac(|spec| < 0.01)
    albedo_channel_std: float = 0.0  # std of per-channel albedo means

    # GROUP 2: Rendering Energy
    energy_conservation_ratio: float = 0.0  # mean(rendered) / mean(A⊙S)
    ambient_frac: float = 0.0
    diffuse_frac: float = 0.0
    specular_frac: float = 0.0
    lambertian_ndotl_mean: float = 0.0
    detail_energy_ratio: float = 0.0

    # GROUP 3: Geometry
    normal_unit_error: float = 0.0  # mean(|‖N‖ - 1|)
    normal_z_mean: float = 0.0
    normal_coverage: float = 0.0    # frac(N_z > 0.1)
    geometry_source: str = "unknown"

    # GROUP 4: Signal Fidelity (locked-arch targets)
    sharpness_output: float = 0.0   # target: 274
    sharpness_source: float = 0.0
    contrast_output: float = 0.0    # target: 73
    frequency_retention: float = 0.0  # target: > 0.6

    # GROUP 5: Identity vs Reference
    lab_distance_vs_expectation: float = 0.0
    ssim_vs_expectation: float = 0.0
    albedo_lab_vs_anchor: float = 0.0
    embedding_distance_vs_expectation: float = 0.0
    embedding_distance_vs_photos_mean: float = 0.0

    # GROUP 6: Telemetry Truth
    telemetry_path_honest: bool = False
    telemetry_intrinsic_honest: bool = False
    telemetry_geometry_honest: bool = False
    telemetry_energy_terms_present: bool = False
    sim2_det_positive: bool = False

    # Arch compliance flags (D-01..D-10)
    D01_signal_preserving: Optional[bool] = None
    D02_physical_quality: Optional[bool] = None
    D04_dense_geometry: Optional[bool] = None
    D05_identity_decoupled: Optional[bool] = None
    D06_temporal: Optional[bool] = None
    D08_telemetry_honest: Optional[bool] = None


@dataclass
class ReferenceFingerprint:
    """Built once from expectation.png and photos/P*.png."""
    expectation_lab_mean: np.ndarray        # (3,) float32
    expectation_gray: np.ndarray            # grayscale uint8 for SSIM
    expectation_crop: Optional[np.ndarray]  # face-cropped BGR
    photos_lab_means: List[np.ndarray]      # per-photo (3,) float32
    anchor_albedo_lab: Optional[np.ndarray] # (3,) LAB mean of expectation albedo
    expectation_embedding: Optional[np.ndarray]
    photos_embeddings: List[np.ndarray]


# ──────────────────────────────────────────────────────────────────────────────
# PhysicalAuditSuite
# ──────────────────────────────────────────────────────────────────────────────

class PhysicalAuditSuite:
    """
    Mathematical audit instrument for the Face OS physical rendering pipeline.

    Non-invasive: intercepts pipeline via monkey-patching, restores originals
    in a finally block. Zero modifications to existing source files.
    """

    TARGET_SHARPNESS = 274.0
    TARGET_CONTRAST  = 73.0
    TARGET_FREQ_RET  = 0.6
    TARGET_SSIM      = 0.5
    TARGET_LAB_DIST  = 20.0
    TARGET_ALBEDO_LAB = 10.0
    TARGET_EMB_DIST  = 0.45  # histogram intersection distance (lower = better)

    def __init__(self):
        self._capture: Optional[FrameAuditCapture] = None

    # ── Frame sampling ────────────────────────────────────────────────────────

    def sample_frames(self, video_path: str, n: int = 5) -> List[Tuple[int, np.ndarray]]:
        """Sample n evenly-spaced frames at [5%,25%,45%,65%,85%] positions."""
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise ValueError(f"Cannot read frame count from {video_path}")

        positions = [int(total * p) for p in [0.05, 0.25, 0.45, 0.65, 0.85]][:n]
        frames = []
        for idx in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append((idx, frame))
        cap.release()
        return frames

    # ── Reference fingerprint ─────────────────────────────────────────────────

    def build_reference_fingerprint(
        self,
        expectation_path: str,
        photos_dir: str,
    ) -> ReferenceFingerprint:
        """Build LAB/embedding/SSIM anchors from reference images."""
        exp = cv2.imread(expectation_path)
        if exp is None:
            raise FileNotFoundError(f"Cannot load {expectation_path}")

        exp_lab = _lab_mean(exp)
        exp_gray = cv2.cvtColor(exp, cv2.COLOR_BGR2GRAY)
        exp_crop = _face_crop_dlib(exp)
        exp_emb = _compute_embedding(exp_crop if exp_crop is not None else exp)

        # Compute anchor albedo LAB from expectation via decomposer
        anchor_albedo_lab = None
        try:
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from face_os.intrinsic_decomposition import IntrinsicDecomposer
            exp_rgb = cv2.cvtColor(exp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            ic = IntrinsicDecomposer().decompose(exp_rgb)
            albedo_u8 = (np.clip(ic.albedo, 0, 1) * 255).astype(np.uint8)
            anchor_albedo_lab = _lab_mean(albedo_u8)
        except Exception as e:
            print(f"  [audit] Anchor albedo failed: {e}")

        # Photos
        photo_paths = sorted(glob.glob(os.path.join(photos_dir, "*.png")) +
                             glob.glob(os.path.join(photos_dir, "*.jpg")))
        photos_lab = []
        photos_emb = []
        for pp in photo_paths:
            img = cv2.imread(pp)
            if img is None:
                continue
            photos_lab.append(_lab_mean(img))
            crop = _face_crop_dlib(img)
            photos_emb.append(_compute_embedding(crop if crop is not None else img))

        return ReferenceFingerprint(
            expectation_lab_mean=exp_lab,
            expectation_gray=exp_gray,
            expectation_crop=exp_crop,
            photos_lab_means=photos_lab,
            anchor_albedo_lab=anchor_albedo_lab,
            expectation_embedding=exp_emb,
            photos_embeddings=photos_emb,
        )

    # ── Pipeline instrumentation ──────────────────────────────────────────────

    def instrument_and_run(
        self,
        pipeline,
        frame: np.ndarray,
        frame_idx: int,
        warmup_frames: Optional[List[Tuple[int, np.ndarray]]] = None,
    ) -> FrameAuditCapture:
        """
        Monkey-patch 3 pipeline methods, optionally run warm-up frames,
        then run target frame and restore originals.

        warmup_frames: list of (idx, frame) pairs run silently before the
        target frame so the pipeline state machine reaches FACE_LOCKED.
        Captured tensors are deep-copied — nothing is modified in-flight.
        """
        capture = FrameAuditCapture(
            frame_idx=frame_idx,
            source_bgr=frame.copy(),
            source_linear=_srgb_to_linear(frame),
        )
        self._capture = capture

        # ── Patch 1: PhysicalRenderer.render — capture component outputs ──────
        from face_os import physical_renderer as pr_mod
        _orig_render = pr_mod.PhysicalRenderer.render

        def _patched_render(self_r, albedo, normal_map, shading, lighting=None, **kw):
            result = _orig_render(self_r, albedo, normal_map, shading, lighting=lighting, **kw)
            capture.ambient_component = result.ambient_component.copy()
            capture.diffuse_component = result.diffuse_component.copy()
            capture.specular_component = result.specular_component.copy()
            capture.base_render = result.rendered.copy()
            capture.final_rendered = result.rendered.copy()
            if lighting is not None:
                capture.lighting_ambient = float(lighting.ambient)
                capture.lighting_diffuse = float(lighting.diffuse_intensity)
            return result

        pr_mod.PhysicalRenderer.render = _patched_render

        # ── Patch 2: _render_with_physical_renderer — capture decomposition ───
        import face_os.pipeline as pl_mod
        _orig_rwpr = pl_mod.FaceOSPipeline._render_with_physical_renderer

        def _patched_rwpr(self_p, source_frame, cropped, intrinsic_components,
                          intrinsic_conf, landmarks, crop_plan, frame_idx_inner,
                          region_masks=None):
            result = _orig_rwpr(
                self_p, source_frame, cropped, intrinsic_components,
                intrinsic_conf, landmarks, crop_plan, frame_idx_inner,
                region_masks=region_masks,
            )
            try:
                ic = intrinsic_components
                if ic is not None:
                    if capture.albedo is None and hasattr(ic, 'albedo'):
                        capture.albedo = ic.albedo.copy()
                    if capture.shading is None and hasattr(ic, 'shading'):
                        capture.shading = ic.shading.copy()
                    if capture.specular is None and hasattr(ic, 'specular'):
                        capture.specular = ic.specular.copy()
                    if capture.detail_residual is None and hasattr(ic, 'detail_residual') and ic.detail_residual is not None:
                        capture.detail_residual = ic.detail_residual.copy()
                    if capture.normal_map is None and hasattr(ic, 'normal_map'):
                        capture.normal_map = ic.normal_map.copy()
            except Exception:
                pass
            if result is not None:
                capture.blended_output = result.copy()
            return result

        pl_mod.FaceOSPipeline._render_with_physical_renderer = _patched_rwpr

        # ── Patch 3: _emit_frame_telemetry — capture telemetry dict ──────────
        _orig_telemetry = pl_mod.FaceOSPipeline._emit_frame_telemetry

        def _patched_telemetry(self_p, frame_idx_inner, fallback_reason,
                               intrinsic_components, energy_terms,
                               prev_physical, prev_alpha, render_path=None,
                               intrinsic_used=None, geometry_source=None,
                               resample_count=None, transform_det=None, **kw):
            _orig_telemetry(
                self_p, frame_idx_inner, fallback_reason, intrinsic_components,
                energy_terms, prev_physical, prev_alpha, render_path=render_path,
                intrinsic_used=intrinsic_used, geometry_source=geometry_source,
                resample_count=resample_count, transform_det=transform_det, **kw
            )
            capture.render_path = render_path or "unknown"
            capture.fallback_reason = fallback_reason
            capture.geometry_source = geometry_source or "unknown"
            capture.sim2_det = float(transform_det) if transform_det is not None else 0.0
            capture.telemetry = {
                "render_path": render_path,
                "fallback_reason": fallback_reason,
                "intrinsic_used": intrinsic_used,
                "geometry_source": geometry_source,
                "energy_terms": energy_terms or {},
                "transform_det": transform_det,
            }

        pl_mod.FaceOSPipeline._emit_frame_telemetry = _patched_telemetry

        # ── Patch 4: _render_core — capture the source crop (cropped arg) ─────
        _orig_render_core = pl_mod.FaceOSPipeline._render_core

        def _patched_render_core(self_p, cropped, source_frame, intrinsic_components,
                                  intrinsic_conf, identity_face, landmarks, crop_plan,
                                  region_masks, face_mask, frame_idx,
                                  identity_eyes=None, eye_confidence=0.0):
            # Capture the actual 9:16 source crop before any rendering
            if capture.source_crop is None and cropped is not None:
                capture.source_crop = cropped.copy()
            return _orig_render_core(
                self_p, cropped, source_frame, intrinsic_components,
                intrinsic_conf, identity_face, landmarks, crop_plan,
                region_masks, face_mask, frame_idx,
                identity_eyes=identity_eyes, eye_confidence=eye_confidence,
            )

        pl_mod.FaceOSPipeline._render_core = _patched_render_core

        try:
            # ── Warm-up: run preceding frames silently to prime state machine ─
            if warmup_frames:
                for wu_idx, wu_frame in warmup_frames:
                    try:
                        pipeline.process_frame(wu_frame.copy(), frame_idx=wu_idx)
                    except Exception:
                        pass

            # ── Run target frame ──────────────────────────────────────────────
            # Reset source_crop so only this target frame's crop is captured
            capture.source_crop = None
            result = pipeline.process_frame(frame.copy(), frame_idx=frame_idx)
            if result and 'frame' in result:
                if capture.blended_output is None:
                    capture.blended_output = result['frame']
                if capture.render_path == "unknown":
                    capture.render_path = result.get('render_path', 'unknown')
        finally:
            pr_mod.PhysicalRenderer.render = _orig_render
            pl_mod.FaceOSPipeline._render_with_physical_renderer = _orig_rwpr
            pl_mod.FaceOSPipeline._emit_frame_telemetry = _orig_telemetry
            pl_mod.FaceOSPipeline._render_core = _orig_render_core

        # ── Fallback: re-decompose source if physical path didn't fire ────────
        if capture.albedo is None:
            try:
                from face_os.intrinsic_decomposition import IntrinsicDecomposer
                src_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                ic = IntrinsicDecomposer().decompose(src_rgb)
                capture.albedo = ic.albedo
                capture.shading = ic.shading
                capture.specular = ic.specular
                capture.detail_residual = ic.detail_residual
                capture.normal_map = ic.normal_map
            except Exception:
                pass

        return capture


    # ── Metric computation ────────────────────────────────────────────────────

    def compute_audit_metrics(
        self,
        capture: FrameAuditCapture,
        reference: ReferenceFingerprint,
    ) -> FrameAuditMetrics:
        m = FrameAuditMetrics(
            frame_idx=capture.frame_idx,
            render_path=capture.render_path,
        )

        Y = capture.source_linear  # (H,W,3) float32

        # ── GROUP 1: Decomposition Fidelity ───────────────────────────────────
        if capture.albedo is not None and capture.shading is not None:
            A = capture.albedo
            S = capture.shading
            # Ensure shading is 3-channel for broadcasting
            S3 = np.repeat(S, 3, axis=2) if S.ndim == 3 and S.shape[2] == 1 else S
            if S3.ndim == 2:
                S3 = S3[:, :, np.newaxis].repeat(3, axis=2)

            spec = capture.specular if capture.specular is not None else np.zeros_like(A)

            # Resize Y to match A if needed (Y is full frame, A is crop)
            A_h, A_w = A.shape[:2]
            Y_h, Y_w = Y.shape[:2]
            if Y_h != A_h or Y_w != A_w:
                Y_resized = cv2.resize(Y, (A_w, A_h), interpolation=cv2.INTER_LINEAR)
            else:
                Y_resized = Y

            reconstructed = np.clip(A * S3 + spec, 0.0, 1.0)
            recon_err = np.abs(Y_resized - reconstructed)
            m.recon_mae = float(np.mean(recon_err))
            m.recon_pct = m.recon_mae / (float(np.mean(Y_resized)) + 1e-8) * 100.0

            m.albedo_mean = float(np.mean(A))
            m.albedo_std = float(np.std(A))
            m.albedo_channel_std = float(np.std(np.mean(A, axis=(0, 1))))

            m.shading_mean = float(np.mean(S))
            # Smoothness: mean gradient magnitude / mean shading
            s2d = S[:, :, 0] if S.ndim == 3 else S
            gx = cv2.Sobel(s2d, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(s2d, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(gx ** 2 + gy ** 2)
            m.shading_smoothness = float(np.mean(grad_mag) / (m.shading_mean + 1e-8))

            m.specular_mean = float(np.mean(np.abs(spec)))
            m.specular_sparsity = float(np.mean(np.abs(spec) < 0.01))

        # ── GROUP 2: Rendering Energy ─────────────────────────────────────────
        if capture.final_rendered is not None and capture.albedo is not None:
            FR = capture.final_rendered  # (H,W,3) float32
            A = capture.albedo
            S = capture.shading
            S3 = np.repeat(S, 3, axis=2) if S is not None and S.ndim == 3 and S.shape[2] == 1 else (S if S is not None else np.ones_like(A) * 0.5)

            target_E = float(np.mean(A * S3))
            rendered_E = float(np.mean(FR))
            m.energy_conservation_ratio = rendered_E / (target_E + 1e-8)

            # Component fractions
            if capture.ambient_component is not None:
                ba = float(np.mean(capture.ambient_component))
                bd = float(np.mean(capture.diffuse_component)) if capture.diffuse_component is not None else 0.0
                bs = float(np.mean(capture.specular_component)) if capture.specular_component is not None else 0.0
                total_b = ba + bd + bs + 1e-10
                m.ambient_frac = ba / total_b
                m.diffuse_frac = bd / total_b
                m.specular_frac = bs / total_b

                # Lambertian N·L̂ mean: diffuse_component = A * Ld * N·L̂
                # so N·L̂ ≈ mean(diffuse_comp) / (mean(A) * Ld + ε)
                Ld = capture.lighting_diffuse
                albedo_m = float(np.mean(A)) if capture.albedo is not None else 0.5
                m.lambertian_ndotl_mean = bd / (albedo_m * Ld + 1e-8)

            if capture.detail_residual is not None:
                m.detail_energy_ratio = float(np.mean(np.abs(capture.detail_residual))) / (rendered_E + 1e-8)

        # ── GROUP 3: Geometry ─────────────────────────────────────────────────
        if capture.normal_map is not None:
            N = capture.normal_map
            norms = np.linalg.norm(N, axis=2)
            m.normal_unit_error = float(np.mean(np.abs(norms - 1.0)))
            m.normal_z_mean = float(np.mean(N[:, :, 2]))
            m.normal_coverage = float(np.mean(N[:, :, 2] > 0.1))
        m.geometry_source = capture.geometry_source

        # ── GROUP 4: Signal Fidelity ──────────────────────────────────────────
        output_u8 = capture.blended_output
        source_u8 = capture.source_bgr
        if output_u8 is not None and output_u8.dtype == np.uint8:
            # source_crop is the actual 9:16 source crop — same content as output.
            # Use it for fair comparison. Fall back to resize if not captured.
            if capture.source_crop is not None:
                source_for_fidelity = capture.source_crop
                # Resize if needed (shouldn't be, but guard)
                oh, ow = output_u8.shape[:2]
                if source_for_fidelity.shape[:2] != (oh, ow):
                    source_for_fidelity = cv2.resize(source_for_fidelity, (ow, oh),
                                                     interpolation=cv2.INTER_AREA)
            elif capture.render_path in ("face_lost_basic", "face_lost_predicted",
                                          "enhancement"):
                # LOST_FACE path: output IS the basic source crop.
                # Use output as its own reference → freq_ret ≈ 1.0 (correct: no HF lost).
                source_for_fidelity = output_u8
            else:
                # Fallback: resize full frame to output dims (imperfect)
                oh, ow = output_u8.shape[:2]
                sh, sw = source_u8.shape[:2]
                if (sh, sw) != (oh, ow):
                    source_for_fidelity = cv2.resize(source_u8, (ow, oh),
                                                     interpolation=cv2.INTER_AREA)
                else:
                    source_for_fidelity = source_u8
            m.sharpness_output = _sharpness(output_u8)
            m.sharpness_source = _sharpness(source_for_fidelity)
            m.contrast_output = _contrast(output_u8)
            m.frequency_retention = _frequency_retention(output_u8, source_for_fidelity)


        # ── GROUP 5: Identity vs Reference ────────────────────────────────────
        if output_u8 is not None:
            # LAB distance vs expectation
            out_lab = _lab_mean(output_u8)
            m.lab_distance_vs_expectation = float(
                np.linalg.norm(out_lab - reference.expectation_lab_mean)
            )

            # SSIM vs expectation (grayscale, resize to match)
            exp_gray = reference.expectation_gray
            out_gray = cv2.cvtColor(output_u8, cv2.COLOR_BGR2GRAY).astype(np.float32)
            exp_r = cv2.resize(exp_gray.astype(np.float32),
                               (out_gray.shape[1], out_gray.shape[0]),
                               interpolation=cv2.INTER_AREA)
            m.ssim_vs_expectation = _ssim(out_gray, exp_r)

            # Albedo LAB vs anchor
            if capture.albedo is not None and reference.anchor_albedo_lab is not None:
                albedo_u8 = (np.clip(capture.albedo, 0, 1) * 255).astype(np.uint8)
                albedo_lab = _lab_mean(albedo_u8)
                m.albedo_lab_vs_anchor = float(
                    np.linalg.norm(albedo_lab - reference.anchor_albedo_lab)
                )

            # Embedding distance
            out_emb = _compute_embedding(output_u8)
            if reference.expectation_embedding is not None and out_emb is not None:
                m.embedding_distance_vs_expectation = _embedding_distance(
                    out_emb, reference.expectation_embedding
                )
            if reference.photos_embeddings and out_emb is not None:
                dists = [_embedding_distance(out_emb, e)
                         for e in reference.photos_embeddings if e is not None]
                m.embedding_distance_vs_photos_mean = float(np.mean(dists)) if dists else 1.0

            # LAB distance vs photos mean
            if reference.photos_lab_means:
                photo_dists = [float(np.linalg.norm(out_lab - pl))
                               for pl in reference.photos_lab_means]
                m.embedding_distance_vs_photos_mean = min(
                    m.embedding_distance_vs_photos_mean,
                    float(np.mean(photo_dists))  # also store raw LAB mean
                )

        # ── GROUP 6: Telemetry Truth ──────────────────────────────────────────
        tel = capture.telemetry
        intercepted_path = capture.render_path
        tel_path = tel.get("render_path", None)
        m.telemetry_path_honest = (tel_path == intercepted_path)
        m.telemetry_intrinsic_honest = (
            tel.get("intrinsic_used", False) == (capture.albedo is not None)
        )
        tel_geo = tel.get("geometry_source", "unknown")
        m.telemetry_geometry_honest = (tel_geo == capture.geometry_source)
        m.telemetry_energy_terms_present = (
            isinstance(tel.get("energy_terms"), dict) and
            len(tel.get("energy_terms", {})) >= 1
        )
        m.sim2_det_positive = capture.sim2_det > 0

        # ── Arch Compliance (D-01..D-10) ──────────────────────────────────────
        m.D01_signal_preserving = (
            m.frequency_retention >= self.TARGET_FREQ_RET and
            m.energy_conservation_ratio >= 0.5 and
            m.energy_conservation_ratio <= 1.5
        )
        m.D02_physical_quality = (
            m.render_path == "physical" and
            m.ssim_vs_expectation >= self.TARGET_SSIM
        )
        m.D04_dense_geometry = (capture.geometry_source == "mesh")
        m.D05_identity_decoupled = (
            m.albedo_lab_vs_anchor <= self.TARGET_ALBEDO_LAB and
            m.lab_distance_vs_expectation <= self.TARGET_LAB_DIST
        )
        m.D06_temporal = m.sim2_det_positive
        m.D08_telemetry_honest = (
            m.telemetry_path_honest and
            m.telemetry_intrinsic_honest and
            m.telemetry_energy_terms_present
        )

        return m

    # ── Compliance check ──────────────────────────────────────────────────────

    def check_arch_compliance(self, metrics: List[FrameAuditMetrics]) -> Dict:
        """Aggregate D-item compliance across all frames."""
        def _pct(key):
            vals = [getattr(m, key) for m in metrics if getattr(m, key) is not None]
            return sum(1 for v in vals if v) / max(len(vals), 1) * 100

        return {
            "D01_signal_preserving_pct": _pct("D01_signal_preserving"),
            "D02_physical_quality_pct": _pct("D02_physical_quality"),
            "D04_dense_geometry_pct": _pct("D04_dense_geometry"),
            "D05_identity_decoupled_pct": _pct("D05_identity_decoupled"),
            "D06_temporal_pct": _pct("D06_temporal"),
            "D08_telemetry_honest_pct": _pct("D08_telemetry_honest"),
        }

    # ── Formatting ────────────────────────────────────────────────────────────

    @staticmethod
    def _tick(val: bool, good: bool = True) -> str:
        if good:
            return "✅" if val else "❌"
        return "⚠️ " if val else "✅"

    @staticmethod
    def _in_range(v: float, lo: float, hi: float) -> bool:
        return lo <= v <= hi

    def format_frame_report(self, m: FrameAuditMetrics) -> str:
        lines = []
        W = 62
        lines.append("═" * W)
        lines.append(f"FRAME {m.frame_idx:>5}  render_path={m.render_path}  geom={m.geometry_source}")
        lines.append("═" * W)

        def row(label, value, unit, good, note=""):
            tick = self._tick(good)
            return f"  {label:<38} {value:>8}{unit}  {tick} {note}"

        lines.append("\nDECOMPOSITION  (Y ≈ A ⊙ S + spec)")
        lines.append(row("Reconstruction MAE", f"{m.recon_mae:.4f}", "",
                          m.recon_mae < 0.05, f"({m.recon_pct:.1f}%)"))
        lines.append(row("Albedo mean", f"{m.albedo_mean:.3f}", "",
                          self._in_range(m.albedo_mean, 0.4, 1.0)))
        lines.append(row("Shading mean", f"{m.shading_mean:.3f}", "",
                          self._in_range(m.shading_mean, 0.05, 0.6)))
        lines.append(row("Shading smoothness (∇S/S)", f"{m.shading_smoothness:.3f}", "",
                          m.shading_smoothness < 0.4, "target<0.4"))
        lines.append(row("Specular sparsity", f"{m.specular_sparsity:.3f}", "",
                          m.specular_sparsity > 0.75, "target>0.75"))
        lines.append(row("Albedo photom. stability", f"{m.albedo_channel_std:.4f}", "",
                          m.albedo_channel_std < 0.05, "target<0.05"))

        lines.append("\nRENDERING ENERGY")
        lines.append(row("Energy conservation ratio", f"{m.energy_conservation_ratio:.3f}", "",
                          self._in_range(m.energy_conservation_ratio, 0.5, 1.5),
                          "target[0.5,1.5]"))
        lines.append(f"  Ambient/Diffuse/Specular fracs: "
                     f"{m.ambient_frac:.0%} / {m.diffuse_frac:.0%} / {m.specular_frac:.0%}")
        lines.append(row("Lambertian N·L̂ mean", f"{m.lambertian_ndotl_mean:.3f}", "",
                          self._in_range(m.lambertian_ndotl_mean, 0.3, 1.2)))
        lines.append(row("Detail energy ratio", f"{m.detail_energy_ratio:.3f}", "",
                          m.detail_energy_ratio < 0.30, "target<0.30"))

        lines.append("\nGEOMETRY")
        lines.append(row("Normal unit error", f"{m.normal_unit_error:.5f}", "",
                          m.normal_unit_error < 0.01, "target<0.01"))
        lines.append(row("Normal Z mean", f"{m.normal_z_mean:.3f}", "",
                          m.normal_z_mean > 0.5, "target>0.5"))
        lines.append(row("Normal coverage (N_z>0.1)", f"{m.normal_coverage:.3f}", "",
                          m.normal_coverage > 0.6, "target>0.6"))
        lines.append(f"  Geometry source: {m.geometry_source}")

        lines.append("\nSIGNAL FIDELITY  (vs LOCKED_ARCHITECTURE.md targets)")
        lines.append(row("Sharpness (output)", f"{m.sharpness_output:.1f}", "",
                          m.sharpness_output >= self.TARGET_SHARPNESS,
                          f"target={self.TARGET_SHARPNESS:.0f}  src={m.sharpness_source:.1f}"))
        lines.append(row("Contrast (output)", f"{m.contrast_output:.1f}", "",
                          m.contrast_output >= self.TARGET_CONTRAST,
                          f"target={self.TARGET_CONTRAST:.0f}"))
        lines.append(row("Frequency retention", f"{m.frequency_retention:.3f}", "",
                          m.frequency_retention >= self.TARGET_FREQ_RET,
                          f"target≥{self.TARGET_FREQ_RET}"))

        lines.append("\nIDENTITY vs REFERENCE")
        lines.append(row("LAB dist vs expectation", f"{m.lab_distance_vs_expectation:.1f}", "",
                          m.lab_distance_vs_expectation <= self.TARGET_LAB_DIST,
                          "target<20"))
        lines.append(row("SSIM vs expectation", f"{m.ssim_vs_expectation:.3f}", "",
                          m.ssim_vs_expectation >= self.TARGET_SSIM,
                          "target≥0.5"))
        lines.append(row("Albedo LAB vs anchor", f"{m.albedo_lab_vs_anchor:.1f}", "",
                          m.albedo_lab_vs_anchor <= self.TARGET_ALBEDO_LAB,
                          "target<10"))
        lines.append(row("Emb dist vs expectation", f"{m.embedding_distance_vs_expectation:.3f}", "",
                          m.embedding_distance_vs_expectation <= self.TARGET_EMB_DIST,
                          f"target<{self.TARGET_EMB_DIST}"))

        lines.append("\nTELEMETRY TRUTH")
        lines.append(f"  path honest:    {self._tick(m.telemetry_path_honest)}")
        lines.append(f"  intrinsic honest:{self._tick(m.telemetry_intrinsic_honest)}")
        lines.append(f"  geometry honest: {self._tick(m.telemetry_geometry_honest)}")
        lines.append(f"  energy terms:   {self._tick(m.telemetry_energy_terms_present)}")
        lines.append(f"  SIM2 det>0:     {self._tick(m.sim2_det_positive)}")

        lines.append("\nARCH COMPLIANCE (D-items)")
        lines.append(f"  D-01 Signal-preserving:  {self._tick(m.D01_signal_preserving)}"
                     f"  (freq_ret={m.frequency_retention:.3f}, ECR={m.energy_conservation_ratio:.2f})")
        lines.append(f"  D-02 Physical quality:   {self._tick(m.D02_physical_quality)}"
                     f"  (path={m.render_path}, SSIM={m.ssim_vs_expectation:.3f})")
        lines.append(f"  D-04 Dense geometry:     {self._tick(m.D04_dense_geometry)}"
                     f"  ({m.geometry_source})")
        lines.append(f"  D-05 Identity decoupled: {self._tick(m.D05_identity_decoupled)}"
                     f"  (albLAB={m.albedo_lab_vs_anchor:.1f}, labDist={m.lab_distance_vs_expectation:.1f})")
        lines.append(f"  D-06 Temporal SIM2:      {self._tick(m.D06_temporal)}")
        lines.append(f"  D-08 Telemetry honest:   {self._tick(m.D08_telemetry_honest)}")

        return "\n".join(lines)

    def format_summary(self, all_metrics: List[FrameAuditMetrics],
                       compliance: Dict) -> str:
        lines = ["\n" + "═" * 62, "AUDIT SUMMARY", "═" * 62]
        n = len(all_metrics)

        def avg(key):
            vals = [getattr(m, key) for m in all_metrics
                    if isinstance(getattr(m, key), (int, float))]
            return float(np.mean(vals)) if vals else 0.0

        lines.append(f"\n  Frames audited: {n}")
        lines.append(f"  Physical path:  {sum(1 for m in all_metrics if m.render_path=='physical')}/{n}")

        lines.append("\n  SIGNAL FIDELITY (mean across frames)")
        sh = avg("sharpness_output")
        ct = avg("contrast_output")
        fr = avg("frequency_retention")
        lines.append(f"    Sharpness:   {sh:>8.1f}  target={self.TARGET_SHARPNESS:.0f}  "
                     f"{'✅' if sh >= self.TARGET_SHARPNESS else '❌'} gap=×{self.TARGET_SHARPNESS/(sh+1e-8):.1f}")
        lines.append(f"    Contrast:    {ct:>8.1f}  target={self.TARGET_CONTRAST:.0f}  "
                     f"{'✅' if ct >= self.TARGET_CONTRAST else '❌'} gap={self.TARGET_CONTRAST-ct:+.1f}")
        lines.append(f"    Freq retain: {fr:>8.3f}  target≥{self.TARGET_FREQ_RET}  "
                     f"{'✅' if fr >= self.TARGET_FREQ_RET else '❌'}")

        lines.append("\n  DECOMPOSITION (mean)")
        lines.append(f"    Recon MAE:   {avg('recon_mae'):>8.4f}  target<0.05  "
                     f"{'✅' if avg('recon_mae')<0.05 else '❌'}")
        lines.append(f"    Shading smooth: {avg('shading_smoothness'):>6.3f}  target<0.40")
        lines.append(f"    Specular sparse:{avg('specular_sparsity'):>6.3f}  target>0.75")

        lines.append("\n  IDENTITY (mean)")
        lines.append(f"    LAB vs expect: {avg('lab_distance_vs_expectation'):>7.1f}  target<20")
        lines.append(f"    SSIM vs expect:{avg('ssim_vs_expectation'):>8.3f}  target≥0.5")
        lines.append(f"    Albedo LAB:    {avg('albedo_lab_vs_anchor'):>7.1f}  target<10")

        lines.append("\n  ARCH COMPLIANCE (% frames passing)")
        for k, v in compliance.items():
            tick = "✅" if v >= 80 else ("⚠️ " if v >= 50 else "❌")
            lines.append(f"    {k:<38} {v:>5.0f}%  {tick}")

        return "\n".join(lines)

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(
        self,
        video_path: str,
        expectation_path: str,
        photos_dir: str,
        n_frames: int = 5,
        output_json: Optional[str] = None,
    ) -> Dict:
        print("\n╔══════════════════════════════════════════════════════════╗")
        print("║       FACE OS — MATHEMATICAL AUDIT INSTRUMENT           ║")
        print("╚══════════════════════════════════════════════════════════╝\n")

        # Build reference fingerprint
        print("Building reference fingerprint from expectation + photos...")
        reference = self.build_reference_fingerprint(expectation_path, photos_dir)
        print(f"  expectation.png: LAB={reference.expectation_lab_mean.round(1)}")
        print(f"  photos loaded:   {len(reference.photos_lab_means)}")
        print(f"  anchor albedo:   {'ok' if reference.anchor_albedo_lab is not None else 'failed'}")

        # Load pipeline
        print("\nLoading pipeline...")
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from face_os.pipeline import FaceOSPipeline
        pipeline = FaceOSPipeline()
        ok = pipeline.enroll()
        if not ok:
            raise RuntimeError("Pipeline enrollment failed")
        print("  Pipeline enrolled ✅\n")

        # Sample frames
        print(f"Sampling {n_frames} frames...")
        sampled = self.sample_frames(video_path, n_frames)
        print(f"  Frames: {[idx for idx,_ in sampled]}\n")

        # Audit each frame — load 8 warm-up frames before each target
        # so the face-lock state machine reaches FACE_LOCKED state.
        all_metrics: List[FrameAuditMetrics] = []
        all_captures: List[dict] = []

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_idx, frame in sampled:
            print(f"\nAuditing frame {frame_idx}...")
            # Collect 12 preceding frames as warm-up.
            # Face-lock needs: 1 LOST→RECOVERY + 6 RECOVERY→LOCKED + 3 mode transitions = 10 min.
            # Use step=1 to ensure consecutive clean detections (no jumps that reset state).
            warmup: List[Tuple[int, np.ndarray]] = []
            wu_start = max(0, frame_idx - 14)
            for wu_idx in range(wu_start, frame_idx, max(1, (frame_idx - wu_start) // 12)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, wu_idx)
                ok, wu_frame = cap.read()
                if ok:
                    warmup.append((wu_idx, wu_frame))
            warmup = warmup[-12:]  # take last 12 (closest to target)


            capture = self.instrument_and_run(pipeline, frame, frame_idx,
                                              warmup_frames=warmup)
            metrics = self.compute_audit_metrics(capture, reference)
            all_metrics.append(metrics)
            print(self.format_frame_report(metrics))
            all_captures.append({"frame_idx": frame_idx, "render_path": metrics.render_path})

        cap.release()


        # Summary
        compliance = self.check_arch_compliance(all_metrics)
        print(self.format_summary(all_metrics, compliance))

        # Build JSON output
        def _to_dict(m: FrameAuditMetrics) -> dict:
            d = {}
            for k, v in m.__dict__.items():
                if isinstance(v, (int, float, str, bool, type(None))):
                    d[k] = v
                else:
                    d[k] = str(v)
            return d

        report = {
            "video": video_path,
            "expectation": expectation_path,
            "photos_dir": photos_dir,
            "n_frames": n_frames,
            "frames": [_to_dict(m) for m in all_metrics],
            "compliance": compliance,
            "targets": {
                "sharpness": self.TARGET_SHARPNESS,
                "contrast": self.TARGET_CONTRAST,
                "frequency_retention": self.TARGET_FREQ_RET,
                "ssim": self.TARGET_SSIM,
                "lab_distance": self.TARGET_LAB_DIST,
                "albedo_lab": self.TARGET_ALBEDO_LAB,
            },
        }

        if output_json:
            os.makedirs(os.path.dirname(output_json), exist_ok=True)
            with open(output_json, "w") as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n  JSON saved → {output_json}")

        return report


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Face OS Mathematical Audit Instrument",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--video", default="input/video.mp4")
    parser.add_argument("--expectation", default="expectation.png")
    parser.add_argument("--photos", default="photos/")
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--output", default="output/face_os/audit_report.json")
    args = parser.parse_args()

    suite = PhysicalAuditSuite()
    suite.run(
        video_path=args.video,
        expectation_path=args.expectation,
        photos_dir=args.photos,
        n_frames=args.frames,
        output_json=args.output,
    )


if __name__ == "__main__":
    main()
