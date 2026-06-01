"""Subsystem B — Identity Estimation.

Estimates stable identity representation and OWNS the lighting-invariant
identity latent (``IdentityLatent``).

Output: IdentityEstimatorState (from face_os.types)
Delegates to: identity_state.py, intrinsic_decomposition.py

BOUNDARY CONTRACT:
- MUST NOT perform RGB EMA blending
- MUST NOT accumulate raw frames
- MUST NOT handle geometry estimation

LATENT OWNERSHIP (D-05):
- IdentityEstimator is the SOLE owner of the IdentityLatent instance.
- The pipeline accesses the latent only through public methods
  (``set_anchor`` here; ``update_latent``/``synthesize_identity``/
  ``query_uncertainty`` arrive in Tasks 2.2-2.4).
- The latent stores reflectance + structure ONLY — never illumination,
  never raw RGB frames, never geometry.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from face_os.types import IdentityEstimatorState, IdentityLatent

logger = logging.getLogger(__name__)

# ── Fusion tuning constants (named, justified — NOT magic EMA rates) ──────────
_EPS = 1e-6
_K_TEMPORAL_INFLATE = 0.5
_WB_SCALE_MIN = 0.5
_WB_SCALE_MAX = 2.0

# ── Appearance encoder constants (Task 2.5 — manifold wiring) ─────────────────
_APPEARANCE_DIM = 16
_PROJECTION_SEED = 42
_PROJECTION_INPUT_DIM = 478 * 3
_MAX_APPEARANCE_DISTANCE = 10.0
_MAX_MANIFOLD_OBSERVATIONS = 64
_APPEARANCE_OBS_NOISE_SIGMA_SQ = 0.1
_K_EXPRESSION_GAIN = 2.0


def _build_projection_matrix() -> np.ndarray:
    """Johnson-Lindenstrauss projection: (16, 1434) random matrix.

    Preserves pairwise distances in the deformation field with high probability,
    no training data required. Fixed seed for deterministic reproducibility.
    """
    rng = np.random.RandomState(_PROJECTION_SEED)
    return rng.randn(_APPEARANCE_DIM, _PROJECTION_INPUT_DIM).astype(np.float32) / np.sqrt(_PROJECTION_INPUT_DIM)


def _build_projection_pinv(P: np.ndarray) -> np.ndarray:
    """Moore-Penrose pseudoinverse of the JL projection matrix.

    P^T @ (P @ P^T)^{-1} — reconstructs approximate deformation field from
    the 16-D appearance_code. The reconstruction captures the principal
    expression deformation; residual high-frequency detail is lost in the
    JL compression (which is by design).
    """
    return np.linalg.pinv(P).astype(np.float32)


class IdentityEstimator:
    """Subsystem B: Stable identity representation and latent owner.

    Thin wrapper that delegates to existing identity_state.py for the legacy
    query surface, and additionally OWNS an ``IdentityLatent`` instance that is
    the renderer's primary input (once the latent path is live).

    FORBIDDEN: RGB EMA blending, raw frame accumulation, geometry estimation
    """

    def __init__(
        self,
        identity_state,
        manifold=None,
        atlas_size: Tuple[int, int] = (256, 256),
    ):
        """Args:
        identity_state: IdentityState instance from identity_state.py.
        manifold: Optional IdentityManifold. The geometry-conditioned
            appearance_code wiring lands in Task 2.5; for now a zero-vector
            placeholder is used, so this stays optional.
        atlas_size: Canonical UV size (H, W) for the owned latent.

        Backward compatibility: existing call sites construct
        ``IdentityEstimator(self.identity_state)`` with a single positional
        arg, so ``manifold`` and ``atlas_size`` are optional keyword args.
        """
        self._state = identity_state
        self._manifold = manifold
        self._atlas_size = atlas_size
        self._latent = IdentityLatent(atlas_size=atlas_size)
        self._last_normal_source = "face_prior"
        self._best_quality: Optional[np.ndarray] = None
        self._enrollment_mesh: Optional[np.ndarray] = None
        self._canonical_lm_2d: Optional[np.ndarray] = None
        self._projection_matrix: np.ndarray = _build_projection_matrix()
        self._projection_pinv: np.ndarray = _build_projection_pinv(self._projection_matrix)
        self._observation_points: list = []
        self._observation_weights: list = []
        self._smoothed_appearance: Optional[np.ndarray] = None
        if self._manifold is None:
            from face_os.identity_manifold import IdentityManifold, ManifoldConfig
            self._manifold = IdentityManifold(ManifoldConfig(dimension=self._appearance_dim(), max_geodesic_distance=_MAX_APPEARANCE_DISTANCE))

    def latent(self) -> IdentityLatent:
        return self._latent

    def store_enrollment_mesh(self, mesh: np.ndarray) -> None:
        """Store the enrollment reference mesh for appearance encoding.

        The enrollment mesh defines the "neutral expression" origin on the
        appearance manifold. Per-frame deformation is measured relative to
        this reference and projected to 16-D appearance_code.

        Args:
            mesh: (478, 3) float32 landmark positions from the enrollment frame.
        """
        mesh_arr = np.asarray(mesh, dtype=np.float32)
        if mesh_arr.ndim != 2 or mesh_arr.shape[0] < 468 or mesh_arr.shape[1] < 3:
            logger.warning(
                "store_enrollment_mesh: invalid shape %s; appearance encoding disabled.",
                mesh_arr.shape,
            )
            self._enrollment_mesh = None
            return
        self._enrollment_mesh = mesh_arr
        lm_xy = mesh_arr[:, :2].astype(np.float32)
        x_min, x_max = lm_xy[:, 0].min(), lm_xy[:, 0].max()
        y_min, y_max = lm_xy[:, 1].min(), lm_xy[:, 1].max()
        scale = max(x_max - x_min, y_max - y_min, 1.0)
        lm_uv_x = (lm_xy[:, 0] - x_min) / scale * 200.0 + 28.0
        lm_uv_y = (lm_xy[:, 1] - y_min) / scale * 200.0 + 28.0
        self._canonical_lm_2d = np.stack([lm_uv_x, lm_uv_y], axis=-1).astype(np.float32)
        self._invalidate_appearance_code()

    def _encode_appearance(self, mesh: np.ndarray) -> Optional[np.ndarray]:
        """Encode landmark deformation → 16-D appearance_code.

        Computes the deformation field between the current mesh and the stored
        enrollment mesh, then projects to 16-D via a fixed Johnson-Lindenstrauss
        random projection. Scale-invariant (normalizes by face bounding box).

        Args:
            mesh: (N, 3+) float32 current frame landmarks.

        Returns:
            (16,) float32 appearance_code, or None if enrollment not available.
        """
        if self._enrollment_mesh is None:
            return None
        mesh_arr = np.asarray(mesh, dtype=np.float32)
        if mesh_arr.ndim != 2 or mesh_arr.shape[0] < 468 or mesh_arr.shape[1] < 3:
            return None
        if mesh_arr.shape != self._enrollment_mesh.shape:
            return None
        deformation = (mesh_arr - self._enrollment_mesh).flatten().astype(np.float32)
        bbox_size = float(
            max(
                mesh_arr[:, 0].max() - mesh_arr[:, 0].min(),
                mesh_arr[:, 1].max() - mesh_arr[:, 1].min(),
                1.0,
            )
        )
        deformation = deformation / bbox_size
        code = self._projection_matrix @ deformation
        return code.astype(np.float32)

    def _invalidate_appearance_code(self) -> None:
        """Reset appearance_code to neutral (zeros) and uncertainty to 0.0.

        Called when enrollment mesh changes so stale codes don't leak.
        """
        if self._latent.initialized:
            self._latent.appearance_code = np.zeros(
                self._appearance_dim(), dtype=np.float32
            )
            self._latent.appearance_uncertainty = 0.0
            self._smoothed_appearance = None

    def _compute_deformation_map(self, atlas_h: int, atlas_w: int) -> np.ndarray:
        """Reconstruct expression deformation from smoothed appearance_code.

        Inverts the JL projection to recover the principal (478, 3) deformation
        field, then interpolates its magnitude onto the atlas grid via Delaunay
        triangulation of the canonical 2D landmark positions.

        Returns:
            (atlas_h, atlas_w) float32 scalar deformation magnitude map, or zeros
            if appearance code / canonical landmarks are unavailable.
        """
        if self._smoothed_appearance is None or self._canonical_lm_2d is None:
            return np.zeros((atlas_h, atlas_w), dtype=np.float32)

        deform = self._projection_pinv @ np.asarray(self._smoothed_appearance, dtype=np.float32)
        deform = deform.reshape(-1, 3)
        n_lm = deform.shape[0]
        if n_lm != self._canonical_lm_2d.shape[0]:
            return np.zeros((atlas_h, atlas_w), dtype=np.float32)

        deform_mag = np.linalg.norm(deform[:, :2], axis=-1)

        try:
            from scipy.interpolate import griddata
            grid_x, grid_y = np.meshgrid(
                np.arange(atlas_w, dtype=np.float32),
                np.arange(atlas_h, dtype=np.float32),
            )
            deform_map = griddata(
                self._canonical_lm_2d.astype(np.float64),
                deform_mag.astype(np.float64),
                (grid_x.astype(np.float64), grid_y.astype(np.float64)),
                method="linear",
                fill_value=0.0,
            )
            return np.clip(deform_map, 0.0, None).astype(np.float32)
        except Exception:
            return np.zeros((atlas_h, atlas_w), dtype=np.float32)

    def _riemannian_gain(self, metric: np.ndarray, innovation: np.ndarray) -> np.ndarray:
        """Apply metric-aware Kalman gain to an innovation vector.

        Eigendecomposes the learned manifold metric tensor, then applies per-direction
        gain ``λ_i / (λ_i + σ²)``. Directions with high observed variance (large λ_i)
        are familiar — gain ≈ 1, trust the observation. Directions with only the
        regularization floor (λ_i ≈ ε) are novel — gain ≈ 0, stick with current state.

        Returns the gain-weighted innovation vector in the original basis.
        """
        vals, vecs = np.linalg.eigh(metric)
        vals = np.clip(vals, 1e-6, None)
        gains = vals / (vals + _APPEARANCE_OBS_NOISE_SIGMA_SQ)
        innovation_basis = vecs.T @ innovation
        return vecs @ (gains * innovation_basis)

    def set_anchor(
        self,
        reference_face_bgr: np.ndarray,
        enrollment_mesh: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize the latent from an enrollment reference frame.

        The reference is decomposed into intrinsic components; its albedo is
        white-balance normalized (canonicalizing color temperature) and stored
        as the latent's reflectance in canonical UV. Lighting/shading is
        deliberately discarded — the latent is lighting-invariant.

        Defensive: a ``None`` / wrong-shape / degenerate reference never raises;
        it is logged and the latent is left uninitialized so the shadow path can
        keep running without corrupting state.

        Args:
            reference_face_bgr: (H, W, 3) uint8 BGR enrollment crop.
            enrollment_mesh: Optional (478, 3) float32 landmarks for appearance
                encoding (Task 2.5 — manifold wiring).
        """
        # ── 0. Guard the input — never raise on a degenerate reference ──────────
        ref = np.asarray(reference_face_bgr) if reference_face_bgr is not None else None
        if ref is None or ref.ndim != 3 or ref.shape[2] != 3 or ref.size == 0:
            logger.warning(
                "IdentityEstimator.set_anchor: invalid reference (%s); "
                "leaving latent uninitialized.",
                None if ref is None else getattr(ref, "shape", None),
            )
            return

        try:
            self._set_anchor_impl(ref)
        except Exception as exc:  # noqa: BLE001 — degenerate refs must not crash enrollment
            logger.warning(
                "IdentityEstimator.set_anchor: latent init failed (%s); "
                "leaving latent uninitialized.",
                exc,
            )
            self._latent = IdentityLatent(atlas_size=self._atlas_size)
            return

        # ── Store enrollment mesh for appearance encoding (Task 2.5) ─────────
        if enrollment_mesh is not None:
            self.store_enrollment_mesh(enrollment_mesh)

        # ── Keep the legacy anchor consistent during shadow mode (best-effort) ──
        try:
            self._state.set_anchor(reference_face_bgr)
        except Exception as exc:  # noqa: BLE001 — legacy sync must not break latent init
            logger.warning(
                "IdentityEstimator.set_anchor: legacy state.set_anchor failed (%s); "
                "latent stays initialized.",
                exc,
            )

    def _set_anchor_impl(self, reference_face_bgr: np.ndarray) -> None:
        """Build and store the latent from a validated BGR reference."""
        # ── 1. Decompose the reference (reuse the state's machinery if present) ──
        # decompose() expects RGB float in [0, 1].
        reference_rgb = (
            cv2.cvtColor(reference_face_bgr.astype(np.uint8), cv2.COLOR_BGR2RGB).astype(
                np.float32
            )
            / 255.0
        )

        decomposer = getattr(self._state, "_intrinsic_decomposer", None)
        if decomposer is None:
            # Fallback: construct a local decomposer (state lacks the machinery).
            from face_os.intrinsic_decomposition import IntrinsicDecomposer

            decomposer = IntrinsicDecomposer()
        intrinsic = decomposer.decompose(reference_rgb)

        # ── 2. White-balance normalize the albedo (reuse state's EMA-backed WB) ──
        normalize_wb = getattr(self._state, "_normalize_white_balance", None)
        if callable(normalize_wb):
            albedo = normalize_wb(intrinsic.albedo)
        else:
            albedo = np.clip(intrinsic.albedo, 0.0, 1.0).astype(np.float32)

        # ── 3. Resize to canonical UV (atlas) so the latent is pose-decoupled ──
        atlas_h, atlas_w = self._atlas_size
        albedo = self._resize_to_atlas(albedo, channels=3)

        # ── 4. Microdetail: identity HF residual, zero-meaned (never averaged) ──
        detail = intrinsic.detail_residual
        if detail is not None:
            microdetail = self._resize_to_atlas(detail.astype(np.float32), channels=1)
            # Zero-mean so it is a pure HF residual (no DC/color offset).
            microdetail = microdetail - np.mean(microdetail, axis=(0, 1), keepdims=True)
            microdetail = microdetail.astype(np.float32)
        else:
            microdetail = np.zeros((atlas_h, atlas_w, 3), dtype=np.float32)

        # ── 5. Uncertainty maps: from decomposition if available, else moderate ──
        albedo_unc = self._uncertainty_to_atlas(intrinsic.albedo_uncertainty)
        microdetail_unc = self._uncertainty_to_atlas(
            getattr(intrinsic, "shading_uncertainty", None)
        )

        # ── 6. wb_reference: mean color of the WB-normalized albedo ──
        # Choice: the mean of the already-normalized albedo is the canonical
        # color the latent has converged to, so future observations are
        # normalized toward THIS reference (rather than raw _wb_scale_ema,
        # which is a per-channel gain, not a reference color).
        wb_reference = np.mean(albedo, axis=(0, 1)).astype(np.float32)

        # ── 7. Appearance code: zero-vector placeholder (Task 2.5 wires manifold) ──
        appearance_code = np.zeros(self._appearance_dim(), dtype=np.float32)

        self._manifold.add_point("enrollment", appearance_code, confidence=1.0)
        self._observation_points = []
        self._observation_weights = []

        # ── 8. Populate and store the latent ──
        self._latent = IdentityLatent(
            atlas_size=self._atlas_size,
            albedo=albedo,
            appearance_code=appearance_code,
            microdetail=microdetail,
            wb_reference=wb_reference,
            albedo_uncertainty=albedo_unc,
            appearance_uncertainty=1.0,
            microdetail_uncertainty=microdetail_unc,
            observation_count=np.ones((atlas_h, atlas_w), dtype=np.float32),
            initialized=True,
        )

    # ── Task 2.2 — update_latent (uncertainty-weighted fusion) ─────────────────

    def update_latent(
        self,
        canonical_face: np.ndarray,
        geometry,
        quality_map: np.ndarray,
        temporal=None,
        intrinsic=None,
    ) -> IdentityLatent:
        """Fuse one observation into the latent (uncertainty-weighted, NOT EMA).

        The source crop is an *observation* of the latent, never identity memory.
        This decomposes the observation, white-balance normalizes its albedo
        against ``wb_reference``, and fuses it into the stored latent with a
        Kalman-like gain so confident-and-low-uncertainty regions move more
        while high-uncertainty / low-quality regions stay conservative.

        Phase 1 (shadow mode): this populates the latent but does NOT drive
        rendering. It is additive and never raises on a degenerate observation.

        Args:
            canonical_face: (atlas_H, atlas_W, 3) uint8 BGR — source warped into
                canonical UV.
            geometry: ``GeometryState`` (provides ``mesh_478`` / canonical
                transform for normals; may be ``None``-valued fields).
            quality_map: (atlas_H, atlas_W) float32 in [0, 1] — per-pixel
                observation quality.
            temporal: optional ``TemporalState`` (read-only). Its ``drift_score``
                inflates stored uncertainty BEFORE fusion (predict step).
            intrinsic: optional pre-computed ``IntrinsicComponents`` for THIS
                observation. When the caller already decomposed ``canonical_face``
                this frame (e.g. the pipeline's identity update), passing it here
                AVOIDS a redundant second decomposition. When ``None`` the
                observation is decomposed internally.

        Returns:
            The updated (and now ``initialized``) ``IdentityLatent``.
        """
        # ── 0. Guard inputs — shadow mode must never crash the pipeline ────────
        obs_bgr = np.asarray(canonical_face) if canonical_face is not None else None
        if obs_bgr is None or obs_bgr.ndim != 3 or obs_bgr.shape[2] != 3 or obs_bgr.size == 0:
            logger.warning(
                "IdentityEstimator.update_latent: invalid canonical_face (%s); "
                "skipping update.",
                None if obs_bgr is None else getattr(obs_bgr, "shape", None),
            )
            return self._latent

        try:
            return self._update_latent_impl(obs_bgr, geometry, quality_map, temporal, intrinsic)
        except Exception as exc:  # noqa: BLE001 — a bad observation must not corrupt state
            logger.warning(
                "IdentityEstimator.update_latent: fusion failed (%s); "
                "latent left unchanged.",
                exc,
            )
            return self._latent

    def _update_latent_impl(
        self,
        canonical_face: np.ndarray,
        geometry,
        quality_map: np.ndarray,
        temporal,
        intrinsic=None,
    ) -> IdentityLatent:
        """Core fusion on a validated BGR observation (see ``update_latent``)."""
        atlas_h, atlas_w = self._atlas_size

        from face_os.identity_manifold import IdentityPoint

        # ── 1. Decompose the OBSERVATION (source is telemetry, not memory) ─────
        # Reuse a caller-provided decomposition when present (avoids a redundant
        # second decompose of the same canonical_face this frame); otherwise
        # decompose locally. decompose() expects RGB float in [0, 1].
        if intrinsic is not None and getattr(intrinsic, "albedo", None) is not None:
            intrinsic = intrinsic
        else:
            obs_rgb = (
                cv2.cvtColor(canonical_face.astype(np.uint8), cv2.COLOR_BGR2RGB).astype(
                    np.float32
                )
                / 255.0
            )
            # Resize observation to atlas so fusion is per-pixel aligned with the latent.
            if obs_rgb.shape[0] != atlas_h or obs_rgb.shape[1] != atlas_w:
                obs_rgb = cv2.resize(
                    obs_rgb, (atlas_w, atlas_h), interpolation=cv2.INTER_LINEAR
                )

            mesh_478, warp_M = self._geometry_normal_inputs(geometry)
            decomposer = self._decomposer()
            intrinsic = decomposer.decompose(obs_rgb, mesh_478=mesh_478, warp_M=warp_M)

        obs_albedo = self._resize_to_atlas(
            np.asarray(intrinsic.albedo, dtype=np.float32), channels=3
        )
        obs_detail = self._extract_observation_detail(intrinsic, atlas_h, atlas_w)
        obs_unc = self._uncertainty_to_atlas(
            getattr(intrinsic, "albedo_uncertainty", None)
        )

        # ── 2. Quality map -> atlas, clamped [0,1] ─────────────────────────────
        quality = self._quality_to_atlas(quality_map)

        # First observation: seed the latent and return
        if not self._latent.initialized:
            obs_albedo = np.clip(obs_albedo, 0.0, 1.0).astype(np.float32)
            wb_reference = np.mean(obs_albedo, axis=(0, 1)).astype(np.float32)
            obs_albedo = self._normalize_albedo(obs_albedo, wb_reference)
            self._latent = IdentityLatent(
                atlas_size=self._atlas_size,
                albedo=obs_albedo,
                appearance_code=np.zeros(self._appearance_dim(), dtype=np.float32),
                microdetail=obs_detail,
                wb_reference=wb_reference,
                albedo_uncertainty=obs_unc,
                appearance_uncertainty=1.0,
                microdetail_uncertainty=obs_unc.copy(),
                observation_count=quality.copy(),
                initialized=True,
            )
            self._best_quality = quality.copy()
            return self._latent

        latent = self._latent

        # ── WB-normalize the incoming albedo against the stored reference ──────
        obs_albedo = self._normalize_albedo(obs_albedo, latent.wb_reference)

        stored_albedo = np.asarray(latent.albedo, dtype=np.float32)
        stored_unc = np.asarray(latent.albedo_uncertainty, dtype=np.float32)
        obs_count = np.asarray(latent.observation_count, dtype=np.float32)
        best_quality = self._resolve_best_quality(latent)

        # ── 3. Temporal inflation of uncertainty BEFORE fusion (predict step) ──
        # Inflation only ever RAISES uncertainty, preserving monotonicity.
        if temporal is not None:
            drift = float(getattr(temporal, "drift_score", 0.0) or 0.0)
            if drift > 0.0:
                stored_unc = np.clip(
                    stored_unc + _K_TEMPORAL_INFLATE * drift, 0.0, 1.0
                ).astype(np.float32)

        # ── 4. Per-pixel uncertainty-weighted fusion (Kalman-like gain) ────────
        # gain high where stored is uncertain AND observation is confident;
        # scaled by observation quality. This is NOT a fixed-rate EMA.
        obs_unc_b = obs_unc[:, :, np.newaxis]
        stored_unc_b = stored_unc[:, :, np.newaxis]
        quality_b = quality[:, :, np.newaxis]

        gain = (stored_unc_b / (stored_unc_b + obs_unc_b + _EPS)) * quality_b
        gain = np.clip(gain, 0.0, 1.0).astype(np.float32)

        deform_map = self._compute_deformation_map(atlas_h, atlas_w)
        if np.any(deform_map > 0):
            deform_gain = np.clip(
                1.0 + deform_map[:, :, np.newaxis] * _K_EXPRESSION_GAIN, 1.0, 3.0
            ).astype(np.float32)
            gain = gain * deform_gain
            gain = np.clip(gain, 0.0, 1.0).astype(np.float32)

        # Albedo fuses unconditionally (design.md:360). The quality-scaled gain
        # already self-suppresses motion as quality → 0, so an occluded frame
        # barely moves the stored albedo without any explicit "freeze" gate.
        new_albedo = (1.0 - gain) * stored_albedo + gain * obs_albedo
        new_albedo = np.clip(new_albedo, 0.0, 1.0).astype(np.float32)

        # ── Honest Bayesian uncertainty (design.md:361): pure Kalman shrink.
        #    new_unc = (1 - gain) * stored_unc. Since gain ∈ [0,1], every
        #    POSITIVE-quality observation is information that can only TIGHTEN the
        #    posterior (non-increasing). A zero-quality (occluded) frame gives
        #    gain → 0, so uncertainty simply HOLDS — no information cannot make us
        #    more certain. The ONLY source of inflation is the temporal predict
        #    step above (drift_score), never a "failed to beat best-seen" ratchet.
        #    This is what keeps latent_confidence from collapsing to ~0. ─────────
        new_unc = ((1.0 - gain[:, :, 0]) * stored_unc).astype(np.float32)
        new_unc = np.clip(new_unc, 0.0, 1.0).astype(np.float32)

        # ── 5. Microdetail: BEST-OBSERVATION-ONLY (never average pores) ────────
        better = quality > best_quality  # strictly improves on best-seen quality
        better_b = better[:, :, np.newaxis]
        stored_detail = np.asarray(latent.microdetail, dtype=np.float32)
        new_detail = np.where(better_b, obs_detail, stored_detail).astype(np.float32)

        stored_md_unc = np.asarray(latent.microdetail_uncertainty, dtype=np.float32)
        new_md_unc = np.where(better, obs_unc, stored_md_unc).astype(np.float32)

        # ── 6. Bookkeeping: accumulate quality; track best-seen quality ────────
        new_count = (obs_count + quality).astype(np.float32)
        self._best_quality = np.maximum(best_quality, quality).astype(np.float32)

        latent.albedo = new_albedo
        latent.albedo_uncertainty = new_unc
        latent.microdetail = new_detail
        latent.microdetail_uncertainty = new_md_unc
        latent.observation_count = new_count

        # ── 7. Encode appearance from current geometry (Task 2.5) ───────────
        mesh = getattr(geometry, "mesh", None)
        if mesh is not None:
            code = self._encode_appearance(mesh)
            if code is not None:
                self._observation_points.append(code.copy())
                frame_weight = float(np.mean(quality).item())
                self._observation_weights.append(max(frame_weight, 1e-6))
                if len(self._observation_points) > _MAX_MANIFOLD_OBSERVATIONS:
                    self._observation_points = self._observation_points[-_MAX_MANIFOLD_OBSERVATIONS:]
                    self._observation_weights = self._observation_weights[-_MAX_MANIFOLD_OBSERVATIONS:]

                enrollment = self._manifold.get_point("enrollment")
                if enrollment is not None:
                    if len(self._observation_points) >= 3:
                        neighbors = [IdentityPoint(coordinates=p) for p in self._observation_points]
                        w_arr = np.array(self._observation_weights, dtype=np.float64)
                        metric = self._manifold.compute_metric_tensor(enrollment, neighbors, weights=w_arr)
                        enrollment.metric_tensor = metric
                    if self._smoothed_appearance is not None and getattr(enrollment, "metric_tensor", None) is not None:
                        innovation = code - self._smoothed_appearance
                        weighted = self._riemannian_gain(enrollment.metric_tensor, innovation)
                        self._smoothed_appearance = self._smoothed_appearance + weighted
                    else:
                        self._smoothed_appearance = code.copy()
                    smoothed_code = self._smoothed_appearance
                else:
                    smoothed_code = code

                latent.appearance_code = smoothed_code.astype(np.float32)
                current = IdentityPoint(coordinates=smoothed_code)
                if enrollment is not None:
                    distance = self._manifold.geodesic_distance(enrollment, current)
                    latent.appearance_uncertainty = float(min(
                        distance / self._manifold.config.max_geodesic_distance, 1.0
                    ))
                else:
                    latent.appearance_uncertainty = float(np.clip(
                        np.linalg.norm(smoothed_code) / _MAX_APPEARANCE_DISTANCE, 0.0, 1.0
                    ))

        return latent

    # ── Task 2.3 — synthesize_identity (latent is the PRIMARY render input) ────

    def synthesize_identity(self, geometry):
        """Synthesize the stored identity into the current geometry.

        PRIMARY render input: warps the stored latent ``albedo`` + ``microdetail``
        from canonical UV into the current geometry, attaches geometry/face-prior
        normals, and leaves ``shading`` as a NEUTRAL unit field so the renderer
        applies lighting. Provenance is the latent ONLY — never a source crop.

        If the latent is uninitialized, returns NEUTRAL components (mid-gray
        albedo, unit shading, face-prior normals, low confidence) so callers can
        gracefully fall back without raising.

        Args:
            geometry: ``GeometryState`` (provides the render size + transform).

        Returns:
            ``IntrinsicComponents`` whose albedo/detail derive from the latent,
            validated against ``assert_intrinsic_contract(..., mode='warn')``.
        """
        from face_os.intrinsic_decomposition import (
            IntrinsicComponents,
            assert_intrinsic_contract,
        )

        render_hw = self._render_hw(geometry)
        h, w = render_hw

        if not self._latent.initialized:
            components = self._neutral_components(h, w)
            assert_intrinsic_contract(components, expect_hw=render_hw, mode="warn")
            return components

        latent = self._latent

        # Warp stored albedo + microdetail from canonical UV into current geometry.
        albedo = self._warp_from_canonical(
            np.asarray(latent.albedo, dtype=np.float32), geometry, render_hw, channels=3
        )
        albedo = np.clip(albedo, 0.0, 1.0).astype(np.float32)

        detail = self._warp_from_canonical(
            np.asarray(latent.microdetail, dtype=np.float32), geometry, render_hw, channels=3
        ).astype(np.float32)

        appear_conf = 1.0 - float(latent.appearance_uncertainty)
        detail = detail * appear_conf

        # Normals: geometry mesh if available, else face-prior ellipsoid.
        normal_map = self._normals_for(geometry, h, w)

        # Shading: neutral unit field — lighting is the renderer's responsibility.
        shading = np.ones((h, w, 1), dtype=np.float32)

        # Confidence = 1 - uncertainty (warped into the current geometry).
        uncertainty = self.query_uncertainty(geometry)
        confidence = np.clip(1.0 - uncertainty, 0.0, 1.0).astype(np.float32)[:, :, np.newaxis]

        specular = np.zeros((h, w, 3), dtype=np.float32)

        components = IntrinsicComponents(
            albedo=albedo,
            shading=shading,
            specular=specular,
            normal_map=normal_map,
            confidence=confidence,
            reconstruction_error=0.0,
            albedo_uncertainty=uncertainty[:, :, np.newaxis].astype(np.float32),
            detail_residual=detail,
            normal_source=self._last_normal_source,
        )

        assert_intrinsic_contract(components, expect_hw=render_hw, mode="warn")
        return components

    # ── Task 2.4 — query_uncertainty ───────────────────────────────────────────

    def query_uncertainty(self, geometry) -> np.ndarray:
        """Latent albedo uncertainty warped into the current geometry.

        Used by render gating and by ``synthesize_identity`` confidence. If the
        latent is uninitialized, returns an all-ones (max uncertainty) map at the
        render size so callers treat it as fully unknown.

        Args:
            geometry: ``GeometryState`` (provides the render size + transform).

        Returns:
            (H, W) float32 in [0, 1] uncertainty in the CURRENT geometry.
        """
        render_hw = self._render_hw(geometry)
        h, w = render_hw

        if not self._latent.initialized or self._latent.albedo_uncertainty is None:
            return np.ones((h, w), dtype=np.float32)

        unc = np.asarray(self._latent.albedo_uncertainty, dtype=np.float32)
        warped = self._warp_from_canonical(unc, geometry, render_hw, channels=1)
        return np.clip(warped, 0.0, 1.0).astype(np.float32)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _appearance_dim(self) -> int:
        """Dimension of the appearance_code placeholder.

        Uses the manifold's configured dimension when a manifold is wired
        (Task 2.5 makes this the real geometry-conditioned code); otherwise a
        16-D default matching ``ManifoldConfig.dimension``.
        """
        manifold = self._manifold
        if manifold is not None:
            for attr in ("_dimension", "dimension"):
                dim = getattr(manifold, attr, None)
                if isinstance(dim, (int, np.integer)) and dim > 0:
                    return int(dim)
            cfg = getattr(manifold, "config", None)
            dim = getattr(cfg, "dimension", None) if cfg is not None else None
            if isinstance(dim, (int, np.integer)) and dim > 0:
                return int(dim)
        return 16

    # ── Decomposition / observation helpers (Tasks 2.2-2.4) ────────────────────

    def _decomposer(self):
        """The intrinsic decomposer to use for observations.

        Reuses the state's decomposer (so WB/normal config matches enrollment)
        when present, else constructs a local one. Cached for reuse.
        """
        decomposer = getattr(self._state, "_intrinsic_decomposer", None)
        if decomposer is not None:
            return decomposer
        cached = getattr(self, "_local_decomposer", None)
        if cached is None:
            from face_os.intrinsic_decomposition import IntrinsicDecomposer

            cached = IntrinsicDecomposer()
            self._local_decomposer = cached
        return cached

    @staticmethod
    def _geometry_normal_inputs(geometry):
        """Extract (mesh_478, warp_M) for the decomposer's mesh-normal path.

        The decomposer only uses mesh normals when BOTH a (>=468, 3+) mesh and a
        (2, 3) warp are present; otherwise it falls back to the face prior. We
        derive a 2x3 warp from ``geometry.canonical_transform`` when possible.
        """
        if geometry is None:
            return None, None
        mesh_478 = getattr(geometry, "mesh_478", None)
        if mesh_478 is None:
            mesh_478 = getattr(geometry, "mesh", None)
        mesh_arr = np.asarray(mesh_478) if mesh_478 is not None else None
        if mesh_arr is None or mesh_arr.ndim != 2 or mesh_arr.shape[0] < 468 or mesh_arr.shape[1] < 3:
            return None, None

        M = getattr(geometry, "canonical_transform", None)
        if M is None:
            return None, None
        M = np.asarray(M, dtype=np.float32)
        if M.shape == (3, 3):
            warp_M = M[:2, :]
        elif M.shape == (2, 3):
            warp_M = M
        else:
            return mesh_arr, None
        return mesh_arr, warp_M.astype(np.float32)

    def _extract_observation_detail(self, intrinsic, atlas_h: int, atlas_w: int) -> np.ndarray:
        """Resize + zero-mean the decomposition's detail residual to atlas size."""
        detail = getattr(intrinsic, "detail_residual", None)
        if detail is None:
            return np.zeros((atlas_h, atlas_w, 3), dtype=np.float32)
        detail = np.asarray(detail, dtype=np.float32)
        if detail.ndim == 2:
            detail = detail[:, :, np.newaxis]
        if detail.shape[2] == 1:
            detail = np.repeat(detail, 3, axis=2)
        if detail.shape[0] != atlas_h or detail.shape[1] != atlas_w:
            detail = cv2.resize(detail, (atlas_w, atlas_h), interpolation=cv2.INTER_LINEAR)
            if detail.ndim == 2:
                detail = detail[:, :, np.newaxis]
        detail = detail - np.mean(detail, axis=(0, 1), keepdims=True)
        return detail.astype(np.float32)

    def _quality_to_atlas(self, quality_map: Optional[np.ndarray]) -> np.ndarray:
        """Squeeze/resize a quality map to (H, W) atlas size, clamped to [0, 1]."""
        atlas_h, atlas_w = self._atlas_size
        if quality_map is None:
            return np.ones((atlas_h, atlas_w), dtype=np.float32)
        q = np.asarray(quality_map, dtype=np.float32)
        if q.ndim == 3:
            q = np.mean(q, axis=2)
        if q.shape[0] != atlas_h or q.shape[1] != atlas_w:
            q = cv2.resize(q, (atlas_w, atlas_h), interpolation=cv2.INTER_LINEAR)
        return np.clip(q, 0.0, 1.0).astype(np.float32)

    def _resolve_best_quality(self, latent) -> np.ndarray:
        """Per-pixel best-seen observation quality at atlas size.

        Returns the tracked ``_best_quality`` when present and shape-matched;
        otherwise derives a conservative proxy from the latent's confidence
        (``1 - albedo_uncertainty``) so a latent seeded by ``set_anchor`` (which
        does not track quality) still behaves monotonically.
        """
        atlas_h, atlas_w = self._atlas_size
        bq = self._best_quality
        if bq is not None:
            bq = np.asarray(bq, dtype=np.float32)
            if bq.shape == (atlas_h, atlas_w):
                return bq
        # Fallback proxy: confidence of the stored albedo.
        unc = np.asarray(latent.albedo_uncertainty, dtype=np.float32)
        proxy = np.clip(1.0 - unc, 0.0, 1.0).astype(np.float32)
        self._best_quality = proxy.copy()
        return proxy

    @staticmethod
    def _normalize_albedo(albedo: np.ndarray, wb_reference: Optional[np.ndarray]) -> np.ndarray:
        """White-balance normalize albedo so its mean matches ``wb_reference``.

        Scales each channel toward the reference color (canonicalizing color
        temperature), with the per-channel scale bounded so a near-black region
        can never blow the albedo up. Returns clipped float32 in [0, 1].
        """
        albedo = np.asarray(albedo, dtype=np.float32)
        if wb_reference is None:
            return np.clip(albedo, 0.0, 1.0).astype(np.float32)
        wb = np.asarray(wb_reference, dtype=np.float32).reshape(-1)
        if wb.size != 3:
            return np.clip(albedo, 0.0, 1.0).astype(np.float32)
        mean_per_channel = np.mean(albedo, axis=(0, 1))
        scale = wb / (mean_per_channel + 1e-8)
        scale = np.clip(scale, _WB_SCALE_MIN, _WB_SCALE_MAX)
        normalized = albedo * scale[np.newaxis, np.newaxis, :]
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)

    # ── Synthesis / warp helpers (Tasks 2.3-2.4) ───────────────────────────────

    @staticmethod
    def _render_hw(geometry) -> Tuple[int, int]:
        """Resolve the (H, W) render size from the geometry.

        Prefers an explicit canonical_face/mask shape; falls back to the mesh
        normal / landmark-derived size, else a (256, 256) default. Always
        returns positive ints.
        """
        if geometry is not None:
            for attr in ("canonical_face", "mask"):
                arr = getattr(geometry, attr, None)
                if arr is not None:
                    a = np.asarray(arr)
                    if a.ndim >= 2 and a.shape[0] > 0 and a.shape[1] > 0:
                        return int(a.shape[0]), int(a.shape[1])
        return (256, 256)

    def _warp_from_canonical(
        self, arr: np.ndarray, geometry, render_hw: Tuple[int, int], channels: int
    ) -> np.ndarray:
        """Warp a canonical-UV map into the current geometry.

        Uses ``geometry.inverse_transform`` (canonical -> current) when a valid
        affine is present; otherwise resizes the canonical map to the render
        size as a pose-neutral fallback. Never raises.
        """
        h, w = render_hw
        arr = np.asarray(arr, dtype=np.float32)

        M = self._affine_2x3(getattr(geometry, "inverse_transform", None))
        if M is None:
            # No transform: return the canonical map resized to render size.
            if arr.shape[0] != h or arr.shape[1] != w:
                arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
            return arr.astype(np.float32)

        try:
            warped = cv2.warpAffine(
                arr,
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
        except Exception:  # noqa: BLE001 — degenerate transform -> resize fallback
            if arr.shape[0] != h or arr.shape[1] != w:
                arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
            return arr.astype(np.float32)

        if channels == 3 and warped.ndim == 2:
            warped = warped[:, :, np.newaxis]
        return warped.astype(np.float32)

    @staticmethod
    def _affine_2x3(transform) -> Optional[np.ndarray]:
        """Coerce a transform into a (2, 3) affine matrix, or None."""
        if transform is None:
            return None
        M = np.asarray(transform, dtype=np.float32)
        if M.shape == (3, 3):
            return M[:2, :].astype(np.float32)
        if M.shape == (2, 3):
            return M.astype(np.float32)
        return None

    def _normals_for(self, geometry, h: int, w: int) -> np.ndarray:
        """Normal map for synthesis: geometry mesh if available, else face prior.

        Records the provenance in ``self._last_normal_source`` so the produced
        ``IntrinsicComponents.normal_source`` is honest.
        """
        mesh_478, warp_M = self._geometry_normal_inputs(geometry)
        if mesh_478 is not None and warp_M is not None:
            try:
                from face_os.landmarks import mesh_normal_map

                normal_map = mesh_normal_map(
                    np.asarray(mesh_478, dtype=np.float32),
                    np.asarray(warp_M, dtype=np.float32),
                    (h, w),
                )
                normal_map = np.asarray(normal_map, dtype=np.float32)
                if normal_map.shape == (h, w, 3):
                    nrm = np.linalg.norm(normal_map, axis=2, keepdims=True)
                    nrm = np.where(nrm > 1e-8, nrm, 1.0)
                    self._last_normal_source = "mesh"
                    return (normal_map / nrm).astype(np.float32)
            except Exception:  # noqa: BLE001 — fall back to the face prior
                pass
        self._last_normal_source = "face_prior"
        return self._face_prior_normals(h, w)

    @staticmethod
    def _face_prior_normals(h: int, w: int) -> np.ndarray:
        """Ellipsoidal face-prior normal map, (H, W, 3) unit vectors."""
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w / 2.0, h / 2.0
        nx = (x - cx) / max(w / 2.0, 1.0)
        ny = (y - cy) / max(h / 2.0, 1.0)
        nz = np.sqrt(np.maximum(1.0 - nx**2 - ny**2, 0.0))
        normals = np.stack([nx, ny, nz], axis=-1)
        nrm = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8
        return (normals / nrm).astype(np.float32)

    def _neutral_components(self, h: int, w: int):
        """Neutral IntrinsicComponents for an uninitialized latent.

        Mid-gray albedo, unit shading, face-prior normals, low confidence so
        callers can gracefully fall back. Provenance is still latent-only (it is
        a synthetic neutral, never a source crop).
        """
        from face_os.intrinsic_decomposition import IntrinsicComponents

        albedo = np.full((h, w, 3), 0.5, dtype=np.float32)
        shading = np.ones((h, w, 1), dtype=np.float32)
        specular = np.zeros((h, w, 3), dtype=np.float32)
        normal_map = self._face_prior_normals(h, w)
        self._last_normal_source = "face_prior"
        confidence = np.zeros((h, w, 1), dtype=np.float32)
        return IntrinsicComponents(
            albedo=albedo,
            shading=shading,
            specular=specular,
            normal_map=normal_map,
            confidence=confidence,
            reconstruction_error=0.0,
            albedo_uncertainty=np.ones((h, w, 1), dtype=np.float32),
            detail_residual=np.zeros((h, w, 3), dtype=np.float32),
            normal_source="face_prior",
        )

    def _resize_to_atlas(self, arr: np.ndarray, channels: int) -> np.ndarray:
        """Resize an (H, W, C) array to canonical UV (atlas_size) if needed."""
        atlas_h, atlas_w = self._atlas_size
        if arr.shape[0] != atlas_h or arr.shape[1] != atlas_w:
            # cv2.resize takes (width, height) order.
            arr = cv2.resize(arr, (atlas_w, atlas_h), interpolation=cv2.INTER_LINEAR)
        return np.clip(arr, 0.0, 1.0).astype(np.float32) if channels == 3 else arr.astype(np.float32)

    def _uncertainty_to_atlas(self, unc: Optional[np.ndarray]) -> np.ndarray:
        """Squeeze an uncertainty map to (H, W) at atlas size, or a 0.5 constant."""
        atlas_h, atlas_w = self._atlas_size
        if unc is None:
            return np.full((atlas_h, atlas_w), 0.5, dtype=np.float32)
        unc = np.asarray(unc, dtype=np.float32)
        if unc.ndim == 3:
            unc = np.mean(unc, axis=2)
        if unc.shape[0] != atlas_h or unc.shape[1] != atlas_w:
            unc = cv2.resize(unc, (atlas_w, atlas_h), interpolation=cv2.INTER_LINEAR)
        return np.clip(unc, 0.0, 1.0).astype(np.float32)

    def query(self, quality_map: np.ndarray) -> IdentityEstimatorState:
        """Query lighting-invariant identity.

        Uses query_albedo (not query_identity) for lighting invariance.

        Args:
            quality_map: Per-pixel quality (H, W) float32

        Returns:
            IdentityEstimatorState with albedo-based identity
        """
        if not self._state.is_initialized():
            return IdentityEstimatorState()

        albedo, albedo_conf = self._state.query_albedo(quality_map)
        rgb_face, rgb_conf = self._state.query_identity(quality_map)
        intrinsic, intrinsic_conf = self._state.query_intrinsic(quality_map)

        return IdentityEstimatorState(
            appearance_latent=rgb_face,
            anchor_basis=[self._state._anchor_albedo]
            if hasattr(self._state, "_anchor_albedo") and self._state._anchor_albedo is not None
            else [],
            identity_uncertainty=(
                1.0 - float(np.mean(albedo_conf))
                if albedo_conf is not None
                else 1.0
            ),
            initialized=True,
        )

    def query_albedo(self, quality_map: np.ndarray):
        """Query lighting-invariant albedo directly.

        Args:
            quality_map: Per-pixel quality (H, W) float32

        Returns:
            (albedo_face, albedo_conf) tuple
        """
        if not self._state.is_initialized():
            return None, None
        return self._state.query_albedo(quality_map)

    def query_intrinsic(self, quality_map: np.ndarray):
        """Query intrinsic decomposition components.

        Args:
            quality_map: Per-pixel quality (H, W) float32

        Returns:
            (intrinsic_components, intrinsic_conf) tuple
        """
        if not self._state.is_initialized():
            return None, None
        return self._state.query_intrinsic(quality_map)
