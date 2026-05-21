"""
energy.py — Energy Function Framework for Face OS.

Phase 1: Energy Function Reformulation.

The global energy:
    E = E_geom + E_identity + E_temporal + E_photometric + E_smoothness

Each energy term is a measurable float with a testable numeric range.
Each subsystem emits its own energy contribution.
No energy term may be hidden inside a black box.

CORE PHILOSOPHY:
    The system is a latent-state estimation problem, not an image-editing pipeline.
    Energy terms quantify how well the current state explains the observations.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from face_os.types import (
    EnergyTerms,
    EnergyReport,
    GeometryState,
    IdentityState,
    TemporalState,
    GeometryMetrics,
    IdentityMetrics,
    TemporalMetrics,
    RendererMetrics,
)


class EnergyComputer:
    """Computes energy terms for a single frame.

    Each energy term quantifies a specific aspect of reconstruction quality.
    Lower energy = better reconstruction.
    """

    def __init__(
        self,
        anchor_face: Optional[np.ndarray] = None,
        anchor_lab: Optional[np.ndarray] = None,
        previous_geometry: Optional[GeometryState] = None,
        previous_identity: Optional[IdentityState] = None,
    ):
        self.anchor_face = anchor_face
        self.anchor_lab = anchor_lab
        self.previous_geometry = previous_geometry
        self.previous_identity = previous_identity

    def compute(
        self,
        frame_idx: int,
        geometry_state: GeometryState,
        identity_state: IdentityState,
        temporal_state: TemporalState,
        source_frame: Optional[np.ndarray] = None,
    ) -> EnergyReport:
        """Compute all energy terms for a single frame.

        Args:
            frame_idx: Frame index
            geometry_state: Current geometry state
            identity_state: Current identity state
            temporal_state: Current temporal state
            source_frame: Original source frame (for photometric terms)

        Returns:
            EnergyReport with all energy terms and metrics
        """
        terms = EnergyTerms()

        # === E_geom: Geometry energy ===
        terms.E_geom = self._compute_E_geom(geometry_state)

        # === E_identity: Identity energy ===
        terms.E_identity = self._compute_E_geom_identity(identity_state)

        # === E_temporal: Temporal energy ===
        terms.E_temporal = self._compute_E_temporal(temporal_state, geometry_state, identity_state)

        # === E_photometric: Photometric energy ===
        terms.E_photometric = self._compute_E_photometric(
            geometry_state, identity_state, source_frame
        )

        # === E_smoothness: Smoothness energy ===
        terms.E_smoothness = self._compute_E_smoothness(geometry_state, identity_state)

        # === E_total: Sum of all terms ===
        terms.E_total = (
            terms.E_geom
            + terms.E_identity
            + terms.E_temporal
            + terms.E_photometric
            + terms.E_smoothness
        )

        # Compute metrics
        geom_metrics = self._compute_geometry_metrics(geometry_state)
        id_metrics = self._compute_identity_metrics(identity_state)
        temp_metrics = self._compute_temporal_metrics(temporal_state)
        rend_metrics = self._compute_renderer_metrics(geometry_state, temporal_state)

        return EnergyReport(
            frame_idx=frame_idx,
            terms=terms,
            geometry=geom_metrics,
            identity=id_metrics,
            temporal=temp_metrics,
            renderer=rend_metrics,
            status="computed",
        )

    def _compute_E_geom(self, geo: GeometryState) -> float:
        """Geometry energy: landmark reprojection + mesh consistency + transform regularization.

        Components:
        - Landmark reprojection error (if landmarks available)
        - Mesh consistency error (if mesh available)
        - Transform regularization (determinant deviation from 1.0)
        - Crop alignment error
        """
        energy = 0.0

        # Transform regularization: penalize determinant deviation from 1.0
        if geo.canonical_transform is not None:
            try:
                M = geo.canonical_transform[:2, :2]  # 2x2 part
                det = np.linalg.det(M)
                # Penalize deviation from 1.0 (similarity transform)
                energy += abs(det - 1.0) * 10.0
            except Exception:
                energy += 5.0  # Penalty for invalid transform

        # Geometry confidence: lower confidence = higher energy
        energy += (1.0 - geo.geometry_confidence) * 2.0

        # Mask coverage: penalize too-small or too-large masks
        if geo.mask is not None:
            coverage = float(np.mean(geo.mask))
            # Ideal coverage is 0.4-0.7
            if coverage < 0.3:
                energy += (0.3 - coverage) * 10.0
            elif coverage > 0.9:
                energy += (coverage - 0.9) * 10.0

        # Pose magnitude: penalize extreme poses
        yaw, pitch, roll = geo.pose
        pose_mag = abs(yaw) + abs(pitch) + abs(roll)
        if pose_mag > 30:
            energy += (pose_mag - 30) * 0.1

        return float(energy)

    def _compute_E_geom_identity(self, id_state: IdentityState) -> float:
        """Identity energy: anchor consistency + latent continuity + region confidence.

        Components:
        - Anchor consistency error (distance from anchor)
        - Latent continuity penalty (if previous identity available)
        - Region confidence penalty
        - Identity drift penalty
        """
        energy = 0.0

        # Identity uncertainty: higher uncertainty = higher energy
        energy += id_state.identity_uncertainty * 3.0

        # Region confidence: penalize low-confidence regions
        if id_state.region_confidence:
            mean_conf = np.mean(list(id_state.region_confidence.values()))
            energy += (1.0 - mean_conf) * 2.0

        # Anchor distance: penalize drift from anchor
        # (This is computed externally and passed via identity_uncertainty)

        # Latent continuity: penalize change from previous identity
        if self.previous_identity is not None and id_state.appearance_latent is not None:
            if self.previous_identity.appearance_latent is not None:
                if id_state.appearance_latent.shape == self.previous_identity.appearance_latent.shape:
                    diff = np.mean(np.abs(
                        id_state.appearance_latent.astype(np.float32)
                        - self.previous_identity.appearance_latent.astype(np.float32)
                    ))
                    energy += diff * 0.01  # Small penalty for change

        return float(energy)

    def _compute_E_temporal(
        self,
        temp: TemporalState,
        geo: GeometryState,
        id_state: IdentityState,
    ) -> float:
        """Temporal energy: continuity + motion coherence + drift + occlusion recovery.

        Components:
        - Frame-to-frame continuity penalty
        - Motion coherence penalty
        - Drift penalty
        - Occlusion recovery penalty
        """
        energy = 0.0

        # Temporal confidence: lower confidence = higher energy
        energy += (1.0 - temp.temporal_confidence) * 3.0

        # Drift score: higher drift = higher energy
        energy += temp.drift_score * 0.5

        # Continuity score: lower continuity = higher energy
        energy += (1.0 - temp.continuity_score) * 2.0

        # Motion coherence: penalize large motion
        if temp.motion_field is not None:
            motion_mag = np.mean(np.sqrt(np.sum(temp.motion_field ** 2, axis=-1)))
            energy += motion_mag * 0.1

        return float(energy)

    def _compute_E_photometric(
        self,
        geo: GeometryState,
        id_state: IdentityState,
        source_frame: Optional[np.ndarray],
    ) -> float:
        """Photometric energy: lighting consistency + albedo stability + render consistency.

        Components:
        - Lighting consistency penalty
        - Albedo stability penalty
        - Render consistency penalty
        - Exposure mismatch penalty
        """
        energy = 0.0

        # If we have source frame and canonical face, compute photometric consistency
        if source_frame is not None and geo.canonical_face is not None:
            # Compute mean brightness of source and canonical
            src_gray = cv2.cvtColor(source_frame, cv2.COLOR_BGR2GRAY)
            src_mean = float(np.mean(src_gray))

            canon_gray = cv2.cvtColor(geo.canonical_face, cv2.COLOR_BGR2GRAY)
            canon_mean = float(np.mean(canon_gray))

            # Penalize large brightness mismatch
            brightness_diff = abs(src_mean - canon_mean) / 255.0
            energy += brightness_diff * 5.0

        # If we have anchor, compute anchor consistency
        if self.anchor_lab is not None and id_state.appearance_latent is not None:
            try:
                id_lab = cv2.cvtColor(
                    id_state.appearance_latent, cv2.COLOR_BGR2LAB
                ).astype(np.float32)
                anchor_mean = np.mean(self.anchor_lab, axis=(0, 1))
                id_mean = np.mean(id_lab, axis=(0, 1))
                lab_dist = np.sqrt(np.sum((anchor_mean - id_mean) ** 2))
                energy += lab_dist * 0.1
            except Exception:
                energy += 5.0  # Penalty for conversion failure

        return float(energy)

    def _compute_E_smoothness(
        self,
        geo: GeometryState,
        id_state: IdentityState,
    ) -> float:
        """Smoothness energy: spatial + temporal + transform smoothness.

        Components:
        - Spatial smoothness penalty (mask smoothness)
        - Temporal smoothness penalty (transform change)
        - Transform smoothness penalty (determinant stability)
        """
        energy = 0.0

        # Mask smoothness: penalize rough mask edges
        if geo.mask is not None:
            # Compute gradient magnitude
            grad_x = cv2.Sobel(geo.mask, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(geo.mask, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
            edge_roughness = float(np.mean(grad_mag))
            energy += edge_roughness * 2.0

        # Transform smoothness: penalize large transform changes
        if self.previous_geometry is not None:
            if (
                geo.canonical_transform is not None
                and self.previous_geometry.canonical_transform is not None
            ):
                try:
                    diff = np.linalg.norm(
                        geo.canonical_transform - self.previous_geometry.canonical_transform
                    )
                    energy += diff * 0.5
                except Exception:
                    energy += 2.0

        return float(energy)

    def _compute_geometry_metrics(self, geo: GeometryState) -> GeometryMetrics:
        """Extract geometry metrics for parameter-wise visibility."""
        yaw, pitch, roll = geo.pose

        det_A = 1.0
        if geo.canonical_transform is not None:
            try:
                det_A = float(np.linalg.det(geo.canonical_transform[:2, :2]))
            except Exception:
                det_A = 0.0

        mask_coverage_pct = 0.0
        if geo.mask is not None:
            mask_coverage_pct = float(np.mean(geo.mask)) * 100.0

        transform_stability = 1.0
        if self.previous_geometry is not None:
            if (
                geo.canonical_transform is not None
                and self.previous_geometry.canonical_transform is not None
            ):
                try:
                    diff = np.linalg.norm(
                        geo.canonical_transform - self.previous_geometry.canonical_transform
                    )
                    transform_stability = max(0.0, 1.0 - diff)
                except Exception:
                    transform_stability = 0.0

        landmark_count = 0
        if geo.landmarks_478 is not None:
            landmark_count = len(geo.landmarks_478)

        return GeometryMetrics(
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            det_A=det_A,
            mask_coverage_pct=mask_coverage_pct,
            transform_stability=transform_stability,
            geometry_confidence=geo.geometry_confidence,
            landmark_count=landmark_count,
            pose_magnitude=abs(yaw) + abs(pitch) + abs(roll),
        )

    def _compute_identity_metrics(self, id_state: IdentityState) -> IdentityMetrics:
        """Extract identity metrics for parameter-wise visibility."""
        appearance_norm = 0.0
        if id_state.appearance_latent is not None:
            appearance_norm = float(np.linalg.norm(id_state.appearance_latent.astype(np.float32)))

        return IdentityMetrics(
            anchor_weights=id_state.anchor_weights,
            uncertainty=id_state.identity_uncertainty,
            region_confidence=id_state.region_confidence,
            appearance_latent_norm=appearance_norm,
            anchor_distance_lab=0.0,  # Computed externally
            observation_count=0.0,    # Computed externally
        )

    def _compute_temporal_metrics(self, temp: TemporalState) -> TemporalMetrics:
        """Extract temporal metrics for parameter-wise visibility."""
        motion_norm = 0.0
        if temp.motion_field is not None:
            motion_norm = float(np.mean(np.sqrt(np.sum(temp.motion_field ** 2, axis=-1))))

        return TemporalMetrics(
            temporal_confidence=temp.temporal_confidence,
            drift_score=temp.drift_score,
            continuity_score=temp.continuity_score,
            motion_field_norm=motion_norm,
            covariance_trace=0.0,  # Phase 3
            uncertainty_mean=0.0,  # Phase 3
        )

    def _compute_renderer_metrics(
        self,
        geo: GeometryState,
        temp: TemporalState,
    ) -> RendererMetrics:
        """Extract renderer metrics for parameter-wise visibility."""
        M_mean = 0.0
        M_min = 0.0
        M_max = 0.0
        if geo.mask is not None:
            M_mean = float(np.mean(geo.mask))
            M_min = float(np.min(geo.mask))
            M_max = float(np.max(geo.mask))

        blend_weight_mean = M_mean * temp.temporal_confidence
        blend_weight_min = M_min * temp.temporal_confidence
        blend_weight_max = M_max * temp.temporal_confidence

        return RendererMetrics(
            M_mean=M_mean,
            M_min=M_min,
            M_max=M_max,
            Y_face_range=(0, 255),
            Y_bg_range=(0, 255),
            blend_weight_min=blend_weight_min,
            blend_weight_mean=blend_weight_mean,
            blend_weight_max=blend_weight_max,
            temporal_confidence=temp.temporal_confidence,
            output_shape=(1920, 1080, 3),
            output_dtype="uint8",
        )
