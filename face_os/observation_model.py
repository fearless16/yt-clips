"""§16.1 Observation Model — forward prediction residual and noise estimate.

arch.md §16.1:
    O_t = R(I_t, p_t, l_t) + ε_t

The source frame is a noisy observation of identity under pose and lighting.
The latent render IS the forward prediction:

    Ô_t = R(latent_albedo, pose_t, lighting_t)

The residual ``‖O_t − Ô_t‖`` measures how well the latent explains the
observation. A small residual means the stored identity + estimated lighting
faithfully reproduce the scene; a large residual means the latent is wrong
(identity drift, lighting misestimate, or extreme expression).

ε_t is the per-pixel observation noise — the part of the observation the
forward model cannot explain. It is represented as a per-pixel LAB-space
difference, masked to the face interior so background never pollutes it.

Invariant (arch §16.1): the forward-model residual is finite, bounded, and
decreases as the latent converges on a held frame.

This module is pure computation — no pipeline imports, no side effects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class ObservationResidual:
    """Forward-model residual ``O_t − Ô_t`` (arch §16.1).

    All comparisons are in CIELAB space (perceptually uniform, consistent with
    the rest of the pipeline — photometric.py, reconstruction_confidence.py).
    """

    residual_map: np.ndarray
    noise_map: np.ndarray
    residual_mean: float
    residual_max: float
    noise_mean: float
    observation_confidence: float


def _to_lab(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB).astype(np.float32)


def compute_observation_residual(
    predicted: np.ndarray,
    observed: np.ndarray,
    mask: Optional[np.ndarray] = None,
    sigma: float = 3.0,
) -> ObservationResidual:
    """Compute the forward-model residual ``O_t − Ô_t`` in LAB space (§16.1).

    Both inputs are low-passed before comparison so the residual captures
    STRUCTURAL lighting/identity mismatch, not high-frequency texture that the
    latent is not expected to reproduce.

    Args:
        predicted: ``Ô_t`` — the pure latent render (float32 [0,1] or uint8 BGR).
        observed: ``O_t`` — the source crop observation (same dtype/space).
        mask: Optional ``(H, W)`` float face mask. Residual is computed only
            where ``mask > 0.5``. If ``None``, the full image is used.
        sigma: Gaussian low-pass sigma for structural comparison.

    Returns:
        :class:`ObservationResidual` with per-pixel and scalar summaries.
    """
    pred = np.asarray(predicted, dtype=np.float32)
    obs = np.asarray(observed, dtype=np.float32)

    if pred.max() > 1.5:
        pred = pred / 255.0
    if obs.max() > 1.5:
        obs = obs / 255.0

    if pred.shape[:2] != obs.shape[:2]:
        pred = cv2.resize(pred, (obs.shape[1], obs.shape[0]))

    ksize = max(3, int(round(sigma * 6)) | 1)
    pred_lp = cv2.GaussianBlur(pred, (ksize, ksize), sigma)
    obs_lp = cv2.GaussianBlur(obs, (ksize, ksize), sigma)

    pred_lab = _to_lab(pred_lp)
    obs_lab = _to_lab(obs_lp)

    diff = obs_lab - pred_lab
    per_pixel = np.sqrt(np.sum(diff ** 2, axis=2))

    if mask is not None:
        m = np.asarray(mask, dtype=np.float32)
        if m.shape[:2] != per_pixel.shape[:2]:
            m = cv2.resize(m, (per_pixel.shape[1], per_pixel.shape[0]))
        interior = m > 0.5
    else:
        interior = np.ones(per_pixel.shape[:2], dtype=bool)

    n = int(interior.sum())
    if n == 0:
        empty = np.zeros(per_pixel.shape[:2], dtype=np.float32)
        return ObservationResidual(
            residual_map=empty,
            noise_map=diff,
            residual_mean=0.0,
            residual_max=0.0,
            noise_mean=0.0,
            observation_confidence=0.0,
        )

    masked_residual = per_pixel * interior.astype(np.float32)
    residual_mean = float(np.sum(masked_residual) / n)
    residual_max = float(np.max(masked_residual))

    noise_per_pixel = np.sqrt(np.sum(diff ** 2, axis=2))
    noise_masked = noise_per_pixel * interior.astype(np.float32)
    noise_mean = float(np.sum(noise_masked) / n)

    observation_confidence = float(np.exp(-residual_mean / 30.0))
    observation_confidence = float(np.clip(observation_confidence, 0.0, 1.0))

    return ObservationResidual(
        residual_map=masked_residual.astype(np.float32),
        noise_map=diff.astype(np.float32),
        residual_mean=residual_mean,
        residual_max=residual_max,
        noise_mean=noise_mean,
        observation_confidence=observation_confidence,
    )
