# Face OS v3.8 — Compact State Reference

**Last updated:** 2026-05-30 | **Tests:** 236 collected in tests/face_os/ (228 fast + 8 slow runtime-truth) | **Source:** ~15,100 lines

This file is the current source of truth for Face OS. Older status files may
describe historical drift; use this file for current runtime alignment.

---

## Architecture Map

```
pipeline.py (single orchestration runtime)
    ├── detect_track.py / landmarks.py / crop_planner.py
    ├── canonical_map.py
    ├── identity_state.py / patch_memory.py
    ├── intrinsic_decomposition.py
    ├── dense_geometry.py
    ├── physical_renderer.py
    ├── compositor.py
    ├── photometric.py
    ├── state_evolution.py / temporal_solve.py / lie_group.py
    ├── energy_scaling.py (now gates rendering decisions)
    ├── ab_validation.py (A/B comparison harness)
    └── subsystems/
        ├── identity_estimator.py
        ├── temporal_estimator.py
        └── renderer.py
```

## Drift Status

| ID | Requirement | Status | Current Evidence |
|---|---|---|---|
| D-01 | Signal-preserving render path | **ALIGNED** | Linear-light compositing via sRGB↔linear conversion. HF detail injection in linear space (fixed C-01). Consistent detail→sharpen→photometric order across all paths. |
| D-02 | PhysicalRenderer improves quality vs alpha | **ALIGNED MECHANISM** | Real Lambertian+Blinn-Phong shading, real mesh normals. A/B framework with SSIM, LAB drift, Procrustes metrics. |
| D-03 | Benchmark corpus | **PARTIAL** | Real video integration tests exist using input/video.mp4. Synthetic hard-condition tests also present. |
| D-04 | Dense geometry integration | **ALIGNED** | Physical path calls DenseGeometryEstimator.estimate() with anatomical anchor-based landmark mapping (fixed H-07). |
| D-05 | Identity decoupling | **PARTIAL — Phase 2A render live (quality WIP)** | Lighting-invariant `IdentityLatent` is owned by `IdentityEstimator`. Fusion is an **honest per-pixel Kalman filter** (design.md:354-361): pure shrink `unc <- (1-gain)*unc` on every positive-quality observation; the ONLY inflation source is the temporal predict step (`drift_score`). A running-max "ratchet" (removed) had collapsed confidence to ~0 on real video (0.234→0.006); the corrected model rises with evidence to a fixed-point plateau (0.234→0.257), governed honestly by the decomposer's `albedo_uncertainty`. Pinned by P4a/P4b/P4c. **Phase 2 (latent DRIVES pixels):** with `render_source='latent'` the latent renders the face via `synthesize_identity` → `estimate_lighting` (closed-form Lambertian inverse, from the OBSERVATION) → `FaceRenderer.render_from_latent` (delegates to `render_with_intrinsic`, `observed=None`, fatal contract), composited as a PEER branch in `_render_core` that skips the source-HF tail — retiring A-2/A-3/A-5 on that path. PROVEN on real video: `TestLatentRenderModeOnRealVideo` 4/4 (latent_primary=True, render_path='latent', source_pixel_fraction≈0.80). **Default stays `legacy`** (shadow `TestLatentShadowModeOnRealVideo` 4/4 unchanged; fast 228, 0 regressions). **Phase 2B OPEN (re-measured 2026-05-30, real clip, 6 frames, both paths):** the earlier "~2.1× too bright (176 vs 82)" note was WRONG — never verified against the composited frame. The honest A/B is: latent vs legacy face-bbox mean **0.93×**, and **SSIM(latent,legacy)=0.9947** — the latent output is ~indistinguishable from legacy paste-then-relight. The decoupling (the entire point of D-05) is NOT yet visible in pixels. Two stacked root causes: (1) **flat render** — `geom_state` is only built when the verification gate accepts the frame (`identity_updated`, pipeline.py:1327); on RECOVERY frames it is None, so `_render_with_latent` gets `mesh=None` and `_normals_for` falls back to the ellipsoid face-prior (identity_estimator.py:729) → the Lambertian fit collapses → near-flat face. Fix = source real mesh normals from `face_track.mesh_478` into the render geometry independent of the fusion gate. (2) **telemetry measured the wrong quantity** — `source_pixel_fraction` was computed as `1 - mean(feathered_mask)` over the WHOLE crop (≈0.80 = background fraction), but the spec (requirements.md:32, design.md:480) defines it as the fraction INSIDE the face mask traceable to source (target <0.02). The "no-leak" proof was meaningless until this is measured over the mask interior. NOT hot-fixed by rescaling. Standing verification: real-video HTML A/B report run after every change.
| D-06 | Predictive temporal belief | **ALIGNED MECHANISM** | SIM(2) velocity prediction computed and used for 1-2 frame occlusion recovery (fixed H-02). True SE(2)/SIM(2) Lie algebra exp/log (fixed H-04). |
| D-07 | State-space runtime brain | **PARTIAL** | Energy scaling now gates rendering decisions (fixed H-03). Runtime remains procedural. optimizer_architecture.py is stranded. |
| D-08 | Per-frame truthful telemetry | **ALIGNED** | `_emit_frame_telemetry()` accepts explicit branch truth. All paths emit explicit telemetry including energy gating reasons. |
| D-09 | Visual regression validation | **ALIGNED** | Integration tests validate sharpness, contrast ratio, flicker, NaN/Inf, telemetry schema on real video. |
| D-10 | Probabilistic architectural closure | **PARTIAL / PHASE C** | Subsystem wrappers are real delegates. Factor graph not implemented. |

**Current honest summary:** 5 aligned mechanisms/areas, 5 partial architectural areas.

## Key Entry Points

| Function | File:Line | Purpose |
|---|---:|---|
| `process_frame` | pipeline.py:698 | Public single-frame API for tests and A/B validation |
| `_process_frame_v2` | pipeline.py:~1000 | Forward-only frame processing |
| `_render_core` | pipeline.py:~1680 | Single render branch selector with energy gating |
| `_render_with_physical_renderer` | pipeline.py:~1770 | Physical renderer + dense-geometry path |
| `_emit_frame_telemetry` | pipeline.py:~1519 | Per-frame explicit telemetry schema |
| `_postprocess_rendered_crop` | pipeline.py:~1600 | Shared sharpen + photometric lock |
| `multiband_blend` | compositor.py:92 | Laplacian pyramid blend |
| `photometric_lock` | photometric.py:29 | LAB temporal luminance lock |
| `render_with_mesh` | physical_renderer.py:329 | Mesh-derived normal rendering |
| `compare_render_methods` | ab_validation.py:270 | Physical vs alpha A/B harness |
| `SE2Transform.exp/log` | lie_group.py:43-100 | True SE(2) Lie algebra maps |
| `SIM2Transform.exp/log` | lie_group.py:100-140 | True SIM(2) Lie algebra maps |

## Test Suite

| File | Current Role |
|---|---|
| test_integration.py | End-to-end pipeline, telemetry, render quality, compositor, Lie group, geometry, identity, A/B |

## Run Commands

```bash
# All tests
.venv/bin/python -m pytest tests/face_os/ -v

# Fast tests only (no video)
.venv/bin/python -m pytest tests/face_os/ -v -m "not slow"

# Slow tests only (requires input/video.mp4)
.venv/bin/python -m pytest tests/face_os/ -v -m "slow"
```
