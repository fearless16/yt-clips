# Face OS v3.9 — Compact State Reference

**Last updated:** 2026-06-02 | **Tests:** 441 passed, 0 failed, 3 skipped | **Source:** ~17,300 lines

This file is the current source of truth for Face OS. Older status files may
describe historical drift; use this file for current runtime alignment.

---

## Architecture Map

```
pipeline.py (single orchestration runtime)
    ├── config.py
    ├── ingest.py
    ├── detect_track.py / landmarks.py / crop_planner.py
    ├── canonical_map.py
    ├── identity_state.py / patch_memory.py
    ├── intrinsic_decomposition.py
    ├── dense_geometry.py
    ├── physical_renderer.py
    ├── compositor.py
    ├── photometric.py
    ├── face_enhance.py
    ├── state_evolution.py / temporal_solve.py / lie_group.py
    ├── energy_scaling.py (gates rendering decisions)
    ├── renderer_mode.py (physical/alpha/hybrid state machine)
    ├── accept_gate.py (central accept/reject gate)
    ├── visibility.py (per-UV geometric visibility)
    ├── reconstruction_confidence.py (C_recon composite)
    ├── ab_validation.py (validation harness)
    ├── export_qc.py
    ├── subsystems/
    │   ├── identity_estimator.py
    │   ├── temporal_estimator.py
    │   ├── geometry_estimator.py
    │   └── renderer.py
    └── stranded/ (not imported by pipeline — see below)
        ├── identity_manifold.py
        ├── crop_objective.py
        ├── mesh_mask.py
        ├── benchmark_suite.py
        ├── audit.py
        ├── gen_report.py
        └── architectural_completeness.py
```

## Stranded / Dormant Modules

These modules exist but are **not imported by the pipeline runtime**:

| Module | Lines | Status | Notes |
|--------|------:|--------|-------|
| `identity_manifold.py` | 422 | Dormant | Riemannian manifold (16-D). arch.md confirms deferred. |
| `crop_objective.py` | 497 | Dormant | Alternative crop planner (`C* = argmin(E_crop)`). Has tests but no pipeline integration. |
| `mesh_mask.py` | 397 | Dormant | Mesh-derived semantic masking with SDF feathering. Has tests but no pipeline integration. |
| `benchmark_suite.py` | 397 | Test-only | Synthetic clip generators + benchmark metrics. Used by `test_benchmark_suite.py` only. |
| `audit.py` | 1,004 | CLI tool | Standalone mathematical diagnostic. Not imported. |
| `gen_report.py` | 382 | CLI tool | Standalone HTML report generator. Not imported. |
| `architectural_completeness.py` | 195 | Orphaned | Tracks module completeness levels. Zero imports anywhere. |

Utility files (standalone, not pipeline modules): `colab_server.py`, `colab_client.py`, `colab_drive.py`, `colab_notebook.py`.

## Drift Status

| ID | Requirement | Status | Current Evidence |
|---|---|---|---|
| D-01 | Signal-preserving render path | **ALIGNED** | Linear-light compositing via sRGB↔linear conversion. HF detail injection in linear space (fixed C-01). Consistent detail→sharpen→photometric order across all paths. |
| D-02 | PhysicalRenderer improves quality vs alpha | **ALIGNED MECHANISM** | Real Lambertian+Blinn-Phong shading, real mesh normals. A/B framework with SSIM, LAB drift, Procrustes metrics. |
| D-03 | Benchmark corpus | **ALIGNED** | 13 synthetic hard-condition generators (overexposure, webcam_noise, rolling_shutter, beard_shadow, face_cutoff, etc.). `run_benchmark()` populates `BenchmarkMetrics` from suite clips. `create_default_suite()` ships 12-clip default corpus. 24 dedicated tests in `test_benchmark_suite.py`. |
| D-04 | Dense geometry integration | **ALIGNED** | Physical path calls DenseGeometryEstimator.estimate() with anatomical anchor-based landmark mapping (fixed H-07). Face-prior normal map + normal-variance edge protection wired into all render paths via `_postprocess_rendered_crop`. |
| D-05 | Identity decoupling | **ALIGNED — Latent is the SOLE render path (v3.9)** | Legacy paste-then-relight path completely removed. `render_source` selector, production/forced gate policy, and shadow mode deleted from config defaults (2026-06-02). Latent drives the face unconditionally via `synthesize_identity` → `estimate_lighting` → `render_from_latent`. `corpus_validate()` replaces `corpus_compare_sources()` for single-pass latent validation. |
| D-06 | Predictive temporal belief | **ALIGNED MECHANISM** | SIM(2) velocity prediction computed and used for 1-2 frame occlusion recovery (fixed H-02). True SE(2)/SIM(2) Lie algebra exp/log (fixed H-04). |
| D-07 | State-space runtime brain | **NOT NEEDED (v3.x)** | Current Kalman filter (state_evolution.py) + SIM(2) velocity prediction + energy_scaling is architecturally sufficient for temporal consistency. optimizer_architecture.py has been deleted (stranded, zero runtime integration). |
| D-08 | Per-frame truthful telemetry | **ALIGNED** | `_emit_frame_telemetry()` accepts explicit branch truth. All paths emit explicit telemetry including energy gating reasons. `gate_state` hardcoded to "engaged" post-legacy-removal. |
| D-09 | Visual regression validation | **ALIGNED** | Integration tests validate sharpness, contrast ratio, flicker, NaN/Inf, telemetry schema on real video. |
| D-10 | Probabilistic architectural closure | **NOT NEEDED (v3.x)** | Subsystem wrappers (IdentityEstimator, TemporalEstimator, GeometryEstimator, FaceRenderer) are real runtime delegates. Factor-graph inference, uncertainty propagation, MAP runtime are explicitly deferred to Phase C. |

**Current honest summary:** 7 aligned mechanisms/areas, 3 Phase C deferred, 0 broken.

## Key Entry Points

| Function | File:Line | Purpose |
|---|---:|---|
| `process_frame` | pipeline.py:834 | Public single-frame API for tests and validation |
| `_process_frame_v2` | pipeline.py:1155 | Forward-only frame processing |
| `_render_core` | pipeline.py:1985 | Single render branch selector with energy gating |
| `_render_with_physical_renderer` | pipeline.py:2299 | Physical renderer + dense-geometry path |
| `_emit_frame_telemetry` | pipeline.py:1736 | Per-frame explicit telemetry schema |
| `_postprocess_rendered_crop` | pipeline.py:1909 | Shared sharpen + photometric lock |
| `multiband_blend` | compositor.py:71 | Laplacian pyramid blend |
| `photometric_lock` | photometric.py:44 | LAB temporal luminance lock |
| `render_with_mesh` | physical_renderer.py:612 | Mesh-derived normal rendering |
| `compare_render_methods` | ab_validation.py:378 | Physical vs alpha A/B harness |
| `corpus_validate` | ab_validation.py:480 | Single-pass latent validation over corpus |
| `SE2Transform.exp/log` | lie_group.py:43-100 | True SE(2) Lie algebra maps |
| `SIM2Transform.exp/log` | lie_group.py:100-140 | True SIM(2) Lie algebra maps |

## Test Suite

| File | Tests | Role |
|---|---:|---|
| test_integration.py | 72 | End-to-end pipeline, telemetry, render quality, compositor, Lie group, geometry, identity, A/B |
| test_latent_identity.py | 59 | Latent identity path: synthesis, update, gate, uncertainty, no-leak |
| test_appearance_encoder.py | 38 | Appearance encoder (JL projection, geodesic outlier rejection) |
| test_signal_fidelity.py | 34 | Signal-preserving render path validation |
| test_ab_comparator_latent.py | 29 | A/B physical-vs-alpha comparison + corpus validation |
| test_benchmark_suite.py | 24 | Synthetic clip generators, benchmark metrics |
| test_physical_renderer.py | 22 | Lambertian + Blinn-Phong rendering, energy conservation |
| test_compositor.py | 21 | Linear-light compositing, multiband blend |
| test_intrinsic_decomposition.py | 21 | Albedo/shading/specular decomposition contract |
| test_lie_group.py | 19 | SE(2)/SIM(2) exp/log, composition, inverse |
| test_lighting_coverage.py | 18 | Lighting bin, coverage tracking |
| test_pose_coverage.py | 16 | Pose bin, coverage tracking |
| test_dense_geometry.py | 13 | Dense mesh estimation from landmarks |
| test_telemetry_temporal.py | 13 | Temporal telemetry schema |
| test_renderer_mode.py | 12 | Renderer mode state machine |
| test_visibility.py | 12 | Per-UV visibility field, memory gating invariant |
| test_reconstruction_confidence.py | 8 | C_recon composite confidence |
| test_appearance_hybrid.py | 7 | Appearance hybrid blend |
| test_mesh_mask.py | 6 | Mesh-derived semantic masking |
| test_crop_objective.py | 5 | Crop objective function |
| test_temporal_confidence_only.py | 5 | Temporal confidence propagation |
| test_color_cast.py | 4 | Color cast detection/correction |
| test_accept_gate.py | 3 | Accept/reject gate |
| **Total** | **441** | |

## Run Commands

```bash
# All tests
.venv/bin/python -m pytest face_os/tests/ -v

# Fast tests only (no video)
.venv/bin/python -m pytest face_os/tests/ -v -m "not slow"

# Slow tests only (requires input/video.mp4)
.venv/bin/python -m pytest face_os/tests/ -v -m "slow"
```
