# Face OS v3.8 — Compact State Reference

**Last updated:** 2026-05-23 | **Tests:** 28 integration | **Source:** ~15,100 lines

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
| D-05 | Identity decoupling | **PARTIAL** | Albedo query path exists. Core BeliefPixel is still RGB pixel buffer, but intrinsic decomposition feeds renderer. |
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
