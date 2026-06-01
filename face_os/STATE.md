# Face OS v3.9 — Compact State Reference

**Last updated:** 2026-06-01 | **Tests:** 441 passed, 0 failed, 3 skipped | **Source:** ~12,600 lines

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
    ├── ab_validation.py (validation harness)
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
| D-03 | Benchmark corpus | **ALIGNED** | 13 synthetic hard-condition generators (overexposure, webcam_noise, rolling_shutter, beard_shadow, face_cutoff, etc.). `run_benchmark()` populates `BenchmarkMetrics` from suite clips. `create_default_suite()` ships 12-clip default corpus. 57 dedicated tests in `test_benchmark_suite.py`. |
| D-04 | Dense geometry integration | **ALIGNED** | Physical path calls DenseGeometryEstimator.estimate() with anatomical anchor-based landmark mapping (fixed H-07). Face-prior normal map + normal-variance edge protection wired into all render paths via `_postprocess_rendered_crop`. |
| D-05 | Identity decoupling | **ALIGNED — Latent is the SOLE render path (v3.9)** | Legacy paste-then-relight path completely removed. `render_source` selector, production/forced gate policy, and shadow mode ALL deleted. Latent drives the face unconditionally via `synthesize_identity` → `estimate_lighting` → `render_from_latent`. `corpus_validate()` replaces `corpus_compare_sources()` for single-pass latent validation. |
| D-06 | Predictive temporal belief | **ALIGNED MECHANISM** | SIM(2) velocity prediction computed and used for 1-2 frame occlusion recovery (fixed H-02). True SE(2)/SIM(2) Lie algebra exp/log (fixed H-04). |
| D-07 | State-space runtime brain | **NOT NEEDED (v3.x)** | Current Kalman filter (state_evolution.py) + SIM(2) velocity prediction + energy_scaling is architecturally sufficient for temporal consistency. optimizer_architecture.py has been deleted (stranded, zero runtime integration). |
| D-08 | Per-frame truthful telemetry | **ALIGNED** | `_emit_frame_telemetry()` accepts explicit branch truth. All paths emit explicit telemetry including energy gating reasons. `gate_state` hardcoded to "engaged" post-legacy-removal. |
| D-09 | Visual regression validation | **ALIGNED** | Integration tests validate sharpness, contrast ratio, flicker, NaN/Inf, telemetry schema on real video. |
| D-10 | Probabilistic architectural closure | **NOT NEEDED (v3.x)** | Subsystem wrappers (IdentityEstimator, TemporalEstimator, GeometryEstimator, FaceRenderer) are real runtime delegates. Factor-graph inference, uncertainty propagation, MAP runtime are explicitly deferred to Phase C. |

**Current honest summary:** 7 aligned mechanisms/areas, 3 Phase C deferred, 0 broken.

## Key Entry Points

| Function | File:Line | Purpose |
|---|---:|---|
| `process_frame` | pipeline.py:698 | Public single-frame API for tests and validation |
| `_process_frame_v2` | pipeline.py:~1000 | Forward-only frame processing |
| `_render_core` | pipeline.py:~1680 | Single render branch selector with energy gating |
| `_render_with_physical_renderer` | pipeline.py:~1770 | Physical renderer + dense-geometry path |
| `_emit_frame_telemetry` | pipeline.py:~1519 | Per-frame explicit telemetry schema |
| `_postprocess_rendered_crop` | pipeline.py:~1600 | Shared sharpen + photometric lock |
| `multiband_blend` | compositor.py:92 | Laplacian pyramid blend |
| `photometric_lock` | photometric.py:29 | LAB temporal luminance lock |
| `render_with_mesh` | physical_renderer.py:329 | Mesh-derived normal rendering |
| `compare_render_methods` | ab_validation.py:378 | Physical vs alpha A/B harness |
| `corpus_validate` | ab_validation.py:480 | Single-pass latent validation over corpus |
| `SE2Transform.exp/log` | lie_group.py:43-100 | True SE(2) Lie algebra maps |
| `SIM2Transform.exp/log` | lie_group.py:100-140 | True SIM(2) Lie algebra maps |

## Test Suite

| File | Current Role |
|---|---|
| test_integration.py | End-to-end pipeline, telemetry, render quality, compositor, Lie group, geometry, identity, A/B |
| test_ab_comparator_latent.py | A/B physical-vs-alpha comparison + corpus validation |

## Run Commands

```bash
# All tests
.venv/bin/python -m pytest tests/face_os/ -v

# Fast tests only (no video)
.venv/bin/python -m pytest tests/face_os/ -v -m "not slow"

# Slow tests only (requires input/video.mp4)
.venv/bin/python -m pytest tests/face_os/ -v -m "slow"
```
