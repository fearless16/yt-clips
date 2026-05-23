# Face OS v3.7 — Compact State Reference

**Last updated:** 2026-05-23 | **Tests:** 614 collected | **Source:** 14,899 lines

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
    └── subsystems/
        ├── geometry_estimator.py
        ├── identity_estimator.py
        ├── temporal_estimator.py
        └── renderer.py
```

## Drift Status

| ID | Requirement | Status | Current Evidence |
|---|---|---|---|
| D-01 | Signal-preserving render path | **ALIGNED MECHANISM / QUALITY STILL BENCHMARKED** | Default compositor mode is implemented Laplacian multiband (`face_os_config.yaml`), identity warp combines face+mask in one `warpAffine`, `_postprocess_rendered_crop()` centralizes sharpening + photometric lock, and physical/alpha/enhancement paths call it. |
| D-02 | PhysicalRenderer improves quality vs alpha | **PARTIAL** | A/B framework exists and `render_mode_override='alpha'` is wired. Wrapper signature drift was fixed so mesh rendering can execute, but corpus-level SSIM/LPIPS/temporal superiority is not yet proven. |
| D-03 | Benchmark corpus | **PARTIAL** | Synthetic hard-condition tests exist and pipeline smoke tests are being added; still missing real hard/adversarial clip corpus with pass/fail visual metrics. |
| D-04 | Dense geometry integration | **ALIGNED MECHANISM** | Physical path calls `DenseGeometryEstimator.estimate()` then `FaceRenderer.render_with_mesh(..., shading=..., image_shape=...)`; telemetry records `mesh` vs `face_prior`. |
| D-05 | Identity decoupling | **PARTIAL** | Albedo query path exists, but canonical RGB memory still participates in reconstruction. |
| D-06 | Predictive temporal belief | **PARTIAL** | SIM(2) velocity prediction exists; no long-horizon particle/multi-hypothesis belief runtime yet. |
| D-07 | State-space runtime brain | **PARTIAL / PHASE C** | Latent/math modules exist but runtime remains procedural orchestration, not joint factor-graph inference. |
| D-08 | Per-frame truthful telemetry | **ALIGNED** | `_emit_frame_telemetry()` accepts explicit branch truth: `render_path`, `intrinsic_used`, `geometry_source`, `resample_count`, `transform_det`; lost-face/enhancement cannot inherit stale counters. |
| D-09 | Visual regression validation | **PARTIAL** | Visual tests cover sharpness, frequency, contrast, texture, flicker, linear-light blending, and wrapper/telemetry truth. Full perceptual regression over real clips remains open. |
| D-10 | Probabilistic architectural closure | **PARTIAL / PHASE C** | Subsystem wrappers are real delegates, but geometry/identity/lighting/temporal state are not jointly optimized as `argmax P(...)`. |

**Current honest summary:** 3 aligned mechanisms, 7 partial architectural/validation areas.

## Key Entry Points

| Function | File:Line | Purpose |
|---|---:|---|
| `process_frame` | pipeline.py:574 | Public single-frame API for tests and A/B validation |
| `_process_frame_v2` | pipeline.py:894 | Forward-only frame processing |
| `_render_frame_v2` | pipeline.py:1123 | Bidirectional render pass |
| `_composite_identity_to_crop` | pipeline.py:1232 | Identity+mask single-warp composite |
| `_update_v3_modules` | pipeline.py:1291 | Intrinsic/temporal/renderer telemetry updates |
| `_emit_frame_telemetry` | pipeline.py:1388 | Per-frame explicit telemetry schema |
| `_resolve_blend_mode` | pipeline.py:1445 | Maps config to implemented compositor modes |
| `_postprocess_rendered_crop` | pipeline.py:1453 | Shared sharpen + photometric lock |
| `_render_core` | pipeline.py:1504 | Single render branch selector |
| `_render_with_physical_renderer` | pipeline.py:1645 | Physical renderer + dense-geometry path |
| `multiband_blend` | compositor.py:92 | Laplacian pyramid blend |
| `photometric_lock` | photometric.py:29 | LAB temporal luminance lock |
| `render_with_mesh` | physical_renderer.py:329 | Mesh-derived normal rendering |
| `compare_render_methods` | ab_validation.py:608 | Physical vs alpha A/B harness |

## Test Suite

| File | Current Role |
|---|---|
| test_v31_consolidation.py | Runtime architecture, energy, telemetry truth |
| test_visual_regression.py | Sharpness, frequency, contrast, texture, flicker |
| test_phase3b_physical_renderer.py | Physical renderer and wrapper contract |
| test_benchmark_conditions.py | Synthetic hard/adversarial condition generation |
| test_arch_regression.py | Photometric lock, linear-light blend, renderer modes |
| test_phase3c_dense_geometry.py | Dense geometry estimator |
| test_phase3a_intrinsic.py | Intrinsic decomposition |
| test_strict_regression.py | Frame contracts and strict pipeline invariants |

## Run Commands

```bash
.venv/bin/python -m pytest tests/face_os/ --collect-only -q
.venv/bin/python -m pytest tests/face_os/test_v31_consolidation.py tests/face_os/test_phase3b_physical_renderer.py -q
.venv/bin/python -m pytest tests/face_os/test_arch_regression.py tests/face_os/test_visual_regression.py tests/face_os/test_phase3b_physical_renderer.py -q
```
