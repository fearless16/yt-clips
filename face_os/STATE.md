# Face OS v3.5 — Compact State Reference

**Last updated:** 2026-05-22 | **Tests:** 603/603 | **Source:** 14,547 lines

---

## Architecture Map (memory graph: `graphify-out/memory_graph.json`)

```
pipeline.py (2050L) ──orchestrates──→ subsystems/ (real delegates)
    ├── detect_track.py (548L)        geometry_estimator.py (105L)
    ├── landmarks.py (355L)           identity_estimator.py (100L)
    ├── canonical_map.py (423L)       temporal_estimator.py (70L)
    ├── identity_state.py (1230L)     renderer.py (100L)
    ├── patch_memory.py (571L)        boundary.py (39L)
    ├── temporal_solve.py (410L)
    ├── face_enhance.py (784L)
    ├── crop_planner.py (390L)
    ├── compositor.py (320L)          ← D-01a: linear-light, D-01c: multiband
    ├── physical_renderer.py (557L)   ← D-04: render_with_mesh()
    ├── intrinsic_decomposition.py (607L)
    ├── state_evolution.py (379L)     ← D-06: predict_with_velocity
    ├── renderer_mode.py (141L)
    ├── energy_scaling.py (239L)
    ├── lie_group.py (400L)
    └── types.py (527L)
```

## Drift Status (LOCKED_ARCHITECTURE.md)

| ID | Requirement | Status | Evidence |
|---|---|---|---|
| D-01a | Linear-light compositing | **ALIGNED** | `compositor.py:33-89` — full sRGB→linear→blend→sRGB |
| D-01b | Single-resample | **ALIGNED** | `pipeline.py:1214-1227` — single warpAffine with mask concat |
| D-01c | Multi-band compositing | **ALIGNED** | `compositor.py:92-160` — Laplacian pyramid blend |
| D-01d | Post-composite sharpening | **ALIGNED** | `pipeline.py:1162,1492,1523` — `amount=0.8, radius=0.8` |
| D-01e | Photometric lock | **ALIGNED** | `photometric.py:29-64` — LAB EMA, called in all 3 render paths |
| D-02 | PhysicalRenderer A/B | **ALIGNED** | `ab_validation.py:601-738` — uses `process_frame()` + `render_mode_override` |
| D-03 | Benchmark conditions | **PARTIAL** | Tests validate frame generation only, not pipeline components |
| D-04 | Dense geometry | **ALIGNED** | `pipeline.py:1620-1634` calls `estimate()` then `render_with_mesh()` |
| D-05 | Identity decoupling | **ALIGNED** | `identity_state.py:930-999` `query_albedo()` called via `IdentityEstimator` |
| D-06 | Temporal prediction | **ALIGNED** | `state_evolution.py:290-319` called via `TemporalEstimator` |
| D-07 | Factor-graph inference | **NOT NEEDED** | Phase C — current Kalman + SIM(2) is sufficient |
| D-08 | Per-frame telemetry | **ALIGNED** | `pipeline.py:1334-1364` — all 8 fields emitted from all 3 paths |
| D-09 | Visual regression tests | **ALIGNED** | `test_visual_regression.py` — sharpness, frequency, contrast, flicker |
| D-10 | Subsystem architecture | **ALIGNED** | All 4 wrappers called from pipeline runtime path |

**10 ALIGNED, 1 PARTIAL, 1 NOT NEEDED (D-07)**

## Key Entry Points

| Function | File:Line | Purpose |
|---|---|---|
| `_render_core` | pipeline.py:1451 | Single render path (all 3 modes) |
| `_process_frame_v2` | pipeline.py:890 | Forward-only frame processing |
| `process_frame` | pipeline.py:569 | Public API for A/B validation |
| `_render_frame_v2` | pipeline.py:1103 | Bidirectional render pass |
| `_update_v3_modules` | pipeline.py:1304 | Kalman + velocity + telemetry |
| `_emit_frame_telemetry` | pipeline.py:1340 | Per-frame JSON log |
| `_composite_identity_to_crop` | pipeline.py:1220 | Identity warp + blend (single resample) |
| `_render_with_physical_renderer` | pipeline.py:1555 | Physical renderer path |
| `photometric_lock` | compositor.py:140 | Temporal luminance EMA |
| `multiband_blend` | compositor.py:92 | Laplacian pyramid blend |
| `_blend_linear` | compositor.py:52 / pipeline.py:95 | Linear-light alpha blend |
| `query_albedo` | identity_state.py:935 | Lighting-invariant identity |
| `predict_with_velocity` | state_evolution.py:290 | SIM(2) velocity extrapolation |
| `render_with_mesh` | physical_renderer.py:370 | Mesh-derived normals render |
| `compare_render_methods` | ab_validation.py:608 | A/B PhysicalRenderer vs alpha |

## Test Suite

| File | Tests | Purpose |
|---|---|---|
| test_strict_regression.py | 31 | Frame contract, mask, NaN, EMA |
| test_v31_consolidation.py | 49 | V3.1 rules, telemetry, per-frame log |
| test_visual_regression.py | 32 | Sharpness, frequency, contrast, flicker |
| test_benchmark_conditions.py | 15 | Rotation, occlusion, lowlight, blur, noise |
| test_math_hardening.py | 37 | 10 invariant classes |
| test_phase1_hardening.py | 37 | Long-horizon, identifiability, renderer |
| test_adversarial.py | 31 | Pathological inputs |
| test_state_space.py | 39 | LatentState, transitions |
| test_arch_regression.py | 16 | Photometric lock, blend_linear |
| Others (18 files) | 316 | Module-specific tests |

## Run Commands

```bash
.venv/bin/python -m pytest tests/face_os/ -v                    # all 603
.venv/bin/python -m pytest tests/face_os/test_strict_regression.py -v  # 31 strict
.venv/bin/python -m pytest tests/face_os/test_visual_regression.py -v  # 32 visual
```
