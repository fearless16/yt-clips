# Face OS v3.2 — Compact State Reference

**Last updated:** 2026-05-22 | **Tests:** 603/603 | **Source:** 14,202 lines

---

## Architecture Map (memory graph: `graphify-out/memory_graph.json`)

```
pipeline.py (1992L) ──orchestrates──→ subsystems/ (thin wrappers)
    ├── detect_track.py (537L)        geometry_estimator.py (105L)
    ├── landmarks.py (355L)           identity_estimator.py (63L)
    ├── canonical_map.py (442L)       temporal_estimator.py (70L)
    ├── identity_state.py (1230L)     renderer.py (63L)
    ├── patch_memory.py (571L)        boundary.py (39L)
    ├── temporal_solve.py (410L)
    ├── face_enhance.py (784L)
    ├── crop_planner.py (390L)
    ├── compositor.py (388L)          ← D-01a: linear-light, D-01c: multiband
    ├── physical_renderer.py (557L)   ← D-04: render_with_mesh()
    ├── intrinsic_decomposition.py (607L)
    ├── state_evolution.py (379L)     ← D-06: predict_with_velocity
    ├── renderer_mode.py (141L)
    ├── energy_scaling.py (239L)
    ├── lie_group.py (400L)
    └── types.py (527L)
```

## Drift Status (LOCKED_ARCHITECTURE.md)

| ID | Requirement | Status |
|---|---|---|
| D-01a | Linear-light compositing | ALIGNED |
| D-01b | Single-resample | ALIGNED |
| D-01c | Multi-band compositing | ALIGNED |
| D-01d | Consistent sharpening | ALIGNED |
| D-01e | Photometric locking | ALIGNED |
| D-02 | PhysicalRenderer A/B | PARTIAL |
| D-03 | Benchmark corpus | ALIGNED |
| D-04 | Dense geometry | PARTIAL |
| D-05 | Identity decoupling | ALIGNED |
| D-06 | Temporal prediction | ALIGNED |
| D-07 | State-space runtime | NOT ALIGNED (Phase C) |
| D-08 | Per-frame telemetry | ALIGNED |
| D-09 | Visual regression tests | ALIGNED |
| D-10 | Subsystem architecture | PARTIAL |

**10 ALIGNED, 3 PARTIAL, 1 NOT ALIGNED (D-07)**

## Key Entry Points

| Function | File:Line | Purpose |
|---|---|---|
| `_render_core` | pipeline.py:1407 | Single render path (all 3 modes) |
| `_process_frame_v2` | pipeline.py:890 | Forward-only frame processing |
| `_render_frame_v2` | pipeline.py:1103 | Bidirectional render pass |
| `_update_v3_modules` | pipeline.py:1304 | Kalman + velocity + telemetry |
| `_emit_frame_telemetry` | pipeline.py:1380 | Per-frame JSON log |
| `_composite_identity_to_crop` | pipeline.py:1220 | Identity warp + blend (single resample) |
| `_render_with_physical_renderer` | pipeline.py:1555 | Physical renderer path |
| `photometric_lock` | compositor.py:140 | Temporal luminance EMA |
| `multiband_blend` | compositor.py:68 | Laplacian pyramid blend |
| `_blend_linear` | compositor.py:46 / pipeline.py:95 | Linear-light alpha blend |
| `query_albedo` | identity_state.py:935 | Lighting-invariant identity |
| `predict_with_velocity` | state_evolution.py:290 | SIM(2) velocity extrapolation |
| `render_with_mesh` | physical_renderer.py:370 | Mesh-derived normals render |
| `compare_render_methods` | ab_validation.py:620 | A/B PhysicalRenderer vs alpha |

## Stranded Modules (STRANDED_MODULES.md)

| Module | File | Tests | Phase |
|---|---|---|---|
| IdentityManifold | identity_manifold.py | 26 | Phase C |
| DenseGeometryEstimator | dense_geometry.py | 23 | Phase B |
| OptimizationEngine | optimizer_architecture.py | 32 | Phase C |
| VisibilityCalibrator | visibility_calibration.py | 16 | Phase D |
| ObservabilityAnalyzer | (optimizer_architecture.py) | 28 | Phase C |
| StateSeparator | (types.py) | 34 | Phase C |
| MAPOptimizer | (types.py) | 19 | Phase C |
| RecoveryTransitionMatrix | (types.py) | 38 | Phase C |

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
