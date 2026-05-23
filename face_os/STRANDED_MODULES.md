# Stranded Modules — D-10 / I-10

Last updated: 2026-05-23

| Module | File | Tests | Decision | Reason |
|---|---|---|---|---|
| DenseGeometryEstimator | dense_geometry.py | 23 | **ACTIVE** | Wired into physical path via `DenseGeometryEstimator.estimate()` → `FaceRenderer.render_with_mesh()` |
| IdentityManifold | identity_manifold.py | 26 | STRANDED | Riemannian identity space — not needed for current anchor-based system |
| OptimizationEngine | optimizer_architecture.py | 32 | STRANDED / PHASE C | Factor-graph solver remains future work; runtime is still procedural orchestration plus Kalman/SIM(2) |
| VisibilityCalibrator | visibility_calibration.py | 16 | STRANDED | Metric calibration — not needed until hard-scene validation phase |
| SE2Transform | lie_group.py | (shared) | STRANDED | SE(2) unused at runtime; only SIM(2) is active |
| ObservabilityAnalyzer | (in optimizer_architecture.py) | 28 | DELETED | Degeneracy analysis for factor-graph — Phase C NOT NEEDED |
| StateSeparator | (test_state_separation.py) | 34 | DELETED | Physical/Belief/Meta state decomposition — Phase C NOT NEEDED |
| MAPOptimizer | (test_map_estimation.py) | 19 | DELETED | MAP optimization for inference graph — Phase C NOT NEEDED |
| RecoveryTransitionMatrix | (test_recovery_dynamics.py) | 38 | STRANDED | Bayesian recovery transitions — keep for future occlusion recovery |

## Rules

1. Every module must satisfy ONE: Active, Scheduled, Experimental, or Deleted
2. Active modules have runtime call paths in pipeline.py
3. Tests for stranded modules should still pass (they test internal correctness)
4. No new features should be added to stranded modules without integration plan

## Architecture Honesty

D-07 (factor-graph inference) is **NOT NEEDED** for the current architecture:
- Current system uses Kalman filter (state_evolution.py) + SIM(2) velocity prediction
- This is sufficient for temporal consistency and occlusion recovery
- Factor-graph inference would require rewriting the entire runtime — not worth the cost
- D-07 is marked as **NOT ALIGNED — NOT NEEDED**

## Integration Order

No further integration phases planned. Current architecture is complete for v3.x.
