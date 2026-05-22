# Stranded Modules — D-10 / I-10

Last updated: 2026-05-22

| Module | File | Tests | Decision | Integration Phase | Reason |
|---|---|---|---|---|---|
| IdentityManifold | identity_manifold.py | 26 | SCHEDULED | Phase C | Riemannian identity space — needed when replacing discrete anchors |
| DenseGeometryEstimator | dense_geometry.py | 23 | SCHEDULED | Phase B | Dense mesh from landmarks — needed for true geometry normals |
| OptimizationEngine | optimizer_architecture.py | 32 | SCHEDULED | Phase C | Iterative optimizer — needed for factor-graph inference |
| VisibilityCalibrator | visibility_calibration.py | 16 | SCHEDULED | Phase D | Metric calibration — needed for hard-scene validation |
| SE2Transform | lie_group.py | (shared) | STRANDED | Unknown | SE(2) unused at runtime; only SIM(2) is active |
| ObservabilityAnalyzer | (in optimizer_architecture.py) | 28 | SCHEDULED | Phase C | Degeneracy analysis for factor-graph |
| StateSeparator | (test_state_separation.py) | 34 | SCHEDULED | Phase C | Physical/Belief/Meta state decomposition |
| MAPOptimizer | (test_map_estimation.py) | 19 | SCHEDULED | Phase C | MAP optimization for inference graph |
| RecoveryTransitionMatrix | (test_recovery_dynamics.py) | 38 | SCHEDULED | Phase C | Bayesian recovery transitions |

## Rules

1. Every module must satisfy ONE: Active, Scheduled, Experimental, or Deleted
2. Scheduled modules keep code + tests but are NOT modified until integration phase
3. Tests for stranded modules should still pass (they test internal correctness)
4. No new features should be added to stranded modules without integration plan

## Integration Order

Phase B (Geometry Realism):
  1. DenseGeometryEstimator → wire into GeometryEstimator subsystem

Phase C (Probabilistic Runtime):
  1. IdentityManifold → replace discrete anchors
  2. OptimizationEngine → factor-graph solver
  3. ObservabilityAnalyzer → degeneracy detection
  4. StateSeparator → decompose latent state
  5. MAPOptimizer → MAP inference
  6. RecoveryTransitionMatrix → recovery dynamics

Phase D (Hard Reality Validation):
  1. VisibilityCalibrator → metric calibration for adversarial clips
