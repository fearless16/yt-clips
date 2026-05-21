# Face OS V3.1 — Missing Items & Next Actions

**Last Updated:** 2026-05-21  
**Status:** Runtime-active, telemetry-backed, P0+P1 complete, 893 tests passing

---

## Current Honest Status

V3.1 is now:
- runtime-active (96% PhysicalRenderer, 100% IntrinsicDecomposer)
- telemetry-backed (14/14 keys, timing, fallback reasons)
- mathematically coherent (SIM(2) validated, energy normalization default-on, mesh normals)
- structurally consolidated (_render_core, no duplicate logic)
- benchmark-validated (synthetic generators for all 4 categories)
- A/B framework proven (SIM(2) vs EMA: lower jitter, better det stability)
- identity decoupled (albedo-based anchor correction, white-balance normalization)
- occlusion-validated (13 tests: dropped frames, rapid motion, 1000-frame drift)

**What's proven:** modules run AND improve geometric stability  
**What's NOT proven:** modules improve perceptual quality on real hard clips

---

## P2 — Decision Items (Remaining)

### I-08: Stranded Module Final Decisions
**Status:** AUDITED — decisions made

| Module | Lines | Tests | Decision | Reasoning |
|---|---|---|---|---|
| IdentityManifold | 399 | 26 | **SCHEDULE** | For I-05 full decoupling. Manifold is flat (no curvature). Wire when replacing discrete anchors. |
| VisibilityCalibration | 286 | 16 | **SCHEDULE** | QA tool for benchmark suite. Wire when validating A/B metric-truth correlation. |
| OptimizationEngine | 340 | 25 | **SCHEDULE** | Wrapper for optimizer.py. Wire when MAP estimation is integrated. |
| DenseGeometry | 534 | 23 | **SCHEDULE** | For I-04 mesh normals. De-scoped for V3. Wire when PhysicalRenderer needs dense normals. |
| StateSpace | 537 | 39 | **INTEGRATE** | Foundation for StateSeparation, Observability, MAPEstimation. Replaces simpler state_evolution.py. |
| StateSeparation | 242 | 34 | **SCHEDULE** | Depends on StateSpace. Clean Physical/Belief/Meta decomposition. |
| Observability | 231 | 28 | **SCHEDULE** | Depends on StateSpace. Diagnostic tool for degeneracy analysis. |
| MAPEstimation | 282 | 19 | **SCHEDULE** | Depends on StateSpace + optimizer.py. Bayesian frame-local inference. |
| RecoveryDynamics | 354 | 38 | **SCHEDULE** | No dependencies. Probabilistic recovery > discrete state machine. Integrate when ready. |

**Dependency chain:** StateSpace → {StateSeparation, Observability, MAPEstimation} → optimizer.py

**Rule:** Dead with NO plan is not allowed. All modules are SCHEDULED or marked INTEGRATE.

---

## What SHOULD NOT Happen

DO NOT:
- Add more theoretical modules
- Add neural buzzword systems
- Add unvalidated optimizers
- Bypass telemetry
- Weaken tests

The architecture is deep enough. Next phase is **reality validation**, not theory expansion.

---

## Success Criteria

V3.1 is production-ready when:
1. ~~SIM(2) proves better than linear EMA on geometric consistency~~ ✅ (11 tests)
2. ~~Identity survives lighting changes without drift~~ ✅ (11 tests, albedo decoupling)
3. ~~StateEvolution recovers from occlusion within 5 frames~~ ✅ (13 tests)
4. ~~All stranded modules have clear fate~~ ✅ (9 modules audited)
5. Real-world A/B runs on hard clips prove perceptual quality improvement
