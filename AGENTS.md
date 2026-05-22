# AGENTS.md — Source of Truth

---

## Face OS (`face_os/`)

**Full documentation → `face_os/STATE.md`** (single source of truth).

Quick links:
- [Architecture map](face_os/STATE.md#architecture-map)
- [Drift status](face_os/STATE.md#drift-status-locked_architecturemd)
- [Key entry points](face_os/STATE.md#key-entry-points)
- [Test suite](face_os/STATE.md#test-suite)

```bash
.venv/bin/python -m pytest tests/face_os/ -v
```

**Last updated:** 2026-05-22 | **Tests:** 140/140 | **Source:** 14,757 lines | **Version:** v3.7

### Architecture Summary

- **D-01 through D-10:** 11 ALIGNED, 1 PARTIAL (D-03), 1 NOT NEEDED (D-07)
- **Subsystem wrappers:** 4 real runtime delegates (IdentityEstimator, TemporalEstimator, FaceRenderer, GeometryEstimator)
- **A/B validation:** ABComparator wired to real pipeline API via `process_frame()` + `render_mode_override`
- **Telemetry:** Per-frame JSON in ALL paths including LOST_FACE
- **Compositing:** Linear-light + Laplacian pyramid multiband blend

---

## Automation Module (`automation/`)

**Full documentation → `automation/AGENTS.md`** (single source of truth).

```bash
.venv/bin/python -m pytest tests/test_automation.py -v
```
