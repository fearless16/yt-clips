# AGENTS.md — Source of Truth

---

## Testing (MANDATORY)

**Every code change MUST be validated before commit:**

```bash
# 1. Run unit tests
.venv/bin/python -m pytest tests/ -v --ignore=tests/test_face_detect.py

# 2. Run pipeline dry run (stubs all external APIs)
.venv/bin/python dry_run.py https://youtu.be/test --skip-download --skip-transcribe
```

If either fails, DO NOT commit. Fix first.

---

## Face OS (`face_os/`)

**Full documentation → `face_os/STATE.md`** (single source of truth).

Quick links:
- [Architecture map](face_os/STATE.md#architecture-map)
- [Drift status](face_os/STATE.md#drift-status-locked_architecturemd)
- [Key entry points](face_os/STATE.md#key-entry-points)
- [Test suite](face_os/STATE.md#test-suite)

```bash
.venv/bin/python -m pytest face_os/tests/ -v
```

**Last updated:** 2026-06-02 | **Tests:** 441 passed, 0 failed, 3 skipped | **Source:** ~17,300 lines | **Version:** v3.9

### Architecture Summary

- **D-01 through D-10:** 3 aligned mechanisms, 7 partial architecture/validation areas
- **Subsystem wrappers:** 4 real runtime delegates (IdentityEstimator, TemporalEstimator, FaceRenderer, GeometryEstimator)
- **A/B validation:** ABComparator wired to real pipeline API via `process_frame()` + `render_mode_override`
- **Telemetry:** Per-frame explicit JSON in all paths including LOST_FACE; branch truth is no longer inferred from counters
- **Compositing:** Linear-light + default Laplacian pyramid multiband blend

---

## Automation Module (`automation/`)

**Full documentation → `automation/AGENTS.md`** (single source of truth).

```bash
.venv/bin/python -m pytest tests/test_automation.py -v
```

---

## Shorts Intelligence (`shorts_intelligence/`)

Canonical cricket Shorts outcome learner. It accepts only this channel's exact
Shorts shelf, joins real YouTube Analytics outcomes, and uses calibrated,
recency-aware evidence. Generated metadata is never treated as an outcome.

```bash
.venv/bin/python -m pytest tests/shorts_intelligence/ -v
.venv/bin/python -m shorts_intelligence sync
.venv/bin/python -m shorts_intelligence status
```

`shorts_intelligence.db` is the only persistent learning database. Selection
adjustments are bounded and cannot alter clip boundaries or bypass the
complete-thought gate.
