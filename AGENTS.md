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

## Self-Learner Module (`self_learner/`)

**SQLite-backed persistent-memory learning engine.** Zero FaceOS deps.

```bash
.venv/bin/python -m pytest tests/test_self_learner.py -v --cov=self_learner --cov-report=term-missing
```

### Key APIs

| Class | Module | Role |
|-------|--------|------|
| `PersistentMemory` | `self_learner/memory.py` | Thread-safe SQLite KV store with TTL, confidence, source |
| `KnowledgeBase` | `self_learner/knowledge.py` | Fact storage with relation index |
| `Learner` | `self_learner/learner.py` | Observe events, extract patterns, predict |
| `Runner` | `self_learner/runner.py` | CLI: `observe`, `predict`, `insights`, `stats`, `daemon` |

### Integration

Wired into `automation/orchestrator.py` Stage 9 — observes pipeline metrics after every run with `auto_upload` or `auto_schedule`.```
