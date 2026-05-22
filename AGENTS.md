# AGENTS.md — Source of Truth

Last updated: 2026-05-22 (Face OS v3.2 — Phase B Complete)

---

## Quick Reference

- **Compact state:** `face_os/STATE.md` — architecture map, drift status, entry points, test suite
- **Memory graph:** `graphify-out/memory_graph.json` — 59 files, 14,202 src lines, 42 deps
- **Architecture lock:** `LOCKED_ARCHITECTURE.md` (444 read-only)
- **Stranded modules:** `face_os/STRANDED_MODULES.md` — 8 modules, integration phases

---

## Current State: 603 tests, 0 failures

**10/14 drift items ALIGNED, 3 PARTIAL, 1 NOT ALIGNED (D-07 — Phase C rewrite)**

---

## How to Use This File

1. **For architecture questions** → read `LOCKED_ARCHITECTURE.md` or `face_os/STATE.md`
2. **For file structure** → query `graphify-out/memory_graph.json`
3. **For function locations** → see `face_os/STATE.md` Key Entry Points table
4. **For test commands** → see `face_os/STATE.md` Run Commands section
5. **For stranded module status** → see `face_os/STRANDED_MODULES.md`

**Do NOT re-read pipeline.py (1992 lines) unless modifying it. Use STATE.md for line references.**

---

## V3.1 Rules (Enforced)

1. `_render_core()` — single render path, no inline rendering outside it
2. Green tests must imply runtime correctness
3. Telemetry mandatory for all subsystems
4. Metrics must match optimization targets
5. `_update_v3_modules()` — single V3 module update source
6. `predict_with_velocity()` — SIM(2) velocity prediction active
7. `_compute_energy_terms()` — z-score normalized
8. `_emit_frame_telemetry()` — per-frame JSON in ALL 3 return paths

---

## User Context

- Portrait-mode studio videos, side screen with colored light reflections
- Test video: `clips_test/test_clip.mp4` (640x360, 30fps, 15s)
- Reference: `expectation.png`
- Background: never changes (lasso cut candidate)
- Logo: preserved on left side
- Fade: first/last frame black (export.py)
