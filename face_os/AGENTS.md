# AGENTS.md — Face OS Codebase Analysis Agent

## Core Directive

You are a codebase analysis and execution agent for the Face OS pipeline (`face_os/`). Your priority is to minimize context usage, avoid unnecessary file reads, and work efficiently with parallel agents.

## Rules

1. **Do not open or read files blindly.** First build a lightweight mental/codebase graph from directory structure, entry points, imports, symbols, and obvious dependencies.
2. **Use that graph to decide which files actually matter before reading anything.**
3. **Prefer narrow, targeted reads over broad exploration.**
4. **Split work across parallel agents** whenever tasks can be independent. Assign each agent a small, clear scope.
5. **Keep context compact.** Summarize findings into short working notes and discard noise.
6. **Use a cache for intermediate thinking**, decisions, and discovered facts so repeated reasoning is not redone.
7. **Reuse cached conclusions before rereading files or recomputing.**
8. **Preserve and reuse thinking state aggressively** to reduce token usage.
9. **Always optimize for the smallest possible context window while still being correct.**
10. **Before any deep dive, ask: "Can I infer this from structure, symbols, or cached knowledge first?"**

## Operating Style

1. **First map the codebase.** Entry points → imports → symbols → dependencies.
2. **Identify the minimum set of files needed** for the task.
3. **Parallelize independent investigations** across sub-agents.
4. **Merge results** into a concise final answer or action plan.
5. **Never expand context unless it clearly improves correctness.**

## Face OS Architecture Map

| Layer | Files | Role |
|-------|-------|------|
| Orchestration | `pipeline.py` | Main pipeline, telemetry, visibility logging |
| Tracking | `detect_track.py`, `landmarks.py` | Face detection, tracking, landmark extraction |
| Canonical | `canonical_map.py` | Face space alignment |
| Identity | `identity_state.py`, `identity_manifold.py`, `patch_memory.py` | Identity belief state |
| Rendering | `physical_renderer.py`, `renderer_mode.py` | Face rendering engine |
| Intrinsic | `intrinsic_decomposition.py` | Albedo/shading/specular decomposition |
| Geometry | `dense_geometry.py` | Mesh estimation |
| Compositing | `compositor.py` | Frequency-aware blending |
| Subsystems | `subsystems/` | Thin wrappers: IdentityEstimator, TemporalEstimator, GeometryEstimator, FaceRenderer |
| Types | `types.py` | All data structures and contracts |

## Key Entry Points

- `pipeline.py:FaceOSPipeline.process()` — main processing entry
- `pipeline.py:FaceOSPipeline.enroll()` — identity enrollment
- `pipeline.py:FaceOSPipeline._render_core()` — single shared rendering core
- `pipeline.py:FaceOSPipeline._emit_frame_telemetry()` — per-frame JSON telemetry
- `intrinsic_decomposition.py:IntrinsicDecomposer.decompose()` — produces `IntrinsicComponents` with `detail_residual`

## Telemetry & Logging

- `_telemetry` dict — runtime counters
- `_frame_telemetry_log` — per-frame JSON list
- `_start_visibility_run()` — per-clip JSONL + summary logging to `output/face_os/visibility/`
- `_log_event()` — human log + JSONL log
- `get_telemetry_report()` — aggregated report with rates and averages
- `_inject_detail_residual()` — re-injects HF detail from intrinsic decomposition

## Test Suite

```bash
.venv/bin/python -m pytest tests/face_os/ -v
```

## Goal

Deliver accurate results with the fewest tokens, minimal file reads, and maximum parallel efficiency.
