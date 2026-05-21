# Face OS Codebase Audit Report

**Date:** 2026-05-21  
**Branch:** `feat/face-os-pipeline`  
**Status:** V4 migration complete | 157 tests passing

---

## Summary

| Severity | Count | Description |
|---|---|---|
| CRITICAL | 3 | Orphaned files, stale config default |
| WARNING | 13 | Dead code, duplicated functions, legacy branches |
| INFO | 18 | Unused imports, hardcoded thresholds, comments |

---

## CRITICAL Issues

### 1. `temporal_stabilize.py` — Orphaned file
- **Status:** No imports anywhere in codebase. Not tested, not referenced.
- **Replacement:** `temporal_solve.py` (bidirectional solver) handles this.
- **Action:** Delete.

### 2. `identity_memory.py` — Orphaned file
- **Status:** No imports anywhere in codebase. Not tested, not referenced.
- **Replacement:** `identity_state.py` (BeliefPixel + IdentityState) handles this.
- **Action:** Delete.

### 3. `config.py:24` — dlib model type
- **Issue:** `"model": "hog"` is a dlib/face_recognition detector type.
- **Fix:** Change to `"model": "mediapipe"`.

---

## WARNING Issues

### Dead Code

| File | Lines | Function | Status |
|---|---|---|---|
| `pipeline.py` | 867-939 | `_apply_anchor_to_frame()` | Never called. Anchor correction now in `identity_state.query()`. |
| `compositor.py` | 94-144 | `composite_with_memory()` | Never called. Pipeline does inline blending. |
| `compositor.py` | 170-204 | `_match_lighting()` | Never called from pipeline. |
| `neural_codec.py` | 308-410 | `IdentityOperatingSystem` | Never instantiated. Wrapper with no pipeline integration. |
| `appearance_field.py` | 363-548 | `DynamicAppearanceField` | Never instantiated. UV deformation unused. |
| `export_qc.py` | 258-281 | `validate_av_sync()` | Duplicate of `ingest.py` version. Never called. |
| `ingest.py` | 169-195 | `validate_av_sync()` | Duplicate of `export_qc.py` version. Never called. |
| `identity_state.py` | 531 | `_anchor_strength = 0.35` | Field defined but never read. |

### Legacy Code (V3 artifacts)

| File | Lines | Issue |
|---|---|---|
| `canonical_map.py` | 149-151 | dlib 68-point anchor branch (dead — always takes `>= 468` path) |
| `canonical_map.py` | 167-173 | dlib 68-point affine branch (dead) |
| `canonical_map.py` | 187-190 | dlib 68-point perspective branch (dead) |

### Silent Exception Swallowing

| File | Line | Issue |
|---|---|---|
| `landmarks.py` | 115-116 | `except Exception: pass` — PnP failures silently ignored, pose stays (0,0,0) |
| `pipeline.py` | 395-396 | `except Exception:` — canonical alignment failure swallowed |
| `pipeline.py` | 580-581 | `except Exception: pass` — canonical alignment failure swallowed |

### Hardcoded Values (should use config)

| File | Line | Value | Should be |
|---|---|---|---|
| `pipeline.py` | 274-275 | `1080`, `1920` | `cfg.crop.output_size` |
| `pipeline.py` | 423-424 | `1080`, `1920` | Same (duplicated) |
| `identity_state.py` | 822-829 | `region_bounds` dict | Reuse `REGION_DEFS` from L156 |

---

## INFO Issues

### Unused Imports

| File | Line | Import |
|---|---|---|
| `pipeline.py` | 36 | `List` from `typing` (use lowercase `list`) |
| `patch_memory.py` | 30 | `defaultdict` from `collections` |
| `temporal_solve.py` | 28 | `deque` from `collections` |

### Hardcoded Thresholds (acceptable, but could be config)

| File | Line | Value | Purpose |
|---|---|---|---|
| `detect_track.py` | 44 | `min_detection_confidence=0.6` | MediaPipe detection threshold |
| `detect_track.py` | 59 | `num_faces=1` | Max faces to detect |
| `face_enhance.py` | 60-61 | `ear_threshold=0.20` | Blink detection threshold |
| `face_enhance.py` | 376 | `blend = 0.3` | Eye preservation blend |
| `crop_planner.py` | 231 | `min_headroom = 0.15` | Minimum headroom ratio |

### Comments Mentioning dlib/Haar (informational only, no code impact)

| File | Line | Context |
|---|---|---|
| `canonical_map.py` | 128 | "V4 ROBUST: Dynamically handles both MediaPipe 478-point and dlib 68-point" |
| `canonical_map.py` | 425 | "Use MediaPipe instead of Haar cascade" |
| `landmarks.py` | 43 | "V4: Replaces dlib 68-point extraction" |
| `detect_track.py` | 5 | "No Haar cascade. No fallback." |

---

## Stale Files

| File | Status | Action |
|---|---|---|
| `tests/check_colab_version.py` | References `ref_grade.py` (legacy). Not a pytest test. | Delete |
| `face_os/ARCHITECTURE_V2.md` | Superseded by `FULL_REFERENCE.md` | Delete |
| `architecture-appearence-field.md` | Root-level, stale | Delete |
| `FIX_PLAN.md` | Root-level, stale | Delete |
| `Colab.md` | Root-level, stale | Delete |
| `output/face_os/debug.qc.json` | Debug artifact | Delete |

---

## Recommendations (Priority Order)

1. ~~Delete orphaned files~~ `temporal_stabilize.py`, `identity_memory.py` → `archive/`
2. ~~Delete stale files~~ `check_colab_version.py`, `ARCHITECTURE_V2.md`, root-level stale `.md` files
3. ~~Fix config default~~ `config.py:24` → `"model": "mediapipe"`
4. ~~Remove dead code~~ `_apply_anchor_to_frame()`, `composite_with_memory()`, `IdentityOperatingSystem`, `DynamicAppearanceField`
5. Remove legacy dlib branches in `canonical_map.py`
6. Fix silent exceptions — add logging to `landmarks.py:115`, `pipeline.py:395,580`
7. Deduplicate `validate_av_sync()` — keep one in `export_qc.py`
8. Extract hardcoded `1080`/`1920` → single source from config

---

## File Status Matrix

| File | Lines | Status | Issues |
|---|---|---|---|
| `pipeline.py` | 1133 | ✅ Working | Dead code, silent exceptions, hardcoded values |
| `detect_track.py` | 537 | ✅ Working | Hardcoded thresholds |
| `identity_state.py` | 892 | ✅ Working | Dead field, duplicated region bounds |
| `face_enhance.py` | 784 | ✅ Working | Hardcoded thresholds |
| `config.py` | 218 | ⚠️ Fix needed | `"model": "hog"` → `"mediapipe"` |
| `types.py` | 238 | ✅ Clean | No issues |
| `canonical_map.py` | 442 | ✅ Working | Legacy dlib branches (dead code) |
| `landmarks.py` | 196 | ✅ Working | Silent PnP exception |
| `crop_planner.py` | 390 | ✅ Working | Minor hardcoded value |
| `compositor.py` | 208 | ⚠️ Mostly dead | `composite_with_memory()` never called |
| `patch_memory.py` | 571 | ✅ Working | Unused import |
| `temporal_solve.py` | 368 | ✅ Working | Unused import |
| `appearance_field.py` | 548 | ⚠️ Partially dead | `DynamicAppearanceField` never used |
| `neural_codec.py` | 410 | ⚠️ Partially dead | `IdentityOperatingSystem` never used |
| `export_qc.py` | 341 | ✅ Working | Duplicate `validate_av_sync` |
| `ingest.py` | 195 | ✅ Working | Duplicate `validate_av_sync` |
| `temporal_stabilize.py` | 163 | ❌ Orphaned | Delete |
| `identity_memory.py` | 244 | ❌ Orphaned | Delete |
| `__init__.py` | 27 | ✅ Clean | No issues |
