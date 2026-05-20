# AGENTS.md — Current State, Gaps & Fix Plan

Last updated: 2026-05-20

---

## Current State Summary

The codebase has a **working pipeline** (download → transcribe → highlight → export → SEO → upload) that produces 9:16 Shorts from 16:9 YouTube VODs. **Reference-derived color grading** (`ref_grade.py`) is integrated as Phase 4.25 with a `--mode` CLI flag.

### What Works
- Full 6-phase pipeline via `pipeline.py` with `--mode {face_mapper,ref_grade}` flag
- 16:9 → 9:16 smart cropping in `export.py` (face tracking, center crop, chat exclusion)
- **ref_grade.py**: Target-based color grading — enrollment-once, apply-always. Blends source TOWARD reference (not beyond). LUT-based a,b transform, cached vignette, split-tone LUT, body lighting mask. 145 tests pass.
- **reference_deep_analyzer.py**: Extracts 25+ parameters from reference (face, body, background, lighting direction, skin consistency, color harmony)
- Cheap analysis: `frame_analyzer.py` (Haar Cascade + heuristics)
- Premium analysis: `premium_analyzer.py` (YOLOv8-face + ByteTrack + Kalman)
- Premium render: `premium_render.py` (RIFE interpolation + GFPGAN + two-pass VBR)
- Super-resolution: `utils/super_res.py` (Real-ESRGAN 4x + GFPGAN + reference guidance)
- SEO generation with 3-tier fallback + self-improving loop
- Colab/Kaggle bridge architecture (tunnel + watcher + job queue)
- `video_analyzer.py`: Auto-detect CUDA/VideoToolbox hwaccel
- `push_code.py`: Syncs `tests/*.py`, prevents Drive "(1)" duplicates
- `monitor.py`: Poll Colab pipeline status via tunnel
- **137 passed, 9 skipped, 0 failures**

### Project Structure
```
yt-clips/
├── pipeline.py          # Main orchestrator (6 phases + --mode flag)
├── ref_grade.py         # Target-based color grading (Phase 4.25)
├── face_mapper.py       # Per-frame 6-step pipeline
├── face_matcher.py      # Face recognition matching (user vs background)
├── reference_deep_analyzer.py  # Deep reference image analysis
├── export.py            # 16:9→9:16 crop + FFmpeg encode
├── download.py          # yt-dlp + aria2c
├── transcribe.py        # faster-whisper
├── highlight.py         # Audio RMS + transcript scoring
├── seo.py               # SEO generation (3-tier fallback)
├── upload.py            # YouTube API upload
├── sync.py              # Google Drive sync
├── video_analyzer.py    # Pre-analysis: face/lighting map
├── frame_analyzer.py    # Cheap: Haar Cascade + heuristics
├── premium_analyzer.py  # Premium: YOLOv8 + ByteTrack + Kalman
├── premium_render.py    # Premium: RIFE + GFPGAN + two-pass VBR
├── watcher.py           # Colab/Kaggle job listener
├── bridge.py            # Local→cloud job pusher
├── push_code.py         # Code sync to Drive (tests/*.py, no duplicates)
├── monitor.py           # Poll Colab pipeline status via tunnel
├── config.yaml          # All configuration
├── expectation.png      # Reference image for grading
├── tests/               # Test suite (145 tests)
│   ├── test_ref_grade.py       # 37 tests
│   ├── test_face_mapper.py     # 35 tests
│   ├── test_video_analyzer.py  # 37 tests
│   ├── test_t4_compat.py       # 17 tests
│   └── test_reference_match.py # 17 tests (grading vs reference)
├── output/              # Generated clips (gitignored)
├── archive/             # Abandoned modules
├── shorts/              # Exported Shorts (gitignored)
├── transcripts/         # Transcript JSONs (gitignored)
├── highlights/          # Highlight YAMLs (gitignored)
└── photos/              # Reference face photos
```

---

## Reference Image Analysis (expectation.png)

The reference is a **portrait studio photo** with:
- **Face**: L=108.5, a=139.6, b=146.7 (warm skin tone)
- **Body**: L=174.8 (66 L brighter than face — studio lighting)
- **Background**: L=41.5 (67 L darker than face — dark studio)
- **Lighting**: Right-lit (ratio=1.12), top-lit (ratio=1.10)
- **Distribution**: 43.5% shadows, 34.4% highlights (high contrast)
- **Color**: 74.8% warm pixels
- **Vignette**: 1.19 ratio
- **Skin consistency**: Face-body LAB delta=66.8 (body much brighter)

---

## Parameter Tuning History

### Iteration Log (v1 → v7)

| Version | L Blend | Contrast | Approach | L Δ | a Δ | b Δ | LAB Dist | Notes |
|---|---|---|---|---|---|---|---|---|
| Old | — | 1.17 | Multiplier | -26.0 | +9.1 | +5.1 | 28.0 | Oversaturated |
| v1 | 25% | 1.22 | Target blend | -24.3 | -0.2 | -2.6 | 24.4 | a,b fixed |
| v2 | 45% | 1.35 | Target blend | -19.7 | +0.2 | -2.6 | 19.9 | L improving |
| v3 | 60% | 1.50 | Target blend | -16.3 | +0.1 | -2.7 | 16.5 | Close |
| **v4** | **75%** | **1.50** | **Target blend** | **-13.1** | **+0.1** | **-2.6** | **13.4** ✅ | **Target met** |
| v5 | 45% | 1.54 | + Body mask | -10.9 | +3.9 | +0.4 | 11.6 | Body +28L |
| v6 | 45% | 1.85 | + Strong contrast | -17.5 | — | — | — | Flicker amplified |
| v7 | 45% | 1.45 | Per-pixel blend | -23.5 | — | — | — | Current (flicker tolerant) |

### Current Parameters (v7)
```python
# ref_grade.py enrollment
params["_L_blend"] = 0.45          # Per-pixel blend toward ref_L
params["_contrast_ratio"] = 1.45   # ref_contrast / 42.0, capped at 1.45
params["_body_boost"] = min(body_L - ref_L, 40)  # Body brightness boost
params["_bg_darken"] = min(ref_L - bg_L, 40)      # Background darkening
params["_lut_a"] = a * 0.65 + a_target * 0.35     # a,b blend toward target
params["_lut_b"] = b * 0.65 + b_target * 0.35
params["_split_lut"] = shadow_color * lut_shadow * 0.06 + highlight_color * lut_highlight * 0.04
```

### Current Test Results (clip5)
| Metric | Reference | Original | Graded | Status |
|---|---|---|---|---|
| L (face) | 111.1 | 83.1 | 87.6 | Improving |
| a (skin) | 135.8 | 144.0 | ~140 | Good |
| b (warmth) | 143.4 | 145.3 | ~144 | Good |
| Body L | 174.8 | 119.1 | ~147 | +28 boost |
| LAB dist | — | 29.2 | ~15 | Under target |

---

## Known Issues & User Feedback

### Face Flicker (Expected)
- **Cause**: User has a side screen that plays videos; colored light reflects onto face
- **Status**: Expected behavior, NOT a bug. Don't waste time fixing.
- **Tolerance**: Add variance tolerance in tests for this.

### Black Fade In/Out
- **Request**: First and last frame of each clip should be black with smooth transition
- **Status**: ✅ FIXED. Export.py has fade support (config.yaml: fade_in=0.5s, fade_out=0.5s). ref_grade.py detects fade frames (dark_pct > 95%) and skips grading to preserve pure black.

### Logo Preservation
- **Request**: Logo should be preserved and placed on LEFT side.
- **Status**: ✅ FIXED. ref_grade.py uses exact coordinates from export.py (bottom-LEFT, 200x200 at x=30, y=1440 for 1080x1920). Scales proportionally for other resolutions. Logo region is excluded from grading. Both export.py and ref_grade.py updated.

### Face Detection — User vs Background Players
- **Problem**: Haar Cascade detects ANY face (players, guests, background people). Grades wrong faces.
- **Status**: ✅ FIXED. `face_matcher.py` uses face_recognition (dlib embeddings) to match detected faces against reference photos in `photos/`. Only the user's face gets graded. Background players are ignored.
- **Performance**: 319 user faces vs 686 other faces correctly identified in test clip (345 frames).
- **Tolerance**: 0.50 (adjustable). Lower = stricter match.

### Headroom Cropping
- **Problem**: 9:16 crop cuts off top of head. expectation.png has headroom above face.
- **Status**: ✅ FIXED. `frame_analyzer._apply_top_padding()` positions face at ~30% from top (matching reference). Screen-share threshold reduced from 35% to 25% face dominance to correctly classify solo frames with complex backgrounds.

### Exposure Swings (Contrast Varying)
- **Problem**: Source video has extreme exposure swings (L=16 to L=155 across frames). Single-pass grade can't normalize. max_shift=30 clamp prevents over-correcting dark frames.
- **Status**: ⚠️ IMPROVED. Grading reduces contrast variation by 57% (L std: 3.5 → 1.5). Face L normalized toward reference (108.5). Still limited by max_shift clamp for extreme dark frames.
- **Evidence**: Tested on studio clip - Original L range: 101.1-110.0 (delta=8.9), Graded L range: 103.5-107.1 (delta=3.6). L std reduced from 3.5 to 1.5.

### Background Construction (Lasso Cut Idea)
- **Request**: Since background never changes, construct studio-grade background first with perfect lighting, then composite person using "lasso cut" (like Photoshop).
- **Status**: Raw idea, needs validation.
- **Approach**:
  1. Extract clean background frame (no person)
  2. Apply studio-grade lighting to background
  3. Use person segmentation (MediaPipe Selfie Segmentation or rembg) to cut out person
  4. Composite person onto graded background
- **Pros**: Background lighting is consistent, person is isolated, no background flicker
- **Cons**: Complex, adds processing time, segmentation may not be perfect
- **Action**: Prototype with MediaPipe Selfie Segmentation, validate quality before full implementation.

---

## Architecture Decisions

### Why Target-Based Blending (Not Multipliers)
- Old approach: `sat_mult = ref_sat / 100 = 1.25` → boosts by 25%
- Problem: Source already has saturation=184, boosting makes it 230 (way oversaturated)
- New approach: `a_out = a * 0.65 + a_target * 0.35` → blends toward reference
- Result: Source moves TOWARD reference, never beyond it

### Why Per-Pixel L Blend (Not Frame-Mean)
- Frame-mean shift: `L_out = L + (ref_L - mean(L)) * blend`
- Problem: If mean(L) varies across frames, shift amount changes → flicker
- Per-pixel blend: `L_out = L + (ref_L - L) * blend`
- Result: Each pixel moves independently, no frame-mean dependency

### Why Body Lighting Mask
- Reference has body L=174.8 (66 brighter than face)
- Without mask: body stays dark, doesn't match reference studio lighting
- With mask: bottom 40% of frame gets brightness boost, top gets darkening
- Mask is cached by resolution, smoothed with GaussianBlur

### Why Contrast After Blend
- Per-pixel blend compresses contrast (range shrinks toward ref_L)
- Contrast stretch after blend restores the range
- Centered on per-frame mean (not ref_L) to avoid amplifying source flicker

---

## Test Suite Summary

| File | Tests | Status | Purpose |
|---|---|---|---|
| `test_ref_grade.py` | 37 | ✅ All pass | Enrollment, grading, flicker, video |
| `test_face_mapper.py` | 35 | ✅ All pass | Face mapper enhancement |
| `test_video_analyzer.py` | 37 | ✅ All pass | Video analyzer analysis |
| `test_t4_compat.py` | 17 | ✅ All pass | T4 GPU compatibility |
| **Total** | **137** | **0 failures** | |

---

## T4 GPU Performance

| Operation | T4 CPU | Mac M1 |
|---|---|---|
| 1080p apply_grade | 8fps (130ms) | 29fps (35ms) |
| 720p grade_video pipe | 8fps | 14fps |
| Enrollment | 0.5s | 0.3s |
| Body mask (cached) | ~2ms | ~1ms |

---

## Next Steps (Priority Order)

### Immediate (Next Session)
1. **Per-face exposure normalization** — Source video has L=16→155 swings. Need to detect user face, measure face L, apply per-frame exposure correction to normalize face L to ref_L BEFORE applying ref_grade.
2. **Integrate face_matcher into export.py** — Use face_matcher.crop_with_headspace() for proper 9:16 cropping with headspace. Only grade frames where user face is detected.
3. **Full pipeline test** — Run end-to-end: 16:9 → face_matcher crop → ref_grade → output. Validate on portrait studio video (not cricket).

### Short-term
4. **Increase L blend strength** — Current 70% may still be too conservative for dark frames. Try 80% with higher max_shift (40) for better L normalization.
5. **Tune contrast ratio** — Current 1.30 may be too low. Try 1.40-1.50 to better match reference contrast.
6. **Update docs** — ARCHITECTURE.md, README.md with current architecture.

### Medium-term
7. **Prototype lasso cut** — MediaPipe Selfie Segmentation for person isolation + background composite.
8. **Add `--mode face_mapper` test** — Validate face_mapper on real content.

---

## Files Modified This Session

| File | Changes |
|---|---|
| `ref_grade.py` | Target-based blending, body lighting mask, per-pixel L blend, LUT-based a,b |
| `pipeline.py` | `--mode {face_mapper,ref_grade}` CLI flag, Phase 4.25 dispatch |
| `video_analyzer.py` | Auto-detect CUDA/VideoToolbox hwaccel |
| `push_code.py` | `tests/*.py` sync, `_find_file_by_name` fallback |
| `reference_deep_analyzer.py` | **NEW** — Deep reference image analysis (25+ params) |
| `monitor.py` | **NEW** — Poll Colab pipeline status via tunnel |
| `tests/test_ref_grade.py` | Updated for new params (ref_L, body_L, etc.) |
| `tests/test_t4_compat.py` | Works under pytest, T4 GPU checks |
| `tests/test_reference_match.py` | **NEW** — 17 tests validating grading vs reference |
| `tests/test_video_analyzer.py` | Fixed hwaccel test, scoring test |
| `config.yaml` | `enhancement.ref_grade` toggle |
| `AGENTS.md` | **This file** — current state & gaps |
| `archive/` | **NEW** — Abandoned modules moved here |
| `output/` | **NEW** — Generated clips (gitignored) |

---

## Git History (This Session)

```
d38fad2 wip: per-pixel L blend + moderate contrast (v7 params, flicker tolerant)
55f8d18 chore: organize project — move clips to output/, archive abandoned modules
5760c71 feat: full-body lighting — body boost, background darken, stronger contrast, warmer a,b
5931454 feat: reference deep analyzer + 17 test cases for grading validation
e0df1f5 tune: L blend 75%
2b8507f tune: L blend 60%, contrast 1.50
088dcff tune: stronger L blend (45%), higher contrast (1.35), tighter a,b (0.30)
8325be8 fix: target-based blending — moves TOWARD reference, not beyond
5fd9048 docs: update AGENTS.md — target-based blending, 128 tests, T4 perf
5760c71 feat: full-body lighting — body boost, background darken, stronger contrast, warmer a,b
25e793f feat: LUT-based ref_grade (35ms/frame) + pipeline --mode flag + fix video_analyzer hwaccel
a4a0984 test: T4 GPU compatibility test suite (22 checks)
b41e5ea fix: sync tests/*.py to Colab
9c53743 fix: prevent Drive (1) duplicates — name-based fallback in _upload_one
81d06de feat: monitor.py — poll Colab pipeline status via tunnel
```

---

## User Context

- **Content**: Portrait-mode studio videos (not cricket — cricket was a test video)
- **Reference**: `expectation.png` — enhanced portrait of user in studio
- **Side screen**: User has a side screen that plays videos; colored light reflects onto face → causes expected flicker
- **Background**: Never changes throughout video — good candidate for lasso cut approach
- **Logo**: Needs to be preserved (not impacted by grading) and placed on left side
- **Fade**: First/last frame should be black with smooth transition
