# Face OS — Complete Architecture & Parameter Reference (V3)

**Version:** 0.3.0  
**Branch:** `feat/face-os-pipeline`  
**Date:** 2026-05-21  
**Status:** Quality gates active | Compositor fixed | LAB distance 20.3 (target <5)

---

## Table of Contents

1. [What Changed From V2](#1-what-changed-from-v2)
2. [Architecture Overview](#2-architecture-overview)
3. [Module-by-Module Deep Dive](#3-module-by-module-deep-dive)
4. [Quality Gates](#4-quality-gates)
5. [Verification Gate](#5-verification-gate)
6. [Configuration Reference](#6-configuration-reference)
7. [Test Results & Metrics](#7-test-results--metrics)
8. [Known Issues & Next Steps](#8-known-issues--next-steps)

---

## 1. What Changed From V2

### V2 → V3 Changes

| Component | V2 | V3 | Why Changed |
|---|---|---|---|
| **Face Detection** | Haar Cascade | MediaPipe FaceDetection | Haar detects posters/backgrounds. MediaPipe has real confidence scores. |
| **Identity Matching** | face_recognition only | face_recognition + VerificationGate | Need to reject wrong identity, tiny faces, static posters before updating identity memory |
| **Compositor Input** | `rendered` (face_enhance output) | `identity_face` (from identity_state.query()) | **CRITICAL BUG FIX** — identity correction was being discarded |
| **Quality Gates** | None | Procrustes, Jitter, Occupancy | Prevent poster lock, tiny face lock, wrong identity lock |
| **Anchor Lambda** | 0.65-0.95 | 0.60-0.75 | Too aggressive anchor was over-correcting |
| **Blend Formula** | Fixed low=0.75, high=0.25 | Dynamic: low=0.7+0.15*conf, high=0.3-0.15*conf | Confidence-weighted blending |

### Critical Fix: Compositor Now Uses Identity Face

**Location:** `pipeline.py:630`

```python
# V2 (BROKEN):
output = self.compositor.composite(cropped, rendered, ...)
# rendered = face_enhance output (NO brightness correction)
# identity_face was computed but DISCARDED

# V3 (FIXED):
output = self.compositor.composite(cropped, identity_face, ...)
# identity_face = anchor-corrected identity from identity_state.query()
# Now identity correction is actually applied to output
```

---

## 2. Architecture Overview

```
INPUT: 16:9 source video + reference face images (expectation.png + photos/)
                    │
    ┌───────────────┼───────────────┐
    │               │               │
    ▼               ▼               ▼
┌─────────┐  ┌───────────┐  ┌──────────────┐
│ Module 1│  │ Module 2  │  │  Module 3    │
│ Ingest  │  │ Detect +  │  │  Landmarks + │
│ + Sync  │  │ Track     │  │  Pose        │
└────┬────┘  └─────┬─────┘  └──────┬───────┘
     │             │               │
     └──────┬──────┴───────────────┘
            │
            ▼
    ┌───────────────┐
    │   Module 4    │ ◄── Canonical UV alignment
    │   Canonical   │     Appearance Field A(u,v,θ,L,t)
    │   Face Map    │     Pose-conditioned patch database
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │   Module 5    │ ◄── Reference-based composition
    │   Crop Plan   │     Matches expectation.png layout
    │   + Headroom  │     Preserves source headroom (never reduces)
    └───────┬───────┘
            │
    ┌───────────────┐
    │   Module 6    │ ◄── Bidirectional temporal solver
    │   Temporal    │     Forward pass: collect confidence
    │   Solve       │     Backward pass: HQ frames repair past
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │   Module 7    │ ◄── Structure-preserving rendering
    │   Face        │     Eyes: enhance definition (NOT hallucinate)
    │   Enhance     │     Beard: preserve texture
    │               │     Skin: gentle smoothing + cinematic noise
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │   Module 8    │ ◄── Frequency decomposition
    │   Identity    │     Low freq (skin tone): EMA
    │   State       │     High freq (pores): BEST observation only
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │   Module 9    │ ◄── Confidence-weighted compositing
    │   Compositor  │     High conf = identity memory
    │               │     Low conf = source pixels
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │   Pipeline    │ ◄── 3-pass orchestrator
    │   Orchestrator│     Pass 1: Forward collection
    │               │     Pass 2: Bidirectional solve
    │               │     Pass 3: Rendering
    └───────────────┘
            │
            ▼
OUTPUT: 9:16 stabilized video (1080x1920)
```

---

## 3. Module-by-Module Deep Dive

### Module 1: `ingest.py` — Video Ingestion

**What it does:**
- Loads video file and extracts metadata (dimensions, fps, codec, duration)
- Provides frame-by-frame generator with seeking support
- Loads reference face images for identity enrollment
- Validates audio-video sync

---

### Module 2: `detect_track.py` — Face Detection + Tracking + Quality Gates

**What it does:**
- Detects faces using **MediaPipe FaceDetection** (model_selection=1, min_conf=0.6)
- Matches detected faces to target identity via embeddings
- Maintains persistent face tracks across frames
- Smooths bounding boxes with EMA
- **Runs quality gates before returning track**

**Detection strategy:**
```
Frame 0:  DETECT → find faces → match identity → create track
Frame 1-4: TRACK → predict position (use last known bbox)
Frame 5:  DETECT → find faces → match identity → update track
```

**Quality Gates (all must pass):**
```python
# Gate 1: Procrustes disparity < 0.09
disparity = compute_procrustes_disparity(mesh, reference_mesh)
if disparity > 0.09:
    return None  # Reject

# Gate 2: Landmark jitter > 0.0008 (real face, not poster)
jitter = compute_landmark_jitter(landmark_history)
if jitter < 0.0008:
    return None  # Reject (static poster)

# Gate 3: Occupancy > 0.25
occupancy = face_area / bbox_area
if occupancy < 0.25:
    return None  # Reject (face too small in bbox)
```

**Face Mesh extraction:**
```python
# Uses dlib 68-point landmarks (fallback to None if unavailable)
mesh = extract_face_mesh(frame)  # Returns (68, 2) array
```

---

### Module 3: `landmarks.py` — Landmarks + Head Pose

**What it does:**
- Extracts 68-point facial landmarks (dlib or geometric fallback)
- Estimates head pose (yaw, pitch, roll) using PnP algorithm
- Creates per-region masks (eyes, brows, nose, mouth, skin, face contour)

**Face mask generation (FIXED):**
```python
# Uses ALL 68 landmarks + forehead extension
all_face_pts = pts[0:68]
hull = cv2.convexHull(all_face_pts)

# Extend upward to include forehead
brow_top = int(np.min(pts[17:26, 1]))
jaw_top = int(np.min(pts[0:17, 1]))
forehead_height = jaw_top - brow_top
forehead_top = max(0, brow_top - forehead_height)
```

---

### Module 4: `canonical_map.py` — Canonical Face Mapping

**What it does:**
- Aligns detected face to canonical UV space (frontal, neutral pose)
- Builds Appearance Field A(u,v,θ,L,t)
- Accumulates pixel observations over time (Photic Memory)
- **Extracts embeddings from face region (not full image)**

**Embedding extraction (FIXED):**
```python
# V2 (BROKEN): extracted from full image
encodings = face_recognition.face_encodings(rgb)

# V3 (FIXED): extracts from face region only
detections = detect_faces(img)
x, y, w, h, conf = detections[0]
locations = [(y, x + w, y + h, x)]
encodings = face_recognition.face_encodings(rgb, locations)
```

---

### Module 5: `crop_planner.py` — Reference-Based Crop Planning

**What it does:**
- Analyzes reference image (expectation.png) at startup for composition targets
- Plans 16:9 → 9:16 crop that matches reference composition
- Preserves source headroom (never reduces it)
- Smooths crop transitions with EMA

---

### Module 6: `temporal_solve.py` — Bidirectional Temporal Solver

**What it does:**
- **Forward pass (Pass 1):** Collects per-frame quality metrics, identifies HQ frames
- **Backward pass (Pass 2):** HQ frames repair past blurry frames
- This is the offline pipeline's superpower — future frames can fix the past

---

### Module 7: `face_enhance.py` — Structure-Preserving Rendering

**What it does:**
- Enhances face regions while PRESERVING source structure
- Does NOT hallucinate details (eyelashes, pores, etc.)
- Adds cinematic noise for realism

---

### Module 8: `identity_state.py` — Frequency Decomposition + Verification Gate

**What it does:**
- Decomposes identity into LOW frequency (skin tone, lighting) and HIGH frequency (pores, edges)
- LOW freq: EMA over time (smooth, stable)
- HIGH freq: BEST observation only (never average — averaging pores = blur)
- **Verification Gate rejects bad observations before updating**

**Frequency decomposition:**
```python
# Low frequency: Gaussian blur (removes fine detail)
low_freq = cv2.GaussianBlur(canonical_face, (ksize, ksize), sigma)

# High frequency: residual (fine detail only)
high_freq = canonical_face - low_freq
```

**Dynamic blending (V3):**
```python
# V2: Fixed blend
low_blend = 0.75
high_blend = 0.25

# V3: Confidence-weighted blend
mean_conf = np.mean(confidence)
low_blend = 0.7 + 0.15 * mean_conf   # 0.7-0.85
high_blend = 0.3 - 0.15 * mean_conf  # 0.15-0.3
```

**Anchor correction (V3 — reduced lambda):**
```python
# V2: lambda up to 0.95 (too aggressive)
lambda_clamped = np.clip(lambda_conf, 0.65, 0.95)

# V3: lambda max 0.75 (gentler correction)
lambda_clamped = np.clip(lambda_conf, 0.60, 0.75)
```

---

### Module 9: `compositor.py` — Confidence-Weighted Compositing

**What it does:**
- Composites **identity_face** onto original frame using per-pixel confidence
- High confidence pixels → use identity memory (stable, clean)
- Low confidence pixels → use original frame (noisy but authentic)
- Feathered edge blending prevents visible seams

**V3 compositing (uses identity_face, not rendered):**
```python
# Blend weight = face_mask * confidence
blend_weight = feathered * confidence

# Composite: identity_face (anchor-corrected) blended with source
result = original * (1 - blend_weight) + identity_face * blend_weight
```

---

### Pipeline Orchestrator: `pipeline.py` — 3-Pass Architecture

**What it does:**
- Coordinates all modules in a 3-pass pipeline
- Pass 1 (Forward): Collect telemetry, build identity state
- Pass 2 (Backward): Bidirectional temporal solve, HQ frame repair
- Pass 3 (Render): Structure-preserving enhancement + compositing

**Face Lock State Machine:**
```
FACE_LOCKED: face detected, occupancy > 0.25, conf > 0.5
LOST_FACE:  no detection → skip identity update, return source
RECOVERY:   face returns → normal processing
```

**Identity update with verification (V3):**
```python
# Get verification parameters
face_bbox = face_track.smooth_bbox
landmarks_pts = np.array(landmarks.xy)
embedding = face_track.detection.embedding

# Update with verification gate
self.identity_state.update(
    canonical_face, masked_quality, pose=pose,
    face_bbox=face_bbox,
    landmarks_pts=landmarks_pts,
    embedding=embedding,
)
```

---

## 4. Quality Gates

### Overview

Quality gates prevent bad observations from corrupting identity memory.

| Gate | Threshold | Purpose |
|---|---|---|
| Procrustes disparity | < 0.09 | Face shape must match reference |
| Landmark jitter | > 0.0008 | Real face moves (poster is static) |
| Occupancy | > 0.25 | Face must fill enough of bbox |

### Procrustes Disparity

Measures shape difference between current face landmarks and reference.

```python
def compute_procrustes_disparity(landmarks, reference_landmarks):
    # Center and normalize both
    lm = landmarks - landmarks.mean(axis=0)
    ref = reference_landmarks - reference.mean(axis=0)
    
    # Scale to unit Frobenius norm
    lm_norm = lm / sqrt(sum(lm^2))
    ref_norm = ref / sqrt(sum(ref^2))
    
    # Disparity = Frobenius norm of difference
    disparity = sqrt(sum((lm_norm - ref_norm)^2))
    return disparity
```

### Landmark Jitter

Measures temporal movement of landmarks across frames.

```python
def compute_landmark_jitter(landmark_history):
    displacements = []
    for i in range(1, len(history)):
        disp = mean(sqrt(sum((curr - prev)^2, axis=1)))
        disp_norm = disp / 200.0  # Normalize by face size
        displacements.append(disp_norm)
    return mean(displacements)
```

### Occupancy

Measures how much of the bounding box is filled by the face.

```python
def compute_occupancy(landmarks, bbox):
    hull = cv2.convexHull(landmarks)
    face_area = cv2.contourArea(hull)
    bbox_area = w * h
    return face_area / bbox_area
```

---

## 5. Verification Gate

### Overview

Verification gate runs BEFORE identity_state.update(). All checks must pass.

| Check | Threshold | Purpose |
|---|---|---|
| Face pixels | >= 4000 | Reject tiny faces |
| Embedding distance | <= 0.45 | Reject wrong identity |
| Liveness (jitter) | >= 0.5 | Reject static posters |

### Implementation

```python
class VerificationGate:
    def __init__(self, embedding_tolerance=0.45, min_face_pixels=4000, liveness_threshold=0.5):
        self.embedding_tolerance = embedding_tolerance
        self.min_face_pixels = min_face_pixels
        self.liveness_threshold = liveness_threshold
    
    def check(self, face_bbox, landmarks_pts, embedding) -> Tuple[bool, str]:
        # Gate 1: Face pixel count
        if w * h < self.min_face_pixels:
            return False, "face_too_small"
        
        # Gate 2: Embedding identity check
        if embedding is not None:
            dist = self._embedding_distance(embedding, reference)
            if dist > self.embedding_tolerance:
                return False, "identity_mismatch"
        
        # Gate 3: Liveness check (landmark jitter)
        if len(history) >= 2:
            jitter = self._compute_jitter()
            if jitter < self.liveness_threshold:
                return False, "static_poster"
        
        return True, "passed"
```

### Integration

```python
# In IdentityState.update():
def update(self, canonical_face, quality_map, ..., face_bbox=None, landmarks_pts=None, embedding=None):
    # VERIFICATION GATE: Check all gates before updating
    if face_bbox is not None:
        passed, reason = self.verification_gate.check(face_bbox, landmarks_pts, embedding)
        if not passed:
            return False  # Reject this observation
    
    # ... proceed with update ...
    return True
```

---

## 6. Configuration Reference

**File:** `face_os_config.yaml`

```yaml
identity:
  reference_dir: "photos/"
  reference_image: "expectation.png"
  embedding_tolerance: 0.45  # Stricter than V2 (was 0.50)

detection:
  model: "mediapipe"  # Changed from "hog" to MediaPipe
  min_face_size: 60
  detection_interval: 5
  max_lost_frames: 30
  smoothing_alpha: 0.3

quality_gates:
  procrustes_threshold: 0.09  # Face shape must match reference
  jitter_threshold: 0.0008    # Real face moves (poster is static)
  occupancy_threshold: 0.25   # Face must fill enough of bbox

verification_gate:
  embedding_tolerance: 0.45   # Reject wrong identity
  min_face_pixels: 4000       # Reject tiny faces
  liveness_threshold: 0.5     # Reject static posters

landmarks:
  model: "dlib_68"
  pose_smoothing: 0.4

canonical:
  atlas_size: [256, 256]
  alignment_mode: "similarity"
  enrollment_frames: 30

crop:
  output_size: [1080, 1920]
  headroom_ratio: 0.30
  face_target_width: 270
  smoothing_alpha: 0.25
  max_crop_velocity: 50
  protect_forehead: true

temporal:
  bidirectional: true
  hq_threshold: 0.6
  temporal_window: 5
  repair_strength: 0.3

enhance:
  structure_preserving: true
  eye_definition_boost: 1.3
  brow_texture_boost: 1.2
  beard_preservation: 0.9
  skin_smoothing: 0.3
  sharpen_amount: 0.3
  use_cinematic_noise: true
  noise_strength: 0.02

identity_state:
  low_freq_ema_rate: 0.1
  high_freq_best_only: true
  confidence_modulation: true
  base_confidence: 0.7
  anchor_lambda_max: 0.75  # Reduced from 0.95
  low_blend_base: 0.7      # Dynamic: 0.7 + 0.15*conf
  high_blend_base: 0.3     # Dynamic: 0.3 - 0.15*conf

compositor:
  confidence_threshold: 0.3
  feather_pixels: 10
  use_light_matching: true

export:
  codec: "libx264"
  crf: 18
  preset: "slow"
  bitrate: "25M"
  audio_bitrate: "320k"
  fps: 30
  fade_in: 0.5
  fade_out: 0.5

qc:
  min_face_detection_rate: 0.80
  max_identity_drift: 20.0
  max_flicker_score: 5.0
  min_sharpness: 10.0
```

---

## 7. Test Results & Metrics

**Test clip:** `clips_test/test_clip.mp4` (640x360, 30fps, 15s, 450 frames)  
**Reference:** `expectation.png` (941x1672, portrait)  
**Reference face:** L=108.4, a=139.6, b=146.7

### Test Suite

| File | Tests | Status | Purpose |
|---|---|---|---|
| `test_detection.py` | 14 | ✅ All pass | MediaPipe, poster rejection, identity matching |
| `test_quality_gates.py` | 13 | ✅ All pass | Procrustes, jitter, occupancy, SSIM, Laplacian |
| `test_identity_state.py` | — | ✅ | Identity state logic |
| `test_patch_memory.py` | — | ✅ | Patch memory |
| `test_temporal_solve.py` | — | ✅ | Bidirectional solver |
| `test_face_enhance.py` | — | ✅ | Face rendering |
| `test_appearance_field.py` | — | ✅ | Appearance field |
| `test_neural_codec.py` | — | ✅ | Neural codec |
| `test_strict_quality.py` | 8 | 5 pass, 3 fail | Strict LAB/SSIM targets |
| **Total** | **154+** | **3 skipped** | |

### Strict Test Results (test_strict_quality.py)

| Test | Status | Value | Target |
|---|---|---|---|
| test_compositor_uses_identity_face | ✅ PASS | 197 > 190 | Proves identity used |
| test_disparity_0095_rejected | ✅ PASS | — | — |
| test_static_poster_low_jitter | ✅ PASS | — | — |
| test_small_face_occupancy_024_rejected | ✅ PASS | — | — |
| test_mean_abs_diff_under_20 | ✅ PASS | — | — |
| test_lab_distance_under_5 | ❌ FAIL | 20.3 | <5 |
| test_ssim_above_075 | ❌ FAIL | ~0.6 | >0.75 |
| test_laplacian_variance_above_120 | ❌ FAIL | 7.0 | >120 |

### Face Identity Metrics

| Metric | Reference | Source | Output | Target | Status |
|---|---|---|---|---|---|
| **L (brightness)** | 108.4 | 99.2 | 92.9 | ~108 | ⚠️ Δ15.4 |
| **a (skin tone)** | 139.6 | 139.0 | 138.1 | ~140 | ✅ Δ1.5 |
| **b (warmth)** | 146.7 | 128.4 | 133.6 | ~147 | ⚠️ Δ13.1 |
| **LAB distance** | — | 18.5 | 20.3 | <5 | ❌ |
| **Flicker (L std)** | — | 6.68 | 4.62 | <1.5 | ⚠️ |
| **Face detection** | — | 100% | 100% | — | ✅ |

### Metrics History

| Version | L Δ | a Δ | b Δ | LAB Dist | Notes |
|---|---|---|---|---|---|
| V1 (broken) | -21.1 | -2.1 | -12.9 | 24.8 | Compositor using rendered |
| V2 (compositor fix) | -15.4 | -1.5 | -13.1 | 20.3 | Compositor using identity_face |
| V3 (target) | <5 | <2 | <5 | <5 | Need stronger anchor correction |

---

## 8. Known Issues & Next Steps

### Issue 1: LAB Distance Still 20.3 (Target <5)

**Root cause:** Identity state blending not aggressive enough

**Current blending:**
```python
low_blend = 0.7 + 0.15 * mean_conf  # ~0.82
high_blend = 0.3 - 0.15 * mean_conf  # ~0.18
```

**Possible fixes:**
1. Increase base blend: `low_blend = 0.8 + 0.1 * mean_conf`
2. Use query_identity() for raw identity (no source blending)
3. Increase anchor lambda max back to 0.85

### Issue 2: Laplacian Variance 7.0 (Target >120)

**Root cause:** Ghost mask artifact — output face is blurry

**Possible fixes:**
1. Use face_enhance output for high frequencies instead of identity
2. Increase high_blend to preserve more source detail
3. Apply sharpening after identity blend

### Issue 3: Still Using Haar Cascade in Some Places

**Locations:**
- `canonical_map.py:394` — face detection during enrollment
- `crop_planner.py` — face detection in crop planning

**Fix:** Replace with MediaPipe FaceDetection

### Next Steps (Priority Order)

1. **Fix LAB distance** — Increase identity blending strength
2. **Fix ghost mask** — Preserve more source detail in high frequencies
3. **Replace remaining Haar** — Use MediaPipe everywhere
4. **Multi-anchor system** — Currently 1 anchor, need 7+ (frontal, smile, left/right yaw, etc.)

---

## File Structure (V3)

```
face_os/
├── __init__.py              # Package init
├── types.py                 # Core data structures (FaceTrack with face_mesh, quality_metrics)
├── config.py                # YAML config loader
├── ingest.py                # Module 1: Video loading, frame reader
├── detect_track.py          # Module 2: MediaPipe detection + tracking + quality gates
├── landmarks.py             # Module 3: 68-point landmarks + PnP pose + region masks
├── canonical_map.py         # Module 4: Canonical UV alignment + Appearance Field
├── crop_planner.py          # Module 5: Reference-based crop planning
├── temporal_solve.py        # Module 6: Bidirectional temporal solver
├── face_enhance.py          # Module 7: Structure-preserving rendering
├── identity_state.py        # Module 8: Frequency decomposition + VerificationGate
├── compositor.py            # Module 9: Confidence-weighted compositing
├── appearance_field.py      # AppearanceField + DynamicAppearanceField
├── neural_codec.py          # PersonalizedSpace + NeuralCodec
└── pipeline.py              # Orchestrator (3-pass architecture)

face_os_config.yaml          # All tuning parameters
face_detector.tflite         # MediaPipe face detection model

tests/face_os/
├── test_detection.py        # 14 tests (MediaPipe, poster, identity, occupancy)
├── test_quality_gates.py    # 13 tests (Procrustes, jitter, occupancy, SSIM, Laplacian)
├── test_identity_state.py   # Identity state tests
├── test_patch_memory.py     # Patch memory tests
├── test_temporal_solve.py   # Bidirectional solver tests
├── test_face_enhance.py     # Face rendering tests
├── test_appearance_field.py # Appearance field tests
├── test_neural_codec.py     # Neural codec tests
└── conftest.py              # Shared fixtures

tests/
├── test_strict_quality.py   # 8 strict tests (LAB, SSIM, ghost mask, gates, compositor)
└── ...

output/face_os_v2/
├── output.mp4               # Generated video (1080x1920, 30fps)
└── face_map.png             # Face visualization (reference | source | output)
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| OpenCV (cv2) | ≥4.5 | Image processing, face detection |
| NumPy | ≥1.20 | Array operations |
| dlib | ≥19.22 | 68-point landmarks, face embeddings |
| face_recognition | ≥1.3 | Identity matching (wraps dlib) |
| mediapipe | ≥0.10 | Face detection (replaces Haar) |
| FFmpeg | ≥5.0 | Video encoding (external binary) |
| PyYAML | ≥5.0 | Config file parsing |
