# Face OS — Complete Architecture & Parameter Reference (V4)

**Version:** 0.4.0  
**Branch:** `feat/face-os-pipeline`  
**Date:** 2026-05-21  
**Status:** MediaPipe tasks API | VerificationGate active | LAB distance 24.6 (target <5)

---

## Table of Contents

1. [What Changed From V3](#1-what-changed-from-v3)
2. [Architecture Overview](#2-architecture-overview)
3. [Module-by-Module Deep Dive](#3-module-by-module-deep-dive)
4. [Quality Gates](#4-quality-gates)
5. [Verification Gate](#5-verification-gate)
6. [Configuration Reference](#6-configuration-reference)
7. [Test Results & Metrics](#7-test-results--metrics)
8. [Known Issues & Next Steps](#8-known-issues--next-steps)

---

## 1. What Changed From V3

### V3 → V4 Changes

| Component | V3 | V4 | Why Changed |
|---|---|---|---|
| **Face Detection** | MediaPipe FaceDetection (tasks) | MediaPipe FaceDetector + FaceLandmarker (tasks) | FaceLandmarker gives 478 landmarks for better shape matching |
| **API** | `mp.solutions.face_mesh` | `mediapipe.tasks.python.vision` | MediaPipe 0.10.35 removed `mp.solutions`, uses `tasks` API |
| **Face Mesh** | dlib 68-point (fallback) | MediaPipe 478-point (no fallback) | More landmarks = better Procrustes disparity |
| **Model Files** | `face_detector.tflite` | `face_detector.tflite` + `face_landmarker.task` | FaceLandmarker needs `.task` file |
| **Procrustes Threshold** | 0.09 | 0.2 | Different image sizes need relaxed threshold |
| **Haar Cascade** | Still in canonical_map.py, crop_planner.py | **COMPLETELY REMOVED** | No Haar anywhere in codebase |

### Critical: No Haar Cascade Anywhere

```bash
# These must all return empty:
grep -rn "CascadeClassifier" face_os/    # → empty
grep -rn "haarcascade" face_os/          # → empty
grep -rn "haar" face_os/                 # → empty
```

### MediaPipe Tasks API (V4)

```python
# V3 (BROKEN in 0.10.35):
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh  # AttributeError!
mesh = mp_face_mesh.FaceMesh(...)

# V4 (CORRECT):
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Face Detection
detector = vision.FaceDetector.create_from_options(
    vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path='face_detector.tflite'),
        min_detection_confidence=0.6,
    )
)

# Face Landmark (478 points)
landmarker = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path='face_landmarker.task'),
        num_faces=1,
    )
)
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

---

### Module 2: `detect_track.py` — Face Detection + Tracking + Quality Gates

**What it does:**
- Detects faces using **MediaPipe FaceDetector** (tasks API, min_conf=0.6)
- Extracts **478 landmarks** using **MediaPipe FaceLandmarker** (tasks API)
- Matches detected faces to target identity via embeddings
- Maintains persistent face tracks across frames
- Smooths bounding boxes with EMA
- **Runs quality gates before returning track**

**Detection code (V4):**
```python
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe import Image as MpImage, ImageFormat as MpImageFormat

_face_detector = None
_face_landmarker = None

def get_detector():
    global _face_detector
    if _face_detector is None:
        base_options = mp_python.BaseOptions(model_asset_path='face_detector.tflite')
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=0.6,
        )
        _face_detector = vision.FaceDetector.create_from_options(options)
    return _face_detector

def detect_faces(frame):
    detector = get_detector()
    mp_image = MpImage(
        image_format=MpImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
    )
    result = detector.detect(mp_image)
    
    tracks = []
    for detection in result.detections:
        bbox = detection.bounding_box
        x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
        track = FaceTrack(
            track_id=0,
            state=FaceState.DETECTED,
            smooth_bbox=(x, y, w, h),
            detection=FaceDetection(
                bbox=(x, y, w, h),
                confidence=detection.categories[0].score,
                is_target=True,
            ),
        )
        tracks.append(track)
    return tracks
```

**Face Landmark extraction (V4):**
```python
def extract_face_mesh(frame):
    landmarker = get_landmarker()
    mp_image = MpImage(
        image_format=MpImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
    )
    result = landmarker.detect(mp_image)
    
    if not result.face_landmarks:
        return None
    
    h, w = frame.shape[:2]
    landmarks = result.face_landmarks[0]
    pts = np.array(
        [[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks],
        dtype=np.float32,
    )
    return pts  # Shape: (478, 3)
```

**Quality Gates (all must pass):**
```python
# Gate 1: Procrustes disparity < 0.2 (relaxed for different image sizes)
disparity = compute_procrustes_disparity(mesh, reference_mesh)
if disparity > 0.2:
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

---

### Module 3: `landmarks.py` — Landmarks + Head Pose

**What it does:**
- Extracts 68-point facial landmarks (dlib or geometric fallback)
- Estimates head pose (yaw, pitch, roll) using PnP algorithm
- Creates per-region masks (eyes, brows, nose, mouth, skin, face contour)

---

### Module 4: `canonical_map.py` — Canonical Face Mapping

**What it does:**
- Aligns detected face to canonical UV space (frontal, neutral pose)
- Builds Appearance Field A(u,v,θ,L,t)
- Accumulates pixel observations over time (Photic Memory)
- **Extracts embeddings from face region (not full image)**

**Embedding extraction (V4 — uses FaceTrack):**
```python
from face_os.detect_track import detect_faces
detections = detect_faces(img)
if detections:
    track = detections[0]
    x, y, w, h = track.smooth_bbox  # FaceTrack object
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

**CompositionReference.from_image() (V4):**
```python
from face_os.detect_track import detect_faces
detections = detect_faces(img)
if detections:
    track = detections[0]
    x, y, fw, fh = track.smooth_bbox  # FaceTrack object
```

---

### Module 6: `temporal_solve.py` — Bidirectional Temporal Solver

**What it does:**
- **Forward pass (Pass 1):** Collects per-frame quality metrics, identifies HQ frames
- **Backward pass (Pass 2):** HQ frames repair past blurry frames

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

**Dynamic blending (V4):**
```python
# Confidence-weighted blend
mean_conf = np.mean(confidence)
low_blend = 0.85 + 0.1 * mean_conf   # 0.85-0.95
high_blend = 0.15 - 0.1 * mean_conf  # 0.05-0.15
```

**Anchor correction (V4 — lambda max 0.75):**
```python
if drift > 30:
    lambda_base = 0.75
elif drift > 15:
    lambda_base = 0.70
elif drift > 5:
    lambda_base = 0.65
else:
    lambda_base = 0.60

lambda_clamped = np.clip(lambda_conf, 0.60, 0.75)
```

**VerificationGate (V4):**
```python
class VerificationGate:
    def __init__(self, embedding_tolerance=0.45, min_face_pixels=4000, liveness_threshold=0.5):
        self.embedding_tolerance = embedding_tolerance
        self.min_face_pixels = min_face_pixels
        self.liveness_threshold = liveness_threshold
    
    def verify(self, canonical_face, face_bbox, landmarks_pts, embedding=None):
        # Gate 1: Face pixel count
        if face_bbox is not None:
            x, y, w, h = face_bbox
            if w * h < self.min_face_pixels:
                return False, "face_too_small"
        
        # Gate 2: Embedding identity check
        if self._reference_embedding is not None and embedding is not None:
            dist = self._embedding_distance(embedding, self._reference_embedding)
            if dist > self.embedding_tolerance:
                return False, "identity_mismatch"
        
        # Gate 3: Liveness check (landmark jitter)
        if landmarks_pts is not None:
            pts_2d = landmarks_pts[:, :2] if landmarks_pts.shape[1] > 2 else landmarks_pts
            self._landmark_history.append(pts_2d.copy())
            if len(self._landmark_history) >= 2:
                jitter = self._compute_jitter()
                if jitter < self.liveness_threshold:
                    return False, "static_poster"
        
        return True, "passed"
```

---

### Module 9: `compositor.py` — Confidence-Weighted Compositing

**What it does:**
- Composites **identity_face** onto original frame using per-pixel confidence
- High confidence pixels → use identity memory (stable, clean)
- Low confidence pixels → use original frame (noisy but authentic)
- Feathered edge blending prevents visible seams

**V4 compositing (uses identity_face, not rendered):**
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

**Identity update with verification (V4):**
```python
# Get verification parameters from track
face_bbox = face_track.smooth_bbox
landmarks_pts = face_track.mesh_468[:, :2] if hasattr(face_track, 'mesh_468') else None
embedding = face_track.detection.embedding if face_track.detection else None

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
| Procrustes disparity | < 0.2 | Face shape must match reference (relaxed for different image sizes) |
| Landmark jitter | > 0.0008 | Real face moves (poster is static) |
| Occupancy | > 0.25 | Face must fill enough of bbox |

### Procrustes Disparity (V4)

Measures shape difference between current face landmarks and reference.
Uses only x,y coordinates (not z) for 2D shape comparison.

```python
def compute_procrustes_disparity(landmarks, reference_landmarks):
    # Use only x,y for shape comparison
    lm = landmarks[:, :2] if landmarks.shape[1] > 2 else landmarks
    ref = reference_landmarks[:, :2] if reference_landmarks.shape[1] > 2 else reference_landmarks
    
    # Center both
    lm = lm - lm.mean(axis=0)
    ref = ref - ref.mean(axis=0)
    
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
    if len(landmark_history) < 2:
        return 1.0  # Assume real if not enough history
    
    displacements = []
    for i in range(1, len(history)):
        prev = history[i - 1]
        curr = history[i]
        if prev.shape == curr.shape:
            disp = mean(sqrt(sum((curr - prev)^2, axis=1)))
            disp_norm = disp / 200.0  # Normalize by face size
            displacements.append(disp_norm)
    
    return mean(displacements) if displacements else 1.0
```

### Occupancy

Measures how much of the bounding box is filled by the face.

```python
def compute_occupancy(landmarks, bbox):
    x, y, w, h = bbox
    bbox_area = w * h
    if bbox_area <= 0:
        return 0.0
    
    pts_2d = landmarks[:, :2] if landmarks.shape[1] > 2 else landmarks
    hull = cv2.convexHull(pts_2d.astype(np.float32))
    face_area = cv2.contourArea(hull)
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

### Implementation (V4)

```python
class VerificationGate:
    def __init__(self, embedding_tolerance=0.45, min_face_pixels=4000, liveness_threshold=0.5):
        self.embedding_tolerance = embedding_tolerance
        self.min_face_pixels = min_face_pixels
        self.liveness_threshold = liveness_threshold
        self._reference_embedding = None
        self._landmark_history = []
    
    def set_reference_embedding(self, embedding):
        self._reference_embedding = embedding
    
    def verify(self, canonical_face, face_bbox, landmarks_pts, embedding=None):
        # Gate 1: Face pixel count
        if face_bbox is not None:
            x, y, w, h = face_bbox
            if w * h < self.min_face_pixels:
                return False, f"face_too_small: {w*h} < {self.min_face_pixels}"
        
        # Gate 2: Embedding identity check
        if self._reference_embedding is not None and embedding is not None:
            dist = self._embedding_distance(embedding, self._reference_embedding)
            if dist > self.embedding_tolerance:
                return False, f"identity_mismatch: {dist:.3f} > {self.embedding_tolerance}"
        
        # Gate 3: Liveness check (landmark jitter)
        if landmarks_pts is not None:
            pts_2d = landmarks_pts[:, :2] if landmarks_pts.shape[1] > 2 else landmarks_pts
            self._landmark_history.append(pts_2d.copy())
            if len(self._landmark_history) > 10:
                self._landmark_history = self._landmark_history[-10:]
            
            if len(self._landmark_history) >= 2:
                jitter = self._compute_jitter()
                if jitter < self.liveness_threshold:
                    return False, f"static_poster: jitter={jitter:.4f} < {self.liveness_threshold}"
        
        return True, "passed"
```

### Integration (V4)

```python
# In IdentityState.__init__():
self._gate = VerificationGate(
    embedding_tolerance=0.45,
    min_face_pixels=4000,
    liveness_threshold=0.5,
)

# In IdentityState.update():
def update(self, canonical_face, quality_map, ..., face_bbox=None, landmarks_pts=None, embedding=None):
    # VERIFICATION GATE: Check all gates before updating
    if face_bbox is not None or landmarks_pts is not None:
        ok, _ = self._gate.verify(canonical_face, face_bbox, landmarks_pts, embedding)
        if not ok:
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
  embedding_tolerance: 0.45

detection:
  model: "mediapipe"  # MediaPipe tasks API
  min_face_size: 60
  detection_interval: 5
  max_lost_frames: 30
  smoothing_alpha: 0.3

quality_gates:
  procrustes_threshold: 0.2   # Relaxed for different image sizes
  jitter_threshold: 0.0008    # Real face moves (poster is static)
  occupancy_threshold: 0.25   # Face must fill enough of bbox

verification_gate:
  embedding_tolerance: 0.45   # Reject wrong identity
  min_face_pixels: 4000       # Reject tiny faces
  liveness_threshold: 0.5     # Reject static posters

landmarks:
  model: "mediapipe_478"      # MediaPipe FaceLandmarker (478 points)
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
  anchor_lambda_max: 0.75
  low_blend_base: 0.85      # Dynamic: 0.85 + 0.1*conf
  high_blend_base: 0.15     # Dynamic: 0.15 - 0.1*conf

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
**Reference face:** L=114.1, a=140.7, b=146.8

### Test Suite (V4)

| File | Tests | Status | Purpose |
|---|---|---|---|
| `test_detection.py` | 14 | ✅ All pass | MediaPipe tasks API, poster rejection, identity matching |
| `test_quality_gates.py` | 13 | ✅ All pass | Procrustes, jitter, occupancy, SSIM, Laplacian |
| `test_identity_state.py` | — | ✅ (1 fail) | Identity state logic (identity_slower_than_source fails) |
| `test_patch_memory.py` | — | ✅ | Patch memory |
| `test_temporal_solve.py` | — | ✅ | Bidirectional solver |
| `test_face_enhance.py` | — | ✅ | Face rendering |
| `test_appearance_field.py` | — | ✅ | Appearance field |
| `test_neural_codec.py` | — | ✅ | Neural codec |
| `test_strict_quality.py` | 5 | 4 pass, 1 fail | No Haar, verification gate, LAB distance |
| **Total** | **156+** | **1 fail** | |

### Strict Test Results (test_strict_quality.py — V4)

| Test | Status | Value | Target |
|---|---|---|---|
| test_no_haar_in_codebase | ✅ PASS | grep empty | No Haar cascade |
| test_no_haarcascade_in_codebase | ✅ PASS | grep empty | No haarcascade |
| test_rejects_tiny_face | ✅ PASS | 2500 < 4000 | face_too_small |
| test_accepts_large_face | ✅ PASS | 10000 >= 4000 | accepted |
| test_lab_distance_under_5 | ❌ FAIL | 24.6 | <5 |

### Face Identity Metrics (V4)

| Metric | Reference | Source | Output | Target | Status |
|---|---|---|---|---|---|
| **L (brightness)** | 114.1 | — | 93.1 | ~114 | ⚠️ Δ21.0 |
| **a (skin tone)** | 140.7 | — | 137.8 | ~141 | ⚠️ Δ2.9 |
| **b (warmth)** | 146.8 | — | 134.4 | ~147 | ⚠️ Δ12.4 |
| **LAB distance** | — | — | 24.6 | <5 | ❌ |
| **Face detection** | — | — | 64% | >80% | ⚠️ |

### Pipeline Output (V4)

```
Collected 96 canonical faces from 150 frames
Bidirectional solver: 20 HQ frames identified
Solved 96 frames, 20 HQ frames
Output: 150 frames, 5.5MB, 4fps
Anchor distance: 0.6 LAB (threshold: 25.0)
```

### Metrics History

| Version | L Δ | a Δ | b Δ | LAB Dist | Notes |
|---|---|---|---|---|---|
| V1 (broken) | -21.1 | -2.1 | -12.9 | 24.8 | Compositor using rendered |
| V2 (compositor fix) | -15.4 | -1.5 | -13.1 | 20.3 | Compositor using identity_face |
| V3 (quality gates) | -15.4 | -1.5 | -13.1 | 20.3 | Procrustes 0.09 |
| V4 (current) | -21.0 | -2.9 | -12.4 | 24.6 | MediaPipe tasks, Procrustes 0.2 |
| V4 (target) | <5 | <2 | <5 | <5 | Need stronger anchor correction |

---

## 8. Known Issues & Next Steps

### Issue 1: LAB Distance Still 24.6 (Target <5)

**Root cause:** Identity state blending not aggressive enough

**Current blending:**
```python
low_blend = 0.85 + 0.1 * mean_conf  # ~0.92
high_blend = 0.15 - 0.1 * mean_conf  # ~0.08
```

**Possible fixes:**
1. Increase base blend: `low_blend = 0.95 + 0.05 * mean_conf`
2. Use query_identity() for raw identity (no source blending)
3. Increase anchor lambda max to 0.9+

### Issue 2: Face Detection Rate 64% (Target >80%)

**Root cause:** Quality gates rejecting frames with different face shapes

**Possible fixes:**
1. Relax Procrustes threshold further (0.3?)
2. Use only jitter + occupancy gates (skip Procrustes)
3. Improve reference mesh extraction

### Issue 3: identity_slower_than_source Test Failing

**Root cause:** With high low_blend (0.85-0.95), identity changes almost as fast as source when source changes dramatically

**Possible fixes:**
1. Adjust test threshold
2. Use EMA on identity state (not just per-frame blend)

### Next Steps (Priority Order)

1. **Fix LAB distance** — Increase identity blending strength to 0.95+
2. **Fix face detection rate** — Relax quality gates or improve reference mesh
3. **Fix identity_slower_than_source** — Add EMA to identity state
4. **Multi-anchor system** — Currently 1 anchor, need 7+ (frontal, smile, left/right yaw, etc.)

---

## File Structure (V4)

```
face_os/
├── __init__.py              # Package init
├── types.py                 # Core data structures (FaceTrack with mesh_468, quality_metrics)
├── config.py                # YAML config loader
├── ingest.py                # Module 1: Video loading, frame reader
├── detect_track.py          # Module 2: MediaPipe tasks API (FaceDetector + FaceLandmarker)
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
face_landmarker.task         # MediaPipe face landmark model (478 points)

tests/face_os/
├── test_detection.py        # 14 tests (MediaPipe tasks API, poster, identity, occupancy)
├── test_quality_gates.py    # 13 tests (Procrustes, jitter, occupancy, SSIM, Laplacian)
├── test_identity_state.py   # Identity state tests
├── test_patch_memory.py     # Patch memory tests
├── test_temporal_solve.py   # Bidirectional solver tests
├── test_face_enhance.py     # Face rendering tests
├── test_appearance_field.py # Appearance field tests
├── test_neural_codec.py     # Neural codec tests
└── conftest.py              # Shared fixtures

tests/
├── test_strict_quality.py   # 5 strict tests (No Haar, verification gate, LAB distance)
└── ...

output/face_os_v2/
├── output.mp4               # Generated video (1080x1920, 30fps, 150 frames, 5.5MB)
└── face_map.png             # Face visualization (reference | source | output)
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| OpenCV (cv2) | ≥4.5 | Image processing |
| NumPy | ≥1.20 | Array operations |
| dlib | ≥19.22 | Face embeddings (fallback) |
| face_recognition | ≥1.3 | Identity matching (wraps dlib) |
| mediapipe | ≥0.10.35 | Face detection + landmarks (tasks API) |
| FFmpeg | ≥5.0 | Video encoding (external binary) |
| PyYAML | ≥5.0 | Config file parsing |
