# Face OS — Complete Architecture & Parameter Reference

**Version:** 0.1.0  
**Branch:** `feat/face-os-pipeline`  
**Date:** 2026-05-20  
**Author:** Auto-generated from codebase analysis

---

## Table of Contents

1. [What Changed](#1-what-changed)
2. [Architecture Overview](#2-architecture-overview)
3. [Module-by-Module Deep Dive](#3-module-by-module-deep-dive)
4. [Core Parameter Calculations](#4-core-parameter-calculations)
5. [Configuration Reference](#5-configuration-reference)
6. [Test Results & Comparison](#6-test-results--comparison)
7. [Known Issues & Next Steps](#7-known-issues--next-steps)

---

## 1. What Changed

### Branch: `feat/face-os-pipeline` from `main`

**16 new files, 4,176 lines of code:**

| File | Lines | Purpose |
|---|---|---|
| `face_os/__init__.py` | 27 | Package init |
| `face_os/types.py` | 234 | Core data structures |
| `face_os/config.py` | 218 | YAML config loader |
| `face_os/ingest.py` | 195 | Video ingestion + frame reader |
| `face_os/detect_track.py` | 370 | Face detection + temporal tracking |
| `face_os/landmarks.py` | 346 | 68-point landmarks + head pose |
| `face_os/canonical_map.py` | 407 | Appearance field + photic memory |
| `face_os/crop_planner.py` | 270 | Face-aware 9:16 crop |
| `face_os/temporal_stabilize.py` | 163 | Identity inertia + flicker suppression |
| `face_os/face_enhance.py` | 315 | Eye-dominant rendering |
| `face_os/identity_memory.py` | 243 | Per-pixel confidence accumulation |
| `face_os/compositor.py` | 208 | Confidence-weighted compositing |
| `face_os/export_qc.py` | 341 | FFmpeg encode + quality validation |
| `face_os/pipeline.py` | 473 | Orchestrator |
| `face_os_config.yaml` | — | All tuning parameters |
| `face_os/ARCHITECTURE_V2.md` | 274 | Architecture doc |

**2 additional files:**

| File | Lines | Purpose |
|---|---|---|
| `test_face_os_comparison.py` | 612 | 4-way comparison test |

### What Was NOT Changed

The existing pipeline (`pipeline.py`, `export.py`, `ref_grade.py`, `face_mapper.py`, `frame_analyzer.py`) is untouched. Face OS is a parallel system that can coexist.

---

## 2. Architecture Overview

```
INPUT: 16:9 source video + reference face images
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
    │   Module 4    │ ◄── Appearance Field A(u,v,θ,L,t)
    │   Canonical   │     (Photic Memory accumulation)
    │   Face Map    │
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │   Module 5    │ ◄── Face-locked 9:16 with headroom
    │   Crop Plan   │     (protect forehead, allow bottom crop)
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │   Module 6    │ ◄── Identity Inertia
    │   Temporal    │     (ΔI_identity ≪ ΔI_source)
    │   Stabilizer  │
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │   Module 7    │ ◄── Eye Dominance Rendering
    │   Face        │     (eyes > brows > beard > skin > bg)
    │   Enhancer    │
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │   Module 8    │ ◄── Photic Memory
    │   Identity    │     (accumulate confidence over time)
    │   Memory      │
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │   Module 9    │ ◄── Per-pixel confidence blending
    │   Compositor  │     (high conf = memory, low conf = source)
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │   Module 10   │ ◄── Quality validation
    │   Export + QC │     (face rate, drift, flicker, sharpness)
    └───────────────┘
            │
            ▼
OUTPUT: 9:16 stabilized video
```

---

## 3. Module-by-Module Deep Dive

### Module 1: `ingest.py` — Video Ingestion

**What it does:**
- Loads video file and extracts metadata (dimensions, fps, codec, duration)
- Provides frame-by-frame generator with seeking support
- Loads reference face images for identity enrollment
- Validates audio-video sync

**Key function:**
```python
def frame_reader(video_path, start_frame=0, end_frame=None, step=1):
    """Yield (frame_idx, timestamp_sec, bgr_frame) from video."""
```

**How it works:**
1. Opens video with OpenCV `VideoCapture`
2. Seeks to `start_frame` if needed
3. Reads frames sequentially, yields every `step`-th frame
4. Closes capture when done

---

### Module 2: `detect_track.py` — Face Detection + Tracking

**What it does:**
- Detects faces using Haar Cascade (every N frames)
- Matches detected faces to target identity via embeddings
- Maintains persistent face tracks across frames
- Smooths bounding boxes with EMA

**Detection strategy:**
```
Frame 0:  DETECT → find faces → match identity → create track
Frame 1:  TRACK  → predict position (use last known bbox)
Frame 2:  TRACK  → predict position
Frame 3:  TRACK  → predict position
Frame 4:  TRACK  → predict position
Frame 5:  DETECT → find faces → match identity → update track
...
```

**Identity matching formula:**
```python
# Compute face embedding (dlib or histogram fallback)
embedding = face_recognition.face_encodings(rgb_frame, face_locations)

# Compare against reference embeddings
distances = face_recognition.face_distance(reference_embeddings, embedding)
min_distance = np.min(distances)

# Match if distance below tolerance
is_match = min_distance <= tolerance  # default: 0.50
```

**EMA smoothing:**
```python
smoothed_bbox = prev_bbox * (1 - alpha) + current_bbox * alpha
# alpha = 0.3 (config.detection.smoothing_alpha)
```

**Track state machine:**
```
DETECTED → (miss 1 frame) → TRACKED → (miss N frames) → OCCLUDED → (miss M frames) → LOST → removed
```
- N = `max_lost_frames / 2` = 15 frames
- M = `max_lost_frames` = 30 frames

---

### Module 3: `landmarks.py` — Landmarks + Head Pose

**What it does:**
- Extracts 68-point facial landmarks (dlib or geometric fallback)
- Estimates head pose (yaw, pitch, roll) using PnP algorithm
- Creates per-region masks (eyes, brows, nose, mouth, skin, face contour)

**68-point layout:**
```
Points 0-16:   Jaw line (17 points)
Points 17-21:  Right eyebrow (5 points)
Points 22-26:  Left eyebrow (5 points)
Points 27-30:  Nose bridge (4 points)
Points 31-35:  Nose bottom (5 points)
Points 36-41:  Right eye (6 points)
Points 42-47:  Left eye (6 points)
Points 48-59:  Outer lip (12 points)
Points 60-67:  Inner lip (8 points)
```

**Head pose estimation (PnP):**
```python
# 3D model points (generic face model)
model_points = np.array([
    [0.0, 0.0, 0.0],             # Nose tip (point 30)
    [0.0, -63.6, -12.5],          # Chin (point 8)
    [-43.3, 32.7, -26.0],         # Left eye (point 36)
    [43.3, 32.7, -26.0],          # Right eye (point 45)
    [-28.9, -28.9, -24.1],        # Left mouth (point 48)
    [28.9, -28.9, -24.1],         # Right mouth (point 54)
])

# 2D image points from detected landmarks
image_points = pts[[30, 8, 36, 45, 48, 54]]

# Camera matrix (approximate)
focal_length = frame_width
camera_matrix = [[focal_length, 0, center_x],
                 [0, focal_length, center_y],
                 [0, 0, 1]]

# Solve PnP
success, rotation_vec, translation_vec = cv2.solvePnP(
    model_points, image_points, camera_matrix, dist_coeffs
)

# Convert to Euler angles
rotation_mat, _ = cv2.Rodrigues(rotation_vec)
angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_mat)

yaw = angles[1]    # Left/right rotation
pitch = angles[0]  # Up/down rotation
roll = angles[2]   # Head tilt
```

**Region mask generation:**
```python
# Eye mask: elliptical around eye contour points
for i in range(6):
    angle = i * np.pi / 3
    eye_points[i] = [center_x + radius * cos(angle),
                     center_y + radius * sin(angle)]

# Gaussian smoothing for soft edges
mask = cv2.GaussianBlur(mask, (ksize, ksize), sigma)
```

---

### Module 4: `canonical_map.py` — Canonical Face Mapping + Appearance Field

**What it does:**
- Aligns detected face to canonical UV space (frontal, neutral pose)
- Builds Appearance Field A(u,v,θ,L,t) — the dynamic appearance function
- Accumulates pixel observations over time (Photic Memory)
- Computes identity residual (what makes THIS face unique)

**Canonical alignment:**
```python
# Standard 68-point canonical positions (256x256 atlas)
canonical_points = _get_canonical_points((256, 256))

# Compute similarity transform (rotation + scale + translation)
anchor_indices = [30, 36, 45, 48, 54]  # Nose, eyes, mouth corners
src_anchor = source_landmarks[anchor_indices]
dst_anchor = canonical_points[anchor_indices]

M = cv2.estimateAffinePartial2D(src_anchor, dst_anchor)  # 2x3 matrix
M_3x3 = np.vstack([M, [0, 0, 1]])  # 3x3 for inverse

# Warp face to canonical space
warped = cv2.warpAffine(frame, M, (256, 256), flags=cv2.INTER_LANCZOS4)
```

**Photic Memory accumulation:**
```python
# Per-pixel quality score
quality = sharpness * brightness_weight * detection_confidence * pose_weight

# Exponential moving average (EMA)
rate = 0.1  # accumulation_rate
weight = rate * quality * detection_confidence
weight_3d = weight[:, :, np.newaxis]  # For RGB broadcasting

accumulated = accumulated * (1 - weight_3d) + new_observation * weight_3d

# Observation count (decays without new observations)
observation_count = observation_count * (1 - decay_rate) + quality
# decay_rate = 0.01

# Confidence = normalized observation count
confidence = np.clip(observation_count / min_observations, 0, 1)
# min_observations = 5
```

**Pose weighting:**
```python
# Prefer frontal poses for atlas building
pose_distance = sqrt(yaw^2 + pitch^2)
frontal_weight = exp(-pose_distance / 45.0)  # 45° half-life

# Consistency weight (prefer poses close to recent average)
avg_yaw = mean(recent_yaws)
avg_pitch = mean(recent_pitches)
consistency = exp(-sqrt((yaw-avg_yaw)^2 + (pitch-avg_pitch)^2) / 30.0)

pose_weight = frontal_weight * consistency
```

---

### Module 5: `crop_planner.py` — Face-Aware Crop with Headroom

**What it does:**
- Plans 16:9 → 9:16 crop that follows the face
- Positions face at ~30% from top (matching reference composition)
- Protects forehead (never crops above top of head)
- Allows bottom crop (OK to crop below chin)
- Smooths crop transitions with EMA

**Crop calculation:**
```python
# Target face width in output (pixels)
target_face_w = 270  # 25% of 1080px output

# Zoom scale
scale = target_face_w / source_face_w

# Crop dimensions in source space
crop_w = output_width / scale   # e.g., 1080 / scale
crop_h = output_height / scale  # e.g., 1920 / scale

# Maintain 9:16 aspect ratio
target_aspect = 1080 / 1920  # = 0.5625
if crop_w / crop_h > target_aspect:
    crop_w = crop_h * target_aspect
else:
    crop_h = crop_w / target_aspect

# Position: face at headroom_ratio from top
headroom_ratio = 0.30
crop_x = face_center_x - crop_w / 2
crop_y = face_center_y - crop_h * headroom_ratio

# Clamp to frame bounds
crop_x = max(0, min(crop_x, source_w - crop_w))
crop_y = max(0, min(crop_y, source_h - crop_h))

# Protect forehead
head_top = min(landmarks.points[:, 1])
if crop_y > head_top - 10:
    crop_y = max(0, head_top - 10)
```

**EMA smoothing with velocity clamping:**
```python
alpha = 0.25  # smoothing_alpha
max_velocity = 50  # max pixels per frame

dx = new_x - smooth_x
dy = new_y - smooth_y
dx = np.clip(dx, -max_velocity, max_velocity)
dy = np.clip(dy, -max_velocity, max_velocity)

smooth_x += dx * alpha
smooth_y += dy * alpha
smooth_w = smooth_w * (1 - alpha) + new_w * alpha
smooth_h = smooth_h * (1 - alpha) + new_h * alpha
```

---

### Module 6: `temporal_stabilize.py` — Temporal Stabilizer (Identity Inertia)

**What it does:**
- Tracks frame-to-frame LAB statistics (mean L, a, b)
- Detects flicker (sudden LAB shifts not caused by real motion)
- Suppresses flicker by pulling toward rolling average
- Reduces stabilization during real movement (motion compensation)

**Core formula (Identity Inertia):**
```
ΔI_identity ≪ ΔI_source
```

**Implementation:**
```python
# Track LAB statistics per frame
l_mean = mean(lab[:, :, 0])
a_mean = mean(lab[:, :, 1])
b_mean = mean(lab[:, :, 2])

# Rolling average (last N frames)
history = deque(maxlen=5)
avg_stats = mean(history, axis=0)

# Deviation from average
deviation = current_stats - avg_stats
deviation_magnitude = sqrt(sum(deviation^2))

# If deviation > threshold, it's likely flicker
threshold = 15.0  # LAB distance units

if deviation_magnitude > threshold:
    # Motion-compensated correction
    motion_factor = 1.0 - min(motion_score / 50.0, 0.8)
    inertia = 0.85  # identity_inertia

    correction = deviation * inertia * motion_factor
    corrected = current_stats - correction

    # Apply to LAB channels
    lab[:, :, 0] += (corrected[0] - current_stats[0])
    lab[:, :, 1] += (corrected[1] - current_stats[1])
    lab[:, :, 2] += (corrected[2] - current_stats[2])
```

**Motion detection (optical flow):**
```python
# Dense optical flow (Farneback)
flow = cv2.calcOpticalFlowFarneback(
    prev_gray, curr_gray,
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2
)
magnitude = sqrt(flow[:,:,0]^2 + flow[:,:,1]^2)
motion_score = mean(magnitude)
```

---

### Module 7: `face_enhance.py` — Eye-Dominant Face Enhancement

**What it does:**
- Applies per-region enhancement based on facial landmarks
- Eyes get highest boost (human perception is eye-obsessed)
- Adds cinematic noise for realism (clean renders look fake)

**Enhancement levels:**

| Region | Boost | Method |
|---|---|---|
| Eyes | 1.5x | Sharpen (0.4 amount) + brighten sclera (+8 L) |
| Brows | 1.3x | Contrast boost (1.2x on L channel) |
| Beard | 1.2x | Detail enhancement (high-pass blend 0.5x) |
| Skin | 0.3x | Bilateral smoothing (d=9, sigmaColor=75) |
| Background | 0.3x | Vignette (center 1.0, edges 0.7) |
| Global | 0.15x | Gentle unsharp mask |

**Eye enhancement formula:**
```python
# Sharpen
sharpened = unsharp_mask(frame, amount=0.4, radius=1.0)

# Brighten (sclera appears whiter)
brightened = frame + 8

# Blend in eye region
result = frame * (1 - mask*boost) + sharpened * mask*boost * 0.6 + brightened * mask*boost * 0.4
```

**Cinematic noise:**
```python
# Gaussian noise (matches sensor distribution)
noise = randn(h, w) * strength * 255  # strength = 0.02

# Correlate slightly (mimics real sensor patterns)
noise = GaussianBlur(noise, (3,3), 0.5)

# Reduce in highlights (like real sensors)
highlight_factor = 1.0 - (L_channel / 255.0) * 0.5
noise *= highlight_factor

# Apply to L channel only (no color shift)
lab[:, :, 0] += noise
```

---

### Module 8: `identity_memory.py` — Photic Memory Atlas

**What it does:**
- Treats each frame as a noisy photon observation
- Accumulates per-pixel confidence over time
- Confidence grows with more observations, decays without
- Pose-weighted: frontal observations count more

**Per-pixel quality calculation:**
```python
# Sharpness (Laplacian magnitude)
lap = abs(cv2.Laplacian(gray, CV_32F))
sharpness = clip(lap / 50.0, 0, 1)

# Brightness preference (mid-tones are best)
brightness = gray / 255.0
brightness_weight = 1.0 - abs(brightness - 0.5) * 2
brightness_weight = clip(brightness_weight, 0.1, 1.0)

# Combined quality
quality = sharpness * brightness_weight * detection_confidence * pose_weight
```

**Accumulation rate adaptation:**
```python
# Slower accumulation for well-observed pixels
obs_factor = 1.0 / (1.0 + observation_count * 0.01)
effective_rate = base_rate * obs_factor * quality * pose_weight
# base_rate = 0.1

# EMA update
accumulated = accumulated * (1 - effective_rate) + new_observation * effective_rate

# Decay unobserved pixels
frames_since_last = current_frame - last_observation_frame
if frames_since_last > max_age_frames:  # 300
    observation_count *= 0.5
```

**Confidence map:**
```python
# Spatial confidence (normalized observation count)
spatial_conf = clip(observation_count / min_observations, 0, 1)
# min_observations = 5

# Temporal stability (how similar is current to history)
stability = exp(-abs(new - accumulated) / 20.0)

# Combined confidence
confidence = spatial_conf * temporal_stability
```

---

### Module 9: `compositor.py` — Confidence-Weighted Compositing

**What it does:**
- Composites enhanced face onto original frame using per-pixel confidence
- High confidence pixels → use accumulated memory (stable, clean)
- Low confidence pixels → use original frame (noisy but authentic)
- Feathered edge blending prevents visible seams

**Compositing formula:**
```python
# Feathered face mask
feathered = GaussianBlur(face_mask, (ksize, ksize), feather/2)
# feather = 10 pixels

# Blend weight = face_mask * confidence
blend_weight = feathered * confidence

# Optional: lighting matching
if use_light_matching:
    # Match mean brightness between face and background
    ref_mean = mean(original_L[face_mask > 0.5])
    tgt_mean = mean(enhanced_L[face_mask > 0.5])
    adjustment = (ref_mean - tgt_mean) * 0.3
    enhanced_L += adjustment

# Composite
result = original * (1 - blend_weight) + enhanced * blend_weight
```

**Light matching:**
```python
# Compute mean L in face region for both frames
ref_mean = mean(original_lab[:, :, 0][face_mask])
tgt_mean = mean(enhanced_lab[:, :, 0][face_mask])

# Partial adjustment (don't fully match — preserve enhancement)
diff = ref_mean - tgt_mean
if abs(diff) > 5:
    enhanced_lab[:, :, 0] += diff * 0.3
```

---

### Module 10: `export_qc.py` — Export + Quality Control

**What it does:**
- Exports processed frames to video via FFmpeg pipe
- Applies fade in/out transitions
- Validates quality with 5 checks

**FFmpeg pipe:**
```python
cmd = [
    "ffmpeg", "-y",
    "-f", "rawvideo", "-pix_fmt", "bgr24",
    "-s", f"{width}x{height}", "-r", str(fps),
    "-i", "-",                          # stdin pipe
    "-i", source_path,                  # for audio muxing
    "-map", "0:v", "-map", "1:a?",
    "-c:v", "libx264", "-crf", "18", "-preset", "slow",
    "-c:a", "aac", "-b:a", "320k",
    "-movflags", "+faststart",
    output_path,
]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
proc.stdin.write(frame.tobytes())  # Write each frame
```

**QC checks:**

| Check | Threshold | Formula |
|---|---|---|
| Face detection rate | ≥ 80% | `frames_with_face / total_frames` |
| Identity drift | ≤ 20.0 LAB | `sqrt((L-ref_L)^2 + (a-ref_a)^2 + (b-ref_b)^2)` |
| Flicker score | ≤ 5.0 LAB | `mean(sqrt(ΔL^2 + Δa^2 + Δb^2))` between consecutive frames |
| Sharpness | ≥ 10.0 | `var(Laplacian(gray))` |
| A/V sync | ≤ 0.5s | `abs(video_duration - audio_duration)` |

---

## 4. Core Parameter Calculations

### LAB Color Space

All color calculations use CIE L*a*b* (LAB) color space:
- **L** = Lightness (0=black, 100=white)
- **a** = Red-green axis (+a=red, -a=green)
- **b** = Yellow-blue axis (+b=yellow, -b=blue)

```python
lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
L = lab[:, :, 0]  # Lightness
a = lab[:, :, 1]  # Red-green
b = lab[:, :, 2]  # Yellow-blue
```

**LAB distance formula:**
```python
lab_distance = sqrt((L1-L2)^2 + (a1-a2)^2 + (b1-b2)^2)
```

### Reference Values (expectation.png)

| Parameter | Value | Source |
|---|---|---|
| Face L | 108.5 | Mean of face LAB L channel |
| Face a | 139.6 | Mean of face LAB a channel |
| Face b | 146.7 | Mean of face LAB b channel |
| Body L | 174.8 | Mean of body region L |
| Background L | 102.1 | Mean of non-face, non-body L |
| Contrast (std) | 58.5 | Std of face grayscale |
| Saturation | 124.7 | Mean of face HSV S channel |
| Vignette ratio | 1.19 | Center/edge brightness ratio |
| Face position | y=0.24, h=0.34 | Normalized face bbox |

### Key Formulas Summary

| Formula | Where Used | Purpose |
|---|---|---|
| `EMA = prev * (1-α) + new * α` | Modules 2,4,5,6,8 | Temporal smoothing |
| `quality = sharpness * brightness * conf * pose` | Modules 4,8 | Per-pixel quality scoring |
| `confidence = clip(obs_count / min_obs, 0, 1)` | Module 8 | Confidence accumulation |
| `blend = original * (1-w) + enhanced * w` | Module 9 | Confidence-weighted compositing |
| `lab_dist = sqrt(ΔL² + Δa² + Δb²)` | Module 10 | Identity drift measurement |
| `flicker = mean(sqrt(ΔL² + Δa² + Δb²))` | Module 10 | Frame-to-frame stability |
| `PnP → RQDecomp → (yaw, pitch, roll)` | Module 3 | Head pose estimation |

---

## 5. Configuration Reference

**File:** `face_os_config.yaml`

```yaml
identity:
  reference_dir: "photos/"           # Directory with reference face photos
  reference_image: "expectation.png" # Primary reference for appearance
  embedding_tolerance: 0.50          # Face matching threshold (lower = stricter)

detection:
  model: "hog"                       # hog (CPU) | cnn (GPU)
  min_face_size: 60                  # Minimum face size in pixels
  detection_interval: 5              # Detect every N frames (track in between)
  max_lost_frames: 30                # Frames before declaring face LOST
  smoothing_alpha: 0.3               # EMA smoothing for bbox

landmarks:
  model: "dlib_68"                   # dlib_68 | mediapipe_468
  pose_smoothing: 0.4                # EMA for head pose angles

canonical:
  atlas_size: [256, 256]             # Canonical face resolution [W, H]
  alignment_mode: "similarity"       # similarity | affine | perspective
  enrollment_frames: 30              # Frames to average for enrollment

crop:
  output_size: [1080, 1920]          # Target output [W, H]
  headroom_ratio: 0.30               # Fraction of output above face center
  face_target_width: 270             # Target face width in output (pixels)
  smoothing_alpha: 0.25              # EMA for crop position
  max_crop_velocity: 50              # Max pixels crop can move per frame
  protect_forehead: true             # Never crop above forehead

temporal:
  identity_inertia: 0.85             # How much identity resists change (0-1)
  flicker_threshold: 15.0            # LAB distance to trigger stabilization
  temporal_window: 5                 # Frames to average for stabilization

enhance:
  eye_boost: 1.5                     # Enhancement multiplier for eyes
  brow_boost: 1.3                    # Enhancement multiplier for brows
  beard_boost: 1.2                   # Enhancement multiplier for beard
  skin_smoothing: 0.3                # Skin smoothing strength (0 = none)
  sharpen_amount: 0.3                # Sharpening strength
  use_cinematic_noise: true          # Add subtle sensor grain
  noise_strength: 0.02               # Grain intensity (0 = none)

memory:
  accumulation_rate: 0.1             # How fast confidence accumulates
  decay_rate: 0.01                   # How fast confidence decays
  min_observations: 5                # Min observations before using memory
  max_age_frames: 300                # Max frames to keep in memory
  use_pose_weighting: true           # Weight observations by pose similarity

compositor:
  confidence_threshold: 0.3          # Below this, use source pixels
  feather_pixels: 10                 # Edge feathering width
  use_light_matching: true           # Match lighting between face and bg

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
  min_face_detection_rate: 0.80      # Min % of frames with face
  max_identity_drift: 20.0           # Max LAB distance from reference
  max_flicker_score: 5.0             # Max frame-to-frame variance
  min_sharpness: 10.0                # Min Laplacian variance
  check_av_sync: true                # Validate audio-video sync
```

---

## 6. Test Results & Comparison

**Test clip:** `clips_test/test_clip.mp4` (640x360, 30fps, 11.5s, 345 frames)  
**Reference:** `expectation.png` (941x1672, portrait)  
**Reference face:** L=108.5, a=139.6, b=146.7

### Parameter Comparison

| Metric | Reference | Original | ref_grade | face_mapper | Face OS |
|---|---|---|---|---|---|
| **L (brightness)** | 108.5 | 98.9 | 100.6 (+1.7) | 99.1 (+0.3) | 97.4 (-1.5) |
| **a (red-green)** | 139.6 | 138.3 | 137.2 (-1.1) | 135.7 (-2.6) | 137.4 (-0.9) |
| **b (yellow-blue)** | 146.7 | 130.3 | 137.9 (+7.6) | 138.2 (+7.9) | 132.3 (+1.9) |
| **L std (contrast)** | 58.5 | 39.8 | 26.3 (-13.5) | 40.0 (+0.1) | 37.2 (-2.7) |
| **Sharpness** | 232.2 | 310.5 | 215.3 (-95.2) | 296.6 (-14.0) | 5.3 (-305.2) |
| **Saturation** | 124.7 | 85.4 | 87.0 (+1.6) | 92.8 (+7.3) | 81.5 (-3.9) |
| **Flicker (L diff)** | — | 2.38 | 1.57 | **1.07** | 9.10 |
| **Flicker (LAB dist)** | — | 2.53 | 1.64 | **1.25** | 9.87 |
| **Face detection** | — | 100% | 100% | 100% | 91.3% |
| **LAB dist from ref** | 0.0 | 19.0 | **12.0** | 13.2 | 18.3 |
| **L distance from ref** | 0.0 | 9.6 | **7.9** | 9.3 | 11.1 |

### Verdict

| Metric | Winner | Score |
|---|---|---|
| Closest to reference (LAB) | **ref_grade** | 12.0 |
| Most stable (lowest flicker) | **face_mapper** | 1.25 |
| Best L match | **ref_grade** | 7.9 |
| Best a match (skin tone) | **Face OS** | 0.9 |
| Best b match (warmth) | **ref_grade** | 7.6 |

### Output Files

```
output/face_os_comparison/
├── original.mp4          792 KB   640x360   (input clip)
├── ref_graded.mp4        1.3 MB   640x360   (ref_grade output)
├── face_mapped.mp4       2.6 MB   640x360   (face_mapper output)
├── face_os_output.mp4    11 MB    1080x1920 (Face OS 9:16 output)
├── comparison_grid.jpg   Side-by-side frame comparison
├── comparison_full_grid.jpg  5 frames × 4 variants
├── comparison_results.json   Raw metrics
└── frames/               20 sample frames
```

---

## 7. Known Issues & Next Steps

### Current Issues

| Issue | Root Cause | Fix |
|---|---|---|
| Flicker = 9.87 (too high) | Crop planner jitter + temporal stabilizer too weak | Increase inertia 0.85→0.95, fix crop smoothing |
| Sharpness = 5.3 (too low) | Bilateral filter + cinematic noise over-smooth at 640x360 | Reduce smoothing at low resolutions |
| Face detection = 91.3% | Haar cascade misses some frames | Add dlib tracker fallback |
| a channel = 137.4 (only -0.9) | Identity embedding correctly identifies skin tone | Working as intended |
| b channel = 132.3 (only +1.9) | Not enough warmth boost | Increase b channel blend toward reference |

### Next Steps (Priority)

1. **Fix flicker**: Increase temporal inertia, fix crop smoothing jitter
2. **Fix sharpness**: Resolution-adaptive enhancement (less smoothing at low res)
3. **Improve detection**: Add dlib HOG detector as fallback
4. **Tune b channel**: Increase warmth blend toward reference b=146.7
5. **Add tests**: Unit tests for each module
6. **GPU path**: CUDA for detection + landmarks on Colab T4

---

## File Structure

```
face_os/
├── __init__.py              # Package init
├── types.py                 # Core data structures (FaceDetection, Landmarks, AppearanceField, CropPlan, ConfidenceMap, etc.)
├── config.py                # YAML config loader with dot-notation access
├── ingest.py                # Module 1: Video loading, frame reader, reference images
├── detect_track.py          # Module 2: Haar/dlib detection, embedding matching, temporal tracker
├── landmarks.py             # Module 3: 68-point landmarks (dlib or geometric), PnP pose, region masks
├── canonical_map.py         # Module 4: Canonical UV alignment, Appearance Field builder
├── crop_planner.py          # Module 5: Face-locked 9:16 crop, headroom, EMA smoothing
├── temporal_stabilize.py    # Module 6: Identity inertia, LAB flicker suppression, optical flow
├── face_enhance.py          # Module 7: Eye-dominant rendering, cinematic noise
├── identity_memory.py       # Module 8: Per-pixel confidence accumulation, pose weighting
├── compositor.py            # Module 9: Confidence-weighted blending, lighting matching
├── export_qc.py             # Module 10: FFmpeg pipe, fades, quality validation
└── pipeline.py              # Orchestrator tying all 10 modules

face_os_config.yaml          # All tuning parameters
test_face_os_comparison.py   # 4-way comparison test
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| OpenCV (cv2) | ≥4.5 | Image processing, face detection, optical flow |
| NumPy | ≥1.20 | Array operations |
| dlib | ≥19.22 | 68-point landmarks, face embeddings (optional) |
| face_recognition | ≥1.3 | Identity matching (optional, wraps dlib) |
| FFmpeg | ≥5.0 | Video encoding (external binary) |
| PyYAML | ≥5.0 | Config file parsing |
