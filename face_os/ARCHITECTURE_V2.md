# Face OS — Architecture V2

## What Is This?

A **Personal Face Operating System** — a face-aware vertical video pipeline
that treats identity as a persistent appearance field, not a per-frame
rendering problem.

**Core philosophy: Overfit is the feature.**

Generic models serve everyone. This serves ONE face, ONE environment,
ONE camera, ONE motion pattern. The probability P(your_setup) → 1.

---

## Architecture Overview

```
INPUT: 16:9 source video + reference face images
                    │
    ┌───────────────┼───────────────┐
    │               │               │
    ▼               ▼               ▼
┌─────────┐  ┌───────────┐  ┌──────────────┐
│ Ingest  │  │ Detection │  │  Landmarks   │
│ + Sync  │  │ + Track   │  │ + Pose       │
└────┬────┘  └─────┬─────┘  └──────┬───────┘
     │             │               │
     └──────┬──────┴───────────────┘
            │
            ▼
    ┌───────────────┐
    │  Canonical    │ ◄── Appearance Field A(u,v,θ,L,t)
    │  Face Map     │     (Photic Memory accumulation)
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │  Crop Planner │ ◄── Face-locked 9:16 with headroom
    │  + Headroom   │     (protect forehead, allow bottom crop)
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │  Temporal     │ ◄── Identity Inertia
    │  Stabilizer   │     (ΔI_identity ≪ ΔI_source)
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │  Face Region  │ ◄── Eye Dominance Rendering
    │  Enhancer     │     (eyes > brows > beard > skin > bg)
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │  Identity     │ ◄── Photic Memory
    │  Memory Atlas │     (accumulate confidence over time)
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │  Confidence   │ ◄── Per-pixel confidence blending
    │  Compositor   │     (high conf = memory, low conf = source)
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │  Export + QC  │ ◄── Quality validation
    │               │     (face rate, drift, flicker, sharpness)
    └───────────────┘
            │
            ▼
OUTPUT: 9:16 stabilized video
```

---

## The 10 Modules

### 1. `ingest.py` — Video Ingestion
- Load video + extract metadata (dimensions, fps, codec)
- Frame-by-frame reading with seeking
- Reference image loading
- A/V sync validation

### 2. `detect_track.py` — Face Detection + Tracking
- Detect sparsely (every N frames), track densely (every frame)
- Match to target identity via face embeddings (dlib)
- Maintain persistent tracks across occlusions
- EMA-smoothed bounding boxes

### 3. `landmarks.py` — Landmarks + Head Pose
- 68-point facial landmarks (dlib or geometric fallback)
- Head pose estimation (yaw, pitch, roll via PnP)
- Per-region mask generation (eyes, brows, nose, mouth, skin)

### 4. `canonical_map.py` — Canonical Face Mapping + Appearance Field
- Align face to canonical UV space (frontal, neutral)
- **Appearance Field**: A(u,v,θ,L,t) — dynamic appearance function
- **Photic Memory**: accumulate pixel observations over time
- Identity residual: what makes THIS face unique

### 5. `crop_planner.py` — Face-Aware Crop with Headroom
- 16:9 → 9:16 crop that follows the face
- Face at ~30% from top (matching reference composition)
- Protect forehead, allow bottom crop
- EMA smoothing + velocity clamping

### 6. `temporal_stabilize.py` — Temporal Stabilizer (Identity Inertia)
- Track frame-to-frame LAB statistics
- Suppress flicker (sudden L/a/b shifts)
- Motion-compensated: reduce stabilization during real movement
- Core rule: ΔI_identity ≪ ΔI_source

### 7. `face_enhance.py` — Eye-Dominant Face Enhancement
- Eyes: 1.5x boost (sharpen + brighten sclera)
- Brows: 1.3x boost (contrast)
- Beard: 1.2x boost (texture detail)
- Skin: gentle bilateral smoothing
- Background: vignette
- **Cinematic noise**: subtle grain for realism

### 8. `identity_memory.py` — Photic Memory Atlas
- Per-pixel confidence accumulation
- Each frame = noisy photon observation
- Confidence grows with more observations
- Pose-weighted: prefer frontal for atlas building
- Memory decays without new observations

### 9. `compositor.py` — Confidence-Weighted Compositing
- High confidence → use accumulated memory (stable, clean)
- Low confidence → use current frame (noisy but authentic)
- Feathered edge blending
- Lighting matching between face and background

### 10. `export_qc.py` — Export + Quality Control
- FFmpeg pipe encoding (no intermediate files)
- Audio muxing + fade in/out
- QC checks: face rate, identity drift, flicker, sharpness, A/V sync

---

## Key Concepts

### Appearance Field (Not Renderer)

Traditional approach: mesh → texture → lighting → render
Problem: uncanny, rigid, CG feel

**Appearance Field approach:**
A(u,v,θ,L,t) → color, normals, microdetail, reflectance, temporal memory

Face is not an object. Face is a **dynamic appearance function**.

### Dynamic UV Flow

UV coordinates deform with expression:
(u',v') = Φ(u,v,θ)

Pores move. Skin stretches. Beard density changes with angle.
Static UV atlas eventually looks fake. Dynamic UV flow = cinematic realism.

### Identity Residual Space

I = I_base + R_identity

- I_base = generic face physics
- R_identity = YOUR exact beard flow, pore layout, asymmetry

Model stays lightweight, stable, controllable.

### Photic Memory

Video pixels are NOT RGB. They are **noisy photon observations**.

Over time:
- Confidence accumulates
- Hidden appearance refines
- Lighting response improves

Your face engine **evolves** with every stream.

### Identity Inertia

ΔI_identity ≪ ΔI_source

Source video fluctuates wildly. Identity stays stable.
This is why humans look real in video.

### Eye Dominance

Human brain is obsessed with eyes. So:
- Eyes always highest quality
- Sclera stable, iris sharp, eyelid motion coherent
- Even if cheeks are medium quality, brain marks whole frame as "premium"

### Intentional Cinematic Noise

Perfect clean renders look fake. Tiny:
- Sensor grain
- Compression-like micro variation
- Subtle skin shimmer

Brain says: "real camera footage"

---

## Usage

```python
from face_os.pipeline import FaceOSPipeline

# Enroll (once)
pipeline = FaceOSPipeline()
pipeline.enroll("expectation.png", reference_dir="photos/")

# Process (per video)
pipeline.process("input/video.mp4", "output/shorts/clip1.mp4")
```

CLI:
```bash
python -m face_os.pipeline --video input/video.mp4 --reference expectation.png
```

---

## File Structure

```
face_os/
├── __init__.py
├── types.py              # Core data structures
├── config.py             # Configuration loader
├── ingest.py             # Module 1: Video ingestion
├── detect_track.py       # Module 2: Face detection + tracking
├── landmarks.py          # Module 3: Landmarks + pose
├── canonical_map.py      # Module 4: Canonical mapping + appearance field
├── crop_planner.py       # Module 5: Crop planning with headroom
├── temporal_stabilize.py # Module 6: Temporal stabilization
├── face_enhance.py       # Module 7: Face enhancement
├── identity_memory.py    # Module 8: Identity memory atlas
├── compositor.py         # Module 9: Confidence-weighted compositing
├── export_qc.py          # Module 10: Export + QC
└── pipeline.py           # Orchestrator

face_os_config.yaml       # Configuration file
```

---

## What Makes This Different

| Aspect | Generic AI | Face OS |
|--------|-----------|---------|
| Target | Everyone | ONE person |
| Model | Average | Overfitted |
| Approach | Per-frame | Temporal memory |
| Face | Rendered | Appearance field |
| Quality | Uniform | Eye-dominant |
| Noise | Clean (fake) | Cinematic (real) |
| Identity | Varies | Inertial |

---

## Dependencies

- OpenCV (cv2)
- NumPy
- dlib (optional, for 68-point landmarks)
- face_recognition (optional, for identity matching)
- FFmpeg (for video encoding)
- PyYAML (for config)
