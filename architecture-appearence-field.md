# PERSONAL FACE OS — REFERENCE ARCHITECTURE DOC 😭

## PROJECT NAME

**Persistent Identity Reconstruction Engine (PIRE)**

Alternative internal codename:

* FaceOS
* Photonic Identity Engine
* Appearance Field Runtime
* Identity Inertia Renderer

---

# 0. CORE PHILOSOPHY 🚨

## MOST IMPORTANT RULE

```text
SOURCE VIDEO IS NOT GROUND TRUTH.
SOURCE VIDEO IS TELEMETRY.
```

Source stream only provides:

* pose
* expression
* motion
* lighting
* temporal observations

Final face is reconstructed from:

* identity memory
* temporal consistency
* appearance priors
* accumulated observations

---

## THE MENTAL SHIFT 🧠

OLD (pixel-centric):

```text
identity = image
enhance each frame
```

NEW (belief-centric):

```text
identity = latent appearance manifold
maintain persistent belief about what THIS human looks like
through noisy observations over time
```

THIS IS THE FUNDAMENTAL DIFFERENCE.

---

# 1. NORTH STAR 🎯

Goal is NOT:

* super resolution
* sharpening
* denoise

Goal IS:

```text
PERCEPTUAL CONTINUITY OF A PERSISTENT HUMAN IDENTITY
UNDER DEGRADED OBSERVATION
```

---

# 2. SYSTEM OVERVIEW 🧠

```text
RAW STREAM
    ↓
FACE TRACKING (telemetry extraction)
    ↓
CANONICAL ALIGNMENT (stabilize coordinate space)
    ↓
PATCH EXTRACTION (per-region decomposition)
    ↓
PATCH BELIEF UPDATE (per-patch confidence accumulation)
    ↓
IDENTITY ANCHOR CORRECTION (prevent drift)
    ↓
PATCH DATABASE QUERY (best patch retrieval)
    ↓
CONFIDENCE-WEIGHTED RECONSTRUCTION
    ↓
BIDIRECTIONAL TEMPORAL SOLVE (future repairs past)
    ↓
CINEMATIC POST PROCESS (temporally coherent grain)
    ↓
FINAL 9:16 OUTPUT
```

---

# 3. CORE MODULES ⚙️

# MODULE A — FACE TELEMETRY EXTRACTION

## INPUT

* raw frame

## OUTPUT

```python
telemetry = {
    yaw,
    pitch,
    roll,
    mouth_open,
    blink_left,
    blink_right,
    eye_direction,
    expression_vector,
    lighting_vector,
    face_bbox,
    confidence
}
```

## PURPOSE

Extract ONLY dynamic information.

NOT identity.

---

# MODULE B — CANONICAL ALIGNMENT

## PURPOSE

Convert every frame into:

```text
same face space
same orientation
same coordinate system
```

## METHOD

* landmarks
* mesh alignment
* affine/TPS warp
* UV mapping

## OUTPUT

```python
canonical_face
canonical_uv
```

---

# MODULE C — PATCH BELIEF ENGINE 🔥

## CORE IDEA

Each frame is:

```text
partial noisy observation
```

NOT final truth.

---

## THE PATCH-FIRST APPROACH 🚨

DO NOT THINK:

```text
identity = whole face image
```

THINK:

```text
identity = collection of patch beliefs
```

Each patch has INDEPENDENT dynamics:

```python
patch_belief = {
    'left_eye': {
        best_observation,
        confidence_distribution,
        lighting_model,
        pose_memory,
        temporal_variance,
        observation_count
    },
    'right_eye': { ... },
    'beard': { ... },
    'forehead': { ... },
    'lips': { ... },
    'nose': { ... },
    'left_cheek': { ... },
    'right_cheek': { ... },
    'jaw': { ... }
}
```

---

## WHY PATCH-FIRST? 😭

Because:

* Eyes blink → eye patch needs freeze logic
* Beard is stable → beard patch accumulates fast
* Forehead barely moves → highest confidence
* Lips change expression → need pose-conditioned storage

ONE global memory = WRONG.

PER-PATCH belief = CORRECT.

---

## PATCH MEMORY STRUCTURE

```python
class PatchBelief:
    # Frequency decomposition (per-patch)
    best_low: np.ndarray      # Low freq: EMA (skin tone, lighting)
    best_high: np.ndarray     # High freq: BEST observation only (pores, edges)
    quality_max: np.ndarray   # Quality of best observation

    # Semantic confidence
    confidence: float         # Per-patch confidence (NOT global)
    observation_count: int    # How many observations accumulated
    temporal_variance: float  # How noisy observations are

    # Pose-conditioned storage
    pose_at_best: Tuple[float, float, float]  # Pose when best was captured
    lighting_at_best: np.ndarray               # Lighting when best was captured

    # Independent dynamics
    stability: float          # How stable this patch is (forehead=high, lips=low)
    decay_rate: float         # How fast confidence decays without observations
```

---

## MEMORY STRUCTURE (per patch)

```python
{
    low_frequency,      # skin tone / lighting (EMA)
    high_frequency,     # pores / beard / edges (BEST only)
    confidence,         # per-patch confidence
    best_observation,   # highest quality seen
    temporal_variance,  # observation noise
    lighting_history    # lighting conditions when observed
}
```

---

## IMPORTANT RULE 🚨

DO NOT STORE PURE RGB ONLY.

Separate:

```text
LOW FREQUENCY  = skin tone / lighting
HIGH FREQUENCY = pores / beard / edges
```

---

## PATCH DYNAMICS (INDEPENDENT) 🔥

Each patch has its own:

```python
# Stability (how fast it changes)
forehead_stability = 0.95  # barely moves
beard_stability = 0.90     # very stable
eye_stability = 0.70       # blinks, gaze shifts
lip_stability = 0.60       # expression changes

# Confidence decay (how fast confidence drops without observations)
forehead_decay = 0.001  # very slow decay
beard_decay = 0.002     # slow decay
eye_decay = 0.01        # medium decay (blinks)
lip_decay = 0.02        # fast decay (expression changes)

# Accumulation rate (how fast it learns)
forehead_rate = 0.20   # fast learning (stable)
beard_rate = 0.15      # medium learning
eye_rate = 0.10        # slow learning (noisy)
lip_rate = 0.08        # slowest learning (very noisy)
```

THIS IS THE KEY INNOVATION.

Different face regions have DIFFERENT temporal dynamics.

---

# MODULE D — IDENTITY ANCHOR SYSTEM 👑

## PURPOSE

Prevent:

```text
identity drift
average-face syndrome
```

---

## ANCHOR SET

Store:

* frontal neutral
* frontal smile
* left yaw
* right yaw
* slight up/down
* beard variations
* eyes open/closed

---

## RULE

Every reconstruction must satisfy:

```text
distance(output_identity, anchor_identity) < threshold
```

---

## IDENTITY GRAVITY EQUATION (Formalized) 👑

The anchor correction is formalized as **identity gravity**:

```text
I_t = (1 - λ) * I_t + λ * I_anchor
```

Where λ (lambda) is conditioned on:
- **drift**: higher drift → stronger pull (gravity increases with distance)
- **confidence**: lower confidence → stronger pull (unstable identity needs anchor)
- **observation_count**: fewer observations → stronger pull (new identity needs anchor)

This creates **identity gravity** — the anchor pulls the identity toward it like a gravitational field. The pull is stronger when:
1. Identity has drifted far from anchor (high drift)
2. Identity confidence is low (unstable observations)
3. Few observations accumulated (new or reset identity)

λ is clamped to [0.1, 0.95] to prevent:
- λ=0: anchor has no effect (identity drifts freely)
- λ=1: identity is always anchor (no source influence)

---

## ANCHOR CORRECTION MATH

```python
# Compute λ (identity gravity strength)
if drift > 30:
    lambda_base = 0.85  # Very strong pull for large drift
elif drift > 15:
    lambda_base = 0.60  # Strong pull
elif drift > 5:
    lambda_base = 0.35  # Moderate pull
else:
    lambda_base = 0.15  # Gentle pull (maintenance)

# Modulate by confidence (lower confidence → stronger anchor pull)
obs_count = mean(observation_count)
confidence_factor = 1.0 / (1.0 + obs_count * 0.01)  # Saturates at ~100 obs
lambda_conf = lambda_base * (0.5 + 0.5 * confidence_factor)

# Clamp λ to safe range
lambda_clamped = clip(lambda_conf, 0.1, 0.95)

# Apply identity gravity equation
# I_t = (1 - λ) * I_t + λ * I_anchor
best_low = (1 - lambda_clamped) * best_low + lambda_clamped * anchor_low

# High freq: weaker pull (preserve source detail)
# λ_high = λ * 0.2 (much less than low freq)
lambda_high = lambda_clamped * 0.2
best_high = (1 - lambda_high) * best_high + lambda_high * anchor_high
```

---

# MODULE E — SEMANTIC CONFIDENCE ENGINE ⚡

## CONFIDENCE IS NOT:

```python
confidence = sharpness
```

TOO NAIVE.

---

## REAL CONFIDENCE (per-patch, semantic)

```python
# Per-patch semantic confidence
confidence = {
    'left_eye': f(sharpness, blink_state, gaze_quality, occlusion),
    'right_eye': f(sharpness, blink_state, gaze_quality, occlusion),
    'beard': f(sharpness, pose, lighting, occlusion),
    'forehead': f(sharpness, lighting, hair_occlusion),
    'lips': f(sharpness, expression, occlusion),
    'nose': f(sharpness, pose, lighting),
    'skin': f(sharpness, lighting, compression),
}
```

---

## SEMANTIC CONFIDENCE RULES 🚨

```python
# Eye confidence
if is_blinking:
    eye_confidence = 0.0  # DON'T learn blink frames
elif gaze_away:
    eye_confidence = 0.3  # learn slowly

# Beard confidence
if frontal_pose:
    beard_confidence = 0.9  # learn fast (stable)
elif side_pose:
    beard_confidence = 0.5  # learn slower

# Forehead confidence
forehead_confidence = 0.95  # almost always high (barely moves)

# Lip confidence
if talking:
    lip_confidence = 0.2  # learn very slowly (changes fast)
else:
    lip_confidence = 0.7  # learn normally
```

---

## PURPOSE

Decide PER-PATCH:

```text
trust source?
or trust identity memory?
```

NOT globally — PER REGION.

---

# MODULE F — IDENTITY RECONSTRUCTION 🧬

## CORE EQUATION

```text
FINAL = source * confidence
      + identity_memory * (1 - confidence)
```

BUT:

* patch-wise
* temporally stabilized
* frequency-aware

---

## PATCH-WISE RECONSTRUCTION

```python
# For each patch independently
for patch_name in patches:
    patch_conf = semantic_confidence[patch_name]

    # Frequency-aware blending
    low_final = low_id * patch_conf * low_blend + low_curr * (1 - patch_conf * low_blend)
    high_final = high_id * patch_conf * high_blend + high_curr * (1 - patch_conf * high_blend)

    # Apply anchor correction
    low_final = low_final + (anchor_low - low_final) * anchor_pull

    result_patch = low_final + high_final
```

---

# MODULE G — TEMPORAL INERTIA ENGINE ⏳

## MOST IMPORTANT REALISM RULE

```text
IDENTITY SHOULD CHANGE
SLOWER THAN SOURCE PIXELS
```

Mathematically:

```text
Δ(identity) << Δ(source)
```

---

## BIDIRECTIONAL TEMPORAL SOLVE 🚨

THE OFFLINE PIPELINE'S SUPERPOWER:

```text
future sharp frame repairs past blurry frame
```

Algorithm:

```python
# Forward pass: collect per-frame quality + appearance
for frame in video:
    store(canonical_face, quality_map, pose, sharpness)

# Identify HQ frames
hq_frames = [f for f in frames if f.quality > threshold]

# Backward pass: HQ frames repair past
for frame in frames:
    nearest_hq = find_nearest_hq(frame, direction='both')

    # Weighted by temporal distance
    weight = 1.0 / (1.0 + temporal_distance * 0.1)

    # Pose similarity bonus
    pose_sim = exp(-pose_distance / 30.0)

    # Fuse
    result = (current * current_quality + hq * weight * pose_sim) / total_weight
```

---

## PURPOSE

Prevent:

* flicker
* beard dancing
* pore instability
* eye inconsistency

---

# MODULE H — EYE DOMINANCE SYSTEM 👁️

## HUMAN BRAIN PRIORITY MAP

Highest quality:

* eyes
* eyelids
* eyebrows
* beard contour
* lips

Medium:

* nose
* forehead

Lowest:

* cheeks
* neck

---

## RULE

Allocate compute based on:

```text
perceptual importance
NOT area size
```

---

## EYE PRESERVATION RULES 🚨

```python
# NEVER hallucinate eyelashes
# NEVER hallucinate iris detail
# NEVER hallucinate sclera brightness

# PRESERVE eye structure
# PRESERVE temporal stability (no eye flicker)
# PRESERVE gaze direction

# If identity memory has good eye data, prefer it
if eye_confidence > 0.6:
    result = identity_eyes * 0.7 + source_eyes * 0.3
else:
    result = source_eyes  # don't enhance, just preserve
```

---

# MODULE I — PATCH DATABASE 🗄️

## THE NEXT LEAP 🚨

Instead of:

```text
store pixel observations
```

Store:

```text
patch hypotheses
```

---

## POSE-CONDITIONED PATCH RETRIEVAL (THE REAL SAUCE) 👑

```python
# THE REAL FUCKING SAUCE
query(
    yaw=15,
    expression='smile',
    lighting='warm'
)

# Returns:
# - best beard patch
# - best eye patch
# - best lip patch
```

THIS IS THE REAL BREAKTHROUGH.

---

## PATCH DATABASE STRUCTURE

```python
patch_database = {
    'left_eye': {
        'frontal_neutral_neutral': best_patch,
        'frontal_smile_warm': best_patch,
        'left_15_neutral_cool': best_patch,
        'right_15_smile_neutral': best_patch,
        # ... etc
    },
    'beard': {
        'frontal_neutral_neutral': best_patch,
        'frontal_smile_warm': best_patch,
        'left_15_neutral_cool': best_patch,
        # ... etc
    },
    'lips': {
        'frontal_neutral_neutral': best_patch,
        'frontal_talk_neutral': best_patch,
        'frontal_smile_warm': best_patch,
        # ... etc
    },
    # ... etc
}
```

---

## QUERY LOGIC (Multi-dimensional)

```python
def query_patch(patch_name, yaw, expression, lighting):
    """Find best matching patch from database.

    POSE-CONDITIONED PATCH RETRIEVAL:
      query(yaw=15, expression='smile', lighting='warm')
      → returns best patch for that condition

    Priority:
      1. Composite condition (pose + expression + lighting)
      2. Pose-only condition
      3. Closest pose bin
      4. Overall best patch
    """
    # Create composite condition key
    cond_key = f'{pose_bin(yaw)}_{expression}_{lighting}'

    # Try composite condition first
    if cond_key in patch_database[patch_name]:
        return patch_database[patch_name][cond_key]

    # Try partial match (pose + expression, without lighting)
    partial_key = f'{pose_bin(yaw)}_{expression}_any'
    if partial_key in patch_database[patch_name]:
        return patch_database[patch_name][partial_key] * 0.9

    # Try pose-only condition
    pose_key = pose_bin(yaw)
    for key in patch_database[patch_name]:
        if key.startswith(pose_key):
            return patch_database[patch_name][key] * 0.8

    # Fallback: overall best
    return best_patch
```

---

## WHY THIS MATTERS 😭

Instead of averaging all observations:

```text
pixel = mean(observations)  # wax museum
```

We query the BEST observation for current conditions:

```text
pixel = best_match(current_pose, current_expression)  # real face
```

HUGE DIFFERENCE.

---

# MODULE J — APPEARANCE FIELD (FUTURE PHASE) 🌌

## LONG TERM GOAL

Instead of:

```text
mesh → texture → render
```

Learn:

```text
A(u,v,θ,L,t)
```

Where:

* `(u,v)` = canonical coordinates
* `θ` = expression/pose
* `L` = lighting
* `t` = temporal memory state

Outputs:

* color
* normals
* microdetail
* reflectance
* dynamic texture behavior

---

# MODULE K — DYNAMIC UV FLOW 🌀

## FUTURE PHASE

Skin is NOT rigid.

Need:

```text
expression-dependent pore movement
skin stretching
beard directional deformation
```

Equation:

```text
(u',v') = Φ(u,v,θ)
```

---

# MODULE L — CINEMATIC REALISM 🎥

## PERFECT CLEAN OUTPUT = FAKE

Need:

* subtle grain
* sensor noise
* micro shimmer
* tiny temporal randomness

---

## TEMPORALLY COHERENT GRAIN 🚨

DO NOT:

```python
# Independent random grain per frame = micro flicker
noise = randn(h, w) * strength
```

DO:

```python
# Temporally coherent grain = cinematic
# Use low-frequency noise that evolves slowly
noise_t = base_noise * (1 - alpha) + new_noise * alpha
# alpha = 0.05 (very slow evolution)
```

---

## TEMPORAL NOISE FIELD IMPLEMENTATION

```python
class TemporalNoiseField:
    """Temporally coherent sensor grain field.

    The problem with independent random noise per frame:
      frame 1: noise_1
      frame 2: noise_2  (completely different)
      → micro shimmer flicker (brain detects as fake)

    The solution: temporally coherent noise field
      - Base noise field persists across frames
      - Slowly evolves over time (low-frequency temporal drift)
      - Sensor-pattern persistence (same hot pixels, same grain structure)
      - Like real camera sensor noise: consistent pattern, slow drift
    """

    def __init__(self, h, w, alpha=0.05):
        self.alpha = alpha  # Evolution rate (0=static, 1=independent)
        self._base_noise = randn(h, w)  # Persists across frames
        self._sensor_pattern = self._generate_sensor_pattern()
        self._drift_noise = randn(h, w) * 0.1  # Slow temporal drift

    def _generate_sensor_pattern(self):
        """Generate persistent sensor pattern (like real camera).

        Real sensors have:
        - Hot pixels (always bright)
        - Column/row noise (readout pattern)
        - Fixed pattern noise (manufacturing defects)
        """
        pattern = zeros(h, w)
        # Hot pixels (sparse, persistent)
        pattern[hot_y, hot_x] = randn(num_hot) * 0.5
        # Column noise (readout pattern)
        pattern += col_noise
        return pattern

    def get_noise(self, strength=0.015):
        """Get temporally coherent noise for current frame."""
        new_noise = randn(h, w)

        # Evolve base noise slowly (temporal coherence)
        # noise_t = base * (1-α) + new * α
        self._base_noise = self._base_noise * (1 - self.alpha) + new_noise * self.alpha

        # Combine: base noise + sensor pattern + temporal drift
        noise = (
            self._base_noise * 0.7 +           # Main noise (temporally coherent)
            self._sensor_pattern * 0.2 +         # Persistent sensor pattern
            self._drift_noise * 0.1              # Slow temporal drift
        )

        # Correlate slightly (mimics sensor readout)
        noise = GaussianBlur(noise, (3, 3), 0.5)

        return noise * strength * 255
```

---

## RULE

Noise MUST:

* vary spatially
* stay statistically consistent
* evolve temporally (NOT independent per frame)
* include sensor-pattern persistence (hot pixels, column noise)

---

# 4. EDGE CASES 💀

# EDGE CASE 1 — FAST HEAD TURN

## PROBLEM

Memory alignment breaks.

## FIX

* reduce memory influence
* trust source more
* temporary fallback mode

---

# EDGE CASE 2 — MOTION BLUR

## PROBLEM

Buffer learns blur.

## FIX

Reject update:

```python
if blur > threshold:
    skip_memory_update()
```

---

# EDGE CASE 3 — FACE OCCLUSION

Examples:

* hand on face
* mic
* glasses reflection

## FIX

* occlusion masks
* partial patch freeze

---

# EDGE CASE 4 — LIGHTING CHANGE

## PROBLEM

Memory becomes inconsistent.

## FIX

Store:

```python
lighting-conditioned memory
```

---

# EDGE CASE 5 — EXTREME EXPRESSIONS

## PROBLEM

Identity deformation.

## FIX

Expression manifold constraints.

---

# EDGE CASE 6 — EYE FAILURE 🚨

MOST IMPORTANT FAILURE.

Even tiny eye artifact:

```text
= uncanny valley
```

## FIX

* minimal hallucination
* maximum temporal stability
* patch-level freeze during blinks

---

# EDGE CASE 7 — COMPRESSION BLOCKING

## PROBLEM

Memory learns artifacts.

## FIX

Pre-clean:

* deblocking
* chroma cleanup
* ringing suppression

BEFORE memory update.

---

# EDGE CASE 8 — LONG STREAM DRIFT

## PROBLEM

Identity slowly mutates.

## FIX

Periodic:

```text
anchor re-projection
```

---

# EDGE CASE 9 — ASYMMETRIC LIGHTING

## PROBLEM

One side face over-enhanced.

## FIX

Per-region lighting estimation.

---

# EDGE CASE 10 — LOW CONFIDENCE CASCADE

## PROBLEM

Entire frame bad.

## FIX

Fallback hierarchy:

```text
source > anchor > memory > render
```

---

# 5. DEVELOPMENT PHASES 🚀

# PHASE 1 — MVP ✅

Build:

* face tracking
* canonical alignment
* memory buffer
* confidence blending

GOAL:

```text
prove temporal accumulation works
```

STATUS: DONE

---

# PHASE 2 ✅

Add:

* patch memory
* eye priority
* anchor correction
* high/low frequency split

STATUS: DONE

---

# PHASE 3 ✅

Add:

* best observation cache
* bidirectional temporal solve
* lighting conditioning

STATUS: DONE

---

# PHASE 4 — COMPLETE ✅

Add:

* patch database (pose-conditioned)
* semantic confidence (per-patch)
* identity hypotheses (not just observations)
* temporally coherent grain

STATUS: DONE

---

# PHASE 5 — CURRENT 🔄

Add:

* appearance field
* dynamic UV flow
* microdetail synthesis

STATUS: IN PROGRESS

---

# PHASE 6 — FAR FUTURE

Add:

* personalized neural codec
* full identity operating system

---

# 6. FAILURE CONDITIONS ❌

System FAILS if:

* face becomes too smooth
* eyes unstable
* beard flickers
* temporal consistency breaks
* identity drifts
* pores hallucinate randomly
* output looks "AI clean"

---

# 7. SUCCESS CONDITIONS 👑

System WINS if:

* viewer stops noticing enhancement
* identity feels persistent
* eyes feel alive
* beard stable
* lighting believable
* output feels like expensive camera

---

# 8. IMPLEMENTATION STATUS 📊

## What's Actually Built

| Module | Status | Notes |
|---|---|---|
| A: Telemetry | ✅ Done | Haar Cascade + dlib landmarks |
| B: Canonical | ✅ Done | Similarity transform, 256x256 atlas |
| C: Patch Belief | ✅ Done | Frequency decomposition, per-patch dynamics |
| D: Anchor | ✅ Done | **Identity gravity equation** — I_t = (1-λ)I_t + λI_anchor |
| E: Confidence | ✅ Done | Semantic confidence, multifactor, quality modulation |
| F: Reconstruction | ✅ Done | Frequency-aware blending, anchor correction |
| G: Temporal | ✅ Done | Bidirectional solver, HQ frame identification |
| H: Eye Dominance | ⚠️ Partial | Structure-preserving rendering, blink detection TODO |
| I: Patch Database | ✅ Done | **Pose-conditioned retrieval** — query(yaw, expression, lighting) |
| J: Appearance Field | 🔄 Phase 5 | Hybrid system |
| K: Dynamic UV | 🔄 Phase 5 | Skin stretch model |
| L: Cinematic | ✅ Done | **Temporally coherent grain** — noise field with sensor persistence |

## Phase Status

| Phase | Status | Items |
|---|---|---|
| Phase 1 (MVP) | ✅ Done | Face tracking, canonical alignment, memory buffer |
| Phase 2 | ✅ Done | Patch memory, eye priority, anchor correction |
| Phase 3 | ✅ Done | Best observation cache, bidirectional solve |
| Phase 4 | ✅ Done | Patch DB, semantic confidence, identity hypotheses, temporal grain |
| Phase 5 | 🔄 Current | Appearance field, dynamic UV flow, microdetail synthesis |
| Phase 6 | ❌ Future | Personalized neural codec |

---

## Current Test Results

| Metric | Reference | Output | Status |
|---|---|---|---|
| L (brightness) | 108.4 | 101.4 | ⚠️ Δ7.0 (was Δ37!) |
| a (skin tone) | 139.6 | 139.2 | ✅ Δ0.4 (PERFECT) |
| b (warmth) | 146.7 | 141.6 | ⚠️ Δ5.1 (was Δ8.4) |
| Face detection | — | 100% | ✅ |
| Flicker | — | 0.22 | ✅ |
| Anchor distance | — | 0.9 LAB | ✅ |
| LAB distance | — | 8.6 | ✅ (was 36.7!) |
| Tests | — | 51/51 | ✅ |

---

## Key Fixes Applied

| Fix | Impact |
|---|---|
| Compositor was undoing anchor correction | L 72→99 (+27 points!) |
| Pre-populate identity from reference (50 obs) | Confidence 0.09→0.33 |
| Don't reset identity between clips | Preserves anchor + observations |
| Increase anchor pull: 0.6→0.85 | Stronger correction for large drift |

---

## Known Issues

| Issue | Root Cause | Fix |
|---|---|---|
| Face L still 9.4 dark | Source blending with low confidence | Increase low-freq blend toward identity |
| b channel 5.2 cold | Source b=128 vs ref b=147 | Increase b anchor correction |
| Temporal grain | Independent random noise | Implement coherent grain |

---

# 9. FINAL PHILOSOPHY 😭

```text
DO NOT ENHANCE PIXELS.

INFER THE MOST PLAUSIBLE
PERSISTENT VERSION
OF THIS HUMAN OVER TIME.
```

---

# 10. THE REAL MOAT 🚨

The REAL competitive advantage is:

```text
persistent identity coherence
```

NOT:

```text
fake 4K pores
```

Anyone can sharpen.

Nobody else maintains:

```text
temporal belief about what THIS person looks like
across degraded observations
over time
```

THIS IS THE MOAT.
