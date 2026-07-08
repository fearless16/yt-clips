"""
ref_grade.py — Reference-Derived Color Grade.

Apple Face ID-style enrollment:
  Phase 1 — ENROLLMENT (once, ~0.3s):
    Analyze reference photo → extract all enhancement parameters
    No source video needed

  Phase 2 — INFERENCE (per frame, ~2ms for 1080p):
    Apply the same fixed parameters to every pixel of every frame
    No face detection, no per-frame analysis = zero flicker

Architecture:
  The enrollment extracts CONSTANTS from the reference — contrast ratio,
  color temperature, saturation multiplier, skin tone target, split-tone
  colors, vignette strength.  These are not frame-dependent.

  The inference applies these constants using the same proven algorithms
  from face_mapper.py, but as pure 1D LUTs and fixed multipliers.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import time

from utils.config import load_config
from utils.face_detect import detect_face
from utils.logger import get_logger

cfg = load_config()
log = get_logger("ref_grade", cfg["logging"]["log_file"], cfg["logging"]["level"])


# ─── Phase 1: ENROLLMENT — Extract fixed params from reference ──────────

def _detect_face_once(img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    face = detect_face(img, score_threshold=0.5)
    if face is None:
        return None
    return tuple(int(v) for v in face)


def enroll(reference_path: str = "expectation.png") -> Dict:
    """Analyze reference photo and extract ALL color grading parameters.

    Returns a dict of CONSTANTS (not frame-dependent).
    These constants encode the reference's COLOR STYLE as mathematical values.

    Approach: TARGET-BASED BLENDING — moves source TOWARD reference, not beyond it.
    Includes full-body lighting: body brightness boost + background darkening.
    """
    img = cv2.imread(reference_path)
    if img is None:
        raise ValueError(f"Cannot read reference: {reference_path}")

    h, w = img.shape[:2]
    log.info("Enrolling: %s (%dx%d)", reference_path, w, h)

    bbox = _detect_face_once(img)
    if bbox is None:
        log.error("No face detected")
        raise ValueError("No face detected in reference")

    fx, fy, fw, fh = bbox
    face = img[fy:fy+fh, fx:fx+fw]
    face_lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    full_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    full_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # ── 1. Face brightness ──────────────────────────────────────────────
    ref_L = float(np.mean(face_lab[:, :, 0]))
    log.info("  Face brightness: L=%.1f", ref_L)

    # ── 2. Body brightness (below face) ─────────────────────────────────
    body_y1 = fy + fh
    body_y2 = min(h, int(body_y1 + fh * 1.5))
    body_x1 = max(0, fx - fw // 2)
    body_x2 = min(w, fx + fw + fw // 2)
    body = img[body_y1:body_y2, body_x1:body_x2]
    body_L = float(np.mean(cv2.cvtColor(body, cv2.COLOR_BGR2LAB)[:, :, 0])) if body.size > 0 else ref_L
    body_boost = body_L - ref_L  # How much brighter body is than face
    log.info("  Body brightness: L=%.1f (boost=%+.1f)", body_L, body_boost)

    # ── 3. Background darkness ──────────────────────────────────────────
    bg_mask = np.ones((h, w), dtype=np.uint8) * 255
    bg_mask[fy:fy+fh, fx:fx+fw] = 0
    bg_mask[body_y1:body_y2, body_x1:body_x2] = 0
    bg_pixels = img[bg_mask > 0]
    if bg_pixels.size > 0:
        bg_lab = cv2.cvtColor(bg_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).astype(np.float32)
        bg_L = float(np.mean(bg_lab[:, 0]))
    else:
        bg_L = 40.0
    bg_darken = ref_L - bg_L  # How much darker background is than face
    log.info("  Background brightness: L=%.1f (darken=%.1f)", bg_L, bg_darken)

    # ── 4. Skin tone target (a, b means of face) ────────────────────────
    a_target = float(np.mean(face_lab[:, :, 1]))
    b_target = float(np.mean(face_lab[:, :, 2]))
    log.info("  Skin: a=%.1f b=%.1f", a_target, b_target)

    # ── 5. Contrast (face std) ──────────────────────────────────────────
    ref_contrast = float(np.std(face_gray))
    log.info("  Contrast: std=%.1f", ref_contrast)

    # ── 6. Saturation (face HSV) ────────────────────────────────────────
    face_hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    ref_sat = float(np.mean(face_hsv[:, :, 1]))
    log.info("  Saturation: ref=%.1f", ref_sat)

    # ── 7. Split tone (shadow/highlight colors from full frame) ─────────
    low = full_gray < 51
    high = full_gray > 204
    shadow_color = np.mean(img[low], axis=0) if low.any() else np.zeros(3)
    highlight_color = np.mean(img[high], axis=0) if high.any() else np.full(3, 255)

    # ── 8. Vignette (center/edge ratio) ─────────────────────────────────
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    maxd = np.sqrt(cx**2 + cy**2)
    center = full_gray[dist < maxd * 0.3].mean()
    edge = full_gray[dist > maxd * 0.85].mean()
    vignette_ratio = float(np.clip(center / max(edge, 1), 1.0, 2.0))
    log.info("  Vignette: %.2f", vignette_ratio)

    # ── 9. Face position (normalized) ───────────────────────────────────
    face_y_norm = fy / h  # Where face starts (0=top, 1=bottom)
    face_h_norm = fh / h  # Face height as fraction of frame
    log.info("  Face position: y=%.2f h=%.2f", face_y_norm, face_h_norm)

    params = {
        "ref_L": ref_L,
        "body_L": body_L,
        "bg_L": bg_L,
        "a_target": a_target,
        "b_target": b_target,
        "ref_contrast": ref_contrast,
        "ref_sat": ref_sat,
        "shadow_color": shadow_color.tolist(),
        "highlight_color": highlight_color.tolist(),
        "vignette_ratio": vignette_ratio,
        "face_y_norm": face_y_norm,
        "face_h_norm": face_h_norm,
    }

    # ── Pre-build LUTs (computed once during enrollment) ────────────────
    # TARGET-BASED: blend toward reference, don't multiply

    # Brightness: uniform per-pixel blend toward ref_L
    # 70% blend + max_shift=30 allows dark frames to catch up to reference
    params["_ref_L"] = ref_L
    params["_L_blend"] = 0.70

    # Body boost: how much brighter body should be (capped at 50 L)
    # Reference has +66.3 body boost; 50 is aggressive but safe
    params["_body_boost"] = min(max(body_boost, 0), 50)
    params["_bg_darken"] = min(max(bg_darken, 0), 50)

    # Face position for spatial mask
    params["_face_y_norm"] = face_y_norm
    params["_face_h_norm"] = face_h_norm

    # Contrast: light — preserve L gains from blend step
    # ref_contrast / 45.0 with cap 1.30
    params["_contrast_ratio"] = max(1.0, min(ref_contrast / 45.0, 1.30))

    # a,b LUTs: blend toward reference target
    blend = 0.45  # Stronger blend for better skin matching
    x = np.arange(256, dtype=np.float32)
    params["_lut_a"] = np.clip(x * (1.0 - blend) + a_target * blend, 0, 255).astype(np.uint8)
    params["_lut_b"] = np.clip(x * (1.0 - blend) + b_target * blend, 0, 255).astype(np.uint8)

    # Split tone: moderate
    ss = 0.06
    hs = 0.04
    params["_shadow_strength"] = ss
    params["_highlight_strength"] = hs
    lut_shadow = np.clip((128.0 - x) / 128.0, 0, 1).astype(np.float32)
    lut_highlight = np.clip((x - 128.0) / 128.0, 0, 1).astype(np.float32)
    params["_split_lut"] = (
        np.array(shadow_color)[:, None] * lut_shadow * ss
        + np.array(highlight_color)[:, None] * lut_highlight * hs
    ).T.astype(np.float32)

    log.info("Enrollment complete: %d params extracted", len(params))
    return params


# ─── Phase 2: INFERENCE — Apply grade to any frame ──────────────────────

# Cache for vignette masks: keyed by (h, w, vr_int)
_vignette_cache: Dict[tuple, np.ndarray] = {}

# Cache for body lighting masks: keyed by (h, w, face_y, face_h, body_boost, bg_darken)
_body_mask_cache: Dict[tuple, np.ndarray] = {}


def _get_vignette(h: int, w: int, ratio: float) -> np.ndarray:
    """Cached vignette mask — computed once per resolution."""
    key = (h, w, round(ratio * 100))
    if key not in _vignette_cache:
        Y, X = np.ogrid[:h, :w]
        cx, cy = w / 2.0, h / 2.0
        d = np.sqrt((X - cx)**2 + (Y - cy)**2)
        vig = 1.0 - (d / np.sqrt(cx**2 + cy**2)) * (1.0 - 1.0 / ratio) * 0.5
        _vignette_cache[key] = vig
    return _vignette_cache[key]


def _get_body_mask(h: int, w: int, face_y_norm: float, face_h_norm: float,
                   body_boost: float, bg_darken: float) -> np.ndarray:
    """Cached body brightness / background darkening mask.

    Returns a float32 mask of shape (h, w) with L offsets:
    - Top region (background): negative (darken)
    - Middle region (face): near zero (neutral)
    - Bottom region (body): positive (brighten)
    """
    key = (h, w, round(face_y_norm * 100), round(face_h_norm * 100),
           round(body_boost), round(bg_darken))
    if key not in _body_mask_cache:
        mask = np.zeros((h, w), dtype=np.float32)

        # Face region: y from face_y_norm to face_y_norm + face_h_norm
        face_top = int(face_y_norm * h)
        face_bot = int((face_y_norm + face_h_norm) * h)

        # Background (top): darken gradient from top to face
        if face_top > 0 and bg_darken > 0:
            for y in range(face_top):
                # Stronger darken at top, fading toward face
                t = y / face_top  # 0 at top, 1 at face
                mask[y, :] = -bg_darken * (1.0 - t) * 0.80

        # Body (bottom): boost gradient from face to bottom
        if face_bot < h and body_boost > 0:
            for y in range(face_bot, h):
                t = (y - face_bot) / max(h - face_bot, 1)  # 0 at face, 1 at bottom
                mask[y, :] = body_boost * t * 0.75

        # Smooth the mask to avoid harsh transitions
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=w * 0.05, sigmaY=h * 0.03)

        _body_mask_cache[key] = mask
    return _body_mask_cache[key]


def _detect_logo_regions(frame: np.ndarray) -> np.ndarray | None:
    """Detect logo region using known export.py position.

    export.py places the logo at a fixed position:
      overlay=W-w-30:H-h-280, scaled to 200px wide
    For 1080x1920 output: x=850, y=1440, size=200x200

    For other resolutions, scale proportionally.
    Returns a boolean mask (True = logo pixel) or None if no logo.
    """
    h, w = frame.shape[:2]

    # Only detect logos on portrait frames (exported shorts)
    # Landscape frames (raw clips) don't have logos yet
    if w > h:
        return None

    # Scale logo position proportionally from 1080x1920 reference (LEFT side)
    logo_w = int(200 * w / 1080)
    logo_h = int(200 * h / 1920)
    logo_x = int(30 * w / 1080)
    logo_y = h - logo_h - int(280 * h / 1920)

    # Clamp to frame bounds
    logo_x = max(0, logo_x)
    logo_y = max(0, logo_y)
    logo_w = min(logo_w, w - logo_x)
    logo_h = min(logo_h, h - logo_y)

    if logo_w < 10 or logo_h < 10:
        return None

    logo_mask = np.zeros((h, w), dtype=bool)
    logo_mask[logo_y:logo_y+logo_h, logo_x:logo_x+logo_w] = True
    return logo_mask


def apply_grade(frame: np.ndarray, params: Dict) -> np.ndarray:
    """Apply the reference grade to a single BGR frame.

    TARGET-BASED BLENDING: moves source TOWARD reference, not beyond it.
    Full-body lighting: spatial L adjustment for body boost + background darken.
    Logo preservation: high-contrast rectangular regions (logos) are excluded.
    2 cvtColor calls (BGR→LAB, LAB→BGR).  No HSV conversion.
    """
    if "_contrast_ratio" not in params:
        return frame

    # ── 0. Detect & preserve logo regions ───────────────────────────────
    # Logos are high-contrast, saturated rectangular overlays that don't
    # belong to the video content. Grading them changes their appearance.
    # Strategy: detect logo-like regions, grade the frame, then restore.
    logo_mask = _detect_logo_regions(frame)

    # Detect fade frames BEFORE any processing — check BGR directly
    # LAB conversion of near-black BGR produces L≈15-20, too high for detection
    # Use percentile-based check: fade frames have >95% pixels below threshold
    src_bgr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dark_pct = float(np.mean(src_bgr_gray < 15)) * 100
    is_fade_frame = dark_pct > 95.0
    if is_fade_frame:
        return frame  # Preserve black fade frames unchanged

    # ── 1. BGR → LAB ────────────────────────────────────────────────────
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # ── 2. Extract L channel ────────────────────────────────────────────
    L = lab[:, :, 0].astype(np.float32)

    # ── 3. Brightness shift toward reference (uniform, clamped) ─────────
    ref_L = params["_ref_L"]
    blend = params["_L_blend"]
    # Per-pixel blend with clamping: max shift per frame to prevent flicker
    shift = (ref_L - L) * blend
    max_shift = 30.0  # Max L shift per frame (allows dark frames to catch up)
    shift = np.clip(shift, -max_shift, max_shift)
    L = np.clip(L + shift, 0, 255)

    # ── 4. Contrast on L (centered on post-blend frame mean, not ref_L) ─
    r = params["_contrast_ratio"]
    if r > 1.0:
        # Center on post-blend mean to avoid shifting brightness
        # (ref_L centering was wrong: frame hasn't reached ref_L yet,
        #  so centering on it pushes everything darker)
        frame_mean = float(np.mean(L))
        L = np.clip((L - frame_mean) * r + frame_mean, 0, 255)

    # ── 4b. Body lighting: spatial L adjustment ─────────────────────────
    body_boost = params.get("_body_boost", 0)
    bg_darken = params.get("_bg_darken", 0)
    if body_boost > 0 or bg_darken > 0:
        h, w = frame.shape[:2]
        body_mask = _get_body_mask(
            h, w,
            params.get("_face_y_norm", 0.25),
            params.get("_face_h_norm", 0.35),
            body_boost, bg_darken,
        )
        L = np.clip(L + body_mask, 0, 255)

    # ── 5. a,b transform via LUT (blend toward target) ──────────────────
    lab[:, :, 1] = cv2.LUT(lab[:, :, 1], params["_lut_a"])
    lab[:, :, 2] = cv2.LUT(lab[:, :, 2], params["_lut_b"])
    lab[:, :, 0] = np.clip(L, 0, 255).astype(np.uint8)

    # ── 6. LAB → BGR ────────────────────────────────────────────────────
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32)

    # ── 7. Split tone via L-indexed LUT ────────────────────────────────
    result += params["_split_lut"][L.astype(np.int32)]

    # ── 8. Vignette (cached) ────────────────────────────────────────────
    h, w = frame.shape[:2]
    vig = _get_vignette(h, w, params["vignette_ratio"])
    result *= vig[:, :, None]

    # ── 9. Restore logo regions (preserve original pixels) ──────────────
    if logo_mask is not None:
        result = np.where(logo_mask[:, :, None], frame.astype(np.float32), result)

    return np.clip(result, 0, 255).astype(np.uint8)


# ─── Convenience class ──────────────────────────────────────────────────

class ReferenceGrade:
    """Enroll once from reference, apply to any frame.

    Usage:
        grade = ReferenceGrade("expectation.png")
        frame = grade.apply(frame)    # for each frame
    """

    def __init__(self, reference_path: str = "expectation.png"):
        self.valid = False
        self.params: Dict = {}
        t0 = time.perf_counter()
        try:
            self.params = enroll(reference_path)
            self.valid = True
        except (ValueError, AttributeError, OSError) as e:
            log.error("Enrollment failed: %s", e)
        log.info("Total enrollment: %.1fs", time.perf_counter() - t0)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.valid:
            return frame
        return apply_grade(frame, self.params)


def grade_video(source_path: str, reference_path: str, output_path: str) -> str:
    """Full pipeline: enroll once, pipe graded frames to ffmpeg.

    Uses ffmpeg stdin pipe (rawvideo) — no intermediate JPEG writes.
    Optimized for T4 GPU: /tmp/ temp file for output, then atomic rename.
    """
    t_start = time.perf_counter()
    grade = ReferenceGrade(reference_path)
    if not grade.valid:
        return source_path

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        return source_path

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log.info("Video: %dx%d @ %.1ffps (%d frames)", w, h, fps, total)

    import subprocess
    import shutil
    import tempfile
    tmp_out = os.path.join(tempfile.gettempdir(), f"yt_clips_grade_{os.getpid()}.mp4")

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
        "-i", source_path,
        "-map", "0:v", "-map", "1:a?",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
        "-shortest", "-movflags", "+faststart",
        tmp_out,
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    fi = 0
    tp = time.perf_counter()
    try:
        while True:
            ret, f = cap.read()
            if not ret:
                break
            g = grade.apply(f)
            proc.stdin.write(g.tobytes())
            fi += 1
            if fi % 100 == 0:
                elapsed = max(time.perf_counter() - tp, 0.001)
                log.info("  %d/%d (%.0f fps)", fi, total, fi / elapsed)
    except BrokenPipeError:
        log.error("ffmpeg pipe broken — check output path and codec support")
    except Exception:
        raise
    finally:
        cap.release()
        proc.stdin.close()
        proc.wait()

    if proc.returncode != 0:
        log.error("ffmpeg failed (code %d)", proc.returncode)
        if os.path.exists(tmp_out):
            os.remove(tmp_out)
        return source_path

    shutil.move(tmp_out, output_path)

    if Path(output_path).exists():
        log.info("Done: %s (%.1f MB, %.1fs)", output_path,
                 Path(output_path).stat().st_size / 1e6, time.perf_counter() - t_start)
    return output_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Reference-Derived Color Grade")
    p.add_argument("--source", required=True)
    p.add_argument("--reference", default="expectation.png")
    p.add_argument("--output", "-o", default=None)
    a = p.parse_args()
    grade_video(a.source, a.reference, a.output or "temp/ref_graded.mp4")
