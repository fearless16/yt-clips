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
from utils.logger import get_logger

cfg = load_config()
log = get_logger("ref_grade", cfg["logging"]["log_file"], cfg["logging"]["level"])


# ─── Phase 1: ENROLLMENT — Extract fixed params from reference ──────────

def _detect_face_once(img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data_mod = getattr(cv2, "data", None)
    if data_mod is not None:
        cascade_path = getattr(data_mod, "haarcascades", None)
    else:
        cascade_path = None
    if cascade_path is None:
        cascade_path = os.path.join(os.path.dirname(cv2.__file__), "data")
    cascade = cv2.CascadeClassifier(os.path.join(str(cascade_path), "haarcascade_frontalface_default.xml"))
    faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
    if not len(faces):
        return None
    return tuple(int(v) for v in max(faces, key=lambda f: f[2]*f[3]))


def enroll(reference_path: str = "expectation.png") -> Dict:
    """Analyze reference photo and extract ALL color grading parameters.

    Returns a dict of CONSTANTS (not frame-dependent).
    These constants encode the reference's COLOR STYLE as mathematical values.
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

    x, y, fw, fh = bbox
    face = img[y:y+fh, x:x+fw]
    face_lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    full_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    full_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # ── 1. Contrast ratio ───────────────────────────────────────────────
    # Reference face contrast / typical face contrast (~50 std)
    ref_contrast = float(np.std(face_gray))
    contrast_ratio = max(1.0, min(ref_contrast / 50.0, 1.5))
    log.info("  Contrast: ref=%.1f ratio=%.2f", ref_contrast, contrast_ratio)

    # ── 2. Skin tone target (a, b means of face) ────────────────────────
    a_target = float(np.mean(face_lab[:, :, 1]))
    b_target = float(np.mean(face_lab[:, :, 2]))
    log.info("  Skin: a=%.1f b=%.1f", a_target, b_target)

    # ── 3. Color temperature (R/B ratio of face) ────────────────────────
    b_face = float(np.mean(face[:, :, 0]))
    r_face = float(np.mean(face[:, :, 2]))
    color_temp = r_face / max(b_face, 0.1)
    log.info("  Color temp: R/B=%.2f", color_temp)

    # ── 4. Saturation multiplier ────────────────────────────────────────
    face_hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    ref_sat = float(np.mean(face_hsv[:, :, 1]))
    sat_mult = max(0.8, min(ref_sat / 100.0, 1.5))
    log.info("  Saturation: ref=%.1f mult=%.2f", ref_sat, sat_mult)

    # ── 5. Split tone (shadow/highlight colors from full frame) ─────────
    low = full_gray < 51
    high = full_gray > 204
    shadow_color = np.mean(img[low], axis=0) if low.any() else np.zeros(3)
    highlight_color = np.mean(img[high], axis=0) if high.any() else np.full(3, 255)

    # ── 6. Vignette (center/edge ratio) ─────────────────────────────────
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    maxd = np.sqrt(cx**2 + cy**2)
    center = full_gray[dist < maxd * 0.3].mean()
    edge = full_gray[dist > maxd * 0.85].mean()
    vignette_ratio = float(np.clip(center / max(edge, 1), 1.0, 2.0))
    log.info("  Vignette: %.2f", vignette_ratio)

    params = {
        "contrast_ratio": contrast_ratio,
        "a_target": a_target,
        "b_target": b_target,
        "color_temp": color_temp,
        "sat_mult": sat_mult,
        "shadow_color": shadow_color.tolist(),
        "highlight_color": highlight_color.tolist(),
        "vignette_ratio": vignette_ratio,
    }

    # ── Pre-build LUTs (computed once during enrollment) ────────────────
    params["_contrast_ratio"] = contrast_ratio
    ss = 0.08
    hs = 0.05
    params["_shadow_strength"] = ss
    params["_highlight_strength"] = hs
    sa = sat_mult
    ao = (a_target - 128.0) * 0.15
    bo = (b_target - 128.0) * 0.15
    ta = (color_temp - 1.0) * 3.2
    tb = (color_temp - 1.0) * 1.8

    # LUTs for a,b transform: maps 0..255 → transformed value (no float32 math)
    x = np.arange(256, dtype=np.float32)
    params["_lut_a"] = np.clip((x - 128.0) * sa + 128.0 + ao + ta, 0, 255).astype(np.uint8)
    params["_lut_b"] = np.clip((x - 128.0) * sa + 128.0 + bo + tb, 0, 255).astype(np.uint8)

    # Split-tone LUT: 3 separate (256,) uint8 LUTs for B, G, R offsets
    lut_shadow = np.clip((128.0 - x) / 128.0, 0, 1).astype(np.float32)
    lut_highlight = np.clip((x - 128.0) / 128.0, 0, 1).astype(np.float32)
    params["_split_lut"] = (
        np.array(shadow_color)[:, None] * lut_shadow * ss
        + np.array(highlight_color)[:, None] * lut_highlight * hs
    ).T.astype(np.float32)  # (256, 3) — indexed by L value

    log.info("Enrollment complete: %d params extracted", len(params))
    return params


# ─── Phase 2: INFERENCE — Apply grade to any frame ──────────────────────

# Cache for vignette masks: keyed by (h, w, vr_int)
_vignette_cache: Dict[tuple, np.ndarray] = {}


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


def apply_grade(frame: np.ndarray, params: Dict) -> np.ndarray:
    """Apply the reference grade to a single BGR frame.

    2 cvtColor calls (BGR→LAB, LAB→BGR).  No HSV conversion.
    Split tone applied via pre-computed 256x3 LUT (no per-frame clip/div).
    Vignette mask cached by resolution.
    """
    if "_contrast_ratio" not in params:
        return frame

    # ── 1. BGR → LAB ────────────────────────────────────────────────────
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # ── 2. Contrast on L (linear stretch around mean) ───────────────────
    L = lab[:, :, 0].astype(np.float32)
    r = params["_contrast_ratio"]
    if r > 1.0:
        ml = float(np.mean(L))
        L = np.clip((L - ml) * r + ml, 0, 255)

    # ── 3. Combined a,b transform via LUT (no float32 math) ────────────
    lab[:, :, 1] = cv2.LUT(lab[:, :, 1], params["_lut_a"])
    lab[:, :, 2] = cv2.LUT(lab[:, :, 2], params["_lut_b"])
    lab[:, :, 0] = np.clip(L, 0, 255).astype(np.uint8)

    # ── 4. LAB → BGR ────────────────────────────────────────────────────
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32)

    # ── 5. Split tone via L-indexed LUT ────────────────────────────────
    result += params["_split_lut"][L.astype(np.int32)]

    # ── 6. Vignette (cached) ────────────────────────────────────────────
    h, w = frame.shape[:2]
    vig = _get_vignette(h, w, params["vignette_ratio"])
    result *= vig[:, :, None]

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
    tmp_out = f"/tmp/yt_clips_grade_{os.getpid()}.mp4"

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
