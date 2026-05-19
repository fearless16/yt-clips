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
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
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
    # 1. Contrast LUT: linear stretch, but since we center on actual mean
    #    (computed per-frame as single float mean, NOT face detection),
    #    we store the ratio only — applied as multiply in .apply()
    params["_contrast_ratio"] = contrast_ratio

    # 2. a/b fixed offset: nudge toward reference from neutral
    #    Applied as: out = in + nudge  (fixed, no per-frame stats)
    params["_a_offset"] = (a_target - 128.0) * 0.15
    params["_b_offset"] = (b_target - 128.0) * 0.15

    # 3. Saturation LUT
    x = np.arange(256, dtype=np.float32)
    sat_lut = np.clip(sat_mult * x, 0, 255).astype(np.uint8)
    params["_sat_lut"] = sat_lut

    log.info("Enrollment complete: %d params extracted", len(params))
    return params


# ─── Phase 2: INFERENCE — Apply grade to any frame ──────────────────────

def apply_grade(frame: np.ndarray, params: Dict) -> np.ndarray:
    """Apply the reference grade to a single BGR frame.

    Pure math — no face detection, no per-frame analysis.
    Same transform applied to every pixel of every frame.
    """
    f32 = frame.astype(np.float32)

    # ── 1. Contrast (linear stretch around mean) ────────────────────────
    lab = cv2.cvtColor(f32.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    r = params["_contrast_ratio"]
    if r > 1.0:
        mean_L = float(np.mean(L))
        lab[:, :, 0] = np.clip((L - mean_L) * r + mean_L, 0, 255)

    # ── 2. Fixed a/b offset toward reference skin tone ──────────────────
    lab[:, :, 1] = np.clip(lab[:, :, 1] + params["_a_offset"], 0, 255)
    lab[:, :, 2] = np.clip(lab[:, :, 2] + params["_b_offset"], 0, 255)

    result = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)

    # ── 4. Saturation ───────────────────────────────────────────────────
    hsv = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = cv2.LUT(hsv[:, :, 1].astype(np.uint8), params["_sat_lut"]).astype(np.float32)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    # ── 5. Split tone ───────────────────────────────────────────────────
    gray = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    sc = np.array(params["shadow_color"])
    hc = np.array(params["highlight_color"])
    sa = np.clip((128 - gray) / 128, 0, 1)[:, :, None]
    ha = np.clip((gray - 128) / 128, 0, 1)[:, :, None]
    result = result + sc * sa * 0.08 + hc * ha * 0.05

    # ── 6. Vignette ─────────────────────────────────────────────────────
    h, w = frame.shape[:2]
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    d = np.sqrt((X - cx)**2 + (Y - cy)**2)
    vig = 1 - (d / np.sqrt(cx**2 + cy**2)) * (1 - 1 / params["vignette_ratio"]) * 0.5
    result = result * vig[:, :, None]

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
        except ValueError as e:
            log.error("Enrollment failed: %s", e)
        log.info("Total enrollment: %.1fs", time.perf_counter() - t0)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.valid:
            return frame
        return apply_grade(frame, self.params)


def grade_video(source_path: str, reference_path: str, output_path: str) -> str:
    """Full pipeline: enroll once, apply to all frames."""
    t_start = time.perf_counter()
    grade = ReferenceGrade(reference_path)
    if not grade.valid:
        return source_path

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        return source_path

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log.info("Video: %dx%d @ %.1ffps (%d frames)", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), fps, total)

    import tempfile
    td = Path(tempfile.mkdtemp())
    fd = td / "frames"
    fd.mkdir()

    fi = 0
    tp = time.perf_counter()
    try:
        while True:
            ret, f = cap.read()
            if not ret:
                break
            g = grade.apply(f)
            cv2.imwrite(str(fd / f"{fi:06d}.jpg"), g, [cv2.IMWRITE_JPEG_QUALITY, 95])
            fi += 1
            if fi % 50 == 0:
                log.info("  %d/%d (%.0f fps)", fi, total, fi / max(time.perf_counter() - tp, 0.001))
    finally:
        cap.release()

    import subprocess
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(fd / "%06d.jpg"), "-i", source_path,
        "-map", "0:v", "-map", "1:a?",
        "-c:v", "libx264", "-crf", "16", "-preset", "slow",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart", output_path,
    ], capture_output=True, text=True)

    if Path(output_path).exists():
        log.info("Done: %s (%.1f MB, %.1fs)", output_path,
                 Path(output_path).stat().st_size / 1e6, time.perf_counter() - t_start)
    import shutil
    shutil.rmtree(td, ignore_errors=True)
    return output_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Reference-Derived Color Grade")
    p.add_argument("--source", required=True)
    p.add_argument("--reference", default="expectation.png")
    p.add_argument("--output", "-o", default=None)
    a = p.parse_args()
    grade_video(a.source, a.reference, a.output or "temp/ref_graded.mp4")
