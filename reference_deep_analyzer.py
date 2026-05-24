"""
reference_deep_analyzer.py — Extract ALL parameters from expectation.png.

Not just face — full body, lighting, background, scene contrast, color harmony.
These parameters drive the ref_grade LUT and are validated by tests.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


def analyze_reference(image_path: str = "expectation.png") -> Dict:
    """Deep analysis of reference image — extracts every grading-relevant parameter."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read: {image_path}")

    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # ── 1. Face Detection ───────────────────────────────────────────────
    from utils.face_detect import detect_face
    face_bbox = detect_face(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==2 else img, score_threshold=0.5)
    if face_bbox is None:
        raise ValueError("No face detected in reference")

    fx, fy, fw, fh = face_bbox
    face = img[fy:fy+fh, fx:fx+fw]
    face_lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB).astype(np.float32)
    face_hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV).astype(np.float32)
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # ── 2. Body Region (below face, assuming portrait) ──────────────────
    body_y1 = fy + fh
    body_y2 = min(h, int(body_y1 + fh * 1.5))
    body_x1 = max(0, fx - fw // 2)
    body_x2 = min(w, fx + fw + fw // 2)
    body = img[body_y1:body_y2, body_x1:body_x2]
    body_lab = cv2.cvtColor(body, cv2.COLOR_BGR2LAB).astype(np.float32) if body.size > 0 else face_lab

    # ── 3. Background Regions (sides + top) ─────────────────────────────
    # Left background
    bg_left = img[:, :max(1, fx)]
    # Right background
    bg_right = img[:, fx+fw:]
    # Top background
    bg_top = img[:max(1, fy), :]
    # Combined background (non-face, non-body)
    bg_mask = np.ones((h, w), dtype=np.uint8) * 255
    bg_mask[fy:fy+fh, fx:fx+fw] = 0
    bg_mask[body_y1:body_y2, body_x1:body_x2] = 0
    bg_pixels = img[bg_mask > 0]
    if bg_pixels.size > 0:
        bg_reshaped = bg_pixels.reshape(-1, 1, 3)
        bg_lab_all = cv2.cvtColor(bg_reshaped, cv2.COLOR_BGR2LAB).astype(np.float32)
        bg_lab = bg_lab_all.reshape(-1, 3)
    else:
        bg_lab = np.array([[0.0, 128.0, 128.0]])

    # ── 4. Lighting Direction Analysis ──────────────────────────────────
    # Split face into left/right halves
    face_mid = fw // 2
    face_left_L = float(np.mean(face_lab[:, :face_mid, 0]))
    face_right_L = float(np.mean(face_lab[:, face_mid:, 0]))
    lr_ratio = max(face_left_L, face_right_L) / max(min(face_left_L, face_right_L), 1)
    light_dir = "left" if face_left_L > face_right_L else "right"

    # Split face into top/bottom halves
    face_top_L = float(np.mean(face_lab[:fh//2, :, 0]))
    face_bot_L = float(np.mean(face_lab[fh//2:, :, 0]))
    tb_ratio = max(face_top_L, face_bot_L) / max(min(face_top_L, face_bot_L), 1)

    # ── 5. Full Frame Statistics ────────────────────────────────────────
    full_L = float(np.mean(lab[:, :, 0]))
    full_a = float(np.mean(lab[:, :, 1]))
    full_b = float(np.mean(lab[:, :, 2]))
    full_contrast = float(np.std(gray))
    full_sat = float(np.mean(hsv[:, :, 1]))

    # ── 6. Face Statistics ──────────────────────────────────────────────
    face_L = float(np.mean(face_lab[:, :, 0]))
    face_a = float(np.mean(face_lab[:, :, 1]))
    face_b = float(np.mean(face_lab[:, :, 2]))
    face_contrast = float(np.std(face_gray))
    face_sat = float(np.mean(face_hsv[:, :, 1]))

    # ── 7. Body Statistics ──────────────────────────────────────────────
    body_L = float(np.mean(body_lab[:, :, 0]))
    body_a = float(np.mean(body_lab[:, :, 1]))
    body_b = float(np.mean(body_lab[:, :, 2]))

    # ── 8. Background Statistics ────────────────────────────────────────
    bg_L = float(np.mean(bg_lab[:, 0]))
    bg_a = float(np.mean(bg_lab[:, 1]))
    bg_b = float(np.mean(bg_lab[:, 2]))

    # ── 9. Shadow/Highlight Distribution ────────────────────────────────
    shadow_pct = float(np.mean(gray < 85)) * 100
    midtone_pct = float(np.mean((gray >= 85) & (gray <= 170))) * 100
    highlight_pct = float(np.mean(gray > 170)) * 100

    # ── 10. Color Harmony (warm/cool balance) ───────────────────────────
    # a > 128 = warm (red), a < 128 = cool (green)
    # b > 128 = warm (yellow), b < 128 = cool (blue)
    warm_pct = float(np.mean((lab[:, :, 1] > 130) | (lab[:, :, 2] > 130))) * 100

    # ── 11. Edge Contrast (sharpness) ───────────────────────────────────
    laplacian = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F)
    edge_sharpness = float(np.std(laplacian))

    # ── 12. Skin Tone Consistency (face vs body) ────────────────────────
    skin_L_diff = abs(face_L - body_L)
    skin_a_diff = abs(face_a - body_a)
    skin_b_diff = abs(face_b - body_b)
    skin_consistency = np.linalg.norm([skin_L_diff, skin_a_diff, skin_b_diff])

    # ── 13. Vignette ────────────────────────────────────────────────────
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    maxd = np.sqrt(cx**2 + cy**2)
    center_L = float(np.mean(gray[dist < maxd * 0.3]))
    edge_L = float(np.mean(gray[dist > maxd * 0.85]))
    vignette_ratio = center_L / max(edge_L, 1)

    # ── 14. Split Tone Colors ───────────────────────────────────────────
    low_mask = gray < 51
    high_mask = gray > 204
    shadow_color = np.mean(img[low_mask], axis=0).tolist() if low_mask.any() else [0, 0, 0]
    highlight_color = np.mean(img[high_mask], axis=0).tolist() if high_mask.any() else [255, 255, 255]

    return {
        # Image info
        "image_path": image_path,
        "dimensions": (w, h),
        "face_bbox": (fx, fy, fw, fh),

        # Face metrics
        "face_L": face_L,
        "face_a": face_a,
        "face_b": face_b,
        "face_contrast": face_contrast,
        "face_saturation": face_sat,

        # Body metrics
        "body_L": body_L,
        "body_a": body_a,
        "body_b": body_b,

        # Full frame metrics
        "full_L": full_L,
        "full_a": full_a,
        "full_b": full_b,
        "full_contrast": full_contrast,
        "full_saturation": full_sat,

        # Background metrics
        "bg_L": bg_L,
        "bg_a": bg_a,
        "bg_b": bg_b,

        # Lighting
        "lr_ratio": lr_ratio,
        "light_direction": light_dir,
        "tb_ratio": tb_ratio,
        "face_top_L": face_top_L,
        "face_bot_L": face_bot_L,

        # Distribution
        "shadow_pct": shadow_pct,
        "midtone_pct": midtone_pct,
        "highlight_pct": highlight_pct,

        # Color
        "warm_pct": warm_pct,

        # Quality
        "edge_sharpness": edge_sharpness,
        "vignette_ratio": vignette_ratio,

        # Consistency
        "skin_consistency": skin_consistency,

        # Split tone
        "shadow_color": shadow_color,
        "highlight_color": highlight_color,
    }


def print_report(params: Dict) -> None:
    """Print a formatted report of all extracted parameters."""
    print("=" * 60)
    print("  REFERENCE DEEP ANALYSIS: %s" % params["image_path"])
    print("  Dimensions: %dx%d" % params["dimensions"])
    print("  Face bbox: %s" % (params["face_bbox"],))
    print("=" * 60)

    print("\n  FACE METRICS")
    print("    L=%.1f  a=%.1f  b=%.1f" % (params["face_L"], params["face_a"], params["face_b"]))
    print("    Contrast=%.1f  Saturation=%.1f" % (params["face_contrast"], params["face_saturation"]))

    print("\n  BODY METRICS")
    print("    L=%.1f  a=%.1f  b=%.1f" % (params["body_L"], params["body_a"], params["body_b"]))

    print("\n  FULL FRAME METRICS")
    print("    L=%.1f  a=%.1f  b=%.1f" % (params["full_L"], params["full_a"], params["full_b"]))
    print("    Contrast=%.1f  Saturation=%.1f" % (params["full_contrast"], params["full_saturation"]))

    print("\n  BACKGROUND METRICS")
    print("    L=%.1f  a=%.1f  b=%.1f" % (params["bg_L"], params["bg_a"], params["bg_b"]))

    print("\n  LIGHTING")
    print("    Left/Right ratio=%.2f (%s lit)" % (params["lr_ratio"], params["light_direction"]))
    print("    Top/Bottom ratio=%.2f" % params["tb_ratio"])
    print("    Face top L=%.1f  bottom L=%.1f" % (params["face_top_L"], params["face_bot_L"]))

    print("\n  DISTRIBUTION")
    print("    Shadows=%.1f%%  Midtones=%.1f%%  Highlights=%.1f%%" % (
        params["shadow_pct"], params["midtone_pct"], params["highlight_pct"]))

    print("\n  COLOR")
    print("    Warm pixels=%.1f%%" % params["warm_pct"])

    print("\n  QUALITY")
    print("    Edge sharpness=%.1f" % params["edge_sharpness"])
    print("    Vignette ratio=%.2f" % params["vignette_ratio"])

    print("\n  CONSISTENCY")
    print("    Skin consistency=%.1f (face vs body LAB delta)" % params["skin_consistency"])

    print("\n  SPLIT TONE")
    print("    Shadow color (BGR)=%s" % [round(x, 1) for x in params["shadow_color"]])
    print("    Highlight color (BGR)=%s" % [round(x, 1) for x in params["highlight_color"]])


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "expectation.png"
    params = analyze_reference(path)
    print_report(params)
