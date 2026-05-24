"""
expectation_analyzer.py — Extract face metrics from expectation.png
Output: reports/face_detection/expectation_metrics.json + expectation_face_roi.png
"""

import cv2
import json
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.face_detect import detect_face

ROOT = Path(__file__).parent.parent
REPORTS = ROOT / "reports" / "face_detection"

IMG_PATH = ROOT / "expectation.png"
OUT_JSON = REPORTS / "expectation_metrics.json"
OUT_ROI = REPORTS / "expectation_face_roi.png"


def laplacian_variance(img_gray: np.ndarray) -> float:
    """Return variance of Laplacian (sharpness measure)."""
    lap = cv2.Laplacian(img_gray, cv2.CV_64F)
    return float(lap.var())


def main():
    frame = cv2.imread(str(IMG_PATH))
    if frame is None:
        print(f"ERROR: Could not read {IMG_PATH}", file=sys.stderr)
        sys.exit(1)

    fh, fw = frame.shape[:2]

    face = detect_face(frame)
    if face is None:
        print("ERROR: No face detected in expectation.png", file=sys.stderr)
        sys.exit(1)

    x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])

    # ---- Face ROI ----
    face_roi = frame[y : y + h, x : x + w]
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_ROI), face_roi)

    # ---- Grayscale variants ----
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

    # ---- Position metrics ----
    cx, cy = x + w / 2.0, y + h / 2.0
    face_center_norm = [round(cx / fw, 4), round(cy / fh, 4)]
    face_top_norm = round(y / fh, 4)
    face_size_pct = [round(w / fw * 100, 1), round(h / fh * 100, 1)]
    face_area_pct = round((w * h) / (fw * fh) * 100, 1)

    # ---- Quality: brightness / contrast / sharpness ----
    brightness = float(np.mean(gray_roi))
    contrast = float(np.std(gray_roi))
    sharpness = laplacian_variance(gray_roi)

    # ---- LAB skin tone ----
    lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    skin_lab = [round(float(v), 2) for v in cv2.mean(lab)[:3]]

    # ---- Saturation (HSV S-channel) ----
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    saturation = float(np.mean(hsv[:, :, 1]))

    # ---- Color temperature proxy: B/R channel ratio ----
    b_mean = float(np.mean(face_roi[:, :, 0]))
    r_mean = float(np.mean(face_roi[:, :, 2]))
    color_temp = round(b_mean / r_mean, 4) if r_mean > 0 else 0.0

    # ---- Lighting: left/right brightness ratio ----
    mid = x + w // 2
    left_half = frame[y : y + h, x:mid]
    right_half = frame[y : y + h, mid : x + w]
    l_bright = float(np.mean(cv2.cvtColor(left_half, cv2.COLOR_BGR2GRAY))) if left_half.size else 0
    r_bright = float(np.mean(cv2.cvtColor(right_half, cv2.COLOR_BGR2GRAY))) if right_half.size else 0
    lr_ratio = round(l_bright / r_bright, 4) if r_bright > 0 else 1.0
    light_direction = "left" if l_bright >= r_bright else "right"

    # ---- Full-frame metrics ----
    frame_brightness = float(np.mean(gray_full))
    frame_contrast = float(np.std(gray_full))
    frame_sharpness = laplacian_variance(gray_full)

    # ---- Assemble output ----
    metrics = {
        "face_bbox": [x, y, w, h],
        "face_center_norm": face_center_norm,
        "face_top_norm": face_top_norm,
        "face_size_pct": face_size_pct,
        "face_area_pct": face_area_pct,
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "sharpness": round(sharpness, 2),
        "skin_lab": skin_lab,
        "saturation": round(saturation, 2),
        "color_temp": color_temp,
        "lr_ratio": lr_ratio,
        "light_direction": light_direction,
        "frame_brightness": round(frame_brightness, 2),
        "frame_contrast": round(frame_contrast, 2),
        "frame_sharpness": round(frame_sharpness, 2),
    }

    with open(OUT_JSON, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
