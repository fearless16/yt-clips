"""
crop_analyzer.py — Analyze cropped face quality vs expectation ROI.
Output: reports/face_detection/cropped_faces/*.jpg + cropped_metrics.json + cropped_summary.json
"""

import cv2
import json
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).parent.parent
REPORTS = ROOT / "reports" / "face_detection"
TOOLS = Path(__file__).parent

FRAMES_DIR = REPORTS / "cropped_faces"
PER_FRAME = REPORTS / "per_frame_results.json"
OUT_METRICS = REPORTS / "cropped_metrics.json"
OUT_SUMMARY = REPORTS / "cropped_summary.json"
EXPECTED = REPORTS / "expectation_metrics.json"

def _sharpness(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def compute_roi_metrics(face_roi):
    """Same metrics as expectation_analyzer.py for direct comparison."""
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)

    h, w = face_roi.shape[:2]

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = _sharpness(gray)
    skin_lab = [float(v) for v in cv2.mean(lab)[:3]]
    saturation = float(np.mean(hsv[:, :, 1]))

    b_mean = float(np.mean(face_roi[:, :, 0]))
    r_mean = float(np.mean(face_roi[:, :, 2]))
    color_temp = round(b_mean / r_mean, 4) if r_mean > 0 else 0.0

    mid = w // 2
    left_half = gray[:, :mid]
    right_half = gray[:, mid:]
    l_bright = float(np.mean(left_half)) if left_half.size else 0
    r_bright = float(np.mean(right_half)) if right_half.size else 0
    lr_ratio = round(l_bright / r_bright, 4) if r_bright > 0 else 1.0
    light_direction = "left" if l_bright >= r_bright else "right"

    return {
        "roi_size": [w, h],
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "sharpness": round(sharpness, 2),
        "skin_lab": [round(v, 2) for v in skin_lab],
        "saturation": round(saturation, 2),
        "color_temp": color_temp,
        "lr_ratio": lr_ratio,
        "light_direction": light_direction,
    }

def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    with open(PER_FRAME) as f:
        frames = json.load(f)

    with open(EXPECTED) as f:
        expected = json.load(f)

    video_path = Path("input/video.mp4")
    cap = cv2.VideoCapture(str(video_path))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cropped_results = []
    saved_count = 0

    for frame_data in frames:
        if not frame_data["face_detected"]:
            cropped_results.append({
                "frame_idx": frame_data["frame_idx"],
                "face_detected": False,
                "roi_metrics": None,
            })
            continue

        idx = frame_data["frame_idx"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        x, y, w, h = map(int, frame_data["face_bbox"])
        face_roi = frame[y : y + h, x : x + w]
        if face_roi.size == 0:
            continue

        ext = cv2.borderInterpolate if h < 2 or w < 2 else None

        metrics = compute_roi_metrics(face_roi)

        if saved_count < 15:
            roi_path = FRAMES_DIR / f"face_{idx:06d}.jpg"
            cv2.imwrite(str(roi_path), face_roi, [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved_count += 1

        face_area_pct = frame_data["face_area_pct"]
        confidence = frame_data["confidence"]

        deltas = {}
        for key in ["brightness", "contrast", "sharpness", "saturation"]:
            delta = metrics[key] - expected[key]
            deltas[f"{key}_delta"] = round(delta, 2)
            deltas[f"{key}_pct"] = round((delta / expected[key]) * 100, 1) if expected[key] != 0 else 0

        for i, axis in enumerate(['L', 'a', 'b']):
            d = metrics["skin_lab"][i] - expected["skin_lab"][i]
            deltas[f"skin_{axis}_delta"] = round(d, 2)
            deltas[f"skin_{axis}_pct"] = round((d / expected["skin_lab"][i]) * 100, 1) if expected["skin_lab"][i] != 0 else 0

        cropped_results.append({
            "frame_idx": idx,
            "timestamp_sec": frame_data["timestamp_sec"],
            "face_detected": True,
            "roi_bbox": [x, y, w, h],
            "face_area_pct": face_area_pct,
            "confidence": confidence,
            "roi_metrics": metrics,
            "deltas": deltas,
            "host_match_confidence": frame_data.get("host_match_confidence", 0.0),
        })

    cap.release()

    with open(OUT_METRICS, "w") as f:
        json.dump(cropped_results, f, indent=2)

    detected = [r for r in cropped_results if r["face_detected"]]
    metrics_list = [r["roi_metrics"] for r in detected]

    summary = {
        "total_frames": len(frames),
        "detected_frames": len(detected),
        "detection_rate": round(len(detected) / len(frames), 3) if frames else 0,
        "avg_brightness": round(float(np.mean([m["brightness"] for m in metrics_list])), 2),
        "avg_contrast": round(float(np.mean([m["contrast"] for m in metrics_list])), 2),
        "avg_sharpness": round(float(np.mean([m["sharpness"] for m in metrics_list])), 2),
        "avg_saturation": round(float(np.mean([m["saturation"] for m in metrics_list])), 2),
        "avg_skin_lab": [
            round(float(np.mean([m["skin_lab"][0] for m in metrics_list])), 2),
            round(float(np.mean([m["skin_lab"][1] for m in metrics_list])), 2),
            round(float(np.mean([m["skin_lab"][2] for m in metrics_list])), 2),
        ],
        "std_brightness": round(float(np.std([m["brightness"] for m in metrics_list])), 2),
        "std_contrast": round(float(np.std([m["contrast"] for m in metrics_list])), 2),
        "std_sharpness": round(float(np.std([m["sharpness"] for m in metrics_list])), 2),
        "std_saturation": round(float(np.std([m["saturation"] for m in metrics_list])), 2),
        "expected": {
            "brightness": expected["brightness"],
            "contrast": expected["contrast"],
            "sharpness": expected["sharpness"],
            "saturation": expected["saturation"],
            "skin_lab": expected["skin_lab"],
        },
        "avg_deltas": {
            "brightness_delta": round(float(np.mean([r["deltas"]["brightness_delta"] for r in detected])), 2),
            "contrast_delta": round(float(np.mean([r["deltas"]["contrast_delta"] for r in detected])), 2),
            "sharpness_delta": round(float(np.mean([r["deltas"]["sharpness_delta"] for r in detected])), 2),
            "saturation_delta": round(float(np.mean([r["deltas"]["saturation_delta"] for r in detected])), 2),
            "brightness_pct": round(float(np.mean([r["deltas"]["brightness_pct"] for r in detected])), 1),
            "contrast_pct": round(float(np.mean([r["deltas"]["contrast_pct"] for r in detected])), 1),
            "sharpness_pct": round(float(np.mean([r["deltas"]["sharpness_pct"] for r in detected])), 1),
            "saturation_pct": round(float(np.mean([r["deltas"]["saturation_pct"] for r in detected])), 1),
        },
    }

    with open(OUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
