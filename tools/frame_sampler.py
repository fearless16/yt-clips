#!/usr/bin/env python3
"""Sample 30 random frames from input/video.mp4, run DNN face detection, save metrics."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from utils.face_detect import detect_face, detect_faces
from utils.face_matcher import find_host_in_frame

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports" / "face_detection"


def _dnn_detect_with_conf(frame, score_threshold=0.7):
    from utils.face_detect import _get_net
    net = _get_net()
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    results = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < score_threshold:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        results.append((x1, y1, x2 - x1, y2 - y1, conf))
    return results


def _sharpness(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    video_path = Path("input/video.mp4")
    if not video_path.exists():
        print(f"Error: {video_path} not found", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames < 30:
        sample_indices = list(range(total_frames))
    else:
        sample_indices = sorted(np.random.choice(total_frames, 30, replace=False).tolist())

    frames_dir = REPORTS / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        timestamp_sec = idx / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame_brightness = float(np.mean(gray))
        frame_contrast = float(np.std(gray))
        frame_sharpness = _sharpness(gray)

        face_bbox = detect_face(frame)
        all_faces = detect_faces(frame)
        detections = _dnn_detect_with_conf(frame)

        face_detected = face_bbox is not None
        num_faces = len(all_faces)

        result = {
            "frame_idx": idx,
            "timestamp_sec": round(timestamp_sec, 3),
            "face_detected": face_detected,
            "face_bbox": None,
            "face_center_norm": None,
            "face_top_norm": None,
            "face_size_pct": None,
            "face_area_pct": None,
            "num_faces": num_faces,
            "confidence": 0.0,
            "brightness": 0.0,
            "contrast": 0.0,
            "sharpness": 0.0,
            "skin_lab": [0.0, 0.0, 0.0],
            "saturation": 0.0,
            "host_match_confidence": 0.0,
            "frame_brightness": round(frame_brightness, 2),
            "frame_contrast": round(frame_contrast, 2),
            "frame_sharpness": round(frame_sharpness, 2),
        }

        if face_detected:
            x, y, w, h = map(int, face_bbox)
            cx, cy = x + w / 2, y + h / 2

            best_conf = 0.0
            for dx, dy, dw, dh, conf in detections:
                ox = max(0, min(x + w, dx + dw) - max(x, dx))
                oy = max(0, min(y + h, dy + dh) - max(y, dy))
                if ox > 0 and oy > 0:
                    union = w * h + dw * dh - ox * oy
                    if union > 0:
                        iou = (ox * oy) / union
                        if iou > 0.5 and conf > best_conf:
                            best_conf = conf

            if best_conf == 0.0 and detections:
                detections_sorted = sorted(detections, key=lambda d: d[2] * d[3], reverse=True)
                best_conf = detections_sorted[0][4]

            face_roi = frame[y:y + h, x:x + w]
            if face_roi.size > 0:
                face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                face_lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
                face_hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
                brightness = float(np.mean(face_gray))
                contrast = float(np.std(face_gray))
                sharpness = _sharpness(face_gray)
                skin_lab = [
                    float(np.mean(face_lab[:, :, 0])),
                    float(np.mean(face_lab[:, :, 1])),
                    float(np.mean(face_lab[:, :, 2])),
                ]
                saturation = float(np.mean(face_hsv[:, :, 1]))
            else:
                brightness = contrast = sharpness = 0.0
                skin_lab = [0.0, 0.0, 0.0]
                saturation = 0.0

            host_match = find_host_in_frame(frame)
            host_match_confidence = host_match.get("confidence", 0.0) if host_match else 0.0

            result.update({
                "face_bbox": [x, y, w, h],
                "face_center_norm": [round(cx / fw, 4), round(cy / fh, 4)],
                "face_top_norm": round(y / fh, 4),
                "face_size_pct": [round(w / fw * 100, 2), round(h / fh * 100, 2)],
                "face_area_pct": round((w * h) / (fw * fh) * 100, 2),
                "confidence": round(best_conf, 2),
                "brightness": round(brightness, 2),
                "contrast": round(contrast, 2),
                "sharpness": round(sharpness, 2),
                "skin_lab": [round(v, 2) for v in skin_lab],
                "saturation": round(saturation, 2),
                "host_match_confidence": round(host_match_confidence, 2),
            })

        results.append(result)

        sample_pos = sample_indices.index(idx)
        if sample_pos < 10:
            out_path = frames_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

    cap.release()

    per_frame_path = REPORTS / "per_frame_results.json"
    with open(per_frame_path, "w") as f:
        json.dump(results, f, indent=2)

    face_frames = [r for r in results if r["face_detected"]]
    host_match_frames = [r for r in results if r["host_match_confidence"] > 0]
    detection_rate = len(face_frames) / len(results) if results else 0

    summary = {
        "video_path": str(video_path),
        "frame_width": fw,
        "frame_height": fh,
        "total_frames": total_frames,
        "fps": round(fps, 2),
        "samples": len(results),
        "detection_rate": round(detection_rate, 2),
        "avg_face_area_pct": round(float(np.mean([r["face_area_pct"] for r in face_frames])), 2) if face_frames else 0,
        "avg_confidence": round(float(np.mean([r["confidence"] for r in face_frames])), 2) if face_frames else 0,
        "avg_brightness": round(float(np.mean([r["brightness"] for r in face_frames])), 2) if face_frames else 0,
        "avg_contrast": round(float(np.mean([r["contrast"] for r in face_frames])), 2) if face_frames else 0,
        "avg_sharpness": round(float(np.mean([r["sharpness"] for r in face_frames])), 2) if face_frames else 0,
        "frames_with_host_match": len(host_match_frames),
        "avg_host_match_confidence": round(
            float(np.mean([r["host_match_confidence"] for r in face_frames])), 2
        ) if face_frames else 0,
    }

    summary_path = REPORTS / "detection_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
