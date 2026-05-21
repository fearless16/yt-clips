"""Deep metrics extraction for input, output, and expectation videos/images.

Extracts: resolution, color stats, sharpness, temporal consistency,
face detection stats, LAB color distribution, frequency content.
"""

import cv2
import numpy as np
import json
import sys
from pathlib import Path


def extract_video_metrics(video_path: str, max_frames: int = 100) -> dict:
    """Extract deep metrics from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Cannot open {video_path}"}

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    lab_means = []
    lab_stds = []
    sharpness_vals = []
    brightness_vals = []
    contrast_vals = []
    temporal_diffs = []
    prev_gray = None

    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        count += 1
    cap.release()

    if not frames:
        return {"error": "No frames read"}

    for i, frame in enumerate(frames):
        # LAB analysis
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_means.append(np.mean(lab, axis=(0, 1)))
        lab_stds.append(np.std(lab, axis=(0, 1)))

        # Sharpness (Laplacian variance)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness_vals.append(float(np.var(lap)))

        # Brightness
        brightness_vals.append(float(np.mean(gray)))

        # Contrast (std of grayscale)
        contrast_vals.append(float(np.std(gray)))

        # Temporal difference
        if prev_gray is not None:
            diff = np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32)))
            temporal_diffs.append(float(diff))
        prev_gray = gray

    lab_means = np.array(lab_means)
    lab_stds = np.array(lab_stds)

    return {
        "file": str(video_path),
        "resolution": f"{w}x{h}",
        "width": w,
        "height": h,
        "fps": float(fps),
        "total_frames": total,
        "analyzed_frames": len(frames),
        "duration_s": len(frames) / fps if fps > 0 else 0,
        "color": {
            "L_mean": float(np.mean(lab_means[:, 0])),
            "L_std": float(np.mean(lab_stds[:, 0])),
            "a_mean": float(np.mean(lab_means[:, 1])),
            "a_std": float(np.mean(lab_stds[:, 1])),
            "b_mean": float(np.mean(lab_means[:, 2])),
            "b_std": float(np.mean(lab_stds[:, 2])),
        },
        "sharpness": {
            "laplacian_var_mean": float(np.mean(sharpness_vals)),
            "laplacian_var_std": float(np.std(sharpness_vals)),
            "laplacian_var_min": float(np.min(sharpness_vals)),
            "laplacian_var_max": float(np.max(sharpness_vals)),
        },
        "brightness": {
            "mean": float(np.mean(brightness_vals)),
            "std": float(np.std(brightness_vals)),
            "min": float(np.min(brightness_vals)),
            "max": float(np.max(brightness_vals)),
        },
        "contrast": {
            "mean": float(np.mean(contrast_vals)),
            "std": float(np.std(contrast_vals)),
        },
        "temporal": {
            "frame_diff_mean": float(np.mean(temporal_diffs)) if temporal_diffs else 0,
            "frame_diff_std": float(np.std(temporal_diffs)) if temporal_diffs else 0,
            "flicker_score": float(np.std(temporal_diffs)) if temporal_diffs else 0,
        },
    }


def extract_image_metrics(image_path: str) -> dict:
    """Extract deep metrics from a single image."""
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"Cannot read {image_path}"}

    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lap = cv2.Laplacian(gray, cv2.CV_64F)

    return {
        "file": str(image_path),
        "resolution": f"{w}x{h}",
        "width": w,
        "height": h,
        "color": {
            "L_mean": float(np.mean(lab[:, :, 0])),
            "L_std": float(np.std(lab[:, :, 0])),
            "a_mean": float(np.mean(lab[:, :, 1])),
            "a_std": float(np.std(lab[:, :, 1])),
            "b_mean": float(np.mean(lab[:, :, 2])),
            "b_std": float(np.std(lab[:, :, 2])),
        },
        "sharpness": {
            "laplacian_var": float(np.var(lap)),
        },
        "brightness": {
            "mean": float(np.mean(gray)),
            "std": float(np.std(gray)),
        },
        "contrast": {
            "std": float(np.std(gray)),
        },
    }


def extract_face_metrics(video_path: str, max_frames: int = 100) -> dict:
    """Extract face detection metrics from video."""
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Cannot open {video_path}"}

    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    )

    face_detected = 0
    face_areas = []
    bbox_sizes = []
    count = 0

    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        if results.detections:
            face_detected += 1
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                area = bbox.width * bbox.height
                face_areas.append(area)
                bbox_sizes.append((bbox.width, bbox.height))

    cap.close()
    detector.close()

    return {
        "frames_analyzed": count,
        "face_detected_frames": face_detected,
        "face_detection_rate": face_detected / max(count, 1),
        "face_area_mean": float(np.mean(face_areas)) if face_areas else 0,
        "face_area_std": float(np.std(face_areas)) if face_areas else 0,
        "face_area_min": float(np.min(face_areas)) if face_areas else 0,
        "face_area_max": float(np.max(face_areas)) if face_areas else 0,
    }


if __name__ == "__main__":
    input_vid = sys.argv[1] if len(sys.argv) > 1 else "clips_test/test_clip.mp4"
    output_vid = sys.argv[2] if len(sys.argv) > 2 else "output/face_os/output.mp4"
    expectation = sys.argv[3] if len(sys.argv) > 3 else "expectation.png"

    print("=" * 60)
    print("DEEP METRICS EXTRACTION")
    print("=" * 60)

    # Input video
    print("\n[1/3] Input video metrics...")
    input_m = extract_video_metrics(input_vid, max_frames=50)
    print(json.dumps(input_m, indent=2))

    # Output video
    print("\n[2/3] Output video metrics...")
    output_m = extract_video_metrics(output_vid, max_frames=50)
    print(json.dumps(output_m, indent=2))

    # Expectation image
    print("\n[3/3] Expectation image metrics...")
    expect_m = extract_image_metrics(expectation)
    print(json.dumps(expect_m, indent=2))

    # Save all
    all_metrics = {
        "input": input_m,
        "output": output_m,
        "expectation": expect_m,
    }
    with open("output/face_os/deep_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("\nSaved to output/face_os/deep_metrics.json")
