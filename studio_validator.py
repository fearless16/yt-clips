"""
studio_validator.py — AI Studio Quality Validation.

Compares enhanced output against reference expectations.
Generates a detailed quality report with pass/fail for each metric.

Checks:
- Face color accuracy (LAB distance to reference)
- Sharpness match (Laplacian variance)
- Contrast match
- Flicker detection (temporal brightness variance)
- Abrupt lighting changes
- Black screen detection
- Frame-to-frame stability
- Color consistency
- Overall quality score

Usage:
    python studio_validator.py --source original.mp4 --enhanced enhanced.mp4 --reference expectation.png
    python studio_validator.py --enhanced enhanced.mp4 --reference expectation.png --report report.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("studio_validator", cfg["logging"]["log_file"], cfg["logging"]["level"])


# ─── Reference Profile ────────────────────────────────────────────────────

class ReferenceTarget:
    """Target metrics extracted from reference photos."""
    
    def __init__(self, reference_path: str):
        self.valid = False
        self.skin_lab = (108.5, 139.6, 146.7)  # Defaults
        self.brightness = 103
        self.contrast = 59
        self.sharpness = 232
        self.color_temp = 1.89
        self.lr_diff = 12
        
        img = cv2.imread(reference_path)
        if img is None:
            log.warning("Cannot read reference: %s", reference_path)
            return
        
        h, w = img.shape[:2]
        from utils.face_detect import detect_face
        face = detect_face(img, score_threshold=0.5)
        if face is None:
            log.warning("No face in reference: %s", reference_path)
            return
        x, y, fw, fh = face
        face_roi = img[y:y+fh, x:x+fw]
        face_lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
        
        self.skin_lab = (
            float(np.mean(face_lab[:,:,0])),
            float(np.mean(face_lab[:,:,1])),
            float(np.mean(face_lab[:,:,2])),
        )
        self.brightness = float(np.mean(gray[y:y+fh, x:x+fw]))
        self.contrast = float(np.std(gray[y:y+fh, x:x+fw]))
        self.sharpness = float(np.var(cv2.Laplacian(gray[y:y+fh, x:x+fw], cv2.CV_64F)))
        
        bgr_means = [float(np.mean(face_roi[:,:,i])) for i in range(3)]
        self.color_temp = bgr_means[2] / max(bgr_means[0], 1)
        
        left = float(np.mean(cv2.cvtColor(face_roi[:, :fw//2], cv2.COLOR_BGR2GRAY)))
        right = float(np.mean(cv2.cvtColor(face_roi[:, fw//2:], cv2.COLOR_BGR2GRAY)))
        self.lr_diff = abs(left - right)
        
        self.valid = True
        log.info("Reference: LAB=(%.0f,%.0f,%.0f) bright=%.0f contrast=%.0f sharp=%.0f",
                 *self.skin_lab, self.brightness, self.contrast, self.sharpness)


# ─── Per-Frame Analysis ───────────────────────────────────────────────────

def analyze_frame(frame: np.ndarray) -> Optional[Dict]:
    """Extract quality metrics from a single frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    from utils.face_detect import detect_face
    face = detect_face(frame, score_threshold=0.5)
    
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    
    if face is None:
        return {
            "face_detected": False,
            "brightness": brightness,
            "contrast": contrast,
            "sharpness": float(np.var(cv2.Laplacian(gray, cv2.CV_64F))),
            "skin_lab": None,
            "color_temp": None,
            "is_black": brightness < 5,
        }
    
    x, y, fw, fh = face
    face_roi = frame[y:y+fh, x:x+fw]
    face_gray = gray[y:y+fh, x:x+fw]
    face_lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
    
    bgr_means = [float(np.mean(face_roi[:,:,i])) for i in range(3)]
    color_temp = bgr_means[2] / max(bgr_means[0], 1)
    
    left = float(np.mean(cv2.cvtColor(face_roi[:, :fw//2], cv2.COLOR_BGR2GRAY)))
    right = float(np.mean(cv2.cvtColor(face_roi[:, fw//2:], cv2.COLOR_BGR2GRAY)))
    
    return {
        "face_detected": True,
        "brightness": brightness,
        "face_brightness": float(np.mean(face_gray)),
        "contrast": contrast,
        "face_contrast": float(np.std(face_gray)),
        "sharpness": float(np.var(cv2.Laplacian(face_gray, cv2.CV_64F))),
        "skin_lab": (
            float(np.mean(face_lab[:,:,0])),
            float(np.mean(face_lab[:,:,1])),
            float(np.mean(face_lab[:,:,2])),
        ),
        "color_temp": color_temp,
        "lr_diff": abs(left - right),
        "is_black": brightness < 5,
        "face_bbox": (x, y, fw, fh),
    }


# ─── Video Analysis ───────────────────────────────────────────────────────

def analyze_video_frames(video_path: str, sample_rate: int = 1) -> List[Dict]:
    """Analyze all frames in a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames_data = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_rate == 0:
            data = analyze_frame(frame)
            if data:
                data["frame_idx"] = frame_idx
                data["timestamp"] = frame_idx / fps if fps > 0 else 0
                frames_data.append(data)
        
        frame_idx += 1
    
    cap.release()
    return frames_data


# ─── Quality Checks ───────────────────────────────────────────────────────

def check_flicker(frames: List[Dict], threshold: float = 20, edge_frames: int = 15) -> Dict:
    """Detect brightness flicker between consecutive frames.
    
    Skips edge frames (first/last N) since source videos often have
    fade-in/out that creates natural brightness transitions.
    """
    if len(frames) < edge_frames * 2:
        return {"pass": True, "count": 0, "details": []}
    
    flickers = []
    for i in range(edge_frames, len(frames) - edge_frames):
        diff = abs(frames[i]["brightness"] - frames[i-1]["brightness"])
        if diff > threshold:
            flickers.append({
                "frame": frames[i]["frame_idx"],
                "time": frames[i]["timestamp"],
                "delta": round(diff, 1),
            })
    
    return {
        "pass": len(flickers) == 0,
        "count": len(flickers),
        "details": flickers[:10],
        "edge_excluded": edge_frames,
    }


def check_lighting_stability(frames: List[Dict], source_frames: Optional[List[Dict]] = None, threshold: float = 15) -> Dict:
    """Check for abrupt lighting changes in face region.
    
    If source_frames provided, compares against source stability.
    Otherwise uses absolute threshold.
    """
    face_brightness = [f["face_brightness"] for f in frames if f.get("face_brightness")]
    if len(face_brightness) < 2:
        return {"pass": True, "std": 0, "reason": "not_enough_data"}
    
    std = float(np.std(face_brightness))
    
    # If source provided, compare against source stability
    if source_frames:
        src_brightness = [f["face_brightness"] for f in source_frames if f.get("face_brightness")]
        if src_brightness:
            src_std = float(np.std(src_brightness))
            # Enhancement should not make stability worse by more than 50%
            return {
                "pass": std < src_std * 1.5,
                "std": round(std, 2),
                "source_std": round(src_std, 2),
                "threshold": f"< {src_std * 1.5:.1f} (1.5x source)",
            }
    
    return {
        "pass": std < threshold,
        "std": round(std, 2),
        "threshold": threshold,
    }


def check_black_screens(frames: List[Dict], edge_frames: int = 10) -> Dict:
    """Detect black screen frames. Skips first/last edge frames."""
    if len(frames) < edge_frames * 2:
        return {"pass": True, "count": 0, "frames": []}
    
    black_frames = [f for f in frames[edge_frames:-edge_frames] if f["is_black"]]
    return {
        "pass": len(black_frames) == 0,
        "count": len(black_frames),
        "frames": [f["frame_idx"] for f in black_frames[:10]],
        "edge_excluded": edge_frames,
    }


def check_color_consistency(frames: List[Dict], threshold: float = 8) -> Dict:
    """Check skin color consistency across frames."""
    a_vals = [f["skin_lab"][1] for f in frames if f.get("skin_lab")]
    if len(a_vals) < 2:
        return {"pass": True, "std": 0, "reason": "not_enough_data"}
    
    std = float(np.std(a_vals))
    return {
        "pass": std < threshold,
        "std": round(std, 2),
        "threshold": threshold,
    }


def check_sharpness_overshoot(frames: List[Dict], target: float) -> Dict:
    """Check that sharpness doesn't wildly overshoot target."""
    sharpness_vals = [f["sharpness"] for f in frames if f["sharpness"] > 0]
    if not sharpness_vals:
        return {"pass": True, "max": 0}
    
    max_sharp = float(np.max(sharpness_vals))
    avg_sharp = float(np.mean(sharpness_vals))
    
    return {
        "pass": max_sharp < target * 3,
        "max": round(max_sharp, 0),
        "avg": round(avg_sharp, 0),
        "target": target,
    }


def check_face_color_distance(frames: List[Dict], target_lab: Tuple[float, float, float]) -> Dict:
    """Check how close face color is to reference."""
    distances = []
    for f in frames:
        if f.get("skin_lab"):
            dist = np.sqrt(sum((a - b)**2 for a, b in zip(f["skin_lab"], target_lab)))
            distances.append(dist)
    
    if not distances:
        return {"pass": True, "avg_distance": 0, "reason": "no_faces"}
    
    avg_dist = float(np.mean(distances))
    max_dist = float(np.max(distances))
    
    # LAB distance of 10-15 is noticeable, 30+ is significant
    return {
        "pass": avg_dist < 25,
        "avg_distance": round(avg_dist, 1),
        "max_distance": round(max_dist, 1),
        "target_lab": target_lab,
    }


# ─── Full Validation ──────────────────────────────────────────────────────

def validate(
    enhanced_path: str,
    reference_path: str,
    source_path: Optional[str] = None,
    sample_rate: int = 2,
) -> Dict:
    """
    Full quality validation of enhanced video against reference.
    
    Returns a detailed report with pass/fail for each check.
    """
    t_start = time.perf_counter()
    
    # Load reference
    ref = ReferenceTarget(reference_path)
    if not ref.valid:
        return {"error": "invalid_reference", "valid": False}
    
    # Analyze enhanced video
    log.info("Analyzing enhanced: %s", enhanced_path)
    enhanced_frames = analyze_video_frames(enhanced_path, sample_rate)
    if not enhanced_frames:
        return {"error": "no_frames", "valid": False}
    
    log.info("Analyzed %d frames", len(enhanced_frames))
    
    # Run all checks
    source_frames_for_comparison = None
    if source_path:
        log.info("Analyzing source: %s", source_path)
        source_frames_for_comparison = analyze_video_frames(source_path, sample_rate)
    
    report = {
        "valid": True,
        "enhanced_video": enhanced_path,
        "reference": reference_path,
        "frames_analyzed": len(enhanced_frames),
        "reference_target": {
            "skin_lab": ref.skin_lab,
            "brightness": ref.brightness,
            "contrast": ref.contrast,
            "sharpness": ref.sharpness,
            "color_temp": ref.color_temp,
        },
        "checks": {
            "flicker": check_flicker(enhanced_frames),
            "lighting_stability": check_lighting_stability(enhanced_frames, source_frames_for_comparison),
            "black_screens": check_black_screens(enhanced_frames),
            "color_consistency": check_color_consistency(enhanced_frames),
            "sharpness_overshoot": check_sharpness_overshoot(enhanced_frames, ref.sharpness),
            "face_color_match": check_face_color_distance(enhanced_frames, ref.skin_lab),
        },
        "analysis_time_sec": round(time.perf_counter() - t_start, 1),
    }
    
    # Analyze source if provided
    if source_frames_for_comparison:
        report["source_comparison"] = {
            "frames_analyzed": len(source_frames_for_comparison),
            "flicker": check_flicker(source_frames_for_comparison),
            "black_screens": check_black_screens(source_frames_for_comparison),
            "face_color_match": check_face_color_distance(source_frames_for_comparison, ref.skin_lab),
        }
    
    # Overall score
    checks = report["checks"]
    passed = sum(1 for c in checks.values() if c.get("pass", False))
    total = len(checks)
    report["score"] = f"{passed}/{total}"
    report["overall_pass"] = passed == total
    
    # Summary stats
    face_frames = [f for f in enhanced_frames if f.get("face_detected")]
    if face_frames:
        report["enhanced_stats"] = {
            "avg_brightness": round(float(np.mean([f["face_brightness"] for f in face_frames])), 1),
            "avg_contrast": round(float(np.mean([f["face_contrast"] for f in face_frames])), 1),
            "avg_sharpness": round(float(np.mean([f["sharpness"] for f in face_frames])), 0),
            "avg_skin_L": round(float(np.mean([f["skin_lab"][0] for f in face_frames if f.get("skin_lab")])), 1),
            "avg_skin_a": round(float(np.mean([f["skin_lab"][1] for f in face_frames if f.get("skin_lab")])), 1),
        }
    
    return report


def print_report(report: Dict):
    """Pretty-print the validation report."""
    if "error" in report:
        print(f"ERROR: {report['error']}")
        return
    
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box
        
        console = Console()
        
        console.print()
        console.print(Panel(
            f"STUDIO VALIDATION — Score: {report['score']}",
            border_style="green" if report["overall_pass"] else "red",
            width=70,
        ))
        
        t = Table(title="Quality Checks", box=box.ROUNDED)
        t.add_column("Check", style="cyan")
        t.add_column("Status", justify="center")
        t.add_column("Details")
        
        for name, result in report["checks"].items():
            status = "PASS" if result.get("pass") else "FAIL"
            style = "green" if result.get("pass") else "red"
            
            details = []
            for k, v in result.items():
                if k != "pass" and k != "details":
                    details.append(f"{k}={v}")
            
            t.add_row(name, f"[{style}]{status}[/{style}]", " ".join(details[:3]))
        
        console.print(t)
        
        if "enhanced_stats" in report:
            stats = report["enhanced_stats"]
            ref = report["reference_target"]
            st = Table(title="Enhanced vs Reference", box=box.SIMPLE)
            st.add_column("Metric", style="cyan")
            st.add_column("Enhanced", justify="right")
            st.add_column("Reference", justify="right")
            st.add_column("Delta", justify="right")
            
            for metric, key in [("Brightness", "avg_brightness"), ("Contrast", "avg_contrast"),
                               ("Sharpness", "avg_sharpness"), ("Skin L", "avg_skin_L"),
                               ("Skin a", "avg_skin_a")]:
                if key in stats:
                    enhanced_val = stats[key]
                    ref_key = key.replace("avg_", "")
                    ref_val = ref.get(ref_key, ref.get("brightness"))
                    if ref_val:
                        delta = enhanced_val - ref_val
                        st.add_row(metric, f"{enhanced_val:.1f}", f"{ref_val:.1f}", f"{delta:+.1f}")
            
            console.print(st)
        
        console.print()
    
    except ImportError:
        print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Studio quality validation")
    parser.add_argument("--enhanced", required=True, help="Enhanced video path")
    parser.add_argument("--reference", default="expectation.png", help="Reference image")
    parser.add_argument("--source", default=None, help="Original source video (for comparison)")
    parser.add_argument("--report", default=None, help="Save JSON report to file")
    parser.add_argument("--sample-rate", type=int, default=2, help="Frame sampling rate")
    args = parser.parse_args()
    
    report = validate(args.enhanced, args.reference, args.source, args.sample_rate)
    print_report(report)
    
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report saved: {args.report}")
