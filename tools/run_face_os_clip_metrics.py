#!/usr/bin/env python3
"""Run a streaming Face OS validation pass and write a markdown metrics report."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
import sys
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from face_os.pipeline import FaceOSPipeline


def _laplacian_var(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _contrast(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray))


def _hf_energy(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    low = cv2.GaussianBlur(gray, (0, 0), 2.0)
    high = gray - low
    return float(np.mean(high * high))


def _mean_lab(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.mean(lab, axis=(0, 1))


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if abs(den) > 1e-8 else 0.0


class RunningStats:
    def __init__(self) -> None:
        self.values: list[float] = []

    def add(self, value: float) -> None:
        if math.isfinite(value):
            self.values.append(float(value))

    @property
    def mean(self) -> float:
        return float(np.mean(self.values)) if self.values else 0.0

    @property
    def minimum(self) -> float:
        return float(np.min(self.values)) if self.values else 0.0

    @property
    def maximum(self) -> float:
        return float(np.max(self.values)) if self.values else 0.0


def _video_writer(path: Path, fps: float, frame_shape: tuple[int, int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frame_shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 30.0,
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output writer: {path}")
    return writer


def run(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    max_frames: Optional[int],
    max_seconds: Optional[float],
) -> int:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input clip: {input_path}")

    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pipeline = FaceOSPipeline(use_bidirectional=False)
    if not pipeline.enroll(reference_image="expectation.png", reference_dir="photos/"):
        raise RuntimeError("Face OS enrollment failed")

    input_sharp = RunningStats()
    output_sharp = RunningStats()
    input_contrast = RunningStats()
    output_contrast = RunningStats()
    input_hf = RunningStats()
    output_hf = RunningStats()
    input_luma = RunningStats()
    output_luma = RunningStats()
    interframe_delta = RunningStats()
    lab_flicker = RunningStats()

    writer: Optional[cv2.VideoWriter] = None
    prev_output: Optional[np.ndarray] = None
    prev_lab: Optional[np.ndarray] = None
    processed = 0
    failures = 0
    start = time.perf_counter()

    while True:
        if max_frames is not None and processed >= max_frames:
            break
        if max_seconds is not None and (time.perf_counter() - start) >= max_seconds:
            break
        ok, frame = cap.read()
        if not ok:
            break

        input_sharp.add(_laplacian_var(frame))
        input_contrast.add(_contrast(frame))
        input_hf.add(_hf_energy(frame))
        input_luma.add(float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))))

        try:
            result = pipeline.process_frame(frame, frame_idx=processed)
            output = result["frame"]
        except Exception:
            failures += 1
            output = cv2.resize(frame, (1080, 1920), interpolation=cv2.INTER_LINEAR)

        if writer is None:
            writer = _video_writer(output_path, source_fps, output.shape)
        writer.write(output)

        output_sharp.add(_laplacian_var(output))
        output_contrast.add(_contrast(output))
        output_hf.add(_hf_energy(output))
        output_luma.add(float(np.mean(cv2.cvtColor(output, cv2.COLOR_BGR2GRAY))))

        lab = _mean_lab(output)
        if prev_lab is not None:
            lab_flicker.add(float(np.linalg.norm(lab - prev_lab)))
        prev_lab = lab

        if prev_output is not None:
            interframe_delta.add(float(np.mean(np.abs(
                output.astype(np.float32) - prev_output.astype(np.float32)
            ))))
        prev_output = output
        processed += 1
        if processed == 1 or processed % 10 == 0:
            print(f"processed {processed} frames", flush=True)

    cap.release()
    if writer is not None:
        writer.release()

    elapsed = time.perf_counter() - start
    telemetry = pipeline.get_telemetry_report()
    frame_telemetry = pipeline.get_frame_telemetry()
    render_paths: dict[str, int] = {}
    fallback_reasons: dict[str, int] = {}
    geometry_sources: dict[str, int] = {}
    resample_counts: dict[int, int] = {}
    transform_dets: list[float] = []
    for row in frame_telemetry:
        render_paths[row.get("render_path", "unknown")] = render_paths.get(row.get("render_path", "unknown"), 0) + 1
        reason = row.get("fallback_reason")
        if reason:
            fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
        source = row.get("geometry_source", "unknown")
        geometry_sources[source] = geometry_sources.get(source, 0) + 1
        resample = int(row.get("resample_count", -1))
        resample_counts[resample] = resample_counts.get(resample, 0) + 1
        try:
            transform_dets.append(float(row.get("transform_det", 1.0)))
        except Exception:
            pass

    goals = {
        "sharpness_target": 274.0,
        "flicker_target": 1.0,
        "contrast_target": 73.0,
    }
    achieved = {
        "sharpness": output_sharp.mean >= goals["sharpness_target"],
        "flicker": lab_flicker.mean < goals["flicker_target"],
        "contrast": output_contrast.mean >= goals["contrast_target"],
        "runtime": processed > 0 and failures == 0,
        "telemetry": len(frame_telemetry) == processed,
    }
    completed = all(achieved.values())

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join([
            "# Face OS Real Clip Metrics",
            "",
            f"**Input:** `{input_path}`",
            f"**Output:** `{output_path}`",
            f"**Frames processed:** {processed} / {source_frames}",
            f"**Stopped by max frames:** {'YES' if max_frames is not None and processed >= max_frames else 'NO'}",
            f"**Stopped by max seconds:** {'YES' if max_seconds is not None and elapsed >= max_seconds else 'NO'}",
            f"**Source:** {source_w}x{source_h} @ {source_fps:.2f} fps",
            f"**Processing time:** {elapsed:.2f}s ({processed / elapsed if elapsed > 0 else 0.0:.2f} fps)",
            f"**Pipeline failures:** {failures}",
            "",
            "## Verdict",
            "",
            f"**Project completed by target metrics:** {'YES' if completed else 'NO'}",
            "",
            "| Gate | Target | Actual | Pass |",
            "|---|---:|---:|:---:|",
            f"| Sharpness | >= {goals['sharpness_target']:.1f} | {output_sharp.mean:.2f} | {'YES' if achieved['sharpness'] else 'NO'} |",
            f"| Flicker | < {goals['flicker_target']:.1f} | {lab_flicker.mean:.2f} | {'YES' if achieved['flicker'] else 'NO'} |",
            f"| Contrast | >= {goals['contrast_target']:.1f} | {output_contrast.mean:.2f} | {'YES' if achieved['contrast'] else 'NO'} |",
            f"| Runtime | 0 failures | {failures} failures | {'YES' if achieved['runtime'] else 'NO'} |",
            f"| Per-frame telemetry | {processed} rows | {len(frame_telemetry)} rows | {'YES' if achieved['telemetry'] else 'NO'} |",
            "",
            "## Signal Metrics",
            "",
            "| Metric | Input mean | Output mean | Output/Input | Output min | Output max |",
            "|---|---:|---:|---:|---:|---:|",
            f"| Sharpness, Laplacian variance | {input_sharp.mean:.2f} | {output_sharp.mean:.2f} | {_safe_ratio(output_sharp.mean, input_sharp.mean):.3f} | {output_sharp.minimum:.2f} | {output_sharp.maximum:.2f} |",
            f"| High-frequency energy | {input_hf.mean:.2f} | {output_hf.mean:.2f} | {_safe_ratio(output_hf.mean, input_hf.mean):.3f} | {output_hf.minimum:.2f} | {output_hf.maximum:.2f} |",
            f"| Contrast, grayscale std | {input_contrast.mean:.2f} | {output_contrast.mean:.2f} | {_safe_ratio(output_contrast.mean, input_contrast.mean):.3f} | {output_contrast.minimum:.2f} | {output_contrast.maximum:.2f} |",
            f"| Luminance mean | {input_luma.mean:.2f} | {output_luma.mean:.2f} | {_safe_ratio(output_luma.mean, input_luma.mean):.3f} | {output_luma.minimum:.2f} | {output_luma.maximum:.2f} |",
            f"| LAB flicker | n/a | {lab_flicker.mean:.2f} | n/a | {lab_flicker.minimum:.2f} | {lab_flicker.maximum:.2f} |",
            f"| Output inter-frame delta | n/a | {interframe_delta.mean:.2f} | n/a | {interframe_delta.minimum:.2f} | {interframe_delta.maximum:.2f} |",
            "",
            "## Runtime Telemetry",
            "",
            f"- Render paths: `{render_paths}`",
            f"- Geometry sources: `{geometry_sources}`",
            f"- Resample counts: `{resample_counts}`",
            f"- Fallback reasons: `{fallback_reasons}`",
            f"- Physical render rate: `{telemetry.get('physical_render_rate', 0.0):.4f}`",
            f"- Alpha fallback rate: `{telemetry.get('alpha_fallback_rate', 0.0):.4f}`",
            f"- Intrinsic success rate: `{telemetry.get('intrinsic_success_rate', 0.0):.4f}`",
            f"- Average intrinsic confidence: `{telemetry.get('avg_intrinsic_confidence', 0.0):.4f}`",
            f"- Average decomposition error: `{telemetry.get('avg_decomposition_error', 0.0):.4f}`",
            f"- Mesh normal rate: `{telemetry.get('mesh_normal_rate', 0.0):.4f}`",
            f"- Shading normal rate: `{telemetry.get('shading_normal_rate', 0.0):.4f}`",
            f"- Renderer mode transitions: `{telemetry.get('renderer_mode_transitions', 0)}`",
            f"- Transform determinant mean/std: `{float(np.mean(transform_dets)) if transform_dets else 0.0:.4f}` / `{float(np.std(transform_dets)) if transform_dets else 0.0:.4f}`",
            "",
            "## Completion Assessment",
            "",
            "The project should only be called complete when runtime succeeds and the measured visual gates pass on real clips.",
            "This report uses the explicit D-01 targets supplied for sharpness, flicker, and contrast.",
            "",
        ]),
        encoding="utf-8",
    )
    print(f"Wrote {report_path}")
    print(f"Verdict completed={completed}")
    return 0 if processed > 0 and failures == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="clips_test/test_clip.mp4")
    parser.add_argument("--output", default="output/face_os/test_clip_validated.mp4")
    parser.add_argument("--report", default="reports/face_os_real_clip_metrics.md")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=None)
    args = parser.parse_args()
    return run(Path(args.input), Path(args.output), Path(args.report), args.max_frames, args.max_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
