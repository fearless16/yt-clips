"""Measured encoded-file QC. Does not certify meaning, framing or intelligibility."""

import argparse
import json
import math
import re
import subprocess
from pathlib import Path

from automation.artifacts import atomic_json
from automation.release_review import sha256


def av_timing_issues(streams: list[dict], *, start_tolerance=0.1, end_tolerance=0.15) -> list[str]:
    """Compare encoded start/end timestamps; unknown timing is a hold, not PASS.

    Container timing alone cannot prove transcript alignment or lip sync.
    """
    if not all(math.isfinite(t) and t >= 0 for t in (start_tolerance, end_tolerance)):
        return ["Invalid A/V timing tolerance"]
    timing = {}
    for kind in ("video", "audio"):
        try:
            stream = next(s for s in streams if s.get("codec_type") == kind)
            start, duration = float(stream["start_time"]), float(stream["duration"])
            if not all(math.isfinite(t) for t in (start, duration, start + duration)) or duration <= 0:
                raise ValueError("Invalid stream timing")
            timing[kind] = (start, start + duration)
        except (StopIteration, KeyError, ValueError, TypeError, AttributeError):
            return [f"Missing or invalid {kind} stream timing"]
    issues = []
    if abs(timing["video"][0] - timing["audio"][0]) > start_tolerance:
        issues.append("Encoded A/V start offset exceeds tolerance")
    if abs(timing["video"][1] - timing["audio"][1]) > end_tolerance:
        issues.append("Encoded A/V end offset exceeds tolerance")
    return issues


def check_delivery(clip: Path, expected_duration: float | None = None) -> dict:
    clip = Path(clip).resolve()
    result = {"schema_version": 1, "media_sha256": sha256(clip), "status": "FAIL", "errors": [],
              "visual_review": "NOT CHECKED", "listening_review": "NOT CHECKED"}
    try:
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(clip)],
                               capture_output=True, text=True, check=True, timeout=30)
        info = json.loads(probe.stdout)
        timing_errors = av_timing_issues(info["streams"])
        result["av_timing"] = {"status": "FAIL" if timing_errors else "PASS", "errors": timing_errors,
                               "method": "encoded stream start/end timestamps; not content alignment"}
        result["errors"].extend(timing_errors)
        video = next(s for s in info["streams"] if s.get("codec_type") == "video")
        duration = float(info["format"]["duration"])
        if not math.isfinite(duration) or not 0 < duration <= 180:
            result["errors"].append("Invalid Shorts duration")
        if int(video["width"]) != 1080 or int(video["height"]) != 1920:
            result["errors"].append("Expected 1080x1920 delivery")
        if not any(s.get("codec_type") == "audio" for s in info["streams"]):
            result["errors"].append("Missing audio stream")
        if expected_duration is not None and abs(duration - expected_duration) > 0.15:
            result["errors"].append("Encoded duration differs from edit map")
        result["duration"] = duration
        # xerror ensures decode errors cannot be hidden behind an exit-0 render.
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(clip),
                        "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"],
                       capture_output=True, text=True, check=True, timeout=300)
        result["full_decode"] = "PASS"
        loudness = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-i", str(clip), "-vn",
                                   "-af", "loudnorm=I=-16:TP=-2:LRA=11:print_format=json", "-f", "null", "-"],
                                  capture_output=True, text=True, check=True, timeout=300)
        matches = re.findall(r"\{[^{}]*\"input_i\"[^{}]*\}", loudness.stderr, re.DOTALL)
        measured = json.loads(matches[-1])
        integrated, peak = float(measured["input_i"]), float(measured["input_tp"])
        result["encoded_audio"] = {"integrated_lufs": integrated if math.isfinite(integrated) else None,
                                   "true_peak_dbtp": peak if math.isfinite(peak) else None}
        if not math.isfinite(integrated) or not -17 <= integrated <= -15:
            result["errors"].append("Encoded loudness outside project -16 +/-1 LUFS target")
        if not math.isfinite(peak) or peak > -1:
            result["errors"].append("Encoded true peak exceeds project -1 dBTP ceiling")
        if not result["errors"]:
            result["status"] = "PASS"
    except (OSError, ValueError, TypeError, KeyError, StopIteration, IndexError, subprocess.SubprocessError) as exc:
        result["errors"].append(str(exc))
    atomic_json(clip.with_name(f"{clip.stem}_qc.json"), result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", type=Path)
    parser.add_argument("--expected-duration", type=float)
    args = parser.parse_args()
    result = check_delivery(args.clip, args.expected_duration)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
