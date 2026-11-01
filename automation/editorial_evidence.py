"""Source-bound discovery evidence. Coverage is measured, not inferred from success."""

import json
import math
import subprocess
from pathlib import Path

from automation.artifacts import atomic_json
from automation.release_review import sha256


def transcript_coverage(segments: list[dict], duration: float | None) -> dict:
    intervals = []
    invalid = []
    for index, cue in enumerate(segments):
        try:
            start, end = float(cue["start"]), float(cue["end"])
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
                raise ValueError("Invalid interval")
            if duration is not None and end > duration + 0.1:
                raise ValueError("Cue exceeds source duration")
            intervals.append((start, end))
        except (ValueError, TypeError, KeyError):
            invalid.append(index)
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    gaps = []
    cursor = 0.0
    for start, end in merged:
        if start > cursor:
            gaps.append([cursor, start])
        cursor = end
    if duration is not None and duration > cursor:
        gaps.append([cursor, duration])
    return {"source_duration": duration, "cue_count": len(segments), "invalid_cues": invalid,
            "covered_intervals": merged, "untranscribed_intervals": gaps,
            "covered_seconds": sum(end - start for start, end in merged),
            "gap_interpretation": "unverified: may be silence, missed speech or non-speech events",
            "audio_visual_coverage": "NOT CHECKED"}


def source_identity(video_path: str) -> dict:
    path = Path(video_path).resolve()
    result = {"path": str(path), "sha256": None, "duration": None}
    try:
        result["sha256"] = sha256(path)
    except OSError as exc:
        result["error"] = str(exc)
        return result
    try:
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                "-of", "json", str(path)], capture_output=True, text=True, timeout=30)
        duration = float(json.loads(probe.stdout)["format"]["duration"])
        if probe.returncode == 0 and math.isfinite(duration) and duration > 0:
            result["duration"] = duration
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired):
        pass
    return result


def write_discovery_evidence(output_path: str, source: dict, transcript_path: str,
                             segments: list[dict], candidates: list[dict], selected: list[dict],
                             decision: dict | None = None) -> Path:
    selected_ranges = {(c["start"], c["end"]) for c in selected}
    alternatives = []
    for index, candidate in enumerate(candidates):
        alternatives.append({
            "candidate_id": index + 1, "source_range": [candidate["start"], candidate["end"]],
            "text": candidate.get("text", ""), "heuristic_score": candidate.get("final_score", candidate.get("score")),
            "selected": (candidate["start"], candidate["end"]) in selected_ranges,
            "hard_rejection": candidate.get("should_reject", False),
            "rejection_reasons": candidate.get("rejection_reasons", []),
        })
    path = Path(output_path).with_suffix(".editorial.json")
    atomic_json(path, {
        "schema_version": 1, "source": source,
        "transcript": {"path": str(Path(transcript_path).resolve()), "sha256": sha256(Path(transcript_path))},
        "coverage": transcript_coverage(segments, source.get("duration")),
        "discovery_method": "full transcript heuristic windows then diverse shortlist; not full audiovisual review",
        "candidates": alternatives, "editorial_decision": decision or {"status": "NOT CHECKED"},
        "selected": [{"source_range": [c["start"], c["end"]], "reason": c.get("ai_reason", ""),
                      "semantic_review": c.get("semantic_review", "NOT CHECKED")} for c in selected],
        "release_status": "REVIEW",
    })
    return path
