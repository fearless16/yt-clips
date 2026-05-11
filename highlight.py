import argparse
import json
import subprocess
import sys
import wave
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import re
import numpy as np
import yaml  # type: ignore

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("highlight", cfg["logging"]["log_file"], cfg["logging"]["level"])


# ─── Audio energy extraction (reliable method) ───────────────────────────────

def _extract_audio_rms(video_path: str, chunk_seconds: float = 1.0) -> List[Tuple[float, float]]:
    """
    Extract per-second RMS energy from the audio track.

    Method: Extract audio to raw PCM via FFmpeg, compute RMS per chunk
    using pure Python (no numpy needed). This is 100% reliable compared
    to parsing FFmpeg's astats metadata output.
    """
    # Extract audio to raw 16-bit PCM
    temp_dir = Path(cfg["paths"]["temp"])
    temp_dir.mkdir(parents=True, exist_ok=True)
    pcm_path = str(temp_dir / "audio_analysis.wav")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                       # no video
        "-acodec", "pcm_s16le",      # 16-bit signed little-endian
        "-ar", "16000",              # 16kHz sample rate (fast processing)
        "-ac", "1",                  # mono
        pcm_path,
    ]
    log.info("Extracting audio for RMS analysis …")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("Audio extraction failed:\n%s", result.stderr[-1000:])
        return []

    # Read WAV and compute RMS per chunk
    rms_values: list[tuple[float, float]] = []
    try:
        with wave.open(pcm_path, "rb") as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            total_frames = wf.getnframes()

            chunk_frames = int(sample_rate * chunk_seconds)
            timestamp = 0.0

            while True:
                raw = wf.readframes(chunk_frames)
                if not raw:
                    break

                # Unpack 16-bit samples
                n_samples = len(raw) // 2
                if n_samples == 0:
                    break

                # Compute RMS
                arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                rms = float(np.sqrt(np.mean(arr**2))) / 32768.0

                rms_values.append((timestamp, rms))
                timestamp += chunk_seconds

    except Exception as e:
        log.error("Error reading WAV file: %s", e)
        return []
    finally:
        # Clean up temp file
        try:
            Path(pcm_path).unlink(missing_ok=True)
        except Exception:
            pass

    log.info("Extracted %d RMS samples (%.1f minutes of audio)",
             len(rms_values), len(rms_values) * chunk_seconds / 60)

    return rms_values


def _get_video_duration(video_path: str) -> float:
    """Return video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


# ─── Heuristic scoring ────────────────────────────────────────────────────────

def _words_per_minute(text: str, duration_sec: float) -> float:
    if duration_sec <= 0:
        return 0.0
    words = len(text.split())
    return (words / duration_sec) * 60.0


def _silence_seconds(text: str, duration_sec: float) -> float:
    """Rough estimate: silence ≈ duration − (words × avg_word_duration)."""
    words = len(text.split())
    estimated_speech = words * 0.35          # ~0.35 s per word on average
    silence = max(0.0, duration_sec - estimated_speech)
    return silence


def _score_segment(
    seg: dict,
    rms_map: dict[int, float],
    avg_rms: float,
    max_rms: float,
    h_cfg: dict,
) -> float:
    """
    Compute a float score for a transcript segment.
    Higher score → more likely to be a highlight.
    """
    start: float = seg["start"]
    end: float   = seg["end"]
    text: str    = seg.get("text", "")
    duration     = max(end - start, 0.1)

    score = 0.0

    # 1. Audio energy — average RMS across all 1-second buckets in segment
    buckets = [rms_map.get(int(t), 0.0) for t in range(int(start), int(end) + 1)]
    seg_rms = sum(buckets) / len(buckets) if buckets else 0.0

    if avg_rms > 0:
        energy_ratio = seg_rms / avg_rms
        score += energy_ratio * 3.0            # weight: audio energy matters most

    # Bonus for peaks — if any bucket is in top 10% of max RMS
    if max_rms > 0:
        peak_buckets = sum(1 for b in buckets if b > max_rms * 0.8)
        if peak_buckets > 0:
            score += peak_buckets * 0.5

    # 2. Fast speech bonus
    wpm = _words_per_minute(text, duration)
    if wpm >= h_cfg["fast_speech_wpm"]:
        score += 1.5
    elif wpm >= h_cfg["fast_speech_wpm"] * 0.8:
        # Partial bonus for moderately fast speech
        score += 0.5

    # 3. Silence penalty (enhanced — aggressively penalise dead air)
    silence = _silence_seconds(text, duration)
    silence_ratio = silence / duration if duration > 0 else 0
    if silence > h_cfg["silence_penalty_seconds"]:
        score -= 0.5 * (silence / duration)
    # Heavy penalty for segments that are mostly silence (>60% dead air)
    if silence_ratio > 0.6:
        score -= 3.0  # Effectively skip these segments
    # Minimum word count — avoid clips with barely any speech
    word_count = len(text.split())
    if word_count < 5:
        score -= 1.5  # Penalise very low speech content

    # 4. Keyword bonus — reaction and analysis words (expanded for scenario/analytics)
    reaction_words = {
        # Original reaction words
        "oh", "wow", "wait", "what", "no", "yes", "let's", "go",
        "insane", "crazy", "bro", "dude", "actually", "holy", "damn",
        "unbelievable", "incredible", "amazing", "clutch", "huge",
        "perfect", "beautiful", "massive", "destroyed", "killed",
        "wicket", "six", "four", "boundary", "out", "catch",
        "shot", "brilliant", "superb", "excellent", "fantastic",
        # Hindi / Hinglish additions:
        "arre", "kya", "bhai", "yaar", "baap", "pagal", "gajab",
        "khatarnak", "chhakka", "chauka", "maar", "maro", "gaya",
        "jeet", "shandar", "dhamaakedaar", "zabardast", "sixer",
        "catch", "dekho", "khatam", "bawaal", "machaa",
        # Analytics / Scenario Shorts additions (from audit):
        "analytics", "scenario", "playoff", "playoffs", "points", "table", 
        "qualification", "stats", "record", "chances", "probability", "equation",
        "prediction", "predict", "calculate", "math", "net run rate", "nrr",
        "standings", "qualify", "eliminate", "knockout"
    }
    words_lower = set(re.findall(r'\b\w+\b', text.lower()))
    hits = len(words_lower & reaction_words)
    score += hits * 0.5  # slightly increased weight for keywords

    # 5. Exclamation/question mark density bonus
    exclaim_count = text.count("!") + text.count("?")
    score += exclaim_count * 0.2

    return score


# ─── Merging overlapping/nearby windows ──────────────────────────────────────

def _merge_windows(windows: List[Dict], gap: float) -> List[Dict]:
    """Merge adjacent windows separated by less than `gap` seconds."""
    if not windows:
        return []
    merged = [dict(windows[0])]
    for w in windows[1:]:
        prev = merged[-1]
        if w["start"] - prev["end"] <= gap:
            prev["end"] = max(prev["end"], w["end"])
            prev["score"] = max(prev["score"], w["score"])
        else:
            merged.append(dict(w))
    return merged


def _format_ts(seconds: float) -> str:
    """Convert seconds → HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ─── Main detection logic ─────────────────────────────────────────────────────

def detect_highlights(
    transcript_path: Optional[str] = None,
    video_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> List[Dict]:
    """
    Analyse transcript + audio energy and return highlight windows.

    Returns:
        List of dicts: [{id, start, end, start_ts, end_ts, score}, ...]
    """
    h_cfg   = cfg["highlight"]
    paths   = cfg["paths"]
    dl_cfg  = cfg["download"]

    if video_path is None:
        video_path = str(Path(paths["input"]) / dl_cfg["output_filename"])
    if transcript_path is None:
        stem = Path(video_path).stem
        transcript_path = str(Path(paths["transcripts"]) / f"{stem}.json")
    if output_path is None:
        stem = Path(video_path).stem
        output_path = str(Path(paths["highlights"]) / f"{stem}.yaml")

    # Load transcript
    t_path = Path(transcript_path)
    if not t_path.exists():
        log.error("Transcript not found: %s", t_path)
        sys.exit(1)

    with open(t_path, encoding="utf-8") as f:
        segments: list[dict] = json.load(f)

    log.info("Loaded %d transcript segments from %s", len(segments), t_path)

    # Extract audio energy (reliable WAV-based method)
    rms_list = _extract_audio_rms(video_path)
    rms_map  = {int(t): v for t, v in rms_list}

    if rms_list:
        all_rms = [v for _, v in rms_list]
        avg_rms = sum(all_rms) / len(all_rms)
        max_rms = max(all_rms)
        # Standard deviation for better threshold
        std_rms = float(np.std(all_rms))
    else:
        avg_rms = 1.0
        max_rms = 1.0
        std_rms = 0.0

    log.info("Audio RMS — avg: %.4f | max: %.4f | std: %.4f", avg_rms, max_rms, std_rms)

    # Score every segment
    scored: list[dict] = []
    for seg in segments:
        score = _score_segment(seg, rms_map, avg_rms, max_rms, h_cfg)
        scored.append({
            "start": seg["start"],
            "end":   seg["end"],
            "text":  seg.get("text", ""),
            "score": round(score, 4),
        })

    # Keep only segments above energy threshold (relative to max score)
    all_scores = [s["score"] for s in scored]
    max_score = max(all_scores) if all_scores else 1.0
    min_score = min(all_scores) if all_scores else 0.0
    threshold = min_score + (max_score - min_score) * h_cfg["audio_energy_threshold"]
    candidates = [s for s in scored if s["score"] >= threshold]

    log.info("Score range: %.2f → %.2f | threshold: %.2f | candidates: %d/%d",
             min_score, max_score, threshold, len(candidates), len(scored))

    # Expand each candidate into a [start-pad, end+pad] window
    min_dur = h_cfg["min_duration"]
    max_dur = h_cfg["max_duration"]

    # Get video duration for boundary clamping
    video_duration = _get_video_duration(video_path)

    windows: list[dict] = []
    for c in candidates:
        seg_duration = c["end"] - c["start"]
        # If segment is shorter than min_dur, pad symmetrically
        if seg_duration < min_dur:
            pad = (min_dur - seg_duration) / 2
            win_start = max(0.0, c["start"] - pad)
            win_end = min(video_duration, c["end"] + pad) if video_duration > 0 else c["end"] + pad
        else:
            win_start = c["start"]
            win_end = c["end"]

        # Enforce max duration
        if win_end - win_start > max_dur:
            win_end = win_start + max_dur

        windows.append({"start": win_start, "end": win_end, "score": c["score"]})

    # Sort by start time and merge nearby windows
    windows.sort(key=lambda w: w["start"])
    merged = _merge_windows(windows, h_cfg["merge_gap"])

    # Enforce max duration after merge
    for w in merged:
        if w["end"] - w["start"] > max_dur:
            center = (w["start"] + w["end"]) / 2.0
            w["start"] = max(0.0, center - max_dur / 2.0)
            w["end"] = w["start"] + max_dur

    # Sort by score and take top N
    merged.sort(key=lambda w: w["score"], reverse=True)
    
    # CRITICAL FIX: Ensure intro coverage (first 30 seconds)
    # If there's a high-scoring segment in the intro, force-include it
    intro_threshold = max_score * 0.45  # 45% of max score
    intro_segments = [w for w in merged if w["start"] < 30 and w["score"] >= intro_threshold]
    
    top = []
    remaining = list(merged)
    # Add best intro segment if exists
    if intro_segments:
        best_intro = max(intro_segments, key=lambda w: w["score"])
        top.append(best_intro)
        remaining.remove(best_intro)
        log.info("🎬 FORCED INTRO: Including highlight from start (%.2f score)", best_intro["score"])
    
    # Fill remaining slots with highest scoring segments (excluding already selected)
    remaining_slots = h_cfg["max_clips"] - len(top)
    top.extend(remaining[:remaining_slots])
    
    log.info("Selected %d clips (%d intro + %d others)", len(top), 1 if intro_segments else 0, len(top) - (1 if intro_segments else 0))

    # Re-sort by time for output
    top.sort(key=lambda w: w["start"])

    # Build YAML output — include transcript text for each window
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    yaml_data: dict = {}
    highlights: list[dict] = []
    for i, w in enumerate(top, start=1):
        key = f"clip{i}"

        # Aggregate transcript text overlapping this window
        window_text_parts = []
        for seg in segments:
            # Check if segment overlaps with the highlight window
            if seg["end"] > w["start"] and seg["start"] < w["end"]:
                window_text_parts.append(seg.get("text", ""))
        window_text = " ".join(window_text_parts).strip() or "Cricket Highlights"

        yaml_data[key] = {
            "start": _format_ts(w["start"]),
            "end":   _format_ts(w["end"]),
            "start_sec": round(w["start"], 2),
            "end_sec":   round(w["end"], 2),
            "score": w["score"],
            "text":  window_text,
        }
        highlights.append({
            "id":       key,
            "start":    w["start"],
            "end":      w["end"],
            "start_ts": _format_ts(w["start"]),
            "end_ts":   _format_ts(w["end"]),
            "score":    w["score"],
            "text":     window_text,
        })
        log.info("  %s: %s → %s (score=%.3f)", key, _format_ts(w["start"]), _format_ts(w["end"]), w["score"])

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

    log.info("Highlights saved → %s (%d clips)", output_path, len(highlights))
    return highlights


# ─── CLI entry-point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Detect highlights using audio + transcript heuristics.")
    parser.add_argument("--transcript", "-t", default=None, help="Transcript JSON path")
    parser.add_argument("--video",      "-v", default=None, help="Source video path (for audio analysis)")
    parser.add_argument("--output",     "-o", default=None, help="Output highlights YAML path")
    args = parser.parse_args()

    detect_highlights(args.transcript, args.video, args.output)


if __name__ == "__main__":
    main()
