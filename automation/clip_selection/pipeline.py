"""Pipeline integration — drop-in replacement for highlight.detect_highlights().

Usage:
    from automation.clip_selection.pipeline import detect_highlights
    highlights = detect_highlights(transcript_path, video_path, highlights_path)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from utils.config import load_config
from utils.logger import get_logger

from automation.clip_selection.selector import ClipSelector
from automation.clip_selection.arbiter import fmt_ts

from prompts import MAX_CANDIDATES, MAX_SELECTED_CLIPS, MIN_QUALITY_THRESHOLD

cfg = load_config()
log = get_logger("clip_pipeline")


# ── Copied from highlight.py for audio RMS extraction ──────────────────

def _extract_audio_rms(video_path: str, chunk_seconds: float = 1.0) -> list[tuple[float, float]]:
    import subprocess
    import wave
    temp_dir = Path(cfg["paths"]["temp"])
    temp_dir.mkdir(parents=True, exist_ok=True)
    pcm_path = str(temp_dir / "audio_analysis.wav")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", pcm_path,
    ]
    log.info("Extracting audio for RMS analysis ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("Audio extraction failed:\n%s", result.stderr[-1000:])
        return []
    rms_values = []
    try:
        with wave.open(pcm_path, "rb") as wf:
            sample_rate = wf.getframerate()
            chunk_frames = int(sample_rate * chunk_seconds)
            timestamp = 0.0
            while True:
                raw = wf.readframes(chunk_frames)
                if not raw:
                    break
                n_samples = len(raw) // 2
                if n_samples == 0:
                    break
                arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                rms = float(np.sqrt(np.mean(arr**2))) / 32768.0
                rms_values.append((timestamp, rms))
                timestamp += chunk_seconds
    except Exception as e:
        log.error("Error reading WAV file: %s", e)
        return []
    finally:
        try:
            Path(pcm_path).unlink(missing_ok=True)
        except Exception:
            pass
    log.info("Extracted %d RMS samples (%.1f minutes of audio)",
             len(rms_values), len(rms_values) * chunk_seconds / 60)
    return rms_values


def _words_per_minute(text: str, duration_sec: float) -> float:
    if duration_sec <= 0:
        return 0.0
    return (len(text.split()) / duration_sec) * 60.0


def _silence_seconds(text: str, duration_sec: float) -> float:
    if not text:
        return max(0.0, duration_sec)
    words = len(text.split())
    estimated_speech = words * 0.35
    return max(0.0, duration_sec - estimated_speech)


def _get_video_duration(video_path: str) -> float:
    import subprocess
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def _merge_windows(windows: list[dict], gap: float) -> list[dict]:
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


# ── Heuristic pre-filter (same as highlight.py) ─────────────────────────

def _score_segment(
    seg: dict, rms_map: dict, avg_rms: float, max_rms: float, h_cfg: dict
) -> float:
    start = seg["start"]
    end = seg["end"]
    text = seg.get("text", "")
    duration = max(end - start, 0.1)
    score = 0.0

    buckets = [rms_map.get(int(t), 0.0) for t in range(int(start), int(end) + 1)]
    seg_rms = sum(buckets) / len(buckets) if buckets else 0.0
    if avg_rms > 0:
        score += (seg_rms / avg_rms) * 3.0
    if max_rms > 0:
        peak_buckets = sum(1 for b in buckets if b > max_rms * 0.8)
        if peak_buckets > 0:
            score += peak_buckets * 0.5

    wpm = _words_per_minute(text, duration)
    if wpm >= h_cfg["fast_speech_wpm"]:
        score += 1.5
    elif wpm >= h_cfg["fast_speech_wpm"] * 0.8:
        score += 0.5

    silence = _silence_seconds(text, duration)
    silence_ratio = silence / duration if duration > 0 else 0
    if silence > h_cfg["silence_penalty_seconds"]:
        score -= 0.3 * (silence / duration)
    if silence_ratio > 0.6:
        score -= 1.5
    if len(text.split()) < 5:
        score -= 0.5

    hook_buckets = [rms_map.get(int(t), 0.0) for t in range(int(start), min(int(start) + 3, int(end) + 1))]
    if hook_buckets and max_rms > 0:
        hook_energy = sum(hook_buckets) / len(hook_buckets)
        if hook_energy > avg_rms * 1.2:
            score += 1.0

    import re
    reaction_words = {
        "oh", "wow", "wait", "what", "no", "yes", "whoa",
        "insane", "crazy", "bro", "dude", "holy", "damn",
        "unbelievable", "incredible", "amazing", "clutch", "huge",
        "perfect", "beautiful", "massive", "destroyed", "killed",
        "wicket", "six", "four", "boundary", "out", "catch",
        "shot", "brilliant", "superb", "fantastic",
        "arre", "kya", "bhai", "yaar", "baap", "pagal", "gajab",
        "khatarnak", "chhakka", "chauka", "maar", "maro", "gaya",
        "jeet", "shandar", "dhamaakedaar", "zabardast", "sixer",
        "dekho", "khatam", "bawaal", "machaa", "haan", "nahi",
        "oho", "accha", "abe", "teri", "re", "arey",
        "chhod", "dekh", "jaa", "nikal", "aagaya",
    }
    words_lower = set(re.findall(r'\b\w+\b', text.lower()))
    score += len(words_lower & reaction_words) * 0.6

    text_lower = text.lower()
    reaction_phrases = [
        "kya baat", "oh ho", "are yaar", "kya shot", "maine kya",
        "haan haan", "arre arre", "are bhai", "kya hua", "yeh kya",
        "oh my god", "oh god", "what a", "kya cheez", "baap re",
        "nahi yaar", "haan bhai", "oho ho", "gajab ka", "chhakka maar",
        "dhamaakedaar shot", "what a shot", "what a six", "what a catch",
    ]
    score += sum(2 for p in reaction_phrases if p in text_lower)

    words_list = re.findall(r'\b\w+\b', text_lower)
    for w in set(words_list):
        if words_list.count(w) >= 3 and len(w) > 1:
            score += 1.5
            break

    if max_rms > 0:
        segment_peaks = [rms_map.get(int(t), 0.0) for t in range(int(start), int(end) + 1)]
        spike_count = sum(1 for v in segment_peaks if v > max_rms * 0.85)
        if spike_count >= 2:
            score += spike_count * 0.8

    score += (text.count("!") + text.count("?")) * 0.3
    return score


# ── Main detection API ───────────────────────────────────────────────────

def detect_highlights(
    transcript_path: str | None = None,
    video_path: str | None = None,
    output_path: str | None = None,
) -> list[dict]:
    """Replace ``highlight.detect_highlights()`` with 7-agent clip selection.

    Same signature, same YAML output format — drop-in replacement.
    """
    h_cfg = cfg["highlight"]
    paths = cfg["paths"]
    dl_cfg = cfg["download"]

    if video_path is None:
        video_path = str(Path(paths["input"]) / dl_cfg["output_filename"])
    if transcript_path is None:
        stem = Path(video_path).stem
        transcript_path = str(Path(paths["transcripts"]) / f"{stem}.json")
    if output_path is None:
        stem = Path(video_path).stem
        output_path = str(Path(paths["highlights"]) / f"{stem}.yaml")

    t_path = Path(transcript_path)
    if not t_path.exists():
        log.error("Transcript not found: %s", t_path)
        sys.exit(1)

    with open(t_path, encoding="utf-8") as f:
        data = json.load(f)
    segments = data if isinstance(data, list) else data.get("segments", [])
    log.info("Loaded %d transcript segments from %s", len(segments), t_path)

    # ── Audio RMS extraction ───────────────────────────────────────────────
    rms_list = _extract_audio_rms(video_path)
    _rms_sums = defaultdict(float)
    _rms_counts = defaultdict(int)
    for t, v in rms_list:
        key = int(t)
        _rms_sums[key] += v
        _rms_counts[key] += 1
    rms_map = {k: _rms_sums[k] / _rms_counts[k] for k in _rms_sums}

    if rms_list:
        all_rms = [v for _, v in rms_list]
        avg_rms = sum(all_rms) / len(all_rms)
        max_rms = max(all_rms)
        std_rms = float(np.std(all_rms))
    else:
        avg_rms = 1.0
        max_rms = 1.0

    log.info("Audio RMS - avg: %.4f | max: %.4f", avg_rms, max_rms)

    # ── Heuristic pre-filter (same as highlight.py) ────────────────────────
    scored = []
    for seg in segments:
        s = _score_segment(seg, rms_map, avg_rms, max_rms, h_cfg)
        scored.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg.get("text", ""),
            "score": round(s, 4),
        })

    all_scores = [s["score"] for s in scored]
    max_score = max(all_scores) if all_scores else 1.0
    min_score = min(all_scores) if all_scores else 0.0
    threshold = min_score + (max_score - min_score) * h_cfg["audio_energy_threshold"]
    candidates = [s for s in scored if s["score"] >= threshold]

    log.info("Score range: %.2f -> %.2f | threshold: %.2f | candidates: %d/%d",
             min_score, max_score, threshold, len(candidates), len(scored))

    min_dur = h_cfg["min_duration"]
    max_dur = h_cfg["max_duration"]
    video_duration = _get_video_duration(video_path)

    windows = []
    for c in candidates:
        seg_duration = c["end"] - c["start"]
        if seg_duration < min_dur:
            pad = (min_dur - seg_duration) / 2
            win_start = max(0.0, c["start"] - pad)
            win_end = min(video_duration, c["end"] + pad) if video_duration > 0 else c["end"] + pad
        else:
            win_start = c["start"]
            win_end = c["end"]
        if win_end - win_start > max_dur:
            win_end = win_start + max_dur
        windows.append({"start": win_start, "end": win_end, "score": c["score"]})

    windows.sort(key=lambda w: w["start"])
    merged = _merge_windows(windows, h_cfg["merge_gap"])

    for w in merged:
        if w["end"] - w["start"] > max_dur:
            center = (w["start"] + w["end"]) / 2.0
            w["start"] = max(0.0, center - max_dur / 2.0)
            w["end"] = w["start"] + max_dur

    merged.sort(key=lambda w: w["score"], reverse=True)
    merged = merged[:MAX_CANDIDATES]

    # ── 7-Agent scoring ────────────────────────────────────────────────────
    log.info("Running 7-agent clip selection on %d candidates...", len(merged))

    # Load learned weights + entity biases from previous runs
    adaptive_weights = None
    weights_path = Path("clip_selection_weights.json")
    if weights_path.exists():
        try:
            with open(weights_path) as f:
                adaptive_weights = json.load(f)
            log.info("Loaded adaptive weights from %s", weights_path)
        except Exception:
            pass

    entity_biases = {}
    biases_path = Path("clip_selection_biases.json")
    if biases_path.exists():
        try:
            with open(biases_path) as f:
                entity_biases = json.load(f)
            log.info("Loaded entity biases: %d top players, %d avoid",
                     len(entity_biases.get("top_players", [])),
                     len(entity_biases.get("avoid_players", [])))
        except Exception:
            pass

    selector = ClipSelector(
        use_llm_arbiter=cfg.get("clip_selection", {}).get("use_llm_arbiter", True),
        weights=adaptive_weights,
    )

    # Load match context
    match_context = {}
    match_file = Path(paths["input"]) / "match_context.json"
    if match_file.exists():
        try:
            with open(match_file) as f:
                match_context = json.load(f)
        except Exception:
            pass

    context_for_agents = {
        "rms_map": rms_map,
        "avg_rms": avg_rms,
        "max_rms": max_rms,
        "transcript_segments": segments,
        "match_context": match_context,
        "entity_bias": entity_biases,
    }

    # Score all candidates through 7 agents
    scored_candidates = selector.score_candidates(merged, context_for_agents)

    # Filter and select top clips
    min_quality = cfg.get("clip_selection", {}).get("min_quality", 20.0)
    max_selected = cfg.get("clip_selection", {}).get("max_selected", MAX_SELECTED_CLIPS)
    top = selector.select(scored_candidates, context_for_agents,
                          max_selected=max_selected, min_quality=min_quality)

    top.sort(key=lambda w: w["start"])

    # ── Build YAML output ─────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    yaml_data = {}
    highlights = []

    for i, w in enumerate(top, start=1):
        key = f"clip{i}"
        window_text_parts = [
            seg.get("text", "") for seg in segments
            if seg["end"] > w["start"] and seg["start"] < w["end"]
        ]
        window_text = " ".join(window_text_parts).strip() or "Cricket Highlights"

        yaml_data[key] = {
            "start": fmt_ts(w["start"]),
            "end": fmt_ts(w["end"]),
            "start_sec": round(w["start"], 2),
            "end_sec": round(w["end"], 2),
            "score": round(w.get("final_score", w.get("score", 0)), 2),
            "text": window_text,
        }

        if "agent_scores" in w:
            yaml_data[key]["agent_scores"] = {
                name: {"score": data["score"], "reasoning": data.get("reasoning", "")}
                for name, data in w["agent_scores"].items()
            }
        if "final_score" in w:
            yaml_data[key]["final_score"] = w["final_score"]
        if "hook_score" in w:
            yaml_data[key]["hook_score"] = w["hook_score"]

        highlights.append({
            "id": key,
            "start": w["start"],
            "end": w["end"],
            "start_ts": fmt_ts(w["start"]),
            "end_ts": fmt_ts(w["end"]),
            "score": w.get("final_score", w.get("score", 0)),
            "text": window_text,
            "agent_scores": w.get("agent_scores", {}),
            "hook_score": w.get("hook_score"),
        })

        log.info("  %s: %s -> %s (score=%.1f)", key, fmt_ts(w["start"]),
                 fmt_ts(w["end"]), w.get("final_score", 0))

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

    log.info("Highlights saved -> %s (%d clips)", output_path, len(highlights))
    return highlights
