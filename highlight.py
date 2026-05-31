import argparse
import json
import subprocess
import sys
import wave
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import numpy as np
import yaml

from utils.config import load_config
from utils.logger import get_logger
from utils.ai_client import AIClient

cfg = load_config()
log = get_logger("highlight", cfg["logging"]["log_file"], cfg["logging"]["level"])
ai = AIClient()


# --- Audio energy extraction ---

def _extract_audio_rms(video_path: str, chunk_seconds: float = 1.0) -> List[Tuple[float, float]]:
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
    rms_values: list = []
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


def _get_video_duration(video_path: str) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


# --- Heuristic scoring ---

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


def _score_segment(seg: dict, rms_map: dict, avg_rms: float, max_rms: float, h_cfg: dict) -> float:
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

    if len(buckets) >= 4:
        first_half = buckets[:len(buckets) // 2]
        second_half = buckets[len(buckets) // 2:]
        avg_first = sum(first_half) / len(first_half) if first_half else 0
        avg_second = sum(second_half) / len(second_half) if second_half else 0
        if avg_second > avg_first * 1.3 and avg_first > 0:
            score += 0.8

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


def _merge_windows(windows: List[Dict], gap: float) -> List[Dict]:
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
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# --- 6-dimension parallel scoring ---

def _score_candidate_dimensions(
    candidate: dict,
    segments: List[dict],
    rms_map: dict,
    avg_rms: float,
    max_rms: float,
) -> dict:
    """Score a single candidate across 6 dimensions in parallel."""
    start = candidate["start"]
    end = candidate["end"]
    duration = max(end - start, 0.1)

    overlap_text = " ".join(
        s.get("text", "") for s in segments
        if s["end"] > start and s["start"] < end
    )

    buckets = [rms_map.get(int(t), 0.0) for t in range(int(start), int(end) + 1)]
    seg_rms = sum(buckets) / len(buckets) if buckets else 0.0

    hook_buckets = [rms_map.get(int(t), 0.0) for t in range(int(start), min(int(start) + 2, int(end) + 1))]
    hook_energy = sum(hook_buckets) / len(hook_buckets) if hook_buckets else 0.0

    # 1. Hook strength (0-10)
    hook_strength = 0.0
    if avg_rms > 0 and hook_energy > avg_rms * 1.5:
        hook_strength += 4.0
    elif avg_rms > 0 and hook_energy > avg_rms * 1.2:
        hook_strength += 2.0
    first_words = overlap_text[:50].lower()
    reaction_openers = {"oh", "wow", "wait", "what", "no", "yes", "bro", "arre", "kya", "dekho"}
    if any(first_words.startswith(w) for w in reaction_openers):
        hook_strength += 3.0
    if "!" in overlap_text[:30]:
        hook_strength += 1.5
    hook_strength = min(10.0, hook_strength)

    # 2. Clarity (0-10)
    word_count = len(overlap_text.split())
    wpm = _words_per_minute(overlap_text, duration)
    clarity = 5.0
    if word_count >= 10:
        clarity += 2.0
    elif word_count >= 5:
        clarity += 1.0
    if 100 <= wpm <= 200:
        clarity += 2.0
    silence_ratio = _silence_seconds(overlap_text, duration) / duration if duration > 0 else 0
    if silence_ratio > 0.5:
        clarity -= 3.0
    clarity = max(0.0, min(10.0, clarity))

    # 3. Emotional peak (0-10)
    emotional_peak = 0.0
    if avg_rms > 0:
        emotional_peak += min(5.0, (seg_rms / avg_rms) * 2.5)
    if max_rms > 0:
        spike_count = sum(1 for v in buckets if v > max_rms * 0.85)
        emotional_peak += min(3.0, spike_count * 1.0)
    reaction_words = {"oh", "wow", "insane", "crazy", "unbelievable", "incredible", "amazing",
                      "wicket", "six", "four", "out", "catch", "shot", "brilliant",
                      "arre", "kya", "bhai", "chhakka", "chauka", "maar"}
    words_set = set(re.findall(r'\b\w+\b', overlap_text.lower()))
    emotional_peak += min(2.0, len(words_set & reaction_words) * 0.5)
    emotional_peak = min(10.0, emotional_peak)

    # 4. Topic completeness (0-10)
    topic_completeness = 4.0
    if word_count >= 15:
        topic_completeness += 2.0
    elif word_count >= 8:
        topic_completeness += 1.0
    if 10 <= duration <= 25:
        topic_completeness += 2.0
    elif 5 <= duration < 10:
        topic_completeness += 1.0
    if overlap_text.strip().endswith((".", "!", "?")):
        topic_completeness += 1.0
    topic_completeness = min(10.0, topic_completeness)

    # 5. Punchline/payoff (0-10)
    payoff = 0.0
    last_3s = [rms_map.get(int(t), 0.0) for t in range(max(int(start), int(end) - 3), int(end) + 1)]
    if last_3s and avg_rms > 0:
        end_energy = sum(last_3s) / len(last_3s)
        if end_energy > avg_rms * 1.3:
            payoff += 4.0
    excl_count = overlap_text.count("!")
    payoff += min(3.0, excl_count * 1.0)
    if overlap_text.rstrip().endswith("!"):
        payoff += 2.0
    payoff = min(10.0, payoff)

    # 6. Cut safety (0-10)
    cut_safety = 5.0
    if silence_ratio > 0.2:
        cut_safety += 2.0
    if overlap_text.strip().endswith((".", "!", "?")):
        cut_safety += 1.5
    if duration >= 8:
        cut_safety += 1.0
    if duration > 28:
        cut_safety -= 2.0
    cut_safety = max(0.0, min(10.0, cut_safety))

    # 7. Replay value (0-10)
    replay = 0.0
    if emotional_peak >= 7:
        replay += 3.0
    elif emotional_peak >= 5:
        replay += 1.5
    if hook_strength >= 7:
        replay += 2.0
    words_list = re.findall(r'\b\w+\b', overlap_text.lower())
    for w in set(words_list):
        if words_list.count(w) >= 3 and len(w) > 1:
            replay += 2.0
            break
    if len(buckets) >= 4:
        first_half = buckets[:len(buckets) // 2]
        second_half = buckets[len(buckets) // 2:]
        avg_first = sum(first_half) / len(first_half) if first_half else 0
        avg_second = sum(second_half) / len(second_half) if second_half else 0
        if avg_second > avg_first * 1.3 and avg_first > 0:
            replay += 1.5
    replay = min(10.0, replay)

    return {
        "hook_strength": round(hook_strength, 2),
        "clarity": round(clarity, 2),
        "emotional_peak": round(emotional_peak, 2),
        "topic_completeness": round(topic_completeness, 2),
        "punchline_or_payoff": round(payoff, 2),
        "cut_safety": round(cut_safety, 2),
        "replay_value": round(replay, 2),
    }


def _compute_weighted_score(scores: dict, weights: dict) -> float:
    return sum(scores.get(dim, 0) * weights.get(dim, 0) for dim in weights)


def _parallel_score_candidates(
    candidates: List[dict],
    segments: List[dict],
    rms_map: dict,
    avg_rms: float,
    max_rms: float,
    max_workers: int = 8,
) -> List[dict]:
    """Score all candidates across 6 dimensions using parallel ThreadPoolExecutor."""
    from prompts import DEFAULT_SCORING_WEIGHTS

    scored = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _score_candidate_dimensions, c, segments, rms_map, avg_rms, max_rms
            ): c
            for c in candidates
        }
        for future in as_completed(futures):
            candidate = futures[future]
            try:
                dimensions = future.result()
                weighted = _compute_weighted_score(dimensions, DEFAULT_SCORING_WEIGHTS)
                candidate_with_scores = dict(candidate)
                candidate_with_scores["dimension_scores"] = dimensions
                candidate_with_scores["weighted_score"] = round(weighted, 4)
                scored.append(candidate_with_scores)
            except Exception as e:
                log.warning("Scoring failed for candidate %.2f-%.2f: %s",
                            candidate["start"], candidate["end"], e)
                candidate_with_scores = dict(candidate)
                candidate_with_scores["dimension_scores"] = {}
                candidate_with_scores["weighted_score"] = candidate.get("score", 0)
                scored.append(candidate_with_scores)

    scored.sort(key=lambda x: x["weighted_score"], reverse=True)
    return scored


# --- AI refinement with new prompts ---

def _refine_highlights_with_ai(
    segments: List[dict],
    candidates: List[dict],
    video_title: str = "",
    max_clips: int = 10,
) -> List[dict]:
    """Use LLM ranker prompt to select best clips from candidates."""
    if not candidates or len(candidates) <= 1:
        return candidates

    from prompts import HIGHLIGHT_RANKER_SYSTEM, HIGHLIGHT_RANKER_USER_TEMPLATE, MAX_CANDIDATES

    capped_candidates = candidates[:MAX_CANDIDATES]

    candidate_min = min(c["start"] for c in capped_candidates)
    candidate_max = max(c["end"] for c in capped_candidates)
    overlapping_segments = [
        seg for seg in segments
        if seg["end"] > candidate_min and seg["start"] < candidate_max
    ]

    transcript_snippets = []
    for seg in overlapping_segments[:40]:
        transcript_snippets.append(f"[{_format_ts(seg['start'])}] {seg.get('text', '')}")
    transcript_text = "\n".join(transcript_snippets)

    candidate_list = []
    for i, c in enumerate(capped_candidates, 1):
        overlap_text = " ".join(
            s.get("text", "") for s in segments
            if s["end"] > c["start"] and s["start"] < c["end"]
        )[:150]
        dims = c.get("dimension_scores", {})
        dims_str = " | ".join(f"{k}:{v}" for k, v in dims.items()) if dims else ""
        candidate_list.append(
            f"{i}. [{_format_ts(c['start'])}-{_format_ts(c['end'])}] "
            f"weighted={c.get('weighted_score', 0):.2f} dims=[{dims_str}] "
            f"text={overlap_text[:120]}"
        )
    candidates_str = "\n".join(candidate_list)

    user_prompt = HIGHLIGHT_RANKER_USER_TEMPLATE.format(
        video_title=video_title,
        candidate_count=len(capped_candidates),
        transcript_context=transcript_text,
        candidate_list=candidates_str,
    )

    try:
        log.info("Sending %d candidates to AI ranker...", len(capped_candidates))
        response = ai.generate_text(user_prompt, system_instruction=HIGHLIGHT_RANKER_SYSTEM)

        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            result = json.loads(match.group(0))
            selected = result.get("selected", [])
            rejected = result.get("rejected", [])
            notes = result.get("notes", {})

            log.info("AI ranker: %d selected, %d rejected, quality=%s",
                     len(selected), len(rejected), notes.get("overall_quality", "unknown"))

            selected_ids = {s.get("candidate_id") for s in selected}
            refined = []
            for sel in selected:
                try:
                    idx = int(sel.get("candidate_id", "").replace("candidate_", "")) - 1
                except (ValueError, TypeError):
                    idx = -1
                if 0 <= idx < len(capped_candidates):
                    c = dict(capped_candidates[idx])
                    c["ai_score"] = sel.get("score", c.get("weighted_score", 0))
                    c["ai_reason"] = sel.get("reason", "")
                    c["ai_confidence"] = sel.get("confidence", 0.5)
                    if "best_start_sec" in sel:
                        c["start"] = sel["best_start_sec"]
                    if "best_end_sec" in sel:
                        c["end"] = sel["best_end_sec"]
                    refined.append(c)

            if len(refined) < max_clips:
                for i, c in enumerate(capped_candidates):
                    cid = f"candidate_{i+1}"
                    if cid not in selected_ids and len(refined) < max_clips:
                        refined.append(c)

            refined.sort(key=lambda x: x.get("ai_score", x.get("weighted_score", 0)), reverse=True)
            return refined[:max_clips]

    except Exception as e:
        log.warning("AI ranker failed: %s, using heuristic ranking", e)

    return candidates[:max_clips]


# --- Main detection logic ---

def detect_highlights(
    transcript_path: Optional[str] = None,
    video_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> List[Dict]:
    """Analyse transcript + audio energy, score candidates in parallel, rank with AI."""
    from prompts import MAX_CANDIDATES, MAX_SELECTED_CLIPS, MIN_QUALITY_THRESHOLD

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

    rms_list = _extract_audio_rms(video_path)
    from collections import defaultdict
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
        std_rms = 0.0

    log.info("Audio RMS - avg: %.4f | max: %.4f | std: %.4f", avg_rms, max_rms, std_rms)

    # Heuristic scoring pass
    scored: list = []
    for seg in segments:
        score = _score_segment(seg, rms_map, avg_rms, max_rms, h_cfg)
        scored.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg.get("text", ""),
            "score": round(score, 4),
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

    windows: list = []
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

    # Cap to MAX_CANDIDATES before parallel scoring
    merged = merged[:MAX_CANDIDATES]

    # Stage 4a: Parallel 6-dimension scoring
    log.info("Running parallel 6-dimension scoring on %d candidates...", len(merged))
    parallel_scored = _parallel_score_candidates(merged, segments, rms_map, avg_rms, max_rms)

    # Filter by minimum quality threshold
    quality_candidates = [
        c for c in parallel_scored
        if c.get("weighted_score", 0) >= MIN_QUALITY_THRESHOLD
    ]
    log.info("After quality filter: %d/%d candidates pass threshold %.1f",
             len(quality_candidates), len(parallel_scored), MIN_QUALITY_THRESHOLD)

    if not quality_candidates:
        log.warning("No candidates passed quality threshold — using top scored anyway")
        quality_candidates = parallel_scored[:5]

    # Stage 4b: AI ranker refinement
    video_title = ""
    meta_file = Path(cfg["paths"]["input"]) / "video_metadata.json"
    if meta_file.exists():
        try:
            with open(meta_file, "r") as f:
                meta = json.load(f)
                video_title = meta.get("title", "")
        except Exception:
            pass

    use_ai_refinement = cfg.get("highlight", {}).get("use_ai_refinement", True)
    if use_ai_refinement and len(quality_candidates) >= 2:
        top = _refine_highlights_with_ai(
            segments, quality_candidates, video_title, MAX_SELECTED_CLIPS
        )
    else:
        top = quality_candidates[:MAX_SELECTED_CLIPS]

    top.sort(key=lambda w: w["start"])

    # Build YAML output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    yaml_data: dict = {}
    highlights: list = []
    for i, w in enumerate(top, start=1):
        key = f"clip{i}"
        window_text_parts = [
            seg.get("text", "") for seg in segments
            if seg["end"] > w["start"] and seg["start"] < w["end"]
        ]
        window_text = " ".join(window_text_parts).strip() or "Cricket Highlights"

        yaml_data[key] = {
            "start": _format_ts(w["start"]),
            "end": _format_ts(w["end"]),
            "start_sec": round(w["start"], 2),
            "end_sec": round(w["end"], 2),
            "score": w.get("weighted_score", w.get("score", 0)),
            "text": window_text,
        }
        # Include dimension scores in YAML if available
        if "dimension_scores" in w:
            yaml_data[key]["dimension_scores"] = w["dimension_scores"]
        if "ai_score" in w:
            yaml_data[key]["ai_score"] = w["ai_score"]
        if "ai_reason" in w:
            yaml_data[key]["ai_reason"] = w["ai_reason"]

        highlights.append({
            "id": key,
            "start": w["start"],
            "end": w["end"],
            "start_ts": _format_ts(w["start"]),
            "end_ts": _format_ts(w["end"]),
            "score": w.get("weighted_score", w.get("score", 0)),
            "text": window_text,
            "dimension_scores": w.get("dimension_scores", {}),
        })
        log.info("  %s: %s -> %s (score=%.3f)", key, _format_ts(w["start"]),
                 _format_ts(w["end"]), w.get("weighted_score", w.get("score", 0)))

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

    log.info("Highlights saved -> %s (%d clips)", output_path, len(highlights))
    return highlights


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect highlights using audio + transcript heuristics + parallel scoring.")
    parser.add_argument("--transcript", "-t", default=None, help="Transcript JSON path")
    parser.add_argument("--video", "-v", default=None, help="Source video path")
    parser.add_argument("--output", "-o", default=None, help="Output highlights YAML path")
    args = parser.parse_args()
    detect_highlights(args.transcript, args.video, args.output)


if __name__ == "__main__":
    main()
