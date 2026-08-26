"""Pipeline integration — drop-in replacement for highlight.detect_highlights().

Usage:
    from automation.clip_selection.pipeline import detect_highlights
    highlights = detect_highlights(transcript_path, video_path, highlights_path)
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from utils.config import load_config
from utils.logger import get_logger

from automation.clip_selection.selector import ClipSelector
from automation.clip_selection.arbiter import fmt_ts
from automation.clip_selection.topic_segmenter import TopicSegmenter
from automation.clip_selection.cricket_heuristics import score_all_topics

from prompts import MAX_CANDIDATES, MAX_SELECTED_CLIPS, MIN_QUALITY_THRESHOLD

cfg = load_config()
log = get_logger("clip_pipeline")


def _sentence_parts(text: str) -> list[str]:
    """Split text only at explicit sentence/thought punctuation."""
    parts = re.findall(r"[^.!?।]+(?:[.!?।]+|$)", text or "")
    return [part.strip() for part in parts if part.strip()]


def _valid_word_timings(segment: dict) -> list[dict]:
    words = []
    for word in segment.get("words", []) or []:
        try:
            start = float(word["start"])
            end = float(word["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            words.append({"start": start, "end": end, "word": word.get("word", "")})
    return sorted(words, key=lambda item: item["start"])


def _merge_caption_fragments(segments: list[dict]) -> list[dict]:
    """Join rolling YouTube caption cues until a spoken sentence completes."""
    merged: list[dict] = []
    current: dict | None = None
    for segment in segments or []:
        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        text = re.sub(r"\s+", " ", str(segment.get("text", ""))).strip()
        if end <= start or not text:
            continue

        if current is not None and start > float(current["end"]) + 1.0:
            merged.append(current)
            current = None

        if current is None:
            current = {"start": start, "end": end, "text": text}
            if segment.get("words"):
                current["words"] = list(segment["words"])
        else:
            current["end"] = max(float(current["end"]), end)
            current["text"] = f"{current['text']} {text}".strip()
            if segment.get("words"):
                current.setdefault("words", []).extend(segment["words"])

        if re.search(r"[.!?।]+[\"')\]]*\s*$", text):
            merged.append(current)
            current = None

    if current is not None:
        merged.append(current)
    return merged


def _split_segments_on_pauses(
    segments: list[dict],
    pause_seconds: float = 0.55,
    max_block_seconds: float = 20.0,
    words_per_second: float = 2.5,
) -> list[dict]:
    """Split whisper's long punctuation-free blocks at speech pauses.

    whisper.cpp emits ~30s blocks with no sentence punctuation, so the
    complete-thought merger cannot find boundaries and glues whole minutes
    into a single 'thought'. Word timestamps carry real silence gaps — cut
    there; when timings are missing, hard-split oversized blocks
    proportionally by word count. Boundaries stay inside the source segment;
    no text is dropped or reordered.
    """
    out: list[dict] = []
    max_words = max(4, int(max_block_seconds * words_per_second))
    for segment in segments:
        try:
            seg_start = float(segment["start"])
            seg_end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if seg_end <= seg_start or not str(segment.get("text", "")).strip():
            continue
        words = _valid_word_timings(segment)

        # Boundary indices where the speaker paused.
        bounds: list[int] = [0]
        if len(words) >= 2:
            for i in range(1, len(words)):
                gap = float(words[i]["start"]) - float(words[i - 1]["end"])
                if gap >= pause_seconds:
                    bounds.append(i)
        bounds.append(len(words))

        pieces: list[tuple[float, float, list[str]]] = []
        if len(words) >= 1 and len(bounds) > 2:
            for a, b in zip(bounds, bounds[1:]):
                if b <= a:
                    continue
                toks = [str(w.get("word", "")).strip() for w in words[a:b]]
                pieces.append((
                    float(words[a]["start"]), float(words[b - 1]["end"]), toks,
                ))
        else:
            pieces.append((
                seg_start, seg_end,
                [t for t in str(segment.get("text", "")).split()],
            ))

        for p_start, p_end, toks in pieces:
            toks = [t for t in toks if t]
            span = max(p_end - p_start, 0.01)
            n_chunks = max(
                1,
                int(-(-span // max_block_seconds)),
                int(-(-len(toks) // max_words)),
            )
            per_chunk = -(-len(toks) // n_chunks)
            for idx in range(n_chunks):
                chunk = toks[idx * per_chunk:(idx + 1) * per_chunk]
                if not chunk:
                    continue
                frac_a = (idx * per_chunk) / max(len(toks), 1)
                frac_b = ((idx + 1) * per_chunk) / max(len(toks), 1)
                c_start = p_start + (p_end - p_start) * min(frac_a, 1.0)
                c_end = p_start + (p_end - p_start) * min(frac_b, 1.0)
                out.append({
                    "start": round(c_start, 3),
                    "end": round(c_end, 3),
                    "text": " ".join(chunk),
                })
    return out


def _prepare_complete_thoughts(segments: list[dict]) -> list[dict]:
    """Canonicalize entities, split complete thoughts, and trim outer silence."""
    from automation.seo.cricket_context import correct_cricket_spelling

    prepared: list[dict] = []
    for segment in _merge_caption_fragments(segments):
        try:
            seg_start = float(segment["start"])
            seg_end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if seg_end <= seg_start:
            continue
        original_parts = _sentence_parts(str(segment.get("text", "")))
        if not original_parts:
            continue

        timings = _valid_word_timings(segment)
        counts = [max(1, len(re.findall(r"\b\w+\b", part))) for part in original_parts]
        total_count = sum(counts)
        cursor = 0

        for index, (part, count) in enumerate(zip(original_parts, counts)):
            if timings:
                next_cursor = len(timings) if index == len(counts) - 1 else round(
                    len(timings) * sum(counts[:index + 1]) / total_count
                )
                next_cursor = max(cursor + 1, min(len(timings), next_cursor))
                selected_words = timings[cursor:next_cursor]
                start = max(seg_start, selected_words[0]["start"] - 0.1)
                end = min(seg_end, selected_words[-1]["end"] + 0.1)
                cursor = next_cursor
            else:
                elapsed_before = sum(counts[:index]) / total_count
                elapsed_after = sum(counts[:index + 1]) / total_count
                start = seg_start + (seg_end - seg_start) * elapsed_before
                end = seg_start + (seg_end - seg_start) * elapsed_after

            prepared.append({
                "start": round(start, 3),
                "end": round(end, 3),
                "text": correct_cricket_spelling(part),
            })

    # Word margins can overlap at a sentence boundary. Resolve to one clean cut.
    for index in range(1, len(prepared)):
        previous = prepared[index - 1]
        current = prepared[index]
        if previous["end"] > current["start"]:
            boundary = round((previous["end"] + current["start"]) / 2.0, 3)
            previous["end"] = boundary
            current["start"] = boundary
    return [item for item in prepared if item["end"] > item["start"]]


def _filter_cricket_candidates(candidates: list[dict], source_context: str) -> list[dict]:
    """Remove setup, gaming, tech, and other non-cricket chatter."""
    from automation.seo.cricket_context import is_cricket_content
    return [
        candidate for candidate in candidates
        if is_cricket_content(str(candidate.get("text", "")), source_context)
    ]


def _filter_source_match_candidates(
    candidates: list[dict],
    source_title: str,
    minimum_matches: int = 3,
) -> list[dict]:
    """Prefer the advertised match without turning a generic stream into a hard gate."""
    from automation.seo.cricket_context import find_canonical_entities

    source_teams = set(find_canonical_entities(source_title).get("teams", []))
    if len(source_teams) < 2:
        return candidates
    matched = [
        candidate for candidate in candidates
        if source_teams.intersection(
            find_canonical_entities(str(candidate.get("text", ""))).get("teams", [])
        )
    ]
    return matched if len(matched) >= max(1, int(minimum_matches)) else candidates


def _load_source_context(input_dir: str | Path) -> str:
    """Load title and match metadata used to disambiguate short utterances."""
    pieces = []
    for filename in ("video_metadata.json", "match_context.json"):
        path = Path(input_dir) / filename
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            pieces.append(json.dumps(data, ensure_ascii=False))
        except (OSError, json.JSONDecodeError, TypeError):
            continue
    return " ".join(pieces)


def _load_source_title(input_dir: str | Path) -> str:
    """Return the source title without sending its keyword-stuffed description to AI."""
    path = Path(input_dir) / "video_metadata.json"
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return str(data.get("title", "") or "").strip()[:240]
    except (OSError, json.JSONDecodeError, TypeError):
        return ""


def _write_empty_highlights(output_path: str | Path) -> None:
    """Atomically clear stale highlights when a source has no cricket clips."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        yaml.safe_dump({}, handle)
    temporary.replace(path)


def _match_key(input_dir: str, output_path: str) -> str:
    """Return a stable dedup-history key for the current match.

    History must be namespaced per match, NOT per filename stem: the
    orchestrator always writes to ``input/video.mp4`` so the stem is the same
    for every match, and re-using one history file would make a new match's
    highlights get rejected against a *different* match's previously selected
    windows. The key is derived from ``video_metadata.json`` (URL, falling
    back to title) which ``download.py`` writes for every download. If no
    metadata exists, falls back to the output filename stem so direct
    ``detect_highlights`` callers still get isolated history.
    """
    metadata_file = Path(input_dir) / "video_metadata.json"
    try:
        with open(metadata_file, encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        meta = {}
    url = str(meta.get("url", "") or "").strip()
    if url:
        video_id = _extract_youtube_id(url)
        if video_id:
            return video_id
        return _slugify(url)
    title = str(meta.get("title", "") or "").strip()
    if title:
        return _slugify(title)
    return Path(output_path).stem


def _extract_youtube_id(url: str) -> str | None:
    """Extract a YouTube video ID from a watch/shorts/live/embed URL."""
    import re
    m = re.search(r"(?:v=|youtu\.be/|shorts/|live/|embed/)([A-Za-z0-9_-]{11})(?:[?&#/]|$)", url)
    return m.group(1) if m else None


def _slugify(text: str, max_len: int = 64) -> str:
    """Normalize an arbitrary string into a filesystem-safe key."""
    slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in text)
    slug = "-".join(part for part in slug.split("-") if part)
    return (slug[:max_len] or "video").rstrip("-").lower()


def _compute_speed_factor(
    window_duration: float,
    target_duration: float,
    max_speedup: float,
    output_min: float = 0.0,
    output_max: float | None = None,
) -> float:
    """Return the speed multiplier for a window, normalizing EXPORT duration.

    Base behaviour: windows at or below ``target_duration`` run at 1.0x,
    longer windows are compressed to complete within the target seconds,
    capped at ``max_speedup``.

    When ``output_min``/``output_max`` are set, the resulting exported
    duration (``window / speed``) is normalized into that window by adjusting
    the speed factor only. Clip boundaries are never altered here — a window
    shorter than ``output_min`` stays at 1.0x (stretching audio is worse than
    a short complete thought) and an over-cap window keeps the cap (the caller
    logs when the ceiling cannot be met).
    """
    if window_duration <= 0 or target_duration <= 0:
        return 1.0
    cap = max(1.0, max_speedup)
    if window_duration <= target_duration:
        speed = 1.0
    else:
        speed = min(cap, window_duration / target_duration)

    def _clamp(value: float, lo: float, hi: float | None) -> float:
        if value < lo:
            return lo
        if hi is not None and value > hi:
            return hi
        return value

    out_floor = max(0.0, float(output_min or 0.0))
    out_ceiling = float(output_max) if output_max else None
    raw_output = window_duration / max(speed, 1e-9)
    target_output = _clamp(raw_output, out_floor, out_ceiling)
    if abs(target_output - raw_output) < 1e-6:
        return round(speed, 2)
    adjusted = window_duration / target_output
    if adjusted < 1.0:
        return round(speed, 2)
    return round(min(cap, adjusted), 2)


def _build_clip_yaml_entry(
    start: float,
    end: float,
    score: float,
    text: str,
    target_duration: float,
    max_speedup: float,
    output_min: float = 0.0,
    output_max: float | None = None,
) -> dict:
    """Build a clip YAML entry, attaching the duration-compression speed_factor."""
    duration = max(0.0, end - start)
    return {
        "start": fmt_ts(start),
        "end": fmt_ts(end),
        "start_sec": round(start, 2),
        "end_sec": round(end, 2),
        "score": round(score, 2),
        "speed_factor": _compute_speed_factor(
            duration, target_duration, max_speedup,
            output_min=output_min, output_max=output_max,
        ),
        "text": text,
    }


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
        if w["start"] - prev["end"] < gap:
            prev["end"] = max(prev["end"], w["end"])
            prev["score"] = max(prev["score"], w["score"])
            # Merge text from both windows (longer text wins)
            curr_text = w.get("text", "")
            prev_text = prev.get("text", "")
            if len(curr_text) > len(prev_text):
                prev["text"] = curr_text
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
    match_key: str | None = None,
) -> list[dict]:
    """Replace ``highlight.detect_highlights()`` with 7-agent clip selection.

    Same signature, same YAML output format — drop-in replacement.

    Args:
        transcript_path: Path to transcript JSON (defaults to config paths).
        video_path: Path to source video (defaults to config paths).
        output_path: Path to write highlights YAML.
        match_key: Explicit dedup-history namespace (e.g. the YouTube video ID
            derived from the run URL). When provided it wins over the
            ``video_metadata.json`` lookup, so flows that never write metadata
            (``--skip-download``, drive sync) still get per-match isolation
            instead of collapsing onto the constant ``video`` stem.
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
        raise FileNotFoundError(f"Transcript not found: {t_path}")

    with open(t_path, encoding="utf-8") as f:
        data = json.load(f)
    segments = data if isinstance(data, list) else data.get("segments", [])
    # Whisper emits Devanagari for language=hi, but every downstream signal
    # (keyword lists, classifiers, SEO grounding) operates on Roman-Hinglish.
    # Transliterate once here so selection sees what the audience speaks.
    from utils.devanagari import to_roman
    for seg in segments:
        seg["text"] = to_roman(seg.get("text", ""))
    segments = _split_segments_on_pauses(segments)
    segments = _prepare_complete_thoughts(segments)
    source_context = _load_source_context(paths["input"])
    source_title = _load_source_title(paths["input"])
    stream_context = " ".join([source_context] + [
        str(segment.get("text", "")) for segment in segments
    ])
    from automation.seo.cricket_context import is_cricket_content
    if not is_cricket_content(stream_context):
        _write_empty_highlights(output_path)
        log.warning("Cricket-only gate rejected non-cricket source; wrote empty highlights")
        return []
    segments = _filter_cricket_candidates(segments, source_context)
    if not segments:
        _write_empty_highlights(output_path)
        log.warning("Cricket-only gate removed every non-cricket segment")
        return []
    log.info("Loaded %d complete cricket thoughts from %s", len(segments), t_path)

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

    # ── Topic segmentation + cricket heuristics ───────────────────────────
    topics = []
    topic_heuristics = {}
    try:
        segmenter = TopicSegmenter()
        topics = segmenter.segment(segments)
        heuristic_scores = score_all_topics(topics)
        for t, hs in zip(topics, heuristic_scores):
            t.update(hs)
            cand_key = f"{t['start']:.1f}-{t['end']:.1f}"
            topic_heuristics[cand_key] = hs
        log.info("Topic segmentation: %d topics found", len(topics))
    except Exception as e:
        log.warning("Topic segmentation failed: %s — continuing without", e)

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

    windows = []
    for c in candidates:
        seg_duration = c["end"] - c["start"]
        if seg_duration < min_dur:
            continue
        win_start = c["start"]
        win_end = c["end"]
        windows.append({"start": win_start, "end": win_end, "score": c["score"], "text": c.get("text", "")})

    windows.sort(key=lambda w: w["start"])
    merged = _merge_windows(windows, h_cfg["merge_gap"])

    # Stamp every candidate with its content angle so the LLM arbiter can
    # weigh it and shorts_intelligence can learn which angles win.
    from automation.clip_selection.content_type import classify_content_type
    for w in merged:
        w["content_type"] = classify_content_type(str(w.get("text", "")))

    merged.sort(key=lambda w: w["score"], reverse=True)
    selection_cfg = cfg.get("clip_selection", {})
    if selection_cfg.get("prefer_source_match", True):
        before = len(merged)
        merged = _filter_source_match_candidates(
            merged,
            source_title,
            minimum_matches=int(selection_cfg.get("source_match_min_candidates", 3)),
        )
        if len(merged) != before:
            log.info("Source-match filter: %d/%d candidates match %s", len(merged), before, source_title)
    merged = merged[:MAX_CANDIDATES]

    # ── 7-Agent scoring ────────────────────────────────────────────────────
    log.info("Running 7-agent clip selection on %d candidates...", len(merged))

    selector = ClipSelector(
        use_llm_arbiter=cfg.get("clip_selection", {}).get("use_llm_arbiter", True),
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
        "topics": topics,
        "topic_heuristics": topic_heuristics,
        "source_title": source_title,
    }

    # Cross-run dedup: reject windows overlapping previously selected clips
    # (same match re-processed → same highlight windows get re-selected).
    # History is a sidecar file so it survives yaml overwrites every run.
    from automation.clip_selection.dedupe import load_previous_windows
    dedup_cfg = cfg.get("clip_selection", {})
    if match_key:
        key = match_key
    else:
        key = _match_key(paths["input"], output_path)
    history_path = str(Path(paths["highlights"]) / f"{key}.dedupe_history.yaml")
    previous_windows = []
    if dedup_cfg.get("dedup_enabled", True):
        previous_windows = load_previous_windows(history_path)
    if previous_windows:
        context_for_agents["previous_windows"] = previous_windows
        context_for_agents["dedup_overlap_threshold"] = dedup_cfg.get("dedup_overlap_threshold", 0.6)
        log.info("Cross-run dedup: %d previously selected windows loaded from %s",
                 len(previous_windows), history_path)

    # Score all candidates through 7 agents
    scored_candidates = selector.score_candidates(merged, context_for_agents)

    # Canonical learner is a bounded tie-breaker only. Every candidate here was
    # already built by _prepare_complete_thoughts(), so it cannot shorten or cut.
    for candidate in scored_candidates:
        candidate["complete_thought"] = True
    try:
        from shorts_intelligence.bridge import apply_selection_policy

        matched = apply_selection_policy(cfg, scored_candidates)
        if matched:
            mode = "shadow" if cfg.get("shorts_intelligence", {}).get("shadow_mode", True) else "active"
            log.info("Shorts Intelligence %s policy matched %d candidates", mode, matched)
    except Exception as exc:
        log.warning("Shorts Intelligence policy unavailable: %s", exc)

    # Filter and select top clips
    min_quality = cfg.get("clip_selection", {}).get("min_quality", 20.0)
    max_selected = cfg.get("clip_selection", {}).get("max_selected", MAX_SELECTED_CLIPS)
    top = selector.select(scored_candidates, context_for_agents,
                          max_selected=max_selected, min_quality=min_quality)

    top.sort(key=lambda w: w["start"])

    target_duration = float(h_cfg.get("target_duration", 22.0))
    max_speedup = float(h_cfg.get("max_speedup", 1.6))
    sel_cfg = cfg.get("clip_selection", {})
    output_min = float(sel_cfg.get("output_min_duration", 0) or 0)
    output_max_raw = float(sel_cfg.get("output_max_duration", 0) or 0)
    output_max: float | None = output_max_raw if output_max_raw > 0 else None

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

        yaml_data[key] = _build_clip_yaml_entry(
            start=w["start"],
            end=w["end"],
            score=w.get("final_score", w.get("score", 0)),
            text=window_text,
            target_duration=target_duration,
            max_speedup=max_speedup,
            output_min=output_min,
            output_max=output_max,
        )

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
            "speed_factor": yaml_data[key]["speed_factor"],
            "text": window_text,
            "content_type": w.get("content_type"),
            "agent_scores": w.get("agent_scores", {}),
            "hook_score": w.get("hook_score"),
            "intelligence_adjustment": w.get("intelligence_adjustment", 0.0),
            "intelligence_shadow_adjustment": w.get(
                "intelligence_shadow_adjustment", 0.0
            ),
            "intelligence_segments": w.get("intelligence_segments", []),
        })

        log.info("  %s: %s -> %s (score=%.1f, speed=%.2fx)", key, fmt_ts(w["start"]),
                 fmt_ts(w["end"]), w.get("final_score", 0),
                 yaml_data[key]["speed_factor"])

    # Always commit the current selection atomically. An empty mapping clears
    # stale highlights so the export phase cannot re-export a previous run.
    tmp_output = Path(output_path).with_suffix(".tmp")
    with open(tmp_output, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
    tmp_output.replace(output_path)
    if top:
        log.info("Highlights saved -> %s (%d clips)", output_path, len(highlights))
    else:
        log.warning("No clips selected — wrote empty highlights mapping to %s", output_path)

    return highlights
