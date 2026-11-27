"""Clip discovery: transcript + audio -> ranked complete-thought windows -> editor.

Flow (one gate per predicate, cancer register C4):
  1. Source gate, once: is this stream about cricket at all?
  2. Window gate, once per window at birth: does the window's OWN speech talk
     cricket? Downstream stages (editor ranking, export, SEO) never re-ask.
  3. Rank windows with signals a viewer actually feels: loudness spike versus
     the stream's own baseline, reaction language, a hook in the first seconds,
     a payoff at the end, cricket density, and duration fit.
  4. The editorial LLM (automation.editorial_planner.curate_clips) reads the
     whole transcript and these ranked windows, picks complete thoughts, and
     word-aligns them. If it is down, the top windows go to owner review.

The ranking weights only order suggestions and the outage fallback; they never
hold a clip. They are UNMEASURED priors (A7): calibrate against
shorts_intelligence outcomes before trusting them further.
"""
from __future__ import annotations

import json
import math
import re
import tempfile
from pathlib import Path

import numpy as np
import yaml

from utils.config import load_config
from utils.logger import get_logger
from shorts_intelligence.completeness import completeness_value, set_completeness

cfg = load_config()
log = get_logger("clip_pipeline")


def fmt_ts(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    return f"{int(seconds // 3600):02d}:{int(seconds % 3600 // 60):02d}:{int(seconds % 60):02d}"


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
            raise ValueError("Invalid source word timing") from None
        if not (math.isfinite(start) and math.isfinite(end) and start >= 0 and end > start):
            raise ValueError("Invalid source word timing")
        words.append({**word, "start": start, "end": end})
    words.sort(key=lambda item: item["start"])
    if words and (words[0]["start"] < float(segment["start"]) - .02
                  or words[-1]["end"] > float(segment["end"]) + .02):
        raise ValueError("Source word timing exceeds its parent cue")
    if any(current["start"] < previous["end"] - .02 for previous, current in zip(words, words[1:])):
        raise ValueError("Overlapping source word timing requires review")
    return words


def _merge_caption_fragments(segments: list[dict], max_merge_seconds: float = 20.0) -> list[dict]:
    """Join short sentences; retain cue entry points when punctuation is sparse.

    A long punctuation group can straddle unrelated topics. In that case use
    its original cues for discovery, not one oversized, proportionally timed
    sentence. These are still unverified boundaries, not complete thoughts.
    """
    merged: list[dict] = []
    current: dict | None = None
    members: list[dict] = []

    def flush() -> None:
        if current is not None:
            if len(members) > 1 and current["end"] - current["start"] > max_merge_seconds:
                merged.extend(members)
            else:
                merged.append(current)

    for segment in segments or []:
        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        text = re.sub(r"\s+", " ", str(segment.get("text", ""))).strip()
        if end <= start or not text:
            continue

        if current is not None and (segment.get("discovery_break_before") or start > float(current["end"]) + 1.0):
            flush()
            current = None
            members = []

        members.append({**segment, "start": start, "end": end, "text": text})

        if current is None:
            current = {**segment, "start": start, "end": end, "text": text}
            if segment.get("words"):
                current["words"] = list(segment["words"])
        else:
            current["end"] = max(float(current["end"]), end)
            current["text"] = f"{current['text']} {text}".strip()
            current["source_cue_ids"] = sorted(set(current.get("source_cue_ids", []) + segment.get("source_cue_ids", [])))
            if segment.get("words"):
                current.setdefault("words", []).extend(segment["words"])

        if re.search(r"[.!?।]+[\"')\]]*\s*$", text):
            flush()
            current = None
            members = []

    flush()
    return merged


def _word_piece(segment: dict, words: list[dict], *, margin: float = 0) -> dict:
    """Keep discovery cuts outside source words, including their provenance."""
    return {
        **segment,
        "start": max(float(segment["start"]), words[0]["start"] - margin),
        "end": min(float(segment["end"]), words[-1]["end"] + margin),
        "text": " ".join(str(word.get("word", "")).strip() for word in words).strip(),
        "words": words,
        "boundary_method": "source_words",
        "complete_thought": None,
    }


def _split_segments_on_pauses(
    segments: list[dict],
    pause_seconds: float = 0.55,
    max_block_seconds: float = 20.0,
    words_per_second: float = 2.5,
) -> list[dict]:
    """Split whisper's long punctuation-free blocks at speech pauses.

    Word evidence is never replaced by proportional text timing. Cue-only
    subdivisions remain explicitly marked discovery estimates requiring ASR.
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

        if words:
            group = []
            for word in words:
                if group and (
                    word["start"] - group[-1]["end"] >= pause_seconds
                    or word["end"] - group[0]["start"] > max_block_seconds
                    or len(group) >= max_words
                ):
                    out.append({**_word_piece(segment, group), "discovery_break_before": True})
                    group = []
                group.append(word)
            if group:
                out.append({**_word_piece(segment, group), "discovery_break_before": True})
            continue

        pieces = [(seg_start, seg_end, str(segment.get("text", "")).split())]

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
                    **segment,
                    "start": round(c_start, 3),
                    "end": round(c_end, 3),
                    "text": " ".join(chunk),
                    "boundary_method": "estimated_cue_fraction",
                    "complete_thought": None,
                    "discovery_break_before": n_chunks > 1,
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
        if timings:
            group = []
            for word in timings:
                group.append(word)
                if re.search(r"[.!?।]+[\"')\]]*\s*$", str(word.get("word", ""))):
                    prepared.append(_word_piece(segment, group, margin=0.1))
                    group = []
            if group:
                prepared.append(_word_piece(segment, group, margin=0.1))
            continue
        counts = [max(1, len(re.findall(r"\b\w+\b", part))) for part in original_parts]
        total_count = sum(counts)
        for index, part in enumerate(original_parts):
            elapsed_before = sum(counts[:index]) / total_count
            elapsed_after = sum(counts[:index + 1]) / total_count
            start = seg_start + (seg_end - seg_start) * elapsed_before
            end = seg_start + (seg_end - seg_start) * elapsed_after

            prepared.append({
                **segment,
                "start": round(start, 3),
                "end": round(end, 3),
                "text": correct_cricket_spelling(part),
                "boundary_method": "estimated_cue_fraction",
                "complete_thought": None,
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
        # Keep the exact reviewed range; display timecodes are not edit points.
        "start_sec": float(start),
        "end_sec": float(end),
        "score": round(score, 2),
        "speed_factor": _compute_speed_factor(
            duration, target_duration, max_speedup,
            output_min=output_min, output_max=output_max,
        ),
        "text": text,
    }


def _extract_audio_rms(video_path: str, chunk_seconds: float = 1.0) -> list[tuple[float, float]]:
    temp_dir = Path(cfg["paths"]["temp"])
    temp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="selection-rms-", dir=temp_dir) as folder:
        return _read_audio_rms(video_path, str(Path(folder) / "audio.wav"), chunk_seconds)


def _read_audio_rms(video_path: str, pcm_path: str, chunk_seconds: float) -> list[tuple[float, float]]:
    import subprocess
    import wave
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


_LIVESTREAM_FILLER_PATTERNS = (
    "like the video", "like the stream", "like karate", "laaika karate",
    "subscribe", "sabsakraaiba", "do subscribe", "channel ko",
    "welcome to the stream", "welcome welcome", "velakama",
    "hello bro", "hello hello", "helo bro", "helo helo",
    "support banaye", "support bana", "saporta bana",
    "views ke liye", "vyooja ke lie", "this is strategy", "disa ija a stretajee",
    "live kar raha", "laaiva kara rahaa", "live match hota", "laaiva match hotaa",
    "1080p", "maapha keejiegaa", "idhar-udhara", "idhara-udhara",
)


def _is_livestream_filler(text: str) -> bool:
    """Identify greetings, chat replies, and channel CTA speech."""
    normalized = re.sub(r"\s+", " ", str(text or "").casefold()).strip()
    return any(pattern in normalized for pattern in _LIVESTREAM_FILLER_PATTERNS)


def _select_diverse_windows(
    windows: list[dict],
    max_candidates: int,
    overlap_threshold: float = 0.5,
) -> list[dict]:
    """Keep the strongest variant of a moment instead of near-duplicates."""
    limit = max(0, int(max_candidates))
    if not limit:
        return []
    selected: list[dict] = []
    ranked = sorted(windows, key=lambda item: (-item.get("score", 0), item["start"], item["end"]))
    # Reserve alternatives across source time before filling by score. This is
    # discovery diversity, not a quota of final releases or proof of full review.
    if len(ranked) > limit:
        earliest = min(item["start"] for item in ranked)
        span = max(item["end"] for item in ranked) - earliest
        bins = min(6, limit)
        representatives = {}
        for item in ranked:
            bucket = min(bins - 1, int((item["start"] - earliest) / max(span, .001) * bins))
            representatives.setdefault(bucket, item)
        reserved = list(representatives.values())
        reserved_ids = {id(item) for item in reserved}
        ranked = reserved + [item for item in ranked if id(item) not in reserved_ids]
    for candidate in ranked:
        start = float(candidate["start"])
        end = float(candidate["end"])
        duration = max(end - start, 0.0)
        if duration <= 0:
            continue
        duplicate = False
        for existing in selected:
            overlap = max(
                0.0,
                min(end, float(existing["end"]))
                - max(start, float(existing["start"])),
            )
            shorter = min(duration, float(existing["end"]) - float(existing["start"]))
            if shorter > 0 and overlap / shorter >= overlap_threshold:
                duplicate = True
                break
        if duplicate:
            continue
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return sorted(selected, key=lambda item: (-item.get("score", 0), item["start"], item["end"]))


# ── Cricket label (a ranking input, not a gate) ──────────────────────────────
# Owner policy 2026-11-27: a cricket stream's Shorts may be non-cricket banter or
# a live moment. The only cricket gate is the source gate in detect_highlights;
# a window's own cricket talk is labelled and ranked, never required.

def is_cricket_window(text: str) -> bool:
    """True when the window talks cricket in its OWN words, densely enough that
    a lone name inside a long tangent does not count."""
    return _has_dense_cricket_evidence(text)


# ── Ranking signals ──────────────────────────────────────────────────────────

# Lexicons carry BOTH casual Hinglish and the romanizer's spelling of Whisper/
# YouTube Devanagari (to_roman keeps the inherent vowel: क्या -> kyaa, यार -> yaara).
# Before 2026-11-27 only casual spellings were listed and the signal matched ~0
# words on 3 real streams. Words too common to signal anything (gaya, kya, no,
# what) only count inside phrases.
_REACTION_WORDS = frozenset({
    "oh", "wow", "whoa", "insane", "crazy", "unbelievable", "incredible", "amazing", "huge", "massive",
    "brilliant", "superb", "wicket", "viketa", "six", "siksa", "sixer", "four", "phora", "boundary", "out",
    "aauta", "catch", "kaicha", "bowled", "bolda", "arre", "arey", "oho", "abe", "baap", "baapa", "pagal",
    "paagala", "gajab", "gajba", "khatarnak", "khataranaaka", "chhakka", "chhakkaa", "chauka", "chaukaa",
    "shandar", "shaanadaara", "zabardast", "jbaradasta", "bawaal", "bavaala", "khatam", "khatma", "kamaal",
    "kamaala", "wah", "waah", "vaaha", "teri", "teree", "nikal", "nikala", "aagaya",
})
_REACTION_PHRASES = (
    "kya baat", "kyaa baata", "oh ho", "are yaar", "are yaara", "arre yaar", "kya shot", "kyaa shota",
    "arre arre", "are are", "are bhai", "are bhaaee", "arre bhai", "yeh kya", "ye kya", "ye kyaa",
    "yaha kyaa", "oh my god", "what a", "kya cheez", "kyaa cheeja", "baap re", "baapa re", "nahi yaar",
    "naheen yaara", "gajab ka", "gajba kaa", "chhakka maar", "gaya gaya", "gayaa gayaa", "out ho gaya",
    "aauta ho gayaa", "ud gaya", "uda diya", "udaa diyaa", "maar diya", "maara diyaa",
)
_LEAD_IN = re.compile(
    r"^\W*(so|toh|to|haan|ha|acha|achha|accha|achchhaa|okay|ok|oke|guys|dekho|matlab|matalaba|basically|"
    r"besikalee|um+|uh+|chalo|aur|aura|and|actually)\b", re.IGNORECASE)
# No "?": to_roman writes "?" for characters it cannot map (measured 3-6% of openings).
_STRONG_OPEN = re.compile(
    r"!|\d|\b(what a|kya baat|kyaa baata|arre|arey|oho|wow|out|aauta|six|siksa|wicket|viketa|"
    r"chhakka|chhakkaa|chauka|chaukaa)\b", re.IGNORECASE)


def _energy_z(rms_list: list[tuple[float, float]]) -> dict[int, float]:
    """Per-second loudness as a robust z-score against this stream's own baseline."""
    if not rms_list:
        return {}
    sums: dict[int, list[float]] = {}
    for t, v in rms_list:
        sums.setdefault(int(t), []).append(float(v))
    seconds = {k: sum(v) / len(v) for k, v in sums.items()}
    values = np.array(list(seconds.values()), dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) * 1.4826 or float(np.std(values)) or 1.0
    return {k: (v - median) / mad for k, v in seconds.items()}


def _z_in(zmap: dict[int, float], start: float, end: float) -> list[float]:
    return [zmap.get(t, 0.0) for t in range(int(start), int(math.ceil(end)) + 1)]


def _unit(value: float, full: float) -> float:
    return max(0.0, min(1.0, value / full)) if full > 0 else 0.0


def _reaction_hits(text: str) -> int:
    low = text.casefold()
    words = re.findall(r"\b\w+\b", low)
    return sum(1 for w in words if w in _REACTION_WORDS) + 2 * sum(1 for p in _REACTION_PHRASES if p in low)


def _cricket_signals(text: str) -> tuple[int, float]:
    """(number of cricket terms/known names, share of tokens they cover)."""
    found, covered, tokens = _cricket_counts(text)
    return found, covered / max(tokens, 1)


def _cricket_counts(text: str) -> tuple[int, int, int]:
    """(cricket terms/names found, tokens they cover, tokens). Additive across
    adjacent thoughts, so a window sums its members instead of re-scanning."""
    from automation.seo.cricket_context import correct_cricket_spelling, find_canonical_entities
    from utils.devanagari import to_roman

    canonical = correct_cricket_spelling(to_roman(str(text or ""))).casefold()
    words = re.findall(r"\b\w+\b", canonical)
    names = {n.casefold() for names in find_canonical_entities(canonical).values() for n in names}
    names.add("test match")
    alternation = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    pattern = (r"\b(?:" + alternation + r"|cricket|ipl|match|test|t20|odi|wicket|runs?|score|over|ball|"
               r"batting|bowling|innings?|partnership|four|six|boundary|century|fifty|batsman|batter|"
               r"bowler|catch|drs|lbw|chhakka|chauka|pakistan|england|india|australia|sri\s+lanka|"
               r"south\s+africa|new\s+zealand)\b")
    found = re.findall(pattern, canonical)
    return len(found), sum(len(s.split()) for s in found), len(words)


def _dense(found: int, ratio: float) -> bool:
    return found >= 2 and ratio >= 0.07


def _has_dense_cricket_evidence(text: str) -> bool:
    return _dense(*_cricket_signals(text))


def _duration_fit(duration: float, low: float, high: float) -> float:
    if low <= duration <= high:
        return 1.0
    gap = (low - duration) if duration < low else (duration - high)
    return max(0.0, 1.0 - gap / 10.0)


def score_window(window: dict, zmap: dict[int, float], source_entities: set[str],
                 preferred: tuple[float, float]) -> dict:
    """Return the ranking breakdown for one window; ``final_score`` is 0..100."""
    start, end = float(window["start"]), float(window["end"])
    text = str(window.get("text", ""))
    words = text.split()
    opening = " ".join(words[:8])
    closing = " ".join(words[-10:])
    body_z = sorted(_z_in(zmap, start, end), reverse=True)
    signals = {
        "energy_peak": _unit(sum(body_z[:2]) / max(1, len(body_z[:2])), 4.0),
        "hook": max(_unit(max(_z_in(zmap, start, start + 3), default=0.0), 3.0),
                    0.7 if _STRONG_OPEN.search(opening) or _reaction_hits(opening) else 0.0),
        "payoff": max(_unit(max(_z_in(zmap, end - 3, end), default=0.0), 3.0),
                      0.6 if _reaction_hits(closing) else 0.0),
        "reaction": _unit(_reaction_hits(text), 6.0),
        "cricket_density": _unit(window["cricket_density"] if "cricket_density" in window
                                 else _cricket_signals(text)[1], 0.15),
        "source_match": 1.0 if source_entities and any(e in text.casefold() for e in source_entities) else 0.0,
        "duration_fit": _duration_fit(end - start, *preferred),
        "lead_in_penalty": 1.0 if _LEAD_IN.match(opening) else 0.0,
    }
    weights = {"energy_peak": 25, "hook": 15, "payoff": 10, "reaction": 20,
               "cricket_density": 15, "source_match": 5, "duration_fit": 10, "lead_in_penalty": -10}
    score = sum(weights[k] * v for k, v in signals.items())
    return {"score": round(score, 2), "final_score": round(score, 2),
            "dimension_scores": {k: round(v, 3) for k, v in signals.items()}}


# ── Window construction ──────────────────────────────────────────────────────

def _build_livestream_windows(thoughts: list[dict], min_duration: float, max_duration: float,
                              max_gap: float = 1.5) -> list[dict]:
    """Forward windows of adjacent complete thoughts: cricket, banter or a live moment.

    Livestream speech rarely puts setup and payoff in one caption sentence, so
    every clean thought seeds windows that grow until silence, stream filler
    or the duration ceiling. Cricket talk is labelled for ranking, not required.
    """
    minimum = max(0.0, float(min_duration))
    maximum = max(minimum, float(max_duration))
    ordered = sorted(thoughts, key=lambda item: float(item.get("start", 0)))
    counts = [_cricket_counts(t.get("text", "")) for t in ordered]
    windows, seen = [], set()
    for index, first in enumerate(ordered):
        if _is_livestream_filler(first.get("text", "")):
            continue
        start = float(first.get("start", 0))
        members, previous_end, found, covered, tokens = [], start, 0, 0, 0
        for offset, thought in enumerate(ordered[index:]):
            t_start, t_end = float(thought.get("start", 0)), float(thought.get("end", 0))
            if t_end <= t_start:
                continue
            if members and t_start - previous_end > max_gap:
                break
            if _is_livestream_filler(thought.get("text", "")) or t_end - start > maximum:
                break
            members.append(thought)
            previous_end = t_end
            f, c, n = counts[index + offset]
            found, covered, tokens = found + f, covered + c, tokens + n
            if t_end - start < minimum or (start, t_end) in seen:
                continue
            text = " ".join(str(m.get("text", "")).strip() for m in members).strip()
            if not text:
                continue
            seen.add((start, t_end))
            density = covered / max(tokens, 1)
            windows.append({
                "start": start, "end": t_end, "text": text,
                "cricket": _dense(found, density), "cricket_density": density,
                "source_cue_ids": sorted({c for m in members for c in m.get("source_cue_ids", [])}),
                "words": [w for m in members for w in m.get("words", [])],
                "boundary_method": ("source_words" if all(m.get("boundary_method") == "source_words"
                                                          for m in members) else "estimated_cue_fraction"),
                "complete_thought": None,
            })
    return windows


# ── Main entry point ─────────────────────────────────────────────────────────

def _durations() -> tuple[float, float, tuple[float, float]]:
    """Single source of truth for clip length (C11): clip_selection.output_*."""
    selection = cfg.get("clip_selection", {})
    low = float(selection.get("output_min_duration", 10))
    high = float(selection.get("output_max_duration", 38))
    h_cfg = cfg.get("highlight", {})
    preferred = (float(h_cfg.get("preferred_duration_min", low)), float(h_cfg.get("preferred_duration_max", high)))
    return low, high, preferred


def dedupe_history_path(match_key: str | None, output_path: str) -> str:
    key = match_key or _match_key(cfg["paths"]["input"], output_path)
    return str(Path(cfg["paths"]["highlights"]) / f"{key}.dedupe_history.yaml")


def record_exported_windows(history_path: str, windows: list[dict]) -> None:
    """Remember exported moments so a re-run of the same match picks new ones."""
    from automation.clip_selection.dedupe import append_windows
    clean = [{"start": float(w["start"]), "end": float(w["end"])} for w in windows
             if w.get("start") is not None and w.get("end") is not None]
    if clean:
        append_windows(history_path, clean)


def _load_segments(transcript_path: Path) -> list[dict]:
    import copy
    with open(transcript_path, encoding="utf-8") as f:
        data = json.load(f)
    segments = copy.deepcopy(data if isinstance(data, list) else data.get("segments", []))
    for cue_id, segment in enumerate(segments):
        segment["source_cue_ids"] = [cue_id]
    return segments


def detect_highlights(transcript_path: str | None = None, video_path: str | None = None,
                      output_path: str | None = None, match_key: str | None = None,
                      diagnostics: dict | None = None) -> list[dict]:
    """Discover, rank and curate clips; write the highlights YAML export reads."""
    import copy

    from utils.devanagari import to_roman
    from automation.seo.cricket_context import find_canonical_entities, is_cricket_content
    from automation.clip_selection.content_type import classify_content_type
    from automation.clip_selection.dedupe import load_previous_windows, overlaps_any_previous

    paths = cfg["paths"]
    if video_path is None:
        video_path = str(Path(paths["input"]) / cfg["download"]["output_filename"])
    stem = Path(video_path).stem
    transcript_path = transcript_path or str(Path(paths["transcripts"]) / f"{stem}.json")
    output_path = output_path or str(Path(paths["highlights"]) / f"{stem}.yaml")
    t_path = Path(transcript_path)
    if not t_path.exists():
        raise FileNotFoundError(f"Transcript not found: {t_path}")

    raw_segments = _load_segments(t_path)
    segments = copy.deepcopy(raw_segments)
    # Whisper emits Devanagari for language=hi; ranking and gates read Roman Hinglish.
    for seg in segments:
        seg["text"] = to_roman(seg.get("text", ""))
        for word in seg.get("words", []) or []:
            word["word"] = to_roman(word.get("word", ""))
    segments = _prepare_complete_thoughts(_split_segments_on_pauses(segments))
    source_context = _load_source_context(paths["input"])
    source_title = _load_source_title(paths["input"])
    diagnostics = diagnostics if diagnostics is not None else {}
    diagnostics["dedupe_history"] = history_path = dedupe_history_path(match_key, output_path)

    # The ONE cricket gate (once per source). Windows inside a cricket stream may
    # be banter or a live moment; their own cricket talk is only a ranking input.
    if not is_cricket_content(" ".join([source_context, *(s.get("text", "") for s in segments)])):
        _write_empty_highlights(output_path)
        diagnostics.update(status="NON_CRICKET_SOURCE", candidates=0, hard_rejected=0)
        log.warning("Source is not cricket; wrote empty highlights")
        return []

    selection = cfg.get("clip_selection", {})
    low, high, preferred = _durations()
    zmap = _energy_z(_extract_audio_rms(video_path))
    windows = _build_livestream_windows(segments, low, high)
    previous = load_previous_windows(history_path) if selection.get("dedup_enabled", True) else []
    threshold = float(selection.get("dedup_overlap_threshold", 0.6))
    fresh = [w for w in windows if not overlaps_any_previous(w["start"], w["end"], previous, threshold)]
    entities = {n.casefold() for names in find_canonical_entities(f"{source_title} {source_context}").values()
                for n in names}
    for window in fresh:
        window.update(score_window(window, zmap, entities, preferred))
        window["content_type"] = classify_content_type(window["text"])
    candidates = _select_diverse_windows(fresh, int(selection.get("discovery_candidates", 60)))
    for candidate in candidates:
        # Sentence grouping is not semantic proof of a complete thought.
        set_completeness(candidate, "UNVERIFIED")
    _apply_policy(candidates)
    candidates.sort(key=lambda c: c.get("final_score", 0), reverse=True)
    log.info("Discovery: %d windows, %d new after dedupe (%d previous), %d ranked candidates",
             len(windows), len(fresh), len(previous), len(candidates))

    from automation.editorial_planner import curate_clips
    context = {"source_title": source_title, "previous_windows": previous,
               "dedup_overlap_threshold": threshold}
    top = curate_clips(raw_segments, candidates, video_path, cfg, context,
                       Path(output_path).with_suffix(".curation.json"))
    top = [c for c in top if completeness_value(c) is True]
    _apply_policy(top)
    summary = context.get("selection_summary", {})
    diagnostics.update(summary)

    if cfg.get("release_review", {}).get("required", False):
        from automation.editorial_evidence import source_identity, write_discovery_evidence
        write_discovery_evidence(output_path, source_identity(video_path), transcript_path, raw_segments,
                                 candidates, top, context.get("editorial_decision"), selection_summary=summary)
    return _write_highlights(output_path, sorted(top, key=lambda w: w["start"]),
                             context.get("editorial_reserves", []), video_path, diagnostics)


def _apply_policy(candidates: list[dict]) -> None:
    try:
        from shorts_intelligence.bridge import apply_selection_policy
        apply_selection_policy(cfg, candidates)
    except Exception as exc:
        log.warning("Shorts Intelligence policy unavailable: %s", exc)


def _write_highlights(output_path: str, top: list[dict], reserves: list[dict], video_path: str,
                      diagnostics: dict) -> list[dict]:
    from automation.speech_alignment import _fingerprint

    h_cfg = cfg.get("highlight", {})
    low, high, _ = _durations()
    identity = _fingerprint(video_path) if top or reserves else None
    yaml_data, highlights = {}, []
    for i, w in enumerate(top + reserves, start=1):
        reserve = i > len(top)
        key = f"reserve{i - len(top)}" if reserve else f"clip{i}"
        entry = _build_clip_yaml_entry(start=w["start"], end=w["end"], score=w.get("final_score", w.get("score", 0)),
                                       text=w["text"], target_duration=float(h_cfg.get("target_duration", high)),
                                       max_speedup=float(h_cfg.get("max_speedup", 1.0)),
                                       output_min=low, output_max=high)
        entry.update(selection_reason=w.get("ai_reason", ""), semantic_review=w.get("semantic_review", "NOT CHECKED"),
                     boundary_method=w.get("boundary_method", "unknown"), source_cue_ids=w.get("source_cue_ids", []),
                     complete_thought=completeness_value(w), completeness=w.get("completeness"),
                     speech_segments=w["speech_segments"], role="reserve" if reserve else "primary",
                     source_identity=identity, final_score=w.get("final_score"),
                     dimension_scores=w.get("dimension_scores", {}))
        yaml_data[key] = entry
        highlight = {
            "id": key, "start": w["start"], "end": w["end"], "start_ts": fmt_ts(w["start"]),
            "end_ts": fmt_ts(w["end"]), "score": w.get("final_score", w.get("score", 0)),
            "speed_factor": entry["speed_factor"], "text": w["text"], "content_type": w.get("content_type"),
            "complete_thought": completeness_value(w), "completeness": w.get("completeness"),
            "semantic_review": w.get("semantic_review", "NOT CHECKED"),
            "dimension_scores": w.get("dimension_scores", {}),
            "intelligence_adjustment": w.get("intelligence_adjustment", 0.0),
            "intelligence_shadow_adjustment": w.get("intelligence_shadow_adjustment", 0.0),
            "intelligence_segments": w.get("intelligence_segments", []),
        }
        if reserve:
            diagnostics.setdefault("reserves", []).append(highlight)
        else:
            highlights.append(highlight)
        log.info("  %s: %s -> %s (score=%.1f)", key, fmt_ts(w["start"]), fmt_ts(w["end"]),
                 w.get("final_score") or 0)
    # Always commit atomically; an empty mapping clears stale highlights.
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(output_path).with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
    tmp.replace(output_path)
    log.info("Highlights saved -> %s (%d clips, %d reserves)", output_path, len(highlights), len(reserves))
    return highlights
