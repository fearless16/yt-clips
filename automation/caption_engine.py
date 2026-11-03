"""Word-timed Hinglish caption cards with conservative source-boundary checks.

Required captions use source word intervals, not proportional sentence timing.
ASR timestamps and transliteration still require listening/fidelity review.
The optional legacy cue-only route is an approximation, not verified alignment.
"""

from __future__ import annotations

import html
import json
import logging
import math
import re
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("caption_engine")

# Concurrency lock for Playwright browser instance across worker threads
_PLAYWRIGHT_LOCK = threading.Lock()

# Global in-memory cache for transliterated lines across clips
_HINGLISH_CACHE: Dict[str, str] = {}

# High-impact keywords that receive glowing accents
GOLD_KEYWORDS = {
    "prajjwal", "cricket", "taza", "tarin", "episode", "match", "india",
    "dhoni", "kohli", "rohit", "bumrah", "hardik", "nar-sanhaar", "narsanhaar",
    "palat-waar", "palatwaar", "shandaar", "bawaal", "six", "wicket", "century",
    "bhai", "yaar"
}

CYAN_KEYWORDS = {
    "cricket wali baat", "new zealand", "hazir hoon", "live", "dosto",
    "kya scene", "ye kya he", "telecast", "repeat", "score", "namashkar"
}


@dataclass
class CaptionCue:
    start: float
    end: float
    text: str
    html: str
    words: List[str]


@dataclass
class CaptionCard:
    path: str
    start: float
    end: float
    text: str


def highlight_keywords_html(text: str) -> str:
    """Format text into HTML with signature Quantum Holo-Glass glowing keywords."""
    if not text:
        return ""

    # Sanitize HTML special characters to prevent markup injection
    safe = html.escape(text.strip(), quote=False)

    # Check multi-word cyan keywords first
    formatted = safe
    for phrase in sorted(CYAN_KEYWORDS, key=len, reverse=True):
        pattern = re.compile(rf"\b({re.escape(phrase)})\b", re.IGNORECASE)
        formatted = pattern.sub(r'<span class="holo-pill">\1</span>', formatted)

    words = formatted.split()
    out_words = []
    for w in words:
        if "<span" in w or "</span>" in w:
            out_words.append(w)
            continue
        clean_w = re.sub(r"[^\w]", "", w).lower()
        if clean_w in GOLD_KEYWORDS:
            out_words.append(f'<span class="holo-gold">{w}</span>')
        else:
            out_words.append(w)

    return " ".join(out_words)


def chunk_sentence_into_cues(
    text: str,
    start_time: float,
    end_time: float,
    max_words: int = 4,
    min_duration: float = 0.5,
) -> List[CaptionCue]:
    """Split a sentence into short 2-4 word micro-chunks synchronized across the duration.

    Prevents text overload by ensuring no more than 3-4 words appear at any given moment.
    Timing is strictly partitioned proportionally across chunks to prevent drift or overlap.
    """
    words = text.strip().split()
    if not words:
        return []

    total_duration = max(min_duration, end_time - start_time)
    num_words = len(words)

    # Group into micro-chunks of at most max_words
    if num_words <= max_words:
        chunks = [words]
    else:
        num_chunks = math.ceil(num_words / max_words)
        chunk_size = math.ceil(num_words / num_chunks)
        chunks = [words[i:i + chunk_size] for i in range(0, num_words, chunk_size)]

    # Calculate proportional weights by character length
    weights = [sum(len(w) for w in ch) for ch in chunks]
    total_weight = sum(weights) or 1.0

    cues = []
    accumulated_ratio = 0.0

    for i, (ch, w) in enumerate(zip(chunks, weights)):
        ratio = w / total_weight
        chunk_start = round(start_time + accumulated_ratio * total_duration, 2)
        accumulated_ratio += ratio
        chunk_end = round(start_time + accumulated_ratio * total_duration, 2)

        # Ensure last chunk hits end_time exactly
        if i == len(chunks) - 1:
            chunk_end = round(end_time, 2)

        # Enforce non-negative minimal duration
        if chunk_end <= chunk_start:
            chunk_end = round(chunk_start + min_duration, 2)

        chunk_text = " ".join(ch)
        html_text = highlight_keywords_html(chunk_text)

        cues.append(
            CaptionCue(
                start=chunk_start,
                end=chunk_end,
                text=chunk_text,
                html=html_text,
                words=ch,
            )
        )

    return cues


def convert_hindi_to_hinglish_llm(
    texts: List[str],
    api_url: str = "http://127.0.0.1:8081/v1/chat/completions",
    model: str = "qwen3.8-27b",
    timeout: int = 90,
) -> Optional[List[str]]:
    """Convert a batch of Hindi sentences into natural conversational Hinglish using local LLM."""
    if not texts:
        return []

    # Check cache first
    needed_indices = []
    results = [None] * len(texts)
    for idx, t in enumerate(texts):
        cleaned = t.strip()
        if cleaned in _HINGLISH_CACHE:
            results[idx] = _HINGLISH_CACHE[cleaned]
        else:
            needed_indices.append(idx)

    if not needed_indices:
        return results

    numbered = "\n".join(f"{i+1}. {texts[idx].strip()}" for i, idx in enumerate(needed_indices))
    prompt = (
        "You are an expert Hindi to Hinglish subtitle converter for YouTube Shorts.\n"
        "Convert each Hindi line into natural, casual, modern conversational Hinglish "
        "(English alphabet, Hindi language, WhatsApp/Shorts style like 'Ye kya he', 'Aur bhai kya scene', 'Namashkar dosto').\n"
        "RULES:\n"
        "- Do NOT translate to English! Keep the spoken Hindi words in English alphabet.\n"
        "- Use standard casual Indian Hinglish spellings (e.g. bhai, yaar, hazir, taza, match, New Zealand).\n"
        "- Return EXACTLY the same number of numbered lines (1 to N). Output nothing else.\n\n"
        f"Input:\n{numbered}\n\n"
        "Hinglish Output:"
    )

    req_body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 1024,
        "chat_template_kwargs": {"enable_thinking": False}
    }

    try:
        req = urllib.request.Request(
            api_url,
            data=json.dumps(req_body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"].get("content", "").strip()

        lines = [line.strip() for line in content.split("\n") if line.strip()]
        cleaned_lines = []
        for line in lines:
            line_clean = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if line_clean:
                cleaned_lines.append(line_clean)

        if len(cleaned_lines) == len(needed_indices):
            for orig_idx, converted in zip(needed_indices, cleaned_lines):
                results[orig_idx] = converted
                _HINGLISH_CACHE[texts[orig_idx].strip()] = converted
            return results
        log.warning("LLM returned %d lines for %d requested inputs", len(cleaned_lines), len(needed_indices))
    except Exception as e:
        log.warning("LLM Hinglish conversion failed: %s", e)

    # Fallback to rule-based transliteration or original text for any unfilled lines
    for idx in needed_indices:
        if results[idx] is None:
            try:
                from automation.hinglish_converter import rule_based_transliterate
                results[idx] = rule_based_transliterate(texts[idx])
            except Exception:
                results[idx] = texts[idx]

    return results


def normalize_rolling_transcript_segments(segments: List[Dict]) -> List[Dict]:
    """Approximate display-overlap removal for the legacy cue-only route.

    YouTube ASR subtitles use 2-line rolling display windows where end times reflect
    on-screen persistence (extending into subsequent utterances). This clamps each
    segment's display end near the next segment. This is not speech alignment.
    """
    if not segments:
        return []

    sorted_segs = sorted(segments, key=lambda s: float(s.get("start", 0.0)))
    normalized = []
    for i, s in enumerate(sorted_segs):
        st = float(s.get("start", 0.0))
        raw_et = float(s.get("end", st + 1.5))
        txt = str(s.get("text", "")).strip()
        if not txt:
            continue

        if i + 1 < len(sorted_segs):
            next_st = float(sorted_segs[i + 1].get("start", 0.0))
            if next_st > st:
                et = min(raw_et, next_st + 0.1)
            else:
                et = raw_et
        else:
            et = raw_et

        normalized.append({"start": st, "end": et, "text": txt})
    return normalized


def process_transcript_segments(
    segments: List[Dict],
    batch_size: int = 8,
    max_words_per_cue: int = 4,
    clip_duration: Optional[float] = None,
) -> List[CaptionCue]:
    """Process transcript segments: converts to natural Hinglish and micro-chunks into synchronized cues."""
    all_cues = []
    if not segments:
        return []

    total = len(segments)
    for i in range(0, total, batch_size):
        batch = segments[i:i + batch_size]
        texts = [b["text"] for b in batch]

        # Convert Hindi to natural conversational Hinglish
        hinglish_list = convert_hindi_to_hinglish_llm(texts)
        if not hinglish_list or len(hinglish_list) != len(batch):
            hinglish_list = texts

        for b, h_text in zip(batch, hinglish_list):
            st = float(b["start"])
            et = float(b["end"])
            if clip_duration is not None:
                if st >= clip_duration:
                    continue
                et = min(clip_duration, et)

            cues = chunk_sentence_into_cues(
                h_text,
                start_time=st,
                end_time=et,
                max_words=max_words_per_cue,
            )
            all_cues.extend(cues)

    return all_cues


def render_transparent_cards_batch(cues: List[CaptionCue], output_dir: Path) -> List[CaptionCard]:
    """Render approved neon cards; measured frame/text bounds accompany each PNG."""
    from playwright.sync_api import sync_playwright
    from automation.caption_design import caption_document, DESIGN_VERSION, MEASURE_CAPTION_JS

    output_dir.mkdir(parents=True, exist_ok=True)
    cards = []

    bounds = []
    with _PLAYWRIGHT_LOCK:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1080, "height": 320})
            for i, cue in enumerate(cues):
                html_body = caption_document(cue.text)
                html_path = output_dir / f"card_{i}.html"
                html_path.write_text(html_body, encoding="utf-8")

                png_path = output_dir / f"card_{i}.png"
                # Use domcontentloaded with 5s timeout to guarantee fast rendering even offline
                page.goto(html_path.as_uri(), wait_until="domcontentloaded", timeout=6000)
                page.evaluate("() => document.fonts.ready.then(() => true)")
                measurement = page.evaluate(MEASURE_CAPTION_JS)
                if not measurement:
                    raise ValueError("Caption needs a shorter aligned phrase; exceeds readable card bounds")
                bounds.append({"index": i, "text": cue.text, "start": cue.start, "end": cue.end, **measurement})
                page.screenshot(path=str(png_path), omit_background=True)
                cards.append(CaptionCard(
                    path=str(png_path),
                    start=cue.start,
                    end=cue.end,
                    text=cue.text
                ))
            browser.close()

    (output_dir / "caption_design.json").write_text(json.dumps(
        {"design": DESIGN_VERSION, "canvas": [1080, 320], "cards": bounds}, indent=2), encoding="utf-8")
    return cards


def prepare_clip_captions(
    transcript_segments: Optional[List[Dict]],
    clip_start: float,
    clip_end: float,
    output_dir: Path,
    max_words_per_cue: int = 4,
    require_word_timing: bool = False,
    hinglish_model: Optional[str] = None,
    allow_documented_successor: bool = False,
) -> List[CaptionCard]:
    """Prepare approved neon-style cards; model output remains unapproved speech evidence."""
    if not transcript_segments:
        if require_word_timing:
            raise ValueError("Word-timed captions required; no speech evidence supplied")
        return []

    if require_word_timing:
        if hinglish_model is not None:
            if hinglish_model != "deepseek-v4-flash":
                raise ValueError("Caption route requires exact deepseek-v4-flash")
            from automation.caption_hinglish import romanize_words_flash
            from automation.speech_alignment import words_in_range
            words = words_in_range(transcript_segments, clip_start, clip_end)
            context = "\n".join(str(s.get("text", "")) for s in transcript_segments
                                if float(s.get("end", 0)) > clip_start - 2
                                and float(s.get("start", 0)) < clip_end + 2)
            converted, evidence = romanize_words_flash(words, context=context,
                allow_documented_successor=allow_documented_successor)
            evidence.update({"clip_source_start": clip_start, "clip_source_end": clip_end})
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "caption_processing.json").write_text(
                json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
            transcript_segments = [{"start": clip_start, "end": clip_end, "words": converted}]
        cues = word_timed_cues(transcript_segments, clip_start, clip_end, max_words_per_cue)
        if not cues:
            raise ValueError("Word-timed captions required; cue-level estimates are not accepted")
        return render_transparent_cards_batch(cues, output_dir)

    # 1. Normalize rolling timestamps to true spoken utterance boundaries
    clean_segs = normalize_rolling_transcript_segments(transcript_segments)
    clip_duration = clip_end - clip_start

    # 2. Slice strictly within the clip boundary [clip_start, clip_end]
    rel_segs = []
    for s in clean_segs:
        st = s["start"]
        et = s["end"]
        txt = s["text"]

        if et <= clip_start or st >= clip_end:
            continue

        # If segment straddles clip_start, proportionally slice words to keep only those in clip
        if st < clip_start:
            dur = max(0.1, et - st)
            frac_before = (clip_start - st) / dur
            words = txt.split()
            skip_count = int(round(len(words) * frac_before))
            remaining_words = words[skip_count:]
            if not remaining_words:
                continue
            txt = " ".join(remaining_words)
            st = clip_start

        # If segment straddles clip_end, proportionally slice words to keep only those before clip_end
        if et > clip_end:
            dur = max(0.1, et - st)
            frac_after = (et - clip_end) / dur
            words = txt.split()
            keep_count = max(1, len(words) - int(round(len(words) * frac_after)))
            txt = " ".join(words[:keep_count])
            et = clip_end

        rel_st = round(st - clip_start, 2)
        rel_et = round(et - clip_start, 2)
        if rel_et > rel_st:
            rel_segs.append({"start": rel_st, "end": rel_et, "text": txt})

    if not rel_segs:
        return []

    cues = process_transcript_segments(
        rel_segs,
        max_words_per_cue=max_words_per_cue,
        clip_duration=clip_duration,
    )
    if not cues:
        return []

    return render_transparent_cards_batch(cues, output_dir)


def word_timed_cues(segments: List[Dict], clip_start: float, clip_end: float,
                    max_words: int = 4) -> List[CaptionCue]:
    """Use actual ASR/alignment word intervals; never slice text by character share.

    Timings remain ASR estimates and require listening review. Boundary-straddling
    words are rejected rather than dropping half a syllable into the delivery.
    """
    from automation.hinglish_converter import rule_based_transliterate
    from automation.speech_alignment import words_in_range

    if not isinstance(max_words, int) or not 1 <= max_words <= 8:
        raise ValueError("Invalid caption bounds/chunk size")
    words = []
    for word in words_in_range(segments, clip_start, clip_end):
        text = str(word.get("word", word.get("text", ""))).strip()
        if not text:
            raise ValueError("Missing word text inside selected range")
        words.append((word["start"], word["end"], word.get("roman") or rule_based_transliterate(text)))
    groups = []
    group = []
    for word in words:
        # Joined Hindi source tokens may romanize to multiple display words.
        # Keep their whole original interval, but count readable display words
        # when choosing a phrase break. Never invent per-syllable timestamps.
        display_words = sum(len(item[2].split()) for item in group)
        display_chars = len(" ".join(item[2] for item in [*group, word]))
        if group and (display_words + len(word[2].split()) > max_words
                      or display_chars > 42 or word[0] - group[-1][1] > 0.45):
            groups.append(group)
            group = []
        group.append(word)
    if group:
        groups.append(group)
    # Rebalance an orphan final auxiliary using its real preceding word. Do not
    # extend display into silence or synthesize a new word boundary to hide it.
    if (len(groups) >= 2 and len(groups[-1]) == 1 and len(groups[-2]) >= 3
            and groups[-1][0][2].lower().strip(".,!?") in {"hai", "hain", "tha", "the", "hoon"}
            and groups[-1][0][0] - groups[-2][-1][1] <= .45
            and len(groups[-2][-1][2].split()) + len(groups[-1][0][2].split()) <= max_words):
        groups[-1].insert(0, groups[-2].pop())
    cues = []
    for group in groups:
        text = " ".join(item[2] for item in group)
        cues.append(CaptionCue(group[0][0] - clip_start, group[-1][1] - clip_start,
                               text, highlight_keywords_html(text), [item[2] for item in group]))
    # Clamp tiny timestamp rounding overlaps, never extend a phrase duration.
    for previous, current in zip(cues, cues[1:]):
        previous.end = min(previous.end, current.start)
    return cues
