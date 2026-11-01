"""Local word-timestamp ASR for shortlisted source ranges, not guessed cue timing."""

import subprocess
import math
import tempfile
import threading
from pathlib import Path

_LOCK = threading.Lock()
_MODELS = {}


def words_in_range(segments: list[dict], start: float, end: float) -> list[dict]:
    """Validate source word intervals, then retain the selected range only.

    A small rounding tolerance is not permission to reconstruct clipped speech.
    Callers must supply contextual ASR or reviewed source word evidence.
    """
    if not all(math.isfinite(t) for t in (start, end)) or start < 0 or end <= start:
        raise ValueError("Invalid speech source range")
    words = []
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("Invalid speech segment")
        raw_words = segment.get("words")
        if not raw_words:
            try:
                segment_start, segment_end = float(segment["start"]), float(segment["end"])
            except (KeyError, TypeError, ValueError):
                raise ValueError("Missing word timestamps") from None
            if not all(math.isfinite(t) for t in (segment_start, segment_end)) or segment_end <= segment_start:
                raise ValueError("Invalid segment timestamp")
            if segment_end > start and segment_start < end:
                raise ValueError("Missing word timestamps inside selected range")
            continue
        if not isinstance(raw_words, list):
            raise ValueError("Invalid word timestamps")
        for word in raw_words:
            try:
                word_start, word_end = float(word["start"]), float(word["end"])
            except (KeyError, TypeError, ValueError):
                raise ValueError("Invalid word timestamp") from None
            if not all(math.isfinite(t) for t in (word_start, word_end)) or word_start < 0 or word_end <= word_start:
                raise ValueError("Invalid word timestamp")
            if word_end <= start or word_start >= end:
                continue
            if word_start < start - 0.02 or word_end > end + 0.02:
                raise ValueError("Cut straddles a spoken word; adjust source boundary")
            words.append({**word, "start": max(start, word_start), "end": min(end, word_end)})
    words.sort(key=lambda word: (word["start"], word["end"]))
    for previous, current in zip(words, words[1:]):
        if current["start"] < previous["end"] - 0.02:
            raise ValueError("Overlapping word timestamps require alignment review")
    return words


def alignment_issues(segments: list[dict], start: float, end: float) -> list[str]:
    """Conservative pilot checks; a flagged gap is uncertainty, not proven silence."""
    try:
        words = words_in_range(segments, start, end)
    except (TypeError, ValueError) as exc:
        return [str(exc)]
    if not words:
        return ["No word-timed speech"]
    issues = []
    if words[0]["start"] - start > 1.5:
        issues.append("Unverified opening speech gap exceeds 1.5s")
    if end - words[-1]["end"] > 1.5:
        issues.append("Unverified ending speech gap exceeds 1.5s")
    if any(current["start"] - previous["end"] > 2 for previous, current in zip(words, words[1:])):
        issues.append("Unverified internal speech gap exceeds 2s")
    try:
        probabilities = [float(word["probability"]) for word in words]
    except (KeyError, TypeError, ValueError):
        return issues + ["Missing or invalid ASR confidence"]
    if not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities):
        issues.append("Missing or invalid ASR confidence")
    elif sum(probabilities) / len(probabilities) < 0.75 or sum(p < 0.5 for p in probabilities) / len(probabilities) > 0.15:
        issues.append("Low-confidence ASR requires correction/alignment review")
    return issues


def transcribe_range(video_path: str, start: float, end: float, *, model_name="medium",
                     language="hi", temp_dir="temp", context_seconds=2.0) -> list[dict]:
    if not all(math.isfinite(t) for t in (start, end, context_seconds)) or start < 0 or end <= start or context_seconds <= 0:
        raise ValueError("Invalid ASR source range")
    context_start, context_end = max(0.0, start - context_seconds), end + context_seconds
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="clip-asr-", dir=temp_dir) as folder:
        audio = Path(folder) / "speech.wav"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{context_start:.6f}", "-i", video_path,
                        "-t", f"{context_end-context_start:.6f}",
                        "-vn", "-ac", "1", "-ar", "16000", str(audio)],
                       capture_output=True, check=True, timeout=180)
        with _LOCK:
            from faster_whisper import WhisperModel
            if model_name not in _MODELS:
                _MODELS[model_name] = WhisperModel(model_name, device="cpu", compute_type="int8")
            segments, _ = _MODELS[model_name].transcribe(
                str(audio), language=language or None, word_timestamps=True,
                # Retain surrounding speech to expose boundary-crossing words.
                # VAD can drop quiet Hindi speech; do not discard it here.
                vad_filter=False, beam_size=5,
            )
            result = []
            for segment in segments:
                words = [{"start": context_start + word.start, "end": context_start + word.end,
                          "word": word.word, "probability": word.probability}
                         for word in (segment.words or [])]
                result.append({"start": context_start + segment.start, "end": context_start + segment.end,
                                "text": segment.text, "words": words,
                                "requested_range": [start, end],
                                "requested_context_range": [context_start, context_end],
                               "timing_method": "local_asr_word_estimates"})
    if not any(segment["words"] for segment in result):
        raise ValueError("Local ASR produced no word timestamps; hold for review")
    return result
