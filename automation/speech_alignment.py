"""Local word-timestamp ASR for shortlisted source ranges, not guessed cue timing."""

import subprocess
import math
import tempfile
import threading
from pathlib import Path

_LOCK = threading.Lock()
_MODELS = {}


def alignment_issues(segments: list[dict], start: float, end: float) -> list[str]:
    """Conservative pilot checks; a flagged gap is uncertainty, not proven silence."""
    words = sorted([word for segment in segments for word in segment.get("words", [])], key=lambda w: w["start"])
    if not words:
        return ["No word-timed speech"]
    issues = []
    if words[0]["start"] - start > 1.5:
        issues.append("Unverified opening speech gap exceeds 1.5s")
    if end - words[-1]["end"] > 1.5:
        issues.append("Unverified ending speech gap exceeds 1.5s")
    if any(current["start"] - previous["end"] > 2 for previous, current in zip(words, words[1:])):
        issues.append("Unverified internal speech gap exceeds 2s")
    probabilities = [float(word.get("probability", 0)) for word in words]
    if not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities):
        issues.append("Missing or invalid ASR confidence")
    elif sum(probabilities) / len(probabilities) < 0.75 or sum(p < 0.5 for p in probabilities) / len(probabilities) > 0.15:
        issues.append("Low-confidence ASR requires correction/alignment review")
    return issues


def transcribe_range(video_path: str, start: float, end: float, *, model_name="small",
                     language="hi", temp_dir="temp") -> list[dict]:
    if start < 0 or end <= start:
        raise ValueError("Invalid ASR source range")
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="clip-asr-", dir=temp_dir) as folder:
        audio = Path(folder) / "speech.wav"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", video_path,
                        "-ss", f"{start:.6f}", "-t", f"{end-start:.6f}",
                        "-vn", "-ac", "1", "-ar", "16000", str(audio)],
                       capture_output=True, check=True, timeout=180)
        with _LOCK:
            from faster_whisper import WhisperModel
            if model_name not in _MODELS:
                _MODELS[model_name] = WhisperModel(model_name, device="cpu", compute_type="int8")
            segments, _ = _MODELS[model_name].transcribe(
                str(audio), language=language or None, word_timestamps=True,
                # Shortlisted speech ranges are already bounded. VAD can drop
                # quiet Hindi speech; do not discard it before alignment review.
                vad_filter=False, beam_size=5,
            )
            result = []
            for segment in segments:
                words = [{"start": start + word.start, "end": start + word.end,
                          "word": word.word, "probability": word.probability}
                         for word in (segment.words or [])]
                result.append({"start": start + segment.start, "end": start + segment.end,
                               "text": segment.text, "words": words,
                               "timing_method": "local_asr_word_estimates"})
    if not any(segment["words"] for segment in result):
        raise ValueError("Local ASR produced no word timestamps; hold for review")
    return result
