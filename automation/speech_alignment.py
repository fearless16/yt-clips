"""Local word-timestamp ASR for shortlisted source ranges, not guessed cue timing."""

import subprocess
import math
import tempfile
import threading
import copy
import hashlib
import json
import re
import unicodedata
from pathlib import Path

_LOCK = threading.Lock()
_MODELS = {}
# Short, independent decoding windows reduced degenerate/zero-length words in
# the recorded fast-Hindi failure. These are decoding options, not timing repair.
ASR_DECODING = {"chunk_length": 8, "condition_on_previous_text": False}


def _fingerprint(video_path):
    path = Path(video_path).resolve()
    try:
        stat = path.stat()
        return [str(path), stat.st_size, stat.st_mtime_ns]
    except OSError:
        return [str(path), None, None]


def _anchor_key(text):
    """Script-neutral key: Devanagari is romanised so caption Hinglish and
    Whisper Devanagari compare on the same alphabet."""
    from utils.devanagari import to_roman
    text = to_roman(unicodedata.normalize("NFKC", str(text)))
    return "".join(c for c in text.casefold() if c.isalnum())


# Below this similarity a phrase is not the quoted speech; above it, a
# different transliteration of the same words (e.g. "bhai"/"bhaai").
ANCHOR_MIN_SIMILARITY = 0.8
ANCHOR_TIE_MARGIN = 0.03


def _anchor_span(words, phrase, *, closing=False):
    """Best unique fuzzy match of a quoted phrase against word-timed ASR."""
    from difflib import SequenceMatcher
    key = _anchor_key(phrase)
    if not key:
        raise ValueError("Missing source phrase anchor")
    keys = [_anchor_key(w.get("word", "")) for w in words]
    scored = []
    for i in range(len(words)):
        if not keys[i]:
            continue
        joined = ""
        for j in range(i, len(words)):
            joined += keys[j]
            if len(joined) >= len(key) * 0.6:
                ratio = 1.0 if joined == key else SequenceMatcher(None, joined, key, autojunk=False).ratio()
                scored.append((ratio, i, j))
            if len(joined) >= len(key) * 1.4:
                break
    if not scored:
        raise ValueError("Missing or ambiguous source phrase anchor: " + str(phrase))
    best = max(scored)
    rivals = [s for s in scored if s[0] >= best[0] - ANCHOR_TIE_MARGIN
              and (s[2] < best[1] or s[1] > best[2])]  # non-overlapping competitor
    if best[0] < ANCHOR_MIN_SIMILARITY or rivals:
        raise ValueError("Missing or ambiguous source phrase anchor: " + str(phrase))
    return best[2] if closing else best[1]


def align_candidate(video_path: str, candidate: dict, config: dict, cache_dir, *,
                    allow_boundary_review: bool = False) -> dict:
    """Resolve proposed cuts to real ASR words; semantic approval is invalidated.

    Raw attempts are immutable, source/config keyed artifacts. Supplied word
    evidence is never retranscribed. Fresh ASR gets at most one explicit retry.
    No deletion of invalid words, confidence inflation or proportional timing.
    """
    from automation.artifacts import atomic_json
    from shorts_intelligence.completeness import set_completeness

    start, end = float(candidate["start"]), float(candidate["end"])
    if not all(math.isfinite(t) for t in (start, end)) or start < 0 or end <= start:
        raise ValueError("Invalid alignment source range")
    transcription = config.get("transcription", {})
    selection = config.get("clip_selection", {})
    shift = float(selection.get("alignment_max_boundary_shift", 3))
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError("Invalid alignment boundary shift")
    models = list(dict.fromkeys([transcription.get("alignment_model", "medium"),
               transcription.get("alignment_fallback_model", "large-v3-turbo")]))
    supplied = candidate.get("speech_segments")
    attempts = []
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for attempt, model in enumerate(models if supplied is None else ["supplied"]):
        # More surrounding speech worsened the recorded fast-Hindi failure.
        # Change models once, not both model and context length at once.
        context = max(float(transcription.get("alignment_context_seconds", 2)), shift)
        identity = {"version": 2, "source": _fingerprint(video_path), "range": [start, end],
                    "model": model, "context": context, "language": transcription.get("language", "hi"),
                    "decoding": dict(ASR_DECODING)}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = cache_dir / f"speech-{key}.json"
        segments, nearby = None, []
        try:
            if supplied is not None:
                segments = copy.deepcopy(supplied)
            elif path.exists():
                segments = json.loads(path.read_text(encoding="utf-8"))["segments"]
            else:
                segments = transcribe_range(video_path, start, end, model_name=model,
                    language=transcription.get("language", "hi"),
                    temp_dir=config.get("paths", {}).get("temp", "temp"), context_seconds=context)
                atomic_json(path, {"identity": identity, "segments": segments,
                                   "verification": "ASR estimates; not listening approval"})
            # Validate every raw interval before using any anchor. Do not
            # silently discard zero-length/overlapping hallucinated words.
            all_words = []
            for segment in segments:
                raw_words = segment.get("words")
                if not raw_words:
                    if segment.get("end", end) > start and segment.get("start", start) < end:
                        raise ValueError("Missing word timestamps inside proposed range")
                    continue
                for word in raw_words:
                    a, b = float(word["start"]), float(word["end"])
                    if not all(math.isfinite(t) for t in (a, b)) or a < 0:
                        raise ValueError("Invalid word timestamp")
                    all_words.append({**word, "start": a, "end": b})
            all_words.sort(key=lambda w: (w["start"], w["end"]))
            if not all_words:
                raise ValueError("No word-timed speech")
            nearby = [w for w in all_words if w["end"] > max(0, start-shift)
                      and w["start"] < end+shift]
            needs_review = False
            if candidate.get("opening_phrase") or candidate.get("closing_phrase"):
                try:
                    first = _anchor_span(nearby, candidate.get("opening_phrase"))
                    last = _anchor_span(nearby, candidate.get("closing_phrase"), closing=True)
                    if last < first:
                        raise ValueError("Source phrase anchors are reversed")
                    resolved_start, resolved_end = nearby[first]["start"], nearby[last]["end"]
                except ValueError:
                    if not allow_boundary_review or (supplied is None and attempt < len(models)-1):
                        raise  # try the explicit fallback before editorial repair
                    needs_review = True
            if not (candidate.get("opening_phrase") or candidate.get("closing_phrase")) or needs_review:
                if any(w["end"] <= w["start"] and start <= w["start"] <= end for w in nearby):
                    raise ValueError("Invalid word timestamp inside proposed range")
                retained = [w for w in nearby if w["end"] > start and w["start"] < end]
                if not retained:
                    raise ValueError("No retained source words")
                resolved_start, resolved_end = retained[0]["start"], retained[-1]["end"]
            if resolved_start < start-shift or resolved_end > end+shift:
                raise ValueError("Source phrase anchor exceeds boundary repair limit")
            duration = resolved_end-resolved_start
            if not float(selection.get("output_min_duration", 1)) <= duration <= float(selection.get("output_max_duration", 60)):
                raise ValueError("Aligned duration outside configured bounds")
            issues = alignment_issues(segments, resolved_start, resolved_end)
            if issues:
                raise ValueError("; ".join(issues))
            retained = words_in_range(segments, resolved_start, resolved_end)
            attempts.append({"model": model, "path": str(path), "issues": []})
            result = copy.deepcopy(candidate)
            result.update(start=resolved_start, end=resolved_end, speech_segments=segments,
                          text=" ".join(str(w.get("word", "")).strip() for w in retained),
                          boundary_method="source_words", alignment_attempts=attempts)
            if needs_review:
                result.update(boundary_review_required=True, boundary_words=nearby,
                              proposed_source_range=[start, end])
            set_completeness(result, "UNVERIFIED")
            result["semantic_review"] = "NOT CHECKED"
            return result
        except (ValueError, TypeError, KeyError, RuntimeError, OSError) as exc:
            attempts.append({"model": model, "path": str(path), "issues": [str(exc)]})
            if (allow_boundary_review and supplied is None and nearby and segments is not None
                    and attempt == len(models)-1
                    and (candidate.get("opening_phrase") or candidate.get("closing_phrase"))):
                # Cue edges are proposals, not final cuts. A malformed word or
                # oversized context at an edge may be excluded by a fresh,
                # contiguous word-ID edit. Preserve ALL raw evidence here;
                # _repair_boundaries validates every word in the actual span
                # before it can reach final completeness review or export.
                result = copy.deepcopy(candidate)
                result.update(speech_segments=segments, boundary_review_required=True,
                    boundary_words=nearby, proposed_source_range=[start, end],
                    boundary_method="pending_source_word_review", alignment_attempts=attempts)
                set_completeness(result, "UNVERIFIED")
                result["semantic_review"] = "NOT CHECKED"
                return result
    raise ValueError("Speech alignment unresolved: " + "; ".join(a["issues"][0] for a in attempts))


def words_in_range(segments: list[dict], start: float, end: float) -> list[dict]:
    """Validate source word intervals, then retain the selected range only.

    A small rounding tolerance is not permission to reconstruct clipped speech.
    Callers must supply contextual ASR or reviewed source word evidence.
    """
    if not all(math.isfinite(t) for t in (start, end)) or start < 0 or end <= start:
        raise ValueError("Invalid speech source range")
    words = []
    wordless_overlap = False
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
            # Whisper emits wordless segments over crowd noise or music. They
            # hold no speech to cut through, so they are not a defect unless the
            # range has no word timing at all (cue-only input, not alignment).
            if segment_end > start and segment_start < end:
                if any(c.isalnum() for c in str(segment.get("text", ""))):
                    # Text without word timing is untimed speech: captions would drop it.
                    raise ValueError("Missing word timestamps inside selected range")
                wordless_overlap = True
            continue
        if not isinstance(raw_words, list):
            raise ValueError("Invalid word timestamps")
        for word in raw_words:
            try:
                word_start, word_end = float(word["start"]), float(word["end"])
            except (KeyError, TypeError, ValueError):
                raise ValueError("Invalid word timestamp") from None
            if not all(math.isfinite(t) for t in (word_start, word_end)) or word_start < 0:
                raise ValueError("Invalid word timestamp")
            # Context is not retained speech. A known, finite malformed word
            # strictly outside the cut cannot invalidate an otherwise valid cut.
            # Inside/at the cut it remains a hard failure; raw evidence is kept.
            if word_end <= word_start:
                if max(word_start, word_end) < start-.02 or min(word_start, word_end) > end+.02:
                    continue
                raise ValueError("Invalid word timestamp")
            if word_end <= start or word_start >= end:
                continue
            if word_start < start - 0.02 or word_end > end + 0.02:
                raise ValueError("Cut straddles a spoken word; adjust source boundary")
            words.append({**word, "start": max(start, word_start), "end": min(end, word_end)})
    if not words and wordless_overlap:
        raise ValueError("Missing word timestamps inside selected range")
    words.sort(key=lambda word: (word["start"], word["end"]))
    for previous, current in zip(words, words[1:]):
        if current["start"] < previous["end"] - 0.02:
            raise ValueError("Overlapping word timestamps require alignment review")
    return words


def alignment_issues(segments: list[dict], start: float, end: float) -> list[str]:
    """Hard cut-integrity defects only: invalid/straddled/overlapping words,
    no speech, or dead air at the hook/payoff. Pauses inside the clip and ASR
    confidence are review flags (alignment_flags), never cut-killers: a wicket
    roar or stunned silence is often the best beat of the clip."""
    try:
        words = words_in_range(segments, start, end)
    except (TypeError, ValueError) as exc:
        return [str(exc)]
    if not words:
        return ["No word-timed speech"]
    issues = []
    if words[0]["start"] - start > EDGE_GAP_SECONDS:
        issues.append(f"Dead air at opening exceeds {EDGE_GAP_SECONDS}s")
    if end - words[-1]["end"] > EDGE_GAP_SECONDS:
        issues.append(f"Dead air at ending exceeds {EDGE_GAP_SECONDS}s")
    probabilities = _probabilities(words)
    if probabilities and sum(probabilities) / len(probabilities) < MIN_MEAN_PROBABILITY:
        issues.append(f"ASR mean confidence below {MIN_MEAN_PROBABILITY}: speech unintelligible to ASR")
    return issues


# Edge dead air ruins the hook; interior pauses do not. The confidence floor
# only rejects ASR output that is effectively noise. Both are unmeasured on
# real Hinglish sources (cancer register C2): measure before tightening.
EDGE_GAP_SECONDS = 1.5
MIN_MEAN_PROBABILITY = 0.4
FLAG_GAP_SECONDS = 2.0
FLAG_MEAN_PROBABILITY = 0.75


def _probabilities(words):
    values = []
    for word in words:
        try:
            p = float(word["probability"])
        except (KeyError, TypeError, ValueError):
            return []
        if not (math.isfinite(p) and 0 <= p <= 1):
            return []
        values.append(p)
    return values


def alignment_flags(segments: list[dict], start: float, end: float) -> list[str]:
    """Advisory caption/pacing notes shown to the reviewer. Never a hold."""
    try:
        words = words_in_range(segments, start, end)
    except (TypeError, ValueError):
        return []
    flags = []
    gaps = [round(c["start"] - p["end"], 2) for p, c in zip(words, words[1:])
            if c["start"] - p["end"] > FLAG_GAP_SECONDS]
    if gaps:
        flags.append(f"{len(gaps)} pause(s) over {FLAG_GAP_SECONDS}s (longest {max(gaps)}s): reaction beat or trim")
    probabilities = _probabilities(words)
    if not probabilities:
        flags.append("ASR confidence unavailable: check caption words")
    else:
        mean = sum(probabilities) / len(probabilities)
        low = [w["word"].strip() for w in words if float(w["probability"]) < 0.5]
        if mean < FLAG_MEAN_PROBABILITY or len(low) / len(words) > 0.15:
            flags.append(f"Low ASR confidence (mean {mean:.2f}); check captions: {' '.join(low[:12])}")
    return flags


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
                vad_filter=False, beam_size=5, **ASR_DECODING,
            )
            result = []
            for segment in segments:
                words = [{"start": float(context_start + word.start), "end": float(context_start + word.end),
                          "word": word.word, "probability": float(word.probability)}
                         for word in (segment.words or [])]
                result.append({"start": float(context_start + segment.start), "end": float(context_start + segment.end),
                                "text": segment.text, "words": words,
                                "requested_range": [start, end],
                                "requested_context_range": [context_start, context_end],
                               "timing_method": "local_asr_word_estimates"})
    if not any(segment["words"] for segment in result):
        raise ValueError("Local ASR produced no word timestamps; hold for review")
    return result
