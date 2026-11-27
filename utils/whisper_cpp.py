"""whisper.cpp (whisper-cli) wrapper: the only ASR engine in this project.

Runs on the AMD GPU through the ROCm (HIP) build, then the Vulkan build. There is
no CPU/CUDA fallback engine: a missing binary or model is an explicit, actionable
error, never a silent switch to something slower or absent.

Word timings come from ``-ml 1 -sow``: whisper.cpp then emits one segment per
whole word with correctly joined UTF-8 text (the raw token list splits Devanagari
into byte pieces with empty text). Word probability is the mean of that word's
token probabilities. Raw timings are passed through unchanged; callers hold or
retry bad intervals, this module never repairs them.
"""

import json
import subprocess
import uuid
import wave
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# ROCm first (gfx120X support, measured fastest), Vulkan as cross-vendor fallback.
CLI_CANDIDATES = [
    ("rocm", ROOT / "whisper_rocm" / "whisper-cli.exe"),
    ("vulkan", ROOT / "whisper_vulkan" / "whisper-cli.exe"),
    ("vulkan", ROOT / "whisper_vulkan" / "whisper-cli"),
]

MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{name}"

# Presentation grouping only (words are never altered): a pause or length cap
# starts a new segment so downstream readers get thought-sized blocks.
GROUP_MAX_GAP_SECONDS = 0.5
GROUP_MAX_WORDS = 25


class WhisperCppError(RuntimeError):
    """whisper.cpp failed; the message carries the real reason."""


class WhisperCppUnavailable(WhisperCppError):
    """No usable whisper-cli binary or model on this machine."""


def decode_output(raw: bytes) -> tuple[dict, list[str]]:
    """Decode whisper-cli JSON without ever producing mojibake.

    Windows builds can truncate a multibyte character mid-sequence, so strict
    UTF-8 may fail on an otherwise valid document. Single-byte fallbacks
    (cp1252/latin-1) "succeed" on any bytes and destroy Devanagari: never used.
    """
    warnings: list[str] = []
    try:
        return json.loads(raw.decode("utf-8")), warnings
    except UnicodeDecodeError as exc:
        warnings.append(
            "whisper JSON had invalid UTF-8 at byte %d (%s) — recovered with "
            "errors='ignore'; a few truncated characters were dropped" % (exc.start, exc.reason)
        )
    return json.loads(raw.decode("utf-8", errors="ignore")), warnings


# whisper.cpp pads audio to a 30 s window. Measured on real GPU output it then emits tail
# words at/after the end of the supplied audio: zero-length at the clip end, and the last
# word stretched to the window end. Those are not speech evidence (they lie outside the
# audio) and alignment would hold on them.
AUDIO_END_TOLERANCE_SECONDS = 0.02


def wav_seconds(path) -> Optional[float]:
    """Duration of a PCM wav, or None when it cannot be read as one."""
    try:
        with wave.open(str(path), "rb") as handle:
            return handle.getnframes() / float(handle.getframerate())
    except (wave.Error, EOFError, OSError, ZeroDivisionError):
        return None


def parse_words(data: dict, audio_seconds: Optional[float] = None) -> list[dict]:
    """Word-per-segment whisper JSON (``-ml 1 -sow``) -> [{start,end,word,probability}].

    With ``audio_seconds``, words starting at/after the audio end are dropped and a word
    stretched past it is clamped to it. Nothing inside the audio is ever repaired.
    """
    words = []
    for entry in data.get("transcription", []):
        text = str(entry.get("text", "")).strip()
        if not text:
            continue
        offsets = entry.get("offsets", {})
        word = {"start": float(offsets.get("from", 0)) / 1000.0,
                "end": float(offsets.get("to", 0)) / 1000.0,
                "word": text}
        if audio_seconds is not None:
            if word["start"] >= audio_seconds - AUDIO_END_TOLERANCE_SECONDS:
                continue
            word["end"] = min(word["end"], float(audio_seconds))
        probabilities = [float(t["p"]) for t in entry.get("tokens", [])
                         if "p" in t and not str(t.get("text", "")).startswith("[_")]
        if probabilities:
            word["probability"] = sum(probabilities) / len(probabilities)
        words.append(word)
    return words


def group_words(words: list[dict], max_gap: float = GROUP_MAX_GAP_SECONDS,
                max_words: int = GROUP_MAX_WORDS) -> list[dict]:
    """Group words into pipeline segments {start,end,text,words}."""
    segments, current = [], []

    def flush():
        if current:
            segments.append({"start": current[0]["start"], "end": current[-1]["end"],
                             "text": " ".join(w["word"] for w in current), "words": list(current)})
            current.clear()

    for word in words:
        if current and (word["start"] - current[-1]["end"] > max_gap or len(current) >= max_words):
            flush()
        current.append(word)
    flush()
    return segments


def find_cli() -> tuple[str, Path]:
    """Best available whisper-cli as (backend, path)."""
    for backend, path in CLI_CANDIDATES:
        if Path(path).exists():
            return backend, Path(path)
    raise WhisperCppUnavailable(
        "whisper-cli not found (looked for: %s). Install the whisper.cpp ROCm build into "
        "whisper_rocm/ or the Vulkan build into whisper_vulkan/." % ", ".join(str(p) for _, p in CLI_CANDIDATES))


def resolve_model(name: str, models_dir: Optional[Path] = None) -> Path:
    """ggml model file for a model name. Never downloads (multi-GB, unattended)."""
    ggml = f"ggml-{name}.bin"
    path = Path(models_dir if models_dir is not None else MODELS_DIR) / ggml
    if not path.exists():
        raise WhisperCppUnavailable(
            "whisper model %s not found at %s. Download %s to that folder." % (ggml, path, MODEL_URL.format(name=ggml)))
    return path


def build_command(cli: Path, model: Path, wav: Path, out_base: str, *, language: str,
                  beam_size: int = 5, max_context: int = 0, threads: int = 8,
                  processors: int = 1) -> list[str]:
    return [str(cli), "-m", str(model), "-f", str(wav), "-l", language or "hi",
            "-ojf", "-of", out_base, "--no-prints",
            "-ml", "1", "-sow",
            "-bs", str(int(beam_size)), "-mc", str(int(max_context)),
            "-t", str(int(threads)), "-p", str(int(processors))]


def run_whisper_cli(wav: Path, *, model_name: str, language: str, scratch: Path,
                    beam_size: int = 5, max_context: int = 0, threads: int = 8,
                    processors: int = 1, timeout: int = 3600) -> dict:
    """Transcribe a 16 kHz mono wav. Returns {segments, language, backend, model}.

    Output goes to a path unique to this call, so a failed run can never pick up
    another run's JSON.
    """
    backend, cli = find_cli()
    model = resolve_model(model_name)
    out_base = str(Path(scratch) / f"whisper_{backend}_{uuid.uuid4().hex}")
    cmd = build_command(cli, model, Path(wav), out_base, language=language, beam_size=beam_size,
                        max_context=max_context, threads=threads, processors=processors)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise WhisperCppError("whisper-cli timed out after %ss" % timeout) from exc
    except OSError as exc:
        raise WhisperCppError("whisper-cli could not start: %r" % exc) from exc
    if proc.returncode != 0:
        raise WhisperCppError("whisper-cli exited %s: %s" % (proc.returncode, (proc.stderr or "")[-500:]))
    json_path = Path(out_base + ".json")
    if not json_path.exists():
        raise WhisperCppError("whisper-cli produced no JSON output at %s: %s"
                              % (json_path, (proc.stderr or "")[-300:]))
    try:
        data, _warnings = decode_output(json_path.read_bytes())
    except ValueError as exc:
        raise WhisperCppError("whisper-cli JSON unreadable: %r" % exc) from exc
    return {"segments": group_words(parse_words(data, wav_seconds(wav))),
            "language": data.get("result", {}).get("language") or language or "unknown",
            "backend": backend, "model": model.name}
