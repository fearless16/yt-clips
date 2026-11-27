"""
transcribe.py — Phase 2: speech-to-text with whisper.cpp (AMD GPU via ROCm/Vulkan).

whisper.cpp is the only ASR engine (see utils/whisper_cpp.py). A failure raises with
the real reason; callers (automation/transcript.py) log it and report the transcript
as unavailable. There is no CPU/CUDA fallback engine.
"""

import json
import subprocess
import tempfile
import time
from pathlib import Path

from utils import whisper_cpp
from utils.config import load_config
from utils.logger import get_logger
from utils.whisper_cpp import decode_output as decode_whisper_output  # noqa: F401  (public name kept)

cfg = load_config()
log = get_logger("transcribe", cfg["logging"]["log_file"], cfg["logging"]["level"])


def _extract_audio(video_path: str, wav_path) -> None:
    """16 kHz mono PCM for whisper-cli."""
    try:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-y", "-i", str(video_path),
             "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(wav_path)],
            timeout=600, check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise whisper_cpp.WhisperCppError(
            "audio extraction failed (ffmpeg exit %s): %s"
            % (exc.returncode, (exc.stderr or b"")[-300:].decode("utf-8", "replace"))) from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise whisper_cpp.WhisperCppError("audio extraction failed: %r" % exc) from exc


def transcribe(video_path: str, output_path: str):
    """Transcribe a video's audio with whisper.cpp and write the pipeline transcript JSON.

    Raises whisper_cpp.WhisperCppError (with the real reason) on any failure; nothing is
    written then. All intermediates live in a unique scratch folder removed on every exit.
    """
    t_cfg = cfg["transcription"]
    scratch_root = Path(cfg.get("paths", {}).get("temp", "temp"))
    scratch_root.mkdir(parents=True, exist_ok=True)
    language = t_cfg.get("language") or "hi"

    with tempfile.TemporaryDirectory(prefix="whisper-cli-", dir=scratch_root) as directory:
        scratch = Path(directory)
        wav = scratch / "whisper_input.wav"
        log.info("Extracting audio for whisper.cpp: %s", video_path)
        _extract_audio(video_path, wav)
        t0 = time.monotonic()
        result = whisper_cpp.run_whisper_cli(
            wav, model_name=t_cfg.get("model", "large-v3-turbo"), language=language, scratch=scratch,
            beam_size=t_cfg.get("beam_size", 5), threads=t_cfg.get("threads", 8),
            processors=t_cfg.get("processors", 1),
        )
    log.info("whisper.cpp %s (%s) finished in %.1fs: %d segments",
             result["backend"].upper(), result["model"], time.monotonic() - t0, len(result["segments"]))

    results = result["segments"]
    # Centralized, guarded cricket spelling correction (shared with the remote fetcher).
    from utils.transcript_postproc import correct_segments
    results, n_corr = correct_segments(results)
    if n_corr:
        log.info("Cricket spelling correction: %d substitutions", n_corr)

    # LLM-based cricket context correction pass (validated before applying)
    results = correct_segments_with_llm(results)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "segments": results,
            "source": f"whisper-{result['backend']}",
            "language": result["language"],
        }, f, indent=2, ensure_ascii=False)

    log.info("Transcription complete: %d segments -> %s", len(results), output_path)


def correct_segments_with_llm(segments: list[dict]) -> list[dict]:
    """Correct misheard player/team names in transcript segments using LLM."""
    if not segments:
        return segments

    from utils.ai_client import AIClient
    try:
        ai = AIClient()
    except Exception:
        return segments

    if not (ai.opencode_api_key or ai.nvidia_api_key):
        return segments

    log.info("Running LLM transcript correction pass...")
    lines = [f"{i}: {seg['text']}" for i, seg in enumerate(segments)]
    input_text = "\n".join(lines)

    system_instruction = (
        "You are an expert Indian/Pakistani cricket transcript editor. "
        "Your task is to correct misheard player names (e.g. Kohli, Dhoni, Rohit, Starc, Bumrah, Shami, Hardik), "
        "team names, tournaments, and venues in the provided indexed transcript lines. "
        "Preserve the index (number followed by colon) at the start of each line. "
        "Do NOT change the meaning or translate. Output ONLY the corrected lines, one per line."
    )

    try:
        response = ai.generate_text(f"Correct these transcript lines:\n\n{input_text}", system_instruction=system_instruction)
        if response:
            corrected_map = {}
            for line in response.strip().splitlines():
                parts = line.split(":", 1)
                if len(parts) == 2:
                    try:
                        idx = int(parts[0].strip())
                        val = parts[1].strip()
                        corrected_map[idx] = val
                    except ValueError:
                        pass

            # Validate before applying: index must exist, non-empty, sane length.
            # Stops a misbehaving model from silently rewriting/translating lines.
            from utils.transcript_postproc import validate_and_apply_llm_corrections
            segments, applied, rejected = validate_and_apply_llm_corrections(segments, corrected_map)
            log.info("LLM transcript correction: %d applied, %d rejected", applied, rejected)
    except Exception as e:
        log.warning(f"LLM transcript correction failed: {e}")

    return segments
