"""
transcribe.py — Phase 2: Speech-to-Text with GPU-optimized faster-whisper.
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("transcribe", cfg["logging"]["log_file"], cfg["logging"]["level"])


def _log_vram(tag: str = ""):
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            parts = r.stdout.strip().split(", ")
            if len(parts) == 2:
                used, total = int(parts[0]), int(parts[1])
                log.info("VRAM [%s] %d / %d MB (%.0f%%)", tag, used, total, used / total * 100)
    except Exception:
        pass


def transcribe(video_path: str, output_path: str):
    """
    Transcribe audio from video using faster-whisper (GPU-accelerated).
    Uses configurable batch_size and VAD filter for maximum GPU utilization.
    """
    t_cfg = cfg["transcription"]
    video = str(Path(video_path))

    _log_vram("start")

    target_device = t_cfg.get("device", "cpu")
    compute_type = t_cfg.get("compute_type", "int8")
    batch_size = t_cfg.get("batch_size", 16)
    beam_size = t_cfg.get("beam_size", 5)
    vad_filter = t_cfg.get("vad_filter", True)

    # Normalize "auto" → cuda ONLY on NVIDIA GPUs (faster-whisper/CTranslate2
    # only supports NVIDIA CUDA, NOT AMD ROCm on Windows). AMD GPUs fall back
    # to CPU int8 automatically.
    if target_device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                # Double-check it's actually NVIDIA (not AMD ROCm misreported)
                gpu_name = torch.cuda.get_device_name(0).lower()
                if "nvidia" in gpu_name or "geforce" in gpu_name or "rtx" in gpu_name or "tesla" in gpu_name or "titan" in gpu_name:
                    target_device = "cuda"
                    log.info("NVIDIA GPU detected: %s — using CUDA", torch.cuda.get_device_name(0))
                else:
                    target_device = "cpu"
                    log.warning("Non-NVIDIA GPU detected (%s) — CTranslate2 only supports NVIDIA CUDA. Using CPU.",
                                torch.cuda.get_device_name(0))
            else:
                target_device = "cpu"
        except Exception:
            target_device = "cpu"

    # Warn if user explicitly set "cuda" but no NVIDIA GPU is available
    if target_device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                log.warning("CUDA requested but not available — falling back to CPU")
                target_device = "cpu"
        except Exception:
            log.warning("CUDA requested but torch not available — falling back to CPU")
            target_device = "cpu"

    # Normalize "auto" compute_type → float16 on GPU, int8 on CPU
    if compute_type == "auto":
        compute_type = "float16" if target_device == "cuda" else "int8"

    devices_to_try = []
    if target_device == "cuda":
        devices_to_try = [("cuda", compute_type), ("cpu", "int8")]
    else:
        devices_to_try = [(target_device, compute_type)]

    model = None
    used_device = None
    for device, comp in devices_to_try:
        try:
            log.info("Loading Whisper: model=%s device=%s compute=%s batch_size=%d",
                     t_cfg["model"], device, comp, batch_size)
            model = WhisperModel(
                t_cfg["model"], device=device, compute_type=comp,
                cpu_threads=0, num_workers=1,
            )
            used_device = device
            log.info("Transcription engine ready: %s / %s", device.upper(), comp)
            break
        except Exception as e:
            log.warning("Device %s failed: %s", device.upper(), e)

    if not model:
        log.error("[EXIT] transcribe: all GPU/CPU Whisper devices failed for model=%s", t_cfg["model"])
        log.error("[EXIT] Falling back to CPU/int8 — this will be very slow")
        model = WhisperModel(t_cfg["model"], device="cpu", compute_type="int8")
        used_device = "cpu"

    _log_vram("model_loaded")

    language = t_cfg.get("language") or None
    log.info("Transcribing: %s language=%s batch_size=%d beam_size=%d vad=%s",
             video, language or "auto", batch_size, beam_size, vad_filter)

    transcribe_kwargs = dict(
        language=language,
        beam_size=beam_size,
        word_timestamps=True,
        vad_filter=vad_filter,
        vad_parameters=dict(min_silence_duration_ms=500, threshold=0.5),
    )

    # Wire the (previously dead) batch_size config: faster-whisper's
    # WhisperModel.transcribe has no batch_size; batching requires the
    # BatchedInferencePipeline. Use it on GPU for higher utilization, with a
    # safe fallback to sequential decoding if it's unavailable or errors.
    engine = model
    if batch_size and batch_size > 1 and used_device == "cuda":
        try:
            from faster_whisper import BatchedInferencePipeline
            engine = BatchedInferencePipeline(model=model)
            transcribe_kwargs["batch_size"] = batch_size
            log.info("Using BatchedInferencePipeline (batch_size=%d) for GPU throughput", batch_size)
        except Exception as e:
            log.warning("Batched pipeline unavailable (%s) — using sequential decode", e)
            engine = model

    try:
        segments, info = engine.transcribe(video, **transcribe_kwargs)
    except Exception as e:
        if engine is model:
            raise
        # Batched engine failed (unsupported kwargs, decode error, etc.) —
        # fall back once to sequential decoding on the base model.
        log.warning("Batched transcribe failed (%s) — falling back to sequential", e)
        transcribe_kwargs.pop("batch_size", None)
        segments, info = model.transcribe(video, **transcribe_kwargs)

    _log_vram("transcribing")

    dur = info.duration
    log.info("Audio: language=%s duration=%.1fs", info.language, dur)

    results = []
    last_log = time.monotonic()
    last_pct = -10.0
    pct_step = float(t_cfg.get("progress_percent_step", 10))
    interval = float(t_cfg.get("progress_interval_seconds", 15))
    prog_start = time.monotonic()

    for segment in segments:
        if dur:
            pct = segment.end / dur * 100
        else:
            pct = 0.0
        now = time.monotonic()
        if pct >= last_pct + pct_step or now - last_log >= interval:
            elapsed = now - prog_start
            eta = ""
            if pct > 1:
                eta_s = (elapsed / pct) * (100 - pct)
                eta = ", ETA %.0fs" % eta_s
            log.info("Progress: %.1f%% (%ds/%ds)%s", pct, int(segment.end), int(dur), eta)
            last_pct = pct
            last_log = now

        words_data = []
        if getattr(segment, "words", None):
            for w in segment.words:
                words_data.append({"start": w.start, "end": w.end, "word": w.word})

        results.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
            "words": words_data,
        })

    _log_vram("done")

    # Post-processing: centralized, guarded cricket spelling correction
    # (shared with the remote transcript fetcher so ALL sources are corrected).
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
            "source": "whisper",
            "language": info.language or "unknown",
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
