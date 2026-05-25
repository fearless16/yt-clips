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

    devices_to_try = []
    if target_device == "cuda" or (sys.platform == "linux" and target_device != "cpu"):
        devices_to_try = [("cuda", compute_type), ("cpu", "int8")]
    else:
        devices_to_try = [(target_device, compute_type)]

    model = None
    for device, comp in devices_to_try:
        try:
            log.info("Loading Whisper: model=%s device=%s compute=%s batch_size=%d",
                     t_cfg["model"], device, comp, batch_size)
            model = WhisperModel(
                t_cfg["model"], device=device, compute_type=comp,
                cpu_threads=0, num_workers=1,
            )
            log.info("Transcription engine ready: %s / %s", device.upper(), comp)
            break
        except Exception as e:
            log.warning("Device %s failed: %s", device.upper(), e)

    if not model:
        log.error("[EXIT] transcribe: all GPU/CPU Whisper devices failed for model=%s", t_cfg["model"])
        log.error("[EXIT] Falling back to CPU/int8 — this will be very slow")
        model = WhisperModel(t_cfg["model"], device="cpu", compute_type="int8")

    _log_vram("model_loaded")

    language = t_cfg.get("language") or None
    log.info("Transcribing: %s language=%s batch_size=%d beam_size=%d vad=%s",
             video, language or "auto", batch_size, beam_size, vad_filter)

    segments, info = model.transcribe(
        video, language=language,
        beam_size=beam_size,
        word_timestamps=True,
        batch_size=batch_size,
        vad_filter=vad_filter,
        vad_parameters=dict(min_silence_duration_ms=500, threshold=0.5),
    )

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

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "segments": results,
            "source": "whisper",
            "language": info.language or "unknown",
        }, f, indent=2, ensure_ascii=False)

    log.info("Transcription complete: %d segments -> %s", len(results), output_path)
