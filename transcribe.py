"""
transcribe.py — Phase 2: Speech-to-Text with Progress Heartbeats.
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel  # type: ignore

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("transcribe", cfg["logging"]["log_file"], cfg["logging"]["level"])

def transcribe(video_path: str, output_path: str):
    """
    Transcribe audio from video using faster-whisper.
    Logs progress heartbeats to keep the user informed.
    """
    t_cfg = cfg["transcription"]
    video = str(Path(video_path))
    
    # ─── AUTO-DEVICE SELECTION (Priority: CUDA > CPU) ─────────────────────────
    target_device = t_cfg.get("device", "cpu")
    compute_type = t_cfg.get("compute_type", "int8")

    # If "cuda" is requested or we're on a remote worker, try CUDA first
    devices_to_try = []
    if target_device == "cuda" or (sys.platform == "linux" and target_device != "cpu"):
        devices_to_try = [("cuda", "float16"), ("cpu", "int8")]
    else:
        devices_to_try = [(target_device, compute_type)]

    model = None
    for device, comp in devices_to_try:
        try:
            log.info(f"Attempting to load Whisper model: {t_cfg['model']} (device={device}, compute={comp})")
            model = WhisperModel(t_cfg["model"], device=device, compute_type=comp)
            log.info(f"✅ Success! Using {device.upper()} for transcription.")
            break
        except Exception as e:
            log.warning(f"⚠️  {device.upper()} loading failed: {e}")
            continue

    if not model:
        log.error("❌ Failed to initialize Whisper on any device. Falling back to base CPU/int8...")
        model = WhisperModel(t_cfg["model"], device="cpu", compute_type="int8")

    language = t_cfg.get("language") or None
    log.info(f"Starting Transcription: {video} (language={language or 'auto'})")

    # ─── TRANSCRIPTION WITH HEARTBEATS ────────────────────────────────────────
    segments, info = model.transcribe(video, language=language, beam_size=5)
    
    log.info(f"✅ Audio Stream Ready | Language: {info.language} | Duration: {info.duration:.1f}s")
    log.info("--- Processing Segments ---")

    results = []
    last_progress_log = time.monotonic()
    last_progress_pct = -10.0
    progress_interval = float(t_cfg.get("progress_interval_seconds", 15))
    progress_percent_step = float(t_cfg.get("progress_percent_step", 10))
    
    for segment in segments:
        progress_pct = (segment.end / info.duration * 100) if info.duration else 0.0
        now = time.monotonic()
        if progress_pct >= last_progress_pct + progress_percent_step or now - last_progress_log >= progress_interval:
            log.info("Transcription progress: %.1f%% (%.1fs / %.1fs)", progress_pct, segment.end, info.duration)
            last_progress_pct = progress_pct
            last_progress_log = now
        
        results.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })

    # Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log.info(f"✨ Transcription complete! Saved → {output_path}")

if __name__ == "__main__":
    # Example
    # transcribe("input/video.mp4", "transcripts/video.json")
    pass
