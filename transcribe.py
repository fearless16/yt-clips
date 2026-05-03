"""
transcribe.py — Phase 2: Speech-to-Text with Progress Heartbeats.
"""

import json
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
    
    log.info(f"Loading Whisper model: {t_cfg['model']} (device={t_cfg['device']}, compute={t_cfg['compute_type']})")

    # ─── ROBUST MODEL INITIALIZATION ──────────────────────────────────────────
    try:
        model = WhisperModel(
            t_cfg["model"],
            device=t_cfg["device"],
            compute_type=t_cfg["compute_type"],
        )
    except Exception as e:
        log.warning(f"⚠️  {t_cfg['device'].upper()} Initialization Failed ({e}). Falling back to CPU...")
        model = WhisperModel(
            t_cfg["model"],
            device="cpu",
            compute_type="int8",
        )

    language = t_cfg.get("language") or None
    log.info(f"Starting Transcription: {video} (language={language or 'auto'})")

    # ─── TRANSCRIPTION WITH HEARTBEATS ────────────────────────────────────────
    segments, info = model.transcribe(video, language=language, beam_size=5)
    
    log.info(f"✅ Audio Stream Ready | Language: {info.language} | Duration: {info.duration:.1f}s")
    log.info("--- Processing Segments ---")

    results = []
    processed_seconds = 0
    
    for segment in segments:
        # Progress Heartbeat
        log.info(f"  [{(segment.end / info.duration * 100):4.1f}%] {segment.start:6.1f}s -> {segment.end:6.1f}s | {segment.text.strip()}")
        
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
