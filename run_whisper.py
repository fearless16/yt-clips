#!/usr/bin/env python3
"""Run Whisper on Colab GPU and save transcript."""
import json, time, sys, os

os.environ["PYTHONUNBUFFERED"] = "1"

PROJECT = "/content/drive/MyDrive/yt-clips"
VIDEO = f"{PROJECT}/input/video.mp4"
OUT_PATH = f"{PROJECT}/transcripts/video_whisper.json"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

if not os.path.exists(VIDEO):
    log(f"ERROR: Video not found at {VIDEO}")
    sys.exit(1)

log(f"Video: {VIDEO} ({os.path.getsize(VIDEO) / 1024 / 1024:.0f}MB)")

from faster_whisper import WhisperModel

log("Loading Whisper small on GPU...")
t0 = time.time()
model = WhisperModel("small", device="cuda", compute_type="float16")
log(f"Model loaded in {time.time()-t0:.1f}s")

log("Starting transcription (language=en)...")
t0 = time.time()
segments, info = model.transcribe(
    VIDEO,
    language="en",
    beam_size=5,
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500),
)

result = []
for seg in segments:
    result.append({
        "start": round(seg.start, 3),
        "end": round(seg.end, 3),
        "text": seg.text.strip()
    })
    if len(result) % 50 == 0:
        elapsed = time.time() - t0
        pct = (seg.end / 8397) * 100
        log(f"  {len(result)} segs | {seg.end:.0f}s/8397s ({pct:.0f}%) | {elapsed:.0f}s elapsed")

elapsed = time.time() - t0
log(f"DONE in {elapsed:.0f}s — {len(result)} segments")

for s in result[:10]:
    log(f"  [{s['start']:.1f}-{s['end']:.1f}] {s['text'][:80]}")

out = {"segments": result, "language": info.language, "source": "whisper"}
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(out, f, ensure_ascii=False)
log(f"Saved {len(result)} segments to {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1024:.0f}KB)")
