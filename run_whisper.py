#!/usr/bin/env python3
"""Run Whisper on Colab GPU and save transcript."""
import json, time, sys
from pathlib import Path

PROJECT = "/content/drive/MyDrive/yt-clips"
VIDEO = f"{PROJECT}/input/video.mp4"

if not Path(VIDEO).exists():
    print(f"ERROR: Video not found at {VIDEO}")
    sys.exit(1)

from faster_whisper import WhisperModel

print("Loading Whisper base on GPU...")
t0 = time.time()
model = WhisperModel("base", device="cuda", compute_type="float16")
print(f"Model loaded in {time.time()-t0:.1f}s")

print(f"Transcribing: {VIDEO}")
t0 = time.time()
segments, info = model.transcribe(
    VIDEO,
    language="hi",
    beam_size=5,
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500)
)

result = []
for seg in segments:
    result.append({
        "start": round(seg.start, 3),
        "end": round(seg.end, 3),
        "text": seg.text.strip()
    })

elapsed = time.time() - t0
print(f"Done in {elapsed:.0f}s - {len(result)} segments, lang={info.language}")

for s in result[:15]:
    print(f"  [{s['start']:.1f}-{s['end']:.1f}] {s['text'][:80]}")

out = {"segments": result, "language": info.language, "source": "whisper"}
out_path = f"{PROJECT}/transcripts/video_whisper.json"
with open(out_path, "w") as f:
    json.dump(out, f, ensure_ascii=False)
print(f"Saved to {out_path}")
