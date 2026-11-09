"""Offline visual-only rerender from existing edit and caption evidence.

python -m tools.portrait_review shorts/<batch>/clip1_edit.json

Never imports an AI client, requests a transcript, generates metadata, uploads,
overwrites a draft or grants release approval. Cached caption WORDING is kept.
"""

import argparse
import json
import math
import subprocess
from pathlib import Path


def cached_cues(payload, duration):
    from automation.caption_engine import CaptionCue
    cues, previous = [], 0.0
    for row in payload["cards"]:
        start, end = float(row["start"]), float(row["end"])
        text = row["text"]
        if (not math.isfinite(start+end) or not previous <= start < end <= duration+.02
                or not isinstance(text, str) or not text.strip()):
            raise ValueError("Invalid cached caption interval/text; no silent repair")
        cues.append(CaptionCue(start, end, text, "", []))
        previous = end
    if not cues:
        raise ValueError("No cached captions; review never calls a provider to invent them")
    return cues


def render(edit_path, camera_bounds=None, delogo=None):
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    import export
    from automation.artifacts import atomic_json
    from automation.caption_engine import render_transparent_cards_batch
    from utils.face_detect import detect_faces_yunet

    edit_path = Path(edit_path).resolve()
    edit = json.loads(edit_path.read_text(encoding="utf-8"))
    source = Path(edit["source_path"])
    start, end = float(edit["source_start"]), float(edit["source_end"])
    duration = end-start
    timeline = edit["timeline"]
    if (not math.isfinite(start+end) or start < 0 or duration <= 0 or len(timeline) != 1
            or timeline[0]["speed"] != 1 or timeline[0]["source_start"] != start
            or timeline[0]["source_end"] != end):
        raise ValueError("Review supports unchanged single-range 1x drafts only")
    caption_path = edit_path.with_name(edit_path.name.replace("_edit.json", "_caption_design.json"))
    cues = cached_cues(json.loads(caption_path.read_text(encoding="utf-8")), duration)
    info = export._get_video_info(str(source))
    active = edit["active_crop"]
    anchor = (active["face_x"]+active["face_w"]/2, active["face_y"]+active["face_h"]/2)
    capture = cv2.VideoCapture(str(source))
    samples = []
    try:
        for timestamp in np.linspace(start+.08, end-.08, 18):
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp)*1000)
            ok, frame = capture.read()
            if not ok:
                raise ValueError("Could not inspect source frame")
            boxes = detect_faces_yunet(frame, score_threshold=.5)
            if not boxes:
                raise ValueError("Missing face sample; refusing an unverified crop")
            x, y, w, h = min(boxes, key=lambda b: (b[0]+b[2]/2-anchor[0])**2
                            +(b[1]+b[3]/2-anchor[1])**2)
            samples.append({"time": float(timestamp), "face_x": x, "face_y": y, "face_w": w, "face_h": h})
    finally:
        capture.release()
    analysis = {"source_dimensions": info, "face_samples": samples,
                "export_strategy": {"use_solo_frame": True, "active_crop": active}}
    if camera_bounds:
        analysis["portrait_source_bounds"] = dict(zip(("x", "y", "width", "height"), camera_bounds))
    # Explicit visual mode; unrelated global config is not mutated.
    if (export.cfg["export"].get("composition") != "portrait-caption-v1"
            or export.cfg["export"]["captions"]["style"] != "portrait-glass-v1"):
        raise ValueError("Enable reference portrait and caption styles before review")
    graph = export._build_enhance_stack(analysis, source_fps=info["fps"], use_logo=False)
    if not analysis.get("portrait_layout"):
        raise ValueError("No dominant host portrait; not a supported review source")
    out = edit_path.parent.parent / ("portrait_review_"+export._new_export_batch_id())
    out.mkdir(exist_ok=False)
    cards = render_transparent_cards_batch(cues, out/"cards", style="portrait-glass-v1")
    media = out/"portrait.mp4"
    seek = max(0, start-4)
    cleanup = ""
    if delogo:
        x, y, w, h = delogo
        if (min(x, y) <= 0 or min(w, h) <= 0 or x+w >= info["width"] or y+h >= info["height"]
                or w*h > info["width"]*info["height"]*.01):
            raise ValueError("Control-label cleanup must be a small explicit source rectangle")
        cleanup = f",delogo=x={x}:y={y}:w={w}:h={h}:show=0"
    graph = (f"[0:v]trim=start={start-seek:.6f}:duration={duration:.6f},setpts=PTS-STARTPTS{cleanup}[v_src];"+graph)
    cmd = ["ffmpeg", "-nostdin", "-n", "-v", "error", "-ss", str(seek),
           "-t", str(end-seek), "-i", str(source)]
    last = "[v_out]"
    for i, card in enumerate(cards, 1):
        cmd += ["-i", card.path]
        label = f"[caption_{i}]"
        graph += (f";{last}[{i}:v]overlay=0:{export._caption_card_y()}:"
                  f"enable='gte(t,{card.start:.6f})*lt(t,{card.end:.6f})'{label}")
        last = label
    cmd += ["-filter_complex", graph, "-map", last, "-map", "0:a:0?", "-af",
            export._build_audio_filter(1, trim_start=start-seek, trim_duration=duration, output_duration=duration),
            "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-t", str(duration), "-movflags", "+faststart", str(media)]
    atomic_json(out/"review.json", {"status": "VISUAL_REVIEW_ONLY", "approval": None,
        "original_edit": str(edit_path), "source": str(source), "source_range": [start, end],
        "crop": analysis["portrait_layout"], "face_samples": samples,
        "reviewed_camera_bounds": camera_bounds, "explicit_source_control_cleanup": delogo,
        "caption_source": str(caption_path), "caption_wording": "UNCHANGED; existing ASR defects remain",
        "paid_api_calls": 0, "cut_changed": False, "filter_graph": graph})
    subprocess.run(cmd, check=True, capture_output=True, timeout=600)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(media), "-f", "null", "-"],
                   check=True, capture_output=True, timeout=120)
    if not export._validate_av_sync(str(media), "portrait_review"):
        raise ValueError("Encoded A/V check failed")
    sheet = Image.new("RGB", (1440, 660))
    for i, timestamp in enumerate((.25, duration/3, duration*2/3, duration-.25)):
        target = out/f"frame-{i+1}.png"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", str(timestamp), "-i", str(media),
                        "-frames:v", "1", str(target)], check=True, capture_output=True, timeout=30)
        with Image.open(target) as image:
            image.thumbnail((360, 640))
            sheet.paste(image, (i*360, 0))
        ImageDraw.Draw(sheet).text((i*360+8, 642), f"{timestamp:.2f}s", fill="white")
    sheet.save(out/"contact.jpg")
    print(media)
    return media


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edit", type=Path)
    parser.add_argument("--camera-bounds", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--delogo", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                        help="Explicit reviewed source-control label, never inferred from a face")
    args = parser.parse_args()
    render(args.edit, args.camera_bounds, args.delogo)
