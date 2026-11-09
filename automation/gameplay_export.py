"""Shared gameplay pipeline entry point. Original speech is authoritative for time/audio.

Uses one provenance-bound continuous gameplay range, never a random download,
loop, frozen last frame, automatic retime, or truncated payoff fade. Stock
availability does not fabricate rights clearance or finished-file release approval.
"""

import json
import math
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

from automation.artifacts import atomic_json
from automation.cricket_gameplay import (COMPOSITION, STADIUM_COMPOSITION, load_render_library, select_gameplay,
                                         sha256, source_crop)


def _duration(path):
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True, timeout=20)
    duration = float(json.loads(result.stdout)["streams"][0]["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid gameplay/source duration")
    return duration


def source_audio_filter(source, start, duration):
    """Measure the exact retained voice before two-pass loudness normalization."""
    base = f"atrim=duration={duration:.6f},asetpts=PTS-STARTPTS,aresample=48000"
    target = "loudnorm=I=-16:TP=-1.5:LRA=11"
    measured = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-ss", f"{start:.6f}",
        "-t", f"{duration:.6f}", "-i", str(source), "-map", "0:a:0", "-vn", "-af",
        f"{base},{target}:print_format=json", "-f", "null", "-"],
        check=True, capture_output=True, text=True, timeout=180)
    import re
    packets = re.findall(r'\{[^{}]*"input_i"[^{}]*\}', measured.stderr, re.DOTALL)
    if not packets:
        raise ValueError("Source voice loudness measurement unavailable")
    stats = json.loads(packets[-1])
    fields = {name: float(stats[name]) for name in
              ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")}
    if not all(math.isfinite(value) for value in fields.values()):
        raise ValueError("Source voice loudness measurement is not finite")
    options = (f":measured_I={fields['input_i']}:measured_TP={fields['input_tp']}"
               f":measured_LRA={fields['input_lra']}:measured_thresh={fields['input_thresh']}"
               f":offset={fields['target_offset']}:linear=true")
    # loudnorm's dynamic path can under-flush a non-round final audio block.
    # Supply only an internal silent guard, then discard it after normalization;
    # never pad the delivered ending in place of actual retained source speech.
    audio_filter = (f"{base},apad=pad_dur=0.25,{target}{options},aresample=48000,"
                    f"atrim=duration={duration:.6f}")
    return audio_filter, {"method": "two-pass exact retained source voice",
        "source_range": [start, start+duration], "measurement": fields,
        "target_lufs": -16, "target_true_peak_dbtp": -1.5,
        "internal_flush_guard_seconds": .25, "output_sample_rate": 48000,
        "output_trim_duration": duration}


def pipeline_gameplay_seconds(config):
    """A discovery run must support its configured ceiling before spending on ASR."""
    seconds = config.get("clip_selection", {}).get("output_max_duration",
        config.get("highlight", {}).get("max_duration", 38))
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Gameplay preflight duration ceiling must be finite and positive")
    return seconds


def preflight_library(library, composition=COMPOSITION, *, required_seconds=None):
    """Probe approved files before spending on ASR/captions/editorial processing."""
    for asset in library["assets"]:
        duration = _duration(asset["path"])
        if any(end > duration+.001 for _, end in asset["approved_ranges"]):
            raise ValueError(f"{asset['id']}: approved gameplay range exceeds actual media duration")
        if asset.get("crop") is not None or composition == STADIUM_COMPOSITION:
            from automation.stadium_export import validate_gameplay_crop
            validate_gameplay_crop(asset)
    if required_seconds is not None:
        # Reject obvious duration shortages without trying to decode them.
        select_gameplay(library, required_seconds, "preflight",
                        preferred_game="EA Cricket 07" if composition == STADIUM_COMPOSITION else None)
        from automation.gameplay_visual_qc import screen_library
        screen_library(library)
        select_gameplay(library, required_seconds, "preflight",
                        preferred_game="EA Cricket 07" if composition == STADIUM_COMPOSITION else None)
    if composition == STADIUM_COMPOSITION:
        from automation.stadium_layout import preflight_assets, preflight_runtime
        preflight_assets()
        preflight_runtime()
    return library


def _label_image(path, disclaimer):
    from PIL import Image, ImageDraw, ImageFont
    from automation.caption_design import caption_font
    lines = ["Illustrative gameplay | Not match footage"]
    if disclaimer:
        lines += textwrap.wrap(disclaimer, width=64)
    font = ImageFont.truetype(str(caption_font()), 28)
    image = Image.new("RGBA", (1080, 44+len(lines)*38), (7, 17, 31, 245))
    draw = ImageDraw.Draw(image)
    for index, line in enumerate(lines):
        if draw.textbbox((0, 0), line, font=font)[2] > 1016:
            raise ValueError("Gameplay disclaimer exceeds readable label bounds")
        draw.text((32, 20+index*38), line, font=font, fill="#F8FAFC")
    if image.height > 360:
        raise ValueError("Gameplay disclaimer too long; explicit layout review required")
    image.save(path)


def build_command(source, start, end, output, selection, analysis, cards, scratch, options):
    """Build a complete command; validate actual media duration before encoding."""
    duration = end-start
    if not math.isfinite(start+end) or start < 0 or duration <= 0:
        raise ValueError("Invalid source range")
    if selection["end"] > _duration(selection["asset"]["path"]) + .001:
        raise ValueError("Approved gameplay range exceeds actual media duration")
    if end > _duration(source) + .001:
        raise ValueError("Source range exceeds actual media duration")
    if abs(selection["end"]-selection["start"]-duration) > .000001:
        raise ValueError("Gameplay duration differs from retained source")
    crop = source_crop(analysis)
    x, y, w, h = crop
    fps = int(options.get("fps", 30))
    if not 1 <= fps <= 60:
        raise ValueError("Gameplay FPS must be 1-60")
    # Fit both panels: preserving the whole pitch/action beats blind fill-crops.
    fit = ("scale=1080:840:flags=lanczos:force_original_aspect_ratio=decrease:force_divisible_by=2,"
           "pad=1080:840:(ow-iw)/2:(oh-ih)/2:color=0x07111F,setsar=1")
    graph = (f"[0:v]trim=duration={duration:.6f},setpts=PTS-STARTPTS,crop={w}:{h}:{x}:{y},"
             f"fps={fps},{fit}[host];"
             f"[1:v]trim=duration={duration:.6f},setpts=PTS-STARTPTS,fps={fps},{fit}[game];"
             f"color=c=0x07111F:s=1080x240:r={fps}:d={duration:.6f}[band];"
             "[host][band][game]vstack=inputs=3[stack]")
    cmd = ["ffmpeg", "-nostdin", "-n", "-v", "error", "-xerror", "-ss", f"{start:.6f}",
           "-t", f"{duration:.6f}", "-i", str(source), "-ss", f"{selection['start']:.6f}",
           "-t", f"{duration:.6f}", "-i", selection["asset"]["path"]]
    # Permanent illustrative label; publisher disclaimer is also visible from
    # the first frame (not hidden in metadata or a tiny credits-only final frame).
    label = Path(scratch)/"gameplay-label.png"
    _label_image(label, selection["disclaimer"])
    cmd += ["-i", str(label)]
    graph += ";[stack][2:v]overlay=0:1080:repeatlast=1[labeled]"
    last = "[labeled]"
    for index, card in enumerate(cards, 3):
        if not 0 <= card.start < card.end <= duration+.02:
            raise ValueError("Caption interval outside retained source")
        cmd += ["-i", card.path]
        next_label = f"[cap{index}]"
        graph += (f";{last}[{index}:v]overlay=0:800:repeatlast=1:"
                  f"enable='gte(t,{card.start:.6f})*lt(t,{card.end:.6f})'{next_label}")
        last = next_label
    graph += f";{last}format=yuv420p[vout]"
    audio_filter, audio_evidence = source_audio_filter(source, start, duration)
    cmd += ["-filter_complex", graph, "-map", "[vout]", "-map", "0:a:0",
            "-af", audio_filter,
            "-c:v", "libx264", "-preset", str(options.get("encoder_preset", "medium")),
            "-crf", str(options.get("crf", 18)), "-c:a", "aac", "-b:a", "192k",
            "-t", f"{duration:.6f}", "-movflags", "+faststart", str(output)]
    return cmd, {"composition": COMPOSITION, "canvas": [1080, 1920],
        "top": [0, 0, 1080, 840], "captions": [0, 840, 1080, 240], "bottom": [0, 1080, 1080, 840],
        "source_crop": crop, "fit": "preserve aspect and pad; no synthesized content",
        "audio": "source input 0 only; no fades; natural speed", "audio_normalization": audio_evidence,
        "selection": selection,
        "verification": "encoded QC and owner listening/visual/rights review still required"}


def render_gameplay_clip(source, start, end, output, analysis, transcript, config):
    from automation.caption_engine import prepare_clip_captions
    from automation.delivery_qc import check_delivery
    from automation.speech_alignment import alignment_issues
    options = config.get("export", {})
    composition = analysis.get("composition", options.get("composition", COMPOSITION))
    if (int(options.get("width", 1080)), int(options.get("height", 1920))) != (1080, 1920):
        raise ValueError("Gameplay v1 requires 1080x1920")
    if not transcript or alignment_issues(transcript, start, end):
        raise ValueError("Gameplay requires intact source-word alignment")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Preserve existing gameplay export: {output}")
    # Re-read permissions/media before each render, even when preflight ran.
    manifest = analysis.get("gameplay_manifest") or options.get("gameplay_manifest")
    if not manifest:
        raise ValueError("Gameplay manifest required; no automatic footage substitution")
    library = preflight_library(load_render_library(manifest), composition, required_seconds=end-start)
    selected = select_gameplay(library, end-start,
        f"{Path(source).resolve()}:{Path(source).stat().st_size}:{start:.6f}:{end:.6f}",
        preferred_game="EA Cricket 07" if composition == STADIUM_COMPOSITION else None)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(config.get("paths", {}).get("temp", "temp"))
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gameplay_", dir=temporary_root) as directory:
        scratch = Path(directory)
        captions = options.get("captions", {})
        cards = prepare_clip_captions(transcript, start, end, scratch,
            max_words_per_cue=captions.get("max_words_per_cue", 4), require_word_timing=True,
            hinglish_model=captions.get("model"),
            allow_documented_successor=captions.get("allow_documented_successor", False),
            caption_style="stadium-reference-v1" if composition == STADIUM_COMPOSITION else "gameplay-hud-v1")
        if not cards:
            raise ValueError("Gameplay requires nonempty source-word captions")
        builder = build_command
        if composition == STADIUM_COMPOSITION:
            from automation.stadium_export import build_stadium_command
            builder = build_stadium_command
        cmd, evidence = builder(source, start, end, str(output), selected, analysis, cards, scratch, options)
        subprocess.run(cmd, check=True, capture_output=True,
                       timeout=float(options.get("ffmpeg_timeout_seconds", 900)))
        for name, suffix in (("caption_design.json", "_caption_design.json"),
                             ("caption_processing.json", "_captions.json")):
            if (scratch/name).exists():
                shutil.copy2(scratch/name, output.with_name(output.stem+suffix))
    # Keep the panel manifest available to every delivery-QC pass, including
    # the orchestrator's later recheck. Persist failed-render evidence too.
    evidence["media_sha256"] = sha256(output)
    atomic_json(output.with_name(output.stem+"_gameplay.json"), evidence)
    # A full decode is a technical check, never an owner speech/meaning approval.
    qc = check_delivery(output, expected_duration=end-start)
    if qc.get("status") != "PASS":
        raise ValueError("Gameplay encoded delivery QC failed; preserve draft and QC for diagnosis")
    current_library = load_render_library(manifest)
    if (current_library["manifest_sha256"] != selected["manifest_sha256"]
            or selected["asset"] not in current_library["assets"]):
        raise ValueError("Gameplay permissions or media changed during rendering")
    evidence["media_sha256"] = sha256(output)
    atomic_json(output.with_name(output.stem+"_gameplay.json"), evidence)
    analysis["gameplay_layout"] = evidence
    analysis["edit_timeline"] = [{"source_start": start, "source_end": end, "speed": 1.0,
                                  "output_start": 0.0, "output_end": end-start}]
    return str(output)
