"""
export.py — Cinema-Grade Export Engine with Immersive Studio Quality.

Features:
  - Smart frame layout detection (solo vs multi-panel)
  - Vertical stacking for multi-frame layouts (top/bottom, not side-by-side)
  - Black screen / dead panel detection and skip
  - Dead air (silence) removal
  - Smart speed adjustment (1.25x for slow segments)
  - Backlight / lighting correction
  - Preview snapshot QA before export
  - Cinema-grade enhancement stack (smooth motion, film grading, audio normalization)
  - Smooth fade-in/fade-out transitions (no cheap hard cuts)
  - Motion interpolation for buttery-smooth 60fps output
"""
import subprocess
import json
import sys
import os
import re
from pathlib import Path
from typing import Dict, List, Optional
from utils.config import load_config
from utils.logger import get_logger
import time
from seo import generate_seo

cfg = load_config()
log = get_logger("export", cfg["logging"]["log_file"], cfg["logging"]["level"])

# ─── Check FFmpeg minterpolate support at module load ─────────────────────────
_MINTERPOLATE_AVAILABLE = None

def _check_minterpolate() -> bool:
    """Check if FFmpeg was compiled with minterpolate support."""
    global _MINTERPOLATE_AVAILABLE
    if _MINTERPOLATE_AVAILABLE is not None:
        return _MINTERPOLATE_AVAILABLE
    try:
        result = subprocess.run(
            ["ffmpeg", "-filters"],
            capture_output=True, text=True, timeout=5,
        )
        _MINTERPOLATE_AVAILABLE = "minterpolate" in result.stdout
        if not _MINTERPOLATE_AVAILABLE:
            log.info("ℹ️ minterpolate not available in this FFmpeg build. Using simple FPS conversion.")
    except Exception:
        _MINTERPOLATE_AVAILABLE = False
    return _MINTERPOLATE_AVAILABLE


def _get_video_info(path: str) -> Dict:
    """Get source video metadata."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,avg_frame_rate", "-of", "json", path]
    res = json.loads(subprocess.check_output(cmd))
    fps_eval = res["streams"][0]["avg_frame_rate"]
    fps = eval(fps_eval) if "/" in fps_eval else float(fps_eval)
    return {"w": res["streams"][0]["width"], "h": res["streams"][0]["height"], "fps": fps}


def _build_enhance_stack(lighting_filter: str = "", source_fps: float = 30.0) -> str:
    """
    Build the cinema-grade enhancement filter chain.

    Philosophy:
      - Gentle denoise (preserve texture, don't flatten)
      - Soft sharpening (detail pop without halos)
      - Cinematic color grading (warm shadows, cool highlights)
      - Subtle contrast + saturation lift (natural, not blown out)
      - Motion interpolation to 60fps (optical flow for smoothness)
    """
    export_cfg = cfg.get("export", {})
    target_fps = export_cfg.get("fps", 60)

    # ─── CINEMA-GRADE ENHANCEMENT STACK ───────────────────────────────────────
    filters = []

    # 1. Gentle temporal + spatial denoise (preserve grain/texture)
    filters.append("hqdn3d=2:2:3:3")

    # 2. Soft detail enhancement (wider kernel, lower intensity = no halos)
    filters.append("unsharp=5:5:0.3:5:5:0.0")

    # 3. Cinematic color grading — warm shadows, cool highlights
    #    rs/gs/bs = shadow color shifts, rh/gh/bh = highlight shifts
    filters.append("colorbalance=rs=0.04:gs=0.01:bs=-0.02:rh=-0.02:gh=0.0:bh=0.03")

    # 4. Contrast & saturation — subtle lift, NOT the overbaked 1.1/1.2
    if lighting_filter:
        # Use the analysis-driven lighting correction instead
        filters.append(lighting_filter)
    else:
        filters.append("eq=contrast=1.05:saturation=1.1:brightness=0.02")

    # 5. Motion interpolation — smooth optical flow to target FPS
    #    Only apply if source FPS < target FPS (avoid slowdown)
    if source_fps < target_fps - 5 and _check_minterpolate():
        # Full optical flow interpolation for buttery smoothness
        filters.append(
            f"minterpolate=fps={target_fps}:mi_mode=mci"
            f":mc_mode=aobmc:me_mode=bidir:vsbmc=1"
        )
    elif source_fps < target_fps:
        # Simple FPS boost (fallback if minterpolate unavailable)
        filters.append(f"fps={target_fps}")

    return ",".join(filters)


def _build_fade_filters(duration: float) -> str:
    """
    Build cinematic fade-in and fade-out video filters.
    Creates smooth black transitions instead of hard cuts.
    """
    fade_cfg = cfg.get("export", {}).get("transitions", {})
    fade_in_dur = fade_cfg.get("fade_in_duration", 0.5)
    fade_out_dur = fade_cfg.get("fade_out_duration", 0.5)

    fade_out_start = max(0, duration - fade_out_dur)

    return (
        f"fade=t=in:st=0:d={fade_in_dur}:color=black,"
        f"fade=t=out:st={fade_out_start:.2f}:d={fade_out_dur}:color=black"
    )


def _build_filtergraph_solo(
    sw: int, sh: int, export_cfg: Dict,
    active_crop: Optional[Dict] = None,
    lighting_filter: str = "",
    speed_factor: float = 1.0,
    source_fps: float = 30.0,
    duration: float = 30.0,
) -> str:
    """
    Build filtergraph for SOLO frame mode — full-screen single panel.
    Used when: no facecam, or dead panel detected, or solo preferred.
    """
    out_w = export_cfg["width"]
    out_h = export_cfg["height"]
    enhance = _build_enhance_stack(lighting_filter, source_fps)
    fades = _build_fade_filters(duration)

    if active_crop:
        # Crop to the active region, then scale to output
        crop = f"crop={active_crop['width']}:{active_crop['height']}:{active_crop['x']}:{active_crop['y']}"
    else:
        # Center-crop from 16:9 to 9:16 aspect ratio
        crop_w = int(sh * (out_w / out_h))
        crop_x = max(0, (sw - crop_w) // 2)
        crop = f"crop={crop_w}:{sh}:{crop_x}:0"

    fg = f"[0:v]{crop},scale={out_w}:{out_h}:flags=lanczos,{enhance},{fades}"

    # Speed adjustment
    if speed_factor != 1.0:
        pts_factor = 1.0 / speed_factor
        fg += f",setpts={pts_factor}*PTS"

    fg += "[out]"
    return fg


def _build_filtergraph_vertical_stack(
    sw: int, sh: int, layout_cfg: Dict, export_cfg: Dict,
    lighting_filter: str = "",
    speed_factor: float = 1.0,
    source_fps: float = 30.0,
    duration: float = 30.0,
) -> str:
    """
    Build filtergraph for VERTICAL STACK mode — gameplay on top, facecam on bottom.
    This is the correct layout for Shorts (9:16 vertical video).
    Frames are stacked VERTICALLY (one above the other), NOT horizontally.
    """
    out_w = export_cfg["width"]
    gp_h = layout_cfg["gameplay_output_height"]
    fc_h = layout_cfg["facecam_output_height"]
    enhance = _build_enhance_stack(lighting_filter, source_fps)
    fades = _build_fade_filters(duration)

    # ─── Gameplay crop (top panel) ────────────────────────────────────────────
    gp_crop_w = int(sh * (out_w / gp_h))
    gp_crop_x = max(0, (sw - gp_crop_w) // 2)

    # ─── Facecam crop (bottom panel) ──────────────────────────────────────────
    fc = layout_cfg["facecam"]

    fg = (
        f"[0:v]split=2[raw1][raw2];"
        # Gameplay: crop center, scale to output width × gameplay height
        f"[raw1]crop={gp_crop_w}:{sh}:{gp_crop_x}:0,"
        f"scale={out_w}:{gp_h}:flags=lanczos,"
        f"{enhance}[gameplay];"
        # Facecam: crop facecam region, scale to output width × facecam height
        f"[raw2]crop={fc['width']}:{fc['height']}:{fc['x']}:{fc['y']},"
        f"scale={out_w}:{fc_h}:flags=lanczos,"
        f"{enhance}[facecam];"
        # VERTICAL STACK: gameplay on top, facecam on bottom
        f"[gameplay][facecam]vstack=inputs=2,{fades}"
    )

    # Speed adjustment
    if speed_factor != 1.0:
        pts_factor = 1.0 / speed_factor
        fg += f",setpts={pts_factor}*PTS"

    fg += "[out]"
    return fg


def _build_audio_filter(
    speed_factor: float = 1.0,
    skip_silence: bool = False,
    duration: float = 30.0,
) -> Optional[str]:
    """Build audio filter chain for normalization, speed, silence removal, and fades."""
    filters = []

    # Audio normalization — YouTube LUFS standard
    filters.append("loudnorm=I=-14:TP=-1:LRA=11")

    if speed_factor != 1.0:
        # atempo only supports 0.5-2.0, chain for larger ranges
        filters.append(f"atempo={speed_factor}")

    if skip_silence:
        # Remove silence using silenceremove
        filters.append(
            "silenceremove=start_periods=1:start_silence=0.5:start_threshold=-35dB:"
            "detection=peak,areverse,"
            "silenceremove=start_periods=1:start_silence=0.5:start_threshold=-35dB:"
            "detection=peak,areverse"
        )

    # Audio fade-in/fade-out for smooth transitions
    fade_cfg = cfg.get("export", {}).get("transitions", {})
    audio_fade_in = fade_cfg.get("audio_fade_in", 0.3)
    audio_fade_out = fade_cfg.get("audio_fade_out", 0.4)
    fade_out_start = max(0, duration - audio_fade_out)

    filters.append(f"afade=t=in:d={audio_fade_in}")
    filters.append(f"afade=t=out:st={fade_out_start:.2f}:d={audio_fade_out}")

    return ",".join(filters) if filters else None


def export_clip(
    video_path: str,
    start: float,
    end: float,
    output_path: str,
    clip_id: str = "clip",
    analysis: Optional[Dict] = None,
) -> Path:
    """
    Export a single clip with intelligent frame analysis and cinema-grade quality.

    If `analysis` is provided (from frame_analyzer.analyze_clip), uses
    smart decisions for layout, speed, lighting, and silence handling.
    Otherwise, falls back to config-based defaults.
    """
    info = _get_video_info(video_path)
    layout_cfg = cfg["layout"]
    export_cfg = cfg["export"]
    duration = end - start

    # ─── Extract Strategy from Analysis ───────────────────────────────────────
    strategy = {}
    if analysis:
        strategy = analysis.get("export_strategy", {})

    use_solo = strategy.get("use_solo_frame", not layout_cfg.get("has_facecam", True))
    active_crop = strategy.get("active_crop")
    speed_factor = strategy.get("speed_factor", 1.0)
    apply_lighting = strategy.get("apply_lighting_fix", False)
    lighting_filter = strategy.get("lighting_filter", "")
    skip_silence = strategy.get("skip_silence", False)

    # ─── Smart Encoder Detection ─────────────────────────────────────────────
    encoder = export_cfg["encoder"]
    if sys.platform == "linux" and encoder == "h264_videotoolbox":
        encoder = "libx264"
        log.info(f"[{clip_id}] Remote Mode: Switching to '{encoder}' encoder.")

    # ─── Build Filtergraph ────────────────────────────────────────────────────
    if use_solo:
        log.info(f"[{clip_id}] 🎬 SOLO frame mode (full-screen single panel)")
        fg = _build_filtergraph_solo(
            info["w"], info["h"], export_cfg,
            active_crop=active_crop,
            lighting_filter=lighting_filter if apply_lighting else "",
            speed_factor=speed_factor,
            source_fps=info["fps"],
            duration=duration,
        )
    else:
        log.info(f"[{clip_id}] 📐 VERTICAL STACK mode (gameplay top + facecam bottom)")
        fg = _build_filtergraph_vertical_stack(
            info["w"], info["h"], layout_cfg, export_cfg,
            lighting_filter=lighting_filter if apply_lighting else "",
            speed_factor=speed_factor,
            source_fps=info["fps"],
            duration=duration,
        )

    # ─── Build Audio Filter ──────────────────────────────────────────────────
    audio_filter = _build_audio_filter(speed_factor, skip_silence, duration)

    # ─── Build FFmpeg Command ─────────────────────────────────────────────────
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start), "-t", str(duration),
        "-i", video_path,
        "-filter_complex", fg,
        "-map", "[out]", "-map", "0:a",
    ]

    if audio_filter:
        cmd.extend(["-af", audio_filter])

    # ─── Encoding settings ───────────────────────────────────────────────────
    #  CRF mode for quality-controlled encoding (better than flat bitrate)
    crf_value = export_cfg.get("crf", 18)
    max_bitrate = export_cfg.get("video_bitrate", "25M")

    if encoder == "libx264":
        cmd.extend([
            "-c:v", encoder,
            "-preset", export_cfg.get("encoder_preset", "medium"),
            "-crf", str(crf_value),
            "-maxrate", max_bitrate,
            "-bufsize", "50M",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",  # Web-optimized MP4
        ])
    else:
        # Hardware encoders (videotoolbox, nvenc) don't support CRF the same way
        cmd.extend([
            "-c:v", encoder,
            "-b:v", max_bitrate,
            "-pix_fmt", "yuv420p",
        ])

    cmd.extend([
        "-c:a", "aac",
        "-b:a", export_cfg["audio_bitrate"],
        "-ar", "48000",
        "-progress", "pipe:1",
        output_path,
    ])

    log.info(f"[{clip_id}] 🎬 Exporting {duration:.1f}s segment "
             f"(speed={speed_factor}x, lighting={'✓' if apply_lighting else '✗'}, "
             f"silence_skip={'✓' if skip_silence else '✗'}, "
             f"minterpolate={'✓' if info['fps'] < export_cfg.get('fps', 60) - 5 else '✗'}) ...")

    # ─── RUN FFMPEG WITH LIVE PROGRESS ───────────────────────────────────────
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True
    )

    for line in process.stdout:
        if "out_time=" in line:
            time_val = line.split("=")[1].strip()
            log.info(f"   ↳ {clip_id} Progress: {time_val} / {duration:.1f}s")

    process.wait()
    if process.returncode != 0:
        raise Exception(f"FFmpeg failed with code {process.returncode}")

    output_size = Path(output_path).stat().st_size / 1_048_576
    log.info(f"[{clip_id}] ✅ Export complete: {output_path} ({output_size:.1f} MB)")

    return Path(output_path)


def export_all(highlights_path: str, video_path: str) -> List[Path]:
    """
    Export all highlights with intelligent frame analysis.
    Each clip is analyzed before export for optimal quality decisions.
    """
    import yaml
    with open(highlights_path, "r") as f:
        highlights = yaml.safe_load(f)

    # Load transcript for tempo analysis
    transcript_segments = _load_transcript_for_analysis(video_path)

    timestamp = time.strftime("%Y-%m-%d_%H%M%S")
    output_dir = Path(cfg["paths"]["shorts"]) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save analysis report
    analysis_report = {}

    exported_paths = []
    for clip_id, info in highlights.items():
        log.info(f"─── Processing {clip_id} ───")

        start_sec = info["start_sec"]
        end_sec = info["end_sec"]

        # ─── Pre-Export Analysis ──────────────────────────────────────────────
        try:
            from frame_analyzer import analyze_clip
            analysis = analyze_clip(
                video_path, start_sec, end_sec,
                transcript_segments=transcript_segments,
                save_preview=True,
                clip_id=clip_id,
            )
            analysis_report[clip_id] = {
                "layout": analysis["layout"]["layout_type"],
                "lighting": analysis["lighting"]["correction_type"],
                "dead_air": analysis["dead_air"]["silence_ratio"],
                "speed": analysis["export_strategy"]["speed_factor"],
                "solo_frame": analysis["export_strategy"]["use_solo_frame"],
            }
        except Exception as e:
            log.warning(f"[{clip_id}] Analysis failed ({e}), using defaults")
            analysis = None

        # ─── Export ───────────────────────────────────────────────────────────
        out_file = output_dir / f"{clip_id}.mp4"
        try:
            p = export_clip(
                video_path, start_sec, end_sec,
                str(out_file), clip_id=clip_id,
                analysis=analysis,
            )
            exported_paths.append(p)

            # Generate SEO Metadata
            seo_data = generate_seo(info.get("text", "Cricket Highlights"), clip_id)
            with open(output_dir / f"{clip_id}_metadata.json", 'w') as f:
                json.dump(seo_data, f, indent=4)
        except Exception as e:
            log.error(f"Failed {clip_id}: {e}")

    # Save analysis report
    report_path = output_dir / "analysis_report.json"
    with open(report_path, "w") as f:
        json.dump(analysis_report, f, indent=2)
    log.info(f"📊 Analysis report saved: {report_path}")

    return exported_paths


def _load_transcript_for_analysis(video_path: str) -> List[Dict]:
    """Load transcript segments for tempo analysis."""
    stem = Path(video_path).stem
    transcript_path = Path(cfg["paths"]["transcripts"]) / f"{stem}.json"
    if transcript_path.exists():
        with open(transcript_path, "r") as f:
            return json.load(f)
    return []
