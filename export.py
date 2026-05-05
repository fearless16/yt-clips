import subprocess
import json
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional

from utils.config import load_config
from utils.logger import get_logger
from frame_analyzer import analyze_clip

cfg = load_config()
log = get_logger("export", cfg["logging"]["log_file"], cfg["logging"]["level"])

# Cache for best encoder to avoid re-testing
_BEST_ENCODER = None

def _get_best_encoder() -> str:
    """
    Detects and verifies the most powerful hardware encoder available.
    Includes a 'smoke test' to ensure the encoder actually works in the current environment.
    """
    global _BEST_ENCODER
    if _BEST_ENCODER:
        return _BEST_ENCODER

    # Priority list of hardware encoders
    # videotoolbox (Mac), nvenc (NVIDIA), qsv (Intel), vaapi (Linux)
    candidates = ["h264_nvenc", "h264_videotoolbox", "h264_qsv", "h264_vaapi"]
    
    try:
        result = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True)
        available = result.stdout
        
        for enc in candidates:
            if enc in available:
                log.info(f"🔍 Testing hardware encoder: {enc} ...")
                if _smoke_test_encoder(enc):
                    log.info(f"✅ {enc} passed smoke test. Using for export.")
                    _BEST_ENCODER = enc
                    return enc
                else:
                    log.warning(f"⚠️ {enc} detected but failed smoke test. Falling back...")
                    
    except Exception as e:
        log.error(f"Error during encoder detection: {e}")

    log.info("ℹ️ No functional hardware encoder detected. Using software 'libx264'.")
    _BEST_ENCODER = "libx264"
    return _BEST_ENCODER

def _smoke_test_encoder(encoder: str) -> bool:
    """
    Runs a 0.5s dummy encode to verify the encoder is functional.
    Crucial for Colab where drivers might be missing despite tool availability.
    """
    temp_test = "temp/smoke_test.mp4"
    os.makedirs("temp", exist_ok=True)
    
    # Generate a tiny 0.5-second dummy video
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.5",
        "-vf", "format=yuv420p",
        "-c:v", encoder,
        "-preset", "p4",
        "-t", "0.5",
        temp_test
    ]
    try:
        # Hide output for smoke test
        res = subprocess.run(cmd, capture_output=True, timeout=15)
        success = res.returncode == 0 and Path(temp_test).exists()
        if Path(temp_test).exists():
            Path(temp_test).unlink()
        return success
    except Exception:
        return False

def _get_video_info(path: str) -> Dict:
    """Extract width, height, and FPS using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "json", path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        
        # Calculate FPS from fraction (e.g. "30/1")
        fps_parts = stream["r_frame_rate"].split("/")
        fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0
        
        return {
            "width": stream["width"],
            "height": stream["height"],
            "fps": fps
        }
    except Exception as e:
        log.error(f"Failed to get video info: {e}")
        return {"width": 1920, "height": 1080, "fps": 30.0}

def _build_enhance_stack(analysis: Dict, source_fps: float = 30.0) -> str:
    """
    Builds the filtergraph based on Frame Analysis decisions.
    Supports SOLO mode (full-screen) and VERTICAL STACK (blurred background).
    """
    strategy = analysis["export_strategy"]
    use_solo = strategy.get("use_solo_frame", False)
    
    target_w = cfg["export"]["width"]   # 1080
    target_h = cfg["export"]["height"]  # 1920
    
    # 1. Base Layer Construction
    if use_solo:
        log.info("🎬 SOLO frame mode (full-screen single panel)")
        # Solo Mode: Take the center 9:16 slice and scale it to fill
        # We assume 16:9 source, so we crop the center 607x1080 (for 1920x1080)
        # More generally: crop width = (9/16) * height
        filter_base = (
            f"crop=ih*9/16:ih,"
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
            f"crop={target_w}:{target_h}"
        )
    else:
        log.info("🎬 STACK mode (blurred background + sharp center)")
        # Stack Mode: Blur background (split layer 1) + Sharp center (split layer 2)
        filter_base = (
            f"split=2[bg][fg];"
            f"[bg]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h},boxblur=20:10[bg_fin];"
            f"[fg]scale={target_w}:-1,crop={target_w}:min(ih\\,{target_h})[fg_scaled];"
            f"[bg_fin][fg_scaled]overlay=(W-w)/2:(H-h)/2"
        )

    # 2. Add Lighting/Color Fixes if recommended
    if strategy.get("apply_lighting_fix"):
        l_filter = strategy.get("lighting_filter", "")
        if l_filter:
            filter_base += f",{l_filter}"

    # 3. Add Motion Interpolation (Min-terpolate) for 60fps look
    # Only apply if target FPS > source FPS
    target_fps = cfg["export"].get("fps", 60)
    if target_fps > source_fps + 5:
        # Using 'scdet' to avoid blending across cuts
        filter_base += f",minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsfm=1"

    # 4. Add Branding / Logo overlay
    logo_path = cfg["thumbnail"].get("template_path", "channel_logo.png")
    if Path(logo_path).exists():
        # This requires a second input [1:v], handled in export_clip
        # Position top-right with margin
        filter_base = f"[v_src]{filter_base}[v_final];[1:v]scale=120:-1[logo];[v_final][logo]overlay=W-w-40:40,format=yuv420p"
        return filter_base
    
    return f"[v_src]{filter_base},format=yuv420p"

def export_clip(video_path: str, start: float, end: float, output_path: str, clip_id: str = "clip") -> str:
    """
    Exports a single clip using Hybrid CPU/GPU pipeline.
    CPU handles the heavy filtergraph (stabilized), GPU handles the final encoding (fast).
    """
    t_start = time.perf_counter()
    
    # 1. Run Intelligence Analysis
    analysis = analyze_clip(video_path, start, end, clip_id=clip_id)
    strategy = analysis["export_strategy"]
    
    # 2. Get hardware info
    info = _get_video_info(video_path)
    encoder = _get_best_encoder()
    
    # 3. Build filter chain
    v_filter = _build_enhance_stack(analysis, source_fps=info["fps"])
    
    # 4. Build audio chain
    speed = strategy.get("speed_factor", 1.0)
    # Removed atrim because input `-t` already perfectly frames the segment.
    # Applied atempo for speed adjustment.
    a_filter = f"aresample=44100,atempo={speed},loudnorm=I=-16:TP=-1.5:LRA=11"
    
    # 5. Execute FFmpeg
    logo_path = cfg["thumbnail"].get("template_path", "channel_logo.png")
    
    cmd =[
        "ffmpeg", "-y",
        "-threads", "0",
        "-ss", str(start),
        "-t", str(end - start),  # FIX: Read the exact source duration, speed reduction happens in filters
        "-i", video_path
    ]
    
    if Path(logo_path).exists():
        cmd.extend(["-i", logo_path])
        
    cmd.extend([
        "-filter_complex", f"[0:v]setpts=PTS/{speed}[v_src];{v_filter}",
        "-af", a_filter,
        "-c:v", encoder,
        "-b:v", str(cfg["export"].get("video_bitrate", "8M")),
        "-maxrate", "15M",
        "-bufsize", "24M",
        "-c:a", "aac",
        "-b:a", str(cfg["export"].get("audio_bitrate", "192k"))
    ])
    
    # Presets based on encoder
    if encoder == "h264_nvenc":
        cmd.extend(["-preset", "p4", "-tune", "hq"])
    elif encoder == "libx264":
        cmd.extend(["-preset", "veryfast", "-crf", str(cfg["export"].get("crf", 23))])
    
    # Global optimizations
    cmd.extend(["-movflags", "+faststart"])
    
    cmd.append(output_path)
    
    log.info(f"[{clip_id}] 🎬 Exporting {end-start:.1f}s segment (speed={speed}x, encoder={encoder}) ...")
    
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            log.error(f"FFmpeg failed for {clip_id}: {res.stderr}")
            return None
        
        dur = time.perf_counter() - t_start
        # Realtime calculation factored for actual processing time vs target video time
        output_duration = (end - start) / speed
        log.info(f"✅ [{clip_id}] Export complete in {dur:.1f}s ({output_duration/dur:.1f}x real-time)")
        return output_path
        
    except Exception as e:
        log.error(f"Export crash for {clip_id}: {e}")
        return None

def export_all(highlights, video_path: str) -> List[Path]:
    """
    Orchestrates the export of all clips in parallel.
    Uses low worker count to avoid OOM or CPU starvation during heavy filtering.
    """
    if isinstance(highlights, (str, Path)):
        import yaml
        with open(highlights, "r") as f:
            highlights = yaml.safe_load(f) or {}

    log.info("🚀 Starting Cinema-Grade Export Phase...")
    
    out_dir = Path(cfg["paths"]["shorts"]) / time.strftime("%Y-%m-%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    encoder = _get_best_encoder()
    # If using GPU, we can afford more workers? Actually, filters are CPU bound.
    # Stick to 1 or 2 workers to keep the UI responsive and avoid FFmpeg thread-locks.
    max_workers = 1 if encoder != "libx264" else 2 
    
    exported_clips = []
    
    # Sort highlights by start time
    items = sorted(highlights.items(), key=lambda x: x[1].get("start", 0))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for clip_id, info in items:
            start = info.get("start", info.get("start_sec", 0))
            end = info.get("end", info.get("end_sec", 0))
            out_file = out_dir / f"{clip_id}.mp4"
            
            futures.append(executor.submit(
                export_clip, video_path, start, end, str(out_file), clip_id
            ))
            
        for future in futures:
            path = future.result()
            if path:
                exported_clips.append(Path(path))
                
    log.info(f"✨ Export Phase Complete: {len(exported_clips)} clips ready in {out_dir}")
    return exported_clips

if __name__ == "__main__":
    # Test stub
    pass
