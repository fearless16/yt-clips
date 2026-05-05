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
    candidates =["h264_nvenc", "h264_videotoolbox", "h264_qsv", "h264_vaapi"]
    
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
    
    cmd = ["ffmpeg", "-y"]
    
    # Add CUDA hardware acceleration for the test if testing NVIDIA
    if encoder == "h264_nvenc":
        cmd.extend(["-hwaccel", "cuda"])
        
    # FIX: NVIDIA T4 GPUs crash if the video is smaller than 145x145. 
    # Using 1280x720 safely mimics a real-world export scenario.
    cmd.extend([
        "-f", "lavfi", "-i", "color=c=black:s=1280x720:d=0.5", 
        "-vf", "format=yuv420p",
        "-c:v", encoder
    ])
    
    if encoder == "h264_nvenc":
        cmd.extend(["-preset", "p4"])
        
    cmd.extend(["-t", "0.5", temp_test])
    
    try:
        # Give it stderr output checking so we can debug if it fails
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        success = res.returncode == 0 and Path(temp_test).exists()
        
        if not success:
            log.warning(f"Smoke test failed for {encoder}. Error: {res.stderr}")
            
        if Path(temp_test).exists():
            Path(temp_test).unlink()
            
        return success
    except Exception as e:
        log.warning(f"Smoke test crashed: {e}")
        return False

def _get_video_info(path: str) -> Dict:
    """Extract width, height, and FPS using ffprobe."""
    cmd =[
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

    # 3. Add Motion Interpolation for 60fps look
    target_fps = cfg["export"].get("fps", 60)
    if target_fps > source_fps + 5:
        # T4 OPTIMIZATION: Do not use `minterpolate` as it chokes the CPU.
        # Use `framerate` for standard blending.
        filter_base += f",framerate=fps={target_fps}:interp_start=128:interp_end=128:scene=100"

    # 4. Add Branding / Logo overlay
    logo_path = cfg["thumbnail"].get("template_path", "channel_logo.png")
    if Path(logo_path).exists():
        # Position top-right with margin
        filter_base = f"[v_src]{filter_base}[v_final];[1:v]scale=120:-1[logo];[v_final][logo]overlay=W-w-40:40,format=yuv420p"
        return filter_base
    
    return f"[v_src]{filter_base},format=yuv420p"

def export_clip(video_path: str, start: float, end: float, output_path: str, clip_id: str = "clip") -> Optional[str]:
    """
    Exports a single clip using Hybrid CPU/GPU pipeline.
    CPU handles the heavy filtergraph, GPU handles Decoding and Encoding.
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
    a_filter = f"aresample=44100,atempo={speed},loudnorm=I=-16:TP=-1.5:LRA=11"
    
    # 5. Execute FFmpeg
    logo_path = cfg["thumbnail"].get("template_path", "channel_logo.png")
    
    cmd = ["ffmpeg", "-y"]
    
    # T4 OPTIMIZATION: Enable CUDA Hardware Decoding.
    if encoder == "h264_nvenc":
        cmd.extend(["-hwaccel", "cuda"])

    cmd.extend([
        "-threads", "0",
        "-ss", str(start),
        "-t", str(end - start),  # Read exact duration
        "-i", video_path
    ])
    
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
    
    # T4 OPTIMIZATION: Turing-specific NVENC flags
    if encoder == "h264_nvenc":
        cmd.extend([
            "-preset", "p6",           
            "-tune", "hq",             
            "-rc", "vbr",              
            "-cq", "26",               
            "-spatial-aq", "1",        
            "-b_ref_mode", "middle"    
        ])
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
        output_duration = (end - start) / speed
        log.info(f"✅ [{clip_id}] Export complete in {dur:.1f}s ({output_duration/dur:.1f}x real-time)")
        return output_path
        
    except Exception as e:
        log.error(f"Export crash for {clip_id}: {e}")
        return None

def _parse_time_to_seconds(time_val) -> float:
    """Helper to convert HH:MM:SS, MM:SS, or numeric time into float seconds."""
    if isinstance(time_val, (int, float)):
        return float(time_val)
    if isinstance(time_val, str):
        parts = time_val.split(":")
        secs = 0.0
        for p in parts:
            secs = secs * 60 + float(p)
        return secs
    return 0.0

def export_all(highlights, video_path: str) -> List[Path]:
    """
    Orchestrates the export of all clips in parallel.
    """
    if isinstance(highlights, (str, Path)):
        import yaml
        with open(highlights, "r") as f:
            highlights = yaml.safe_load(f) or {}

    log.info("🚀 Starting Cinema-Grade Export Phase...")
    
    out_dir = Path(cfg["paths"]["shorts"]) / time.strftime("%Y-%m-%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # FFmpeg multi-threading runs optimally without Python fighting it for resources. 
    # Hardcoding to 1 worker prevents Thread-lock/OOM issues on instances like Colab.
    max_workers = 1 
    
    exported_clips =[]
    
    # Sort highlights by start time safely handling both strings and numbers
    items = sorted(
        highlights.items(), 
        key=lambda x: _parse_time_to_seconds(x[1].get("start", x[1].get("start_sec", 0.0)))
    )
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures =[]
        for clip_id, info in items:
            # Safely parse time formats like '00:11:19' or 679.0 into float seconds
            start = _parse_time_to_seconds(info.get("start", info.get("start_sec", 0.0)))
            end = _parse_time_to_seconds(info.get("end", info.get("end_sec", 0.0)))
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