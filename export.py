import subprocess
import json
import os
import shlex
import shutil
import time
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

from utils.config import load_config
from utils.logger import get_logger
from frame_analyzer import analyze_clip
from utils.subtitles import generate_ass_subtitles

cfg = load_config()
log = get_logger("export", cfg["logging"]["log_file"], cfg["logging"]["level"])

# Cache for best encoder to avoid re-testing
_BEST_ENCODER = None
_BEST_ENCODER_LOCK = Lock()
MIN_OUTPUT_BYTES = 5_000
SAFE_LIGHTING_FILTERS = ("eq=", "curves=", "hue=", "unsharp=", "hqdn3d=")

def _compact_text(text: str, max_lines: int = 20) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-max_lines:])

def _normalize_speed(speed: float) -> float:
    """Clamp speed to a practical range so filter loops stay bounded."""
    try:
        speed = float(speed)
    except (TypeError, ValueError):
        speed = 1.0

    if speed <= 0:
        speed = 1.0

    return max(0.25, min(speed, 4.0))

def _sanitize_lighting_filter(value) -> str:
    if not isinstance(value, str):
        return ""

    candidate = value.strip()
    if not candidate or ";" in candidate or "[" in candidate or "]" in candidate or "\n" in candidate:
        return ""
    if not candidate.startswith(SAFE_LIGHTING_FILTERS):
        return ""
    return candidate

def _sanitize_strategy(raw_strategy) -> Dict:
    """Normalize analyzer output before it touches FFmpeg command construction."""
    if not isinstance(raw_strategy, dict):
        raw_strategy = {}

    active_crop = raw_strategy.get("active_crop")
    if isinstance(active_crop, dict):
        try:
            active_crop = {
                "x": max(0, int(active_crop.get("x", 0))),
                "y": max(0, int(active_crop.get("y", 0))),
                "width": max(2, int(active_crop.get("width", 0))),
                "height": max(2, int(active_crop.get("height", 0))),
            }
        except (TypeError, ValueError):
            active_crop = None
    else:
        active_crop = None

    return {
        "use_solo_frame": bool(raw_strategy.get("use_solo_frame", False)),
        "has_black_panel": bool(raw_strategy.get("has_black_panel", False)),
        "black_panel_side": raw_strategy.get("black_panel_side"),
        "active_crop": active_crop,
        "speed_factor": _normalize_speed(raw_strategy.get("speed_factor", 1.0)),
        "apply_lighting_fix": bool(raw_strategy.get("apply_lighting_fix", False)),
        "lighting_filter": _sanitize_lighting_filter(raw_strategy.get("lighting_filter", "")),
        "skip_silence": bool(raw_strategy.get("skip_silence", False)),
        "active_segments": raw_strategy.get("active_segments", []) if isinstance(raw_strategy.get("active_segments"), list) else [],
        "should_drop": bool(raw_strategy.get("should_drop", False)),
        "is_multi_active_frame": bool(raw_strategy.get("is_multi_active_frame", False)),
    }

def _parse_fps(rate: str) -> float:
    """Parse ffprobe frame-rate strings like 30000/1001 without eval or div-by-zero."""
    if not rate:
        return 30.0

    try:
        if "/" in rate:
            num, den = rate.split("/", 1)
            numerator = float(num)
            denominator = float(den)
            if denominator == 0:
                raise ValueError("FPS denominator is zero")
            fps = numerator / denominator
        else:
            fps = float(rate)
    except (TypeError, ValueError):
        return 30.0

    return fps if fps > 0 else 30.0

def _get_best_encoder() -> str:
    """
    Detects and verifies the most powerful hardware encoder available.
    Includes a 'smoke test' to ensure the encoder actually works in the current environment.
    """
    global _BEST_ENCODER
    with _BEST_ENCODER_LOCK:
        if _BEST_ENCODER:
            return _BEST_ENCODER

        # Priority list of hardware encoders
        # videotoolbox (Mac), nvenc (NVIDIA), qsv (Intel), vaapi (Linux)
        candidates =["h264_nvenc", "h264_videotoolbox", "h264_qsv", "h264_vaapi"]
        
        try:
            result = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=15)
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0 or not result.stdout.strip():
            raise ValueError(result.stderr.strip() or "ffprobe returned no video info")

        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        if not streams:
            raise ValueError("No video stream found")

        stream = streams[0]
        fps = _parse_fps(stream.get("r_frame_rate", "30/1"))
        
        return {
            "width": int(stream.get("width") or 1920),
            "height": int(stream.get("height") or 1080),
            "fps": fps
        }
    except Exception as e:
        log.error(f"Failed to get video info: {e}")
        return {"width": 1920, "height": 1080, "fps": 30.0}

def _has_audio_stream(path: str) -> bool:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_type",
        "-of", "json",
        path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0 or not result.stdout.strip():
            return False
        data = json.loads(result.stdout)
        return bool(data.get("streams"))
    except Exception as e:
        log.debug("Audio stream probe failed for %s: %s", path, e)
        return False

def _check_free_space(output_path: str, clip_duration: float) -> bool:
    output_dir = Path(output_path).parent
    usage = shutil.disk_usage(output_dir if output_dir.exists() else Path("."))
    bitrate = str(cfg["export"].get("video_bitrate", "8M")).strip().upper()
    multiplier = 1_000_000 if bitrate.endswith("M") else 1_000
    try:
        bits_per_second = float(bitrate[:-1]) * multiplier if bitrate[-1] in {"M", "K"} else float(bitrate)
    except (ValueError, IndexError):
        bits_per_second = 8_000_000
    estimated_bytes = max(MIN_OUTPUT_BYTES, int((bits_per_second / 8) * max(clip_duration, 1) * 2.5))
    if usage.free < estimated_bytes:
        log.error("Not enough disk space for export: free=%.1f MB need≈%.1f MB", usage.free / 1_048_576, estimated_bytes / 1_048_576)
        return False
    return True

def _build_audio_filter(
    speed: float,
    trim_start: float = 0.0,
    trim_duration: Optional[float] = None,
    output_duration: Optional[float] = None,
) -> str:
    """Build an audio filter chain that keeps atempo inside FFmpeg's 0.5-2.0 range."""
    speed = _normalize_speed(speed)

    filters = []
    if trim_duration is not None:
        filters.append(f"atrim=start={trim_start:.6f}:duration={trim_duration:.6f}")
        filters.append("asetpts=PTS-STARTPTS")

    filters.append("aresample=44100")

    tempo = speed
    while tempo < 0.5:
        filters.append("atempo=0.500000")
        tempo *= 2.0
    while tempo > 2.0:
        filters.append("atempo=2.000000")
        tempo /= 2.0

    filters.append(f"atempo={tempo:.6f}")
    filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")
    chain = ",".join(filters)
    if output_duration is not None:
        chain = _add_audio_fades(chain, output_duration)
    return chain

def _get_fade_durations(output_duration: float) -> Tuple[float, float]:
    """Return safe in/out fade durations for the post-speed output duration."""
    transitions = cfg["export"].get("transitions", {})
    try:
        fade_in = float(transitions.get("fade_in_duration", 0.0))
    except (TypeError, ValueError):
        fade_in = 0.0
    try:
        fade_out = float(transitions.get("fade_out_duration", 0.0))
    except (TypeError, ValueError):
        fade_out = 0.0

    max_each = max(0.0, output_duration / 3.0)
    return max(0.0, min(fade_in, max_each)), max(0.0, min(fade_out, max_each))

def _add_audio_fades(filter_chain: str, output_duration: float) -> str:
    transitions = cfg["export"].get("transitions", {})
    try:
        fade_in = float(transitions.get("audio_fade_in", 0.0))
    except (TypeError, ValueError):
        fade_in = 0.0
    try:
        fade_out = float(transitions.get("audio_fade_out", 0.0))
    except (TypeError, ValueError):
        fade_out = 0.0

    fade_in = max(0.0, min(fade_in, output_duration / 3.0))
    fade_out = max(0.0, min(fade_out, output_duration / 3.0))
    if fade_in > 0:
        filter_chain += f",afade=t=in:st=0:d={fade_in:.6f}"
    if fade_out > 0:
        filter_chain += f",afade=t=out:st={max(0.0, output_duration - fade_out):.6f}:d={fade_out:.6f}"
    return filter_chain

def _load_transcript_segments(transcript_path: Optional[str]) -> Optional[List[Dict]]:
    if not transcript_path:
        return None

    path = Path(transcript_path)
    if not path.exists():
        log.warning("Transcript not found for export analysis: %s", path)
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log.warning("Could not load transcript %s: %s", path, e)
        return None

    if not isinstance(data, list):
        log.warning("Transcript has invalid format: %s", path)
        return None

    segments = []
    for seg in data:
        if not isinstance(seg, dict):
            continue
        try:
            start = float(seg["start"])
            end = float(seg["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            segments.append({"start": start, "end": end, "text": str(seg.get("text", ""))})

    return segments or None

def _is_readable_media_input(path: Path) -> bool:
    """Return True for readable image/video inputs FFmpeg can use as overlays."""
    if not path.exists() or path.stat().st_size == 0:
        return False

    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        return True

    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_type",
        "-of", "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0 or not result.stdout.strip():
            return False
        data = json.loads(result.stdout)
        return bool(data.get("streams"))
    except Exception as e:
        log.debug("Video stream probe failed for %s: %s", path, e)
        return False

def _validate_output(path: str) -> bool:
    output = Path(path)
    min_bytes = int(cfg["export"].get("min_output_bytes", MIN_OUTPUT_BYTES))
    if not output.exists():
        log.error("Output file missing: %s", output)
        return False
    if output.stat().st_size < min_bytes:
        log.error("Output file too small: %s (%d bytes)", output, output.stat().st_size)
        return False
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(output),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0 or not result.stdout.strip():
            log.error("Output file failed ffprobe validation: %s", output)
            return False
        data = json.loads(result.stdout)
        duration = float(data.get("format", {}).get("duration") or 0)
        if duration <= 0:
            log.error("Output file has invalid duration: %s", output)
            return False
    except Exception as e:
        log.error("Output validation failed for %s: %s", output, e)
        return False
    return True

def _ffmpeg_timeout() -> int:
    try:
        return int(cfg["export"].get("ffmpeg_timeout_seconds", 900))
    except (TypeError, ValueError):
        return 900

def _libx264_fallback_cmd(cmd: List[str]) -> List[str]:
    fallback = list(cmd)
    if "-c:v" in fallback:
        fallback[fallback.index("-c:v") + 1] = "libx264"

    remove_with_value = {"-preset", "-tune", "-rc", "-cq", "-spatial-aq", "-b_ref_mode"}
    cleaned = []
    skip_next = False
    for item in fallback:
        if skip_next:
            skip_next = False
            continue
        if item in remove_with_value:
            skip_next = True
            continue
        cleaned.append(item)

    insert_at = cleaned.index("-movflags") if "-movflags" in cleaned else max(0, len(cleaned) - 1)
    cleaned[insert_at:insert_at] = ["-preset", "veryfast", "-crf", str(cfg["export"].get("crf", 23))]
    return cleaned

def _without_logo_cmd(cmd: List[str], logo_file: Path, filter_complex: str) -> List[str]:
    no_logo = list(cmd)
    logo_input = str(logo_file)
    for index in range(len(no_logo) - 1):
        if no_logo[index] == "-i" and no_logo[index + 1] == logo_input:
            del no_logo[index:index + 2]
            break

    if "-filter_complex" in no_logo:
        no_logo[no_logo.index("-filter_complex") + 1] = filter_complex

    return no_logo

def _run_ffmpeg_with_retry(cmd: List[str], output_path: str, clip_id: str, attempts: int = 2) -> Tuple[bool, str]:
    """Run FFmpeg, retry once on failure, and reject missing/tiny output files."""
    last_error = ""
    output = Path(output_path)
    timeout = _ffmpeg_timeout()

    for attempt in range(1, attempts + 1):
        if output.exists():
            try:
                output.unlink()
            except OSError as e:
                log.warning("[%s] Could not remove stale output %s: %s", clip_id, output, e)

        log.debug("[%s] FFmpeg command: %s", clip_id, shlex.join(cmd))
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            last_error = f"FFmpeg timed out after {timeout}s"
            log.warning("[%s] FFmpeg attempt %d/%d failed: %s", clip_id, attempt, attempts, last_error)
            continue

        if res.returncode == 0 and _validate_output(output_path):
            return True, ""

        last_error = _compact_text(res.stderr) or f"FFmpeg exited {res.returncode}"
        log.warning("[%s] FFmpeg attempt %d/%d failed: %s", clip_id, attempt, attempts, last_error)

    return False, last_error

def _build_enhance_stack(
    analysis: Dict,
    source_fps: float = 30.0,
    use_logo: Optional[bool] = None,
    output_duration: Optional[float] = None,
    subtitles_path: Optional[str] = None,
) -> str:
    """
    Builds the filtergraph based on Frame Analysis decisions.
    Supports SOLO mode (full-screen single panel), SPLIT mode (both panels), and BLACK PANEL handling.
    
    CRITICAL FIX FOR GUEST CAMERA OFF:
    - If right panel is black (guest camera off), crop to LEFT half only, then scale to 9:16
    - This prevents awkward center crop that cuts both panels in half
    """
    strategy = analysis.get("export_strategy", {}) if isinstance(analysis, dict) else {}
    strategy = _sanitize_strategy(strategy)
    use_solo = strategy.get("use_solo_frame", False)
    has_black_panel = strategy.get("has_black_panel", False)
    black_panel_side = strategy.get("black_panel_side")
    active_crop = strategy.get("active_crop")
    
    target_w = int(cfg["export"]["width"])   # 1080
    target_h = int(cfg["export"]["height"])  # 1920
    
    # 1. Base Layer Construction with Enhancement Stack
    # Apply quality enhancements BEFORE cropping/scaling for best results
    enhancement_chain = "hqdn3d=4:3:6:4.5,deband=1thr=0.02:2thr=0.02:range=16:blur=1,unsharp=5:5:1.0:5:5:0.0"
    
    if has_black_panel:
        # CRITICAL: Guest camera is OFF - crop to active panel only
        log.debug("BLACK PANEL detected - cropping to active panel only")
        
        if black_panel_side == "right":
            # Right panel is black, crop to LEFT half (your camera)
            filter_base = (
                f"{enhancement_chain},"
                f"crop=iw/2:ih:0:0,"
                f"scale={target_w}:{target_h}:flags=lanczos:force_original_aspect_ratio=increase,"
                f"crop={target_w}:{target_h}"
            )
        else:
            # Left panel is black (rare), crop to RIGHT half
            filter_base = (
                f"{enhancement_chain},"
                f"crop=iw/2:ih:iw/2:0,"
                f"scale={target_w}:{target_h}:flags=lanczos:force_original_aspect_ratio=increase,"
                f"crop={target_w}:{target_h}"
            )
    elif use_solo:
        log.debug("SOLO frame mode (full-screen single panel)")
        if active_crop:
            try:
                crop_x = max(0, int(active_crop.get("x", 0)))
                crop_y = max(0, int(active_crop.get("y", 0)))
                crop_w = max(2, int(active_crop.get("width", target_w)))
                crop_h = max(2, int(active_crop.get("height", target_h)))
                filter_base = (
                    f"{enhancement_chain},"
                    f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},"
                    f"scale={target_w}:{target_h}:flags=lanczos:force_original_aspect_ratio=increase,"
                    f"crop={target_w}:{target_h}"
                )
            except (TypeError, ValueError):
                filter_base = (
                    f"{enhancement_chain},"
                    f"crop='trunc(ih*9/16)':ih,"
                    f"scale={target_w}:{target_h}:flags=lanczos:force_original_aspect_ratio=increase,"
                    f"crop={target_w}:{target_h}"
                )
        else:
            # Solo Mode: Take the center 9:16 slice with Lanczos scaling
            filter_base = (
                f"{enhancement_chain},"
                f"crop='trunc(ih*9/16)':ih,"
                f"scale={target_w}:{target_h}:flags=lanczos:force_original_aspect_ratio=increase,"
                f"crop={target_w}:{target_h}"
            )
    else:
        log.debug("STACK mode (blurred background + sharp center)")
        # Stack Mode with enhancements on foreground layer
        filter_base = (
            f"{enhancement_chain},split=2[bg][fg];"
            f"[bg]scale={target_w}:{target_h}:flags=lanczos:force_original_aspect_ratio=increase,crop={target_w}:{target_h},boxblur=20:10[bg_fin];"
            f"[fg]scale={target_w}:-1:flags=lanczos,crop={target_w}:min(ih\\,{target_h})[fg_scaled];"
            f"[bg_fin][fg_scaled]overlay=(W-w)/2:(H-h)/2"
        )

    # 2. Add Lighting/Color Fixes if recommended
    if strategy.get("apply_lighting_fix"):
        l_filter = _sanitize_lighting_filter(strategy.get("lighting_filter", ""))
        if l_filter:
            filter_base += f",{l_filter}"

    # 3. Add Motion Interpolation for 60fps look
    try:
        target_fps = float(cfg["export"].get("fps", 60))
    except (TypeError, ValueError):
        target_fps = 60.0
    if target_fps > source_fps + 5:
        # T4 OPTIMIZATION: Do not use `minterpolate` as it chokes the CPU.
        # Use `framerate` for standard blending.
        filter_base += f",framerate=fps={target_fps:.6f}"

    if output_duration:
        fade_in, fade_out = _get_fade_durations(output_duration)
        if fade_in > 0:
            filter_base += f",fade=t=in:st=0:d={fade_in:.6f}"
        if fade_out > 0:
            filter_base += f",fade=t=out:st={max(0.0, output_duration - fade_out):.6f}:d={fade_out:.6f}"

    # 4. Add Branding / Logo overlay
    logo_path = cfg["thumbnail"].get("template_path", "channel_logo.png")
    logo_enabled = Path(logo_path).exists() if use_logo is None else use_logo
    
    is_graph = ";" in filter_base
    scale_format = "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p"
    if subtitles_path:
        # Escape colons and backslashes for FFmpeg filter syntax
        safe_path = str(subtitles_path).replace("\\", "/").replace(":", "\\:")
        scale_format += f",subtitles='{safe_path}'"

    if logo_enabled:
        # Position top-right with margin
        return (
            f"[v_src]{filter_base}[v_tmp];"
            f"[1:v]scale=120:-1[logo];"
            f"[v_tmp][logo]overlay=W-w-40:40,{scale_format}[v_out]"
        )
    else:
        if is_graph:
            return f"[v_src]{filter_base}[v_tmp];[v_tmp]{scale_format}[v_out]"
        else:
            return f"[v_src]{filter_base},{scale_format}[v_out]"

def export_clip(
    video_path: str,
    start: float,
    end: float,
    output_path: str,
    clip_id: str = "clip",
    transcript_segments: Optional[List[Dict]] = None,
    analysis: Optional[Dict] = None,
) -> Optional[str]:
    """
    Exports a single clip using Hybrid CPU/GPU pipeline.
    CPU handles decode/filtergraph, GPU handles encoding when NVENC is available.
    """
    t_start = time.perf_counter()

    try:
        start = float(start)
        end = float(end)
    except (TypeError, ValueError):
        log.error("[%s] Invalid clip timestamps: start=%r end=%r", clip_id, start, end)
        return None

    if end <= start:
        log.error("[%s] Invalid clip range: start=%.3f end=%.3f", clip_id, start, end)
        return None
    
    # 1. Run/Use Intelligence Analysis
    if analysis is None:
        analysis = analyze_clip(
            video_path,
            start,
            end,
            transcript_segments=transcript_segments,
            clip_id=clip_id,
        )
    
    if not isinstance(analysis, dict) or not isinstance(analysis.get("export_strategy"), dict):
        log.error("[%s] Invalid analysis result; skipping export.", clip_id)
        return None

    strategy = _sanitize_strategy(analysis["export_strategy"])
    analysis["export_strategy"] = strategy
    
    # CRITICAL: Drop segments with multiple active frames (both host + guest cameras on)
    if strategy.get("should_drop", False):
        log.info("[%s] DROPPING segment - multiple active frames detected (host + guest)", clip_id)
        return None
    
    # 2. Get hardware info
    info = _get_video_info(video_path)
    encoder = _get_best_encoder()

    logo_file = Path(cfg["thumbnail"].get("template_path", "channel_logo.png"))
    use_logo = logo_file.exists()
    if use_logo and not _is_readable_media_input(logo_file):
        log.warning("[%s] Logo file is not readable as image/video input; skipping: %s", clip_id, logo_file)
        use_logo = False
    has_audio = _has_audio_stream(video_path)
    if not has_audio:
        log.warning("[%s] No audio stream found; exporting silent video.", clip_id)
    
    # 3. Build speed/audio/video decisions
    global_speed = _normalize_speed(cfg["export"].get("global_speed_factor", 1.0))
    analysis_speed = _normalize_speed(strategy.get("speed_factor", 1.0))
    speed = _normalize_speed(global_speed * analysis_speed)

    try:
        pre_seek = float(cfg["export"].get("pre_seek", 4.0))
    except (TypeError, ValueError):
        pre_seek = 4.0
    pre_seek = max(0.0, pre_seek)
    input_seek = max(0.0, start - pre_seek)
    trim_start = start - input_seek
    clip_duration = max(0.0, end - start)
    output_duration = clip_duration / speed
    input_duration = trim_start + clip_duration
    if not _check_free_space(output_path, output_duration):
        return None

    a_filter = (
        _build_audio_filter(
            speed,
            trim_start=trim_start,
            trim_duration=clip_duration,
            output_duration=output_duration,
        )
        if has_audio else None
    )

    # 3.5 Generate Subtitles (ASS)
    output_file = Path(output_path)
    ass_path = str(output_file.with_suffix('.ass'))
    if transcript_segments:
        log.info("[%s] Generating dynamic subtitles...", clip_id)
        if generate_ass_subtitles(transcript_segments, ass_path, start, end):
            pass
        else:
            ass_path = None
    else:
        ass_path = None

    # 4. Build filter chain
    v_filter = _build_enhance_stack(
        analysis,
        source_fps=info["fps"],
        use_logo=use_logo,
        output_duration=output_duration,
        subtitles_path=ass_path,
    )
    video_filter_complex = (
        f"[0:v]trim=start={trim_start:.6f}:duration={clip_duration:.6f},"
        f"setpts=(PTS-STARTPTS)/{speed:.6f}[v_src];{v_filter}"
    )
    
    # 5. Execute FFmpeg
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        log.warning("[%s] Output already exists and will be overwritten: %s", clip_id, output_file)

    cmd = ["ffmpeg", "-y"]
    
    cmd.extend([
        "-threads", "0",
        "-ss", f"{input_seek:.6f}",
        "-t", f"{input_duration:.6f}",
        "-i", video_path
    ])
    
    if use_logo:
        cmd.extend(["-i", str(logo_file)])
        
    cmd.extend([
        "-filter_complex", video_filter_complex,
        "-map", "[v_out]",
        "-c:v", encoder,
        "-b:v", str(cfg["export"].get("video_bitrate", "8M")),
        "-maxrate", "15M",
        "-bufsize", "24M",
    ])

    if has_audio:
        cmd.extend([
            "-af", a_filter,
            "-map", "0:a:0?",
            "-c:a", "aac",
            "-b:a", str(cfg["export"].get("audio_bitrate", "192k")),
            "-shortest",
        ])
    
    # T4 OPTIMIZATION: Turing-specific NVENC flags
    if encoder == "h264_nvenc":
        cmd.extend([
            "-preset", "p4",           
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
    
    log.info("[%s] 🎬 Exporting %.1fs segment (speed=%.2fx, encoder=%s) ...", clip_id, end-start, speed, encoder)
    
    try:
        success, error = _run_ffmpeg_with_retry(cmd, output_path, clip_id)
        if not success:
            fallback_source_cmd = cmd
            if use_logo:
                log.warning("[%s] Export failed with logo; retrying without logo.", clip_id)
                no_logo_filter = _build_enhance_stack(
                    analysis,
                    source_fps=info["fps"],
                    use_logo=False,
                    output_duration=output_duration,
                    subtitles_path=ass_path,
                )
                no_logo_complex = (
                    f"[0:v]trim=start={trim_start:.6f}:duration={clip_duration:.6f},"
                    f"setpts=(PTS-STARTPTS)/{speed:.6f}[v_src];{no_logo_filter}"
                )
                fallback_source_cmd = _without_logo_cmd(cmd, logo_file, no_logo_complex)
                success, error = _run_ffmpeg_with_retry(fallback_source_cmd, output_path, clip_id, attempts=1)
                if success:
                    dur = max(time.perf_counter() - t_start, 0.001)
                    log.info("✅ [%s] Export complete in %.1fs (%.1fx real-time, logo skipped)", clip_id, dur, output_duration/dur)
                    return output_path

            if encoder == "h264_nvenc":
                log.warning("[%s] NVENC failed; retrying with libx264 fallback.", clip_id)
                fallback_cmd = _libx264_fallback_cmd(fallback_source_cmd)
                success, error = _run_ffmpeg_with_retry(fallback_cmd, output_path, clip_id, attempts=1)
                if success:
                    dur = max(time.perf_counter() - t_start, 0.001)
                    log.info("✅ [%s] Export complete in %.1fs (%.1fx real-time, encoder=libx264 fallback)", clip_id, dur, output_duration/dur)
                    return output_path
            log.error("[%s] FFmpeg failed: %s", clip_id, error)
            return None
        
        dur = max(time.perf_counter() - t_start, 0.001)
        log.info("✅ [%s] Export complete in %.1fs (%.1fx real-time)", clip_id, dur, output_duration/dur)
        return output_path
        
    except Exception as e:
        log.error("[%s] Export crash: %s", clip_id, e)
        return None

def _parse_time_to_seconds(time_val) -> float:
    """Helper to convert HH:MM:SS, MM:SS, or numeric time into float seconds."""
    if isinstance(time_val, (int, float)):
        return float(time_val)
    if isinstance(time_val, str):
        try:
            parts = time_val.split(":")
            secs = 0.0
            for p in parts:
                secs = secs * 60 + float(p)
            return secs
        except ValueError:
            log.warning("Invalid time value: %r", time_val)
    return 0.0

def export_all(highlights, video_path: str, transcript_path: Optional[str] = None) -> List[Path]:
    """
    Orchestrates the export of all clips in parallel.
    """
    if isinstance(highlights, (str, Path)):
        import yaml
        with open(highlights, "r") as f:
            highlights = yaml.safe_load(f) or {}

    if not isinstance(highlights, dict):
        log.error("Invalid highlights data: expected mapping, got %s", type(highlights).__name__)
        return []

    if transcript_path is None:
        candidate = Path(cfg["paths"]["transcripts"]) / f"{Path(video_path).stem}.json"
        transcript_path = str(candidate) if candidate.exists() else None
    transcript_segments = _load_transcript_segments(transcript_path)

    log.info("🚀 Starting Cinema-Grade Export Phase...")
    
    out_dir = Path(cfg["paths"]["shorts"]) / time.strftime("%Y-%m-%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    exported_clips =[]
    
    # Sort highlights by start time safely handling both strings and numbers.
    items = []
    for clip_id, info in highlights.items():
        if not isinstance(info, dict):
            log.warning("Skipping malformed highlight %s: expected mapping", clip_id)
            continue

        start = _parse_time_to_seconds(info.get("start", info.get("start_sec", 0.0)))
        end = _parse_time_to_seconds(info.get("end", info.get("end_sec", 0.0)))
        if end <= start:
            log.warning("Skipping invalid highlight %s: start=%.3f end=%.3f", clip_id, start, end)
            continue

        items.append((clip_id, start, end))

    items.sort(key=lambda x: x[1])
    
    # FFmpeg already uses all available threads; running clips serially avoids CPU oversubscription.
    for clip_id, start, end in items:
        out_file = out_dir / f"{clip_id}.mp4"

        path = export_clip(
            video_path,
            start,
            end,
            str(out_file),
            clip_id,
            transcript_segments=transcript_segments,
        )
        if path:
            exported_clips.append(Path(path))
                
    log.info(f"✨ Export Phase Complete: {len(exported_clips)} clips ready in {out_dir}")
    return exported_clips

if __name__ == "__main__":
    # Test stub
    pass
