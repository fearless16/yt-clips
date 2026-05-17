import subprocess
import json
import os
import shlex
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

from utils.config import load_config
from utils.logger import get_logger
from frame_analyzer import analyze_clip as cheap_analyze_clip

cfg = load_config()
log = get_logger("export", cfg["logging"]["log_file"], cfg["logging"]["level"])
premium_enabled = cfg.get("premium", {}).get("enabled", False)
super_res_enabled = cfg.get("export", {}).get("super_resolution", False)

if premium_enabled:
    try:
        from premium_analyzer import PremiumAnalyzer
        _premium_analyzer = PremiumAnalyzer()
        analyze_clip = _premium_analyzer.analyze_clip
        log.info("Premium analyzer ACTIVE — YOLOv8-face + ByteTrack")
    except ImportError as e:
        analyze_clip = cheap_analyze_clip
        log.warning("Premium import failed (%s) — using cheap analyzer", e)
else:
    analyze_clip = cheap_analyze_clip

_BEST_ENCODER = None
_BEST_ENCODER_LOCK = Lock()
MIN_OUTPUT_BYTES = 5_000
SAFE_LIGHTING_FILTERS = ("eq=", "curves=", "hue=", "unsharp=", "hqdn3d=")


def _compact_text(text: str, max_lines: int = 20) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-max_lines:])


def _normalize_speed(speed: float) -> float:
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
        "use_vertical_stack": bool(raw_strategy.get("use_vertical_stack", False)),
        "has_black_panel": bool(raw_strategy.get("has_black_panel", False)),
        "black_panel_side": raw_strategy.get("black_panel_side"),
        "guest_cam_on": bool(raw_strategy.get("guest_cam_on", False)),
        "guest_cam_off": bool(raw_strategy.get("guest_cam_off", False)),
        "is_screen_share": bool(raw_strategy.get("is_screen_share", False)),
        "active_crop": active_crop,
        "speed_factor": _normalize_speed(raw_strategy.get("speed_factor", 1.0)),
        "apply_lighting_fix": bool(raw_strategy.get("apply_lighting_fix", False)),
        "lighting_filter": _sanitize_lighting_filter(raw_strategy.get("lighting_filter", "")),
        "skip_silence": bool(raw_strategy.get("skip_silence", False)),
        "active_segments": raw_strategy.get("active_segments", []) if isinstance(raw_strategy.get("active_segments"), list) else [],
        "should_drop": bool(raw_strategy.get("should_drop", False)),
    }


def _parse_fps(rate: str) -> float:
    if not rate:
        return 30.0
    try:
        if "/" in rate:
            num, den = rate.split("/", 1)
            fps = float(num) / float(den)
        else:
            fps = float(rate)
    except (TypeError, ValueError, ZeroDivisionError):
        return 30.0
    return fps if fps > 0 else 30.0


def _get_best_encoder() -> str:
    global _BEST_ENCODER
    with _BEST_ENCODER_LOCK:
        if _BEST_ENCODER:
            return _BEST_ENCODER
        candidates = ["h264_nvenc", "h264_videotoolbox", "h264_qsv", "h264_vaapi"]
        try:
            result = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=15)
            for enc in candidates:
                if enc in result.stdout:
                    log.info("Testing encoder: %s ...", enc)
                    if _smoke_test_encoder(enc):
                        log.info("Using encoder: %s", enc)
                        _BEST_ENCODER = enc
                        return enc
                    log.warning("%s failed smoke test, skipping", enc)
        except Exception as e:
            log.error("Encoder detection error: %s", e)
        log.info("Using software encoder: libx264")
        _BEST_ENCODER = "libx264"
        return _BEST_ENCODER


def _smoke_test_encoder(encoder: str) -> bool:
    temp_test = "temp/smoke_test.mp4"
    os.makedirs("temp", exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=1280x720:d=0.5",
        "-vf", "format=yuv420p", "-c:v", encoder,
    ]
    if encoder == "h264_nvenc":
        cmd.extend(["-preset", "p4"])
    cmd.extend(["-t", "0.5", temp_test])
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        success = res.returncode == 0 and Path(temp_test).exists()
        if not success:
            log.warning("Smoke test failed for %s: %s", encoder, res.stderr[-300:])
        if Path(temp_test).exists():
            Path(temp_test).unlink()
        return success
    except Exception as e:
        log.warning("Smoke test crashed for %s: %s", encoder, e)
        return False


def _get_video_info(path: str) -> Dict:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "json", path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        data = json.loads(result.stdout)
        stream = data.get("streams", [{}])[0]
        fps = _parse_fps(stream.get("r_frame_rate", "30/1"))
        return {"width": int(stream.get("width") or 1920), "height": int(stream.get("height") or 1080), "fps": fps}
    except Exception as e:
        log.error("Failed to get video info: %s", e)
        return {"width": 1920, "height": 1080, "fps": 30.0}


def _has_audio_stream(path: str) -> bool:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "a:0",
           "-show_entries", "stream=codec_type", "-of", "json", path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        data = json.loads(result.stdout)
        return bool(data.get("streams"))
    except Exception:
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
        log.error("Not enough disk space: free=%.1f MB need≈%.1f MB",
                  usage.free / 1_048_576, estimated_bytes / 1_048_576)
        return False
    return True


def _build_audio_filter(
    speed: float,
    trim_start: float = 0.0,
    trim_duration: Optional[float] = None,
    output_duration: Optional[float] = None,
) -> str:
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
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log.warning("Could not load transcript %s: %s", path, e)
        return None
    if not isinstance(data, list):
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
    if not path.exists() or path.stat().st_size == 0:
        return False
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        return True
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=codec_type", "-of", "json", str(path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        data = json.loads(result.stdout)
        return bool(data.get("streams"))
    except Exception:
        return False


def _validate_output(path: str, expected_duration: float = 0.0) -> bool:
    output = Path(path)
    min_bytes = int(cfg["export"].get("min_output_bytes", MIN_OUTPUT_BYTES))
    if not output.exists():
        log.error("Output file missing: %s", output)
        return False
    if output.stat().st_size < min_bytes:
        log.error("Output too small: %s (%d bytes)", output, output.stat().st_size)
        return False
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(output)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        data = json.loads(result.stdout)
        duration = float(data.get("format", {}).get("duration") or 0)
        if duration <= 0:
            log.error("Output has invalid duration: %s", output)
            return False
            
        if expected_duration > 0 and abs(duration - expected_duration) > 2.0:
            log.error("Output duration %.1fs deviates too much from expected %.1fs: %s", duration, expected_duration, output)
            return False

        if duration < 5.0:
            log.error("Output is too short (%.1fs) to be a valid Short: %s", duration, output)
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


def _run_ffmpeg_with_retry(cmd: List[str], output_path: str, clip_id: str, attempts: int = 2, expected_duration: float = 0.0) -> Tuple[bool, str]:
    last_error = ""
    output = Path(output_path)
    timeout = _ffmpeg_timeout()
    for attempt in range(1, attempts + 1):
        if output.exists():
            try:
                output.unlink()
            except OSError:
                pass
        log.debug("[%s] FFmpeg: %s", clip_id, shlex.join(cmd))
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            last_error = f"FFmpeg timed out after {timeout}s"
            log.warning("[%s] attempt %d/%d: %s", clip_id, attempt, attempts, last_error)
            continue
        if res.returncode == 0 and _validate_output(output_path, expected_duration):
            return True, ""
        last_error = _compact_text(res.stderr) or f"FFmpeg exited {res.returncode}"
        log.warning("[%s] attempt %d/%d failed: %s", clip_id, attempt, attempts, last_error)
    return False, last_error


def _build_enhance_stack(
    analysis: Dict,
    source_fps: float = 30.0,
    use_logo: Optional[bool] = None,
    output_duration: Optional[float] = None,
    native_res: bool = False,
) -> str:
    """
    Build filtergraph based on frame analysis.

    Layout routing:
      solo              → center 9:16 crop (just you in frame)
      guest_cam_on      → vertical stack, both panels stacked 9:16
      screen_share      → 9:16 canvas with blurred bg + sharp readable content
      guest_cam_off     → should have been dropped; crop active half as fallback

    When native_res=True, skip final scale to 1080x1920 — export at crop
    resolution for super-resolution processing.
    """
    strategy = analysis.get("export_strategy", {}) if isinstance(analysis, dict) else {}
    strategy = _sanitize_strategy(strategy)

    use_solo = strategy.get("use_solo_frame", False)
    use_vertical_stack = strategy.get("use_vertical_stack", False)
    guest_cam_on = strategy.get("guest_cam_on", False)
    guest_cam_off = strategy.get("guest_cam_off", False)
    has_black_panel = strategy.get("has_black_panel", False)
    black_panel_side = strategy.get("black_panel_side")
    active_crop = strategy.get("active_crop")
    is_screen_share = strategy.get("is_screen_share", False)

    # YouTube classifies Shorts by shape/duration, so every export must remain
    # on the configured vertical canvas.
    target_w = int(cfg["export"]["width"])    # 1080
    target_h = int(cfg["export"]["height"])   # 1920
    enhance = "hqdn3d=4:3:6:4.5,deband=1thr=0.02:2thr=0.02:range=16:blur=1,unsharp=5:5:1.0:5:5:0.0"

    # ── Filter selection ──────────────────────────────────────────────────────

    if guest_cam_on:
        # Guest has video: stack both halves vertically into 9:16
        # Left half → top panel, right half → bottom panel
        log.debug("GUEST CAM ON → vertical stack (both panels)")
        if native_res:
            # Split + crop both halves, stack at native resolution
            filter_base = (
                f"{enhance},split=2[left_raw][right_raw];"
                f"[left_raw]crop=iw/2:ih:0:0[top];"
                f"[right_raw]crop=iw/2:ih:iw/2:0[bot];"
                f"[top][bot]vstack=inputs=2"
            )
        else:
            half_h = target_h // 2  # 960px each
            filter_base = (
                f"{enhance},split=2[left_raw][right_raw];"
                f"[left_raw]crop=iw/2:ih:0:0,scale={target_w}:{half_h}:flags=lanczos[top];"
                f"[right_raw]crop=iw/2:ih:iw/2:0,scale={target_w}:{half_h}:flags=lanczos[bot];"
                f"[top][bot]vstack=inputs=2"
            )

    elif is_screen_share:
        # Screen share → keep the full 16:9 source visible inside a 9:16 Short.
        # A blurred cover-fill background avoids black bars while preserving the
        # foreground content instead of cropping off scorecards/charts.
        log.debug("SCREEN SHARE → 9:16 shorts canvas (blurred bg + fit foreground)")
        if native_res:
            # Export at source resolution, super-res will handle upscaling
            filter_base = f"{enhance}"
        else:
            filter_base = (
                f"{enhance},"
                f"scale={target_w}:{target_h}:flags=lanczos:force_original_aspect_ratio=increase,"
                f"crop={target_w}:{target_h}"
            )

    elif guest_cam_off:
        # Guest cam off: crop to the active (non-black) side only
        # This path should rarely be reached (should be dropped upstream),
        # but acts as a safety fallback.
        log.debug("GUEST CAM OFF fallback → crop active side (%s)", black_panel_side)
        if native_res:
            if black_panel_side == "right":
                filter_base = f"{enhance},crop=iw/2:ih:0:0"
            else:
                filter_base = f"{enhance},crop=iw/2:ih:iw/2:0"
        elif black_panel_side == "right":
            filter_base = (
                f"{enhance},"
                f"crop=iw/2:ih:0:0,"
                f"scale={target_w}:{target_h}:flags=lanczos:force_original_aspect_ratio=increase,"
                f"crop={target_w}:{target_h}"
            )
        else:
            filter_base = (
                f"{enhance},"
                f"crop=iw/2:ih:iw/2:0,"
                f"scale={target_w}:{target_h}:flags=lanczos:force_original_aspect_ratio=increase,"
                f"crop={target_w}:{target_h}"
            )

    elif use_solo and active_crop:
        # Solo with face-detected crop region
        log.debug("SOLO + face crop → precise 9:16 crop")
        try:
            cx = max(0, int(active_crop["x"]))
            cy = max(0, int(active_crop["y"]))
            cw = max(2, int(active_crop["width"]))
            ch = max(2, int(active_crop["height"]))
            if native_res:
                # Crop only — super-res handles upscaling
                filter_base = (
                    f"{enhance},split=2[bg_raw][fg_raw];"
                    f"[bg_raw]crop={cw}:{ch}:{cx}:{cy}[bg];"
                    f"[fg_raw]crop={cw}:{ch}:{cx}:{cy}[fg];"
                    f"[bg][fg]overlay=0:0"
                )
            else:
                # Fill-crop: scale up to cover frame, then crop to exact size
                filter_base = (
                    f"{enhance},"
                    f"crop={cw}:{ch}:{cx}:{cy},"
                    f"scale={target_w}:{target_h}:flags=lanczos:force_original_aspect_ratio=increase,"
                    f"crop={target_w}:{target_h}"
                )
        except (TypeError, ValueError):
            if native_res:
                filter_base = f"{enhance},crop='trunc(ih*9/16)':ih"
            else:
                filter_base = (
                    f"{enhance},"
                    f"crop='trunc(ih*9/16)':ih,"
                    f"scale={target_w}:{target_h}:flags=lanczos:force_original_aspect_ratio=increase,"
                    f"crop={target_w}:{target_h}"
                )

    else:
        # Default solo: center 9:16 crop (you only in frame)
        log.debug("SOLO center crop → 9:16")
        # ── Chat exclusion: shift crop to exclude chat overlay ─────────────────
        chat_exclusion = strategy.get("chat_exclusion")
        if chat_exclusion and chat_exclusion.get("exclude_chat"):
            chat_side = chat_exclusion.get("chat_side", "right")
            chat_w = chat_exclusion.get("chat_exclude_width", 0)

            if chat_side == "right":
                crop_filter = f"crop=iw-{chat_w}:ih:0:0"
            else:
                crop_filter = f"crop=iw-{chat_w}:ih:{chat_w}:0"

            if native_res:
                # Crop only — super-res handles upscaling
                filter_base = f"{enhance},{crop_filter}"
            elif chat_side == "right":
                filter_base = (
                    f"{enhance},"
                    f"crop=iw-{chat_w}:ih:0:0,"
                    f"scale={target_w}:{target_h}:flags=lanczos:force_original_aspect_ratio=increase,"
                    f"crop={target_w}:{target_h}"
                )
                log.debug("Chat exclusion: cropped left side to exclude right chat (%dpx)", chat_w)
            else:
                filter_base = (
                    f"{enhance},"
                    f"crop=iw-{chat_w}:ih:{chat_w}:0,"
                    f"scale={target_w}:{target_h}:flags=lanczos:force_original_aspect_ratio=increase,"
                    f"crop={target_w}:{target_h}"
                )
                log.debug("Chat exclusion: cropped right side to exclude left chat (%dpx)", chat_w)
        else:
            if native_res:
                # Just crop to 9:16 — super-res handles upscaling
                filter_base = (
                    f"{enhance},split=2[bg_raw][fg_raw];"
                    f"[bg_raw]crop='trunc(ih*9/16)':ih[bg];"
                    f"[fg_raw]crop='trunc(ih*9/16)':ih[fg];"
                    f"[bg][fg]overlay=0:0"
                )
            else:
                filter_base = (
                    f"{enhance},"
                    f"crop='trunc(ih*9/16)':ih,"
                    f"scale={target_w}:{target_h}:flags=lanczos:force_original_aspect_ratio=increase,"
                    f"crop={target_w}:{target_h}"
                )

    # ── Lighting fix ──────────────────────────────────────────────────────────
    if strategy.get("apply_lighting_fix"):
        lf = _sanitize_lighting_filter(strategy.get("lighting_filter", ""))
        if lf:
            filter_base += f",{lf}"

    # ── Color boost + sharpening ─────────────────────────────────────────────
    # Proven approach from test: sharpen + contrast + saturation
    # Reference: saturation=133, sharpness=274
    # Our tested params: sharp=1.15x, contrast=1.15x, saturation=1.15x
    filter_base += ",unsharp=5:5:1.0:5:5:0.0,eq=saturation=1.15:contrast=1.15:brightness=0.04"

    # ── Motion interpolation ──────────────────────────────────────────────────
    try:
        target_fps = float(cfg["export"].get("fps", 60))
    except (TypeError, ValueError):
        target_fps = 60.0
    if target_fps > source_fps + 5:
        filter_base += f",framerate=fps={target_fps:.6f}"

    # ── Fades ─────────────────────────────────────────────────────────────────
    if output_duration:
        fade_in, fade_out = _get_fade_durations(output_duration)
        if fade_in > 0:
            filter_base += f",fade=t=in:st=0:d={fade_in:.6f}"
        if fade_out > 0:
            filter_base += f",fade=t=out:st={max(0.0, output_duration - fade_out):.6f}:d={fade_out:.6f}"

    # ── Logo overlay ──────────────────────────────────────────────────────────
    logo_path = cfg["thumbnail"].get("template_path", "channel_logo.png")
    logo_enabled = Path(logo_path).exists() if use_logo is None else use_logo
    is_graph = ";" in filter_base
    scale_format = "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p"

    if logo_enabled:
        # Circular mask: scale to 200px, apply circular alpha, position bottom-right
        # Bottom-right: 30px from right, 280px from bottom (above YouTube UI buttons)
        return (
            f"[v_src]{filter_base}[v_tmp];"
            f"[1:v]scale=200:-1,format=rgba,"
            f"geq=lum='p(X,Y)':a='if(lt(pow(X-W/2,2)+pow(Y-H/2,2),pow(W/2,2)),255,0)'[logo];"
            f"[v_tmp][logo]overlay=W-w-30:H-h-280,{scale_format}[v_out]"
        )
    else:
        if is_graph:
            return f"[v_src]{filter_base}[v_tmp];[v_tmp]{scale_format}[v_out]"
        else:
            return f"[v_src]{filter_base},{scale_format}[v_out]"


def _export_native_res(
    video_path: str,
    input_seek: float,
    trim_start: float,
    clip_duration: float,
    speed: float,
    v_filter: str,
    clip_id: str,
) -> Optional[str]:
    """
    Export clip at native crop resolution (no 1080x1920 upscale).
    Returns path to temp file, or None on failure.
    """
    import tempfile as _tf
    temp_dir = _tf.mkdtemp()
    temp_path = str(Path(temp_dir) / f"{clip_id}_native.mp4")

    video_complex = (
        f"[0:v]trim=start={trim_start:.6f}:duration={clip_duration:.6f},"
        f"setpts=(PTS-STARTPTS)/{speed:.6f}[v_src];{v_filter}"
    )

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{input_seek:.6f}",
        "-t", f"{(trim_start + clip_duration):.6f}",
        "-i", video_path,
        "-filter_complex", video_complex,
        "-map", "[v_out]",
        "-c:v", "libx264", "-crf", "16", "-preset", "fast",
        "-pix_fmt", "yuv420p",
    ]

    # Include audio if available
    if _has_audio_stream(video_path):
        a_filter = _build_audio_filter(speed, trim_start=trim_start,
                                       trim_duration=clip_duration,
                                       output_duration=clip_duration / speed)
        if a_filter:
            cmd.extend(["-map", "0:a:0?", "-af", a_filter,
                        "-c:a", "aac", "-b:a", "192k", "-shortest"])

    cmd.append(temp_path)

    log.info("[%s] Exporting native-res for super-res...", clip_id)
    t0 = time.perf_counter()
    success, error = _run_ffmpeg_with_retry(cmd, temp_path, clip_id,
                                            expected_duration=clip_duration / speed)
    if not success:
        log.error("[%s] Native-res export failed: %s", clip_id, error)
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None
    log.info("[%s] Native-res export done in %.1fs → %s", clip_id,
             time.perf_counter() - t0, Path(temp_path).name)

    return temp_path


def export_clip(
    video_path: str,
    start: float,
    end: float,
    output_path: str,
    clip_id: str = "clip",
    transcript_segments: Optional[List[Dict]] = None,
    analysis: Optional[Dict] = None,
    device: str = None,
) -> Optional[str]:
    t_start = time.perf_counter()

    try:
        start = float(start)
        end = float(end)
    except (TypeError, ValueError):
        log.error("[%s] Invalid timestamps: start=%r end=%r", clip_id, start, end)
        return None

    if end <= start:
        log.error("[%s] Invalid range: start=%.3f end=%.3f", clip_id, start, end)
        return None

    if analysis is None:
        analysis = analyze_clip(video_path, start, end,
                                transcript_segments=transcript_segments, clip_id=clip_id)

    if not isinstance(analysis, dict) or not isinstance(analysis.get("export_strategy"), dict):
        log.error("[%s] Invalid analysis; skipping.", clip_id)
        return None

    strategy = _sanitize_strategy(analysis["export_strategy"])
    analysis["export_strategy"] = strategy

    if strategy.get("should_drop", False):
        log.info("[%s] DROP — guest cam off or no face", clip_id)
        return None

    info = _get_video_info(video_path)
    encoder = _get_best_encoder()
    logo_file = Path(cfg["thumbnail"].get("template_path", "channel_logo.png"))
    use_logo = logo_file.exists()
    if use_logo and not _is_readable_media_input(logo_file):
        log.warning("[%s] Logo not readable; skipping logo.", clip_id)
        use_logo = False
        
    cta_file = Path("cta.mp3")
    use_cta = cta_file.exists()
    
    has_audio = _has_audio_stream(video_path)
    if not has_audio:
        log.warning("[%s] No audio stream; exporting silent video.", clip_id)

    # ── Premium Render Path ──────────────────────────────────────────────────
    if premium_enabled:
        try:
            from premium_render import PremiumRender
            pr = PremiumRender()
            result = pr.render_clip(
                video_path, start, end, output_path,
                clip_id=clip_id,
                face_enhance=cfg.get("premium", {}).get("face_enhancement", True),
                two_pass=True,
            )
            if result:
                dur = max(time.perf_counter() - t_start, 0.001)
                log.info("✅ [%s] Premium render done in %.1fs", clip_id, dur)
                return output_path
            log.error("[%s] Premium render failed", clip_id)
            return None
        except Exception as e:
            log.error("[%s] Premium render error: %s — falling back to standard", clip_id, e)

    enable_var_speed = cfg["export"].get("enable_variable_speed", True)
    global_speed = _normalize_speed(cfg["export"].get("global_speed_factor", 1.0))
    analysis_speed = _normalize_speed(strategy.get("speed_factor", 1.0))

    if enable_var_speed:
        speed = _normalize_speed(analysis_speed)
        log.info("[%s] Using variable speed: %.2fx", clip_id, speed)
    else:
        speed = _normalize_speed(global_speed)
        log.info("[%s] Variable speed disabled, using global speed: %.2fx", clip_id, speed)

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

    # ── Super-Resolution Path ───────────────────────────────────────────────
    if super_res_enabled and not premium_enabled:
        try:
            from utils.super_res import SuperResEnhancer
            sr = SuperResEnhancer(scale=4, device=device)
            if sr.available:
                log.info("[%s] Super-res mode: native-res export → 4x upscale", clip_id)
                v_filter_native = _build_enhance_stack(
                    analysis, source_fps=info["fps"],
                    use_logo=False, output_duration=output_duration,
                    native_res=True,
                )
                native_path = _export_native_res(
                    video_path, input_seek, trim_start, clip_duration,
                    speed, v_filter_native, clip_id,
                )
                if native_path:
                    output_file = Path(output_path)
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    target_w = int(cfg["export"]["width"])
                    target_h = int(cfg["export"]["height"])
                    success = sr.upscale_video(
                        native_path, output_path,
                        target_w=target_w, target_h=target_h,
                    )
                    # Clean up temp native-res file
                    try:
                        temp_dir = str(Path(native_path).parent)
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    except Exception:
                        pass
                    if success:
                        dur = max(time.perf_counter() - t_start, 0.001)
                        log.info("✅ [%s] Super-res done in %.1fs", clip_id, dur)
                        return output_path
                    log.error("[%s] Super-res failed; falling back to standard", clip_id)
                else:
                    log.error("[%s] Native-res export failed; falling back to standard", clip_id)
            else:
                log.info("[%s] Super-res unavailable; using standard pipeline", clip_id)
        except Exception as e:
            log.warning("[%s] Super-res error (%s); falling back to standard", clip_id, e)

    a_filter = (
        _build_audio_filter(speed, trim_start=trim_start,
                            trim_duration=clip_duration, output_duration=output_duration)
        if has_audio else None
    )

    output_file = Path(output_path)

    v_filter = _build_enhance_stack(analysis, source_fps=info["fps"],
                                    use_logo=use_logo, output_duration=output_duration)
    video_filter_complex = (
        f"[0:v]trim=start={trim_start:.6f}:duration={clip_duration:.6f},"
        f"setpts=(PTS-STARTPTS)/{speed:.6f}[v_src];{v_filter}"
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-y",
           "-threads", "0",
           "-ss", f"{input_seek:.6f}",
           "-t", f"{input_duration:.6f}",
           "-i", video_path]

    if use_logo:
        cmd.extend(["-i", str(logo_file)])
        
    if use_cta:
        cmd.extend(["-i", str(cta_file)])
        
    cta_idx = 2 if use_logo else 1

    a_map = "0:a:0?"
    if has_audio:
        if use_cta:
            video_filter_complex += f";[0:a:0?]{a_filter}[main_a];[main_a][{cta_idx}:a]amix=inputs=2:duration=first:dropout_transition=2:weights=0.8 1.5[a_out]"
            a_map = "[a_out]"
            a_filter = None

    cmd.extend([
        "-filter_complex", video_filter_complex,
        "-map", "[v_out]",
        "-c:v", encoder,
        "-b:v", str(cfg["export"].get("video_bitrate", "8M")),
        "-maxrate", "15M",
        "-bufsize", "24M",
    ])

    if has_audio:
        cmd.extend(["-map", a_map])
        if a_filter:
            cmd.extend(["-af", a_filter])
        cmd.extend([
            "-c:a", "aac",
            "-b:a", str(cfg["export"].get("audio_bitrate", "192k")),
            "-shortest",
        ])

    if encoder == "h264_nvenc":
        cmd.extend(["-preset", "p4", "-tune", "hq", "-rc", "vbr",
                    "-cq", "26", "-spatial-aq", "1", "-b_ref_mode", "middle"])
    elif encoder == "libx264":
        cmd.extend(["-preset", "veryfast", "-crf", str(cfg["export"].get("crf", 23))])

    cmd.extend(["-movflags", "+faststart", output_path])

    log.info("[%s] 🎬 Exporting %.1fs (speed=%.2fx, encoder=%s)...",
             clip_id, end - start, speed, encoder)

    try:
        success, error = _run_ffmpeg_with_retry(cmd, output_path, clip_id, expected_duration=output_duration)

        if not success:
            fallback_source_cmd = cmd
            if use_logo:
                log.warning("[%s] Retrying without logo...", clip_id)
                no_logo_filter = _build_enhance_stack(
                    analysis, source_fps=info["fps"], use_logo=False, output_duration=output_duration)
                no_logo_complex = (
                    f"[0:v]trim=start={trim_start:.6f}:duration={clip_duration:.6f},"
                    f"setpts=(PTS-STARTPTS)/{speed:.6f}[v_src];{no_logo_filter}"
                )
                fallback_source_cmd = _without_logo_cmd(cmd, logo_file, no_logo_complex)
                success, error = _run_ffmpeg_with_retry(
                    fallback_source_cmd, output_path, clip_id, attempts=1, expected_duration=output_duration)
                if success:
                    dur = max(time.perf_counter() - t_start, 0.001)
                    log.info("✅ [%s] Done in %.1fs (logo skipped)", clip_id, dur)
                    return output_path

            if encoder == "h264_nvenc":
                log.warning("[%s] NVENC failed; retrying with libx264...", clip_id)
                fallback_cmd = _libx264_fallback_cmd(fallback_source_cmd)
                success, error = _run_ffmpeg_with_retry(
                    fallback_cmd, output_path, clip_id, attempts=1, expected_duration=output_duration)
                if success:
                    dur = max(time.perf_counter() - t_start, 0.001)
                    log.info("✅ [%s] Done in %.1fs (libx264 fallback)", clip_id, dur)
                    return output_path

            log.error("[%s] Export failed: %s", clip_id, error)
            return None

        dur = max(time.perf_counter() - t_start, 0.001)
        log.info("✅ [%s] Done in %.1fs (%.1fx real-time)", clip_id, dur, output_duration / dur)
        return output_path

    except Exception as e:
        log.error("[%s] Export crash: %s", clip_id, e)
        return None


def _validate_av_sync(video_path: str, clip_id: str, tolerance: float = 0.5) -> bool:
    """Validate audio-video duration sync. Returns True if OK."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        v_result = subprocess.run(cmd, capture_output=True, text=True)
        v_dur = float(v_result.stdout.strip()) if v_result.stdout.strip() else 0.0

        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        a_result = subprocess.run(cmd, capture_output=True, text=True)
        a_dur = float(a_result.stdout.strip()) if a_result.stdout.strip() else 0.0

        if v_dur > 0 and a_dur > 0:
            diff = abs(v_dur - a_dur)
            if diff > tolerance:
                log.warning("[%s] A/V sync issue: video=%.2fs audio=%.2fs (diff=%.2fs > %.2fs tolerance)",
                           clip_id, v_dur, a_dur, diff, tolerance)
                return False
            log.debug("[%s] A/V sync OK: diff=%.3fs", clip_id, diff)
        return True
    except Exception as e:
        log.debug("[%s] A/V validation failed: %s", clip_id, e)
        return True  # Don't block on validation failure


def _parse_time_to_seconds(time_val) -> float:
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
            pass
    return 0.0


def export_all(
    highlights,
    video_path: str,
    transcript_path: Optional[str] = None,
    generate_seo: bool = True,
) -> List[Path]:
    """
    Export all clips sequentially.
    After each successful export, immediately generates SEO for that clip
    (one AI call per clip, with breathing room between calls).
    """
    if isinstance(highlights, (str, Path)):
        import yaml
        with open(highlights, "r") as f:
            highlights = yaml.safe_load(f) or {}

    if not isinstance(highlights, dict):
        log.error("Invalid highlights: expected mapping, got %s", type(highlights).__name__)
        return []

    if transcript_path is None:
        candidate = Path(cfg["paths"]["transcripts"]) / f"{Path(video_path).stem}.json"
        transcript_path = str(candidate) if candidate.exists() else None
    transcript_segments = _load_transcript_segments(transcript_path)

    log.info("🚀 Export phase starting...")
    out_dir = Path(cfg["paths"]["shorts"]) / time.strftime("%Y-%m-%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare SEO context once (trend fetch is expensive)
    seo_context = {}
    if generate_seo:
        try:
            from trends import get_trending_context
            import json as _json

            video_title = ""
            live_stream_url = ""
            meta_file = Path(cfg["paths"]["input"]) / "video_metadata.json"
            if meta_file.exists():
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = _json.load(f)
                    video_title = meta.get("title", "")
                    live_stream_url = meta.get("live_stream_url", "")

            trend = get_trending_context(domain="cricket", region="IN", video_title=video_title)
            seo_context = {
                "video_title": video_title,
                "scorecard": trend.get("scorecard", ""),
                "trend_topics": trend.get("topics", []),
                "live_stream_url": live_stream_url or trend.get("live_stream_url", ""),
            }
            log.info("Trend context loaded for SEO.")
        except Exception as e:
            log.warning("Could not load trend context: %s", e)

    # Parse and validate highlights
    items = []
    for clip_id, info in highlights.items():
        if not isinstance(info, dict):
            continue
        start = _parse_time_to_seconds(info.get("start", info.get("start_sec", 0.0)))
        end = _parse_time_to_seconds(info.get("end", info.get("end_sec", 0.0)))
        if end <= start:
            log.warning("Skipping invalid highlight %s: start=%.3f end=%.3f", clip_id, start, end)
            continue
        items.append((clip_id, start, end, info))

    items.sort(key=lambda x: x[1])

    # Pre-filter: run analysis upfront to reject bad clips early
    # Degraded mode: if face detection fails, try center crop instead of dropping
    filtered_items = []
    for clip_id, start, end, info in items:
        analysis = analyze_clip(video_path, start, end,
                                transcript_segments=transcript_segments, clip_id=clip_id)
        strategy = analysis.get("export_strategy", {})
        if strategy.get("should_drop", False):
            layout = analysis.get("layout", {}).get("layout_type", "unknown")
            # Degraded mode: override to center crop instead of dropping
            log.info("[%s] PRE-FILTER layout=%s — trying degraded center-crop mode", clip_id, layout)
            analysis["export_strategy"]["should_drop"] = False
            analysis["export_strategy"]["layout_mode"] = "solo"
            analysis["layout"]["layout_type"] = "solo"
            analysis["layout"]["face_in_frame"] = True
            analysis["layout"]["face_cx"] = None  # Force center detection
            analysis["layout"]["face_cy"] = None
        filtered_items.append((clip_id, start, end, info, analysis))

    log.info("Pre-filter: %d/%d clips passed (degraded mode for dropped)", len(filtered_items), len(items))

    if not filtered_items:
        log.warning("No clips passed pre-filter — nothing to export")
        return []

    # ── Parallel Export ─────────────────────────────────────────────────────────
    # Use thread pool for I/O-bound FFmpeg processes
    encoder = _get_best_encoder()
    super_res = cfg.get("export", {}).get("super_resolution", False)
    if super_res:
        import torch
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
        max_workers = max(1, min(gpu_count, len(filtered_items)))  # 1 worker per GPU
    else:
        max_workers = max(1, min(2 if encoder == "h264_nvenc" else 4, len(filtered_items)))
    log.info(f"🚀 Starting parallel export with {max_workers} workers (super_res={super_res})...")

    exported_clips = []
    seo_queue: List[Dict] = []   # queued after all encoding is done

    def _export_one(args):
        """Worker function for parallel export."""
        idx, total, clip_id, start, end, info, analysis = args
        out_file = out_dir / f"{clip_id}.mp4"
        # Assign GPU round-robin for super-res workers
        device = None
        if super_res:
            import torch
            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
            device = f"cuda:{(idx - 1) % gpu_count}" if gpu_count > 1 else "cuda:0"
        path = export_clip(video_path, start, end, str(out_file),
                           clip_id, transcript_segments=transcript_segments, analysis=analysis,
                           device=device)
        # Post-export A/V sync validation
        if path and Path(path).exists():
            _validate_av_sync(path, clip_id)
        return (idx, total, clip_id, path, info)

    # Run exports in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_export_one, (idx, len(filtered_items), clip_id, start, end, info, analysis))
            for idx, (clip_id, start, end, info, analysis) in enumerate(filtered_items, start=1)
        ]

        for future in as_completed(futures):
            try:
                idx, total, clip_id, path, info = future.result()
                if path:
                    exported_clips.append(Path(path))
                    log.info("[%d/%d] Exported: %s", idx, total, clip_id)
                    # Queue for SEO — don't block encoding with API calls
                    if generate_seo and seo_context:
                        seo_queue.append({
                            "clip_id": clip_id,
                            "transcript": info.get("text", "Cricket Live"),
                        })
                else:
                    log.warning("[%d/%d] Export failed or dropped: %s", idx, total, clip_id)
            except Exception as e:
                log.error(f"Parallel export error: {e}")

    # ── SEO phase: runs after all encoding is done ───────────────────────────
    # One AI call per clip, 8s wait between calls to avoid 429.
    # Encoding is fully unblocked — SEO never delays the next export.
    if generate_seo and seo_context and seo_queue:
        log.info("🏷  SEO phase: %d clips...", len(seo_queue))
        try:
            from seo import generate_seo_for_exported_clip
            for idx, item in enumerate(seo_queue):
                # 8s pause between calls (skip before first call)
                inter_pause = 8.0 if idx > 0 else 0.0
                generate_seo_for_exported_clip(
                    clip_id=item["clip_id"],
                    transcript=item["transcript"],
                    output_dir=str(out_dir),
                    video_title=seo_context.get("video_title", ""),
                    scorecard=seo_context.get("scorecard", ""),
                    trend_topics=seo_context.get("trend_topics", []),
                    live_stream_url=seo_context.get("live_stream_url", ""),
                    inter_clip_pause=inter_pause,
                )
        except Exception as e:
            log.error("SEO phase failed: %s", e)

    log.info("✨ Export complete: %d clips in %s", len(exported_clips), out_dir)
    return exported_clips


if __name__ == "__main__":
    pass
