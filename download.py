"""
download.py — Phase 1: Download a YouTube VOD/stream with yt-dlp.

Downloads at the HIGHEST available quality (up to 4K/2160p).
Prefers VP9/AV1 for quality, remuxes to MP4 container.

Uses yt-dlp's current YouTube clients with Colab-friendly fallbacks.
"""

import argparse
from collections import deque
import json
import os
import re
import shutil
import subprocess
import sys
import time
import random
from typing import Optional
from pathlib import Path

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("download", cfg["logging"]["log_file"], cfg["logging"]["level"])


ACCESS_ERROR_HINTS = (
    "403",
    "forbidden",
    "sign in to confirm",
    "not a bot",
    "po token",
    "proof of origin",
    "bot",
)


def _is_colab() -> bool:
    return bool(os.environ.get("COLAB_GPU") or Path("/content").exists())


def _compact_stderr(stderr: str, max_lines: int = 12) -> str:
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    return "\n".join(lines[-max_lines:])


def _has_access_error(stderr: str) -> bool:
    lower = stderr.lower()
    return any(hint in lower for hint in ACCESS_ERROR_HINTS)


def _cookie_args(dl_cfg: dict) -> list[str]:
    cookie_path = dl_cfg.get("cookies") or dl_cfg.get("cookies_path") or dl_cfg.get("cookiefile")
    candidates = [Path(cookie_path)] if cookie_path else [Path("cookies.txt")]
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return ["--cookies", str(candidate)]
    return []


def _js_runtime_args(dl_cfg: dict) -> list[str]:
    runtime = dl_cfg.get("js_runtime", "auto")
    if not runtime or str(runtime).lower() in {"none", "false", "off"}:
        return []
    if str(runtime).lower() != "auto":
        return ["--js-runtimes", str(runtime)]

    for name in ("deno", "node"):
        path = shutil.which(name)
        if path:
            return ["--js-runtimes", f"{name}:{path}"]
    return []


def _normalise_po_token(raw_token: str, client: str) -> str:
    token = raw_token.strip()
    if not token:
        return ""

    normalised = []
    for part in (p.strip() for p in token.split(",")):
        if not part:
            continue
        if "." in part.split("+", 1)[0]:
            normalised.append(part)
            continue
        if "+" in part:
            legacy_client, legacy_token = part.split("+", 1)
            normalised.append(f"{legacy_client}.gvs+{legacy_token}")
            continue
        normalised.append(f"{client}.gvs+{part}")
    return ",".join(normalised)


def _extractor_args(client: str, dl_cfg: dict) -> list[str]:
    args = [f"player-client={client}"]

    token_client = "mweb" if client == "default" else client
    po_token = _normalise_po_token(str(dl_cfg.get("po_token") or ""), token_client)
    if po_token:
        args.append(f"po_token={po_token}")

    if dl_cfg.get("pot_trace"):
        args.append("pot_trace=true")
    if dl_cfg.get("fetch_pot"):
        args.append(f"fetch_pot={dl_cfg['fetch_pot']}")
    if dl_cfg.get("visitor_data"):
        args.append(f"visitor_data={dl_cfg['visitor_data']}")

    return ["--extractor-args", f"youtube:{';'.join(args)}"]


def _base_yt_dlp_cmd(dl_cfg: dict, template: str) -> list[str]:
    cmd = [
        "yt-dlp",
        "--format", dl_cfg.get("format", "bv*+ba/b"),
        "--merge-output-format", "mp4",
        "--format-sort", dl_cfg.get("format_sort", "res:2160,vbr,abr"),
        "--output", template,
        "--no-playlist",
        "--progress",
        "--newline",
        "--no-warnings",
        "--retries", str(dl_cfg.get("retries", 5)),
        "--fragment-retries", str(dl_cfg.get("fragment_retries", 5)),
        "--concurrent-fragments", str(dl_cfg.get("concurrent_fragments", 8)),
        "--retry-sleep", str(dl_cfg.get("retry_sleep", "fragment:exp=1:20")),
        "--sleep-requests", str(dl_cfg.get("sleep_requests", 0)),
    ]

    # aria2c downloader (2-3x faster on Colab gigabit)
    if dl_cfg.get("use_aria2c", False):
        aria2c_path = shutil.which("aria2c")
        if aria2c_path:
            try:
                subprocess.run([aria2c_path, "--version"], capture_output=True, check=True)
                cmd.extend([
                    "--downloader", "aria2c",
                    "--downloader-args", "aria2c:-x 16 -s 16 -k 1M",
                ])
                log.info("aria2c downloader enabled — 16 connections (%s)", aria2c_path)
            except subprocess.CalledProcessError:
                log.warning("aria2c found but failed version check — falling back to default downloader")
        else:
            log.warning("aria2c not found — install with: apt-get install aria2 (or brew install aria2)")
            if _is_colab():
                log.info("Attempting to install aria2c on Colab...")
                try:
                    subprocess.run(["apt-get", "install", "-y", "-qq", "aria2"],
                                   capture_output=True, check=True, timeout=60)
                    aria2c_path = shutil.which("aria2c")
                    if aria2c_path:
                        cmd.extend([
                            "--downloader", "aria2c",
                            "--downloader-args", "aria2c:-x 16 -s 16 -k 1M",
                        ])
                        log.info("aria2c installed and enabled — 16 connections")
                    else:
                        log.warning("aria2c install completed but binary not found")
                except Exception as e:
                    log.warning("Failed to install aria2c: %s", e)

    proxy = dl_cfg.get("proxy")
    if proxy:
        cmd.extend(["--proxy", str(proxy)])

    # Throttled-rate keepalive — prevents connection drops on long downloads
    cmd.extend(["--throttled-rate", "100K"])

    cmd.extend(_cookie_args(dl_cfg))
    cmd.extend(_js_runtime_args(dl_cfg))
    return cmd


def _extract_percent(line: str) -> Optional[float]:
    match = re.search(r"(\d+(?:\.\d+)?)%", line)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _should_log_download_line(line: str, state: dict, interval: float, percent_step: float) -> bool:
    lower = line.lower()
    now = time.monotonic()

    if any(marker in lower for marker in ("destination:", "merging formats", "deleting original file", "has already been downloaded")):
        return True

    percent = _extract_percent(line)
    if percent is not None:
        last_percent = state.get("last_percent")
        last_time = state.get("last_time", 0.0)
        if last_percent is None or percent >= last_percent + percent_step or now - last_time >= interval or percent >= 99.9:
            state["last_percent"] = percent
            state["last_time"] = now
            return True
        return False

    if "[download]" in lower:
        last_time = state.get("last_time", 0.0)
        if now - last_time >= interval:
            state["last_time"] = now
            return True
        return False

    # Filter aria2c raw fragment noise (hash IDs, byte counters)
    if re.match(r'^[\[#0-9]', line) and any(c in line for c in ('#', '[DL:', 'B/0B', '](+')):
        return False
    if '#' in line[:6] and any(c.isdigit() for c in line[:10]):
        return False
    return not line.startswith("[download]")


def _cleanup_stale_downloads(dest: Path) -> None:
    for candidate in dest.parent.glob(f"{dest.stem}.*"):
        if candidate == dest or not candidate.is_file():
            continue
        try:
            candidate.unlink()
            log.info("🧹 Removed stale download fragment: %s", candidate.name)
        except OSError as e:
            log.warning("Could not remove stale download fragment %s: %s", candidate, e)


def _run_yt_dlp(cmd: list[str], url: str, label: str) -> subprocess.CompletedProcess:
    log.info("🚀 Attempting download with yt-dlp client set: %s", label)
    dl_cfg = cfg.get("download", {})
    progress_interval = float(dl_cfg.get("progress_interval_seconds", 10))
    progress_step = float(dl_cfg.get("progress_percent_step", 5))
    process = subprocess.Popen(
        [*cmd, url],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    tail = deque(maxlen=400)
    progress_state = {"last_time": 0.0, "last_percent": None, "stall_time": None}
    if process.stdout:
        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line:
                continue
            tail.append(line)
            if _should_log_download_line(line, progress_state, progress_interval, progress_step):
                if line.startswith("[download]"):
                    log.info("📥 %s", line.replace("[download]", "", 1).strip())
                else:
                    log.info("%s", line)
            perc = _extract_percent(line)
            if perc is not None and perc >= 99.9:
                if progress_state["stall_time"] is None:
                    progress_state["stall_time"] = time.monotonic()
                elif time.monotonic() - progress_state["stall_time"] > 90:
                    log.warning("Download stalled at 99.9%% for 90s — terminating")
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    break
            else:
                progress_state["stall_time"] = None

    returncode = process.wait()
    output_tail = "\n".join(tail)
    if returncode != 0:
        details = _compact_stderr(output_tail)
        if details:
            log.warning("yt-dlp failed for %s:\n%s", label, details)
    return subprocess.CompletedProcess([*cmd, url], returncode, stdout="", stderr=output_tail)


def _client_attempts(dl_cfg: dict) -> list[str]:
    configured = dl_cfg.get("player_clients")
    if configured:
        if isinstance(configured, str):
            return [item.strip() for item in configured.split(",") if item.strip()]
        return [str(item).strip() for item in configured if str(item).strip()]

    # Defaults first lets yt-dlp choose currently healthy clients; mweb is the
    # recommended fallback when a PO-token provider or PO token is available.
    return [
        "default",
        "mweb",
        "web_safari",
        "android_vr",
        "web_embedded",
    ]


def download(url: str, output_path: Optional[str] = None, sample_minutes: Optional[int] = None) -> Path:
    """
    Download a YouTube video using yt-dlp at the highest available quality.
    If sample_minutes is provided, downloads a random N-minute section from the video.
    """
    dl_cfg = cfg.get("download", {})
    paths_cfg = cfg.get("paths", {"input": "input"})

    if output_path is None:
        output_filename = dl_cfg.get("output_filename", "video.mp4")
        output_path = str(Path(paths_cfg["input"]) / output_filename)

    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _cleanup_stale_downloads(dest)

    stem = dest.stem
    template = str(dest.parent / f"{stem}.%(ext)s")

    base_cmd = _base_yt_dlp_cmd(dl_cfg, template)

    # ─── Capture Video Metadata ──────────────────────────────────────────────
    try:
        title_cmd = [
            "yt-dlp",
            "--dump-json",
            "--no-playlist",
            "--no-warnings",
            *_cookie_args(dl_cfg),
            *_js_runtime_args(dl_cfg),
            url,
        ]
        title_res = subprocess.run(title_cmd, capture_output=True, text=True, timeout=30)
        video_info = json.loads(title_res.stdout) if title_res.returncode == 0 and title_res.stdout else {}
        video_title = video_info.get("title") or "Cricket Highlights"
        log.info(f"📹 Video Title: {video_title}")

        if video_info.get("is_live") or video_info.get("live_status") == "is_live":
            if not dl_cfg.get("allow_live_recording", False):
                log.error(
                    "This URL is an active livestream. Refusing to record indefinitely. "
                    "Use the VOD after the stream ends, or set download.allow_live_recording: true."
                )
                sys.exit(1)
            log.warning("Active livestream detected; yt-dlp will run until the stream ends.")

        meta_file = Path(paths_cfg["input"]) / "video_metadata.json"
        meta_file.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({"title": video_title, "url": url}, f, indent=4)
            
        if sample_minutes is not None and sample_minutes > 0:
            duration = video_info.get("duration", 0)
            if duration > sample_minutes * 60:
                sample_sec = sample_minutes * 60
                max_start = duration - sample_sec
                # pick a random start time from 10% to 90% of max_start to avoid intro/outro if possible
                start_sec = random.uniform(max_start * 0.1, max_start * 0.9)
                end_sec = start_sec + sample_sec
                log.info(f"🎲 Random sample requested: downloading {sample_minutes} min from {start_sec:.1f}s to {end_sec:.1f}s")
                # Add --download-sections
                base_cmd.extend(["--download-sections", f"*{start_sec}-{end_sec}"])
            else:
                log.warning(f"Video duration ({duration}s) is shorter than requested sample ({sample_minutes * 60}s). Downloading full video.")
    except Exception as e:
        log.warning(f"Failed to capture video metadata or set sections: {e}")

    attempts = _client_attempts(dl_cfg)
    last_result: subprocess.CompletedProcess | None = None
    access_error_seen = False
    for index, client in enumerate(attempts, start=1):
        current_cmd = [*base_cmd, *_extractor_args(client, dl_cfg)]
        result = _run_yt_dlp(current_cmd, url, f"{client} ({index})")
        last_result = result
        access_error_seen = access_error_seen or _has_access_error(result.stderr or "")

        if result.returncode == 0:
            log.info("✅ Successfully downloaded video with client set: %s", client)
            break

        if index < len(attempts):
            time.sleep(float(dl_cfg.get("client_retry_delay", 2)))
    else:
        log.error("❌ Download failed for every yt-dlp client attempt.")
        if access_error_seen:
            if _is_colab():
                log.error(
                    "Colab IPs are often challenged by YouTube. Upload cookies.txt, "
                    "set download.po_token, or run a PO-token provider for yt-dlp."
                )
            else:
                log.error(
                    "YouTube returned an access/bot-check error. Try cookies.txt, "
                    "a PO token/provider, or a different network/proxy."
                )
        if not _js_runtime_args(dl_cfg):
            log.error("No Deno/Node JS runtime was found. In Colab, install Deno or Node before running.")
        sys.exit(1)

    # yt-dlp may produce video.mp4 or video.webm → normalise to dest
    produced = dest.parent / f"{stem}.mp4"
    if not produced.exists():
        candidates = list(dest.parent.glob(f"{stem}.*"))
        candidates = [c for c in candidates if not c.suffix.endswith(".part")]
        if not candidates:
            log.error("Download completed but output file not found in %s", dest.parent)
            sys.exit(1)
        produced = candidates[0]

    # Rename to exactly the configured filename if needed
    if produced != dest:
        produced.rename(dest)
        produced = dest

    size_mb = produced.stat().st_size / 1_048_576
    log.info("Download complete → %s (%.1f MB)", produced, size_mb)
    return produced

# ─── CLI entry-point ──────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Download a YouTube VOD or stream.")
    parser.add_argument("url", help="YouTube URL to download")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path (default: input/video.mp4 from config.yaml)",
    )
    args = parser.parse_args()

    download(args.url, args.output)

if __name__ == "__main__":
    main()
