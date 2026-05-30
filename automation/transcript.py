"""transcript.py — Cached YouTube transcript fetcher + LLM formatter.

Priority order:
    1. youtube-transcript-api (Python library, JSON)
    2. yt-dlp (VTT subtitle download, parsed)
    3. empty result with source="unavailable"

Results cached 1h in TRANSCRIPT_CACHE.

Usage::

    from .transcript import fetch, format_for_llm

    data = fetch("https://youtu.be/dQw4w9WgXcQ")
    data["segments"]  # -> [{"start": 0.0, "end": 5.0, "text": "..."}]
    data["source"]    # -> "api" | "vtt" | "unavailable"

    # LLM-ready text:
    text = format_for_llm(data["segments"], max_seconds=120, max_segments=50)
"""

import re
import json
from pathlib import Path

from ._cache import TRANSCRIPT_CACHE


def _extract_video_id(url: str) -> str | None:
    """Extract the 11-char YouTube video ID from a URL.

    Supports: youtu.be/ID, watch?v=ID, shorts/ID, embed/ID, youtube.com/v/ID.
    Returns None if no ID found.
    """
    patterns = [
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:watch\?v=)([a-zA-Z0-9_-]{11})",
        r"(?:shorts/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/v/)([a-zA-Z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def _vtt_timestamp_to_seconds(ts: str) -> float:
    """Convert a VTT timestamp (MM:SS.mmm or HH:MM:SS.mmm) to seconds."""
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    m, s = parts
    return int(m) * 60 + float(s)


def _parse_vtt(text: str) -> list[dict]:
    """Parse VTT subtitle text into segment dicts.

    Each segment: {"start": float, "end": float, "text": str}.
    HTML tags (<c>, </c>) are stripped.
    """
    segments = []
    block_pattern = re.compile(
        r"(\d{2}:\d{2}\.\d{3})\s+-->\s+(\d{2}:\d{2}\.\d{3})\s*\n(.+?)(?=\n\n|\n\d{2}|\Z)",
        re.DOTALL,
    )
    for match in block_pattern.finditer(text):
        start_str = match.group(1)
        end_str = match.group(2)
        raw = match.group(3).strip()
        clean = re.sub(r"<[^>]+>", "", raw).replace("\n", " ").strip()
        if not clean:
            continue
        start_sec = _vtt_timestamp_to_seconds(start_str)
        end_sec = _vtt_timestamp_to_seconds(end_str)
        segments.append({"start": start_sec, "end": end_sec, "text": clean})
    return segments


def _fetch_via_api(video_id: str) -> dict | None:
    """Fetch transcript via youtube-transcript-api. Returns None on failure.

    Tries English first, then any available language (auto-generated or manual).
    Supports youtube-transcript-api v1.x API.
    Passes cookies.txt if available to bypass Colab IP blocks.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        return None
    try:
        # Pass cookies if available (bypasses Colab IP blocks)
        cookie_path = Path("cookies.txt")
        if cookie_path.exists() and cookie_path.stat().st_size > 0:
            api = YouTubeTranscriptApi(cookies=str(cookie_path))
        else:
            api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)

        # Try English first (manual > generated)
        transcript = None
        for lang in ["en", "en-US", "en-GB"]:
            try:
                transcript = transcript_list.find_transcript([lang])
                break
            except Exception:
                continue
        if transcript is None:
            for lang in ["en", "en-US", "en-GB"]:
                try:
                    transcript = transcript_list.find_generated_transcript([lang])
                    break
                except Exception:
                    continue

        # Fallback: use whatever language is available (Hindi, auto-generated, etc.)
        if transcript is None:
            try:
                transcript = next(iter(transcript_list))
            except StopIteration:
                return None

        raw = transcript.fetch()
        # Handle different response formats from youtube-transcript-api
        if hasattr(raw, "snippets"):
            raw_snippets = raw.snippets
        else:
            raw_snippets = raw

        segments = []
        for s in raw_snippets:
            if isinstance(s, dict):
                start = s["start"]
                duration = s["duration"]
                text = s["text"]
            else:
                start = getattr(s, "start")
                duration = getattr(s, "duration")
                text = getattr(s, "text")
            segments.append({
                "start": start,
                "end": start + duration,
                "text": text
            })
        return {"segments": segments, "language": transcript.language_code, "source": "api"}
    except Exception:
        return None


def _fetch_via_ytdlp(video_id: str) -> dict | None:
    """Fetch transcript via yt-dlp (VTT download). Returns None on failure."""
    import subprocess
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(suffix=".vtt", delete=False, mode="w") as f:
            tmp_path = f.name
        cmd = [
            "yt-dlp", "--skip-download", "--write-auto-subs", "--sub-langs", "en",
            "-o", tmp_path.replace(".vtt", ""), f"https://www.youtube.com/watch?v={video_id}",
        ]
        # Pass cookies if available (bypasses Colab IP blocks)
        cookie_path = Path("cookies.txt")
        if cookie_path.exists() and cookie_path.stat().st_size > 0:
            cmd.extend(["--cookies", str(cookie_path)])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        vtt_files = list(Path(tmp_path).parent.glob(f"{Path(tmp_path).stem}*.vtt"))
        if not vtt_files and not result.returncode:
            vtt_files = list(Path(tmp_path).parent.glob("*.vtt"))
        if vtt_files:
            text = Path(vtt_files[0]).read_text(encoding="utf-8", errors="replace")
            for p in vtt_files:
                try:
                    p.unlink()
                except OSError:
                    pass
            segments = _parse_vtt(text)
            return {"segments": segments, "language": "en", "source": "vtt"}
        for p in vtt_files:
            try:
                p.unlink()
            except OSError:
                pass
        return None
    except Exception:
        return None


def format_for_llm(segments: list[dict], max_seconds: float | None = None, max_segments: int = 100) -> str:
    """Format transcript segments for LLM consumption.

    Produces a timestamped plain-text transcript suitable for prompt injection.
    Optionally trims to *max_seconds* of video time or *max_segments* entries.

    Args:
        segments: List of dicts with keys ``start``, ``end``, ``text``.
        max_seconds: If set, only include segments up to this video time.
        max_segments: Maximum number of segments to include.

    Returns:
        Formatted string::

            [00:00] Hello and welcome to the stream
            [00:05] Today we're talking about cricket
    """
    out = []
    for seg in segments:
        if max_seconds is not None and seg["start"] > max_seconds:
            break
        if len(out) >= max_segments:
            break
        ts = int(seg["start"])
        m, s = divmod(ts, 60)
        h, m = divmod(m, 60)
        tag = f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
        out.append(f"[{tag}] {seg.get('text', '').strip()}")
    return "\n".join(out)


def fetch(url: str, output_path: str | None = None, video_path: str | None = None) -> dict:
    """Fetch transcript for *url*.

    Tries youtube-transcript-api first, then yt-dlp VTT fallback.
    If both fail and video_path is provided, falls back to local whisper transcription.
    Results cached 1h per video_id.

    Args:
        url: YouTube video URL (standard, short, shorts, or embed).
        output_path: Optional path to write JSON transcript file.
        video_path: Optional path to local video file for local transcription fallback.

    Returns:
        dict with keys: segments, language, source.
        source is "api", "vtt", "local_whisper", or "unavailable".
    """
    video_id = _extract_video_id(url)
    if not video_id:
        return {"segments": [], "language": "unknown", "source": "unavailable"}
    cached = TRANSCRIPT_CACHE.get(video_id)
    if cached is not None and not (cached.get("source") == "unavailable" and video_path):
        return cached
    result = _fetch_via_api(video_id)
    if result is None:
        result = _fetch_via_ytdlp(video_id)
    if (result is None or result.get("source") == "unavailable") and video_path:
        try:
            from transcribe import transcribe
            import tempfile
            temp_out = output_path or str(Path(tempfile.gettempdir()) / f"transcribe_{video_id}.json")
            transcribe(video_path, temp_out)
            if Path(temp_out).exists():
                with open(temp_out, "r", encoding="utf-8") as f:
                    local_data = json.load(f)
                result = {
                    "segments": local_data.get("segments", []),
                    "language": local_data.get("language", "unknown"),
                    "source": "local_whisper"
                }
                if not output_path:
                    try:
                        Path(temp_out).unlink()
                    except OSError:
                        pass
        except Exception:
            pass
    if result is None:
        result = {"segments": [], "language": "unknown", "source": "unavailable"}

    # Centralized cricket spelling correction for ALL sources (api/vtt/local).
    # The local_whisper path is already corrected inside transcribe(); applying
    # the (idempotent) spelling pass here ensures api/vtt transcripts — the
    # common case — are corrected too. Cheap, deterministic, no network.
    if result.get("segments") and result.get("source") in ("api", "vtt"):
        try:
            from utils.transcript_postproc import correct_segments
            result["segments"], _n = correct_segments(result["segments"])
        except Exception:
            pass

    TRANSCRIPT_CACHE.set(video_id, result)
    if output_path and result.get("segments"):
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
