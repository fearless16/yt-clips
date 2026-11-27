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

import os
import re
import json
import logging
from pathlib import Path

from ._cache import TRANSCRIPT_CACHE

log = logging.getLogger("transcript")


def _language_preference() -> list[str]:
    """The stream's own language first: an English track on a Hindi stream is
    an auto-translation or mis-tagged ASR, worse than the Hindi captions."""
    try:
        from utils.config import load_config
        own = str(load_config()["transcription"].get("language") or "").strip()
    except Exception:
        own = ""
    return list(dict.fromkeys(x for x in (own, "hi", "en", "en-US", "en-GB") if x))


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
        r"(\d{2}:\d{2}(?::\d{2})?\.\d{3})\s+-->\s+(\d{2}:\d{2}(?::\d{2})?\.\d{3})[^\n]*\n(.+?)(?=\n\n|\n\d{2}|\Z)",
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


def _fetch_via_youtube_data_api(video_id: str) -> dict | None:
    """Fetch transcript via YouTube Data API v3 (captions.list + captions.download).

    This is the official API — no bot detection, no cookies needed.
    Requires OAuth2 token with youtube.force-ssl scope.
    Returns None on failure.
    """
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
    except ImportError:
        log.warning("YouTube Data API libraries not available")
        return None

    token_path = Path("yt_channel_token.json")
    if not token_path.exists():
        log.warning("YouTube Data API token not found at %s", token_path)
        return None

    try:
        creds = Credentials.from_authorized_user_file(str(token_path))
        # Check if token has youtube.force-ssl scope
        if creds.scopes and "https://www.googleapis.com/auth/youtube.force-ssl" not in creds.scopes:
            log.debug("YouTube token lacks youtube.force-ssl scope, skipping API method")
            return None
        if not creds.valid and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        if not creds.valid:
            log.warning("YouTube Data API credentials not valid")
            return None

        youtube = build("youtube", "v3", credentials=creds)

        captions_response = youtube.captions().list(
            part="snippet",
            videoId=video_id
        ).execute()

        items = captions_response.get("items", [])
        if not items:
            log.warning("No caption tracks found for video %s via YouTube Data API", video_id)
            return None

        caption_id = None
        lang_code = None
        for pref_lang in _language_preference():
            for item in items:
                snippet = item["snippet"]
                if snippet["language"] == pref_lang and snippet.get("trackKind") != "ASR" and snippet.get("status") == "serving":
                    caption_id = item["id"]
                    lang_code = snippet["language"]
                    break
            if caption_id:
                break

        if not caption_id:
            for pref_lang in _language_preference():
                for item in items:
                    snippet = item["snippet"]
                    if snippet["language"] == pref_lang and snippet.get("status") == "serving":
                        caption_id = item["id"]
                        lang_code = snippet["language"]
                        break
                if caption_id:
                    break

        if not caption_id:
            for item in items:
                if item["snippet"].get("status") == "serving":
                    caption_id = item["id"]
                    lang_code = item["snippet"]["language"]
                    break

        caption_response = youtube.captions().download(
            id=caption_id,
            tfmt="srt"
        ).execute()

        segments = _parse_srt(caption_response.decode("utf-8") if isinstance(caption_response, bytes) else caption_response)
        if segments:
            return {"segments": segments, "language": lang_code, "source": "youtube_api"}

    except Exception as e:
        log.warning("YouTube Data API transcript fetch failed for %s: %s", video_id, e)
    return None


def _parse_srt(text: str) -> list[dict]:
    """Parse SRT subtitle text into segment dicts."""
    segments = []
    blocks = re.split(r'\n\s*\n', text.strip())
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        # Parse timestamp line: "00:00:00,000 --> 00:00:05,000"
        ts_match = re.match(r'(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})', lines[1])
        if not ts_match:
            continue
        start = _srt_timestamp_to_seconds(ts_match.group(1))
        end = _srt_timestamp_to_seconds(ts_match.group(2))
        text = ' '.join(lines[2:]).strip()
        if text:
            segments.append({"start": start, "end": end, "text": text})
    return segments


def _srt_timestamp_to_seconds(ts: str) -> float:
    """Convert SRT timestamp (HH:MM:SS,mmm) to seconds."""
    ts = ts.replace(',', '.')
    parts = ts.split(":")
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    return h * 3600 + m * 60 + s


def _fetch_via_api(video_id: str) -> dict | None:
    """Fetch transcript via youtube-transcript-api. Returns None on failure.

    Tries English first, then any available language (auto-generated or manual).
    Supports youtube-transcript-api v1.x API.
    Passes cookies.txt if available to bypass Colab IP blocks.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        log.warning("youtube-transcript-api not installed")
        return None
    try:
        # Build a requests session with cookies if available (bypasses Colab IP blocks)
        import requests
        session = requests.Session()
        cookie_path = Path("cookies.txt")
        if cookie_path.exists() and cookie_path.stat().st_size > 0:
            # Parse Netscape cookies.txt into a cookie jar
            from http.cookiejar import MozillaCookieJar
            jar = MozillaCookieJar(str(cookie_path))
            jar.load(ignore_discard=True, ignore_expires=True)
            session.cookies = jar
        api = YouTubeTranscriptApi(http_client=session)
        transcript_list = api.list(video_id)

        # The stream's own language first (manual > generated), then English.
        transcript = None
        for lang in _language_preference():
            try:
                transcript = transcript_list.find_manually_created_transcript([lang])
                break
            except Exception:
                continue
        if transcript is None:
            for lang in _language_preference():
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
    except Exception as e:
        log.warning("youtube-transcript-api fetch failed for %s: %s", video_id, e)
        return None


def _find_ytdlp() -> str:
    """Find yt-dlp executable — checks venv Scripts, then PATH."""
    import shutil
    import sys
    # Check venv Scripts first (Windows: yt-dlp.exe, Unix: yt-dlp)
    vtd = Path(sys.executable).parent / ("yt-dlp.exe" if sys.platform == "win32" else "yt-dlp")
    if vtd.exists():
        return str(vtd)
    found = shutil.which("yt-dlp")
    if found:
        return found
    return "yt-dlp"


def _fetch_via_ytdlp(video_id: str) -> dict | None:
    """Fetch only invocation-owned VTTs; missing/empty output permits ASR fallback."""
    import subprocess
    import tempfile
    try:
        with tempfile.TemporaryDirectory(prefix="yt-clips-vtt-") as directory:
            base = Path(directory) / "captions"
            cmd = [
                _find_ytdlp(), "--skip-download", "--write-auto-subs", "--sub-langs", "en,hi",
                "--remote-components", "ejs:github",
                "-o", str(base), f"https://www.youtube.com/watch?v={video_id}",
            ]
            env = os.environ.copy()
            deno_bin = str(Path.home() / ".deno" / "bin")
            if Path(deno_bin).exists() and deno_bin not in env.get("PATH", ""):
                env["PATH"] = deno_bin + os.pathsep + env.get("PATH", "")
            cookie_path = Path("cookies.txt")
            if cookie_path.exists() and cookie_path.stat().st_size > 0:
                cmd.extend(["--cookies", str(cookie_path)])
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
            if result.returncode:
                return None
            for path in sorted(Path(directory).glob("captions*.vtt")):
                segments = _parse_vtt(path.read_text(encoding="utf-8", errors="replace"))
                if segments:
                    language = path.stem.rsplit(".", 1)[-1] if "." in path.stem else "unknown"
                    return {"segments": segments, "language": language, "source": "vtt"}
        log.warning("yt-dlp produced no VTT files for %s", video_id)
        return None
    except Exception as e:
        log.warning("yt-dlp transcript fetch failed for %s: %s", video_id, e)
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

    Priority: YouTube Data API → youtube-transcript-api → yt-dlp VTT → local Whisper.
    Results cached 1h per video_id.

    Args:
        url: YouTube video URL (standard, short, shorts, or embed).
        output_path: Optional path to write JSON transcript file.
        video_path: Optional path to local video file for local transcription fallback.

    Returns:
        dict with keys: segments, language, source.
        source is "youtube_api", "api", "vtt", "local_whisper", or "unavailable".
    """
    video_id = _extract_video_id(url)
    if not video_id:
        return {"segments": [], "language": "unknown", "source": "unavailable"}
    cached = TRANSCRIPT_CACHE.get(video_id)
    if cached is not None and not (cached.get("source") == "unavailable" and video_path):
        return cached
    # Priority 1: YouTube Data API v3 (official, no bot detection)
    result = _fetch_via_youtube_data_api(video_id)
    # Priority 2: youtube-transcript-api (with cookies)
    if not result or not result.get("segments"):
        result = _fetch_via_api(video_id)
    # Priority 3: yt-dlp VTT download
    if not result or not result.get("segments"):
        result = _fetch_via_ytdlp(video_id)
    if (not result or not result.get("segments")) and video_path:
        result = None
        try:
            from transcribe import transcribe
            import tempfile
            # Never use the requested destination as scratch: a failed ASR run
            # must not read a previous source's transcript or overwrite review work.
            with tempfile.TemporaryDirectory(prefix="yt-clips-transcript-") as directory:
                temp_out = Path(directory) / "transcript.json"
                transcribe(video_path, str(temp_out))
                if temp_out.exists():
                    local_data = json.loads(temp_out.read_text(encoding="utf-8"))
                    if local_data.get("segments"):
                        result = {"segments": local_data["segments"],
                                  "language": local_data.get("language", "unknown"),
                                  "source": "local_whisper"}
        except Exception as exc:
            log.warning("Local transcript fallback failed for %s: %s", video_id, exc)
    if not result or not result.get("segments"):
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
