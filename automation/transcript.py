"""transcript.py — Cached YouTube transcript fetcher.
Priority: youtube-transcript-api → yt-dlp VTT → empty."""
import re
import json
from pathlib import Path

from ._cache import TRANSCRIPT_CACHE


def _extract_video_id(url: str) -> str | None:
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


def _parse_vtt(text: str) -> list[dict]:
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


def _vtt_timestamp_to_seconds(ts: str) -> float:
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    m, s = parts
    return int(m) * 60 + float(s)


def _fetch_via_api(video_id: str) -> dict | None:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        return None
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_transcript(["en", "en-US", "en-GB"])
        except Exception:
            transcript = transcript_list.find_generated_transcript(["en", "en-US", "en-GB"])
        segments = [
            {"start": s["start"], "end": s["start"] + s["duration"], "text": s["text"]}
            for s in transcript.fetch()
        ]
        return {"segments": segments, "language": transcript.language_code, "source": "api"}
    except Exception:
        return None


def _fetch_via_ytdlp(video_id: str) -> dict | None:
    import subprocess
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(suffix=".vtt", delete=False, mode="w") as f:
            tmp_path = f.name
        cmd = [
            "yt-dlp", "--skip-download", "--write-auto-subs", "--sub-langs", "en",
            "-o", tmp_path.replace(".vtt", ""), f"https://www.youtube.com/watch?v={video_id}",
        ]
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


def fetch(url: str, output_path: str | None = None) -> dict:
    video_id = _extract_video_id(url)
    if not video_id:
        return {"segments": [], "language": "unknown", "source": "unavailable"}
    cached = TRANSCRIPT_CACHE.get(video_id)
    if cached is not None:
        return cached
    result = _fetch_via_api(video_id)
    if result is None:
        result = _fetch_via_ytdlp(video_id)
    if result is None:
        result = {"segments": [], "language": "unknown", "source": "unavailable"}
    TRANSCRIPT_CACHE.set(video_id, result)
    if output_path and result.get("segments"):
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
