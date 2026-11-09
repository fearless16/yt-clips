"""
dry_run.py — Execute pipeline with real code, only API calls stubbed.

Calls automation/orchestrator.run() directly — every internal function,
import path, data-flow branch, and observability layer runs for real.
Only external APIs (YouTube Data API, yt-dlp, ffmpeg/ffprobe, Google
Drive, AI providers, PIL, faster-whisper, HTTP trend fetches, etc.)
are intercepted at the boundary and return plausible fake data.

This catches bugs in function signatures, import paths, config keys,
phase ordering, skip-flag logic, return-type contracts, logging, the
SEO escalate-not-degrade retry queue, and PipelineResult — all locally
in seconds without any network/GPU/disk-IO to external services.

Usage:
    python dry_run.py https://youtu.be/4ylLhtICj1I
    python dry_run.py https://youtu.be/4ylLhtICj1I --skip-download --skip-transcribe
    python dry_run.py https://youtu.be/4ylLhtICj1I --upload --sync --schedule
    python dry_run.py --list-external        # list all API-call intercept points
    python dry_run.py --validate-config      # only check required config keys
"""
import _fix_encoding  # noqa: F401 — force UTF-8 on Windows cp1252

import argparse
import contextlib
import io
import json
import logging
import os
import subprocess
import sys
import time
import tempfile
import shutil
import copy
from functools import wraps
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dry_run")

from utils.logger import run_phase, new_run_id, JsonStreamHandler

# ── Config validation ─────────────────────────────────────────────────────

REQUIRED_CONFIG_KEYS = [
    ("ai", "retry_base_delay_seconds"),
    ("ai", "retry_max_delay_seconds"),
    ("ai", "race_tier_timeout_seconds"),
    ("premium", "yolo_device"),
    ("premium", "yolo_batch_size"),
    ("premium", "identity_refs"),
    ("youtube", "upload_chunk_size_mb"),
    ("youtube", "upload_max_retries"),
    ("youtube", "upload_deadline_seconds"),
]

EXPECTED_PHASE_STAGES = [
    "transcript_fetch", "download", "transcribe", "highlight", "export",
    "enhancement", "seo", "thumbnails", "sync", "upload", "analytics",
    "shorts_intelligence", "provider_health", "event_store_validation",
]

REQUIRED_PROMPTS = [
    "HIGHLIGHT_RANKER_SYSTEM", "HIGHLIGHT_RANKER_USER_TEMPLATE",
    "CANDIDATE_EVAL_SYSTEM", "CANDIDATE_EVAL_USER_TEMPLATE",
    "SEO_SYSTEM", "SEO_USER_TEMPLATE",
    "ANALYTICS_SYSTEM", "ANALYTICS_USER_TEMPLATE",
    "ORCHESTRATOR_SYSTEM",
    "DEFAULT_SCORING_WEIGHTS", "MIN_QUALITY_THRESHOLD",
    "MAX_SELECTED_CLIPS", "MAX_CANDIDATES",
]


@dataclass
class PipelineResult:
    exported: list = field(default_factory=list)
    uploaded_count: int = 0
    failures: list = field(default_factory=list)
    total_seconds: float = 0.0
    transcript_source: str = "none"
    run_id: str = ""
    selected_clips: int = 0
    seo_generated: int = 0


_FAKE_SEGMENTS = [
    {"start": 0.0, "end": 5.0, "text": "Kohli waits for the slower ball."},
    {"start": 5.0, "end": 25.0, "text": "Today coaly is batting brilliantly. Kohli waits for the slower ball and hits a six over the boundary. That batting patience wins the over."},
    {"start": 30.0, "end": 55.0, "text": "Kohli hits a massive six. The bowler misses the yorker and Kohli clears the boundary to finish the cricket match with that six."},
    {"start": 60.0, "end": 85.0, "text": "bumra bowls a perfect yorker. Bumrah changes the bowling angle and takes the wicket because the batter cannot reach the ball."},
    {"start": 120.0, "end": 145.0, "text": "India take a catch in the deep. That wicket ends the batting partnership and India can defend the target with accurate bowling."},
    {"start": 175.0, "end": 180.0, "text": "And that's the end of the over."},
]

_FAKE_TRANSCRIPT = {
    "segments": _FAKE_SEGMENTS,
    "source": "api",
}


def _ensure_skipped_transcript_fixture(
    paths: dict, video_path: str, skip_transcribe: bool
) -> Path | None:
    """Create a temporary transcript only when a dry-run skips transcription."""
    if not skip_transcribe:
        return None
    transcript_dir = Path(paths.get("transcripts", "transcripts"))
    transcript_path = transcript_dir / f"{Path(video_path).stem}.json"
    if transcript_path.exists():
        return None
    transcript_dir.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(
        json.dumps(_FAKE_TRANSCRIPT, ensure_ascii=False), encoding="utf-8"
    )
    return transcript_path

_FAKE_HIGHLIGHTS = {
    "clip1": {"start": 30, "end": 55, "label": "Kohli Six", "text": "coaly hits a massive six"},
    "clip2": {"start": 60, "end": 85, "label": "Yorker", "text": "bumra bowls a perfect yorker"},
    "clip3": {"start": 120, "end": 145, "label": "Amazing Catch", "text": "What a catch in the deep"},
}

_FAKE_SEO_JSON = json.dumps({
    "cover_text": "KOHLI KA CHHAKKA",
    "title": "🔴 Kohli ne maara CHHAKKA! 💥 | RCB vs CSK IPL 2026 | Live Score #Shorts",
    "description": (
        "📝 Virat Kohli smashes a massive six over long-on! The crowd at Chinnaswamy "
        "goes absolutely wild as Kohli deposits Jadeja into the stands!\n\n"
        "🔥 Current Match Situation:\n"
        "RCB is chasing 185 and Kohli is on fire! After a slow start, "
        "the King has shifted gears with back-to-back boundaries. RCB at 95/2 "
        "after 12 overs, needing 90 from 48 balls.\n\n"
        "👉 If you love cricket, please SUBSCRIBE! We are growing together.\n\n"
        "🏟️ Match Info:\n"
        "Series: Indian Premier League 2026 (Match 42)\n"
        "Teams: Royal Challengers Bengaluru (RCB) vs Chennai Super Kings (CSK)\n"
        "Venue: M. Chinnaswamy Stadium, Bengaluru\n"
        "Toss: CSK won the toss and elected to bat.\n\n"
        "🏏 Key Players Today:\n"
        "RCB: Virat Kohli, Rajat Patidar (c), Jacob Bethell, Krunal Pandya\n"
        "CSK: Ruturaj Gaikwad (c), Ravindra Jadeja, Matheesha Pathirana\n\n"
        "⚠️ Disclaimer: This is a watch-along and scorecard video. No live match "
        "footage or audio from official broadcasters. All logos belong to respective owners.\n\n"
        "🏷️ Tags: kohli six, rcb vs csk live, ipl 2026 live, aaj ka match, "
        "virat kohli batting, cricket live match today\n\n"
        "#️⃣ #RCBvsCSK #ViratKohli #IPL2026 #RCB #CSK #LiveCricket #Shorts"
    ),
    "hashtags": [
        "#Shorts", "#ViratKohli", "#RCB", "#CSK", "#IPL2026",
        "#RCBvsCSK", "#LiveCricket", "#CricketLive", "#IPL",
        "#Kohli", "#Cricket", "#T20", "#LiveScore", "#RCBvCSK", "#KohliSix"
    ],
    "search_terms": [
        "kohli six", "rcb vs csk live score", "ipl 2026 match 42",
        "virat kohli batting today", "rcb vs csk highlights",
        "aaj ka match", "live cricket match", "aaj ka match live",
        "ipl live kaise dekhe", "cricket live match today online",
        "लाइव क्रिकेट स्कोर", "kohli six chinnaswamy",
        "rcb vs csk ipl 2026", "live ipl match", "cricket live",
        "ipl 2026 live", "kohli batting rcb", "csk vs rcb scorecard",
        "ipl live stream", "how to watch ipl live",
        "cricket live commentary hindi", "rcb chase",
        "kohli jadeja six", "ipl live score today", "rcb winning"
    ],
})

_FAKE_FFPROBE_OUTPUT = json.dumps({
    "streams": [{"width": 1080, "height": 1920, "codec_type": "video"}],
    "format": {"duration": "180.0"},
})


def _create_fake_video(path: str, size_mb: int = 10):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_bytes(b"X" * (size_mb * 1024 * 1024))
        log.debug("  [FAKE] Created %s (%d MB)", path, size_mb)


def _fake_ffmpeg_output(cmd, root: Path) -> Path | None:
    """Accept only an explicit final output, never a searched-for input file."""
    if not isinstance(cmd, (list, tuple)) or len(cmd) < 2:
        raise ValueError("Dry-run FFmpeg requires an argument list")
    final = str(cmd[-1])
    if final in ("-", "NUL", "/dev/null") or final.startswith("pipe:"):
        return None
    output = Path(final).resolve()
    inputs = {
        Path(str(cmd[i + 1])).resolve()
        for i, value in enumerate(cmd[:-1]) if value == "-i"
    }
    if output in inputs:
        raise ValueError("Dry-run refuses to overwrite an FFmpeg input")
    if output.suffix.lower() not in {".mp4", ".wav", ".nut", ".jpg", ".png", ".pcm"}:
        return None
    if not output.is_relative_to(root.resolve()):
        raise ValueError("Dry-run FFmpeg output escaped its sandbox")
    return output


def _project_module(module) -> bool:
    """Exclude installed dependencies and Face OS from process-global patches."""
    filename = getattr(module, "__file__", None)
    if not filename:
        return False
    try:
        relative = Path(filename).resolve().relative_to(Path(__file__).resolve().parent)
    except ValueError:
        return False
    return not any(part.startswith(".") or part in {"face_os", "site-packages"} for part in relative.parts)


def _loggers() -> list:
    import logging
    return [logging.getLogger(), *[
        logger for logger in logging.Logger.manager.loggerDict.values()
        if isinstance(logger, logging.Logger)
    ]]


def _close_sandbox_logs(sandbox: Path, preserved: set) -> None:
    """Release only new owned file handles before Windows directory cleanup."""
    for logger in _loggers():
        for handler in list(logger.handlers):
            if handler in preserved:
                continue
            filename = getattr(handler, "baseFilename", None)
            if filename is None:
                filename = getattr(getattr(handler, "_file", None), "name", None)
            if not isinstance(filename, (str, os.PathLike)):
                continue
            path = Path(filename)
            path = path if path.is_absolute() else sandbox / path
            if path.resolve().is_relative_to(sandbox):
                logger.removeHandler(handler)
                handler.close()


def _isolated_run(function):
    """Keep relative files, cached configuration and generated evidence off live paths.

    Like the existing API patches this harness is process-global, not thread-safe.
    Set YT_CLIPS_KEEP_DRY_RUN=1 to retain debugging artifacts explicitly.
    """
    @wraps(function)
    def isolated(*args, **kwargs):
        keep = os.environ.get("YT_CLIPS_KEEP_DRY_RUN") == "1"
        temp_base = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "Temp" / "opencode"
        temp_base.mkdir(parents=True, exist_ok=True)
        sandbox = Path(tempfile.mkdtemp(prefix="yt-clips-dry-run-", dir=temp_base))
        preserved_handlers = {handler for logger in _loggers() for handler in logger.handlers}
        try:
            return _run_in_sandbox(function, args, kwargs, sandbox, keep)
        finally:
            _close_sandbox_logs(sandbox, preserved_handlers)
            if not keep:
                shutil.rmtree(sandbox)
    return isolated


def _run_in_sandbox(function, args, kwargs, sandbox: Path, keep: bool):
    import yaml
    import automation.config as config_module
    import utils.config as utils_config

    original_cwd = Path.cwd()
    original = config_module.load()
    config = copy.deepcopy(original)
    # This synthetic acceptance fixture intentionally tests three candidates
    # and reserve handling, independently of the production two-Short default.
    config.setdefault("clip_selection", {})["max_selected"] = 3
    config["paths"] = {name: str(sandbox / name) for name in
                       ("input", "temp", "transcripts", "highlights", "shorts", "logs")}
    # Exercise the real policy/selection integration against an isolated DB.
    config.setdefault("shorts_intelligence", {})["enabled"] = True
    config["shorts_intelligence"]["db_path"] = str(sandbox / "shorts_intelligence.db")
    # Stubbed media cannot receive a real owner approval. Release gating is
    # tested separately; this run exercises only the fake API happy path.
    config["release_review"] = {"required": False}
    config["delivery_qc"] = {"enabled": False}
    # Keep production word requirements ON. Stub ASR output, not validation.
    config.setdefault("export", {}).setdefault("captions", {})["require_word_timing"] = True
    if kwargs.get("composition") is not None:
        config["export"]["composition"] = kwargs["composition"]
    from automation.cricket_gameplay import GAMEPLAY_COMPOSITIONS, sha256
    if config["export"].get("composition") in GAMEPLAY_COMPOSITIONS:
        # Exercise enabled stock with synthetic provenance, NEVER live permissions.
        game_dir = sandbox / "input/gameplay"
        game_dir.mkdir(parents=True)
        game = game_dir / "fixture.mp4"
        game.write_bytes(b"SYNTHETIC DRY-RUN MEDIA; NOT A PRODUCTION RECORDING")
        grant = game_dir / "test-only-recording.json"
        grant.write_text(json.dumps({"id": "synthetic-stock", "uploader": "SYNTHETIC TEST ONLY",
            "license": "Creative Commons Attribution license (reuse allowed)"}), encoding="utf-8")
        manifest = game_dir / "library.json"
        manifest.write_text(json.dumps({"schema_version":1,"usage":"STOCK","assets":[{
            "id":"dry-run-test-only","game":"EA Cricket 07","publisher":"EA",
            "status":"AVAILABLE","rights_status":"UNRESOLVED",
            "reviewer":"SYNTHETIC FIXTURE ONLY","reviewed_at":"2026-11-08",
            "path":game.name,"sha256":sha256(game),
            "source_video_id":"synthetic-stock","owner":"SYNTHETIC TEST ONLY",
            "license_evidence":{"path":grant.name,"sha256":sha256(grant)},
            "attribution":"SYNTHETIC TEST ONLY; not licensed production gameplay",
            "approved_ranges":[[0,180]],"crop":[0,0,1920,1080]}]}),encoding="utf-8")
        config["export"]["gameplay_manifest"] = str(manifest)
    (sandbox / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=False), encoding="utf-8")
    # Fake credentials only: never copy production tokens/cookies into a test.
    (sandbox / "yt_channel_token.json").write_text('{"token": "fake"}', encoding="utf-8")
    (sandbox / "drive_token.json").write_text('{"token": "fake"}', encoding="utf-8")
    saved = []
    for name, module in list(sys.modules.items()):
        if module is None or name.startswith("face_os"):
            continue
        if not _project_module(module):
            continue
        for attr in ("cfg", "_CONFIG"):
            if hasattr(module, attr):
                saved.append((module, attr, getattr(module, attr)))
                setattr(module, attr, config)
        for attr in ("_ai_instance", "_ai"):
            if hasattr(module, attr):
                saved.append((module, attr, getattr(module, attr)))
                setattr(module, attr, None)
        if name == "automation.speech_alignment":
            saved.append((module, "_MODELS", module._MODELS))
            module._MODELS = {}
        if name == "automation.orchestrator":
            from automation.memory.decision_store import DecisionStore
            from automation.providers.provider_health import ProviderHealth
            for attr, replacement in (("_DECISION_STORE", DecisionStore()),
                                      ("_PROVIDER_HEALTH", ProviderHealth())):
                saved.append((module, attr, getattr(module, attr)))
                setattr(module, attr, replacement)
    previous_modules = set(sys.modules)
    cache = dict(utils_config._config_cache)
    utils_config._config_cache.clear()
    try:
        os.chdir(sandbox)
        result = function(*args, **kwargs)
        result["sandbox"] = str(sandbox)
        result["sandbox_retained"] = keep
        return result
    finally:
        os.chdir(original_cwd)
        for module, attr, value in saved:
            setattr(module, attr, value)
        for name in set(sys.modules) - previous_modules:
            module = sys.modules[name]
            if module is None or not _project_module(module):
                continue
            for attr in ("cfg", "_CONFIG"):
                value = getattr(module, attr, None)
                if isinstance(value, dict) and value.get("paths", {}).get("input") == config["paths"]["input"]:
                    setattr(module, attr, utils_config.Config(original))
            for attr in ("_ai_instance", "_ai"):
                if hasattr(module, attr):
                    setattr(module, attr, None)
            if name == "automation.speech_alignment":
                module._MODELS = {}  # newly imported models were created under ASR stubs
        utils_config._config_cache.clear()
        utils_config._config_cache.update(cache)


def _create_fake_clip(export_dir: str, stem: str):
    d = Path(export_dir)
    d.mkdir(parents=True, exist_ok=True)
    clip = d / f"{stem}.mp4"
    if not clip.exists():
        clip.write_bytes(b"C" * 1024 * 1024)
    return clip


def _make_valid_wav_bytes(num_samples: int = 16000) -> bytes:
    """Return a minimal valid 16-bit mono 16kHz WAV file."""
    import struct
    data_size = num_samples * 2  # 16-bit = 2 bytes per sample
    sample_rate = 16000
    fmt_data = struct.pack('<HHIIHH', 1, 1, sample_rate, sample_rate * 2, 2, 16)
    buf = bytearray()
    buf += b'RIFF'
    buf += struct.pack('<I', 36 + data_size)
    buf += b'WAVE'
    buf += b'fmt '
    buf += struct.pack('<I', 16)
    buf += fmt_data
    buf += b'data'
    buf += struct.pack('<I', data_size)
    buf += b'\x00' * data_size
    return bytes(buf)


# ── External-API intercept context ─────────────────────────────────────────

@contextlib.contextmanager
def _api_intercept_context(video_path: str):
    """Context manager that intercepts every external-API boundary call.

    Every ``unittest.mock.patch`` here targets the lowest point where our
    code touches an external dependency (network, subprocess, ML model, GPU,
    local binary).  All internal business logic runs unchanged.
    """
    # Real local cover bytes allow the uploader's actual image preflight and
    # CDN pixel verification to run, rather than stubbing their verdicts.
    from PIL import Image as _FixtureImage
    _real_image_open = _FixtureImage.open
    _jpeg_buffer = io.BytesIO()
    _FixtureImage.new("RGB", (1080, 1920), (25, 35, 55)).save(_jpeg_buffer, format="JPEG")
    _cover_fixture_bytes = _jpeg_buffer.getvalue()
    _fixture_thumb_url = "https://i.ytimg.com/vi/test_video_id/custom.jpg"
    _fixture_videos = {}

    # ── 1. AI / LLM providers ──────────────────────────────────────────────

    def _fake_generate_fastest_first(self, prompt="", system_instruction=None,
                                     prefer_provider=None, prefer_model=None):
        log.debug("  [FAKE AI] generate_fastest_first (prompt=%.60s...)",
                 prompt.replace("\n", " ")[:60])
        return _FAKE_SEO_JSON

    def _fake_generate_text(self, prompt="", system_instruction=None,
                            prefer_model=None, prefer_provider=None, **options):
        log.debug("  [FAKE AI] generate_text (prompt=%.60s...)",
                 prompt.replace("\n", " ")[:60])
        if "candidate_id" in prompt:
            return json.dumps({
                "selected": [
                    {"candidate_id": 1, "score": 90, "complete_thought": True, "reason": "retained setup and payoff"},
                    {"candidate_id": 2, "score": 85, "complete_thought": True, "reason": "retained setup and payoff"},
                ],
                "rejected": [],
                "notes": {"overall_quality": "high"},
            })
        return "Kohli hit a six. Bumrah bowled a yorker. Amazing catch."

    sdk_requests = []
    globals()["_editorial_sdk_requests"] = sdk_requests

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = self
            self.completions = self
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def create(self, **kwargs):
            from types import SimpleNamespace
            sdk_requests.append({key: kwargs.get(key) for key in
                                 ("model", "reasoning_effort", "extra_body", "max_tokens")})
            prompt = kwargs["messages"][-1]["content"]
            payload, _ = json.JSONDecoder().raw_decode(prompt)
            sdk_requests[-1]["stage"] = payload.get("stage")
            if payload.get("stage") == "discovery":
                proposals = []
                for cue in payload["cues"]:
                    if cue["cue_id"] not in (1, 2, 3, 4):
                        continue
                    tokens = cue["text"].split()
                    proposals.append({"start_cue_id": cue["cue_id"], "end_cue_id": cue["cue_id"],
                        "opening_phrase": " ".join(tokens[:3]), "closing_phrase": " ".join(tokens[-3:]),
                        "score": 95-cue["cue_id"], "reason": "Retained cricket setup and actual payoff"})
                response = {"proposals": proposals}
            elif payload.get("stage") == "boundary_review":
                cuts = []
                for c in payload["candidates"]:
                    a, b = c["original_source_range"]
                    ids = [w["word_id"] for w in c["words"] if w["start"] >= a and w["end"] <= b]
                    cuts.append({"candidate_id": c["candidate_id"], "first_word_id": ids[0],
                        "last_word_id": ids[-1], "reason": "Contiguous actual cricket setup and payoff"})
                response = {"cuts": cuts, "rejected": []}
            elif payload.get("stage") == "final_review":
                choices = [{"candidate_id": c["candidate_id"], "score": 95-c["candidate_id"],
                            "complete_thought": True, "reason": "Retained setup and cricket payoff"}
                           for c in payload["candidates"]]
                response = {"selected": choices[:3], "reserves": choices[3:6], "rejected": []}
            else:
                raise RuntimeError("Missing dry-run structured provider fixture")
            # A completed final object must follow an initially truncated
            # response; exercise real JSONResponseError -> bounded retry.
            finish_reason = "length" if len(sdk_requests) == 1 else "stop"
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish_reason,
                message=SimpleNamespace(content=json.dumps(response)))])

    def _fake_urlopen(request, **kwargs):
        """Run real caption schema/fidelity checks; substitute HTTP only."""
        if not getattr(request, "data", None):
            raise RuntimeError("Dry-run has no urllib fixture for this request")
        body = json.loads(request.data)
        payload = json.loads(body["messages"][-1]["content"])
        if body.get("model") != "deepseek-v4-flash" or "source_words" not in payload:
            raise RuntimeError("Dry-run refuses unrecognized urllib network access")
        entries = [{"id": word["id"], "roman": word["source"], "uncertain": False, "reason": ""}
                   for word in payload["source_words"]]
        return io.BytesIO(json.dumps({"model": "deepseek-v4-flash", "choices": [
            {"finish_reason": "stop", "message": {"content": json.dumps({"words": entries})}}]}).encode())

    def _fake_generate_seo_text(self, prompt="", system_instruction=None, reasoning_effort=None):
        log.debug("  [FAKE AI] generate_seo_text (prompt=%.60s...)",
                 prompt.replace("\n", " ")[:60])
        import re
        match = re.search(r"COMPLETE CLIP TRANSCRIPT:\s*(.*?)\n\nAPPROVED", prompt, re.DOTALL)
        script = match.group(1).strip() if match else "Kohli hits a six over the boundary."
        title = "Kohli hits a massive six" if "kohli" in script.lower() or "coaly" in script.lower() else script.split(".")[0][:60]
        return json.dumps({"title": title, "description": script + "\n\nIndependent cricket commentary and analysis, not official match footage.",
                           "cover_text": "KOHLI SIX" if "kohli" in title.lower() else "CRICKET MOMENT",
                           "hashtags": ["#Shorts", "#Cricket"], "tags": ["cricket commentary"],
                           "search_terms": [], "primary_search_terms": []})

    def _fake_generate_image(self, prompt="", **kwargs):
        log.debug("  [FAKE AI] generate_image (prompt=%.60s...)", prompt[:60])
        return b"JPEG" * 1024

    def _fake_caption_cards(cues, output_dir, *, style="neon-reference-v2"):
        from automation.caption_engine import CaptionCard
        if style not in ("neon-reference-v2", "portrait-glass-v1", "gameplay-hud-v1", "stadium-reference-v1"):
            raise ValueError("Unsupported fixture caption design")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cards = []
        for index, cue in enumerate(cues):
            path = Path(output_dir) / f"card-{index}.png"
            path.write_bytes(b"fake-caption-image")
            cards.append(CaptionCard(str(path), cue.start, cue.end, cue.text))
        return cards

        # ── 2. subprocess — intercept ffmpeg/ffprobe/yt-dlp/aria2c/nvidia-smi ──

    _original_subprocess_run = subprocess.run
    _original_subprocess_popen = subprocess.Popen
    _last_ffmpeg_t = [10.0]  # list to allow mutation in nested function
    _output_durations = {}
    _audio_ranges = {}

    def _fake_subprocess_run(cmd, *args, **kwargs):
        cmd_str = " ".join(str(c) for c in cmd) if isinstance(cmd, list) else str(cmd)
        low = cmd_str.lower()
        text_mode = kwargs.get("text", kwargs.get("universal_newlines", False))
        def _out(s):
            return s if text_mode else s.encode()
        def _empty():
            return "" if text_mode else b""
        # ffprobe: return contextual fake JSON
        if "ffprobe" in low:
            log.debug("  [FAKE ffprobe] %s", cmd_str[:100])
            text = cmd_str.lower()
            target = str(Path(str(cmd[-1])).resolve()) if isinstance(cmd, (list, tuple)) else ""
            is_source = target == str(Path(video_path).resolve()) or Path(target).name == "fixture.mp4"
            probe_w, probe_h = (1920, 1080) if is_source else (1080, 1920)
            if "-show_streams" in text:
                dur = _output_durations.get(target,180.0)
                fake = json.dumps({"streams":[
                    {"codec_type":"video","width":probe_w,"height":probe_h,"start_time":"0.0","duration":str(dur)},
                    {"codec_type":"audio","start_time":"0.0","duration":str(dur)}],"format":{"duration":str(dur)}})
            elif "stream=duration" in text:
                fake = json.dumps({"streams":[{"duration":str(_output_durations.get(target,180.0))}]})
            elif "format=duration" in text and "show_entries" in text:
                # If this looks like a clip validation file, use a small duration.
                # Otherwise fall back to the full video duration.
                target = str(Path(str(cmd[-1])).resolve()) if isinstance(cmd, (list, tuple)) else ""
                dur = _output_durations.get(target, 180.0)
                fake = json.dumps({"format": {"duration": str(dur)}})
            elif "r_frame_rate" in text:
                fake = json.dumps({"streams": [{"width": probe_w, "height": probe_h, "r_frame_rate": "30/1"}]})
            elif "stream=width,height" in text and "codec_type" not in text:
                fake = json.dumps({"streams": [{"width": probe_w, "height": probe_h}]})
            elif "stream=codec_type,start_time,duration" in text:
                target = str(Path(str(cmd[-1])).resolve())
                dur = _output_durations.get(target, 180.0)
                fake = json.dumps({"streams": [
                    {"codec_type": kind, "start_time": "0.0", "duration": str(dur)}
                    for kind in ("video", "audio")]})
            elif "codec_type" in text:
                fake = json.dumps({"streams": [{"codec_type": "audio"}]})
            else:
                fake = _FAKE_FFPROBE_OUTPUT
            return subprocess.CompletedProcess(cmd, 0, _out(fake), _empty())
        # ffmpeg: create output file at last position argument
        if "ffmpeg" in low:
            log.debug("  [FAKE ffmpeg] %s", cmd_str[:120])
            if "blackdetect=" in low:
                # External decoder boundary only: the real no-frame guard,
                # interval subtraction and selector still run. Synthetic
                # non-dark frames are not visual approval of actual media.
                return subprocess.CompletedProcess(cmd, 0, _empty(), _out("frame= 5400 time=00:03:00.00\n"))
            if "print_format=json" in low:
                return subprocess.CompletedProcess(cmd,0,_empty(),_out('{"input_i":"-16.0","input_tp":"-1.5","input_lra":"1.0","input_thresh":"-26.0","target_offset":"0.0"}'))
            # encoders list: return a real-looking list
            if "-encoders" in low:
                return subprocess.CompletedProcess(cmd, 0,
                    _out("Encoders:\n V..... libx264              H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (codec h264)\n"),
                    _empty())
            if "rawvideo" in low and "bgr24" in low and "-vf" in cmd:
                import re
                size = re.search(r"scale=(\d+):(\d+)", cmd[cmd.index("-vf")+1])
                if size:
                    # Decode/model boundary fixture; real face crop selection,
                    # motion-envelope planning and final filter routing still run.
                    w, h = map(int, size.groups())
                    return subprocess.CompletedProcess(cmd, 0, bytes([96])*(w*h*3), b"")
            # Capture -t flag value for later ffprobe duration matching
            command_duration = _last_ffmpeg_t[0]
            for i, c in enumerate(cmd):
                if isinstance(c, str) and c == "-t" and i + 1 < len(cmd):
                    try:
                        command_duration = float(cmd[i + 1])
                        _last_ffmpeg_t[0] = command_duration
                    except (ValueError, IndexError):
                        pass
            out_path = _fake_ffmpeg_output(cmd, Path.cwd())
            if out_path:
                import re
                duration_match = re.search(r"(?<!a)trim=start=[0-9.]+:duration=([0-9.]+)", cmd_str)
                speed_match = re.search(r"setpts=\(PTS-STARTPTS\)/([0-9.]+)", cmd_str)
                duration = float(duration_match.group(1)) if duration_match else command_duration
                if speed_match:
                    duration /= float(speed_match.group(1))
                _output_durations[str(out_path)] = duration
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if out_path.suffix == ".wav":
                    source_start = float(cmd[cmd.index("-ss")+1]) if "-ss" in cmd else 0
                    _audio_ranges[str(out_path)] = (source_start, source_start+duration)
                    out_path.write_bytes(_make_valid_wav_bytes(16000))
                else:
                    out_path.write_bytes(b"F" * 1024 * 500)
            return subprocess.CompletedProcess(cmd, 0, _empty(), _empty())
        # yt-dlp call: create fake video file
        if "yt-dlp" in low or ("yt" in low and "dlp" in low):
            log.debug("  [FAKE yt-dlp] %s", cmd_str[:100])
            _create_fake_video(video_path)
            # --dump-json needs real-looking metadata
            if "--dump-json" in low or "dump" in low:
                fake_meta = json.dumps({
                    "id": "4ylLhtICj1I", "title": "Cricket Highlights",
                    "duration": 180, "ext": "mp4",
                    "width": 1920, "height": 1080,
                })
                return subprocess.CompletedProcess(cmd, 0, _out(fake_meta), _empty())
            return subprocess.CompletedProcess(cmd, 0, _empty(), _empty())
        # aria2c: return version string (download.py checks for it)
        if "aria2c" in low:
            return subprocess.CompletedProcess(cmd, 0, _out("aria2c 1.36.0"), _empty())
        # nvidia-smi: return no-GPU
        if "nvidia-smi" in low:
            return subprocess.CompletedProcess(cmd, 0, _empty(), _empty())
        # apt-get: skip
        if "apt-get" in low or "apt " in low:
            return subprocess.CompletedProcess(cmd, 0, _empty(), _empty())
        # A missing boundary stub must fail, not run pip/scripts/network for real.
        raise RuntimeError(f"Dry-run has no safe subprocess stub for: {cmd_str[:120]}")

    class _FakePopen:
        def __init__(self, cmd, *args, **kwargs):
            cmd_str = " ".join(str(c) for c in cmd) if isinstance(cmd, list) else str(cmd)
            log.debug("  [FAKE Popen] %s", cmd_str[:100])
            if "yt-dlp" in cmd_str.lower():
                _create_fake_video(video_path)
            self.returncode = 0
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO("")
            self.args = cmd
        def communicate(self, input=None, timeout=None):
            return ("", "")
        def wait(self, timeout=None):
            return 0
        def poll(self):
            return 0
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    # ── 3. Google APIs — YouTube Data, Drive ───────────────────────────────

    def _fake_google_build(service_name, version, *args, **kwargs):
        log.debug("  [FAKE Google API] build %s %s", service_name, version)
        svc = MagicMock()
        if service_name == "youtube":
            def _fake_insert(**kwargs):
                def execute():
                    video_id = f"test_video_id_{len(_fixture_videos)+1}"
                    _fixture_videos[video_id] = copy.deepcopy(kwargs["body"]["status"])
                    return {"id": video_id}
                request = MagicMock()
                request.execute.side_effect = execute
                return request
            svc.videos().insert.side_effect = _fake_insert
            # thumbnails().set() → executed
            svc.thumbnails().set().execute.return_value = {"items": [{
                "shortsAspectRatio": {"url": _fixture_thumb_url, "width": 1080, "height": 1920}}]}
            def _fake_video_readback(**kwargs):
                video_id = kwargs["id"]
                item = {"id": video_id}
                if kwargs["part"] == "processingDetails":
                    item["processingDetails"] = {"processingStatus": "succeeded"}
                elif "status" in kwargs["part"]:
                    from utils.config import load_config
                    item["status"] = copy.deepcopy(_fixture_videos.get(video_id, {}))
                    item["snippet"] = {"channelId": load_config()["youtube"].get("channel_id")}
                else:
                    item["snippet"] = {"thumbnails": {"shortsAspectRatio": {
                        "url": _fixture_thumb_url, "width": 1080, "height": 1920}}}
                request = MagicMock()
                request.execute.return_value = {"items": [item]}
                return request
            svc.videos().list.side_effect = _fake_video_readback
            def _fake_update(**kwargs):
                def execute():
                    video_id = kwargs["body"]["id"]
                    if video_id not in _fixture_videos:
                        raise ValueError("Dry-run refuses status update for unknown video")
                    _fixture_videos[video_id] = copy.deepcopy(kwargs["body"]["status"])
                    return {"id": video_id, "status": copy.deepcopy(_fixture_videos[video_id])}
                request = MagicMock()
                request.execute.side_effect = execute
                return request
            svc.videos().update.side_effect = _fake_update
            # videoCategories().list() → assignable categories
            svc.videoCategories().list().execute.return_value = {
                "items": [{"id": "17", "snippet": {"assignable": True}},
                          {"id": "24", "snippet": {"assignable": True}}]
            }
            # captions().list() → empty (so transcript falls through to _fetch_via_api)
            svc.captions().list().execute.return_value = {"items": []}
            # search().list() → empty
            svc.search().list().execute.return_value = {"items": []}
        if service_name == "drive":
            svc.files().list().execute.return_value = {"files": []}
            svc.files().create().execute.return_value = {"id": "fake_file_id", "name": "dummy"}
            svc.files().get_media.return_value = MagicMock()
        if service_name == "youtubeAnalytics":
            svc.reports().query().execute.return_value = {
                "columnHeaders": [{"name": "video"}, {"name": "estimatedMinutesWatched"}],
                "rows": [],
            }
        return svc

    # ── 4. faster-whisper ─────────────────────────────────────────────────

    class _FakeWhisperInfo:
        duration = 180.0
        language = "en"
        duration_after_vad = None
        @property
        def language(self):
            return "en"

    class _FakeWhisperModel:
        def __init__(self, *args, **kwargs):
            log.debug("  [FAKE faster-whisper] WhisperModel.__init__ (skipping GPU)")
            self.model_name = args[0] if args else "fixture"
        def transcribe(self, audio, *args, **kwargs):
            log.debug("  [FAKE faster-whisper] model.transcribe(audio=%s)", str(audio)[:60])
            from types import SimpleNamespace
            offset, end = _audio_ranges.get(str(audio), (0, 180))
            output = []
            for cue in _FAKE_SEGMENTS:
                if cue["start"] < offset or cue["end"] > end:
                    continue
                tokens = cue["text"].split()
                if cue["start"] == 5:
                    # Caption/ASR wording mismatch must use a word-ID edit,
                    # not nearest-time approval or a hidden phrase substitution.
                    tokens[0] = "Aaj"
                step = (cue["end"]-cue["start"]-.2)/len(tokens)
                word_list = [SimpleNamespace(start=cue["start"]-offset+.1+i*step,
                    end=cue["start"]-offset+.1+(i+1)*step, word=token, probability=.99)
                    for i, token in enumerate(tokens)]
                if self.model_name == "medium" and word_list:
                    # Exercise the real zero-duration ASR regression and retry.
                    word_list[0].end = word_list[0].start
                output.append(SimpleNamespace(start=cue["start"]-offset,
                    end=cue["end"]-offset, text=cue["text"], words=word_list))
            return iter(output), _FakeWhisperInfo()

    class _FakeBatchedPipeline:
        def __init__(self, model=None, *args, **kwargs):
            self._model = model
        def transcribe(self, audio, *args, **kwargs):
            seg = MagicMock()
            seg.start = 0.0; seg.end = 5.0; seg.text = "Kohli hit a six"; seg.words = []
            return iter([seg]), _FakeWhisperInfo()

    # ── 5. youtube-transcript-api ──────────────────────────────────────────

    class _FakeTranscript:
        language_code = "en"
        def fetch(self):
            return [{"start": s["start"], "duration": s["end"] - s["start"], "text": s["text"]}
                    for s in _FAKE_SEGMENTS]

    class _FakeTranscriptList:
        def list(self, video_id):
            return self
        def find_transcript(self, langs):
            return _FakeTranscript()
        def find_generated_transcript(self, langs):
            return _FakeTranscript()
        def __iter__(self):
            return iter([_FakeTranscript()])
        def __next__(self):
            return _FakeTranscript()

    class _FakeYouTubeTranscriptApi:
        def __init__(self, http_client=None):
            pass
        def list(self, video_id):
            return _FakeTranscriptList()

    # ── 6. PIL ─────────────────────────────────────────────────────────────

    def _fake_pil_open(*args, **kwargs):
        if args and isinstance(args[0], io.BytesIO):
            return _real_image_open(*args, **kwargs)
        if args and isinstance(args[0], (str, Path)):
            candidate = Path(args[0])
            if candidate.is_file():
                with candidate.open("rb") as handle:
                    header = handle.read(8)
                    if header.startswith(b"\xff\xd8") or header == b"\x89PNG\r\n\x1a\n":
                        return _real_image_open(*args, **kwargs)
        img = MagicMock()
        img.width = 1080
        img.height = 1920
        img.mode = "RGB"
        img.size = (1080, 1920)
        img.convert.return_value = img
        img.resize.return_value = img
        img.crop.return_value = img
        img.__enter__.return_value = img
        def save(path, **options):
            target = Path(path).resolve()
            if not target.is_relative_to(Path.cwd()):
                raise ValueError("Fake image output escaped sandbox")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(_cover_fixture_bytes if target.suffix.lower() in (".jpg", ".jpeg") else b"fake-image")
        img.save.side_effect = save
        return img

    # ── 7. Pillow ImageEnhance ─────────────────────────────────────────────

    class _FakeEnhancer:
        def enhance(self, factor):
            return _fake_pil_open()

    def _fake_image_enhance(*args, **kwargs):
        return _FakeEnhancer()

    # ── 8. Pillow ImageFont ────────────────────────────────────────────────

    class _FakeFont:
        @staticmethod
        def truetype(*args, **kwargs):
            return _FakeFont()
        def getbbox(self, text):
            return (0, 0, len(text) * 10, 20)
        def getlength(self, text):
            return len(text) * 10
        def set_variation_by_axes(self, axes):
            pass

    # ── 9. Pillow ImageDraw ────────────────────────────────────────────────

    class _FakeDraw:
        def __init__(self, *args, **kwargs):
            pass
        def text(self, *args, **kwargs):
            pass
        def rectangle(self, *args, **kwargs):
            pass
        def rounded_rectangle(self, *args, **kwargs):
            pass
        def ellipse(self, *args, **kwargs):
            pass
        def line(self, *args, **kwargs):
            pass
        def textlength(self, text, **kwargs):
            return len(text) * 10
        def textbbox(self, *args, **kwargs):
            return (0, 0, 100, 20)
        def multiline_textbbox(self, *args, **kwargs):
            return (0, 0, 100, 20)

    # ── 10. Pillow ImageFilter ─────────────────────────────────────────────

    class _FakeFilter:
        def __init__(self, *args, **kwargs):
            pass

    # ── 11. requests — HTTP for trends ─────────────────────────────────────

    _fake_response = MagicMock()
    _fake_response.status_code = 200
    _fake_response.text = "<html>fake</html>"
    _fake_response.content = b"fake"
    _fake_response.headers = {"Content-Type": "image/jpeg"}
    _fake_response.iter_content.side_effect = lambda **_kw: iter([_cover_fixture_bytes])
    _fake_response.__enter__.return_value = _fake_response
    _fake_response.__exit__.return_value = False
    _fake_response.json.return_value = {"items": []}
    _fake_response.ok = True

    class _FakeSession:
        def __init__(self):
            self.headers = {}
            self.cookies = MagicMock()
        def get(self, *args, **kwargs):
            log.debug("  [FAKE HTTP] GET %s", str(args[0])[:80] if args else "")
            return _fake_response
        def post(self, *args, **kwargs):
            return _fake_response
        def mount(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    # ── 12. torch (CUDA query for export) ──────────────────────────────────

    def _fake_torch_cuda_device_count():
        return 0

    # ── 14. SuperResEnhancer ───────────────────────────────────────────────

    class _FakeSuperRes:
        def upscale_video(self, input_path, output_path, *args, **kwargs):
            log.debug("  [FAKE SuperRes] upscale %s -> %s", input_path, output_path)
            Path(output_path).write_bytes(Path(input_path).read_bytes())
            return output_path

    # ── 15. face_mapper / ref_grade / premium_analyzer ─────────────────────

    def _fake_face_map_video(*args, **kwargs):
        log.debug("  [FAKE face_mapper] map_video")
        return None

    def _fake_ref_grade(*args, **kwargs):
        log.debug("  [FAKE ref_grade] grade_video")
        return kwargs.get("output_path", args[-1] if args else "")

    # ── 16. upload._probe_video ────────────────────────────────────────────

    def _fake_probe_video(path):
        return {"width": 1080, "height": 1920, "duration": 60.0}

    # ── 17. OAuth token refresh ────────────────────────────────────────────

    from google.oauth2.credentials import Credentials as RealCredentials

    class _FakeCredentials(RealCredentials):
        scopes = ["https://www.googleapis.com/auth/youtube.force-ssl"]
        valid = True
        expired = False
        refresh_token = "fake_refresh"
        def __init__(self, *args, **kwargs):
            pass
        def refresh(self, request):
            pass
        def to_json(self):
            return '{"token": "fake"}'

    @classmethod
    def _fake_from_authorized_user_file(cls, *args, **kwargs):
        log.debug("  [FAKE OAuth] Credentials.from_authorized_user_file")
        return _FakeCredentials()

    @classmethod
    def _fake_from_authorized_user_info(cls, *args, **kwargs):
        log.debug("  [FAKE OAuth] Credentials.from_authorized_user_info")
        return _FakeCredentials()

    # ── 18. scheduler ──────────────────────────────────────────────────────

    def _fake_get_next_upload_time():
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

    # ── 19. Instagram Graph API ───────────────────────────────────────────

    globals()["_insta_stub_calls"] = {"count": 0}

    def _fake_process_instagram_for_clip(*args, **kwargs):
        log.debug("  [FAKE Instagram] process_instagram_for_clip")
        globals()["_insta_stub_calls"]["count"] += 1
        return None

    # ── Apply ALL patches ──────────────────────────────────────────────────

    _patches = [
        # AI providers
        patch("utils.ai_client.AIClient.generate_fastest_first", _fake_generate_fastest_first),
        patch("utils.ai_client.AIClient.generate_text", _fake_generate_text),
        patch("utils.ai_client.OpenAI", _FakeOpenAI),
        patch.dict(os.environ, {"DEEPSEEK_API_KEY": "dry-run-key-not-real"}),
        patch("urllib.request.urlopen", _fake_urlopen),
        patch("socket.create_connection", side_effect=RuntimeError("Dry-run blocked unstubbed network access")),
        patch("utils.ai_client.AIClient.generate_seo_text", _fake_generate_seo_text),
        patch("utils.ai_client.AIClient.generate_image", _fake_generate_image),
        patch("automation.caption_engine.convert_hindi_to_hinglish_llm", lambda texts, **_: texts),
        patch("automation.caption_engine.render_transparent_cards_batch", _fake_caption_cards),
        # Probe and caption rendering launch external browser processes. Stub
        # only that boundary; real rights/range/assets/runtime orchestration runs.
        patch("automation.stadium_layout._check_caption_browser", lambda: None),
        patch("frame_analyzer.find_host_in_frame", lambda frame, **kwargs: {
            "x": 700, "y": 260, "width": 400, "height": 650, "confidence": .99}),
        # subprocess intercept
        patch("subprocess.run", _fake_subprocess_run),
        patch("subprocess.check_output", _fake_subprocess_run),
        patch("subprocess.Popen", _FakePopen),
        # Google APIs
        patch("googleapiclient.discovery.build", _fake_google_build),
        # upload caches a from-import: warm processes need its binding patched too.
        patch("upload.build", _fake_google_build),
        # faster-whisper (module-level mock)
        patch("faster_whisper.WhisperModel", _FakeWhisperModel),
        patch("faster_whisper.BatchedInferencePipeline", _FakeBatchedPipeline),
        # youtube-transcript-api
        patch("youtube_transcript_api.YouTubeTranscriptApi", _FakeYouTubeTranscriptApi),
        # PIL
        patch("PIL.Image.open", _fake_pil_open),
        patch("PIL.Image.new", _fake_pil_open),
        patch("PIL.Image.alpha_composite", lambda *_: _fake_pil_open()),
        patch("PIL.ImageEnhance.Contrast", _fake_image_enhance),
        patch("PIL.ImageEnhance.Brightness", _fake_image_enhance),
        patch("PIL.ImageEnhance.Sharpness", _fake_image_enhance),
        patch("PIL.ImageDraw.Draw", _FakeDraw),
        patch("PIL.ImageFont.truetype", _FakeFont.truetype),
        patch("PIL.ImageFilter.GaussianBlur", _FakeFilter),
        patch("PIL.ImageStat.Stat", lambda *_: type("Stats", (), {"mean": [128.0], "var": [50.0]})()),
        # requests
        patch("requests.Session", _FakeSession),
        patch("requests.get", lambda *_a, **_kw: _fake_response),
        # OpenAI SDK (covered by AIClient patches above — the `chat` attribute
        # is a cached_property so string-path patching won't work here)
        # torch CUDA
        patch("torch.cuda.device_count", _fake_torch_cuda_device_count),
        # SuperResEnhancer
        patch("utils.super_res.SuperResEnhancer", _FakeSuperRes),
        # Upload probe
        patch("upload._probe_video", _fake_probe_video),
        # OAuth credential loading
        patch("google.oauth2.credentials.Credentials.from_authorized_user_file", _fake_from_authorized_user_file),
        patch("google.oauth2.credentials.Credentials.from_authorized_user_info", _fake_from_authorized_user_info),
        # scheduler
        patch("scheduler.get_next_upload_time", _fake_get_next_upload_time),
        # Instagram (no network — Graph API fully stubbed)
        patch("automation.instagram.runner.process_instagram_for_clip",
              _fake_process_instagram_for_clip),
    ]

    with contextlib.ExitStack() as stack:
        for p in _patches:
            stack.enter_context(p)
        yield


# ── Config validation ────────────────────────────────────────────────────────

def validate_config(cfg: dict) -> list[tuple]:
    results = []
    for path in REQUIRED_CONFIG_KEYS:
        node = cfg
        present = True
        for part in path:
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                present = False
                break
        results.append((".".join(path), present, node if present else None))
    return results


# ── Dry-run pipeline ─────────────────────────────────────────────────────────

@_isolated_run
def dry_run(url: str,
            skip_download=False, skip_transcribe=False,
            skip_highlight=False, skip_export=False,
            skip_sync=False, skip_seo=False,
            skip_enhancement=False,
            auto_sync=False, auto_upload=False,
            auto_schedule=False, sample_minutes=None,
            sync_from_drive=False, mode=None, composition=None) -> dict:
    """Execute the REAL pipeline with all external API calls intercepted."""

    from automation.config import load as load_config

    cap_buf = io.StringIO()
    plog = logging.getLogger("dry_run.pipeline")
    plog.handlers = [JsonStreamHandler(cap_buf)]
    plog.setLevel(logging.DEBUG)
    plog.propagate = False

    start = time.monotonic()
    result = PipelineResult()
    result.run_id = new_run_id()
    rid = result.run_id

    cfg = load_config()
    paths = cfg.get("paths", {})
    input_dir = paths.get("input", "input")
    output_filename = cfg.get("download", {}).get("output_filename", "video.mp4")
    video_path = str(Path(input_dir) / output_filename)

    log.info("=" * 64)
    log.info("DRY RUN — real pipeline, ONLY external API calls stubbed")
    log.info("URL: %s   run_id=%s", url, rid)
    log.info("Flags: skip_dl=%s skip_tx=%s skip_hl=%s skip_ex=%s skip_sync=%s "
             "skip_seo=%s sync=%s upload=%s schedule=%s mode=%s",
             skip_download, skip_transcribe, skip_highlight, skip_export,
             skip_sync, skip_seo, auto_sync, auto_upload, auto_schedule, mode)
    log.info("=" * 64)

    from automation.memory.decision_store import DecisionStore as _DS
    decision_store = _DS()
    decision_store.clear()
    event_count_before = 0

    validation = {
        "transcript_phase_ran": False,
        "transcript_corrected": False,
        "transcript_before": "", "transcript_after": "",
        "seo_marker_written": False, "seo_recovered": 0,
    }

    # ── Ensure input directory exists for fake video ──────────────────────
    Path(input_dir).mkdir(parents=True, exist_ok=True)
    _create_fake_video(video_path)
    temporary_transcript = _ensure_skipped_transcript_fixture(
        paths, video_path, skip_transcribe
    )

    # ── Run the REAL orchestrator with API patches ────────────────────────
    try:
        # Previous tests may have contaminated module-level cfg by importing
        # modules while load_config was mocked. Fix by clearing the config
        # cache and patching cfg on already-imported modules (without clearing
        # them from sys.modules, which would break test mock references).
        from automation.config import load as _load_real_config
        _real_cfg = _load_real_config()
        _ig = dict(_real_cfg.get("instagram") or {})
        _ig["enabled"] = True
        _real_cfg["instagram"] = _ig
        import sys as _sys
        for _m in ("transcribe", "highlight", "export", "download"):
            _mod = _sys.modules.get(_m)
            if _mod is not None and hasattr(_mod, "cfg"):
                _mod.cfg = _real_cfg
        with _api_intercept_context(video_path):
            import automation.orchestrator as orchestrator
            decision_store = orchestrator._DECISION_STORE
            event_count_before = decision_store.count()
            handler = JsonStreamHandler(cap_buf)
            orchestrator.log.addHandler(handler)
            try:
                pipeline_result = orchestrator.run(
                    url=url, skip_download=skip_download, skip_transcribe=skip_transcribe,
                    skip_highlight=skip_highlight, skip_export=skip_export,
                    skip_sync=skip_sync, skip_seo=skip_seo, skip_enhancement=skip_enhancement,
                    auto_sync=auto_sync, auto_upload=auto_upload, auto_schedule=auto_schedule,
                    sample_minutes=sample_minutes, sync_from_drive=sync_from_drive, mode=mode,
                )
                rid = pipeline_result.run_id
            finally:
                orchestrator.log.removeHandler(handler)
    except Exception as e:
        log.error("Pipeline crashed: %s", e, exc_info=True)
        result.failures.append(f"pipeline_crash: {e}")
        pipeline_result = type('_', (), {
            'exported': [], 'uploaded_count': 0, 'failures': [str(e)],
            'transcript_source': 'none', 'selected_clips': 0, 'seo_generated': 0,
        })()
    finally:
        if temporary_transcript is not None:
            temporary_transcript.unlink(missing_ok=True)

    # ── Gather results ─────────────────────────────────────────────────────
    result.exported = pipeline_result.exported if hasattr(pipeline_result, 'exported') else []
    result.uploaded_count = getattr(pipeline_result, 'uploaded_count', 0)
    result.selected_clips = getattr(pipeline_result, 'selected_clips', 0)
    result.seo_generated = getattr(pipeline_result, 'seo_generated', 0)
    result.transcript_source = getattr(pipeline_result, 'transcript_source', 'none')
    pipe_failures = getattr(pipeline_result, 'failures', [])
    result.failures.extend(f for f in pipe_failures if f not in result.failures)
    if not skip_export and not result.exported:
        result.failures.append("acceptance: expected fixture exports, produced zero")
    if not skip_export and len(result.exported) != result.selected_clips:
        result.failures.append("acceptance: selected/exported count mismatch")
    if not skip_export and not skip_seo and result.seo_generated != len(result.exported):
        result.failures.append("acceptance: exported/valid metadata count mismatch")

    # Check actual artifacts too: a successful stage log or counter is not a
    # cover. Metadata failures are reported separately, not double-counted here.
    from automation.artifacts import read_metadata
    thumbnail_clips = [Path(clip) for clip in result.exported if read_metadata(Path(clip)) is not None]
    thumbnails_enabled = cfg.get("thumbnail", {}).get("enabled", True)
    missing_covers = []
    if thumbnails_enabled:
        for clip in thumbnail_clips:
            cover = clip.with_name(f"{clip.stem}_thumb.jpg")
            if not cover.is_file() or cover.stat().st_size == 0:
                missing_covers.append(clip.name)
    validation.update(
        thumbnails_expected=len(thumbnail_clips) if thumbnails_enabled else 0,
        thumbnails_generated=getattr(pipeline_result, "thumbnails_generated", 0),
        thumbnails_preserved=getattr(pipeline_result, "thumbnails_preserved", 0),
        thumbnails_missing=len(missing_covers),
    )
    if missing_covers:
        result.failures.append("acceptance: missing thumbnails: " + ", ".join(missing_covers))

    # Independently verify actual word evidence and final edit ranges. Neither
    # an LLM attestation, stage tick nor a nonempty fake MP4 can satisfy this.
    from automation.speech_alignment import alignment_issues, words_in_range
    word_aligned = 0
    portrait_exports = 0
    stadium_exports = 0
    for clip in result.exported:
        try:
            speech = json.loads(clip.with_name(f"{clip.stem}_speech.json").read_text(encoding="utf-8"))
            edit = json.loads(clip.with_name(f"{clip.stem}_edit.json").read_text(encoding="utf-8"))
            start_s, end_s = speech["source_range"]
            if alignment_issues(speech["segments"], start_s, end_s):
                raise ValueError("invalid source word evidence")
            if [edit["source_start"], edit["source_end"]] != [start_s, end_s]:
                raise ValueError("speech/edit ranges disagree")
            retained = " ".join(str(w["word"]).strip() for w in words_in_range(speech["segments"], start_s, end_s))
            if edit["text"] != retained:
                raise ValueError("edit script is not retained speech")
            word_aligned += 1
            layout = edit.get("portrait_layout")
            if (isinstance(layout, dict) and layout["picture_width"] == cfg["export"]["width"]
                    and layout["face_bottom"]+60 <= layout["caption_top"]):
                portrait_exports += 1
            if cfg["export"].get("composition") == "cricket-gameplay-stadium-v1":
                from automation.cricket_gameplay import sha256, credit_text
                from automation.stadium_layout import HOST, GAMEPLAY, OVERLAY_HASH, FONT_HASH
                gameplay = json.loads(clip.with_name(f"{clip.stem}_gameplay.json").read_text(encoding="utf-8"))
                if (edit.get("composition") != "cricket-gameplay-stadium-v1"
                        or gameplay["media_sha256"] != sha256(clip)
                        or gameplay["top"] != HOST or gameplay["bottom"] != GAMEPLAY
                        or gameplay["caption_style"] != "stadium-reference-v1"
                        or gameplay["design_assets"]["overlay"]["sha256"] != OVERLAY_HASH
                        or gameplay["design_assets"]["font"]["sha256"] != FONT_HASH
                        or gameplay["gameplay_crop"] != gameplay["selection"]["asset"]["crop"]):
                    raise ValueError("stadium layout/crop/asset evidence mismatch")
                metadata = read_metadata(clip)
                if not skip_seo and credit_text(gameplay["selection"]) not in metadata.get("description", ""):
                    raise ValueError("gameplay attribution missing after SEO")
                stadium_exports += 1
        except (OSError, ValueError, TypeError, KeyError) as exc:
            result.failures.append(f"acceptance: word alignment {clip.name}: {exc}")
    validation["word_aligned_exports"] = word_aligned
    validation["portrait_exports"] = portrait_exports
    validation["stadium_exports"] = stadium_exports
    sdk_calls = globals().get("_editorial_sdk_requests", [])
    validation["editorial_high_requests"] = sum(r["reasoning_effort"] == "high" and
        r["extra_body"] == {"thinking": {"type": "enabled"}} for r in sdk_calls)
    validation["boundary_review_requests"] = sum(r.get("stage") == "boundary_review" for r in sdk_calls)
    if not skip_export and not skip_highlight and cfg.get("clip_selection", {}).get("editorial", {}).get("enabled"):
        if cfg["export"].get("composition") == "portrait-caption-v1" and portrait_exports != len(result.exported):
            result.failures.append("acceptance: fixture exports did not use full-width face-safe portrait layout")
        if cfg["export"].get("composition") == "cricket-gameplay-stadium-v1" and stadium_exports != len(result.exported):
            result.failures.append("acceptance: fixture exports did not use approved stadium layout and gameplay evidence")
        if not 2 <= len(result.exported) <= 3:
            result.failures.append("acceptance: fixture must produce 2-3 complete drafts")
        if validation["editorial_high_requests"] < 2:
            result.failures.append("acceptance: discovery and final review did not use explicit high thinking")
        if validation["boundary_review_requests"] < 1:
            result.failures.append("acceptance: fixture phrase mismatch did not exercise word-ID boundary review")

    # ── Post-pipeline transcript correction validation ─────────────────────
    try:
        from automation.transcript import TRANSCRIPT_CACHE
        if not skip_transcribe:
            cached = TRANSCRIPT_CACHE.get(Path(url).stem or url)
            if cached and cached.get("segments"):
                segs = cached["segments"]
                if len(segs) >= 2:
                    before = segs[1]["text"]
                    from utils.transcript_postproc import correct_segments
                    segs_corrected, _n = correct_segments(segs)
                    after = segs_corrected[1]["text"]
                    validation["transcript_before"] = before
                    validation["transcript_after"] = after
                    validation["transcript_corrected"] = (
                        "Kohli" in after and "coaly" not in after.lower()
                    )
                    validation["transcript_phase_ran"] = True
    except Exception as e:
        log.warning("Transcript validation skipped: %s", e)

    # ── SEO retry-queue validation ─────────────────────────────────────────
    shorts_dir = paths.get("shorts", "shorts")
    if Path(shorts_dir).exists():
        markers = list(Path(shorts_dir).rglob("*_seo_failed.json"))
        validation["seo_marker_written"] = len(markers) > 0
        recovered = 0
        for m in markers:
            if not m.exists():
                recovered += 1
        validation["seo_recovered"] = recovered

    result.total_seconds = time.monotonic() - start
    status = "partial" if result.failures else "ok"
    log.info("[EXIT] run_id=%s exported=%d uploaded=%d selected=%d seo=%d "
             "failures=%d elapsed=%.2fs transcript=%s",
             rid, len(result.exported), result.uploaded_count,
             result.selected_clips, result.seo_generated,
             len(result.failures), result.total_seconds, result.transcript_source)

    struct_records = []
    for line in cap_buf.getvalue().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            struct_records.append(json.loads(line))
        except Exception:
            pass
    logged_stages = sorted({r.get("stage") for r in struct_records
                             if r.get("run_id") == rid and r.get("stage")
                             and r.get("status") != "skipped"})
    skipped_stages = sorted({r.get("stage") for r in struct_records
                              if r.get("run_id") == rid and r.get("status") == "skipped"})
    if not struct_records:
        result.failures.append("acceptance: no structured pipeline records captured")

    prompts_ok = True
    try:
        import prompts as _p
        for name in REQUIRED_PROMPTS:
            if not hasattr(_p, name):
                prompts_ok = False
                log.warning("Missing prompt: %s", name)
    except Exception as e:
        prompts_ok = False
        log.warning("Prompts module import failed: %s", e)

    event_count_new = decision_store.count() - event_count_before

    return {
        "run_id": rid,
        "exported": [str(p) for p in result.exported],
        "uploaded_count": result.uploaded_count,
        "selected_clips": result.selected_clips,
        "seo_generated": result.seo_generated,
        "thumbnails_generated": validation["thumbnails_generated"],
        "thumbnails_preserved": validation["thumbnails_preserved"],
        "failures": result.failures,
        "total_seconds": result.total_seconds,
        "transcript_source": result.transcript_source,
        "config_checks": validate_config(cfg),
        "struct_records": len(struct_records),
        "logged_stages": logged_stages,
        "skipped_stages": skipped_stages,
        "validation": validation,
        "prompts_ok": prompts_ok,
        "event_count": event_count_new,
        "event_types": sorted({event.event_type.value for event in decision_store.get_all_events()}),
        "export_expected": not skip_export,
        "instagram_expected": not skip_export and not skip_seo and os.environ.get("YT_CLIPS_SKIP_INSTAGRAM") != "1",
    }


def list_external_calls():
    calls = [
        ("subprocess.run (ffmpeg/ffprobe/yt-dlp)", "media processing & download"),
        ("subprocess.Popen (yt-dlp via download.py)", "video download"),
        ("googleapiclient.discovery.build", "YouTube Data API + Drive API"),
        ("utils.ai_client.AIClient.*", "all AI providers (OpenCode, Groq, etc.)"),
        ("faster_whisper.WhisperModel", "local GPU transcription"),
        ("youtube_transcript_api.YouTubeTranscriptApi", "YouTube transcript fetch"),
        ("PIL.Image / ImageDraw / ImageFont", "thumbnail compositing"),
        ("requests.Session", "HTTP trends (cricbuzz, google trends, etc.)"),
        ("openai.OpenAI chat.completions.create", "AI provider HTTP fallback"),
        ("torch.cuda.device_count", "CUDA device query"),
        ("utils.super_res.SuperResEnhancer", "GFPGAN / Real-ESRGAN upscaling"),
        ("upload._probe_video", "ffprobe video metadata"),
        ("google.oauth2.credentials.Credentials", "OAuth token loading/refresh"),
        ("scheduler.get_next_upload_time", "upload scheduling"),
    ]
    print("\nIntercepted External API Calls:")
    print("=" * 70)
    for name, desc in calls:
        print(f"  {name:<50s} {desc}")
    print()


def _print_validation_report(result: dict):
    cfg_checks = result["config_checks"]
    cfg_ok = sum(1 for _, ok, _ in cfg_checks if ok)
    v = result["validation"]
    expected_stages = set(EXPECTED_PHASE_STAGES)
    logged = set(result["logged_stages"])

    print()
    print("=" * 70)
    print("DRY RUN COMPLETE — VALIDATION REPORT")
    print("=" * 70)
    print(f"  run_id:             {result['run_id']}")
    print(f"  Exported:           {len(result['exported'])} clips")
    print(f"  Uploaded:           {result['uploaded_count']}")
    print(f"  Selected clips:     {result.get('selected_clips', 0)}")
    print(f"  SEO generated:      {result.get('seo_generated', 0)}")
    print(f"  Failures:           {len(result['failures'])}")
    print(f"  Elapsed:            {result['total_seconds']:.2f}s")
    print(f"  Transcript source:  {result['transcript_source']}")
    print()
    print("  [1] Config tunables (overhaul):")
    for key, ok, val in cfg_checks:
        mark = "PASS" if ok else "FAIL"
        print(f"        [{mark}] {key} = {val}")
    print(f"      -> {cfg_ok}/{len(cfg_checks)} present")
    print()
    print("  [2] Observability (run_phase + run_id):")
    print(f"        run_id generated:        {'PASS' if result['run_id'] else 'FAIL'}")
    print(f"        structured records:      {result['struct_records']}")
    print(f"        stages logged:           {sorted(logged)}")
    print(f"        explicitly skipped:      {result.get('skipped_stages', [])}")
    print()
    print("  [3] Transcript correction (utils.transcript_postproc):")
    if not v["transcript_phase_ran"]:
        print("        N/A — transcript phase skipped this run")
        transcript_ok = True
    else:
        print(f"        before: {v['transcript_before']!r}")
        print(f"        after:  {v['transcript_after']!r}")
        print(f"        corrected coaly->Kohli:  {'PASS' if v['transcript_corrected'] else 'FAIL'}")
        transcript_ok = v["transcript_corrected"]
    print()
    print("  [4] SEO escalate-not-degrade retry queue:")
    print(f"        failure marker written:  {'PASS' if v['seo_marker_written'] else 'n/a'}")
    print(f"        recovered on retry:      {v['seo_recovered']}")
    print()
    print("  [5] Prompts module:")
    prompts_ok = result.get("prompts_ok", False)
    print(f"        all templates present: {'PASS' if prompts_ok else 'FAIL'}")
    print()
    missing = expected_stages - logged
    print("  [6] Phase coverage:")
    print(f"        expected stages present: {sorted(expected_stages & logged)}")
    if missing:
        print(f"        not run (flag-dependent):{sorted(missing)}")
    print()
    print("  [7] Event store (automation.decision_store):")
    event_count = result.get("event_count", 0)
    print(f"        events emitted:             {event_count}")
    if event_count:
        print(f"        event types:               {result.get('event_types', [])}")
    print()
    insta_gate_ok = not result.get("instagram_expected", False) or _insta_stub_calls["count"] > 0
    print("  [8] Instagram (stubbed, zero network):")
    print(f"        INSTA(STUB): "
          f"{'PASS' if insta_gate_ok else 'FAIL — stub never invoked'} "
          f"(calls={_insta_stub_calls['count']})")
    print()
    print("  [9] Thumbnail artifacts (stubbed media; not visual approval):")
    print(f"        expected/generated/preserved/missing: "
          f"{v.get('thumbnails_expected', 0)}/{v.get('thumbnails_generated', 0)}/"
           f"{v.get('thumbnails_preserved', 0)}/{v.get('thumbnails_missing', 0)}")
    print()
    print("  [10] Editorial preflight (real validation, fixture ASR/HTTP):")
    print(f"        exports with matching word/edit evidence: {v.get('word_aligned_exports', 0)}")
    print(f"        explicit high-thinking SDK requests:     {v.get('editorial_high_requests', 0)}")
    print(f"        word-ID boundary review requests:        {v.get('boundary_review_requests', 0)}")
    print(f"        full-width face-safe portrait exports:   {v.get('portrait_exports', 0)}")
    print(f"        approved stadium/evidence exports:       {v.get('stadium_exports', 0)}")
    print()
    if result["failures"]:
        print("  Failures detail:")
        for f in result["failures"]:
            print(f"    - {f}")
    export_ok = not result.get("export_expected", True) or bool(result["exported"])
    ok = (cfg_ok == len(cfg_checks) and result["run_id"] and export_ok and
          transcript_ok and prompts_ok and insta_gate_ok and
          not v.get("thumbnails_missing", 0) and not result["failures"])
    print("=" * 70)
    print(f"  OVERALL: {'PASS ✅' if ok else 'CHECK ⚠'}")
    print("=" * 70)
    print()
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Dry-run pipeline — real internal code, only external API calls stubbed.")
    parser.add_argument("url", nargs="?", default="https://youtu.be/4ylLhtICj1I")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-transcribe", action="store_true")
    parser.add_argument("--skip-highlight", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-sync", action="store_true")
    parser.add_argument("--skip-seo", action="store_true")
    parser.add_argument("--skip-enhancement", action="store_true")
    parser.add_argument(
        "--skip-instagram", action="store_true",
        help="Skip Instagram Reel publish (sets YT_CLIPS_SKIP_INSTAGRAM=1)")
    parser.add_argument("--sync", action="store_true", dest="auto_sync")
    parser.add_argument("--upload", action="store_true", dest="auto_upload")
    parser.add_argument("--schedule", action="store_true", dest="auto_schedule")
    parser.add_argument("--sample-minutes", type=int, default=None)
    parser.add_argument("--sync-from-drive", action="store_true")
    parser.add_argument("--mode", choices=["face_mapper", "ref_grade"], default=None)
    parser.add_argument("--composition", choices=["portrait-caption-v1", "cricket-gameplay-split-v1", "cricket-gameplay-stadium-v1"], default=None)
    parser.add_argument("--list-external", action="store_true")
    parser.add_argument("--validate-config", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Show stub calls and detailed diagnostics")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        from utils.logger import configure_console
        configure_console(verbose=True)

    if args.skip_instagram:
        os.environ["YT_CLIPS_SKIP_INSTAGRAM"] = "1"

    if args.list_external:
        list_external_calls()
        return

    if args.validate_config:
        from automation.config import load as load_config
        checks = validate_config(load_config())
        print("\nConfig validation:")
        print("=" * 60)
        all_ok = True
        for key, ok, val in checks:
            all_ok = all_ok and ok
            print(f"  [{'PASS' if ok else 'FAIL'}] {key} = {val}")
        print("=" * 60)
        print(f"  {'ALL PRESENT ✅' if all_ok else 'MISSING KEYS ⚠'}\n")
        sys.exit(0 if all_ok else 1)

    result = dry_run(
        url=args.url,
        skip_download=args.skip_download,
        skip_transcribe=args.skip_transcribe,
        skip_highlight=args.skip_highlight,
        skip_export=args.skip_export,
        skip_sync=args.skip_sync,
        skip_seo=args.skip_seo,
        skip_enhancement=args.skip_enhancement,
        auto_sync=args.auto_sync,
        auto_upload=args.auto_upload,
        auto_schedule=args.auto_schedule,
        sample_minutes=args.sample_minutes,
        sync_from_drive=args.sync_from_drive,
        mode=args.mode,
        composition=args.composition,
    )

    ok = _print_validation_report(result)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
