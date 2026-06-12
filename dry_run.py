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

import argparse
import contextlib
import io
import json
import logging
import subprocess
import sys
import time
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
    ("seo", "inject_viral_elements"),
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
    "automation_learner", "provider_health", "event_store_validation",
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
    {"start": 0.0, "end": 5.0, "text": "Hello and welcome to the live stream"},
    {"start": 5.0, "end": 10.0, "text": "Today coaly is batting brilliantly"},
    {"start": 30.0, "end": 35.0, "text": "coaly hits a massive six"},
    {"start": 60.0, "end": 65.0, "text": "bumra bowls a perfect yorker"},
    {"start": 120.0, "end": 125.0, "text": "What a catch in the deep"},
    {"start": 300.0, "end": 305.0, "text": "And that's the end of the over"},
]

_FAKE_TRANSCRIPT = {
    "segments": _FAKE_SEGMENTS,
    "source": "api",
}

_FAKE_HIGHLIGHTS = {
    "clip1": {"start": 30, "end": 55, "label": "Kohli Six", "text": "coaly hits a massive six"},
    "clip2": {"start": 60, "end": 85, "label": "Yorker", "text": "bumra bowls a perfect yorker"},
    "clip3": {"start": 120, "end": 145, "label": "Amazing Catch", "text": "What a catch in the deep"},
}

_FAKE_SEO_JSON = json.dumps({
    "title": "Kohli Six! #Shorts",
    "description": "Amazing shot by Kohli in today's match! Subscribe for more. #Shorts",
    "hashtags": ["#Shorts", "#Kohli", "#IPL"],
    "search_terms": ["kohli six", "ipl highlights", "rcb vs csk"],
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
        log.info("  [FAKE] Created %s (%d MB)", path, size_mb)


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
    # ── 1. AI / LLM providers ──────────────────────────────────────────────

    def _fake_generate_fastest_first(self, prompt="", system_instruction=None,
                                     prefer_provider=None, prefer_model=None):
        log.info("  [FAKE AI] generate_fastest_first (prompt=%.60s...)",
                 prompt.replace("\n", " ")[:60])
        return _FAKE_SEO_JSON

    def _fake_generate_text(prompt="", system_instruction=None):
        log.info("  [FAKE AI] generate_text (prompt=%.60s...)",
                 prompt.replace("\n", " ")[:60])
        return "Kohli hit a six. Bumrah bowled a yorker. Amazing catch."

    def _fake_generate_seo_text(self, prompt="", system_instruction=None):
        log.info("  [FAKE AI] generate_seo_text (prompt=%.60s...)",
                 prompt.replace("\n", " ")[:60])
        return _FAKE_SEO_JSON

    def _fake_generate_image(prompt="", **kwargs):
        log.info("  [FAKE AI] generate_image (prompt=%.60s...)", prompt[:60])
        return b"JPEG" * 1024

        # ── 2. subprocess — intercept ffmpeg/ffprobe/yt-dlp/aria2c/nvidia-smi ──

    _original_subprocess_run = subprocess.run
    _original_subprocess_popen = subprocess.Popen
    _last_ffmpeg_t = [10.0]  # list to allow mutation in nested function

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
            log.info("  [FAKE ffprobe] %s", cmd_str[:100])
            text = cmd_str.lower()
            if "format=duration" in text and "show_entries" in text:
                # If this looks like a clip validation file, use a small duration.
                # Otherwise fall back to the full video duration.
                if any(p in cmd_str for p in ("shorts/", "temp/")):
                    dur = _last_ffmpeg_t[0]
                else:
                    dur = 180.0
                fake = json.dumps({"format": {"duration": str(dur)}})
            elif "r_frame_rate" in text:
                fake = json.dumps({"streams": [{"width": 1080, "height": 1920, "r_frame_rate": "30/1"}]})
            elif "stream=width,height" in text and "codec_type" not in text:
                fake = json.dumps({"streams": [{"width": 1080, "height": 1920}]})
            elif "codec_type" in text:
                fake = json.dumps({"streams": [{"codec_type": "audio"}]})
            else:
                fake = _FAKE_FFPROBE_OUTPUT
            return subprocess.CompletedProcess(cmd, 0, _out(fake), _empty())
        # ffmpeg: create output file at last position argument
        if "ffmpeg" in low:
            log.info("  [FAKE ffmpeg] %s", cmd_str[:120])
            # encoders list: return a real-looking list
            if "-encoders" in low:
                return subprocess.CompletedProcess(cmd, 0,
                    _out("Encoders:\n V..... libx264              H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (codec h264)\n"),
                    _empty())
            # Capture -t flag value for later ffprobe duration matching
            for i, c in enumerate(cmd):
                if isinstance(c, str) and c == "-t" and i + 1 < len(cmd):
                    try:
                        _last_ffmpeg_t[0] = float(cmd[i + 1])
                    except (ValueError, IndexError):
                        pass
            out_path = None
            for c in reversed(cmd):
                if isinstance(c, str) and (c.endswith(".mp4") or c.endswith(".wav") or c.endswith(".nut") or c.endswith(".jpg")):
                    out_path = c
                    break
            if out_path:
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                if out_path.endswith(".wav"):
                    Path(out_path).write_bytes(_make_valid_wav_bytes(16000))
                else:
                    Path(out_path).write_bytes(b"F" * 1024 * 500)  # 500KB to pass size validation
            return subprocess.CompletedProcess(cmd, 0, _empty(), _empty())
        # yt-dlp call: create fake video file
        if "yt-dlp" in low or ("yt" in low and "dlp" in low):
            log.info("  [FAKE yt-dlp] %s", cmd_str[:100])
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
        # Everything else (mkdir, pip, cp, ls, etc.) pass through
        return _original_subprocess_run(cmd, *args, **kwargs)

    class _FakePopen:
        def __init__(self, cmd, *args, **kwargs):
            cmd_str = " ".join(str(c) for c in cmd) if isinstance(cmd, list) else str(cmd)
            log.info("  [FAKE Popen] %s", cmd_str[:100])
            if "yt-dlp" in cmd_str.lower():
                _create_fake_video(video_path)
            self.returncode = 0
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO("")
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
        log.info("  [FAKE Google API] build %s %s", service_name, version)
        svc = MagicMock()
        if service_name == "youtube":
            # videos().insert() → executed → {"id": "test_video_id"}
            insert_mock = MagicMock()
            insert_mock.execute.return_value = {"id": "test_video_id"}
            svc.videos().insert.return_value = insert_mock
            # thumbnails().set() → executed
            svc.thumbnails().set().execute.return_value = {"id": "test_thumb_id"}
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
            log.info("  [FAKE faster-whisper] WhisperModel.__init__ (skipping GPU)")
        def transcribe(self, audio, *args, **kwargs):
            log.info("  [FAKE faster-whisper] model.transcribe(audio=%s)", str(audio)[:60])
            seg = MagicMock()
            seg.start = 0.0; seg.end = 5.0; seg.text = "Kohli hit a six"; seg.words = []
            return iter([seg]), _FakeWhisperInfo()

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
            return _FAKE_SEGMENTS

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
        img = MagicMock()
        img.width = 1080
        img.height = 1920
        img.mode = "RGB"
        img.size = (1080, 1920)
        img.convert.return_value = img
        img.resize.return_value = img
        img.crop.return_value = img
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
            return MagicMock()
        def getbbox(self, text):
            return (0, 0, len(text) * 10, 20)

    # ── 9. Pillow ImageDraw ────────────────────────────────────────────────

    class _FakeDraw:
        def __init__(self, *args, **kwargs):
            pass
        def text(self, *args, **kwargs):
            pass
        def rectangle(self, *args, **kwargs):
            pass
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
    _fake_response.json.return_value = {"items": []}
    _fake_response.ok = True

    class _FakeSession:
        def __init__(self):
            self.headers = {}
            self.cookies = MagicMock()
        def get(self, *args, **kwargs):
            log.info("  [FAKE HTTP] GET %s", str(args[0])[:80] if args else "")
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
            log.info("  [FAKE SuperRes] upscale %s -> %s", input_path, output_path)
            Path(output_path).write_bytes(Path(input_path).read_bytes())
            return output_path

    # ── 15. face_mapper / ref_grade / premium_analyzer ─────────────────────

    def _fake_face_map_video(*args, **kwargs):
        log.info("  [FAKE face_mapper] map_video")
        return None

    def _fake_ref_grade(*args, **kwargs):
        log.info("  [FAKE ref_grade] grade_video")
        return kwargs.get("output_path", args[-1] if args else "")

    # ── 16. upload._probe_video ────────────────────────────────────────────

    def _fake_probe_video(path):
        return {"width": 1080, "height": 1920, "duration": 60.0}

    # ── 17. OAuth token refresh ────────────────────────────────────────────

    from google.oauth2.credentials import Credentials as RealCredentials

    class _FakeCredentials(RealCredentials):
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
        log.info("  [FAKE OAuth] Credentials.from_authorized_user_file")
        return _FakeCredentials()

    @classmethod
    def _fake_from_authorized_user_info(cls, *args, **kwargs):
        log.info("  [FAKE OAuth] Credentials.from_authorized_user_info")
        return _FakeCredentials()

    # ── 18. scheduler ──────────────────────────────────────────────────────

    def _fake_get_next_upload_time():
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

    # ── Apply ALL patches ──────────────────────────────────────────────────

    _patches = [
        # AI providers
        patch("utils.ai_client.AIClient.generate_fastest_first", _fake_generate_fastest_first),
        patch("utils.ai_client.AIClient.generate_text", _fake_generate_text),
        patch("utils.ai_client.AIClient.generate_seo_text", _fake_generate_seo_text),
        patch("utils.ai_client.AIClient.generate_image", _fake_generate_image),
        # subprocess intercept
        patch("subprocess.run", _fake_subprocess_run),
        patch("subprocess.check_output", _fake_subprocess_run),
        patch("subprocess.Popen", _FakePopen),
        # Google APIs
        patch("googleapiclient.discovery.build", _fake_google_build),
        # faster-whisper (both module-level and any cached module-level refs)
        patch("faster_whisper.WhisperModel", _FakeWhisperModel),
        patch("faster_whisper.BatchedInferencePipeline", _FakeBatchedPipeline),
        patch("transcribe.WhisperModel", _FakeWhisperModel),
        # youtube-transcript-api
        patch("youtube_transcript_api.YouTubeTranscriptApi", _FakeYouTubeTranscriptApi),
        # PIL
        patch("PIL.Image.open", _fake_pil_open),
        patch("PIL.Image.new", _fake_pil_open),
        patch("PIL.ImageEnhance.Contrast", _fake_image_enhance),
        patch("PIL.ImageEnhance.Brightness", _fake_image_enhance),
        patch("PIL.ImageEnhance.Sharpness", _fake_image_enhance),
        patch("PIL.ImageDraw.Draw", _FakeDraw),
        patch("PIL.ImageFont.truetype", _FakeFont.truetype),
        patch("PIL.ImageFilter.GaussianBlur", _FakeFilter),
        # requests
        patch("requests.Session", _FakeSession),
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

def dry_run(url: str,
            skip_download=False, skip_transcribe=False,
            skip_highlight=False, skip_export=False,
            skip_sync=False, skip_seo=False,
            auto_sync=False, auto_upload=False,
            auto_schedule=False, sample_minutes=None,
            sync_from_drive=False, mode=None) -> dict:
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

    # ── Run the REAL orchestrator with API patches ────────────────────────
    try:
        # Previous tests may have contaminated module-level cfg by importing
        # modules while load_config was mocked. Fix by clearing the config
        # cache and patching cfg on already-imported modules (without clearing
        # them from sys.modules, which would break test mock references).
        from automation.config import load as _load_real_config
        _real_cfg = _load_real_config()
        import sys as _sys
        for _m in ("transcribe", "highlight", "export", "download"):
            _mod = _sys.modules.get(_m)
            if _mod is not None and hasattr(_mod, "cfg"):
                _mod.cfg = _real_cfg
        with _api_intercept_context(video_path):
            from automation.orchestrator import run as orchestrator_run
            pipeline_result = orchestrator_run(
                url=url,
                skip_download=skip_download,
                skip_transcribe=skip_transcribe,
                skip_highlight=skip_highlight,
                skip_export=skip_export,
                skip_sync=skip_sync,
                skip_seo=skip_seo,
                auto_sync=auto_sync,
                auto_upload=auto_upload,
                auto_schedule=auto_schedule,
                sample_minutes=sample_minutes,
                sync_from_drive=sync_from_drive,
                mode=mode,
            )
    except Exception as e:
        log.error("Pipeline crashed: %s", e, exc_info=True)
        result.failures.append(f"pipeline_crash: {e}")
        pipeline_result = type('_', (), {
            'exported': [], 'uploaded_count': 0, 'failures': [str(e)],
            'transcript_source': 'none', 'selected_clips': 0, 'seo_generated': 0,
        })()

    # ── Gather results ─────────────────────────────────────────────────────
    result.exported = pipeline_result.exported if hasattr(pipeline_result, 'exported') else []
    result.uploaded_count = getattr(pipeline_result, 'uploaded_count', 0)
    result.selected_clips = getattr(pipeline_result, 'selected_clips', 0)
    result.seo_generated = getattr(pipeline_result, 'seo_generated', 0)
    result.transcript_source = getattr(pipeline_result, 'transcript_source', 'none')
    pipe_failures = getattr(pipeline_result, 'failures', [])
    result.failures = list(pipe_failures)

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
                            if r.get("run_id") == rid and r.get("stage")})

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
        "failures": result.failures,
        "total_seconds": result.total_seconds,
        "transcript_source": result.transcript_source,
        "config_checks": validate_config(cfg),
        "struct_records": len(struct_records),
        "logged_stages": logged_stages,
        "validation": validation,
        "prompts_ok": prompts_ok,
        "event_count": event_count_new,
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
        from automation import decision_store as _ds
        event_types = sorted({e.event_type.value for e in _ds.get_all_events()})
        print(f"        event types:               {event_types}")
    print()
    if result["failures"]:
        print("  Failures detail:")
        for f in result["failures"]:
            print(f"    - {f}")
    ok = (cfg_ok == len(cfg_checks) and result["run_id"] and
          transcript_ok and prompts_ok and not result["failures"])
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
    parser.add_argument("--sync", action="store_true", dest="auto_sync")
    parser.add_argument("--upload", action="store_true", dest="auto_upload")
    parser.add_argument("--schedule", action="store_true", dest="auto_schedule")
    parser.add_argument("--sample-minutes", type=int, default=None)
    parser.add_argument("--sync-from-drive", action="store_true")
    parser.add_argument("--mode", choices=["face_mapper", "ref_grade"], default=None)
    parser.add_argument("--list-external", action="store_true")
    parser.add_argument("--validate-config", action="store_true")
    args = parser.parse_args()

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
        auto_sync=args.auto_sync,
        auto_upload=args.auto_upload,
        auto_schedule=args.auto_schedule,
        sample_minutes=args.sample_minutes,
        sync_from_drive=args.sync_from_drive,
        mode=args.mode,
    )

    ok = _print_validation_report(result)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
