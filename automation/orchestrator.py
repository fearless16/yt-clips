"""orchestrator.py — Pipeline orchestrator for yt-clips automation."""

import time
import logging
from dataclasses import dataclass, field

from . import config
from .worker import ParallelPool


@dataclass
class PipelineResult:
    exported: list = field(default_factory=list)
    uploaded_count: int = 0
    failures: list = field(default_factory=list)
    total_seconds: float = 0.0
    transcript_source: str = "none"


def _t(start: float) -> str:
    return f"{time.monotonic() - start:.1f}s"


def run(url: str, skip_download=False, skip_transcribe=False,
        skip_highlight=False, skip_export=False, skip_sync=False,
        skip_seo=False, auto_sync=False, auto_upload=False,
        auto_schedule=False, skip_tests=False, sample_minutes=None,
        sync_from_drive=False, mode=None) -> PipelineResult:
    start = time.monotonic()
    result = PipelineResult()

    if not skip_transcribe:
        t0 = time.monotonic()
        logging.info("[phase 0] YouTube transcript")
        try:
            from .transcript import fetch as fetch_transcript
            transcript = fetch_transcript(url)
            if transcript.get("segments"):
                skip_transcribe = True
                result.transcript_source = transcript.get("source", "api")
                logging.info(f"[phase 0] done {_t(t0)} source={result.transcript_source}")
        except Exception as e:
            result.failures.append(f"phase0: {e}")

    if not skip_download:
        t0 = time.monotonic()
        logging.info("[phase 1] Download")
        try:
            from download import download
            download(url, sample_minutes=sample_minutes)
            logging.info(f"[phase 1] done {_t(t0)}")
        except Exception as e:
            result.failures.append(f"phase1: {e}")

    if not skip_transcribe:
        t0 = time.monotonic()
        logging.info("[phase 2] Transcribe")
        try:
            from transcribe import transcribe
            transcribe()
            logging.info(f"[phase 2] done {_t(t0)}")
        except Exception as e:
            result.failures.append(f"phase2: {e}")

    if not skip_highlight:
        t0 = time.monotonic()
        logging.info("[phase 3] Highlight")
        try:
            from highlight import highlight
            highlight(skip_tests=skip_tests, sample_minutes=sample_minutes, mode=mode)
            logging.info(f"[phase 3] done {_t(t0)}")
        except Exception as e:
            result.failures.append(f"phase3: {e}")

    if not skip_export:
        t0 = time.monotonic()
        logging.info("[phase 4] Export")
        try:
            from export import export
            result.exported = export()
            logging.info(f"[phase 4] done {_t(t0)} exported={len(result.exported)}")
        except Exception as e:
            result.failures.append(f"phase4: {e}")

    do_sync = auto_sync or sync_from_drive
    if not skip_sync and do_sync:
        t0 = time.monotonic()
        logging.info("[phase 5] Sync")
        try:
            from sync import sync
            sync(from_drive=sync_from_drive)
            logging.info(f"[phase 5] done {_t(t0)}")
        except Exception as e:
            result.failures.append(f"phase5: {e}")

    if not skip_seo:
        t0 = time.monotonic()
        logging.info("[phase 6] SEO")
        try:
            from generate_seo import generate_seo
            generate_seo()
            logging.info(f"[phase 6] done {_t(t0)}")
        except Exception as e:
            result.failures.append(f"phase6: {e}")

    if auto_upload or auto_schedule:
        t0 = time.monotonic()
        logging.info("[phase 7] Upload")
        try:
            from upload import upload
            result.uploaded_count = upload()
            logging.info(f"[phase 7] done {_t(t0)} uploaded={result.uploaded_count}")
        except Exception as e:
            result.failures.append(f"phase7: {e}")

    result.total_seconds = time.monotonic() - start
    logging.info(f"pipeline done {result.total_seconds:.1f}s failures={len(result.failures)}")
    return result
