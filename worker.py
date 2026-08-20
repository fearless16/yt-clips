"""Autonomous background worker for cricket Shorts Intelligence.

Handles the exact channel Shorts shelf, real YouTube Analytics outcomes,
canonical model fitting, local reporting, and optional queued video processing.

Usage:
  python worker.py              # Run analytics + learning cycle
  python worker.py --pipeline   # Also check for new videos to process
  python worker.py --report     # Just print current learnings
"""
import _fix_encoding  # noqa: F401 — force UTF-8 on Windows cp1252

import sys
import json
from pathlib import Path

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("worker", cfg["logging"]["log_file"], cfg["logging"]["level"])


def run_analytics_cycle():
    """Sync exact Shorts outcomes and refit the canonical model."""
    log.info("=" * 50)
    log.info("🔄 Worker: Analytics cycle starting...")

    try:
        from automation.seo.analytics import (
            generate_daily_insights,
            sync_clip_performance_from_youtube,
        )
        added = sync_clip_performance_from_youtube()
        result = {"snapshots_added": added, **generate_daily_insights()}
        log.info("Analytics cycle complete: %s", result)
    except Exception as e:
        log.error("Analytics cycle failed: %s", e)
        return None

    log.info("Worker: Analytics cycle done")
    return result


def print_learnings():
    """Print compact local state from the canonical database."""
    try:
        from shorts_intelligence.reporting import load_report

        print(json.dumps(load_report(), ensure_ascii=False, separators=(",", ":"), default=str))
    except Exception as e:
        print(json.dumps({"status": "failed", "error": str(e)}, separators=(",", ":")))


def check_new_videos():
    """Check if there are new videos to process."""
    pending_dir = Path("pending")
    pipeline_script = str(Path(__file__).resolve().parent / "pipeline.py")
    if not Path(pipeline_script).exists():
        log.error("pipeline.py not found at %s", pipeline_script)
        return
    if pending_dir.exists():
        for f in pending_dir.glob("*.url"):
            url = f.read_text().strip()
            log.info(f"🆕 Found pending video: {url}")
            try:
                import subprocess
                result = subprocess.run(
                    [sys.executable, pipeline_script, url, "--sync", "--skip-tests"],
                    capture_output=True, text=True, timeout=3600,
                )
                if result.returncode == 0:
                    f.unlink()  # Remove pending file
                    log.info(f"✅ Processed: {url}")
                else:
                    log.error(f"❌ Pipeline failed: {result.stderr[:500]}")
            except Exception as e:
                log.error(f"❌ Pipeline error: {e}")


def main():
    args = sys.argv[1:]

    if "--report" in args:
        print_learnings()
        return

    # Always run analytics
    run_analytics_cycle()

    # Optionally process new videos
    if "--pipeline" in args:
        check_new_videos()

    print_learnings()
    log.info("Worker cycle complete.")


if __name__ == "__main__":
    main()
