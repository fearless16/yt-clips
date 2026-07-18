"""
worker.py — Autonomous background worker for yt-clips.
Runs on Mac via launchd. Handles:
  1. Fetch YouTube analytics (videos, shorts, lives) every run
  2. Feed SEO learner with performance data
  3. Generate performance report
  4. Optionally trigger new video processing

Usage:
  python worker.py              # Run analytics + learning cycle
  python worker.py --pipeline   # Also check for new videos to process
  python worker.py --report     # Just print current learnings
"""
import _fix_encoding  # noqa: F401 — force UTF-8 on Windows cp1252

import sys
import json
import time
from datetime import datetime
from pathlib import Path

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("worker", cfg["logging"]["log_file"], cfg["logging"]["level"])


def run_analytics_cycle():
    """Fetch analytics for all content types and feed the learner."""
    log.info("=" * 50)
    log.info("🔄 Worker: Analytics cycle starting...")

    try:
        from automation.seo.analytics import generate_daily_insights
        result = generate_daily_insights()
        if result:
            log.info(f"✅ Analytics cycle complete: {result}")
        else:
            log.warning("⚠️ No analytics data returned")
    except Exception as e:
        log.error(f"❌ Analytics cycle failed: {e}")

    log.info("🔄 Worker: Analytics cycle done")
    return True


def print_learnings():
    """Print current SEO learnings from the learner."""
    try:
        from automation.seo.seo_learner import SEOLearner
        learner = SEOLearner()
        suggestions = learner.get_seo_improvement_suggestions()
        prefs = learner.get_learned_title_preferences()

        print("\n" + "=" * 60)
        print("🧠 SEO LEARNER STATUS")
        print("=" * 60)

        clips = learner.learned_insights.get("clips", [])
        print(f"\n📊 Data collected: {len(clips)} clips analyzed")

        if prefs.get("best_patterns"):
            print("\n✅ Winning patterns:")
            for p, score in prefs["best_patterns"][:3]:
                print(f"   {p} (avg: {score:.2f})")

        if prefs.get("avoid_patterns"):
            print("\n❌ Avoid patterns:")
            for p, score in prefs["avoid_patterns"][:3]:
                print(f"   {p} (avg: {score:.2f})")

        if suggestions:
            print("\n💡 Suggestions:")
            for s in suggestions:
                print(f"   • {s}")

        best_prov, best_mod = learner.get_best_model()
        if best_prov:
            print(f"\n🏆 Best model: {best_prov}/{best_mod}")

        print("=" * 60 + "\n")
    except Exception as e:
        print(f"Error: {e}")


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
