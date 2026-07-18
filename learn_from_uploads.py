"""learn_from_uploads.py — Fetch all uploaded Shorts from YouTube and feed
them into the self_learner module so it can build keyword/title/engagement
patterns locally.

This bypasses the YouTube Data API (which is quota-exhausted) by using
yt-dlp with cookies.txt (already on the repo root). Engagement metrics
(view/like/comment counts) come from yt-dlp's public metadata scrape.

Usage:
    .venv/bin/python learn_from_uploads.py
    .venv/bin/python learn_from_uploads.py --limit 30
    .venv/bin/python learn_from_uploads.py --channel UCQtCMKPc41MHd7hujVuGm5g
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# ── Set SELF_LEARNER_DB BEFORE importing self_learner ──────────────────
# memory.py reads SELF_LEARNER_DB at module import time as a module-level
# constant (_MEMORY_DB), so it must be set before any self_learner import.
_DEFAULT_DB = os.environ.get("SELF_LEARNER_DB", "self_learner.db")
os.environ.setdefault("SELF_LEARNER_DB", _DEFAULT_DB)

from self_learner import (
    Learner,
    SEOLearner,
    SEOPerformance,
    TrendAnalyzer,
    TrendPoint,
    RecommendationEngine,
)


CHANNEL_SHORTS_URL = (
    "https://www.youtube.com/channel/UCQtCMKPc41MHd7hujVuGm5g/shorts"
)


def list_channel_shorts(channel_url: str) -> list:
    """Get flat list of video IDs from a channel's shorts tab."""
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--cookies", "cookies.txt",
        "--flat-playlist",
        "--print", "%(id)s",
        channel_url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"  yt-dlp list error: {result.stderr[:300]}")
        return []
    ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return ids


def fetch_video_meta(video_id: str):
    """Fetch full metadata for one video (uses cookies for private/unlisted)."""
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--cookies", "cookies.txt",
        "--dump-json",
        "--no-download",
        f"https://www.youtube.com/shorts/{video_id}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def parse_hashtags(text: str) -> list:
    """Extract #hashtags from title/description text."""
    return re.findall(r"#(\w+)", text or "")


def parse_tags_from_description(desc: str) -> list:
    """Some channels put comma-separated tags at the end of description."""
    if not desc:
        return []
    tags_line = ""
    for line in desc.splitlines():
        if line.lower().startswith("tags:") or line.lower().startswith("keywords:"):
            tags_line = line.split(":", 1)[1]
            break
    if not tags_line:
        return []
    return [t.strip() for t in tags_line.split(",") if t.strip()]


def compute_engagement_rate(views: int, likes: int, comments: int) -> float:
    """Engagement rate as a percentage of views."""
    if views <= 0:
        return 0.0
    return ((likes + comments) / views) * 100.0


def main():
    parser = argparse.ArgumentParser(description="Feed uploaded Shorts into self_learner")
    parser.add_argument("--limit", type=int, default=0, help="Max videos to process (0 = all)")
    parser.add_argument("--channel", type=str, default=CHANNEL_SHORTS_URL,
                        help="Channel shorts URL")
    parser.add_argument("--db", type=str, default=_DEFAULT_DB,
                        help="Path to self_learner SQLite DB")
    args = parser.parse_args()

    db_path = args.db
    print(f"self_learner DB: {db_path}")

    print(f"\n[1/4] Listing Shorts on channel: {args.channel}")
    video_ids = list_channel_shorts(args.channel)
    if not video_ids:
        print("  No videos found. Check cookies.txt / channel URL.")
        sys.exit(1)
    print(f"  Found {len(video_ids)} Shorts")
    if args.limit > 0:
        video_ids = video_ids[:args.limit]
        print(f"  Processing first {args.limit}")

    print(f"\n[2/4] Initializing self_learner modules")
    from self_learner.memory import PersistentMemory
    from self_learner.knowledge import KnowledgeBase
    _mem = PersistentMemory(db_path=db_path)
    _kb = KnowledgeBase(memory=_mem)
    learner = Learner(memory=_mem, knowledge=_kb)
    seo_learner = SEOLearner(memory=_mem, knowledge=_kb)
    trend_analyzer = TrendAnalyzer(memory=_mem)
    rec_engine = RecommendationEngine(seo_learner=seo_learner, trend_analyzer=trend_analyzer)
    print("  Learner / SEOLearner / TrendAnalyzer / RecommendationEngine ready")

    print(f"\n[3/4] Fetching metadata + recording learnings")
    processed = 0
    skipped = 0
    for idx, vid_id in enumerate(video_ids, 1):
        print(f"  [{idx}/{len(video_ids)}] {vid_id} ...", end=" ", flush=True)
        meta = fetch_video_meta(vid_id)
        if not meta:
            print("SKIP (fetch failed)")
            skipped += 1
            time.sleep(0.5)
            continue

        title = meta.get("title", "") or ""
        desc = meta.get("description", "") or ""
        views = int(meta.get("view_count", 0) or 0)
        likes = int(meta.get("like_count", 0) or 0)
        comments = int(meta.get("comment_count", 0) or 0)
        upload_date = meta.get("upload_date", "")
        yt_tags = meta.get("tags", []) or []
        desc_tags = parse_tags_from_description(desc)
        all_tags = list(dict.fromkeys(yt_tags + desc_tags))
        hashtags = parse_hashtags(title + " " + desc)
        engagement = compute_engagement_rate(views, likes, comments)

        learner.observe("upload", {
            "clip_id": vid_id,
            "title": title,
            "duration": meta.get("duration", 0),
            "view_count": views,
            "like_count": likes,
            "comment_count": comments,
            "engagement_rate": engagement,
            "tag_count": len(all_tags),
            "hashtag_count": len(hashtags),
            "upload_date": upload_date,
        })

        seo_perf = SEOPerformance(
            clip_id=vid_id,
            title=title,
            description=desc,
            hashtags=hashtags,
            tags=all_tags,
            is_shorts=True,
            provider="uploaded",
            model="unknown",
            upload_success=True,
            view_count=views,
            like_count=likes,
            engagement_rate=engagement,
        )
        seo_learner.record_seo_outcome(seo_perf)

        trend_analyzer.record_metric(TrendPoint(
            metric="views",
            value=float(views),
            timestamp=time.time(),
            metadata={"clip_id": vid_id},
        ))
        trend_analyzer.record_metric(TrendPoint(
            metric="engagement_rate",
            value=engagement,
            timestamp=time.time(),
            metadata={"clip_id": vid_id},
        ))

        print(f"OK (views={views}, likes={likes}, eng={engagement:.2f}%)")
        processed += 1
        time.sleep(0.4)

    print(f"\n[4/4] Summary")
    print(f"  Processed: {processed}")
    print(f"  Skipped:   {skipped}")
    print(f"  DB:        {os.environ['SELF_LEARNER_DB']}")

    if processed == 0:
        print("  No videos to summarize. Exiting.")
        rec_engine.close()
        trend_analyzer.close()
        seo_learner.close()
        learner.close()
        return

    print(f"\n--- Learned Insights ---")
    print(f"Top keywords (by avg engagement):")
    for kw in seo_learner.get_best_keywords("shorts", limit=15):
        print(f"  {kw['keyword']:30s}  eng={kw['avg_engagement']:.2f}%  n={kw['count']}")

    print(f"\nTitle patterns:")
    for p in seo_learner.get_best_title_patterns("shorts", limit=10):
        print(f"  {p['pattern']:20s}  eng={p['avg_engagement']:.2f}%  n={p['count']}")

    print(f"\nContent type stats:")
    for ct, stats in seo_learner.get_content_type_stats().items():
        print(f"  {ct:12s}  n={stats['count']}  success={stats['success_rate']:.2%}  eng={stats['avg_engagement']:.2f}%")

    print(f"\nProvider performance:")
    for prov, stats in seo_learner.get_provider_performance().items():
        print(f"  {prov:12s}  uses={stats['total_uses']}  success={stats['success_rate']:.2%}  eng={stats['avg_engagement']:.2f}%")

    recs = rec_engine.generate_recommendations()
    print(f"\nTop {min(5, len(recs))} recommendations:")
    for rec in recs[:5]:
        print(f"  [{rec.priority}/{rec.category}] {rec.title}")
        print(f"    -> {rec.action}")

    rec_engine.close()
    trend_analyzer.close()
    seo_learner.close()
    learner.close()

    print(f"\n[5/5] Refreshing v4.0 learned_state (5-learner architecture)")
    try:
        from automation.learner.migrate import migrate as migrate_learner_state
        summary = migrate_learner_state(db_path=db_path, dry_run=False, reset=False)
        if summary.get("events_built", 0) > 0:
            print(f"  Migrated {summary['events_built']} events -> "
                  f"{summary.get('learners_updated', 0)} learner updates")
        else:
            print(f"  No new events to migrate")
    except Exception as e:
        print(f"  WARNING: v4.0 migration failed: {e}")
        print(f"  Run separately: .venv/bin/python -m automation.learner.migrate")

    print(f"\nDB closed. Persistence: {os.environ['SELF_LEARNER_DB']}")


if __name__ == "__main__":
    main()
