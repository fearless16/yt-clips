"""migrate.py — seed new learned_state from existing self_learner.db facts.

Reads upload + seo_outcome facts from the old `memories` table, reconstructs
synthetic ``ClipEvent(metrics_received, ...)`` objects, and dispatches them
through ``LearningDispatcher`` to populate the new ``learned_state``,
``processed_events``, and ``entity_scores`` tables.

This is a one-time operation intended to bootstrap the v4.0 architecture from
the 486 facts accumulated by the v3.x self_learner pipeline.

Usage:
    # Dry run (no writes)
    .venv/bin/python -m automation.learner.migrate --dry-run

    # Real migration
    .venv/bin/python -m automation.learner.migrate

    # Custom DB path
    .venv/bin/python -m automation.learner.migrate --db /path/to/self_learner.db

    # Reset state and re-migrate from scratch
    .venv/bin/python -m automation.learner.migrate --reset
"""

import argparse
import json
import logging
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from automation.learner.entity_learner import EntityLearner
from automation.learner.state_store import PersistentStateStore
from automation.learner.processed_log import ProcessedEventLog
from automation.learner.dispatcher import LearningDispatcher
from automation.learner.format_learner import FormatLearner
from automation.learner.entity_learner import EntityLearner
from automation.learner.timing_learner import TimingLearner
from automation.learner.duration_learner import DurationLearner
from automation.learner.trend_engine import TrendEngine
from automation.memory.event_models import ClipEvent, EventType


log = logging.getLogger(__name__)


HOOK_KEYWORDS = {
    "reaction": ["reaction", "react", "speechless", "stunned", "wtf", "no way"],
    "fan_war": [" vs ", "rivalry", "beef", "fight", "war"],
    "shock": ["shocking", "crazy", "insane", "unbelievable", "omg", "😱", "horror"],
    "live": ["live", "streaming", "happening now", "ongoing"],
    "debate": ["debate", "should", "opinion", "discuss"],
    "question": ["?"],
    "prediction": ["will ", "predict", "forecast", "future", "karenge", "hoga", "honge", "lenge"],
}

HINDI_RANGE = (0x0900, 0x097F)
EMOJIS_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "]+",
    flags=re.UNICODE,
)


def has_devanagari(text: str) -> bool:
    """Return True if text contains any Devanagari (Hindi) characters."""
    return any(HINDI_RANGE[0] <= ord(c) <= HINDI_RANGE[1] for c in text)


def has_emoji(text: str) -> bool:
    """Return True if text contains emoji characters."""
    return bool(EMOJIS_PATTERN.search(text))


def classify_hook_type(title: str) -> str | None:
    """Classify hook type from title using keyword matching.

    Returns the highest-priority match. Priority order (from observed data):
    reaction > fan_war > shock > live > debate > prediction > question.

    Returns None if no hook type matches.
    """
    title_lower = title.lower()
    for hook_type, keywords in HOOK_KEYWORDS.items():
        if any(kw in title_lower for kw in keywords):
            return hook_type
    return None


def extract_title_features(title: str) -> dict:
    """Extract title features for FormatLearner.process_metrics.

    Returns a dict with:
        length: int (character count)
        is_hinglish: bool (mix of Hindi + Latin script)
        has_caps: bool (any uppercase letter, excluding emoji)
        has_emoji: bool
        has_pipe: bool ("|" character)
        has_question: bool
    """
    length = len(title)
    has_hindi = has_devanagari(title)
    has_latin = any(c.isalpha() and ord(c) < 128 for c in title)
    is_hinglish = has_hindi and has_latin

    alpha_chars = [c for c in title if c.isalpha()]
    has_caps = any(c.isupper() for c in alpha_chars)

    return {
        "length": length,
        "is_hinglish": is_hinglish,
        "has_caps": has_caps,
        "has_emoji": has_emoji(title),
        "has_pipe": "|" in title,
        "has_question": "?" in title,
    }


def parse_publish_timing(date_str: str) -> tuple[int | None, int | None]:
    """Parse YYYYMMDD upload_date string to (hour, day_of_week).

    Since the old data only stores YYYYMMDD (no time), we default to 19:00 UTC
    (8pm IST, peak cricket watching time) as the most likely upload hour.

    Returns (hour, day_of_week) where day_of_week is 0=Monday..6=Sunday.
    """
    try:
        dt = datetime.strptime(date_str, "%Y%m%d")
        default_hour = 19
        return default_hour, dt.weekday()
    except (ValueError, TypeError):
        return None, None


def read_old_facts(db_path: str) -> tuple[dict[str, dict], dict[str, dict]]:
    """Read upload and seo_outcome facts from the old memories table.

    Returns:
        (uploads_by_clip, seo_by_clip) — keyed by clip_id, with the latest
        fact's value-object stored as the dict value.

    Returns empty dicts if the memories table does not exist (e.g. fresh DB
    that has not been seeded yet).
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        uploads_by_clip: dict[str, dict] = {}
        seo_by_clip: dict[str, dict] = {}

        # Check if memories table exists (it won't on a fresh DB)
        table_check = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
        ).fetchone()
        if not table_check:
            log.info("No 'memories' table found in %s — nothing to migrate", db_path)
            return uploads_by_clip, seo_by_clip

        for row in conn.execute(
            "SELECT key, value, timestamp FROM memories "
            "WHERE key LIKE 'fact:upload:%' "
            "ORDER BY timestamp ASC"
        ):
            fact = json.loads(row["value"])
            obj = fact.get("object", {})
            clip_id = obj.get("clip_id")
            if not clip_id:
                continue
            uploads_by_clip[clip_id] = obj

        for row in conn.execute(
            "SELECT key, value, timestamp FROM memories "
            "WHERE key LIKE 'fact:seo_outcome:%' "
            "ORDER BY timestamp ASC"
        ):
            fact = json.loads(row["value"])
            obj = fact.get("object", {})
            clip_id = obj.get("clip_id")
            if not clip_id:
                continue
            seo_by_clip[clip_id] = obj
    finally:
        conn.close()

    return uploads_by_clip, seo_by_clip


def build_metrics_event(
    clip_id: str,
    upload_obj: dict,
    seo_obj: dict | None,
    entity_learner: EntityLearner,
    event_index: int,
) -> ClipEvent:
    """Construct a synthetic ClipEvent with metrics_received event type.

    Args:
        clip_id: The YouTube clip ID
        upload_obj: Upload fact's "object" field (view_count, duration, etc.)
        seo_obj: SEO outcome fact's "object" field (tags, description), or None
        entity_learner: EntityLearner instance for entity extraction
        event_index: Sequence number for deterministic event_id

    Returns:
        ClipEvent with synthesized payload ready for LearningDispatcher.
    """
    title = upload_obj.get("title", "")
    description = (seo_obj or {}).get("description", "")
    tags = (seo_obj or {}).get("tags", [])

    hook_type = classify_hook_type(title)
    title_features = extract_title_features(title)

    entity_text = title + " " + description + " " + " ".join(tags)
    entities = entity_learner.extract_entities(title=entity_text)

    upload_date = upload_obj.get("upload_date", "")
    publish_hour, publish_dow = parse_publish_timing(upload_date)

    duration = upload_obj.get("duration", 0)
    views = upload_obj.get("view_count", 0)
    engagement_rate = upload_obj.get("engagement_rate", 0.0)

    if views > 0 and duration > 0:
        completion_rate = min(1.0, engagement_rate / 10.0) if engagement_rate > 0 else 0.5
    else:
        completion_rate = 0.5

    payload: dict[str, Any] = {
        "views": views,
        "duration_seconds": duration,
        "engagement_rate": engagement_rate,
        "completion_rate": completion_rate,
        "player_tags": entities.get("players", []),
        "team_tags": entities.get("teams", []),
        "format_tags": entities.get("series", []),
        "publish_hour": publish_hour,
        "publish_dow": publish_dow,
        "title_pattern": title_features,
        "hook_type": hook_type,
        "migrated_from": "self_learner.db:memories",
    }

    if upload_date:
        try:
            dt = datetime.strptime(upload_date, "%Y%m%d").replace(
                hour=publish_hour or 19, tzinfo=timezone.utc
            )
            timestamp = dt.isoformat()
        except ValueError:
            timestamp = datetime.now(timezone.utc).isoformat()
    else:
        timestamp = datetime.now(timezone.utc).isoformat()

    event_id = f"migrate-{event_index:04d}-{clip_id}"

    return ClipEvent(
        event_id=event_id,
        clip_id=clip_id,
        timestamp=timestamp,
        event_type=EventType.metrics_received,
        payload_json=json.dumps(payload),
    )


def build_dispatcher(state_store: PersistentStateStore) -> tuple[
    ProcessedEventLog,
    LearningDispatcher,
    FormatLearner,
    EntityLearner,
    TimingLearner,
    DurationLearner,
    TrendEngine,
]:
    """Build the dispatcher and all 5 learners, sharing one state store.

    Returns:
        Tuple of (processed_log, dispatcher, format, entity, timing,
        duration, trend).
    """
    processed_log = ProcessedEventLog(state_store._conn)
    fmt = FormatLearner(state_store)
    ent = EntityLearner(state_store)
    tim = TimingLearner(state_store)
    dur = DurationLearner(state_store)
    trend = TrendEngine(state_store)
    dispatcher = LearningDispatcher(fmt, ent, tim, dur, trend, processed_log)
    return processed_log, dispatcher, fmt, ent, tim, dur, trend


def migrate(
    db_path: str = "self_learner.db",
    dry_run: bool = False,
    reset: bool = False,
) -> dict[str, Any]:
    """Run the full migration: old facts -> new learned_state.

    Args:
        db_path: Path to self_learner.db
        dry_run: If True, show plan but don't write
        reset: If True, clear learned_state + processed_events first

    Returns:
        Migration summary dict with counts and stats.
    """
    log.info("Migration starting: db=%s dry_run=%s reset=%s", db_path, dry_run, reset)

    uploads_by_clip, seo_by_clip = read_old_facts(db_path)
    log.info(
        "Read %d unique upload facts, %d unique seo_outcome facts",
        len(uploads_by_clip), len(seo_by_clip),
    )

    if not uploads_by_clip:
        log.warning("No upload facts found. Nothing to migrate.")
        return {"events_built": 0, "learners_updated": 0, "dry_run": dry_run}

    state_store = None
    processed_log = None
    dispatcher = None
    fmt = ent = tim = dur = trend = None

    if not dry_run:
        state_store = PersistentStateStore(db_path)
        if reset:
            log.info("Resetting learned_state and processed_events tables")
            with state_store._lock:
                state_store._conn.execute("DELETE FROM learned_state")
                state_store._conn.execute("DELETE FROM processed_events")
                state_store._conn.commit()
        processed_log, dispatcher, fmt, ent, tim, dur, trend = build_dispatcher(state_store)

    # EntityLearner is used for extract_entities which doesn't need a live store
    ent_for_extract = EntityLearner(state_store=None)

    events: list[ClipEvent] = []
    skipped = 0
    for i, (clip_id, upload_obj) in enumerate(sorted(uploads_by_clip.items())):
        if upload_obj.get("view_count", 0) == 0:
            skipped += 1
            continue
        seo_obj = seo_by_clip.get(clip_id)
        event = build_metrics_event(clip_id, upload_obj, seo_obj, ent_for_extract, i)
        events.append(event)

    log.info("Built %d metrics_received events (skipped %d with 0 views)", len(events), skipped)

    hook_counter: Counter = Counter()
    title_stats = {"long": 0, "medium": 0, "short": 0, "hinglish": 0, "with_emoji": 0, "with_question": 0}
    player_counter: Counter = Counter()
    team_counter: Counter = Counter()
    for event in events:
        payload = json.loads(event.payload_json)
        if payload.get("hook_type"):
            hook_counter[payload["hook_type"]] += 1
        feats = payload.get("title_pattern", {})
        if feats.get("length", 0) >= 60:
            title_stats["long"] += 1
        elif feats.get("length", 0) >= 30:
            title_stats["medium"] += 1
        else:
            title_stats["short"] += 1
        if feats.get("is_hinglish"):
            title_stats["hinglish"] += 1
        if feats.get("has_emoji"):
            title_stats["with_emoji"] += 1
        if feats.get("has_question"):
            title_stats["with_question"] += 1
        for p in payload.get("player_tags", []):
            player_counter[p] += 1
        for t in payload.get("team_tags", []):
            team_counter[t] += 1

    log.info("Hook type distribution: %s", dict(hook_counter.most_common()))
    log.info("Title stats: %s", title_stats)
    log.info("Top players detected: %s", dict(player_counter.most_common(10)))
    log.info("Top teams detected: %s", dict(team_counter.most_common(10)))

    if dry_run:
        log.info("DRY RUN — no writes performed")
        return {
            "events_built": len(events),
            "skipped_zero_views": skipped,
            "hook_distribution": dict(hook_counter),
            "title_stats": title_stats,
            "top_players": dict(player_counter.most_common(10)),
            "top_teams": dict(team_counter.most_common(10)),
            "dry_run": True,
        }

    updated = dispatcher.dispatch_batch(events)
    log.info("Dispatcher updated %d learner-pairs", updated)

    trend.decay_all()
    trend.prune_expired()

    # Recalibrate scorer weights against actual channel data distribution
    from automation.learner.scorer import CricketScorer
    scorer = CricketScorer(fmt, ent, trend, tim, dur, state_store)
    scorer.recalibrate_weights(events)
    log.info("Scorer weights recalibrated: %s",
             {k: round(v, 3) for k, v in scorer.get_weights().items()
              if not k.startswith("last_")})

    summary = {
        "events_built": len(events),
        "events_dispatched": len(events),
        "learners_updated": updated,
        "skipped_zero_views": skipped,
        "hook_distribution": dict(hook_counter),
        "title_stats": title_stats,
        "top_players": dict(player_counter.most_common(10)),
        "top_teams": dict(team_counter.most_common(10)),
        "dry_run": False,
        "format_scores": fmt.get_scores(),
        "duration_scores": dur.get_scores(),
        "top_player_scores": ent.get_top_entities("players", limit=10),
        "top_team_scores": ent.get_top_entities("teams", limit=10),
        "best_hour": tim.best_hour(),
        "best_day": tim.best_day(),
        "active_trends": len(trend.get_active()),
        "scoring_weights": scorer.get_weights(),
    }

    state_store.close()
    log.info("Migration complete: %s", json.dumps(summary, indent=2, default=str))
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate self_learner.db facts to new learned_state schema"
    )
    parser.add_argument(
        "--db", default="self_learner.db",
        help="Path to self_learner.db (default: ./self_learner.db)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be migrated without writing",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Clear learned_state and processed_events before migration",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    summary = migrate(
        db_path=args.db,
        dry_run=args.dry_run,
        reset=args.reset,
    )

    print("\n" + "=" * 60)
    print("MIGRATION SUMMARY")
    print("=" * 60)
    print(f"  Events built:       {summary['events_built']}")
    print(f"  Events dispatched:  {summary.get('events_dispatched', 0)}")
    print(f"  Learners updated:   {summary.get('learners_updated', 0)}")
    print(f"  Skipped (0 views):  {summary['skipped_zero_views']}")
    print(f"  Dry run:            {summary['dry_run']}")

    if not summary["dry_run"]:
        print(f"\n  Active trends:      {summary['active_trends']}")
        print(f"  Best hour:          {summary['best_hour']}:00")
        print(f"  Best day:           {summary['best_day']}")
        print(f"\n  Top players:")
        for p in summary["top_player_scores"][:5]:
            print(f"    {p['name']:15s} score={p['score']:.3f} n={p['n']:3d} avg_views={p['avg_views']:.0f}")
        print(f"\n  Top teams:")
        for t in summary["top_team_scores"][:5]:
            print(f"    {t['name']:15s} score={t['score']:.3f} n={t['n']:3d} avg_views={t['avg_views']:.0f}")

    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
