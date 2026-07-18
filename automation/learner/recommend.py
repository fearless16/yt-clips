"""recommend.py — CLI for cricket learning engine.

Generates and prints actionable recommendations from the learned state.
Also supports scoring hypothetical candidates and inspecting raw scores.

Usage:
    # Show all recommendations
    .venv/bin/python -m automation.learner.recommend

    # Filter by category
    .venv/bin/python -m automation.learner.recommend --category player

    # JSON output for piping
    .venv/bin/python -m automation.learner.recommend --json

    # Score a hypothetical candidate
    .venv/bin/python -m automation.learner.recommend --score \\
        --title "Bumrah magic spell" \\
        --players bumrah \\
        --teams mi

    # Show raw learned state
    .venv/bin/python -m automation.learner.recommend --stats

    # Custom DB path
    .venv/bin/python -m automation.learner.recommend --db /path/to/self_learner.db
"""

import argparse
import json
import logging
import sys
from typing import Any

from automation.learner.entity_learner import EntityLearner
from automation.learner.state_store import PersistentStateStore
from automation.learner.format_learner import FormatLearner
from automation.learner.timing_learner import TimingLearner
from automation.learner.duration_learner import DurationLearner
from automation.learner.trend_engine import TrendEngine
from automation.learner.scorer import CricketScorer
from automation.learner.recommender import CricketRecommendationEngine, CricketRecommendation
from automation.learner.migrate import classify_hook_type, extract_title_features


log = logging.getLogger(__name__)


CATEGORY_ICONS = {
    "player": "[PLAYER]",
    "team": "[TEAM  ]",
    "hook": "[HOOK  ]",
    "title": "[TITLE ]",
    "duration": "[DUR   ]",
    "timing": "[TIME  ]",
    "topic": "[TREND ]",
    "warning": "[WARN  ]",
}

PRIORITY_BAR = {
    "high": "######",
    "medium": "####  ",
    "low": "##    ",
}


def _build_engine(db_path: str) -> tuple[
    PersistentStateStore,
    FormatLearner,
    EntityLearner,
    TimingLearner,
    DurationLearner,
    TrendEngine,
    CricketScorer,
    CricketRecommendationEngine,
]:
    """Build all 5 learners + scorer + recommender from a single state store."""
    state = PersistentStateStore(db_path)
    fmt = FormatLearner(state)
    ent = EntityLearner(state)
    tim = TimingLearner(state)
    dur = DurationLearner(state)
    trend = TrendEngine(state)
    scorer = CricketScorer(fmt, ent, trend, tim, dur, state)
    rec_engine = CricketRecommendationEngine(fmt, ent, trend, tim, dur, state)
    return state, fmt, ent, tim, dur, trend, scorer, rec_engine


def _format_recommendation(rec: CricketRecommendation) -> str:
    """Format a single recommendation for terminal display."""
    icon = CATEGORY_ICONS.get(rec.category, f"[{rec.category.upper():7s}]")
    bar = PRIORITY_BAR.get(rec.priority, "      ")
    lines = [
        f"{icon} {bar} priority={rec.priority:6s} confidence={rec.confidence:.2f}",
        f"  -> {rec.recommendation}",
        f"  >> {rec.reason}",
    ]
    if rec.expires_at:
        lines.append(f"  [expires: {rec.expires_at}]")
    if rec.supporting_data:
        data_str = ", ".join(
            f"{k}={v}" for k, v in list(rec.supporting_data.items())[:4]
        )
        lines.append(f"  [data] {data_str}")
    return "\n".join(lines)


def show_recommendations(
    rec_engine: CricketRecommendationEngine,
    category: str | None = None,
    as_json: bool = False,
) -> int:
    """Generate and display recommendations."""
    recs = rec_engine.generate_all()

    if category:
        recs = [r for r in recs if r.category == category]

    if as_json:
        output = [
            {
                "category": r.category,
                "priority": r.priority,
                "recommendation": r.recommendation,
                "reason": r.reason,
                "confidence": r.confidence,
                "supporting_data": r.supporting_data,
                "expires_at": r.expires_at,
            }
            for r in recs
        ]
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        if not recs:
            print("\nNo recommendations found.")
            print("(Try: python -m automation.learner.migrate to seed the engine first)")
            return 0

        print("\n" + "=" * 72)
        print("CRICKET LEARNING ENGINE — RECOMMENDATIONS")
        print("=" * 72)
        for i, rec in enumerate(recs, 1):
            print(f"\n#{i}")
            print(_format_recommendation(rec))
        print("\n" + "=" * 72)
        print(f"Total: {len(recs)} recommendations")
        high = sum(1 for r in recs if r.priority == "high")
        med = sum(1 for r in recs if r.priority == "medium")
        low = sum(1 for r in recs if r.priority == "low")
        print(f"Priority breakdown: {high} high, {med} medium, {low} low")
        print("=" * 72 + "\n")

    return 0


def score_candidate(
    fmt: FormatLearner,
    ent: EntityLearner,
    trend: TrendEngine,
    tim: TimingLearner,
    dur: DurationLearner,
    scorer: CricketScorer,
    title: str,
    players: list[str],
    teams: list[str],
    series: str | None = None,
    as_json: bool = False,
    state_store: Any = None,
) -> int:
    """Score a hypothetical candidate and explain the result."""
    hook_type = classify_hook_type(title)
    title_features = extract_title_features(title)
    entities = ent.extract_entities(title=title + " ".join(players) + " ".join(teams))

    full_players = list(set(players + entities.get("players", [])))
    full_teams = list(set(teams + entities.get("teams", [])))
    full_series = series or (entities.get("series", [None])[0] if entities.get("series") else None)

    schedule = tim.get_schedule_recommendation()
    best_hour = schedule.get("hour", 19)
    best_day = schedule.get("day", "saturday")

    candidate = {
        "title": title,
        "hook_type": hook_type,
        "title_features": title_features,
        "players": full_players,
        "teams": full_teams,
        "series": full_series,
        "publish_hour": best_hour,
        "publish_dow": best_day,
    }

    score = scorer.score_candidate(candidate)
    trend_score = trend.get_trend_for_entities(players=full_players, teams=full_teams)
    duration_target = dur.select_duration()
    hook_choice = fmt.select_hook_type()
    ent_score = ent.get_entity_score(full_players, full_teams, full_series)
    fmt_score = fmt.get_title_score(title_features)

    if as_json:
        output = {
            "candidate": candidate,
            "score": round(score, 3),
            "breakdown": {
                "trend_score": round(trend_score, 3),
                "format_score": round(fmt_score, 3),
                "entity_score": round(ent_score, 3),
                "timing_score": round(schedule.get("confidence", 0.5), 3),
            },
            "suggestions": {
                "hook_type": hook_choice,
                "duration_seconds": duration_target,
                "publish_hour": best_hour,
                "publish_day": best_day,
            },
            "weights": scorer.get_weights(),
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print("\n" + "=" * 72)
        print(f"CANDIDATE SCORE: {score:.3f} / 1.0")
        print("=" * 72)
        print(f"\n  Title:        {title}")
        print(f"  Hook type:    {hook_type or '(none detected)'}")
        print(f"  Players:      {', '.join(full_players) or '(none)'}")
        print(f"  Teams:        {', '.join(full_teams) or '(none)'}")
        print(f"  Series:       {full_series or '(none)'}")
        print()
        print("  Score breakdown:")
        print(f"    Trend score:  {trend_score:.3f}  (weight {scorer.get_weights()['trend_weight']:.2f})")
        print(f"    Format score: {fmt_score:.3f}  (weight {scorer.get_weights()['format_weight']:.2f})")
        print(f"    Entity score: {ent_score:.3f}  (weight {scorer.get_weights()['entity_weight']:.2f})")
        print(f"    Timing score: {schedule.get('confidence', 0.5):.3f}  (weight {scorer.get_weights()['timing_weight']:.2f})")
        print()
        print(f"  Recommended hook type:  {hook_choice}")
        print(f"  Recommended duration:   {duration_target}s")
        print(f"  Recommended publish:    {best_hour}:00 on {best_day}")
        print()

        # Use calibrated thresholds if available, otherwise defaults
        thresholds = state_store.get("verdict_thresholds") if state_store else None
        t_strong = (thresholds or {}).get("strong", 0.60)
        t_good = (thresholds or {}).get("good", 0.50)
        t_avg = (thresholds or {}).get("average", 0.40)

        verdict = (
            f"STRONG — likely to perform well above baseline (>{t_strong:.2f})"
            if score >= t_strong
            else f"GOOD — expected to perform above average (>{t_good:.2f})"
            if score >= t_good
            else f"AVERAGE — expected to match channel baseline (>{t_avg:.2f})"
            if score >= t_avg
            else "WEAK — likely to underperform, consider revising"
        )
        print(f"  Verdict: {verdict}")
        print("=" * 72 + "\n")

    return 0


def show_stats(
    state: PersistentStateStore,
    fmt: FormatLearner,
    ent: EntityLearner,
    tim: TimingLearner,
    dur: DurationLearner,
    trend: TrendEngine,
    scorer: CricketScorer,
    as_json: bool = False,
) -> int:
    """Display raw learned state for debugging."""
    stats = {
        "weights": scorer.get_weights(),
        "format": {
            "hook_types": fmt.get_scores().get("hook_types", {}),
            "title_patterns": fmt.get_scores().get("title_patterns", {}),
        },
        "entity": {
            "top_players": ent.get_top_entities("players", limit=15),
            "top_teams": ent.get_top_entities("teams", limit=10),
            "top_series": ent.get_top_entities("series", limit=10),
        },
        "timing": {
            "best_hour": tim.best_hour(),
            "best_day": tim.best_day(),
            "best_match_phase": tim.best_match_phase(),
            "schedule": tim.get_schedule_recommendation(),
        },
        "duration": dur.get_scores(),
        "trends": {
            "active_count": len(trend.get_active()),
            "hot_topics": trend.get_hot_topics(limit=5),
        },
    }

    if as_json:
        print(json.dumps(stats, indent=2, default=str, ensure_ascii=False))
    else:
        print("\n" + "=" * 72)
        print("LEARNED STATE — RAW SCORES")
        print("=" * 72)

        print("\n[Scorer weights]")
        for k, v in stats["weights"].items():
            print(f"  {k:25s} = {v:.3f}")

        print("\n[Format — hook types]")
        if stats["format"]["hook_types"]:
            for name, data in sorted(
                stats["format"]["hook_types"].items(),
                key=lambda x: -x[1].get("score", 0)
            ):
                print(f"  {name:12s} score={data['score']:.3f} n={data['n']:3d} "
                      f"avg_views={data['avg_views']:.0f}")
        else:
            print("  (no data)")

        print("\n[Format — title patterns]")
        if stats["format"]["title_patterns"]:
            for name, data in sorted(
                stats["format"]["title_patterns"].items(),
                key=lambda x: -x[1].get("score", 0)
            ):
                print(f"  {name:15s} score={data['score']:.3f} n={data['n']:3d} "
                      f"avg_views={data['avg_views']:.0f}")
        else:
            print("  (no data)")

        print("\n[Entities — top players]")
        for p in stats["entity"]["top_players"]:
            print(f"  {p['name']:15s} score={p['score']:.3f} n={p['n']:3d} "
                  f"avg_views={p['avg_views']:.0f} trend={p.get('trend')}")

        print("\n[Entities — top teams]")
        for t in stats["entity"]["top_teams"]:
            print(f"  {t['name']:10s} score={t['score']:.3f} n={t['n']:3d} "
                  f"avg_views={t['avg_views']:.0f} trend={t.get('trend')}")

        print("\n[Timing]")
        print(f"  Best hour:        {stats['timing']['best_hour']}:00")
        print(f"  Best day:         {stats['timing']['best_day']}")
        print(f"  Best match phase: {stats['timing']['best_match_phase']}")
        schedule = stats["timing"]["schedule"]
        print(f"  Schedule confidence: {schedule.get('confidence', 0):.2f}")
        print(f"  Total observations:  {schedule.get('total_observations', 0)}")

        print("\n[Duration]")
        dur_data = stats["duration"]
        print(f"  Sweet spot:    {dur_data.get('sweet_spot', '?')}")
        print(f"  Total clips:   {dur_data.get('total_n', 0)}")
        for name, data in sorted(
            dur_data.get("buckets", {}).items(),
            key=lambda x: -x[1].get("score", 0)
        ):
            print(f"  {name:10s} score={data['score']:.3f} n={data['n']:3d} "
                  f"avg_views={data['avg_views']:.0f} "
                  f"completion={data['avg_completion']:.2f}")

        print("\n[Trends]")
        print(f"  Active: {stats['trends']['active_count']}")
        if stats["trends"]["hot_topics"]:
            for t in stats["trends"]["hot_topics"]:
                print(f"  - {t.get('query', '?')} (score {t['score']:.2f})")

        print("=" * 72 + "\n")

    return 0


def calibrate_verdicts(
    state: "PersistentStateStore",
    fmt: "FormatLearner",
    ent: "EntityLearner",
    trend: "TrendEngine",
    tim: "TimingLearner",
    dur: "DurationLearner",
    scorer: "CricketScorer",
    as_json: bool = False,
) -> int:
    """Score all historical events and calibrate verdict thresholds.

    Reads upload facts from memories table, scores each one through
    the current scorer, and sets percentile-based thresholds:
      - STRONG: top 25% (p75)
      - GOOD: top 50% (p50/median)
      - AVERAGE: top 75% (p25)
    """
    import sqlite3

    conn = sqlite3.connect(state._db_path if hasattr(state, '_db_path') else "self_learner.db")
    conn.row_factory = sqlite3.Row

    table_check = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
    ).fetchone()
    if not table_check:
        print("No memories table found — cannot calibrate")
        conn.close()
        return 1

    rows = conn.execute(
        "SELECT value FROM memories WHERE key LIKE 'fact:upload:%'"
    ).fetchall()
    conn.close()

    scores = []
    for row in rows:
        try:
            fact = json.loads(row["value"])
            obj = fact.get("object", {})
            views = obj.get("view_count", 0)
            if views == 0:
                continue
            title = obj.get("title", "")
            hook_type = classify_hook_type(title)
            title_features = extract_title_features(title)
            entities = ent.extract_entities(title=title)

            candidate = {
                "title": title,
                "hook_type": hook_type,
                "title_pattern": title_features,
                "players": entities.get("players", []),
                "teams": entities.get("teams", []),
                "format_tags": entities.get("series", []),
                "weighted_score": views / 100.0,
            }
            s = scorer.score_candidate(candidate)
            scores.append({"title": title[:60], "views": views, "score": round(s, 3)})
        except Exception:
            continue

    if not scores:
        print("No scorable events found")
        return 1

    scores.sort(key=lambda x: x["score"])
    score_vals = [s["score"] for s in scores]
    n = len(score_vals)

    p25 = score_vals[int(n * 0.25)]
    p50 = score_vals[int(n * 0.50)]
    p75 = score_vals[int(n * 0.75)]
    p_min = score_vals[0]
    p_max = score_vals[-1]
    avg = sum(score_vals) / n

    thresholds = {
        "strong": round(p75, 3),
        "good": round(p50, 3),
        "average": round(p25, 3),
    }
    state.set("verdict_thresholds", thresholds)

    if as_json:
        output = {
            "n": n,
            "min": p_min,
            "p25": p25,
            "median": p50,
            "p75": p75,
            "max": p_max,
            "mean": round(avg, 3),
            "thresholds": thresholds,
            "bottom_5": scores[:5],
            "top_5": scores[-5:],
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print("\n" + "=" * 72)
        print("SCORE CALIBRATION — PERCENTILE DISTRIBUTION")
        print("=" * 72)
        print(f"  Scored {n} historical events")
        print(f"  Min:    {p_min:.3f}")
        print(f"  P25:    {p25:.3f}  -> AVERAGE threshold")
        print(f"  Median: {p50:.3f}  -> GOOD threshold")
        print(f"  P75:    {p75:.3f}  -> STRONG threshold")
        print(f"  Max:    {p_max:.3f}")
        print(f"  Mean:   {avg:.3f}")
        print()
        print("  Bottom 5 (lowest scores):")
        for s in scores[:5]:
            print(f"    {s['score']:.3f}  views={s['views']:5d}  {s['title']}")
        print()
        print("  Top 5 (highest scores):")
        for s in scores[-5:]:
            print(f"    {s['score']:.3f}  views={s['views']:5d}  {s['title']}")
        print()
        print(f"  Thresholds saved: {thresholds}")
        print("=" * 72 + "\n")

    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Cricket Learning Engine — recommendations, scoring, and stats"
    )
    parser.add_argument(
        "--db", default="self_learner.db",
        help="Path to self_learner.db (default: ./self_learner.db)",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--category", type=str,
        help="Filter recommendations by category (player, team, hook, title, "
             "duration, timing, topic, warning)",
    )
    mode.add_argument(
        "--score", action="store_true",
        help="Score a hypothetical candidate (requires --title)",
    )
    mode.add_argument(
        "--stats", action="store_true",
        help="Show raw learned state for debugging",
    )

    mode.add_argument(
        "--calibrate", action="store_true",
        help="Calibrate verdict thresholds from historical data",
    )

    parser.add_argument(
        "--title", type=str,
        help="Candidate title (used with --score)",
    )
    parser.add_argument(
        "--players", nargs="+", default=[],
        help="Player tags for candidate (used with --score)",
    )
    parser.add_argument(
        "--teams", nargs="+", default=[],
        help="Team tags for candidate (used with --score)",
    )
    parser.add_argument(
        "--series", type=str, default=None,
        help="Series tag for candidate (used with --score)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output JSON instead of formatted text",
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

    state, fmt, ent, tim, dur, trend, scorer, rec_engine = _build_engine(args.db)

    try:
        if args.calibrate:
            return calibrate_verdicts(
                state, fmt, ent, trend, tim, dur, scorer,
                as_json=args.json,
            )

        if args.stats:
            return show_stats(state, fmt, ent, tim, dur, trend, scorer, as_json=args.json)

        if args.score:
            if not args.title:
                print("ERROR: --score requires --title", file=sys.stderr)
                return 1
            return score_candidate(
                fmt, ent, trend, tim, dur, scorer,
                title=args.title,
                players=args.players,
                teams=args.teams,
                series=args.series,
                as_json=args.json,
                state_store=state,
            )

        return show_recommendations(rec_engine, category=args.category, as_json=args.json)
    finally:
        state.close()


if __name__ == "__main__":
    sys.exit(main())
