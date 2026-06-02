"""runner.py — CLI entry point for the self-learner module.

Usage:
    python -m self_learner.runner --help
    python -m self_learner.runner observe pipeline_run '{"duration": 120}'
    python -m self_learner.runner predict pipeline_run
    python -m self_learner.runner insights
    python -m self_learner.runner stats
    python -m self_learner.runner trends
    python -m self_learner.runner anomalies <metric>
    python -m self_learner.runner seo_keywords [shorts|long_form]
    python -m self_learner.runner seo_patterns [shorts|long_form]
    python -m self_learner.runner providers
    python -m self_learner.runner content_types
    python -m self_learner.runner recommendations
    python -m self_learner.runner daemon
"""

import json
import sys
import time
from typing import Any, Optional

from self_learner.learner import Learner
from self_learner.seo_learner import SEOLearner
from self_learner.trends import TrendAnalyzer
from self_learner.recommend import RecommendationEngine


def _format_prediction(pred) -> str:
    lines = [f"event_type:      {pred.event_type}",
             f"confidence:      {pred.confidence:.3f}",
             f"supporting_pats: {pred.supporting_patterns}",
             "predicted:"]
    attrs = pred.predicted_attributes or {}
    for k, v in attrs.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def cmd_observe(learner: Learner, args: list[str]) -> int:
    """Record an observation."""
    if len(args) < 2:
        print("usage: observe <event_type> <json_attributes>", file=sys.stderr)
        return 1
    event_type = args[0]
    try:
        attributes = json.loads(args[1])
    except json.JSONDecodeError as e:
        print(f"invalid JSON: {e}", file=sys.stderr)
        return 1
    obs = learner.observe(event_type, attributes)
    print(f"observed {obs.event_type} at {obs.timestamp:.3f}")
    return 0


def cmd_predict(learner: Learner, args: list[str]) -> int:
    """Predict attributes for an event type."""
    if not args:
        print("usage: predict <event_type>", file=sys.stderr)
        return 1
    pred = learner.predict(args[0])
    if pred is None:
        print(f"no pattern for '{args[0]}'", file=sys.stderr)
        return 1
    print(_format_prediction(pred))
    return 0


def cmd_insights(learner: Learner, args: list[str]) -> int:
    """Print human-readable insights."""
    insights = learner.insights()
    if not insights:
        print("no insights yet")
        return 0
    for line in insights:
        print(line)
    return 0


def cmd_stats(learner: Learner, args: list[str]) -> int:
    """Print learner statistics."""
    stats = learner.stats()
    print(f"observations: {stats['total_observations']}")
    print(f"patterns:     {stats['total_patterns']}")
    print(f"memory size:  {stats['memory_size']}")
    print(f"facts:        {stats['facts_count']}")
    return 0


def cmd_trends(learner: Learner, args: list[str]) -> int:
    """Print trend analysis."""
    trend_analyzer = TrendAnalyzer()
    trends = trend_analyzer.get_all_trends()

    if not trends:
        print("no trend data yet")
        return 0

    print("Metric Trends (last 7 days):")
    print("-" * 60)
    for trend in trends:
        arrow = "↑" if trend.direction == "improving" else "↓" if trend.direction == "degrading" else "→"
        print(f"{trend.metric:20s} {arrow} {trend.direction:12s} "
              f"slope={trend.slope:+.3f} conf={trend.confidence:.2f}")

    trend_analyzer.close()
    return 0


def cmd_anomalies(learner: Learner, args: list[str]) -> int:
    """Print anomalies for a metric."""
    if not args:
        print("usage: anomalies <metric>", file=sys.stderr)
        print("metrics: duration, exported_count, failures_count, selected_clips", file=sys.stderr)
        return 1

    metric = args[0]
    trend_analyzer = TrendAnalyzer()
    anomalies = trend_analyzer.detect_anomalies(metric)

    if not anomalies:
        print(f"no anomalies detected for {metric}")
        return 0

    print(f"Anomalies in {metric}:")
    print("-" * 40)
    for a in anomalies:
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(a["timestamp"]))
        print(f"  {ts}: {a['value']:.2f} ({a['deviation']} by {a['z_score']:.1f} stdev)")

    trend_analyzer.close()
    return 0


def cmd_seo_keywords(learner: Learner, args: list[str]) -> int:
    """Print best SEO keywords."""
    content_type = args[0] if args else "all"
    seo_learner = SEOLearner()
    keywords = seo_learner.get_best_keywords(content_type, limit=10)

    if not keywords:
        print(f"no keyword data yet for {content_type}")
        return 0

    print(f"Best Keywords ({content_type}):")
    print("-" * 50)
    for kw in keywords:
        print(f"  {kw['keyword']:20s} engagement={kw['avg_engagement']:.3f} "
              f"count={kw['count']}")

    seo_learner.close()
    return 0


def cmd_seo_patterns(learner: Learner, args: list[str]) -> int:
    """Print best SEO title patterns."""
    content_type = args[0] if args else "all"
    seo_learner = SEOLearner()
    patterns = seo_learner.get_best_title_patterns(content_type, limit=5)

    if not patterns:
        print(f"no pattern data yet for {content_type}")
        return 0

    print(f"Best Title Patterns ({content_type}):")
    print("-" * 50)
    for p in patterns:
        print(f"  {p['pattern']:20s} engagement={p['avg_engagement']:.3f} "
              f"count={p['count']}")

    seo_learner.close()
    return 0


def cmd_providers(learner: Learner, args: list[str]) -> int:
    """Print provider performance."""
    seo_learner = SEOLearner()
    providers = seo_learner.get_provider_performance()

    if not providers:
        print("no provider data yet")
        return 0

    print("Provider Performance:")
    print("-" * 50)
    for provider, stats in providers.items():
        print(f"  {provider:15s} success={stats['success_rate']:.0%} "
              f"engagement={stats['avg_engagement']:.3f} "
              f"uses={stats['total_uses']}")

    seo_learner.close()
    return 0


def cmd_content_types(learner: Learner, args: list[str]) -> int:
    """Print content type performance."""
    seo_learner = SEOLearner()
    types = seo_learner.get_content_type_stats()

    if not types:
        print("no content type data yet")
        return 0

    print("Content Type Performance:")
    print("-" * 50)
    for ctype, stats in types.items():
        print(f"  {ctype:12s} count={stats['count']} "
              f"success={stats['success_rate']:.0%} "
              f"engagement={stats['avg_engagement']:.3f}")

    seo_learner.close()
    return 0


def cmd_recommendations(learner: Learner, args: list[str]) -> int:
    """Print recommendations."""
    rec_engine = RecommendationEngine()
    recommendations = rec_engine.generate_recommendations()

    if not recommendations:
        print("no recommendations yet")
        return 0

    print("Recommendations:")
    print("=" * 60)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. [{rec.priority.upper()}] {rec.title}")
        print(f"   {rec.description}")
        print(f"   Action: {rec.action}")
        print()

    rec_engine.close()
    return 0


def cmd_daemon(learner: Learner, args: list[str]) -> int:
    """Run as a daemon, observing and learning continuously.

    Reads JSON lines from stdin. Each line should be
    ``{"event_type": "...", "attributes": {...}}``.
    """
    print("daemon mode: reading JSON lines from stdin", file=sys.stderr)
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"invalid JSON: {line}", file=sys.stderr)
                continue
            event_type = data.get("event_type")
            attributes = data.get("attributes", {})
            if not event_type:
                print("missing event_type", file=sys.stderr)
                continue
            learner.observe(event_type, attributes)
            print(json.dumps({"status": "ok", "event_type": event_type}))
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("daemon shutting down", file=sys.stderr)
    return 0


_COMMANDS: dict[str, Any] = {
    "observe": cmd_observe,
    "predict": cmd_predict,
    "insights": cmd_insights,
    "stats": cmd_stats,
    "trends": cmd_trends,
    "anomalies": cmd_anomalies,
    "seo_keywords": cmd_seo_keywords,
    "seo_patterns": cmd_seo_patterns,
    "providers": cmd_providers,
    "content_types": cmd_content_types,
    "recommendations": cmd_recommendations,
    "daemon": cmd_daemon,
}


class Runner:
    """CLI runner for the self-learner module.

    Args:
        learner: A ``Learner`` instance. Creates a default one if not provided.
    """

    def __init__(self, learner: Optional[Learner] = None):
        self._learner = learner or Learner()

    def run(self, args: list[str]) -> int:
        """Execute a CLI command.

        Args:
            args: Command-line arguments (without the program name).

        Returns:
            Exit code (0 for success).
        """
        if not args or args[0] in ("-h", "--help"):
            self._show_help()
            return 0

        command = args[0]
        cmd_fn = _COMMANDS.get(command)
        if cmd_fn is None:
            print(f"unknown command: {command}", file=sys.stderr)
            self._show_help()
            return 1

        return cmd_fn(self._learner, args[1:])

    def _show_help(self):
        print("self_learner v2.0.0 -- persistent-memory learning engine",
              file=sys.stderr)
        print(file=sys.stderr)
        print("commands:", file=sys.stderr)
        for name in _COMMANDS:
            print(f"  {name}", file=sys.stderr)
        print(file=sys.stderr)
        print("examples:", file=sys.stderr)
        print("  python -m self_learner.runner observe pipeline_run "
              '\'{"duration":120,"success":true}\'', file=sys.stderr)
        print("  python -m self_learner.runner predict pipeline_run",
              file=sys.stderr)
        print("  python -m self_learner.runner insights", file=sys.stderr)
        print("  python -m self_learner.runner stats", file=sys.stderr)
        print("  python -m self_learner.runner trends", file=sys.stderr)
        print("  python -m self_learner.runner anomalies duration", file=sys.stderr)
        print("  python -m self_learner.runner seo_keywords shorts", file=sys.stderr)
        print("  python -m self_learner.runner seo_patterns shorts", file=sys.stderr)
        print("  python -m self_learner.runner providers", file=sys.stderr)
        print("  python -m self_learner.runner content_types", file=sys.stderr)
        print("  python -m self_learner.runner recommendations", file=sys.stderr)
        print("  python -m self_learner.runner daemon", file=sys.stderr)


def main():
    runner = Runner()
    sys.exit(runner.run(sys.argv[1:]))


if __name__ == "__main__":  # pragma: no cover
    main()
