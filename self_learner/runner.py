"""runner.py — CLI entry point for the self-learner module.

Usage:
    python -m self_learner.runner --help
    python -m self_learner.runner observe pipeline_run '{"duration": 120}'
    python -m self_learner.runner predict pipeline_run
    python -m self_learner.runner insights
    python -m self_learner.runner stats
    python -m self_learner.runner daemon
"""

import json
import sys
import time
from typing import Any, Optional

from self_learner.learner import Learner


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
        print("self_learner v1.0.0 -- persistent-memory learning engine",
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
        print("  python -m self_learner.runner daemon", file=sys.stderr)


def main():
    runner = Runner()
    sys.exit(runner.run(sys.argv[1:]))


if __name__ == "__main__":  # pragma: no cover
    main()
