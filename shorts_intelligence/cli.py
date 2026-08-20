"""Compact standalone CLI for sync, status, and temporal evaluation."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from shorts_intelligence.config import RuntimeConfig
from shorts_intelligence.learner import BayesianSegmentLearner
from shorts_intelligence.service import ShortsIntelligence
from shorts_intelligence.store import ShortsStore
from shorts_intelligence.youtube_source import YouTubeShortsSource


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(prog="shorts-intelligence")
    parser.add_argument("command", choices=("sync", "status", "evaluate"))
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and emit exactly one JSON document."""
    args = build_parser().parse_args(argv)
    config = RuntimeConfig.from_yaml(args.config)
    if not config.enabled:
        _print({"status": "disabled"}, args.pretty)
        return 0
    with ShortsStore(config.db_path, channel_id=config.channel_id) as store:
        source = YouTubeShortsSource(config.source_config())
        service = ShortsIntelligence(
            store=store,
            source=source,
            learner=BayesianSegmentLearner(config.learner),
        )
        if args.command == "sync":
            result = service.sync_and_fit()
        elif args.command == "evaluate":
            from shorts_intelligence.evaluation import temporal_holdout

            result = temporal_holdout(
                store.training_rows(),
                BayesianSegmentLearner(config.learner),
            )
        else:
            result = service.status()
    _print(result, args.pretty)
    return 0


def _print(payload: dict, pretty: bool) -> None:
    indent = 2 if pretty else None
    separators = None if pretty else (",", ":")
    print(json.dumps(payload, ensure_ascii=False, indent=indent, separators=separators, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
