"""Backward-compatible CLI adapter for the canonical Shorts report."""

from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path

from shorts_intelligence.reporting import load_report, render_html


DEFAULT_OUTPUT = Path("reports/analytics/analytics_report.html")


def generate_html(*, config_path: str = "config.yaml") -> str:
    """Return the canonical local report as standalone HTML."""
    return render_html(load_report(config_path))


def generate_json_dump(*, config_path: str = "config.yaml") -> str:
    """Return the same canonical report as compact JSON."""
    return json.dumps(
        load_report(config_path),
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate canonical cricket Shorts report"
    )
    parser.add_argument("-o", "--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--json", action="store_true", help="Write compact JSON instead of HTML"
    )
    parser.add_argument("--open", action="store_true", help="Open the generated report")
    args = parser.parse_args(argv)

    output = Path(args.output)
    if args.json and output == DEFAULT_OUTPUT:
        output = output.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        generate_json_dump(config_path=args.config)
        if args.json
        else generate_html(config_path=args.config)
    )
    output.write_text(payload, encoding="utf-8")
    print(str(output))
    if args.open:
        webbrowser.open(output.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
