"""Compact local reporting for the canonical Shorts Intelligence store."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from shorts_intelligence.config import RuntimeConfig
from shorts_intelligence.evaluation import temporal_holdout
from shorts_intelligence.learner import BayesianSegmentLearner
from shorts_intelligence.store import ShortsStore


def build_report(
    store: ShortsStore,
    *,
    learner: BayesianSegmentLearner | None = None,
    top_limit: int = 10,
) -> dict[str, Any]:
    """Build one bounded report from SQLite; no network or legacy files."""
    rows = store.training_rows()
    model = store.latest_model()
    top = sorted(
        rows,
        key=lambda row: (float(row["outcome_score"]), int(row["engaged_views"])),
        reverse=True,
    )[:max(0, top_limit)]
    return {
        "store": store.summary(),
        "model": model,
        "recommendations": store.active_recommendations(limit=8),
        "evaluation": temporal_holdout(rows, learner or BayesianSegmentLearner()),
        "top_shorts": [
            {
                "video_id": row["video_id"],
                "title": row["title"],
                "duration_seconds": row["duration_seconds"],
                "outcome_score": round(float(row["outcome_score"]), 6),
                "engaged_views": int(row["engaged_views"]),
                "average_view_percentage": round(
                    float(row["average_view_percentage"]), 2
                ),
            }
            for row in top
        ],
    }


def load_report(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load config and return local report without contacting YouTube."""
    config = RuntimeConfig.from_yaml(config_path)
    if not config.enabled:
        return {"status": "disabled"}
    with ShortsStore(config.db_path, channel_id=config.channel_id) as store:
        return build_report(store, learner=BayesianSegmentLearner(config.learner))


def render_html(report: dict[str, Any]) -> str:
    """Render a small standalone dashboard with escaped channel data."""
    store = report.get("store") or {}
    model = report.get("model") or {}
    evaluation = report.get("evaluation") or {}
    recommendations = report.get("recommendations") or []
    top_shorts = report.get("top_shorts") or []

    cards = "".join(
        _card(label, value)
        for label, value in (
            ("Shorts", store.get("shorts", 0)),
            ("Cricket", store.get("cricket", 0)),
            ("Snapshots", store.get("snapshots", 0)),
            ("Model samples", model.get("observations", 0)),
        )
    )
    rec_rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{:.3f}</td><td>{}</td></tr>".format(
            _escape(item.get("feature_name", "")),
            _escape(item.get("feature_value", "")),
            float(item.get("effect", 0)),
            int(item.get("sample_count", 0)),
        )
        for item in recommendations
    ) or '<tr><td colspan="4">No statistically supported recommendation.</td></tr>'
    short_rows = "".join(
        "<tr><td>{}</td><td>{:.3f}</td><td>{}</td><td>{:.1f}%</td></tr>".format(
            _escape(item.get("title", "")),
            float(item.get("outcome_score", 0)),
            int(item.get("engaged_views", 0)),
            float(item.get("average_view_percentage", 0)),
        )
        for item in top_shorts
    ) or '<tr><td colspan="4">No cricket Shorts outcomes available.</td></tr>'
    evaluation_text = _escape(json.dumps(
        evaluation,
        ensure_ascii=False,
        separators=(",", ":"),
    ))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cricket Shorts Intelligence</title><style>
body{{font:15px system-ui;background:#0b1020;color:#e8ecf4;margin:0;padding:24px}}main{{max-width:1100px;margin:auto}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}}.card,section{{background:#151d31;border:1px solid #26324d;border-radius:12px;padding:16px;margin:14px 0}}
.value{{font-size:28px;font-weight:700}}table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;text-align:left;border-bottom:1px solid #26324d}}code{{white-space:pre-wrap;color:#a7f3d0}}
</style></head><body><main><h1>Cricket Shorts Intelligence</h1><div class="cards">{cards}</div>
<section><h2>Validated recommendations</h2><table><tr><th>Feature</th><th>Value</th><th>Effect</th><th>N</th></tr>{rec_rows}</table></section>
<section><h2>Top cricket Shorts</h2><table><tr><th>Title</th><th>Outcome</th><th>Engaged views</th><th>Avg viewed</th></tr>{short_rows}</table></section>
<section><h2>Temporal holdout</h2><code>{evaluation_text}</code></section></main></body></html>"""


def _card(label: str, value: Any) -> str:
    return (
        f'<div class="card"><div>{_escape(label)}</div>'
        f'<div class="value">{_escape(value)}</div></div>'
    )


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)
