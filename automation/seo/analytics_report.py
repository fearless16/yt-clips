"""
analytics_report.py — Generate a comprehensive HTML analytics report
from collected SEO performance data + YouTube analytics logs.

Usage:
  python -m automation.seo.analytics_report                          # latest only
  python -m automation.seo.analytics_report --output report.html     # custom path
  python -m automation.seo.analytics_report --open                   # open in browser
  python -m automation.seo.analytics_report --json                   # also save LLM-ready JSON
"""

import json
import sys
import os
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import base64
from io import BytesIO

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT))


STYLES = """
:root {
  --bg: #0a0c10;
  --bg-card: #111318;
  --bg-card-hover: #16181e;
  --border: #1e2028;
  --border-light: #282a34;
  --text: #e4e6ef;
  --text-muted: #7a7d8a;
  --text-dim: #4a4d59;
  --accent: #6366f1;
  --accent-glow: rgba(99,102,241,0.15);
  --green: #22c55e;
  --green-bg: rgba(34,197,94,0.1);
  --red: #ef4444;
  --red-bg: rgba(239,68,68,0.1);
  --yellow: #eab308;
  --yellow-bg: rgba(234,179,8,0.1);
  --blue: #3b82f6;
  --blue-bg: rgba(59,130,246,0.1);
  --purple: #a855f7;
  --purple-bg: rgba(168,85,247,0.1);
  --radius: 12px;
  --radius-sm: 8px;
  --shadow: 0 1px 3px rgba(0,0,0,0.3), 0 1px 2px rgba(0,0,0,0.4);
  --shadow-lg: 0 4px 24px rgba(0,0,0,0.4);
}

* { margin:0; padding:0; box-sizing:border-box; }

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', Roboto, Helvetica, Arial, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}

.container { max-width: 1280px; margin: 0 auto; padding: 32px 24px; }

/* Header */
.header {
  margin-bottom: 32px;
  padding-bottom: 24px;
  border-bottom: 1px solid var(--border);
}
.header h1 {
  font-size: 32px;
  font-weight: 700;
  background: linear-gradient(135deg, var(--accent), var(--purple));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 8px;
}
.header .subtitle {
  color: var(--text-muted);
  font-size: 14px;
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}
.header .subtitle span { display: inline-flex; align-items: center; gap: 4px; }
.header .subtitle .dot {
  width: 6px; height: 6px; border-radius: 50%; display: inline-block;
}
.dot.green { background: var(--green); }
.dot.yellow { background: var(--yellow); }
.dot.red { background: var(--red); }
.dot.blue { background: var(--blue); }
.dot.accent { background: var(--accent); }

/* Grid Cards */
.grid-4 { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-bottom: 24px; }
.grid-3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 24px; }
.grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; margin-bottom: 24px; }

.stat-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px;
  transition: border-color 0.2s, box-shadow 0.2s;
  position: relative;
  overflow: hidden;
}
.stat-card:hover {
  border-color: var(--border-light);
  box-shadow: var(--shadow);
}
.stat-card .label {
  font-size: 12px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-muted);
  margin-bottom: 6px;
}
.stat-card .value {
  font-size: 30px;
  font-weight: 700;
  letter-spacing: -0.02em;
  line-height: 1.1;
}
.stat-card .value.green { color: var(--green); }
.stat-card .value.yellow { color: var(--yellow); }
.stat-card .value.red { color: var(--red); }
.stat-card .value.blue { color: var(--blue); }
.stat-card .value.accent { color: var(--accent); }
.stat-card .sub {
  font-size: 13px;
  color: var(--text-muted);
  margin-top: 4px;
}
.stat-card .trend {
  font-size: 13px;
  margin-top: 2px;
  font-weight: 500;
}
.trend.up { color: var(--green); }
.trend.down { color: var(--red); }
.trend.flat { color: var(--yellow); }

/* Bar */
.bar-track {
  height: 6px;
  background: var(--border);
  border-radius: 3px;
  margin-top: 10px;
  overflow: hidden;
}
.bar-fill {
  height: 100%;
  border-radius: 3px;
  transition: width 0.6s ease;
  background: var(--accent);
}
.bar-fill.green { background: linear-gradient(90deg, var(--green), #16a34a); }
.bar-fill.yellow { background: linear-gradient(90deg, var(--yellow), #ca8a04); }
.bar-fill.red { background: linear-gradient(90deg, var(--red), #dc2626); }

/* Section */
.section {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 24px;
  margin-bottom: 20px;
  transition: border-color 0.2s;
}
.section:hover { border-color: var(--border-light); }
.section-title {
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--text);
}
.section-desc {
  font-size: 13px;
  color: var(--text-muted);
  margin-bottom: 16px;
}

/* Tables */
.table-wrap { overflow-x: auto; }
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
thead th {
  text-align: left;
  padding: 10px 12px;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted);
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}
tbody td {
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
  vertical-align: middle;
}
tbody tr:hover td { background: var(--bg-card-hover); }
tbody tr:last-child td { border-bottom: none; }

.code { font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', monospace; font-size: 12px; }

/* Badge */
.badge {
  display: inline-flex;
  align-items: center;
  padding: 2px 10px;
  border-radius: 20px;
  font-size: 11px;
  font-weight: 600;
  white-space: nowrap;
}
.badge.green { background: var(--green-bg); color: var(--green); }
.badge.red { background: var(--red-bg); color: var(--red); }
.badge.yellow { background: var(--yellow-bg); color: var(--yellow); }
.badge.blue { background: var(--blue-bg); color: var(--blue); }
.badge.purple { background: var(--purple-bg); color: var(--purple); }

/* Mini sparkline */
.mini-chart {
  display: flex;
  align-items: flex-end;
  gap: 2px;
  height: 40px;
  margin: 8px 0;
}
.mini-bar {
  width: 8px;
  border-radius: 2px 2px 0 0;
  min-height: 2px;
  transition: height 0.3s;
}

/* Insight blocks */
.insight-box {
  border-left: 3px solid var(--accent);
  background: var(--accent-glow);
  padding: 16px 20px;
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  margin: 12px 0;
  line-height: 1.7;
  font-size: 14px;
  white-space: pre-wrap;
}
.insight-box.start { border-left-color: var(--green); background: var(--green-bg); }
.insight-box.stop { border-left-color: var(--red); background: var(--red-bg); }
.insight-box.continue { border-left-color: var(--blue); background: var(--blue-bg); }

/* Recommendation list */
.rec-list { list-style: none; padding: 0; }
.rec-list li {
  padding: 12px 16px;
  margin-bottom: 8px;
  background: var(--bg);
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
  font-size: 14px;
  display: flex;
  align-items: flex-start;
  gap: 10px;
}
.rec-list li .icon { flex-shrink: 0; font-size: 16px; line-height: 1.4; }
.rec-list li .rec-text { flex: 1; }
.rec-list li .rec-tag {
  flex-shrink: 0;
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 4px;
  font-weight: 600;
}
.rec-tag.high { background: var(--green-bg); color: var(--green); }
.rec-tag.medium { background: var(--yellow-bg); color: var(--yellow); }
.rec-tag.low { background: var(--red-bg); color: var(--red); }

/* No data */
.no-data {
  color: var(--text-muted);
  text-align: center;
  padding: 32px 20px;
  font-style: italic;
  font-size: 14px;
}

/* JSON dump */
.json-dump {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 16px;
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 12px;
  line-height: 1.5;
  white-space: pre-wrap;
  overflow-x: auto;
  max-height: 400px;
  overflow-y: auto;
  color: var(--text-muted);
}

/* Footer */
.footer {
  text-align: center;
  color: var(--text-dim);
  font-size: 12px;
  margin-top: 48px;
  padding: 24px;
  border-top: 1px solid var(--border);
}

/* Animations */
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}
.fade-in { animation: fadeInUp 0.4s ease both; }
.delay-1 { animation-delay: 0.05s; }
.delay-2 { animation-delay: 0.1s; }
.delay-3 { animation-delay: 0.15s; }
.delay-4 { animation-delay: 0.2s; }

@media (max-width: 768px) {
  .container { padding: 16px; }
  .grid-4, .grid-3, .grid-2 { grid-template-columns: 1fr; }
  .stat-card .value { font-size: 24px; }
  .header h1 { font-size: 24px; }
}
"""


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>yt-clips Analytics Report</title>
<style>{styles}</style>
</head>
<body>
<div class="container">
{content}
<div class="footer">
  Generated by <strong>yt-clips Analytics Engine</strong> · {generated_at} ·
  <a href="#" style="color:var(--accent);text-decoration:none" onclick="document.getElementById('llm-json').scrollIntoView({{behavior:'smooth'}})">Jump to LLM Data ↓</a>
</div>
</div>
</body>
</html>"""


def fmt(n, suffix=""):
    if n is None:
        return "—"
    if isinstance(n, float):
        return f"{n:.1f}{suffix}"
    if isinstance(n, int):
        s = str(n)
        if n >= 1_000_000:
            s = f"{n/1_000_000:.1f}M"
        elif n >= 1_000:
            s = f"{n/1_000:.1f}K"
        return f"{s}{suffix}"
    return f"{n}{suffix}"


def mini_sparkline(vals, color="#6366f1", height=40):
    if not vals:
        return ""
    mx = max(vals) or 1
    bars = "".join(
        f'<div class="mini-bar" style="height:{max(3, v/mx*height):.0f}px;background:{color}"></div>'
        for v in vals[-24:]
    )
    return f'<div class="mini-chart">{bars}</div>'


def sparkbar(val, max_val, color="accent"):
    pct = min(100, (val / max_val * 100)) if max_val > 0 else 0
    cls = color if color in ("green", "yellow", "red") else ""
    return f'<div class="bar-track"><div class="bar-fill {cls}" style="width:{pct:.0f}%"></div></div>'


def load_performance_data() -> Dict:
    path = ROOT / "data/seo_performance.json"
    if not path.exists():
        return {"clips": [], "title_patterns": {}, "hashtag_performance": {},
                "hooks_performance": {}, "ctas_performance": {},
                "model_performance": {}, "benchmark_history": [],
                "feature_importance": {}, "llm_insights": []}
    with open(path) as f:
        return json.load(f)


def load_latest_analytics() -> Optional[Dict]:
    log_dir = ROOT / "logs"
    files = sorted(log_dir.glob("analytics_*.json"), reverse=True)
    if not files:
        return None
    with open(files[0]) as f:
        return json.load(f)


def load_all_analytics() -> List[Dict]:
    log_dir = ROOT / "logs"
    files = sorted(log_dir.glob("analytics_*.json"))
    results = []
    for p in files:
        with open(p) as f:
            data = json.load(f)
        ts = p.stem.replace("analytics_", "").replace("_", "T")
        results.append({"timestamp": ts, "data": data})
    return results


def build_header(pd, ad, history):
    clips = pd.get("clips", [])
    total = len(clips)
    unique = len(set(c["clip_id"] for c in clips))
    shorts_count = len(ad.get("shorts", [])) if ad else 0

    daily_views = []
    for h in history:
        v = sum(s.get("views", 0) for s in h["data"].get("shorts", []))
        daily_views.append(v)

    view_trend = ""
    if len(daily_views) >= 2:
        trend = daily_views[-1] - daily_views[-2]
        if trend > 0:
            view_trend = f'<span class="trend up">▲ +{trend}</span> vs yesterday'
        elif trend < 0:
            view_trend = f'<span class="trend down">▼ {trend}</span> vs yesterday'
        else:
            view_trend = '<span class="trend flat">—</span> vs yesterday'

    return f"""
<div class="header fade-in">
  <h1>📊 yt-clips Analytics</h1>
  <div class="subtitle">
    <span><span class="dot accent"></span> {total} clips tracked</span>
    <span><span class="dot blue"></span> {unique} unique videos</span>
    <span><span class="dot green"></span> {shorts_count} shorts in latest fetch</span>
    <span>{view_trend}</span>
  </div>
</div>"""


def build_overview_cards(pd, ad):
    clips = pd.get("clips", [])
    scores = [c["performance_score"] for c in clips]
    avg_score = sum(scores) / len(scores) if scores else 0
    max_score = max(scores) if scores else 0

    total_views = 0
    total_likes = 0
    total_comments = 0
    shorts_count = 0
    if ad:
        for s in ad.get("shorts", []):
            total_views += s.get("views", 0)
            total_likes += s.get("likes", 0)
            total_comments += s.get("comments", 0)
            shorts_count += 1

    eng_rate = ((total_likes + total_comments) / max(1, total_views)) * 100 if total_views else 0

    score_color = "green" if avg_score >= 0.5 else ("yellow" if avg_score >= 0.35 else "red")
    return f"""
<div class="grid-4">
  <div class="stat-card fade-in delay-1">
    <div class="label">Avg Performance Score</div>
    <div class="value {score_color}">{avg_score:.3f}</div>
    <div class="sub">range: 0 – {max_score:.3f}</div>
    {sparkbar(avg_score, 1.0, score_color)}
  </div>
  <div class="stat-card fade-in delay-2">
    <div class="label">Total Shorts Views</div>
    <div class="value blue">{fmt(total_views)}</div>
    <div class="sub">{shorts_count} shorts · avg {fmt(total_views // max(1, shorts_count))}/short</div>
    {sparkbar(min(1.0, total_views / max(1, shorts_count) / 500), 1.0, "blue")}
  </div>
  <div class="stat-card fade-in delay-3">
    <div class="label">Total Engagement</div>
    <div class="value green">{fmt(total_likes + total_comments)}</div>
    <div class="sub">👍 {fmt(total_likes)} likes · 💬 {fmt(total_comments)} comments</div>
    {sparkbar(min(1.0, (total_likes + total_comments) / 500), 1.0, "green")}
  </div>
  <div class="stat-card fade-in delay-4">
    <div class="label">Engagement Rate</div>
    <div class="value {'green' if eng_rate >= 5 else ('yellow' if eng_rate >= 2 else 'red')}">{eng_rate:.1f}%</div>
    <div class="sub">(likes + comments) / views</div>
    {sparkbar(min(1.0, eng_rate / 20), 1.0, "green" if eng_rate >= 5 else ("yellow" if eng_rate >= 2 else "red"))}
  </div>
</div>"""


def build_daily_trend(history):
    daily_views = []
    daily_labels = []
    for h in history:
        v = sum(s.get("views", 0) for s in h["data"].get("shorts", []))
        daily_views.append(v)
        ts = h["timestamp"][:10]
        daily_labels.append(ts)

    if not daily_views:
        return ""

    labels = "".join(f"<span>{l[-5:]}</span>" for l in daily_labels[-7:])
    return f"""
<div class="section fade-in">
  <div class="section-title">📈 Daily Shorts Views</div>
  {mini_sparkline(daily_views, '#22c55e')}
  <div style="display:flex;gap:16px;flex-wrap:wrap;font-size:11px;color:var(--text-muted);margin-top:4px;">
    {labels}
  </div>
</div>"""


def build_llm_insights_section(pd):
    llm = pd.get("llm_insights", [])
    if not llm:
        return ""

    latest = llm[-1].get("insights", "")
    ts = llm[-1].get("timestamp", "")[:10]

    start_items = []
    stop_items = []
    continue_items = []
    for line in latest.split("\n"):
        line = line.strip()
        if not line:
            continue
        clean = line.lstrip("* -0123456789.)")
        if any(w in line.lower() for w in ["start", "do this", "begin", "add", "introduce", "use more", "incorporate"]):
            start_items.append(clean)
        elif any(w in line.lower() for w in ["stop", "avoid", "don't", "remove", "reduce", "cut", "eliminate"]):
            stop_items.append(clean)
        else:
            continue_items.append(clean)

    boxes = ""
    if start_items:
        items = "".join(f"<div>✅ <strong>START:</strong> {s}</div>" for s in start_items)
        boxes += f'<div class="insight-box start">{items}</div>'
    if stop_items:
        items = "".join(f"<div>❌ <strong>STOP:</strong> {s}</div>" for s in stop_items)
        boxes += f'<div class="insight-box stop">{items}</div>'
    if continue_items:
        items = "".join(f"<div>🔄 <strong>CONTINUE:</strong> {s}</div>" for s in continue_items)
        boxes += f'<div class="insight-box continue">{items}</div>'

    if not boxes:
        boxes = f'<div class="insight-box">{latest}</div>'

    return f"""
<div class="section fade-in">
  <div class="section-title">🧠 AI-Powered Insights</div>
  <div class="section-desc">Generated from performance data · {ts}</div>
  {boxes}
</div>"""


def build_top_shorts(ad):
    if not ad:
        return ""
    shorts = ad.get("shorts", [])
    if not shorts:
        return ""

    sorted_shorts = sorted(shorts, key=lambda x: -x.get("views", 0))[:10]
    rows = ""
    for i, s in enumerate(sorted_shorts):
        views = s.get("views", 0)
        likes = s.get("likes", 0)
        comments = s.get("comments", 0)
        eng = ((likes + comments) / max(1, views)) * 100
        badge = "green" if i < 3 else ("blue" if i < 6 else "yellow")
        rows += f"""<tr>
          <td><span class="badge {badge}">#{i+1}</span></td>
          <td class="code" style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{s.get('title','?')[:60]}</td>
          <td style="font-weight:600">{fmt(views)}</td>
          <td>{fmt(likes)}</td>
          <td>{fmt(comments)}</td>
          <td>{eng:.1f}%</td>
          <td>{sparkbar(min(100, views/20), 100, "green" if views >= 200 else ("yellow" if views >= 100 else "red"))}</td>
        </tr>"""

    return f"""
<div class="section fade-in">
  <div class="section-title">🔥 Top Performing Shorts</div>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Rank</th><th>Title</th><th>Views</th><th>Likes</th><th>Comments</th><th>Eng Rate</th><th></th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>"""


def build_performance_scores(pd):
    clips = pd.get("clips", [])
    if not clips:
        return ""

    scores = [(c["performance_score"], c.get("clip_id", "?"), c.get("timestamp", "")[:10], c.get("provider", "?"), c.get("model", "?")) for c in clips]
    scores.sort(key=lambda x: -x[0])

    rows = ""
    for i, (score, cid, ts, prov, model) in enumerate(scores[:15]):
        color = "green" if score >= 0.5 else ("yellow" if score >= 0.35 else "red")
        badge = f"badge {color}"
        pct = score / max(s[0] for s in scores) * 100
        rows += f"""<tr>
          <td><span class="{badge}">#{i+1}</span></td>
          <td class="code">{cid[:12]}</td>
          <td>{ts}</td>
          <td>{fmt(prov)}/{fmt(model)}</td>
          <td style="font-weight:600;color:var(--{color})">{score:.4f}</td>
          <td>{sparkbar(pct, 100, color)}</td>
        </tr>"""

    return f"""
<div class="section fade-in">
  <div class="section-title">📊 Performance Scores</div>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Rank</th><th>Clip ID</th><th>Date</th><th>Provider/Model</th><th>Score</th><th></th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
  {f'<div style="color:var(--text-muted);font-size:13px;margin-top:8px">… and {len(scores) - 15} more</div>' if len(scores) > 15 else ''}
</div>"""


def build_feature_importance(pd):
    fi = pd.get("feature_importance", {})
    if not fi:
        return ""

    rows = ""
    for feat, data in sorted(fi.items(), key=lambda x: -abs(x[1]["delta"])):
        delta = data["delta"]
        color = "green" if delta > 0 else "red"
        arrow = "▲" if delta > 0 else "▼"
        label = feat.replace("has_", "").replace("_", " ").title()
        impact = "Positive" if delta > 0 else "Negative"
        rows += f"""<tr>
          <td>{arrow} {label}</td>
          <td style="color:var(--{color});font-weight:600">{delta:+.3f}</td>
          <td>{data['count_with']} / {data['count_without']}</td>
          <td><span class="badge {color}">{impact}</span></td>
          <td>{sparkbar(abs(delta), max(abs(v["delta"]) for v in fi.values()), color)}</td>
        </tr>"""

    return f"""
<div class="section fade-in">
  <div class="section-title">🔬 Feature Importance</div>
  <div class="section-desc">Score delta when feature is present vs absent (time-decayed)</div>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Feature</th><th>Delta</th><th>Samples (yes/no)</th><th>Impact</th><th></th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>"""


def build_title_patterns(pd):
    tp = pd.get("title_patterns", {})
    if not tp:
        return ""

    patterns = [(k, v["avg_score"], v["count"], v["total_score"]) for k, v in tp.items() if v["count"] >= 2]
    patterns.sort(key=lambda x: -x[1])

    rows = ""
    for pattern, avg, count, total in patterns[:12]:
        color = "green" if avg >= 0.5 else ("yellow" if avg >= 0.35 else "red")
        badge = f"badge {color}"
        readable = pattern.replace("_", " · ").replace(":", ": ")[:80]
        rows += f"""<tr>
          <td class="code" style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{readable}</td>
          <td><span class="{badge}">{avg:.3f}</span></td>
          <td>{count}x</td>
          <td>{sparkbar(avg, 1.0, color)}</td>
        </tr>"""

    if not rows:
        return ""

    best = patterns[0] if patterns else None
    worst = patterns[-1] if len(patterns) > 1 else None
    insights = ""
    if best and best[1] >= 0.5:
        insights += f'<div style="margin-bottom:6px;font-size:13px">✅ <strong>Best pattern:</strong> avg {best[1]:.3f} across {best[2]} clips</div>'
    if worst and worst[1] < 0.4:
        insights += f'<div style="font-size:13px">❌ <strong>Worst pattern:</strong> avg {worst[1]:.3f} across {worst[2]} clips — review</div>'

    return f"""
<div class="section fade-in">
  <div class="section-title">🎯 Title Patterns</div>
  {insights}
  <div class="table-wrap">
    <table>
      <thead><tr><th>Pattern</th><th>Avg Score</th><th>Count</th><th></th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>"""


def build_hashtags(pd):
    hp = pd.get("hashtag_performance", {})
    if not hp:
        return ""

    entries = [(k, sum(v) / len(v) if v else 0, len(v)) for k, v in hp.items()]
    entries.sort(key=lambda x: -x[1])

    rows = ""
    for tag, avg, count in entries[:10]:
        color = "green" if avg >= 0.5 else ("yellow" if avg >= 0.35 else "red")
        badge = f"badge {color}"
        rows += f"""<tr>
          <td class="code">{tag}</td>
          <td><span class="{badge}">{avg:.3f}</span></td>
          <td>{count}x</td>
          <td>{sparkbar(avg, 1.0, color)}</td>
        </tr>"""

    return f"""
<div class="section fade-in">
  <div class="section-title">#️⃣ Hashtag Performance</div>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Hashtag Feature</th><th>Avg Score</th><th>Count</th><th></th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>"""


def build_model_performance(pd):
    mp = pd.get("model_performance", {})
    if not mp:
        return ""

    entries = [(k, v["avg_score"], v["count"]) for k, v in mp.items()]
    entries.sort(key=lambda x: -x[1])

    rows = ""
    for key, avg, count in entries:
        color = "green" if avg >= 0.5 else "yellow"
        rows += f"""<tr>
          <td class="code">{key}</td>
          <td style="font-weight:600;color:var(--{color})">{avg:.3f}</td>
          <td>{count}x</td>
          <td>{sparkbar(avg, 1.0, color)}</td>
        </tr>"""

    best = pd.get("current_best_provider"), pd.get("current_best_model")
    best_str = f"{best[0]}/{best[1]}" if best[0] else "Not selected"

    return f"""
<div class="section fade-in">
  <div class="section-title">🤖 Model Performance</div>
  <div style="margin-bottom:12px;font-size:14px">🏆 <strong>Current best:</strong> {best_str}</div>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Provider/Model</th><th>Avg Score</th><th>Count</th><th></th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>"""


def build_recommendations(pd):
    clips = pd.get("clips", [])
    tp = pd.get("title_patterns", {})
    hp = pd.get("hashtag_performance", {})
    fi = pd.get("feature_importance", {})

    recs = []

    if fi:
        for feat, data in sorted(fi.items(), key=lambda x: -abs(x[1]["delta"]))[:3]:
            label = feat.replace("has_", "").replace("_", " ").title()
            if data["delta"] > 0.03:
                recs.append((f"Include **{label}** in titles — boosts score by +{data['delta']:.0%}", "high"))
            elif data["delta"] < -0.03:
                recs.append((f"Avoid **{label}** — reduces score by {abs(data['delta']):.0%}", "low"))

    if tp:
        patterns = [(k, v["avg_score"]) for k, v in tp.items() if v["count"] >= 2]
        patterns.sort(key=lambda x: -x[1])
        if patterns:
            best = patterns[0]
            readable = best[0].replace("_", " · ").replace(":", ": ")[:60]
            recs.append((f"Best title pattern: `{readable}` (score: {best[1]:.3f}) — use similar", "high"))
            if len(patterns) > 1 and patterns[-1][1] < 0.35:
                recs.append((f"Worst title pattern scores {patterns[-1][1]:.3f} — avoid this structure", "low"))

    if clips:
        with_pipe = [c for c in clips if c["features"].get("has_pipe_format")]
        without_pipe = [c for c in clips if not c["features"].get("has_pipe_format")]
        if with_pipe and without_pipe:
            avg_pipe = sum(c["performance_score"] for c in with_pipe) / len(with_pipe)
            avg_no = sum(c["performance_score"] for c in without_pipe) / len(without_pipe)
            if avg_pipe > avg_no * 1.05:
                recs.append((f"Pipe format (Team vs Team | Tournament) outperforms by {((avg_pipe/avg_no)-1)*100:.0f}%", "high"))

    if hp:
        entries = [(k, sum(v)/len(v) if v else 0) for k, v in hp.items()]
        entries.sort(key=lambda x: -x[1])
        if entries and entries[0][1] > 0.5:
            recs.append((f"Hashtag pattern `{entries[0][0]}` most effective (avg {entries[0][1]:.3f})", "medium"))

    if not recs:
        recs.append(("More data needed for recommendations — keep publishing!", "medium"))

    items = "".join(
        f"""<li>
          <span class="icon">{'✅' if level == 'high' else '📌' if level == 'medium' else '⚠️'}</span>
          <span class="rec-text">{text}</span>
          <span class="rec-tag {level}">{level.upper()}</span>
        </li>"""
        for text, level in recs
    )

    return f"""
<div class="section fade-in">
  <div class="section-title">💡 Actionable Recommendations</div>
  <ul class="rec-list">{items}</ul>
</div>"""


def build_llm_json_dump(pd, ad):
    """Build a clean JSON dump of all data for LLM consumption."""
    clips = pd.get("clips", [])
    clean_data = {
        "report_metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_clips": len(clips),
            "unique_videos": len(set(c["clip_id"] for c in clips)),
            "time_decay_half_life_days": pd.get("config", {}).get("decay_half_life_days", 30),
        },
        "overview": {
            "avg_performance_score": round(sum(c["performance_score"] for c in clips) / max(1, len(clips)), 4) if clips else 0,
            "score_range": {
                "min": round(min(c["performance_score"] for c in clips), 4) if clips else 0,
                "max": round(max(c["performance_score"] for c in clips), 4) if clips else 0,
            },
        },
        "feature_importance": pd.get("feature_importance", {}),
        "title_patterns": {
            k: {"avg_score": v["avg_score"], "count": v["count"]}
            for k, v in pd.get("title_patterns", {}).items()
            if v["count"] >= 2
        },
        "hashtag_performance": {
            k: {"avg_score": round(sum(v) / len(v), 4) if v else 0, "count": len(v)}
            for k, v in pd.get("hashtag_performance", {}).items()
        },
        "model_performance": pd.get("model_performance", {}),
        "best_model": {
            "provider": pd.get("current_best_provider"),
            "model": pd.get("current_best_model"),
        },
        "llm_insights": pd.get("llm_insights", []),
        "recent_analytics": {
            "shorts_count": len(ad.get("shorts", [])) if ad else 0,
            "videos_count": len(ad.get("videos", [])) if ad else 0,
            "lives_count": len(ad.get("lives", [])) if ad else 0,
            "total_shorts_views": sum(s.get("views", 0) for s in (ad.get("shorts", []) if ad else [])),
            "total_shorts_likes": sum(s.get("likes", 0) for s in (ad.get("shorts", []) if ad else [])),
            "total_shorts_comments": sum(s.get("comments", 0) for s in (ad.get("shorts", []) if ad else [])),
        } if ad else {},
    }

    json_str = json.dumps(clean_data, indent=2)
    return f"""
<div id="llm-json" class="section fade-in">
  <div class="section-title">🤖 LLM-Ready Data</div>
  <div class="section-desc">
    Paste this JSON into your AI prompt for data-driven SEO optimization advice.
    Copy the entire block below for the most accurate recommendations.
  </div>
  <div class="json-dump">{json_str}</div>
</div>"""


def generate_html():
    pd = load_performance_data()
    ad = load_latest_analytics()
    history = load_all_analytics()

    sections = [
        build_header(pd, ad, history),
        build_overview_cards(pd, ad),
        build_daily_trend(history),
        build_llm_insights_section(pd),
        build_recommendations(pd),
        build_top_shorts(ad),
        build_performance_scores(pd),
        build_feature_importance(pd),
        build_title_patterns(pd),
        build_hashtags(pd),
        build_model_performance(pd),
        build_llm_json_dump(pd, ad),
    ]

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    content = "\n".join(sections)
    return HTML_TEMPLATE.format(styles=STYLES, content=content, generated_at=now)


def generate_json_dump() -> str:
    """Generate only the LLM-ready JSON."""
    pd = load_performance_data()
    ad = load_latest_analytics()
    clips = pd.get("clips", [])

    clean_data = {
        "report_metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_clips": len(clips),
            "unique_videos": len(set(c["clip_id"] for c in clips)),
            "time_decay_half_life_days": pd.get("config", {}).get("decay_half_life_days", 30),
        },
        "overview": {
            "avg_performance_score": round(sum(c["performance_score"] for c in clips) / max(1, len(clips)), 4) if clips else 0,
            "score_range": {
                "min": round(min(c["performance_score"] for c in clips), 4) if clips else 0,
                "max": round(max(c["performance_score"] for c in clips), 4) if clips else 0,
            },
        },
        "feature_importance": pd.get("feature_importance", {}),
        "title_patterns": {
            k: {"avg_score": v["avg_score"], "count": v["count"]}
            for k, v in pd.get("title_patterns", {}).items()
            if v["count"] >= 2
        },
        "hashtag_performance": {
            k: {"avg_score": round(sum(v) / len(v), 4) if v else 0, "count": len(v)}
            for k, v in pd.get("hashtag_performance", {}).items()
        },
        "model_performance": pd.get("model_performance", {}),
        "best_model": {"provider": pd.get("current_best_provider"), "model": pd.get("current_best_model")},
        "llm_insights": pd.get("llm_insights", []),
        "recent_analytics": {
            "shorts_count": len(ad.get("shorts", [])) if ad else 0,
            "videos_count": len(ad.get("videos", [])) if ad else 0,
            "lives_count": len(ad.get("lives", [])) if ad else 0,
            "total_shorts_views": sum(s.get("views", 0) for s in (ad.get("shorts", []) if ad else [])),
            "total_shorts_likes": sum(s.get("likes", 0) for s in (ad.get("shorts", []) if ad else [])),
            "total_shorts_comments": sum(s.get("comments", 0) for s in (ad.get("shorts", []) if ad else [])),
        } if ad else {},
    }
    return json.dumps(clean_data, indent=2)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate analytics HTML report")
    parser.add_argument("-o", "--output", default="reports/analytics/analytics_report.html",
                        help="Output HTML path (default: reports/analytics/analytics_report.html)")
    parser.add_argument("--open", action="store_true", help="Open in browser")
    parser.add_argument("--json", action="store_true", help="Also save LLM-ready JSON")
    args = parser.parse_args()

    html = generate_html()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    size_kb = len(html) / 1024
    print(f"✅ Report saved: {out_path} ({size_kb:.0f} KB)")

    if args.json:
        json_path = out_path.with_suffix(".json")
        json_path.write_text(generate_json_dump())
        print(f"✅ JSON saved: {json_path}")

    if args.open:
        import subprocess
        if sys.platform == "darwin":
            subprocess.run(["open", str(out_path)], check=True)
        print("✅ Opened in browser")

    return str(out_path.absolute())


if __name__ == "__main__":
    main()
