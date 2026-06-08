"""weight_learner.py — learns adaptive agent weights from historical performance.

Reads ``clip_learner.db``, computes correlation between each agent's score
and actual video views, and produces updated weights that shift emphasis
toward agents that best predict high-performing clips.

Closed-loop: orchestrator Stage 9 → recalibrate_weights() → updated weights
→ injected into next Stage 3-5 run → better clip selection over time.
"""

import json
from pathlib import Path
from typing import Any

from utils.logger import get_logger

log = get_logger("weight_learner")

DEFAULT_WEIGHTS: dict[str, float] = {
    "hook_expert": 0.35,
    "emotion_expert": 0.20,
    "viral_potential": 0.15,
    "cricket_context": 0.10,
    "viewer_psychology": 0.10,
    "retention_expert": 0.05,
    "technical_quality": 0.05,
}


def load_performance_data(
    db_path: str | Path = "clip_learner.db",
) -> list[dict[str, Any]]:
    """Load clip performance records with agent scores + views.

    Returns list of records with deserialized agent_scores_json.
    """
    import sqlite3

    path = Path(db_path)
    if not path.exists():
        log.info("No clip_learner.db found — returning empty")
        return []

    conn = sqlite3.connect(str(path))
    try:
        cur = conn.execute(
            "SELECT clip_id, final_score, agent_scores_json, views, "
            "selected_rank, youtube_video_id "
            "FROM clip_performance WHERE views > 0 ORDER BY views DESC"
        )
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        records = []
        for row in rows:
            rec = dict(zip(columns, row))
            try:
                rec["agent_scores"] = json.loads(rec.get("agent_scores_json", "{}"))
            except (json.JSONDecodeError, TypeError):
                rec["agent_scores"] = {}
            records.append(rec)
        return records
    finally:
        conn.close()


def compute_adaptive_weights(
    performance_data: list[dict],
    min_clips: int = 10,
    drift_rate: float = 0.3,
) -> dict[str, float]:
    """Compute agent weights based on correlation between scores and views.

    Algorithm:
        1. For each agent, compute a "predictiveness" score = how well
           the agent's score correlates with actual views.
        2. Blend the learned weights with defaults using drift_rate
           (0.3 = 30% learned, 70% default), preventing wild swings
           from small sample sizes.
        3. Normalize so all agent weights sum to 1.0.

    Args:
        performance_data: List from load_performance_data()
        min_clips: Minimum records needed before trusting learned weights
        drift_rate: How much to shift toward learned weights (0-1)

    Returns:
        Dict of agent_name → weight (sums to 1.0)
    """
    if len(performance_data) < min_clips:
        log.info(
            "Not enough performance data for learning (%d < %d) — using defaults",
            len(performance_data), min_clips,
        )
        return dict(DEFAULT_WEIGHTS)

    agent_names = list(DEFAULT_WEIGHTS.keys())
    agent_names.remove("technical_quality")  # no agent for this

    agent_totals: dict[str, float] = {name: 0.0 for name in agent_names}
    view_buckets: dict[str, list[tuple[float, float]]] = {
        name: [] for name in agent_names
    }

    for rec in performance_data:
        agent_scores = rec.get("agent_scores", {})
        views = rec.get("views", 0)
        if not views or not agent_scores:
            continue
        for name in agent_names:
            raw = agent_scores.get(name, {}).get("score", 0)
            agent_totals[name] += raw
            view_buckets[name].append((raw, float(views)))

    agent_scores_list: list[dict] = []
    for name in agent_names:
        buckets = view_buckets[name]
        if len(buckets) < 2:
            agent_scores_list.append({"name": name, "correlation_score": 0.0})
            continue

        xs = [b[0] for b in buckets]
        ys = [b[1] for b in buckets]

        avg_x = sum(xs) / len(xs)
        avg_y = sum(ys) / len(ys)

        numerator = sum((x - avg_x) * (y - avg_y) for x, y in zip(xs, ys))
        std_x = (sum((x - avg_x) ** 2 for x in xs) / len(xs)) ** 0.5
        std_y = (sum((y - avg_y) ** 2 for y in ys) / len(ys)) ** 0.5

        if std_x > 0 and std_y > 0:
            corr = numerator / (len(xs) * std_x * std_y)
        else:
            corr = 0.0

        agent_scores_list.append({
            "name": name,
            "correlation_score": round(corr, 4),
            "n_samples": len(buckets),
            "avg_agent_score": round(avg_x, 1),
            "avg_views": round(avg_y, 0),
        })

    agent_scores_list.sort(key=lambda x: x["correlation_score"], reverse=True)

    log.info(
        "Agent-view correlations (n=%d): %s",
        len(performance_data),
        ", ".join(
            f"{a['name']}={a['correlation_score']:.3f}"
            for a in agent_scores_list
        ),
    )

    # Convert correlations to weights (softmax-style, only positive correlations)
    learned_weights: dict[str, float] = {}
    total = 0.0
    for a in agent_scores_list:
        w = max(0.01, a["correlation_score"])
        learned_weights[a["name"]] = w
        total += w

    if total > 0:
        learned_weights = {k: v / total for k, v in learned_weights.items()}

    # Blend with defaults
    blended: dict[str, float] = {}
    for name in agent_names:
        default = DEFAULT_WEIGHTS[name]
        learned = learned_weights.get(name, default)
        blended[name] = default * (1 - drift_rate) + learned * drift_rate

    # Add back technical_quality
    blended["technical_quality"] = DEFAULT_WEIGHTS["technical_quality"]

    # Re-normalize
    total = sum(blended.values())
    if total > 0:
        blended = {k: round(v / total, 4) for k, v in blended.items()}

    log.info("Adaptive weights: %s", blended)
    return blended


def recalibrate_weights(
    db_path: str | Path = "clip_learner.db",
    min_clips: int = 10,
    drift_rate: float = 0.3,
) -> dict[str, float]:
    """Convenience: load data + compute adaptive weights."""
    data = load_performance_data(db_path)
    return compute_adaptive_weights(data, min_clips=min_clips, drift_rate=drift_rate)


def load_entity_biases(
    self_learner_db: str | Path = "self_learner.db",
) -> dict[str, Any]:
    """Load learned entity scores for injection into agent context.

    Returns dict with top_players and top_teams for CricketContextExpert bias.
    """
    import sqlite3

    path = Path(self_learner_db)
    if not path.exists():
        return {}

    conn = sqlite3.connect(str(path))
    try:
        result: dict[str, Any] = {}

        row = conn.execute(
            "SELECT value_json FROM learned_state WHERE state_key='entity_scores'"
        ).fetchone()
        if row:
            data = json.loads(row[0])
            players = data.get("players", {})
            if players:
                top_players = sorted(
                    players.items(),
                    key=lambda x: x[1].get("score", 0),
                    reverse=True,
                )[:10]
                result["top_players"] = top_players
                result["avoid_players"] = [
                    name for name, info in players.items()
                    if info.get("n", 0) > 5 and info.get("avg_views", 0) < 150
                ][:5]

            teams = data.get("teams", {})
            if teams:
                top_teams = sorted(
                    teams.items(),
                    key=lambda x: x[1].get("score", 0),
                    reverse=True,
                )[:5]
                result["top_teams"] = top_teams
                result["avoid_teams"] = [
                    name for name, info in teams.items()
                    if info.get("n", 0) > 5 and info.get("avg_views", 0) < 150
                ][:5]

        row = conn.execute(
            "SELECT value_json FROM learned_state WHERE state_key='format_scores'"
        ).fetchone()
        if row:
            result["format_scores"] = json.loads(row[0])

        row = conn.execute(
            "SELECT value_json FROM learned_state WHERE state_key='timing_scores'"
        ).fetchone()
        if row:
            result["timing_scores"] = json.loads(row[0])

        return result
    finally:
        conn.close()
