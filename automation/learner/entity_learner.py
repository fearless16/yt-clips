"""EntityLearner — tracks player, team, series, format performance.

20% weight in scoring formula. Uses confidence-weighted EMA and
detects entity fatigue for over-covered topics.
"""

import re
from datetime import datetime, timezone
from typing import Any


def _word_match(keyword: str, text: str) -> bool:
    kw = keyword.rstrip()
    if not kw:
        return False
    return bool(re.search(r'\b' + re.escape(kw) + r'\b', text))


CRICKET_ENTITIES = {
    "players": {
        "kohli": ["kohli", "virat", "king kohli"],
        "bumrah": ["bumrah", "jasprit"],
        "rohit": ["rohit", "hitman", "ro"],
        "gill": ["gill", "shubman"],
        "dhoni": ["dhoni", "msd", "thala", "mahendra"],
        "jadeja": ["jadeja", "sir jadeja", "jaddu"],
        "surya": ["surya", "suryakumar", "sky"],
        "samson": ["samson", "sanju"],
        "hardik": ["hardik", "pandya"],
        "rahul": ["kl rahul", "rahul"],
        "pant": ["pant", "rishabh"],
        "arshdeep": ["arshdeep"],
        "siraj": ["siraj"],
        "ashwin": ["ashwin"],
        "vaibhav": ["vaibhav", "suryavanshi"],
    },
    "teams": {
        "mi": ["mi", "mumbai indians", "mumbai"],
        "rcb": ["rcb", "royal challengers", "bangalore"],
        "csk": ["csk", "chennai super", "chennai"],
        "gt": ["gt", "gujarat titans", "gujarat"],
        "kkr": ["kkr", "kolkata"],
        "dc": ["dc", "delhi capitals", "delhi"],
        "rr": ["rr", "rajasthan royals", "rajasthan"],
        "pbks": ["pbks", "punjab kings", "punjab"],
        "srh": ["srh", "sunrisers", "hyderabad"],
        "lsg": ["lsg", "lucknow"],
    },
    "series": {
        "ipl": ["ipl", "indian premier league"],
        "world_cup": ["world cup", "wc"],
        "asia_cup": ["asia cup"],
        "test": ["test cricket", "test match", "test series"],
        "odi": ["odi", "one day"],
        "t20i": ["t20i", "t20 international"],
    }
}


class EntityLearner:
    """Tracks player, team, series, format performance. 20% weight."""

    def __init__(self, state_store, baseline_views: int = 290):
        """Initialize EntityLearner.

        Args:
            state_store: PersistentStateStore instance
            baseline_views: Channel average views for normalization
        """
        self._state = state_store
        self._baseline = baseline_views

    def process_metrics(self, clip_id: str, payload: dict) -> None:
        """Process a metrics_received event and update entity scores.

        Args:
            clip_id: The clip ID
            payload: Event payload with views, player_tags, team_tags, format_tags
        """
        views = payload.get("views", 0)
        if views == 0:
            return

        signal = min(views / self._baseline, 3.0) / 3.0

        for player in payload.get("player_tags", []):
            self._update_entity("players", player, signal, views)
            self._state.add_recent_upload("players", player)

        for team in payload.get("team_tags", []):
            self._update_entity("teams", team, signal, views)
            self._state.add_recent_upload("teams", team)

        for series in payload.get("format_tags", []):
            self._update_entity("series", series, signal, views)
            self._state.add_recent_upload("series", series)

    def _update_entity(self, entity_type: str, name: str,
                       signal: float, views: int) -> None:
        """Update score for a specific entity using confidence-weighted EMA.

        Args:
            entity_type: "players", "teams", "series", or "formats"
            name: Entity name
            signal: Normalized view signal (0-1)
            views: Raw view count
        """
        current = self._state.get_entity(entity_type, name)
        n = current["n"]

        alpha = max(0.10, 0.30 / (1 + n * 0.05))

        old_score = current["score"]
        new_score = (1 - alpha) * old_score + alpha * signal

        old_avg = current["avg_views"]
        new_avg = (old_avg * n + views) / (n + 1)

        recent_avg = current.get("recent_views_avg", views)
        new_recent = (recent_avg * min(n, 3) + views) / (min(n, 3) + 1)

        if views > recent_avg * 1.3:
            trend = "rising"
        elif views < recent_avg * 0.7:
            trend = "declining"
        else:
            trend = "stable"

        self._state.set_entity(entity_type, name, {
            "score": new_score,
            "n": n + 1,
            "avg_views": new_avg,
            "trend": trend,
            "recent_views_avg": new_recent,
            "recent_uploads": current.get("recent_uploads", [])
        })

    def process_override(self, clip_id: str, payload: dict) -> None:
        """Process a manual_override event. Boosts entity scores.

        Args:
            clip_id: The clip ID
            payload: Event payload with override_type, player_tags, team_tags
        """
        override_type = payload.get("override_type", "")
        if override_type != "keep":
            return

        for player in payload.get("player_tags", []):
            current = self._state.get_entity("players", player)
            current["score"] = min(1.0, current["score"] + 0.15)
            self._state.set_entity("players", player, current)

        for team in payload.get("team_tags", []):
            current = self._state.get_entity("teams", team)
            current["score"] = min(1.0, current["score"] + 0.15)
            self._state.set_entity("teams", team, current)

    def get_entity_score(self, players: list[str], teams: list[str],
                         series: str | None = None) -> float:
        """Get composite entity score for a candidate.

        Args:
            players: List of player names
            teams: List of team names
            series: Optional series name

        Returns:
            Composite score (0-1)
        """
        player_scores = [self._state.get_entity("players", p)["score"] for p in players]
        team_scores = [self._state.get_entity("teams", t)["score"] for t in teams]

        max_player = max(player_scores) if player_scores else 0.5
        max_team = max(team_scores) if team_scores else 0.5

        series_score = 0.5
        if series:
            series_score = self._state.get_entity("series", series)["score"]

        return max_player * 0.5 + max_team * 0.3 + series_score * 0.2

    def fatigue_penalty(self, entity_type: str, name: str,
                        window_days: int = 14) -> float:
        """Calculate fatigue penalty for over-covered entities.

        Args:
            entity_type: "players", "teams", etc.
            name: Entity name
            window_days: Lookback window in days

        Returns:
            Penalty (0-0.30), higher means more fatigued
        """
        recent_uploads = self._state.count_recent(entity_type, name, window_days)
        entity = self._state.get_entity(entity_type, name)
        avg_performance = entity["score"]

        if recent_uploads > 10 and avg_performance < 0.50:
            return min(0.30, (recent_uploads - 10) * 0.03)
        return 0.0

    def novelty_bonus(self, entity_type: str, name: str) -> float:
        """Novelty bonus for under-explored high-performers.

        Gives a score boost to entities that have high avg_views but
        low video count (< 5), encouraging the pipeline to explore them.

        Args:
            entity_type: "players", "teams", etc.
            name: Entity name

        Returns:
            Bonus (0-0.15), higher for under-explored high-performers
        """
        entity = self._state.get_entity(entity_type, name)
        n = entity.get("n", 0)
        score = entity.get("score", 0.0)

        # Only give bonus if: few videos AND high score AND at least 1 observation
        if n < 1 or n >= 5 or score < 0.4:
            return 0.0

        # Scale bonus: fewer videos + higher score = bigger bonus
        # Max bonus at n=1, score=1.0 → 0.15
        novelty_factor = min(1.0, max(0.0, (5 - n) / 4.0))  # 1.0 at n=1, 0.0 at n=5
        quality_factor = min(1.0, (score - 0.4) / 0.6)  # 0.0 at 0.4, 1.0 at 1.0
        return round(novelty_factor * quality_factor * 0.15, 4)

    def get_top_entities(self, entity_type: str, limit: int = 5) -> list[dict]:
        """Get top-performing entities of a type.

        Args:
            entity_type: "players", "teams", "series", or "formats"
            limit: Maximum number of entities to return

        Returns:
            List of dicts with name, score, n, avg_views, trend
        """
        entities = self._state.get_all_entities(entity_type)

        result = []
        for name, data in entities.items():
            result.append({
                "name": name,
                "score": data.get("score", 0.5),
                "n": data.get("n", 0),
                "avg_views": data.get("avg_views", 0),
                "trend": data.get("trend", "stable")
            })

        result.sort(key=lambda x: x["score"], reverse=True)
        return result[:limit]

    def extract_entities(self, title: str, transcript: str = "") -> dict:
        """Extract cricket entities from title and transcript.

        Args:
            title: Video title
            transcript: Optional transcript text

        Returns:
            Dict with players, teams, series lists
        """
        text = (title + " " + transcript).lower()
        found = {"players": [], "teams": [], "series": []}

        for entity_type, entities in CRICKET_ENTITIES.items():
            for name, keywords in entities.items():
                if any(_word_match(kw, text) for kw in keywords):
                    found[entity_type].append(name)

        return found
