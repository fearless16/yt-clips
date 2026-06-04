"""CricketRecommendationEngine — generates actionable recommendations.

Combines insights from all learners to recommend next topics,
players, hooks, title styles, durations, and upload times.
Each recommendation includes WHY (traceable to learned state).
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CricketRecommendation:
    """A single actionable recommendation."""
    category: str
    priority: str
    recommendation: str
    reason: str
    confidence: float
    supporting_data: dict = field(default_factory=dict)
    expires_at: str | None = None


PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


class CricketRecommendationEngine:
    """Generates actionable recommendations from all learners."""

    def __init__(self, format_learner, entity_learner, trend_engine,
                 timing_learner, duration_learner, state_store,
                 baseline_views: int = 290):
        self._format = format_learner
        self._entity = entity_learner
        self._trend = trend_engine
        self._timing = timing_learner
        self._duration = duration_learner
        self._state = state_store
        self._baseline = baseline_views

    def generate_all(self) -> list[CricketRecommendation]:
        """Generate all recommendations from all sources.

        Returns:
            Sorted list of CricketRecommendation (high priority first)
        """
        recs = []
        recs.extend(self._player_recommendations())
        recs.extend(self._team_recommendations())
        recs.extend(self._hook_recommendations())
        recs.extend(self._title_recommendations())
        recs.extend(self._duration_recommendations())
        recs.extend(self._timing_recommendations())
        recs.extend(self._trend_recommendations())
        recs.extend(self._anti_pattern_warnings())
        recs.sort(key=lambda r: PRIORITY_ORDER.get(r.priority, 3))
        return recs

    def _player_recommendations(self) -> list[CricketRecommendation]:
        """Recommend players to cover more or less."""
        recs = []
        top = self._entity.get_top_entities("players", limit=5)

        for p in top:
            if p["n"] < 5 and p["avg_views"] > self._baseline * 1.5:
                multiplier = p["avg_views"] / self._baseline
                recs.append(CricketRecommendation(
                    category="player",
                    priority="high",
                    recommendation=f"Cover {p['name']} more aggressively",
                    reason=f"avg {p['avg_views']:.0f} views ({multiplier:.1f}x baseline) but only {p['n']} videos",
                    confidence=min(0.9, p["score"]),
                    supporting_data={
                        "n": p["n"],
                        "avg_views": p["avg_views"],
                        "baseline": self._baseline,
                        "multiplier": round(multiplier, 2)
                    }
                ))

        all_players = self._state.get_all_entities("players")
        for name, data in all_players.items():
            n = data.get("n", 0)
            avg = data.get("avg_views", 0)
            if n >= 10 and avg < self._baseline * 0.8:
                fatigue = self._entity.fatigue_penalty("players", name)
                if fatigue > 0:
                    recs.append(CricketRecommendation(
                        category="player",
                        priority="high",
                        recommendation=f"Reduce {name} coverage",
                        reason=f"{n} videos at avg {avg:.0f} views. Fatigue penalty active ({fatigue:.2f}).",
                        confidence=0.75,
                        supporting_data={
                            "n": n,
                            "avg_views": avg,
                            "fatigue": round(fatigue, 2),
                            "effective_score": round(data.get("score", 0.5) - fatigue, 2)
                        }
                    ))

        return recs

    def _team_recommendations(self) -> list[CricketRecommendation]:
        """Recommend teams to cover or avoid."""
        recs = []
        all_teams = self._state.get_all_entities("teams")

        for name, data in all_teams.items():
            n = data.get("n", 0)
            avg = data.get("avg_views", 0)
            if n >= 3 and avg < self._baseline * 0.3:
                recs.append(CricketRecommendation(
                    category="team",
                    priority="medium",
                    recommendation=f"Avoid {name.upper()} content",
                    reason=f"avg {avg:.0f} views across {n} videos. Worst performing team.",
                    confidence=0.70,
                    supporting_data={
                        "n": n,
                        "avg_views": avg,
                        "baseline": self._baseline
                    }
                ))

        top = self._entity.get_top_entities("teams", limit=3)
        for t in top:
            if t["avg_views"] > self._baseline * 1.3 and t["n"] >= 5:
                recs.append(CricketRecommendation(
                    category="team",
                    priority="medium",
                    recommendation=f"Increase {t['name'].upper()} coverage",
                    reason=f"avg {t['avg_views']:.0f} views ({t['avg_views']/self._baseline:.1f}x baseline) across {t['n']} videos",
                    confidence=min(0.85, t["score"]),
                    supporting_data={
                        "n": t["n"],
                        "avg_views": t["avg_views"],
                        "score": round(t["score"], 2)
                    }
                ))

        return recs

    def _hook_recommendations(self) -> list[CricketRecommendation]:
        """Recommend hook types to use more."""
        recs = []
        scores = self._format.get_scores()
        hook_types = scores.get("hook_types", {})

        if not hook_types:
            return recs

        sorted_hooks = sorted(hook_types.items(), key=lambda x: x[1].get("avg_views", 0), reverse=True)

        if sorted_hooks:
            best_name, best_data = sorted_hooks[0]
            worst_name, worst_data = sorted_hooks[-1]

            if best_data.get("avg_views", 0) > self._baseline * 1.3:
                recs.append(CricketRecommendation(
                    category="hook",
                    priority="high",
                    recommendation=f"Use '{best_name}' hooks more",
                    reason=f"avg {best_data['avg_views']:.0f} views ({best_data['avg_views']/self._baseline:.1f}x baseline) across {best_data['n']} videos",
                    confidence=min(0.85, best_data.get("score", 0.5)),
                    supporting_data={
                        "hook_type": best_name,
                        "n": best_data["n"],
                        "avg_views": best_data["avg_views"],
                        "score": round(best_data.get("score", 0.5), 2)
                    }
                ))

            if worst_data.get("n", 0) >= 3 and worst_data.get("avg_views", 0) < self._baseline * 0.5:
                recs.append(CricketRecommendation(
                    category="hook",
                    priority="medium",
                    recommendation=f"Avoid '{worst_name}' hooks",
                    reason=f"avg {worst_data['avg_views']:.0f} views ({worst_data['avg_views']/self._baseline:.1f}x baseline) — worst performing hook type",
                    confidence=0.65,
                    supporting_data={
                        "hook_type": worst_name,
                        "n": worst_data["n"],
                        "avg_views": worst_data["avg_views"]
                    }
                ))

        return recs

    def _title_recommendations(self) -> list[CricketRecommendation]:
        """Recommend title formatting patterns."""
        recs = []
        scores = self._format.get_scores()
        patterns = scores.get("title_patterns", {})

        if not patterns:
            return recs

        q_data = patterns.get("has_question", {})
        nq_data = patterns.get("no_question", {})
        if q_data.get("n", 0) >= 5 and nq_data.get("n", 0) >= 5:
            q_avg = q_data.get("avg_views", 0)
            nq_avg = nq_data.get("avg_views", 0)
            if nq_avg > q_avg * 1.2:
                pct_diff = ((nq_avg - q_avg) / q_avg) * 100 if q_avg > 0 else 0
                recs.append(CricketRecommendation(
                    category="title",
                    priority="medium",
                    recommendation="Stop using question marks in titles",
                    reason=f"Questions avg {q_avg:.0f} views vs {nq_avg:.0f} without. -{pct_diff:.0f}% impact.",
                    confidence=0.80,
                    supporting_data={
                        "question_n": q_data["n"],
                        "question_avg": q_avg,
                        "no_question_n": nq_data["n"],
                        "no_question_avg": nq_avg
                    }
                ))

        long_data = patterns.get("long_title", {})
        med_data = patterns.get("medium_title", {})
        if long_data.get("n", 0) >= 5 and med_data.get("n", 0) >= 5:
            long_avg = long_data.get("avg_views", 0)
            med_avg = med_data.get("avg_views", 0)
            if long_avg > med_avg * 1.2:
                pct_lift = ((long_avg - med_avg) / med_avg) * 100 if med_avg > 0 else 0
                recs.append(CricketRecommendation(
                    category="title",
                    priority="medium",
                    recommendation="Use longer titles (60+ characters)",
                    reason=f"Long titles avg {long_avg:.0f} views vs {med_avg:.0f} for medium. +{pct_lift:.0f}% lift.",
                    confidence=0.78,
                    supporting_data={
                        "long_n": long_data["n"],
                        "long_avg": long_avg,
                        "medium_n": med_data["n"],
                        "medium_avg": med_avg
                    }
                ))

        hinglish_data = patterns.get("hinglish", {})
        if hinglish_data.get("n", 0) >= 10 and hinglish_data.get("avg_views", 0) > self._baseline:
            recs.append(CricketRecommendation(
                category="title",
                priority="low",
                recommendation="Prefer Hinglish titles",
                reason=f"Hinglish titles avg {hinglish_data['avg_views']:.0f} views across {hinglish_data['n']} videos",
                confidence=0.65,
                supporting_data={
                    "n": hinglish_data["n"],
                    "avg_views": hinglish_data["avg_views"]
                }
            ))

        return recs

    def _duration_recommendations(self) -> list[CricketRecommendation]:
        """Recommend optimal clip duration."""
        recs = []
        scores = self._duration.get_scores()
        buckets = scores.get("buckets", {})
        sweet_spot = scores.get("sweet_spot", "25-34s")

        if not buckets:
            return recs

        best = buckets.get(sweet_spot, {})
        if best.get("n", 0) >= 5:
            recs.append(CricketRecommendation(
                category="duration",
                priority="low",
                recommendation=f"Target {sweet_spot} duration",
                reason=f"Sweet spot with avg {best.get('avg_views', 0):.0f} views and {best.get('avg_completion', 0):.0%} completion",
                confidence=min(0.75, best.get("score", 0.5)),
                supporting_data={
                    "sweet_spot": sweet_spot,
                    "avg_views": best.get("avg_views", 0),
                    "avg_completion": best.get("avg_completion", 0),
                    "n": best.get("n", 0)
                }
            ))

        return recs

    def _timing_recommendations(self) -> list[CricketRecommendation]:
        """Recommend upload timing."""
        recs = []
        schedule = self._timing.get_schedule_recommendation()

        if schedule.get("total_observations", 0) >= 10:
            recs.append(CricketRecommendation(
                category="timing",
                priority="low",
                recommendation=f"Upload at {schedule['hour']}:00 on {schedule['day']}",
                reason=schedule.get("reason", "Best performing time slot"),
                confidence=schedule.get("confidence", 0.5),
                supporting_data={
                    "hour": schedule["hour"],
                    "day": schedule["day"],
                    "avg_views": schedule.get("avg_views_at_hour", 0),
                    "observations": schedule.get("total_observations", 0)
                }
            ))

        return recs

    def _trend_recommendations(self) -> list[CricketRecommendation]:
        """Recommend trending topics to cover."""
        recs = []
        hot = self._trend.get_hot_topics(limit=3)

        for topic in hot:
            entities = topic.get("entities", {})
            players = entities.get("players", [])
            teams = entities.get("teams", [])
            entity_str = ", ".join(players[:2] + teams[:2])

            recs.append(CricketRecommendation(
                category="topic",
                priority="high",
                recommendation=f"Cover trending: {topic.get('query', 'unknown')}",
                reason=f"Active trend (score {topic['score']:.2f}) featuring {entity_str}",
                confidence=min(0.90, topic["score"]),
                supporting_data={
                    "trend_id": topic.get("trend_id", ""),
                    "score": topic["score"],
                    "entities": entities,
                    "category": topic.get("category", "")
                },
                expires_at=topic.get("expires_at")
            ))

        return recs

    def _anti_pattern_warnings(self) -> list[CricketRecommendation]:
        """Generate warnings about anti-patterns."""
        recs = []

        all_players = self._state.get_all_entities("players")
        for name, data in all_players.items():
            n = data.get("n", 0)
            if n >= 15:
                recs.append(CricketRecommendation(
                    category="warning",
                    priority="medium",
                    recommendation=f"Over-coverage alert: {name}",
                    reason=f"{n} videos — consider diversifying to avoid audience fatigue",
                    confidence=0.60,
                    supporting_data={
                        "entity": name,
                        "n": n,
                        "avg_views": data.get("avg_views", 0)
                    }
                ))

        return recs
