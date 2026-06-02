"""recommend.py — Predictive recommendations engine.

Combines SEO learning, trend analysis, and pattern extraction to
provide actionable recommendations for pipeline optimization.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from self_learner.memory import PersistentMemory
from self_learner.knowledge import Fact, KnowledgeBase
from self_learner.seo_learner import SEOLearner
from self_learner.trends import TrendAnalyzer


@dataclass
class Recommendation:
    """A single actionable recommendation."""
    category: str  # "seo", "provider", "content_type", "performance"
    priority: str  # "high", "medium", "low"
    title: str
    description: str
    action: str  # Specific action to take
    confidence: float  # 0-1
    supporting_data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class RecommendationEngine:
    """Generate actionable recommendations based on learned patterns.

    Args:
        memory: A ``PersistentMemory`` instance. Creates a default one if
                not provided.
        knowledge: A ``KnowledgeBase`` instance. Creates a default one if
                   not provided.
        seo_learner: A ``SEOLearner`` instance. Creates a default one if
                     not provided.
        trend_analyzer: A ``TrendAnalyzer`` instance. Creates a default one
                        if not provided.
    """

    def __init__(self, memory: Optional[PersistentMemory] = None,
                 knowledge: Optional[KnowledgeBase] = None,
                 seo_learner: Optional[SEOLearner] = None,
                 trend_analyzer: Optional[TrendAnalyzer] = None):
        self._memory = memory or PersistentMemory()
        self._knowledge = knowledge or KnowledgeBase(memory=self._memory)
        self._seo_learner = seo_learner or SEOLearner(
            memory=self._memory, knowledge=self._knowledge
        )
        self._trend_analyzer = trend_analyzer or TrendAnalyzer(
            memory=self._memory, knowledge=self._knowledge
        )

    def generate_recommendations(self) -> list[Recommendation]:
        """Generate all recommendations based on current state.

        Returns:
            List of recommendations sorted by priority.
        """
        recommendations = []

        # Get SEO recommendations
        recommendations.extend(self._seo_recommendations())

        # Get trend-based recommendations
        recommendations.extend(self._trend_recommendations())

        # Get provider recommendations
        recommendations.extend(self._provider_recommendations())

        # Get content type recommendations
        recommendations.extend(self._content_type_recommendations())

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda r: priority_order.get(r.priority, 3))

        return recommendations

    def _seo_recommendations(self) -> list[Recommendation]:
        """Generate SEO-related recommendations."""
        recs = []

        # Check best keywords for shorts
        best_keywords = self._seo_learner.get_best_keywords("shorts", limit=5)
        if best_keywords:
            keywords_str = ", ".join(kw["keyword"] for kw in best_keywords[:3])
            recs.append(Recommendation(
                category="seo",
                priority="high",
                title="Use top-performing keywords for Shorts",
                description=f"These keywords have the highest engagement: {keywords_str}",
                action=f"Include these keywords in your next Shorts SEO: {keywords_str}",
                confidence=0.8,
                supporting_data={"keywords": best_keywords},
            ))

        # Check title patterns
        best_patterns = self._seo_learner.get_best_title_patterns("shorts", limit=3)
        if best_patterns:
            patterns_str = ", ".join(p["pattern"] for p in best_patterns)
            recs.append(Recommendation(
                category="seo",
                priority="medium",
                title="Use successful title patterns",
                description=f"Titles with these patterns perform better: {patterns_str}",
                action=f"Apply these patterns to your titles: {patterns_str}",
                confidence=0.7,
                supporting_data={"patterns": best_patterns},
            ))

        return recs

    def _trend_recommendations(self) -> list[Recommendation]:
        """Generate trend-based recommendations."""
        recs = []

        # Check pipeline duration trend
        duration_trend = self._trend_analyzer.analyze_trend("duration")
        if duration_trend and duration_trend.direction == "degrading":
            recs.append(Recommendation(
                category="performance",
                priority="high",
                title="Pipeline duration is increasing",
                description=f"Duration is increasing at {duration_trend.slope:.1f} min/hour",
                action="Check for resource bottlenecks or optimize heavy stages",
                confidence=duration_trend.confidence,
                supporting_data={
                    "slope": duration_trend.slope,
                    "current": duration_trend.current_value,
                },
            ))

        # Check export success trend
        export_trend = self._trend_analyzer.analyze_trend("exported_count")
        if export_trend and export_trend.direction == "degrading":
            recs.append(Recommendation(
                category="performance",
                priority="high",
                title="Export count is decreasing",
                description=f"Fewer clips are being exported over time",
                action="Check highlight detection sensitivity or content quality",
                confidence=export_trend.confidence,
                supporting_data={
                    "slope": export_trend.slope,
                    "current": export_trend.current_value,
                },
            ))

        # Check for anomalies
        for metric in ["duration", "failures", "exported_count"]:
            anomalies = self._trend_analyzer.detect_anomalies(metric)
            if anomalies:
                latest = anomalies[-1]
                recs.append(Recommendation(
                    category="performance",
                    priority="medium",
                    title=f"Anomaly detected in {metric}",
                    description=f"{metric} had unusual value: {latest['value']:.2f} "
                               f"({latest['deviation']} by {latest['z_score']:.1f} stdev)",
                    action=f"Investigate what caused the unusual {metric} value",
                    confidence=0.6,
                    supporting_data={"anomaly": latest},
                ))

        return recs

    def _provider_recommendations(self) -> list[Recommendation]:
        """Generate provider-related recommendations."""
        recs = []

        provider_perf = self._seo_learner.get_provider_performance()
        if not provider_perf:
            return recs

        # Find best and worst providers
        best_provider = None
        best_rate = 0.0
        worst_provider = None
        worst_rate = 1.0

        for provider, stats in provider_perf.items():
            if stats["total_uses"] < 2:
                continue
            if stats["success_rate"] > best_rate:
                best_rate = stats["success_rate"]
                best_provider = provider
            if stats["success_rate"] < worst_rate:
                worst_rate = stats["success_rate"]
                worst_provider = provider

        if best_provider and best_rate > 0.8:
            recs.append(Recommendation(
                category="provider",
                priority="medium",
                title=f"Provider '{best_provider}' is performing well",
                description=f"{best_provider} has {best_rate:.0%} success rate",
                action=f"Consider using {best_provider} as primary provider",
                confidence=0.7,
                supporting_data={"provider": best_provider, "success_rate": best_rate},
            ))

        if worst_provider and worst_rate < 0.5:
            recs.append(Recommendation(
                category="provider",
                priority="high",
                title=f"Provider '{worst_provider}' is underperforming",
                description=f"{worst_provider} has only {worst_rate:.0%} success rate",
                action=f"Consider reducing reliance on {worst_provider} or investigating issues",
                confidence=0.7,
                supporting_data={"provider": worst_provider, "success_rate": worst_rate},
            ))

        return recs

    def _content_type_recommendations(self) -> list[Recommendation]:
        """Generate content type recommendations."""
        recs = []

        content_stats = self._seo_learner.get_content_type_stats()
        if not content_stats:
            return recs

        shorts_stats = content_stats.get("shorts", {})
        long_form_stats = content_stats.get("long_form", {})

        # Compare performance
        if shorts_stats.get("count", 0) >= 3 and long_form_stats.get("count", 0) >= 3:
            shorts_engagement = shorts_stats.get("avg_engagement", 0)
            long_form_engagement = long_form_stats.get("avg_engagement", 0)

            if shorts_engagement > long_form_engagement * 1.5:
                recs.append(Recommendation(
                    category="content_type",
                    priority="medium",
                    title="Shorts are outperforming long-form",
                    description=f"Shorts get {shorts_engagement:.1f}x more engagement than long-form",
                    action="Consider focusing more on Shorts content",
                    confidence=0.6,
                    supporting_data={
                        "shorts_engagement": shorts_engagement,
                        "long_form_engagement": long_form_engagement,
                    },
                ))
            elif long_form_engagement > shorts_engagement * 1.5:
                recs.append(Recommendation(
                    category="content_type",
                    priority="medium",
                    title="Long-form is outperforming Shorts",
                    description=f"Long-form gets {long_form_engagement:.1f}x more engagement than Shorts",
                    action="Consider focusing more on long-form content",
                    confidence=0.6,
                    supporting_data={
                        "shorts_engagement": shorts_engagement,
                        "long_form_engagement": long_form_engagement,
                    },
                ))

        return recs

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the recommendation engine state.

        Returns:
            Dict with counts and top recommendations.
        """
        recommendations = self.generate_recommendations()

        by_category: dict[str, int] = {}
        by_priority: dict[str, int] = {}
        for rec in recommendations:
            by_category[rec.category] = by_category.get(rec.category, 0) + 1
            by_priority[rec.priority] = by_priority.get(rec.priority, 0) + 1

        return {
            "total_recommendations": len(recommendations),
            "by_category": by_category,
            "by_priority": by_priority,
            "top_recommendations": [
                {
                    "category": r.category,
                    "priority": r.priority,
                    "title": r.title,
                    "action": r.action,
                }
                for r in recommendations[:5]
            ],
        }

    def close(self):
        """Close underlying resources."""
        self._memory.close()
