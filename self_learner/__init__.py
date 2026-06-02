"""self_learner — persistent-memory learning engine.

Zero FaceOS deps. Built for autonomous learning with SQLite-backed memory.

Submodules:
    memory      SQLite-backed persistent key-value store with metadata
    knowledge   Knowledge base with facts, patterns, and relationships
    learner     Learning engine: observe -> extract -> predict
    seo_learner Learn from SEO performance (keywords, titles, providers)
    trends      Time-series trend analysis and anomaly detection
    recommend   Predictive recommendations engine
    runner      CLI entry point for one-shot and daemon modes

Usage:
    from self_learner import Learner, SEOLearner, TrendAnalyzer, RecommendationEngine
    learner = Learner()
    learner.observe("pipeline_run", {"duration": 120, "success": True})
    pred = learner.predict("pipeline_run")

    seo = SEOLearner()
    seo.record_seo_outcome(seo_performance)
    best_keywords = seo.get_best_keywords("shorts")

    trends = TrendAnalyzer()
    trends.record_metric(TrendPoint("duration", 120))
    trend = trends.analyze_trend("duration")

    recs = RecommendationEngine()
    recommendations = recs.generate_recommendations()
"""

from self_learner.memory import PersistentMemory
from self_learner.knowledge import KnowledgeBase, Fact
from self_learner.learner import Learner, Observation, Pattern, Prediction
from self_learner.seo_learner import SEOLearner, SEOPerformance
from self_learner.trends import TrendAnalyzer, TrendPoint, TrendResult
from self_learner.recommend import RecommendationEngine, Recommendation

VERSION = "2.0.0"

__all__ = [
    "PersistentMemory",
    "KnowledgeBase",
    "Fact",
    "Learner",
    "Observation",
    "Pattern",
    "Prediction",
    "SEOLearner",
    "SEOPerformance",
    "TrendAnalyzer",
    "TrendPoint",
    "TrendResult",
    "RecommendationEngine",
    "Recommendation",
    "VERSION",
]
