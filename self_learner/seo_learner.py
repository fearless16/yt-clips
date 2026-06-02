"""seo_learner.py — Learn from SEO performance.

Tracks which keywords, titles, descriptions, and hashtags lead to better
engagement. Correlates SEO choices with upload success and view counts.
"""

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from self_learner.memory import PersistentMemory
from self_learner.knowledge import Fact, KnowledgeBase


@dataclass
class SEOPerformance:
    """Performance metrics for a specific SEO strategy."""
    clip_id: str
    title: str
    description: str
    hashtags: list[str]
    tags: list[str]
    is_shorts: bool
    provider: str
    model: str
    upload_success: bool
    view_count: int = 0
    like_count: int = 0
    engagement_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)


class SEOLearner:
    """Learn from SEO performance to improve future SEO generation.

    Args:
        memory: A ``PersistentMemory`` instance. Creates a default one if
                not provided.
        knowledge: A ``KnowledgeBase`` instance. Creates a default one if
                   not provided.
    """

    def __init__(self, memory: Optional[PersistentMemory] = None,
                 knowledge: Optional[KnowledgeBase] = None):
        self._memory = memory or PersistentMemory()
        self._knowledge = knowledge or KnowledgeBase(memory=self._memory)

    def record_seo_outcome(self, perf: SEOPerformance) -> None:
        """Record the outcome of an SEO strategy.

        Args:
            perf: The SEO performance metrics.
        """
        fact = Fact(
            subject="seo_outcome",
            relation="has_performance",
            object={
                "clip_id": perf.clip_id,
                "title": perf.title,
                "description": perf.description[:200],  # Truncate for storage
                "hashtags": perf.hashtags,
                "tags": perf.tags,
                "is_shorts": perf.is_shorts,
                "provider": perf.provider,
                "model": perf.model,
                "upload_success": perf.upload_success,
                "view_count": perf.view_count,
                "like_count": perf.like_count,
                "engagement_rate": perf.engagement_rate,
                "timestamp": perf.timestamp,
            },
            confidence=1.0,
            source="seo_learner",
        )
        self._knowledge.add_fact(fact)

    def get_best_keywords(self, content_type: str = "all",
                          limit: int = 10) -> list[dict[str, Any]]:
        """Get the best performing keywords.

        Args:
            content_type: Filter by content type ("shorts", "long_form", "all").
            limit: Maximum number of keywords to return.

        Returns:
            List of dicts with keyword, avg_engagement, count.
        """
        facts = self._knowledge.get_facts(
            subject="seo_outcome", relation="has_performance"
        )

        keyword_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"engagements": [], "count": 0}
        )

        for fact in facts:
            obj = fact.object
            if not isinstance(obj, dict):
                continue

            if content_type != "all":
                is_shorts = obj.get("is_shorts", True)
                if content_type == "shorts" and not is_shorts:
                    continue
                if content_type == "long_form" and is_shorts:
                    continue

            tags = obj.get("tags", [])
            hashtags = obj.get("hashtags", [])
            engagement = obj.get("engagement_rate", 0.0)

            for keyword in tags + hashtags:
                keyword_stats[keyword]["engagements"].append(engagement)
                keyword_stats[keyword]["count"] += 1

        results = []
        for keyword, stats in keyword_stats.items():
            if stats["count"] >= 2:  # Need at least 2 data points
                avg_engagement = sum(stats["engagements"]) / len(stats["engagements"])
                results.append({
                    "keyword": keyword,
                    "avg_engagement": avg_engagement,
                    "count": stats["count"],
                })

        results.sort(key=lambda x: x["avg_engagement"], reverse=True)
        return results[:limit]

    def get_best_title_patterns(self, content_type: str = "all",
                                 limit: int = 5) -> list[dict[str, Any]]:
        """Get patterns in successful titles.

        Args:
            content_type: Filter by content type ("shorts", "long_form", "all").
            limit: Maximum number of patterns to return.

        Returns:
            List of dicts with pattern, avg_engagement, count.
        """
        facts = self._knowledge.get_facts(
            subject="seo_outcome", relation="has_performance"
        )

        patterns: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"engagements": [], "count": 0}
        )

        for fact in facts:
            obj = fact.object
            if not isinstance(obj, dict):
                continue

            if content_type != "all":
                is_shorts = obj.get("is_shorts", True)
                if content_type == "shorts" and not is_shorts:
                    continue
                if content_type == "long_form" and is_shorts:
                    continue

            title = obj.get("title", "")
            engagement = obj.get("engagement_rate", 0.0)

            # Extract patterns from title
            title_patterns = self._extract_title_patterns(title)
            for pattern in title_patterns:
                patterns[pattern]["engagements"].append(engagement)
                patterns[pattern]["count"] += 1

        results = []
        for pattern, stats in patterns.items():
            if stats["count"] >= 2:
                avg_engagement = sum(stats["engagements"]) / len(stats["engagements"])
                results.append({
                    "pattern": pattern,
                    "avg_engagement": avg_engagement,
                    "count": stats["count"],
                })

        results.sort(key=lambda x: x["avg_engagement"], reverse=True)
        return results[:limit]

    def _extract_title_patterns(self, title: str) -> list[str]:
        """Extract reusable patterns from a title."""
        patterns = []

        # Check for emojis
        if re.search(r'[\U0001F600-\U0001F64F]', title):
            patterns.append("has_emoji")

        # Check for ALL CAPS words
        if re.search(r'\b[A-Z]{2,}\b', title):
            patterns.append("has_caps")

        # Check for numbers
        if re.search(r'\d+', title):
            patterns.append("has_numbers")

        # Check for question marks
        if '?' in title:
            patterns.append("is_question")

        # Check for exclamation marks
        if '!' in title:
            patterns.append("has_exclamation")

        # Check for hashtags
        if '#' in title:
            patterns.append("has_hashtag")

        # Check for pipe separator
        if '|' in title:
            patterns.append("has_pipe")

        # Check for hyphen separator
        if ' - ' in title:
            patterns.append("has_hyphen")

        # Check length category
        if len(title) < 30:
            patterns.append("short_title")
        elif len(title) < 60:
            patterns.append("medium_title")
        else:
            patterns.append("long_title")

        # Check for Hinglish patterns
        hinglish_words = ['ka', 'ki', 'ke', 'ne', 'se', 'me', 'hai', 'tha', 'thi',
                          'aur', 'ya', 'ko', 'par', 'bhi', 'nahi', 'kya', 'ye', 'wo']
        title_lower = title.lower()
        for word in hinglish_words:
            if f' {word} ' in f' {title_lower} ':
                patterns.append("is_hinglish")
                break

        return patterns

    def get_provider_performance(self) -> dict[str, dict[str, Any]]:
        """Get performance metrics by provider.

        Returns:
            Dict mapping provider to metrics.
        """
        facts = self._knowledge.get_facts(
            subject="seo_outcome", relation="has_performance"
        )

        provider_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"successes": 0, "failures": 0, "engagements": []}
        )

        for fact in facts:
            obj = fact.object
            if not isinstance(obj, dict):
                continue

            provider = obj.get("provider", "unknown")
            upload_success = obj.get("upload_success", False)
            engagement = obj.get("engagement_rate", 0.0)

            if upload_success:
                provider_stats[provider]["successes"] += 1
                provider_stats[provider]["engagements"].append(engagement)
            else:
                provider_stats[provider]["failures"] += 1

        results = {}
        for provider, stats in provider_stats.items():
            total = stats["successes"] + stats["failures"]
            avg_engagement = (
                sum(stats["engagements"]) / len(stats["engagements"])
                if stats["engagements"] else 0.0
            )
            results[provider] = {
                "success_rate": stats["successes"] / total if total > 0 else 0.0,
                "avg_engagement": avg_engagement,
                "total_uses": total,
            }

        return results

    def get_content_type_stats(self) -> dict[str, dict[str, Any]]:
        """Get performance stats by content type.

        Returns:
            Dict mapping content type to metrics.
        """
        facts = self._knowledge.get_facts(
            subject="seo_outcome", relation="has_performance"
        )

        type_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "successes": 0, "engagements": []}
        )

        for fact in facts:
            obj = fact.object
            if not isinstance(obj, dict):
                continue

            is_shorts = obj.get("is_shorts", True)
            content_type = "shorts" if is_shorts else "long_form"
            upload_success = obj.get("upload_success", False)
            engagement = obj.get("engagement_rate", 0.0)

            type_stats[content_type]["count"] += 1
            if upload_success:
                type_stats[content_type]["successes"] += 1
                type_stats[content_type]["engagements"].append(engagement)

        results = {}
        for content_type, stats in type_stats.items():
            avg_engagement = (
                sum(stats["engagements"]) / len(stats["engagements"])
                if stats["engagements"] else 0.0
            )
            results[content_type] = {
                "count": stats["count"],
                "success_rate": stats["successes"] / stats["count"] if stats["count"] > 0 else 0.0,
                "avg_engagement": avg_engagement,
            }

        return results

    def recommend_seo_strategy(self, content_type: str = "shorts",
                                topic: str = "") -> dict[str, Any]:
        """Recommend SEO strategy based on learned patterns.

        Args:
            content_type: Type of content ("shorts" or "long_form").
            topic: Optional topic to tailor recommendations.

        Returns:
            Dict with recommended keywords, title patterns, etc.
        """
        best_keywords = self.get_best_keywords(content_type, limit=10)
        best_patterns = self.get_best_title_patterns(content_type, limit=5)
        provider_perf = self.get_provider_performance()

        # Find best provider
        best_provider = None
        best_success_rate = 0.0
        for provider, stats in provider_perf.items():
            if stats["success_rate"] > best_success_rate:
                best_success_rate = stats["success_rate"]
                best_provider = provider

        return {
            "recommended_keywords": [kw["keyword"] for kw in best_keywords],
            "title_patterns": [p["pattern"] for p in best_patterns],
            "best_provider": best_provider,
            "best_provider_success_rate": best_success_rate,
            "content_type_stats": self.get_content_type_stats().get(content_type, {}),
        }

    def close(self):
        """Close underlying resources."""
        self._memory.close()
