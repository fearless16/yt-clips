"""
seo_learner.py — Self-improving SEO system that learns from past performance.
Analyzes which titles, hooks, and CTAs drive the most engagement and updates prompts.
"""
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("seo_learner", cfg["logging"]["log_file"], cfg["logging"]["level"])


class SEOLearner:
    """
    Learns from past YouTube Shorts performance to improve future SEO.
    Tracks what titles, hooks, and CTAs perform best and adapts prompts.
    """

    def __init__(self):
        self.performance_db = Path("data/seo_performance.json")
        self.performance_db.parent.mkdir(exist_ok=True)
        self.learned_insights = self._load_performance_data()

    def _load_performance_data(self) -> Dict:
        """Load historical performance data."""
        if self.performance_db.exists():
            try:
                with open(self.performance_db, "r") as f:
                    return json.load(f)
            except Exception as e:
                log.warning(f"Failed to load SEO performance data: {e}")
        return {
            "clips": [],  # List of clip performance records
            "title_patterns": {},  # What title structures work best
            "hooks_performance": {},  # Which hooks drive engagement
            "ctas_performance": {},  # Which CTAs drive action
            "hashtag_performance": {},  # Which hashtags boost discovery
            "last_updated": None
        }

    def _save_performance_data(self):
        """Save performance data to disk."""
        self.learned_insights["last_updated"] = datetime.now().isoformat()
        with open(self.performance_db, "w") as f:
            json.dump(self.learned_insights, f, indent=2)

    def record_performance(self, clip_id: str, title: str, description: str, 
                         hashtags: List[str], analytics: Dict,
                         tags: List[str] = None, search_terms: List[str] = None):
        """
        Record performance data for a clip.
        
        Args:
            clip_id: YouTube video ID
            title: Video title
            description: Video description
            hashtags: List of hashtags
            analytics: Dict with views, likes, comments, avg_view_duration, ctr
            tags: SEO tags for YouTube API
            search_terms: Search phrases
        """
        features = self._extract_seo_features(title, description, hashtags)
        features["tags_count"] = len(tags or [])
        features["search_terms_count"] = len(search_terms or [])
        
        # Calculate performance score (weighted)
        performance_score = self._calculate_performance_score(analytics)
        
        # Store record
        record = {
            "clip_id": clip_id,
            "timestamp": datetime.now().isoformat(),
            "performance_score": performance_score,
            "analytics": analytics,
            "features": features
        }
        
        self.learned_insights["clips"].append(record)
        
        # Keep only last 100 clips to prevent unbounded growth
        if len(self.learned_insights["clips"]) > 100:
            self.learned_insights["clips"] = self.learned_insights["clips"][-100:]
        
        # Update learned patterns
        self._update_learned_patterns(features, performance_score)
        
        self._save_performance_data()
        log.info(f"📊 Recorded SEO performance for {clip_id}: score={performance_score:.2f}")

    def _extract_seo_features(self, title: str, description: str, hashtags: List[str]) -> Dict:
        """Extract SEO features from title, description, and hashtags."""
        has_pipe_format = "|" in title and "vs" in title.lower() and any(
            t in title for t in ["IPL", "TATA", "Cricket", "T20"]
        )
        power_words = [
            "destroys", "smashes", "obliterates", "anchors", "steals",
            "snatches", "demolishes", "shatters", "ignites", "seals",
            "sinks", "crushes", "finishes", "annihilates", "dominates",
        ]
        sections = ["Match Summary", "What Happens", "Key Players", "Match Situation"]
        features = {
            "title_length": len(title),
            "has_pipe_format": has_pipe_format,
            "has_power_word": any(w in title.lower() for w in power_words),
            "has_emoji": bool(re.search(r'[\U0001F600-\U0001F64F]', title)),
            "has_player_name": bool(re.search(r'[A-Z][a-z]+ [A-Z][a-z]+', title)),
            "has_score": bool(re.search(r'\d+\s*(?:off|of|/\d+| runs| wickets| for\s*\d)', title.lower())),
            "has_number": bool(re.search(r'\d', title)),
            "title_words": len(title.split()),
            # Description features
            "description_length": len(description),
            "has_sections": all(s.lower() in description.lower() for s in sections),
            "sections_count": sum(1 for s in sections if s.lower() in description.lower()),
            "has_cta": any(word in description.lower() for word in [
                "like", "subscribe", "share", "comment", "follow", "bell"
            ]),
            # Hashtag features
            "hashtag_count": len(hashtags),
            "has_shorts_hashtag": any(h.lower() == "#shorts" for h in hashtags),
            "has_ipl_hashtag": any("#ipl" in h.lower() for h in hashtags),
            "has_team_hashtag": any(any(team in h.lower() for team in [
                "rcb", "csk", "mi", "kkr", "srh", "dc", "pbks", "rr", "gt", "lsg"
            ]) for h in hashtags),
        }
        return features

    def _calculate_performance_score(self, analytics: Dict) -> float:
        """
        Calculate composite performance score.
        Weights: views (40%), engagement rate (30%), retention (20%), CTR (10%)
        """
        views = analytics.get("viewCount", 0)
        likes = analytics.get("likeCount", 0)
        comments = analytics.get("commentCount", 0)
        # Note: For actual implementation, you'd need to fetch avg_view_duration and CTR
        # from YouTube Analytics API - this is simplified
        
        # Normalize views (log scale to prevent outliers from dominating)
        import math
        views_score = min(1.0, math.log(max(1, views)) / 15)  # Cap at ~1M views
        
        # Engagement rate (likes + comments) / views
        engagement_rate = (likes + comments) / max(1, views)
        engagement_score = min(1.0, engagement_rate * 20)  # Scale to 0-1
        
        # For now, use placeholder for retention and CTR (would come from analytics)
        retention_score = 0.5  # Placeholder
        ctr_score = 0.5  # Placeholder
        
        # Weighted composite
        score = (
            views_score * 0.4 +
            engagement_score * 0.3 +
            retention_score * 0.2 +
            ctr_score * 0.1
        )
        
        return min(1.0, max(0.0, score))  # Clamp to 0-1

    def _update_learned_patterns(self, features: Dict, performance_score: float):
        """Update learned patterns based on what performs well."""
        if performance_score < 0.3 or performance_score > 0.7:
            title_pattern = (
                f"pipe_format:{features['has_pipe_format']}_"
                f"power_word:{features['has_power_word']}_"
                f"player:{features['has_player_name']}_"
                f"score:{features['has_score']}_"
                f"emoji:{features['has_emoji']}_"
                f"sections:{features['sections_count']}"
            )

            if title_pattern not in self.learned_insights["title_patterns"]:
                self.learned_insights["title_patterns"][title_pattern] = {
                    "count": 0,
                    "total_score": 0.0,
                    "avg_score": 0.0
                }

            pattern_data = self.learned_insights["title_patterns"][title_pattern]
            pattern_data["count"] += 1
            pattern_data["total_score"] += performance_score
            pattern_data["avg_score"] = pattern_data["total_score"] / pattern_data["count"]

            if features["has_pipe_format"]:
                pt = "pipe_format"
                self.learned_insights["hooks_performance"].setdefault(pt, []).append(performance_score)

            if features["has_sections"]:
                ct = "full_sections"
                self.learned_insights["ctas_performance"].setdefault(ct, []).append(performance_score)

    def get_learned_title_preferences(self) -> Dict:
        """Get insights about what title structures perform best."""
        preferences = {
            "best_patterns": [],
            "avoid_patterns": [],
            "hook_effectiveness": {},
            "recommendations": []
        }
        
        # Analyze title patterns
        for pattern, data in self.learned_insights["title_patterns"].items():
            if data["count"] >= 3:  # Need minimum samples
                if data["avg_score"] > 0.6:
                    preferences["best_patterns"].append((pattern, data["avg_score"]))
                elif data["avg_score"] < 0.4:
                    preferences["avoid_patterns"].append((pattern, data["avg_score"]))
        
        # Sort by score
        preferences["best_patterns"].sort(key=lambda x: x[1], reverse=True)
        preferences["avoid_patterns"].sort(key=lambda x: x[1])
        
        # Generate recommendations
        if preferences["best_patterns"]:
            best_pattern = preferences["best_patterns"][0][0]
            if "pipe_format:true" in best_pattern:
                preferences["recommendations"].append("Pipe format (Team vs Team | Tournament) drives CTR — keep using it")
            if "power_word:true" in best_pattern:
                preferences["recommendations"].append("Action power words (Destroys, Smashes) boost engagement")
            if "player:true" in best_pattern:
                preferences["recommendations"].append("Player names in title increase search discoverability")
            if "score:true" in best_pattern:
                preferences["recommendations"].append("Including scores in title (e.g. 67 off 34) improves CTR")
            if "emoji:true" in best_pattern:
                preferences["recommendations"].append("1 emoji in title improves thumbnail CTR")
            if "sections:4" in best_pattern or "sections:5" in best_pattern:
                preferences["recommendations"].append("Full description sections improve watch time")
        if preferences["avoid_patterns"]:
            worst = preferences["avoid_patterns"][0][0]
            if "pipe_format:false" in worst:
                preferences["recommendations"].append("Titles WITHOUT pipe format underperform — always add | Team vs Team | Tournament")
            if "power_word:false" in worst:
                preferences["recommendations"].append("Titles without power words underperform — use action verbs")
            if "player:false" in worst:
                preferences["recommendations"].append("Titles without player names underperform — always mention the star")
        
        return preferences

    def get_seo_improvement_suggestions(self) -> List[str]:
        """Get actionable suggestions to improve SEO based on learned data."""
        suggestions = []
        
        # Get title preferences
        title_prefs = self.get_learned_title_preferences()
        suggestions.extend(title_prefs["recommendations"])
        
        # Analyze hook performance
        for hook_type, scores in self.learned_insights["hooks_performance"].items():
            if len(scores) >= 3:
                avg_score = sum(scores) / len(scores)
                if avg_score > 0.6:
                    suggestions.append(f"Use more {hook_type} hooks - they perform well (avg score: {avg_score:.2f})")
                elif avg_score < 0.4:
                    suggestions.append(f"Avoid {hook_type} hooks - they underperform (avg score: {avg_score:.2f})")
        
        # Analyze CTA performance
        for cta_type, scores in self.learned_insights["ctas_performance"].items():
            if len(scores) >= 3:
                avg_score = sum(scores) / len(scores)
                if avg_score > 0.6:
                    suggestions.append(f"Use more {cta_type} style CTAs - they drive engagement")
                elif avg_score < 0.4:
                    suggestions.append(f"Improve {cta_type} CTAs - they're not working well")
        
        if not suggestions:
            suggestions.append("Keep experimenting! Need more data to give specific recommendations.")
        
        return suggestions

    def update_prompt_with_learnings(self, base_prompt: str) -> str:
        """
        Update the SEO prompt with learned insights.
        Returns an enhanced prompt that incorporates what we've learned works.
        """
        # Get learned preferences
        title_prefs = self.get_learned_title_preferences()
        suggestions = self.get_seo_improvement_suggestions()
        
        # Build enhancement section
        enhancement = "\n\n# LEARNED FROM PERFORMANCE DATA\n"
        
        if title_prefs["best_patterns"]:
            enhancement += "## ✅ What WORKS (based on past performance):\n"
            for pattern, score in title_prefs["best_patterns"][:3]:
                enhancement += f"- {pattern} (avg score: {score:.2f})\n"
        
        if title_prefs["avoid_patterns"]:
            enhancement += "\n## ❌ What to AVOID (based on past performance):\n"
            for pattern, score in title_prefs["avoid_patterns"][:3]:
                enhancement += f"- {pattern} (avg score: {score:.2f})\n"

        return base_prompt + enhancement


# Global instance for easy access
seo_learner = SEOLearner()

learn_from_clip_performance = seo_learner.record_performance
get_seo_improvement_suggestions = seo_learner.get_seo_improvement_suggestions


def enhance_seo_prompt(base_prompt: str) -> str:
    """Enhance SEO prompt with learned insights."""
    return seo_learner.update_prompt_with_learnings(base_prompt)


def generate_performance_report() -> str:
    """Generate a report of what we've learned from performance data."""
    learner = seo_learner
    if not learner.learned_insights["clips"]:
        return "No performance data collected yet."

    clips = learner.learned_insights["clips"]
    total_clips = len(clips)
    avg_score = sum(c["performance_score"] for c in clips) / total_clips

    parts = [
        f"Total Clips Analyzed: {total_clips}",
        f"Average Performance Score: {avg_score:.2f}",
    ]

    title_prefs = learner.get_learned_title_preferences()
    if title_prefs["best_patterns"]:
        parts.append("\nWinning Title Patterns:")
        for pattern, score in title_prefs["best_patterns"][:3]:
            parts.append(f"  + {pattern} (avg: {score:.2f})")

    if title_prefs["avoid_patterns"]:
        parts.append("\nLosing Title Patterns:")
        for pattern, score in title_prefs["avoid_patterns"][:3]:
            parts.append(f"  - {pattern} (avg: {score:.2f})")

    suggestions = learner.get_seo_improvement_suggestions()
    if suggestions:
        parts.append("\nSuggestions:")
        for s in suggestions:
            parts.append(f"  * {s}")

    if clips:
        avg_tags = sum(c["features"].get("tags_count", 0) for c in clips) / len(clips)
        avg_terms = sum(c["features"].get("search_terms_count", 0) for c in clips) / len(clips)
        parts.append(f"\nSEO Volume: tags={avg_tags:.0f}/clip, terms={avg_terms:.0f}/clip")

    return "\n".join(parts)


if __name__ == "__main__":
    print(generate_performance_report())