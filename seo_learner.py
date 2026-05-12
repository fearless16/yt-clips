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
                         hashtags: List[str], analytics: Dict):
        """
        Record performance data for a clip.
        
        Args:
            clip_id: YouTube video ID
            title: Video title
            description: Video description
            hashtags: List of hashtags
            analytics: Dict with views, likes, comments, avg_view_duration, ctr
        """
        # Extract features
        features = self._extract_seo_features(title, description, hashtags)
        
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
        features = {
            "title_length": len(title),
            "starts_with_cricket_live": title.lower().startswith("cricket live:"),
            "starts_with_ipl": title.lower().startswith("ipl:"),
            "has_emoji": bool(re.search(r'[\U0001F600-\U0001F64F]', title)),
            "has_number": bool(re.search(r'\d', title)),
            "title_words": len(title.split()),
            # Hook detection
            "has_shock_hook": any(hook in title.lower() for hook in [
                "kya", "yeh kya", "unbelievable", "shocking", " insane ", 
                "wait", "no way", "did they", "can't believe"
            ]),
            # Description features
            "description_length": len(description),
            "has_cta": any(word in description.lower() for word in [
                "like", "subscribe", "share", "comment", "follow", "bell"
            ]),
            # Hashtag features
            "hashtag_count": len(hashtags),
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
        # Only learn from significantly good or bad performance
        if performance_score < 0.3 or performance_score > 0.7:
            # Title patterns
            title_pattern = (
                f"starts_cricket_live:{features['starts_with_cricket_live']}_"
                f"has_emoji:{features['has_emoji']}_"
                f"has_number:{features['has_number']}_"
                f"hook:{features['has_shock_hook']}"
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
            
            # Hook performance
            if features["has_shock_hook"]:
                hook_type = "shock_hook"
                if hook_type not in self.learned_insights["hooks_performance"]:
                    self.learned_insights["hooks_performance"][hook_type] = []
                self.learned_insights["hooks_performance"][hook_type].append(performance_score)
            
            # CTA performance
            if features["has_cta"]:
                cta_type = "has_cta"
                if cta_type not in self.learned_insights["ctas_performance"]:
                    self.learned_insights["ctas_performance"][cta_type] = []
                self.learned_insights["ctas_performance"][cta_type].append(performance_score)

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
            if "starts_cricket_live:true" in best_pattern:
                preferences["recommendations"].append("Always start titles with 'cricket live:'")
            if "has_emoji:true" in best_pattern:
                preferences["recommendations"].append("Include 1 emoji in title for higher CTR")
            if "has_number:true" in best_pattern:
                preferences["recommendations"].append("Include numbers (scores, wickets) in titles")
        
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
        
        if suggestions:
            enhancement += "\n## 💡 ACTIONABLE RECOMMENDATIONS:\n"
            for suggestion in suggestions[:5]:  # Top 5 suggestions
                enhancement += f"- {suggestion}\n"
        
        enhancement += "\n# END LEARNED SECTION\n"
        
        return base_prompt + enhancement


# Global instance for easy access
seo_learner = SEOLearner()


def learn_from_clip_performance(clip_id: str, title: str, description: str, 
                               hashtags: List[str], analytics: Dict):
    """Convenience function to record clip performance."""
    seo_learner.record_performance(clip_id, title, description, hashtags, analytics)


def get_seo_improvement_suggestions() -> List[str]:
    """Get SEO improvement suggestions based on learned data."""
    return seo_learner.get_seo_improvement_suggestions()


def enhance_seo_prompt(base_prompt: str) -> str:
    """Enhance SEO prompt with learned insights."""
    return seo_learner.update_prompt_with_learnings(base_prompt)


def generate_performance_report() -> str:
    """Generate a report of what we've learned from performance data."""
    learner = seo_learner
    if not learner.learned_insights["clips"]:
        return "# SEO Performance Report\n\nNo performance data collected yet."
    
    clips = learner.learned_insights["clips"]
    total_clips = len(clips)
    avg_score = sum(c["performance_score"] for c in clips) / total_clips
    
    lines = [
        "# 📊 SEO Performance & Learning Report",
        f"**Total Clips Analyzed:** {total_clips}",
        f"**Average Performance Score:** {avg_score:.2f}",
        f"**Last Updated:** {learner.learned_insights['last_updated'] or 'Never'}",
        "",
        "## 🎓 What We've Learned",
    ]
    
    # Title patterns
    title_prefs = learner.get_learned_title_preferences()
    if title_prefs["best_patterns"]:
        lines.append("### ✅ Winning Title Patterns:")
        for pattern, score in title_prefs["best_patterns"][:3]:
            lines.append(f"- {pattern} (avg score: {score:.2f})")
    
    if title_prefs["avoid_patterns"]:
        lines.append("### ❌ Losing Title Patterns:")
        for pattern, score in title_prefs["avoid_patterns"][:3]:
            lines.append(f"- {pattern} (avg score: {score:.2f})")
    
    lines.append("")
    lines.append("## 💡 Improvement Suggestions:")
    suggestions = learner.get_seo_improvement_suggestions()
    for suggestion in suggestions:
        lines.append(f"- {suggestion}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test the learner
    print("Testing SEO Learner...")
    print(generate_performance_report())