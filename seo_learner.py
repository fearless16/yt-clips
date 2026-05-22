"""
seo_learner.py — Self-improving SEO system that learns from past performance.
Analyzes which titles, hooks, and CTAs drive the most engagement and updates prompts.
"""
import json
import math
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from utils.config import load_config
from utils.logger import get_logger
from automation._cache import TTLCache

PERF_CACHE = TTLCache(maxsize=2, ttl=60)

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
        """Load historical performance data (cached 60s)."""
        cached = PERF_CACHE.get("perf_data")
        if cached is not None:
            return cached
        default = {
            "clips": [],
            "title_patterns": {},
            "hooks_performance": {},
            "ctas_performance": {},
            "hashtag_performance": {},
            "model_performance": {},
            "benchmark_history": [],
            "current_best_provider": None,
            "current_best_model": None,
            "last_updated": None
        }
        if self.performance_db.exists():
            try:
                with open(self.performance_db, "r") as f:
                    data = json.load(f)
                    PERF_CACHE.set("perf_data", data)
                    return data
            except Exception as e:
                log.warning("Failed to load SEO performance data: %s", e)
        PERF_CACHE.set("perf_data", default)
        return default

    def _save_performance_data(self):
        """Save performance data to disk."""
        self.learned_insights["last_updated"] = datetime.now().isoformat()
        with open(self.performance_db, "w") as f:
            json.dump(self.learned_insights, f, indent=2)

    def record_performance(self, clip_id: str, title: str, description: str, 
                         hashtags: List[str], analytics: Dict,
                         tags: List[str] = None, search_terms: List[str] = None,
                         provider: str = None, model: str = None):
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
            provider: Which provider generated the SEO (groq/openrouter/deepseek etc.)
            model: Which model generated the SEO
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
            "features": features,
            "provider": provider,
            "model": model,
        }
        
        self.learned_insights["clips"].append(record)
        
        # Keep only last 100 clips to prevent unbounded growth
        if len(self.learned_insights["clips"]) > 100:
            self.learned_insights["clips"] = self.learned_insights["clips"][-100:]
        
        # Update learned patterns
        self._update_learned_patterns(features, performance_score)
        
        # Track model/provider performance
        self._update_model_performance(provider, model, performance_score)
        
        self._save_performance_data()
        log.info(f"📊 Recorded SEO performance for {clip_id}: score={performance_score:.2f} [{provider}/{model}]")

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

    def _update_model_performance(self, provider: str = None, model: str = None, performance_score: float = 0):
        """Track which models/providers drive the best engagement."""
        if not provider and not model:
            return
        key = f"{provider or '?'}/{model or '?'}"
        mp = self.learned_insights["model_performance"]
        if key not in mp:
            mp[key] = {"count": 0, "total_score": 0.0, "avg_score": 0.0, "provider": provider, "model": model}
        mp[key]["count"] += 1
        mp[key]["total_score"] += performance_score
        mp[key]["avg_score"] = mp[key]["total_score"] / mp[key]["count"]
        self._update_best_model()

    def _update_best_model(self):
        """Auto-select the best performing provider/model combo."""
        mp = self.learned_insights["model_performance"]
        if not mp:
            return
        candidates = [(k, v["avg_score"], v["count"]) for k, v in mp.items() if v["count"] >= 2]
        if not candidates:
            return
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_key = candidates[0][0]
        best = mp[best_key]
        self.learned_insights["current_best_provider"] = best["provider"]
        self.learned_insights["current_best_model"] = best["model"]
        log.info(f"🏆 Best model auto-selected: {best_key} (avg score: {candidates[0][1]:.3f}, n={candidates[0][2]})")

    def get_best_model(self) -> Tuple[Optional[str], Optional[str]]:
        """Return the best (provider, model) based on learned performance."""
        return (
            self.learned_insights.get("current_best_provider"),
            self.learned_insights.get("current_best_model"),
        )

    def run_auto_benchmark(self):
        """
        Auto-discover best models by running a benchmark across all available
        providers. Tests JSON parsing, content grounding, and SEO quality.
        """
        import re
        log.info("🔬 Running auto-benchmark to discover best model...")
        from utils.ai_client import AIClient
        from seo import _parse_json_response
        ai = AIClient()
        available = ai.get_available_providers()

        # Use a real cricket clip transcript to test grounding
        benchmark_prompt = (
            "CONTEXT:\n"
            "  Match: RCB vs CSK IPL 2026\n"
            "  Scorecard: RCB 198/4 (Kohli 67(34), Faf 45(29)) | CSK 175/9 (Jadeja 42(28))\n\n"
            "CLIP TRANSCRIPT: 'Kohli ne maara six! M Chinnaswamy mein crowd pagal ho gaya! "
            "Faf du Plessis ne bhi acchi opening ki thi. Kohli ne 34 ball mein 67 banaye, "
            "8 fours aur 3 sixes. CSK ke bowlers ko koi chance nahi mila.'\n\n"
            "Generate YouTube SEO metadata for this clip.\n"
            "RULES: Only use players mentioned in transcript (Kohli, Faf). "
            "Description must be PLAIN TEXT, not a Python dict.\n"
            "Return ONLY valid JSON with fields: title, description, hashtags, search_terms."
        )

        # Players actually mentioned in transcript (for grounding check)
        grounded_players = {"kohli", "faf", "du plessis"}

        results = []
        for provider, models in available.items():
            for model in models[:2]:
                log.info(f"  Benchmarking {provider}/{model}...")
                try:
                    old_provider = ai._provider
                    old_model = ai._model
                    ai._provider = provider
                    ai._model = model
                    t0 = time.time()
                    resp = ai.generate_text(
                        benchmark_prompt,
                        system_instruction=(
                            "You are an elite YouTube Shorts SEO expert for Indian cricket. "
                            "Only use player names from the transcript. "
                            "Return ONLY valid JSON — no markdown, no explanation."
                        ),
                    )
                    latency = time.time() - t0
                    ai._provider = old_provider
                    ai._model = old_model

                    score = 0
                    title = ""
                    description = ""

                    # Parse JSON (handles markdown wrapping)
                    data = _parse_json_response(resp)

                    if data is None:
                        score = 5  # Returned something but not parseable
                    else:
                        # STRUCTURE (40 points)
                        if "title" in data and isinstance(data["title"], str) and len(data["title"]) > 10:
                            score += 10
                            title = data["title"]
                        if "description" in data and isinstance(data["description"], str) and len(data["description"]) > 100:
                            score += 10
                            description = data["description"]
                        if "hashtags" in data and isinstance(data["hashtags"], list) and len(data["hashtags"]) >= 3:
                            score += 10
                        if "search_terms" in data and isinstance(data["search_terms"], list) and len(data["search_terms"]) >= 5:
                            score += 10

                        # GROUNDING (30 points) - title uses players from transcript
                        title_lower = title.lower()
                        if any(p in title_lower for p in ["kohli", "virat"]):
                            score += 15
                        if "ipl" in title_lower or "2026" in title_lower:
                            score += 5
                        if "rcb" in title_lower or "csk" in title_lower or "virat" in title_lower:
                            score += 10

                        # QUALITY (30 points) - no dict syntax, unique title, proper format
                        has_dict_syntax = bool(re.search(r"\{[^}]{5,}\}", description))
                        if not has_dict_syntax and description:
                            score += 15  # Clean description
                        if title and not title.startswith("{"):
                            score += 5  # Title is not a dict
                        if any(p in title_lower for p in ["smash", "six", "fire", "brilliant", "clutch", "incredible"]):
                            score += 5  # Has power words
                        if title and ("|" in title or ":" in title):
                            score += 5  # Has pipe/colon format

                    # LATENCY PENALTY - more than 30s gets a penalty
                    if latency > 30:
                        score = max(0, score - 20)
                    elif latency > 15:
                        score = max(0, score - 10)

                    results.append({
                        "provider": provider,
                        "model": model,
                        "score": score,
                        "latency": round(latency, 2),
                        "timestamp": datetime.now().isoformat(),
                        "has_json": data is not None,
                        "title_preview": title[:60] if title else "N/A",
                    })
                    log.info(f"    Score: {score}/100, Latency: {latency:.1f}s, Title: {title[:60]}")
                except Exception as e:
                    log.warning(f"    Failed: {e}")

        results.sort(key=lambda x: x["score"], reverse=True)
        self.learned_insights["benchmark_history"].append({
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "top_result": results[0] if results else None,
        })

        if results and results[0]["score"] >= 40:
            best = results[0]
            self.learned_insights["current_best_provider"] = best["provider"]
            self.learned_insights["current_best_model"] = best["model"]
            log.info(f"🏆 Benchmark complete. Best: {best['provider']}/{best['model']} (score={best['score']}, latency={best['latency']}s)")
        self._save_performance_data()

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


# Lazy singleton — created on first access, no module-level side effects
_seo_learner_instance = None


def _get_learner() -> SEOLearner:
    global _seo_learner_instance
    if _seo_learner_instance is None:
        _seo_learner_instance = SEOLearner()
    return _seo_learner_instance


learn_from_clip_performance = lambda *a, **kw: _get_learner().record_performance(*a, **kw)
get_seo_improvement_suggestions = lambda *a, **kw: _get_learner().get_seo_improvement_suggestions(*a, **kw)
get_best_model = lambda *a, **kw: _get_learner().get_best_model(*a, **kw)
run_auto_benchmark = lambda *a, **kw: _get_learner().run_auto_benchmark(*a, **kw)


def enhance_seo_prompt(base_prompt: str) -> str:
    """Enhance SEO prompt with learned insights."""
    return _get_learner().update_prompt_with_learnings(base_prompt)


def generate_performance_report() -> str:
    """Generate a report of what we've learned from performance data."""
    learner = _get_learner()
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