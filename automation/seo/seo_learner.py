"""
seo_learner.py — Self-improving SEO system with dedup, time decay, trend tracking,
feature importance analysis, and optional LLM-enhanced insights.
"""
import json
import math
import os
import re
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from utils.config import load_config
from utils.logger import get_logger
from automation._cache import TTLCache

PERF_CACHE = TTLCache(maxsize=2, ttl=60)

cfg = load_config()
log = get_logger("seo_learner", cfg["logging"]["log_file"], cfg["logging"]["level"])
DECAY_HALF_LIFE_DAYS = cfg.get("learner", {}).get("decay_half_life_days", 30)
MIN_CLIPS_FOR_PATTERN = cfg.get("learner", {}).get("min_clips_for_pattern", 2)


def _time_decay_weight(timestamp_str: str, now: datetime | None = None) -> float:
    """Exponential decay weight: 1.0 for today, 0.5 after half-life."""
    if not timestamp_str:
        return 0.5
    try:
        ts = datetime.fromisoformat(timestamp_str)
        now = now or datetime.now()
        days_old = max(0, (now - ts).total_seconds() / 86400)
        return math.exp(-math.log(2) * days_old / DECAY_HALF_LIFE_DAYS)
    except Exception:
        return 0.5


def _stable_pattern_key(features: Dict) -> str:
    """Build a deterministic pattern key from LOW-CARDINALITY features only.

    Critically excludes raw numeric features (title_length, description_length,
    title_words, sections_count, tags_count, ...): including them made nearly
    every clip a unique pattern, so patterns never reached MIN_CLIPS_FOR_PATTERN
    and ``title_patterns`` learning was effectively inert. We key on booleans and
    binned/categorical strings (e.g. ``hashtag_count_bin``) which DO recur.
    """
    parts = []
    for k in sorted(features.keys()):
        v = features[k]
        if isinstance(v, bool):
            parts.append(f"{k}:{'true' if v else 'false'}")
        elif isinstance(v, str):
            # Categorical/binned features (e.g. hashtag_count_bin) — recurring.
            parts.append(f"{k}:{v}")
        # Numeric features are intentionally excluded (high cardinality).
    return "_".join(parts)


def _compute_weighted_avg(scores: List[float], weights: List[float]) -> float:
    total_w = sum(w for s, w in zip(scores, weights) if w > 0)
    if total_w == 0:
        return sum(scores) / len(scores) if scores else 0.0
    return sum(s * w for s, w in zip(scores, weights) if w > 0) / total_w


def _bin_hashtag_count(n: int) -> str:
    if n == 0:
        return "0"
    if n <= 3:
        return "1-3"
    if n <= 5:
        return "4-5"
    if n <= 10:
        return "6-10"
    return "10+"


class SEOLearner:
    """
    Learns from past YouTube Shorts performance to improve future SEO.
    Features: dedup, time decay, stable pattern keys, trend tracking,
    feature importance, optional LLM-enhanced insights.
    """

    def __init__(self):
        self.performance_db = Path("data/seo_performance.json")
        self.performance_db.parent.mkdir(exist_ok=True)
        self.learned_insights = self._load_performance_data()
        self._loaded_mtime = self._db_mtime()
        self._benchmark_lock = threading.Lock()

    def _db_mtime(self) -> float:
        """Last-modified time of the perf DB file (0.0 if it doesn't exist)."""
        try:
            return self.performance_db.stat().st_mtime
        except OSError:
            return 0.0

    def _maybe_reload(self):
        """Re-read the perf DB from disk if it changed since we last loaded it.

        Fixes Bug 4: ``get_best_model()`` used to return whatever was loaded at
        process/instance start (or last in-memory write). The benchmark and real
        performance ingestion can run in a *different* process or instance and
        write ``seo_performance.json``; without this, readers never saw those
        updates (stale best-model). We detect file changes via mtime, refresh
        ``learned_insights`` (and the shared TTL cache), so cross-process /
        cross-instance writes become visible — while still avoiding redundant
        disk reads when nothing changed.
        """
        current = self._db_mtime()
        if current <= self._loaded_mtime:
            return
        try:
            with open(self.performance_db, "r") as f:
                data = json.load(f)
            if "version" not in data:
                data["version"] = 1
                data.setdefault("feature_importance", {})
                data.setdefault("llm_insights", [])
            self.learned_insights = data
            PERF_CACHE.set("perf_data", data)
            self._loaded_mtime = current
            log.info("Reloaded seo_performance.json (changed on disk) — best model now %s/%s",
                     data.get("current_best_provider") or "-",
                     data.get("current_best_model") or "-")
        except Exception as e:
            log.warning("Perf DB changed but reload failed; keeping in-memory state: %s", e)

    def _load_performance_data(self) -> Dict:
        cached = PERF_CACHE.get("perf_data")
        if cached is not None:
            return cached
        default = {
            "clips": [],
            "title_patterns": {},
            "hooks_performance": {},
            "ctas_performance": {},
            "hashtag_performance": {},
            "feature_importance": {},
            "model_performance": {},
            "benchmark_history": [],
            "current_best_provider": None,
            "current_best_model": None,
            "last_updated": None,
            "llm_insights": [],
            "version": 2,
        }
        if self.performance_db.exists():
            try:
                with open(self.performance_db, "r") as f:
                    data = json.load(f)
                    if "version" not in data:
                        data["version"] = 1
                        data.setdefault("feature_importance", {})
                        data.setdefault("llm_insights", [])
                    PERF_CACHE.set("perf_data", data)
                    return data
            except Exception as e:
                log.warning("Failed to load SEO performance data: %s", e)
        PERF_CACHE.set("perf_data", default)
        return default

    def _save_performance_data(self):
        import tempfile
        self.learned_insights["last_updated"] = datetime.now().isoformat()
        self.learned_insights["version"] = 2
        # Atomic write: write to temp file then rename
        db_path = Path(self.performance_db)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=str(db_path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.learned_insights, f, indent=2)
            os.replace(tmp_path, str(db_path))
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        # Keep our reload watermark + shared cache in sync with what we just
        # wrote, so a subsequent _maybe_reload() doesn't needlessly re-read our
        # own data (and other in-process readers see it immediately).
        self._loaded_mtime = self._db_mtime()
        PERF_CACHE.set("perf_data", self.learned_insights)

    def _dedup_clips(self, new_record: Dict):
        """Append-only versioned clip storage.

        Never mutates an existing entry (Invariant 1: append-only events).
        Each unique clip_id gets a monotonically increasing version number.
        Oldest entries trimmed when count exceeds 100.
        """
        clips = self.learned_insights["clips"]
        cid = new_record["clip_id"]
        version = 1
        for c in clips:
            if c.get("clip_id") == cid:
                version = max(version, c.get("_version", 0) + 1)
        new_record["_version"] = version
        clips.append(new_record)
        if len(clips) > 100:
            clips.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            self.learned_insights["clips"] = clips[:100]

    def record_performance(self, clip_id: str, title: str, description: str,
                           hashtags: List[str], analytics: Dict,
                           tags: List[str] = None, search_terms: List[str] = None,
                           provider: str = None, model: str = None):
        features = self._extract_seo_features(title, description, hashtags)
        features["tags_count"] = len(tags or [])
        features["search_terms_count"] = len(search_terms or [])

        performance_score = self._calculate_performance_score(analytics)

        # Persist the title inside analytics so downstream LLM-insight prompts
        # (which read analytics.title) show real titles instead of "?".
        analytics = {**analytics, "title": title}

        record = {
            "clip_id": clip_id,
            "timestamp": datetime.now().isoformat(),
            "performance_score": performance_score,
            "analytics": analytics,
            "features": features,
            "provider": provider,
            "model": model,
        }

        self._dedup_clips(record)
        self._update_learned_patterns(features, performance_score, record["timestamp"])
        self._update_model_performance(
            provider, model, performance_score, timestamp=record["timestamp"]
        )
        self._update_feature_importance()
        self._save_performance_data()
        log.info("Recorded SEO performance for %s: score=%.2f [%s/%s]",
                 clip_id, performance_score, provider or "?", model or "?")

    def _extract_seo_features(self, title: str, description: str, hashtags: List[str]) -> Dict:
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
            "description_length": len(description),
            "has_sections": all(s.lower() in description.lower() for s in sections),
            "sections_count": sum(1 for s in sections if s.lower() in description.lower()),
            "has_cta": any(word in description.lower() for word in [
                "like", "subscribe", "share", "comment", "follow", "bell"
            ]),
            "hashtag_count_bin": _bin_hashtag_count(len(hashtags)),
            "has_shorts_hashtag": any(h.lower() == "#shorts" for h in hashtags),
            "has_ipl_hashtag": any("#ipl" in h.lower() for h in hashtags),
            "has_team_hashtag": any(any(team in h.lower() for team in [
                "rcb", "csk", "mi", "kkr", "srh", "dc", "pbks", "rr", "gt", "lsg"
            ]) for h in hashtags),
            "starts_with_live_emoji": title.startswith("🔴"),
            "has_multiple_pipes": title.count("|") >= 2,
        }
        return features

    def _calculate_performance_score(self, analytics: Dict) -> float:
        views = analytics.get("viewCount", 0)
        likes = analytics.get("likeCount", 0)
        comments = analytics.get("commentCount", 0)

        views_score = min(1.0, math.log(max(1, views)) / 15)
        engagement_rate = (likes + comments) / max(1, views)
        # Apply view-based credibility scaling to avoid noise on low-view clips
        credibility = min(1.0, views / 100.0)
        engagement_score = min(1.0, engagement_rate * 20) * credibility

        has_retention = analytics.get("retention") is not None
        has_ctr = analytics.get("ctr") is not None

        if has_retention and has_ctr:
            score = (
                views_score * 0.4 +
                engagement_score * 0.3 +
                min(1.0, max(0.0, analytics["retention"])) * 0.2 +
                min(1.0, max(0.0, analytics["ctr"] * 10)) * 0.1
            )
        elif has_retention:
            score = (
                views_score * 0.5 +
                engagement_score * 0.3 +
                min(1.0, max(0.0, analytics["retention"])) * 0.2
            )
        elif has_ctr:
            score = (
                views_score * 0.5 +
                engagement_score * 0.3 +
                min(1.0, max(0.0, analytics["ctr"] * 10)) * 0.2
            )
        else:
            score = views_score * 0.6 + engagement_score * 0.4

        return min(1.0, max(0.0, score))

    def _update_learned_patterns(self, features: Dict, performance_score: float, timestamp: str):
        now = datetime.now()
        weight = _time_decay_weight(timestamp, now)

        # Stable pattern key
        title_pattern = _stable_pattern_key(features)

        tp = self.learned_insights["title_patterns"]
        if title_pattern not in tp:
            tp[title_pattern] = {
                "count": 0,
                "total_score": 0.0,
                "total_weight": 0.0,
                "avg_score": 0.0,
                "scores": [],
            }

        pd = tp[title_pattern]
        pd["count"] += 1
        pd["total_score"] += performance_score * weight
        pd["total_weight"] += weight
        pd["avg_score"] = pd["total_score"] / pd["total_weight"] if pd["total_weight"] > 0 else performance_score
        pd["scores"].append({"score": performance_score, "weight": weight, "ts": timestamp})
        if len(pd["scores"]) > 50:
            pd["scores"] = pd["scores"][-50:]

        # Track hooks and CTAs
        if features.get("has_pipe_format"):
            self.learned_insights["hooks_performance"].setdefault("pipe_format", []).append(performance_score)
        if features.get("has_sections"):
            self.learned_insights["ctas_performance"].setdefault("full_sections", []).append(performance_score)

        # Track hashtag performance (binned)
        htag_bin = features.get("hashtag_count_bin", "?")
        entry = f"hashtag_count:{htag_bin}"
        self.learned_insights["hashtag_performance"].setdefault(entry, []).append(performance_score)
        for key in ["has_shorts_hashtag", "has_ipl_hashtag", "has_team_hashtag"]:
            if key in features:
                entry2 = f"{key}:{features[key]}"
                self.learned_insights["hashtag_performance"].setdefault(entry2, []).append(performance_score)

    def _update_model_performance(self, provider: str = None, model: str = None,
                                   performance_score: float = 0,
                                   timestamp: str = None):
        """Track per-model performance with time-decay weighting.

        Stores both a raw ``avg_score`` (simple mean, for reference) and a
        ``weighted_avg_score`` (exponentially time-decayed) so that recent
        clip performance drives model selection rather than old history.
        """
        if not provider and not model:
            return
        key = f"{provider or '?'}/{model or '?'}"
        mp = self.learned_insights["model_performance"]
        if key not in mp:
            mp[key] = {
                "count": 0,
                "total_score": 0.0,
                "avg_score": 0.0,
                "weighted_total_score": 0.0,
                "weighted_total_weight": 0.0,
                "weighted_avg_score": 0.0,
                "provider": provider,
                "model": model,
            }
        ts = timestamp or datetime.now().isoformat()
        decay_w = _time_decay_weight(ts)
        mp[key]["count"] += 1
        mp[key]["total_score"] += performance_score
        mp[key]["avg_score"] = mp[key]["total_score"] / mp[key]["count"]
        mp[key]["weighted_total_score"] += performance_score * decay_w
        mp[key]["weighted_total_weight"] += decay_w
        if mp[key]["weighted_total_weight"] > 0:
            mp[key]["weighted_avg_score"] = (
                mp[key]["weighted_total_score"] / mp[key]["weighted_total_weight"]
            )
        self._update_best_model()

    def _update_best_model(self):
        self._recompute_best_model()

    def _recompute_best_model(self) -> Optional[str]:
        """Single source of truth for current_best_provider/model.

        Precedence (real performance always wins over synthetic benchmarks):
          1. Real ``model_performance`` entries with count >= MIN_CLIPS_FOR_PATTERN,
             ranked by avg_score.
          2. Cold-start fallback: the most recent auto-benchmark top_result
             (only if it scored >= 40).

        Returns "real" | "benchmark" | None describing which source was used.
        """
        mp = self.learned_insights.get("model_performance", {})
        candidates = [(k, v) for k, v in mp.items()
                      if v.get("count", 0) >= MIN_CLIPS_FOR_PATTERN]
        if candidates:
            candidates.sort(key=lambda kv: kv[1]["avg_score"], reverse=True)
            best = candidates[0][1]
            self.learned_insights["current_best_provider"] = best["provider"]
            self.learned_insights["current_best_model"] = best["model"]
            log.info("Best model (real data): %s/%s avg=%.3f n=%d",
                     best["provider"], best["model"], best["avg_score"], best["count"])
            return "real"

        # Cold-start fallback: latest benchmark top result.
        history = self.learned_insights.get("benchmark_history", [])
        if history:
            top = history[-1].get("top_result")
            if top and top.get("score", 0) >= 40:
                self.learned_insights["current_best_provider"] = top["provider"]
                self.learned_insights["current_best_model"] = top["model"]
                log.info("Best model (benchmark cold-start): %s/%s score=%d",
                         top["provider"], top["model"], top["score"])
                return "benchmark"
        return None

    def get_best_model(self) -> Tuple[Optional[str], Optional[str]]:
        # Pick up benchmark / real-performance writes made by another instance
        # or process before answering (Bug 4: avoid returning stale data).
        self._maybe_reload()
        return (
            self.learned_insights.get("current_best_provider"),
            self.learned_insights.get("current_best_model"),
        )

    def run_auto_benchmark(self):
        log.info("Running auto-benchmark to discover best model...")
        from utils.ai_client import AIClient
        from automation.seo.seo import _parse_json_response
        ai = AIClient()
        available = ai.get_available_providers()

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

        results = []
        for provider, models in available.items():
            for model in models[:2]:
                log.info("  Benchmarking %s/%s...", provider, model)
                try:
                    with self._benchmark_lock:
                        old_provider = ai._provider
                        old_model = ai._model
                        ai._provider = provider
                        ai._model = model
                        try:
                            t0 = time.time()
                            resp = ai.generate_text(
                                benchmark_prompt,
                                system_instruction=(
                                    "You are an elite YouTube Shorts SEO expert for Indian cricket. "
                                    "Only use player names from the transcript. "
                                    "Return ONLY valid JSON — no markdown, no explanation."
                                ),
                            )
                        finally:
                            ai._provider = old_provider
                            ai._model = old_model
                    latency = time.time() - t0

                    score = 0
                    title = ""
                    description = ""
                    data = _parse_json_response(resp)

                    if data is None:
                        score = 5
                    else:
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

                        title_lower = title.lower()
                        if any(p in title_lower for p in ["kohli", "virat"]):
                            score += 15
                        if "ipl" in title_lower or "2026" in title_lower:
                            score += 5
                        if "rcb" in title_lower or "csk" in title_lower or "virat" in title_lower:
                            score += 10

                        has_dict_syntax = bool(re.search(r"\{[^}]{5,}\}", description))
                        if not has_dict_syntax and description:
                            score += 15
                        if title and not title.startswith("{"):
                            score += 5
                        if any(p in title_lower for p in ["smash", "six", "fire", "brilliant", "clutch", "incredible"]):
                            score += 5
                        if title and ("|" in title or ":" in title):
                            score += 5

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
                    log.info("    Score: %d/100, Latency: %.1fs, Title: %s", score, latency, title[:60])
                except Exception as e:
                    log.warning("    Failed: %s", e)

        results.sort(key=lambda x: x["score"], reverse=True)
        self.learned_insights["benchmark_history"].append({
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "top_result": results[0] if results else None,
        })

        # Commit best model via the unified precedence (real performance data
        # wins; this synthetic benchmark only applies as a cold-start fallback).
        if results and results[0]["score"] >= 40:
            source = self._recompute_best_model()
            best = results[0]
            log.info("Benchmark complete. Top: %s/%s (score=%d, latency=%.1fs); selection source=%s",
                     best['provider'], best['model'], best['score'], best['latency'],
                     source or "none")
        self._save_performance_data()

    def _update_feature_importance(self):
        """Rank boolean/categorical features by correlation with performance."""
        clips = self.learned_insights["clips"]
        if len(clips) < MIN_CLIPS_FOR_PATTERN:
            return

        now = datetime.now()
        bool_features = [
            "has_pipe_format", "has_power_word", "has_player_name",
            "has_score", "has_emoji", "has_sections", "has_cta",
            "has_shorts_hashtag", "has_ipl_hashtag", "has_team_hashtag",
            "starts_with_live_emoji", "has_multiple_pipes",
        ]
        importance = {}
        for feat in bool_features:
            with_feat = []
            without_feat = []
            for c in clips:
                f = c.get("features", {})
                if feat not in f:
                    continue
                w = _time_decay_weight(c.get("timestamp", ""), now)
                if f[feat]:
                    with_feat.append((c["performance_score"], w))
                else:
                    without_feat.append((c["performance_score"], w))
            if len(with_feat) >= MIN_CLIPS_FOR_PATTERN and len(without_feat) >= MIN_CLIPS_FOR_PATTERN:
                avg_with = _compute_weighted_avg(
                    [s for s, _ in with_feat], [w for _, w in with_feat]
                )
                avg_without = _compute_weighted_avg(
                    [s for s, _ in without_feat], [w for _, w in without_feat]
                )
                delta = avg_with - avg_without
                importance[feat] = {
                    "delta": round(delta, 4),
                    "avg_with": round(avg_with, 4),
                    "avg_without": round(avg_without, 4),
                    "count_with": len(with_feat),
                    "count_without": len(without_feat),
                }

        self.learned_insights["feature_importance"] = importance

    def get_feature_importance(self) -> Dict:
        return self.learned_insights.get("feature_importance", {})

    def get_learned_title_preferences(self) -> Dict:
        preferences = {
            "best_patterns": [],
            "avoid_patterns": [],
            "hook_effectiveness": {},
            "recommendations": [],
            "trending_up": [],
            "trending_down": [],
        }

        feat_imp = self.get_feature_importance()
        # Generate recommendations from feature importance (most reliable)
        for feat, data in sorted(feat_imp.items(), key=lambda x: -x[1]["delta"]):
            if data["delta"] > 0.05:
                label = feat.replace("has_", "").replace("_", " ").title()
                preferences["recommendations"].append(
                    f"Use {label} (+{data['delta']:.2f} score impact, {data['count_with']} samples)"
                )
            elif data["delta"] < -0.05:
                label = feat.replace("has_", "").replace("_", " ").title()
                preferences["recommendations"].append(
                    f"Avoid {label} ({data['delta']:.2f} score impact, {data['count_without']} samples)"
                )

        # Pattern-based preferences (time-decayed)
        for pattern, data in self.learned_insights["title_patterns"].items():
            if data["count"] >= MIN_CLIPS_FOR_PATTERN:
                if data["avg_score"] > 0.6:
                    preferences["best_patterns"].append((pattern, data["avg_score"], data["count"]))
                elif data["avg_score"] < 0.4:
                    preferences["avoid_patterns"].append((pattern, data["avg_score"], data["count"]))

                # Trend: compare recent (last 30d) vs older scores
                now = datetime.now()
                recent = [s for s in data.get("scores", [])
                          if s.get("ts") and (now - datetime.fromisoformat(s["ts"])).days <= 30]
                older = [s for s in data.get("scores", [])
                         if s.get("ts") and (now - datetime.fromisoformat(s["ts"])).days > 30]
                if len(recent) >= 2 and len(older) >= 2:
                    avg_recent = sum(s["score"] for s in recent) / len(recent)
                    avg_older = sum(s["score"] for s in older) / len(older)
                    if avg_recent > avg_older * 1.1:
                        preferences["trending_up"].append((pattern, avg_recent - avg_older))
                    elif avg_older > avg_recent * 1.1:
                        preferences["trending_down"].append((pattern, avg_older - avg_recent))

        preferences["best_patterns"].sort(key=lambda x: x[1], reverse=True)
        preferences["avoid_patterns"].sort(key=lambda x: x[1])

        return preferences

    def get_seo_improvement_suggestions(self) -> List[str]:
        suggestions = []
        prefs = self.get_learned_title_preferences()
        suggestions.extend(prefs["recommendations"])

        for hook_type, scores in self.learned_insights["hooks_performance"].items():
            if len(scores) >= 2:
                avg_score = sum(scores) / len(scores)
                if avg_score > 0.6:
                    suggestions.append(f"Use more {hook_type} hooks — they perform well (avg: {avg_score:.2f})")
                elif avg_score < 0.4:
                    suggestions.append(f"Avoid {hook_type} hooks — they underperform (avg: {avg_score:.2f})")

        for cta_type, scores in self.learned_insights["ctas_performance"].items():
            if len(scores) >= 2:
                avg_score = sum(scores) / len(scores)
                if avg_score > 0.6:
                    suggestions.append(f"Use more {cta_type} style CTAs — they drive engagement")
                elif avg_score < 0.4:
                    suggestions.append(f"Improve {cta_type} CTAs — they're not working well")

        # Trend signals
        if prefs.get("trending_up"):
            suggestions.append("📈 New patterns rising in performance — check prompt enhancements")
        if prefs.get("trending_down"):
            suggestions.append("📉 Some previously strong patterns declining — consider refreshing")

        if not suggestions:
            suggestions.append("Keep experimenting! Need more data to give specific recommendations.")
        return suggestions

    def update_prompt_with_learnings(self, base_prompt: str) -> str:
        prefs = self.get_learned_title_preferences()
        suggestions = self.get_seo_improvement_suggestions()
        feat_imp = self.get_feature_importance()

        enhancement = "\n\n# LEARNED FROM PERFORMANCE DATA\n"

        # Feature importance (most data-driven signal)
        positive = {k: v for k, v in feat_imp.items() if v["delta"] > 0.03}
        negative = {k: v for k, v in feat_imp.items() if v["delta"] < -0.03}

        if positive:
            enhancement += "## ✅ Features that boost performance:\n"
            for feat, data in sorted(positive.items(), key=lambda x: -x[1]["delta"])[:4]:
                label = feat.replace("has_", "").replace("_", " ").title()
                enhancement += f"- Include {label}: +{data['delta']:.0%} avg score\n"

        if negative:
            enhancement += "\n## ❌ Features that hurt performance:\n"
            for feat, data in sorted(negative.items(), key=lambda x: x[1]["delta"])[:3]:
                label = feat.replace("has_", "").replace("_", " ").title()
                enhancement += f"- Avoid {label}: {data['delta']:.0%} avg score\n"

        if prefs.get("trending_up"):
            enhancement += "\n## 📈 Trending UP (improving patterns):\n"
            for pattern, delta in prefs["trending_up"][:2]:
                readable = _pattern_to_readable(pattern)
                enhancement += f"- {readable} (rising by {delta:.2f})\n"

        # LLM insights if available
        llm_insights = self.learned_insights.get("llm_insights", [])
        if llm_insights:
            latest = llm_insights[-1]
            if "insights" in latest:
                enhancement += f"\n## 🧠 AI Analysis:\n{latest['insights']}\n"

        if suggestions:
            enhancement += "\n## 📋 Recommendations:\n"
            for s in suggestions[:4]:
                enhancement += f"- {s}\n"

        return base_prompt + enhancement

    def generate_llm_insights(self) -> Optional[str]:
        """Feed performance data to LLM for deep analytical insights."""
        clips = self.learned_insights["clips"]
        if len(clips) < 5:
            log.info("Not enough data for LLM insights (need 5+ clips)")
            return None

        try:
            from utils.ai_client import AIClient
        except ImportError:
            log.warning("AIClient not available — skipping LLM insights")
            return None

        # Build a compact data summary
        total = len(clips)
        scores = [c["performance_score"] for c in clips]
        avg = sum(scores) / total
        top = sorted(clips, key=lambda x: -x["performance_score"])[:3]
        low = sorted(clips, key=lambda x: x["performance_score"])[:3]

        feat_imp = self.get_feature_importance()
        feat_summary = ""
        for feat, data in sorted(feat_imp.items(), key=lambda x: -abs(x[1]["delta"]))[:5]:
            feat_summary += f"  {feat}: +{data['delta']:.2f} impact ({data['count_with']} yes, {data['count_without']} no)\n"

        patterns = sorted(
            self.learned_insights["title_patterns"].items(),
            key=lambda x: -x[1].get("avg_score", 0),
        )[:5]
        pattern_summary = "\n".join(
            f"  {p}: avg={d['avg_score']:.3f} count={d['count']}" for p, d in patterns
        )

        top_titles = "\n".join(f"  - {c.get('analytics', {}).get('title', '?')[:60]} ({c['performance_score']:.3f})" for c in top)
        low_titles = "\n".join(f"  - {c.get('analytics', {}).get('title', '?')[:60]} ({c['performance_score']:.3f})" for c in low)

        prompt = f"""Analyze this YouTube Shorts SEO performance data and give 3-5 specific, actionable insights:

SUMMARY:
- Total clips analyzed: {total}
- Average performance score: {avg:.3f}
- Score range: {min(scores):.3f} – {max(scores):.3f}

FEATURE IMPACT (delta = score change when feature is present):
{feat_summary or "  (insufficient data for feature analysis)"}

TOP TITLE PATTERNS (time-decayed avg score):
{pattern_summary or "  (none yet)"}

TOP PERFORMING TITLES:
{top_titles}

LOWEST PERFORMING TITLES:
{low_titles}

Return concise bullet points. Focus on what the user should START doing, STOP doing, and CONTINUE doing."""
        try:
            ai = AIClient()
            resp = ai.generate_text(
                prompt,
                system_instruction="You are a YouTube SEO data analyst. Give short, specific, actionable insights. No fluff.",
            )
            if resp and len(resp) > 20:
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "insights": resp.strip(),
                }
                self.learned_insights.setdefault("llm_insights", []).append(entry)
                if len(self.learned_insights["llm_insights"]) > 10:
                    self.learned_insights["llm_insights"] = self.learned_insights["llm_insights"][-10:]
                self._save_performance_data()
                log.info("LLM insights generated successfully")
                return resp.strip()
        except Exception as e:
            log.warning("LLM insight generation failed: %s", e)
        return None

    def get_llm_insights(self) -> List[Dict]:
        return self.learned_insights.get("llm_insights", [])


# ─── Module-level convenience functions ─────────────────────────────────────

_seo_learner_instance = None
_seo_learner_lock = threading.Lock()


def _get_learner() -> "SEOLearner":
    """Thread-safe lazy singleton for SEOLearner."""
    global _seo_learner_instance
    # Fast-path: already initialised (no lock needed after first creation)
    if _seo_learner_instance is not None:
        return _seo_learner_instance
    with _seo_learner_lock:
        # Double-checked locking: re-test after acquiring lock
        if _seo_learner_instance is None:
            _seo_learner_instance = SEOLearner()
    return _seo_learner_instance


learn_from_clip_performance = lambda *a, **kw: _get_learner().record_performance(*a, **kw)
get_seo_improvement_suggestions = lambda *a, **kw: _get_learner().get_seo_improvement_suggestions(*a, **kw)
get_best_model = lambda *a, **kw: _get_learner().get_best_model(*a, **kw)
run_auto_benchmark = lambda *a, **kw: _get_learner().run_auto_benchmark(*a, **kw)
generate_llm_insights = lambda *a, **kw: _get_learner().generate_llm_insights(*a, **kw)
get_feature_importance = lambda *a, **kw: _get_learner().get_feature_importance(*a, **kw)
get_llm_insights = lambda *a, **kw: _get_learner().get_llm_insights(*a, **kw)


def enhance_seo_prompt(base_prompt: str) -> str:
    return _get_learner().update_prompt_with_learnings(base_prompt)


def generate_performance_report() -> str:
    learner = _get_learner()
    if not learner.learned_insights["clips"]:
        return "No performance data collected yet."

    clips = learner.learned_insights["clips"]
    total = len(clips)
    scores = [c["performance_score"] for c in clips]
    avg_score = sum(scores) / total
    unique_ids = len(set(c["clip_id"] for c in clips))

    parts = [
        "=" * 60,
        "SEO LEARNER — PERFORMANCE REPORT",
        "=" * 60,
        f"Total records: {total} (unique videos: {unique_ids})",
        f"Score range: {min(scores):.3f} – {max(scores):.3f} (avg: {avg_score:.3f})",
        f"Time decay half-life: {DECAY_HALF_LIFE_DAYS} days",
        "",
    ]

    feat_imp = learner.get_feature_importance()
    if feat_imp:
        parts.append("FEATURE IMPORTANCE (delta when feature present):")
        for feat, data in sorted(feat_imp.items(), key=lambda x: -abs(x[1]["delta"])):
            arrow = "▲" if data["delta"] > 0 else "▼"
            parts.append(f"  {arrow} {feat}: {data['delta']:+.3f} "
                         f"(with={data['count_with']}, without={data['count_without']})")
        parts.append("")

    tp = learner.learned_insights["title_patterns"]
    if tp:
        patterns = [(k, v["avg_score"], v["count"]) for k, v in tp.items() if v["count"] >= MIN_CLIPS_FOR_PATTERN]
        patterns.sort(key=lambda x: -x[1])
        if patterns:
            parts.append("TOP PATTERNS (time-decayed avg, min {MIN_CLIPS_FOR_PATTERN}+ clips):")
            for pattern, score, count in patterns[:8]:
                readable = _pattern_to_readable(pattern)
                bar = "█" * max(1, int(score * 20))
                parts.append(f"  {bar} {score:.3f} ({count}x) — {readable}")
            parts.append("")

    llm = learner.get_llm_insights()
    if llm:
        parts.append("LATEST LLM INSIGHTS:")
        parts.append(f"  {llm[-1].get('insights', '').strip()}")
        parts.append("")

    suggestions = learner.get_seo_improvement_suggestions()
    if suggestions:
        parts.append("SUGGESTIONS:")
        for s in suggestions:
            parts.append(f"  • {s}")

    parts.append("=" * 60)
    return "\n".join(parts)


def _pattern_to_readable(pattern: str) -> str:
    parts = pattern.split("_")
    readable = []
    for part in parts:
        if ":" not in part:
            continue
        key, val = part.split(":", 1)
        if val != "true":
            continue
        label = {
            "has_pipe_format": "Pipe format (Team vs Team | Tournament)",
            "has_power_word": "Power words (smashes, destroys)",
            "has_player_name": "Player names in title",
            "has_score": "Scores/stats in title",
            "has_emoji": "Emoji in title",
            "has_sections": "Full description sections",
            "has_cta": "Call-to-action in description",
            "has_shorts_hashtag": "#Shorts hashtag",
            "has_ipl_hashtag": "#IPL hashtag",
            "has_team_hashtag": "Team hashtag",
            "hashtag_count_bin": "Hashtag count",
            "starts_with_live_emoji": "Live circle emoji prefix (🔴)",
            "has_multiple_pipes": "Multiple structure separators (|)",
        }.get(key, key.replace("_", " ").replace("has ", "").title())
        readable.append(label)
    if not readable:
        missing = []
        for part in parts:
            if ":" not in part:
                continue
            key, val = part.split(":", 1)
            if val != "false":
                continue
            label = {
                "has_pipe_format": "Pipe format",
                "has_power_word": "Power words",
                "has_player_name": "Player names",
                "has_score": "Scores/stats",
                "has_emoji": "Emoji",
                "has_sections": "Description sections",
                "has_cta": "CTA",
                "has_shorts_hashtag": "#Shorts",
                "has_ipl_hashtag": "#IPL",
                "has_team_hashtag": "Team hashtag",
            }.get(key, key)
            missing.append(f"No {label}")
        readable = missing
    return "; ".join(readable) if readable else pattern


if __name__ == "__main__":
    print(generate_performance_report())
