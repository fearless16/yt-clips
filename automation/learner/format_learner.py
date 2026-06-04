"""FormatLearner — tracks hook type and title pattern performance.

40% weight in scoring formula. Uses Bayesian EMA for updates and
Thompson Sampling for exploration to prevent feedback loops.
"""

import random
from datetime import datetime, timezone
from typing import Any


class FormatLearner:
    """Tracks hook type and title pattern performance. 40% weight."""

    HOOK_TYPES = ["reaction", "fan_war", "shock", "live", "debate", "question", "prediction"]

    TITLE_PATTERNS = [
        "long_title", "hinglish", "has_caps", "has_emoji",
        "has_pipe", "no_question", "has_question"
    ]

    def __init__(self, state_store, baseline_views: int = 290, alpha: float = 0.15,
                 exploration_rate: float = 0.15):
        """Initialize FormatLearner.

        Args:
            state_store: PersistentStateStore instance
            baseline_views: Channel average views for normalization
            alpha: EMA learning rate (higher = faster adaptation)
            exploration_rate: Thompson Sampling exploration probability
        """
        self._state = state_store
        self._baseline = baseline_views
        self._alpha = alpha
        self._exploration_rate = exploration_rate

    def process_metrics(self, clip_id: str, payload: dict) -> None:
        """Process a metrics_received event and update format scores.

        Args:
            clip_id: The clip ID
            payload: Event payload with views, hook_type, title_pattern
        """
        views = payload.get("views", 0)
        if views == 0:
            return

        signal = min(views / self._baseline, 3.0) / 3.0

        hook_type = payload.get("hook_type")
        if hook_type:
            self._update_hook_score(hook_type, signal, views)

        title_pattern = payload.get("title_pattern", {})
        if title_pattern:
            self._update_title_scores(title_pattern, signal, views)

    def _update_hook_score(self, hook_type: str, signal: float, views: int) -> None:
        """Update score for a specific hook type using Bayesian EMA.

        Args:
            hook_type: The hook type (e.g., "reaction", "shock")
            signal: Normalized view signal (0-1)
            views: Raw view count
        """
        scores = self._state.get("format_scores")
        if scores is None:
            scores = {"hook_types": {}, "title_patterns": {}}

        hook_types = scores.get("hook_types", {})
        current = hook_types.get(hook_type, {
            "score": 0.5,
            "n": 0,
            "avg_views": 0,
            "avg_engagement": 0,
            "last_updated": ""
        })

        n = current["n"]
        old_score = current["score"]
        old_avg = current["avg_views"]

        new_score = (1 - self._alpha) * old_score + self._alpha * signal
        new_avg = (old_avg * n + views) / (n + 1)

        hook_types[hook_type] = {
            "score": new_score,
            "n": n + 1,
            "avg_views": new_avg,
            "avg_engagement": current.get("avg_engagement", 0),
            "last_updated": datetime.now(timezone.utc).isoformat()
        }

        scores["hook_types"] = hook_types
        self._state.set("format_scores", scores)

    def _update_title_scores(self, title_pattern: dict, signal: float, views: int) -> None:
        """Update scores for title pattern features.

        Args:
            title_pattern: Dict of title features (has_emoji, is_hinglish, etc.)
            signal: Normalized view signal (0-1)
            views: Raw view count
        """
        scores = self._state.get("format_scores")
        if scores is None:
            scores = {"hook_types": {}, "title_patterns": {}}

        patterns = scores.get("title_patterns", {})

        length = title_pattern.get("length", 0)
        if length >= 60:
            self._update_pattern(patterns, "long_title", signal, views)
        elif length >= 30:
            self._update_pattern(patterns, "medium_title", signal, views)
        else:
            self._update_pattern(patterns, "short_title", signal, views)

        if title_pattern.get("is_hinglish"):
            self._update_pattern(patterns, "hinglish", signal, views)

        if title_pattern.get("has_caps"):
            self._update_pattern(patterns, "has_caps", signal, views)

        if title_pattern.get("has_emoji"):
            self._update_pattern(patterns, "has_emoji", signal, views)

        if title_pattern.get("has_pipe"):
            self._update_pattern(patterns, "has_pipe", signal, views)

        has_question = title_pattern.get("has_question", False)
        if has_question:
            self._update_pattern(patterns, "has_question", signal, views)
        else:
            self._update_pattern(patterns, "no_question", signal, views)

        scores["title_patterns"] = patterns
        self._state.set("format_scores", scores)

    def _update_pattern(self, patterns: dict, pattern_name: str,
                        signal: float, views: int) -> None:
        """Update a single title pattern score.

        Args:
            patterns: Dict of pattern scores
            pattern_name: Pattern name (e.g., "long_title", "hinglish")
            signal: Normalized view signal (0-1)
            views: Raw view count
        """
        current = patterns.get(pattern_name, {
            "score": 0.5,
            "n": 0,
            "avg_views": 0
        })

        n = current["n"]
        old_score = current["score"]
        old_avg = current["avg_views"]

        new_score = (1 - self._alpha) * old_score + self._alpha * signal
        new_avg = (old_avg * n + views) / (n + 1)

        patterns[pattern_name] = {
            "score": new_score,
            "n": n + 1,
            "avg_views": new_avg
        }

    def process_override(self, clip_id: str, payload: dict) -> None:
        """Process a manual_override event. Boosts the hook type score.

        Args:
            clip_id: The clip ID
            payload: Event payload with override_type
        """
        override_type = payload.get("override_type", "")
        if override_type == "keep":
            hook_type = payload.get("hook_type")
            if hook_type:
                scores = self._state.get("format_scores")
                if scores is None:
                    return

                hook_types = scores.get("hook_types", {})
                current = hook_types.get(hook_type, {"score": 0.5, "n": 0})
                current["score"] = min(1.0, current["score"] + 0.2)
                hook_types[hook_type] = current
                scores["hook_types"] = hook_types
                self._state.set("format_scores", scores)

    def get_hook_score(self, hook_type: str) -> float:
        """Get the score for a specific hook type.

        Args:
            hook_type: The hook type

        Returns:
            Score (0-1), defaults to 0.5 if not found
        """
        scores = self._state.get("format_scores")
        if scores is None:
            return 0.5

        hook_types = scores.get("hook_types", {})
        return hook_types.get(hook_type, {}).get("score", 0.5)

    def get_title_score(self, features: dict) -> float:
        """Get composite title pattern score.

        Args:
            features: Dict of title features

        Returns:
            Composite score (0-1)
        """
        scores = self._state.get("format_scores")
        if scores is None:
            return 0.5

        patterns = scores.get("title_patterns", {})
        if not patterns:
            return 0.5

        total = 0.0
        count = 0

        length = features.get("length", 0)
        if length >= 60:
            total += patterns.get("long_title", {}).get("score", 0.5)
            count += 1

        if features.get("is_hinglish"):
            total += patterns.get("hinglish", {}).get("score", 0.5)
            count += 1

        if features.get("has_pipe"):
            total += patterns.get("has_pipe", {}).get("score", 0.5)
            count += 1

        has_question = features.get("has_question", False)
        if has_question:
            total += patterns.get("has_question", {}).get("score", 0.5)
        else:
            total += patterns.get("no_question", {}).get("score", 0.5)
        count += 1

        return total / count if count > 0 else 0.5

    def select_hook_type(self) -> str:
        """Select a hook type using Thompson Sampling with exploration.

        Returns:
            Selected hook type
        """
        if random.random() < self._exploration_rate:
            return random.choice(self.HOOK_TYPES)

        scores = self._state.get("format_scores")
        if scores is None:
            return random.choice(self.HOOK_TYPES)

        hook_types = scores.get("hook_types", {})
        if not hook_types:
            return random.choice(self.HOOK_TYPES)

        best_type = None
        best_sample = -1.0

        for hook_type in self.HOOK_TYPES:
            data = hook_types.get(hook_type, {"score": 0.5, "n": 0})
            score = data.get("score", 0.5)
            n = data.get("n", 0)

            alpha_param = score * n + 1
            beta_param = (1 - score) * n + 1

            sample = random.betavariate(alpha_param, beta_param)

            if sample > best_sample:
                best_sample = sample
                best_type = hook_type

        return best_type if best_type else random.choice(self.HOOK_TYPES)

    def get_scores(self) -> dict:
        """Get all format scores.

        Returns:
            Dict with hook_types and title_patterns
        """
        scores = self._state.get("format_scores")
        return scores if scores else {"hook_types": {}, "title_patterns": {}}
