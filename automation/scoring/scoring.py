"""Clip scoring with configurable feature extraction."""

from collections.abc import Callable
from typing import Any

from automation.scoring.feature_extractor import FeatureExtractor


class ClipScorer:
    """Scores clips based on extracted features using a weighted formula.

    Formula: base 0.5 + hook_bonus + payoff_bonus - profanity_penalty - length_penalty
      - hook_bonus: 0.1 per hook word, max 0.3
      - payoff_bonus: 0.1 per indicator, max 0.2
      - profanity_penalty: 0.2 if profanity
      - length_penalty: 0.1 if > 60s
    Score clamped to [0.0, 1.0].
    """

    def __init__(self, feature_extractor: FeatureExtractor | None = None) -> None:
        self._extractor = feature_extractor or FeatureExtractor()

    def score(self, clip_data: dict[str, Any]) -> dict[str, Any]:
        clip_id = clip_data.get("clip_id")
        if clip_id is None:
            raise ValueError("missing required field: clip_id")

        features = self._extractor.extract(clip_data)

        hook_bonus = min(features["hook_words_count"] * 0.1, 0.3)
        payoff_bonus = min(features["payoff_indicators"] * 0.1, 0.2)
        profanity_penalty = 0.2 if features["profanity_flag"] else 0.0
        length_penalty = 0.1 if features["length_category"] == "long" else 0.0

        raw = 0.5 + hook_bonus + payoff_bonus - profanity_penalty - length_penalty
        final_score = max(0.0, min(1.0, raw))

        return {
            "clip_id": clip_id,
            "score": final_score,
            "features": features,
            "breakdown": {
                "base": 0.5,
                "hook_bonus": hook_bonus,
                "payoff_bonus": payoff_bonus,
                "profanity_penalty": profanity_penalty,
                "length_penalty": length_penalty,
            },
        }

    def score_many(self, clips: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.score(c) for c in clips]

    def score_with_provider(
        self, clip_data: dict[str, Any], provider_fn: Callable
    ) -> dict[str, Any]:
        ai_score = self.score(clip_data)["score"]
        provider_score = provider_fn(clip_data)
        combined = (ai_score + provider_score) / 2.0

        return {
            "clip_id": clip_data.get("clip_id"),
            "provider_score": provider_score,
            "ai_score": ai_score,
            "combined_score": combined,
        }
