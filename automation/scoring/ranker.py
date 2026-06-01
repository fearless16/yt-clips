"""Clip ranking by score with top-k and best-selection."""

from typing import Any

from automation.scoring.scoring import ClipScorer


class ClipRanker:
    """Ranks clips by descending score, assigning a rank field."""

    def __init__(self, scorer: ClipScorer) -> None:
        self._scorer = scorer

    def rank(self, clips: list[dict[str, Any]]) -> list[dict[str, Any]]:
        scored = self._scorer.score_many(clips)
        scored.sort(key=lambda x: x["score"], reverse=True)
        return [{**item, "rank": i + 1} for i, item in enumerate(scored)]

    def top_k(self, clips: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
        return self.rank(clips)[:k]

    def select_best(self, clips: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not clips:
            return None
        return self.rank(clips)[0]
