"""SEOGenerator — generates SEO metadata for clips."""

from automation.memory.decision_store import DecisionStore
from automation.seo.analytics import Analytics


class SEOGenerator:
    def __init__(
        self,
        decision_store: DecisionStore,
        analytics: Analytics | None = None,
    ) -> None:
        self._store = decision_store
        self._analytics = analytics

    def generate(self, clip_data: dict) -> dict:
        clip_id = clip_data.get("clip_id", "unknown")
        title = clip_data.get("title", "")
        transcript_summary = clip_data.get("transcript_summary", "")

        seo_title = title[:60] + " - Shorts"
        description = (
            "🎬 " + clip_data.get("title", "") + "\n\n"
            + transcript_summary[:200] + "\n\n"
            + "#shorts #youtubeshorts"
        )

        words = [w for w in title.split() if len(w) > 2 and w.isalpha()]
        title_words = words[:3]
        tags = ["shorts", "youtubeshorts", "viral"] + title_words

        return {
            "clip_id": clip_id,
            "title": seo_title,
            "description": description,
            "tags": tags,
            "category": "Entertainment",
        }

    def generate_batch(self, clips: list[dict]) -> list[dict]:
        return [self.generate(c) for c in clips]

    def enhance_with_analytics(self, clip_data: dict) -> dict:
        result = self.generate(clip_data)
        if self._analytics is not None:
            summary = self._analytics.get_summary()
            tags = result["tags"]
            if summary.get("avg_score", 0) > 0.7:
                tags.append("highly_rated")
            if summary.get("published_count", 0) > 10:
                tags.append("popular_channel")
            result["tags"] = tags
        return result
