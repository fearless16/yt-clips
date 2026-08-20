"""Small plug-and-play facade for ingestion and learning."""

from __future__ import annotations

from typing import Protocol

from shorts_intelligence.learner import BayesianSegmentLearner
from shorts_intelligence.store import ShortsStore
from shorts_intelligence.youtube_source import IngestionBatch


class ShortsSource(Protocol):
    """Port implemented by YouTube and deterministic test sources."""

    def fetch(self, *, captured_at: str | None = None) -> IngestionBatch: ...


class ShortsIntelligence:
    """Independent application service with no legacy learner dependency."""

    def __init__(
        self,
        *,
        store: ShortsStore,
        source: ShortsSource,
        learner: BayesianSegmentLearner | None = None,
    ) -> None:
        self.store = store
        self.source = source
        self.learner = learner or BayesianSegmentLearner()

    def sync_and_fit(self, *, captured_at: str | None = None) -> dict:
        """Atomically sync current data, fit, and return compact evidence."""
        batch = self.source.fetch(captured_at=captured_at)
        ingested = self.store.ingest_batch(batch.records, batch.snapshots)
        summary = self.store.summary()
        model = self.learner.fit(self.store.training_rows())
        self.store.save_model(model, self.learner.config)
        recommendations = [
            {
                "feature_name": item.feature_name,
                "feature_value": item.feature_value,
                "effect": round(item.effect, 6),
                "effect_low": round(item.effect_lower_bound, 6),
                "n": item.sample_count,
                "confidence": round(item.confidence, 4),
            }
            for item in model.recommendations(limit=8)
        ]
        return {
            "shelf": batch.shelf_count,
            "catalog": summary["shorts"],
            "cricket": summary["cricket"],
            "non_cricket": summary["non_cricket"],
            "unknown": summary["unknown"],
            "snapshots_added": ingested["snapshots_added"],
            "issues": batch.issues,
            "model": {
                "observations": model.observations,
                "baseline": round(model.baseline_mean, 6),
                "recommendations": recommendations,
            },
        }

    def status(self) -> dict:
        """Return local status without network access."""
        return {"store": self.store.summary(), "model": self.store.latest_model()}
