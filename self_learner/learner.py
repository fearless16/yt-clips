"""learner.py — Learning engine: observe, extract patterns, predict.

Ties PersistentMemory and KnowledgeBase together into a learning loop.
Observations are stored as facts and analyzed for recurring patterns.
"""

import statistics
import time
from collections import defaultdict
from typing import Any, Optional

from self_learner.memory import PersistentMemory
from self_learner.knowledge import Fact, KnowledgeBase


_OBSERVATION_RELATION = "was_observed"
_PATTERN_RELATION = "is_pattern"
_PREDICTION_RELATION = "predicts"


class Observation:
    """A single observed event.

    Attributes:
        event_type: Category of the event (e.g., ``"pipeline_run"``).
        attributes: Key-value payload describing the event.
        timestamp: When the event occurred.
        source: Origin identifier.
    """

    __slots__ = ("event_type", "attributes", "timestamp", "source")

    def __init__(self, event_type: str, attributes: dict[str, Any],
                 timestamp: Optional[float] = None,
                 source: str = "learner"):
        self.event_type = event_type
        self.attributes = attributes
        self.timestamp = timestamp or time.time()
        self.source = source


class Pattern:
    """A learned pattern extracted from repeated observations.

    Attributes:
        event_type: The event category this pattern describes.
        attributes: Template of expected attribute keys with observed values.
        confidence: How reliable this pattern is (0-1).
        sample_count: Number of observations supporting this pattern.
        last_updated: When the pattern was last refined.
    """

    __slots__ = ("event_type", "attributes", "confidence",
                 "sample_count", "last_updated")

    def __init__(self, event_type: str, attributes: dict[str, Any],
                 confidence: float = 0.5, sample_count: int = 1,
                 last_updated: Optional[float] = None):
        self.event_type = event_type
        self.attributes = attributes
        self.confidence = confidence
        self.sample_count = sample_count
        self.last_updated = last_updated or time.time()


class Prediction:
    """A prediction derived from learned patterns.

    Attributes:
        event_type: The predicted event category.
        predicted_attributes: Expected attribute values.
        confidence: Overall confidence (0-1).
        supporting_patterns: Number of patterns that support this prediction.
        timestamp: When the prediction was made.
    """

    __slots__ = ("event_type", "predicted_attributes", "confidence",
                 "supporting_patterns", "timestamp")

    def __init__(self, event_type: str, predicted_attributes: dict[str, Any],
                 confidence: float = 0.0, supporting_patterns: int = 0,
                 timestamp: Optional[float] = None):
        self.event_type = event_type
        self.predicted_attributes = predicted_attributes
        self.confidence = confidence
        self.supporting_patterns = supporting_patterns
        self.timestamp = timestamp or time.time()


class Learner:
    """Autonomous learning engine.

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

    # -- observation ---------------------------------------------------------

    def observe(self, event_type: str, attributes: dict[str, Any],
                source: str = "learner") -> Observation:
        """Record an observation and update patterns.

        Args:
            event_type: Category of the event.
            attributes: Key-value payload describing the event.
            source: Origin identifier.

        Returns:
            The created ``Observation``.
        """
        obs = Observation(event_type, attributes, source=source)
        fact = Fact(
            subject=event_type,
            relation=_OBSERVATION_RELATION,
            object=obs.attributes,
            timestamp=obs.timestamp,
            confidence=1.0,
            source=source,
        )
        self._knowledge.add_fact(fact)
        self._refine_patterns(event_type)
        return obs

    # -- pattern extraction --------------------------------------------------

    def _refine_patterns(self, event_type: str):
        observations = self._knowledge.get_facts(
            subject=event_type, relation=_OBSERVATION_RELATION
        )
        if len(observations) < 2:
            return

        numeric_attrs: dict[str, list[float]] = defaultdict(list)
        categorical_attrs: dict[str, set[Any]] = defaultdict(set)
        for obs in observations:
            obj = obs.object
            if not isinstance(obj, dict):
                continue
            for k, v in obj.items():
                if isinstance(v, (int, float)):
                    numeric_attrs[k].append(v)
                else:
                    categorical_attrs[k].add(v)

        attributes: dict[str, Any] = {}
        for k, vals in numeric_attrs.items():
            if len(vals) >= 2:
                attributes[k] = {"mean": statistics.mean(vals),
                                 "stdev": statistics.stdev(vals)}
            else:
                attributes[k] = {"mean": vals[0], "stdev": 0.0}
        for k, vals in categorical_attrs.items():
            attributes[k] = list(vals)

        confidence = min(1.0, len(observations) / 10.0)
        existing = self._knowledge.get_facts(
            subject=event_type, relation=_PATTERN_RELATION
        )
        for pat_fact in existing:
            self._knowledge.remove_fact(pat_fact)

        pattern = Pattern(event_type, attributes,
                          confidence=confidence,
                          sample_count=len(observations))
        pattern_fact = Fact(
            subject=event_type,
            relation=_PATTERN_RELATION,
            object={"attributes": pattern.attributes,
                     "confidence": pattern.confidence,
                     "sample_count": pattern.sample_count,
                     "last_updated": pattern.last_updated},
            confidence=pattern.confidence,
            source="learner",
        )
        self._knowledge.add_fact(pattern_fact)

    # -- prediction ----------------------------------------------------------

    def predict(self, event_type: str) -> Optional[Prediction]:
        """Predict attributes for *event_type* based on learned patterns.

        Args:
            event_type: The event category to predict.

        Returns:
            A ``Prediction`` if patterns exist, otherwise ``None``.
        """
        patterns = self._knowledge.get_facts(
            subject=event_type, relation=_PATTERN_RELATION
        )
        if not patterns:
            return None

        best = patterns[-1]
        obj = best.object
        if not isinstance(obj, dict):
            return None

        predicted: dict[str, Any] = obj.get("attributes", {})
        confidence: float = obj.get("confidence", 0.0)
        sample_count: int = obj.get("sample_count", 0)

        return Prediction(
            event_type=event_type,
            predicted_attributes=predicted,
            confidence=confidence,
            supporting_patterns=sample_count,
        )

    # -- insights ------------------------------------------------------------

    def insights(self) -> list[str]:
        """Generate human-readable insights from the knowledge base.

        Returns:
            A list of insight strings.
        """
        result: list[str] = []
        patterns = self._knowledge.get_facts(relation=_PATTERN_RELATION)
        for pat_fact in patterns:
            obj = pat_fact.object
            if not isinstance(obj, dict):
                continue
            event_type = pat_fact.subject
            confidence = obj.get("confidence", 0)
            sample_count = obj.get("sample_count", 0)
            attrs = obj.get("attributes", {})
            attr_summary = "; ".join(
                f"{k}: {v}" for k, v in attrs.items()
            )
            result.append(
                f"[{event_type}] confidence={confidence:.2f} "
                f"samples={sample_count} {attr_summary}"
            )
        return result

    def stats(self) -> dict[str, Any]:
        """Return summary statistics about the learner state.

        Returns:
            Dict with keys ``total_observations``, ``total_patterns``,
            ``memory_size``, ``facts_count``.
        """
        observations = self._knowledge.get_facts(
            relation=_OBSERVATION_RELATION
        )
        patterns = self._knowledge.get_facts(
            relation=_PATTERN_RELATION
        )
        return {
            "total_observations": len(observations),
            "total_patterns": len(patterns),
            "memory_size": self._memory.size(),
            "facts_count": self._knowledge.facts_count(),
        }

    def close(self):
        """Close all underlying resources."""
        self._knowledge.close()
        self._memory.close()
