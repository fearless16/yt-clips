"""LearningDispatcher — routes events to specialized learners with idempotency.

Replaces the monolithic PolicyUpdater. One router, five specialist learners.
Checks ProcessedEventLog before processing to prevent double-learning.
"""

import json
from automation.memory.event_models import ClipEvent, EventType


class LearningDispatcher:
    """Routes events to specialized learners with idempotency."""

    LEARNERS = ["format", "entity", "timing", "duration", "trend"]

    def __init__(self, format_learner, entity_learner, timing_learner,
                 duration_learner, trend_engine, processed_log):
        """Initialize LearningDispatcher.

        Args:
            format_learner: FormatLearner instance
            entity_learner: EntityLearner instance
            timing_learner: TimingLearner instance
            duration_learner: DurationLearner instance
            trend_engine: TrendEngine instance
            processed_log: ProcessedEventLog instance
        """
        self._format = format_learner
        self._entity = entity_learner
        self._timing = timing_learner
        self._duration = duration_learner
        self._trend = trend_engine
        self._log = processed_log

    def dispatch(self, event: ClipEvent) -> int:
        """Process event through all relevant learners.

        Args:
            event: ClipEvent to process

        Returns:
            Count of learners updated
        """
        updated = 0
        payload = json.loads(event.payload_json)

        if event.event_type == EventType.metrics_received:
            for learner_name, learner in [
                ("format", self._format),
                ("entity", self._entity),
                ("timing", self._timing),
                ("duration", self._duration),
            ]:
                if not self._log.is_processed(event.event_id, learner_name):
                    learner.process_metrics(event.clip_id, payload)
                    self._log.mark_processed(event.event_id, learner_name)
                    updated += 1

        elif event.event_type == EventType.trend_ingested:
            if not self._log.is_processed(event.event_id, "trend"):
                from automation.learner.trend_engine import TrendInput
                self._trend.ingest(TrendInput.from_payload(payload))
                self._log.mark_processed(event.event_id, "trend")
                updated += 1

        elif event.event_type == EventType.manual_override:
            for learner_name, learner in [
                ("format", self._format),
                ("entity", self._entity),
            ]:
                if not self._log.is_processed(event.event_id, learner_name):
                    learner.process_override(event.clip_id, payload)
                    self._log.mark_processed(event.event_id, learner_name)
                    updated += 1

        return updated

    def dispatch_batch(self, events: list[ClipEvent]) -> int:
        """Process a batch of events.

        Args:
            events: List of ClipEvents to process

        Returns:
            Total count of learners updated
        """
        total = 0
        for event in events:
            total += self.dispatch(event)
        return total
