"""trends.py — Time-series trend analysis.

Tracks metrics over time and detects trends (improving, degrading, stable).
Uses simple linear regression for trend detection.
"""

import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from self_learner.memory import PersistentMemory
from self_learner.knowledge import Fact, KnowledgeBase


@dataclass
class TrendPoint:
    """A single data point in a time series."""
    metric: str
    value: float
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrendResult:
    """Result of trend analysis."""
    metric: str
    direction: str  # "improving", "degrading", "stable"
    slope: float  # Rate of change per unit time
    confidence: float  # 0-1, how confident we are in the trend
    data_points: int
    time_span_hours: float
    current_value: float
    predicted_next: float


class TrendAnalyzer:
    """Analyze trends in pipeline metrics over time.

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

    def record_metric(self, point: TrendPoint) -> None:
        """Record a metric data point.

        Args:
            point: The trend data point to record.
        """
        fact = Fact(
            subject=f"trend:{point.metric}",
            relation="has_data_point",
            object={
                "value": point.value,
                "timestamp": point.timestamp,
                "metadata": point.metadata,
            },
            confidence=1.0,
            source="trend_analyzer",
        )
        self._knowledge.add_fact(fact)

    def analyze_trend(self, metric: str,
                      window_hours: float = 168.0) -> Optional[TrendResult]:
        """Analyze trend for a specific metric.

        Args:
            metric: The metric to analyze.
            window_hours: Time window to consider (default: 1 week).

        Returns:
            TrendResult if enough data points exist, None otherwise.
        """
        facts = self._knowledge.get_facts(
            subject=f"trend:{metric}", relation="has_data_point"
        )

        if not facts:
            return None

        # Filter to window and sort by time
        now = time.time()
        cutoff = now - (window_hours * 3600)
        points = []
        for fact in facts:
            obj = fact.object
            if not isinstance(obj, dict):
                continue
            ts = obj.get("timestamp", 0)
            if ts >= cutoff:
                points.append((ts, obj.get("value", 0.0)))

        if len(points) < 2:
            return None

        points.sort(key=lambda x: x[0])
        timestamps = [p[0] for p in points]
        values = [p[1] for p in points]

        # Normalize timestamps to hours from start
        t0 = timestamps[0]
        hours = [(t - t0) / 3600.0 for t in timestamps]

        # Simple linear regression: y = mx + b
        n = len(hours)
        sum_x = sum(hours)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(hours, values))
        sum_x2 = sum(x * x for x in hours)

        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            slope = 0.0
        else:
            slope = (n * sum_xy - sum_x * sum_y) / denominator

        # Calculate R-squared for confidence
        if n > 2:
            mean_y = sum_y / n
            ss_tot = sum((y - mean_y) ** 2 for y in values)
            intercept = (sum_y - slope * sum_x) / n
            ss_res = sum((y - (slope * x + intercept)) ** 2
                        for x, y in zip(hours, values))
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            confidence = max(0.0, min(1.0, r_squared))
        else:
            confidence = 0.5

        # Determine direction
        # For metrics where lower is better (duration, failures)
        lower_is_better = metric in ("duration", "failures", "error_rate")
        if abs(slope) < 0.01:  # Essentially flat
            direction = "stable"
        elif (slope > 0 and not lower_is_better) or (slope < 0 and lower_is_better):
            direction = "improving"
        else:
            direction = "degrading"

        # Predict next value (1 hour ahead)
        last_hour = hours[-1]
        predicted_next = slope * (last_hour + 1) + (
            (sum_y - slope * sum_x) / n
        )

        return TrendResult(
            metric=metric,
            direction=direction,
            slope=slope,
            confidence=confidence,
            data_points=n,
            time_span_hours=hours[-1] - hours[0],
            current_value=values[-1],
            predicted_next=predicted_next,
        )

    def get_metric_summary(self, metric: str,
                           window_hours: float = 168.0) -> dict[str, Any]:
        """Get summary statistics for a metric.

        Args:
            metric: The metric to summarize.
            window_hours: Time window to consider.

        Returns:
            Dict with min, max, mean, stdev, trend direction.
        """
        facts = self._knowledge.get_facts(
            subject=f"trend:{metric}", relation="has_data_point"
        )

        if not facts:
            return {"metric": metric, "data_points": 0}

        now = time.time()
        cutoff = now - (window_hours * 3600)
        values = []
        for fact in facts:
            obj = fact.object
            if not isinstance(obj, dict):
                continue
            ts = obj.get("timestamp", 0)
            if ts >= cutoff:
                values.append(obj.get("value", 0.0))

        if not values:
            return {"metric": metric, "data_points": 0}

        trend = self.analyze_trend(metric, window_hours)

        return {
            "metric": metric,
            "data_points": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "current": values[-1],
            "trend_direction": trend.direction if trend else "unknown",
            "trend_confidence": trend.confidence if trend else 0.0,
        }

    def get_all_trends(self, window_hours: float = 168.0) -> list[TrendResult]:
        """Analyze trends for all tracked metrics.

        Args:
            window_hours: Time window to consider.

        Returns:
            List of TrendResult for each metric.
        """
        # Find all unique metrics
        facts = self._knowledge.get_facts(relation="has_data_point")
        metrics = set()
        for fact in facts:
            if fact.subject.startswith("trend:"):
                metrics.add(fact.subject[6:])  # Remove "trend:" prefix

        results = []
        for metric in metrics:
            trend = self.analyze_trend(metric, window_hours)
            if trend:
                results.append(trend)

        return results

    def detect_anomalies(self, metric: str,
                         window_hours: float = 168.0,
                         threshold_stdev: float = 2.0) -> list[dict[str, Any]]:
        """Detect anomalous values in a metric.

        Args:
            metric: The metric to check.
            window_hours: Time window to consider.
            threshold_stdev: Number of standard deviations to consider anomalous.

        Returns:
            List of dicts with value, timestamp, z_score.
        """
        facts = self._knowledge.get_facts(
            subject=f"trend:{metric}", relation="has_data_point"
        )

        if not facts:
            return []

        now = time.time()
        cutoff = now - (window_hours * 3600)
        points = []
        for fact in facts:
            obj = fact.object
            if not isinstance(obj, dict):
                continue
            ts = obj.get("timestamp", 0)
            if ts >= cutoff:
                points.append((ts, obj.get("value", 0.0)))

        if len(points) < 3:
            return []

        values = [p[1] for p in points]
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0

        if stdev == 0:
            return []

        anomalies = []
        for ts, value in points:
            z_score = abs(value - mean) / stdev
            if z_score > threshold_stdev:
                anomalies.append({
                    "value": value,
                    "timestamp": ts,
                    "z_score": z_score,
                    "deviation": "high" if value > mean else "low",
                })

        return anomalies

    def close(self):
        """Close underlying resources."""
        self._memory.close()
