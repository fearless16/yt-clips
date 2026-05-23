"""Energy Scaling Module.

BEAST MODE FIXES:
- Hooked up the dead `smoothing_factor` config.
- Fixed the fake variance math with proper EMA Welford's algorithm.
- Added a variance floor to prevent Z-score energy nukes in early frames.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class EnergyScalingConfig:
    """Configuration for energy scaling."""
    normalization_method: str = 'zscore'
    smoothing_factor: float = 0.01
    min_weight: float = 0.01
    max_weight: float = 10.0
    uncertainty_scale: float = 1.0


@dataclass
class EnergyTermStats:
    """Running statistics for an energy term."""
    name: str
    mean: float = 0.0
    variance: float = 1.0
    min_val: float = float('inf')
    max_val: float = float('-inf')
    count: int = 0

    def update(self, value: float, alpha: Optional[float] = None) -> None:
        """Update statistics with new value using proper EMA math."""
        self.count += 1
        
        # Use provided alpha (from config) or fallback to 1/count for warmup
        a = alpha if alpha is not None else (1.0 / max(self.count, 100))
        
        old_mean = self.mean
        # EMA mean
        self.mean = (1 - a) * self.mean + a * value
        
        # BEAST MODE FIX: Welford's online algorithm adapted for EMA.
        # Prevents variance collapse and underestimation.
        diff = value - old_mean
        self.variance = (1 - a) * self.variance + a * diff * (value - self.mean)

        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)


class EnergyScaler:
    """Energy normalization and adaptive weighting."""

    def __init__(self, config: Optional[EnergyScalingConfig] = None):
        self.config = config or EnergyScalingConfig()
        self._stats: Dict[str, EnergyTermStats] = {}

    def normalize(self, term_name: str, value: float) -> float:
        """Normalize an energy term value."""
        if term_name not in self._stats:
            self._stats[term_name] = EnergyTermStats(name=term_name)
        stats = self._stats[term_name]

        # Pass the config smoothing factor to the update function
        alpha = self.config.smoothing_factor if self.config.smoothing_factor > 0 else None
        stats.update(value, alpha=alpha)

        if self.config.normalization_method == 'zscore':
            # BEAST MODE FIX: Variance floor (1e-4) prevents division by near-zero 
            # which nukes the energy to 100M in the first 10 frames.
            std = np.sqrt(max(stats.variance, 1e-4))
            normalized = (value - stats.mean) / std
        elif self.config.normalization_method == 'minmax':
            range_val = stats.max_val - stats.min_val + 1e-8
            normalized = (value - stats.min_val) / range_val
        else:
            normalized = value

        return float(normalized)

    def compute_weight(self, term_name: str, uncertainty: float = 1.0) -> float:
        """Compute adaptive weight for energy term."""
        weight = 1.0 / (uncertainty + 1e-8)
        weight *= self.config.uncertainty_scale
        weight = np.clip(weight, self.config.min_weight, self.config.max_weight)
        return float(weight)

    def compute_scaled_energy(
        self,
        terms: Dict[str, float],
        uncertainties: Optional[Dict[str, float]] = None,
    ) -> float:
        """Compute scaled energy: E = sum_i lambda_i * E_hat_i"""
        if uncertainties is None:
            uncertainties = {name: 1.0 for name in terms}

        total_energy = 0.0
        for name, value in terms.items():
            normalized = self.normalize(name, value)
            uncertainty = uncertainties.get(name, 1.0)
            weight = self.compute_weight(name, uncertainty)
            total_energy += weight * normalized

        return float(total_energy)

    def get_stats(self) -> Dict[str, dict]:
        """Get statistics for all terms."""
        result = {}
        for name, stats in self._stats.items():
            result[name] = {
                "mean": stats.mean,
                "variance": stats.variance,
                "std": np.sqrt(max(stats.variance, 0.0)),
                "min": stats.min_val,
                "max": stats.max_val,
                "count": stats.count,
            }
        return result

    def reset(self) -> None:
        """Reset all statistics."""
        self._stats.clear()


@dataclass
class EnergyScalingReport:
    """Report for energy scaling metrics."""
    raw_terms: Dict[str, float]
    normalized_terms: Dict[str, float]
    weights: Dict[str, float]
    weighted_terms: Dict[str, float]
    total_energy: float

    def to_dict(self) -> dict:
        return {
            "raw_terms": self.raw_terms,
            "normalized_terms": self.normalized_terms,
            "weights": self.weights,
            "weighted_terms": self.weighted_terms,
            "total_energy": self.total_energy,
        }