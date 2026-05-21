"""Energy Scaling Module.

Defines energy normalization and adaptive weighting:
    E = sum_i lambda_i * E_hat_i

where:
    E_hat_i = normalized energy term
    lambda_i = adaptive weight

This module provides:
- Energy normalization (z-score, min-max)
- Adaptive weighting (uncertainty-weighted)
- Energy scaling strategy
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class EnergyScalingConfig:
    """Configuration for energy scaling."""

    # Normalization method: 'zscore', 'minmax', 'none'
    normalization_method: str = 'zscore'

    # Smoothing factor for running statistics
    smoothing_factor: float = 0.01

    # Minimum weight for any term
    min_weight: float = 0.01

    # Maximum weight for any term
    max_weight: float = 10.0

    # Uncertainty scaling factor
    uncertainty_scale: float = 1.0


@dataclass
class EnergyTermStats:
    """Running statistics for an energy term."""

    # Term name
    name: str

    # Running mean
    mean: float = 0.0

    # Running variance
    variance: float = 1.0

    # Running min
    min_val: float = float('inf')

    # Running max
    max_val: float = float('-inf')

    # Sample count
    count: int = 0

    def update(self, value: float) -> None:
        """Update statistics with new value.

        Args:
            value: New energy value
        """
        self.count += 1

        # Update mean (exponential moving average)
        alpha = 1.0 / max(self.count, 100)
        self.mean = (1 - alpha) * self.mean + alpha * value

        # Update variance
        diff = value - self.mean
        self.variance = (1 - alpha) * self.variance + alpha * diff**2

        # Update min/max
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)


class EnergyScaler:
    """Energy normalization and adaptive weighting.

    Provides:
    - Z-score normalization: E_hat = (E - mean) / std
    - Min-max normalization: E_hat = (E - min) / (max - min)
    - Adaptive weighting: lambda = f(uncertainty)
    """

    def __init__(self, config: Optional[EnergyScalingConfig] = None):
        """Initialize energy scaler.

        Args:
            config: Configuration
        """
        self.config = config or EnergyScalingConfig()
        self._stats: Dict[str, EnergyTermStats] = {}

    def normalize(
        self,
        term_name: str,
        value: float,
    ) -> float:
        """Normalize an energy term value.

        Args:
            term_name: Name of energy term
            value: Raw energy value

        Returns:
            Normalized energy value
        """
        # Get or create stats
        if term_name not in self._stats:
            self._stats[term_name] = EnergyTermStats(name=term_name)
        stats = self._stats[term_name]

        # Update statistics
        stats.update(value)

        # Normalize based on method
        if self.config.normalization_method == 'zscore':
            # Z-score: (x - mean) / std
            std = np.sqrt(stats.variance) + 1e-8
            normalized = (value - stats.mean) / std
        elif self.config.normalization_method == 'minmax':
            # Min-max: (x - min) / (max - min)
            range_val = stats.max_val - stats.min_val + 1e-8
            normalized = (value - stats.min_val) / range_val
        else:
            # No normalization
            normalized = value

        return float(normalized)

    def compute_weight(
        self,
        term_name: str,
        uncertainty: float = 1.0,
    ) -> float:
        """Compute adaptive weight for energy term.

        Weight = 1 / uncertainty (higher uncertainty = lower weight)

        Args:
            term_name: Name of energy term
            uncertainty: Uncertainty of term [0, inf)

        Returns:
            Adaptive weight
        """
        # Base weight from uncertainty
        weight = 1.0 / (uncertainty + 1e-8)

        # Scale by config
        weight *= self.config.uncertainty_scale

        # Clamp to valid range
        weight = np.clip(weight, self.config.min_weight, self.config.max_weight)

        return float(weight)

    def compute_scaled_energy(
        self,
        terms: Dict[str, float],
        uncertainties: Optional[Dict[str, float]] = None,
    ) -> float:
        """Compute scaled energy: E = sum_i lambda_i * E_hat_i

        Args:
            terms: Dict of term_name -> raw_value
            uncertainties: Dict of term_name -> uncertainty (optional)

        Returns:
            Scaled energy value
        """
        if uncertainties is None:
            uncertainties = {name: 1.0 for name in terms}

        total_energy = 0.0
        for name, value in terms.items():
            # Normalize term
            normalized = self.normalize(name, value)

            # Compute weight
            uncertainty = uncertainties.get(name, 1.0)
            weight = self.compute_weight(name, uncertainty)

            # Add weighted term
            total_energy += weight * normalized

        return float(total_energy)

    def get_stats(self) -> Dict[str, dict]:
        """Get statistics for all terms.

        Returns:
            Dict of term_name -> stats dict
        """
        result = {}
        for name, stats in self._stats.items():
            result[name] = {
                "mean": stats.mean,
                "variance": stats.variance,
                "std": np.sqrt(stats.variance),
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
        """Convert to dictionary."""
        return {
            "raw_terms": self.raw_terms,
            "normalized_terms": self.normalized_terms,
            "weights": self.weights,
            "weighted_terms": self.weighted_terms,
            "total_energy": self.total_energy,
        }
