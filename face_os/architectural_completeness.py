"""Architectural Completeness Report.

Separates:
- IMPLEMENTED: code exists
- TESTED: tests pass
- MATHEMATICALLY_COMPLETE: formal definition exists
- INTEGRATED: connected to pipeline
- ACTIVE: used in production
- VALIDATED: measurably improves metrics

Purpose:
    Passing tests ≠ architectural completeness.
    This module provides a formal way to track what's actually complete.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class CompletenessLevel(Enum):
    """Level of architectural completeness."""
    IMPLEMENTED = "implemented"          # Code exists
    TESTED = "tested"                    # Tests pass
    MATHEMATICALLY_COMPLETE = "math"     # Formal definition exists
    INTEGRATED = "integrated"            # Connected to pipeline
    ACTIVE = "active"                    # Used in production
    VALIDATED = "validated"              # Measurably improves metrics


@dataclass
class ModuleCompleteness:
    """Completeness status for a single module."""

    # Module name
    name: str

    # Completeness levels achieved
    levels: List[CompletenessLevel]

    # Missing levels
    missing: List[CompletenessLevel]

    # Notes
    notes: str = ""

    @property
    def is_complete(self) -> bool:
        """Check if module is fully complete."""
        return len(self.missing) == 0

    @property
    def completeness_score(self) -> float:
        """Completeness score [0, 1]."""
        total = len(CompletenessLevel)
        achieved = len(self.levels)
        return achieved / total

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "levels": [l.value for l in self.levels],
            "missing": [l.value for l in self.missing],
            "is_complete": self.is_complete,
            "completeness_score": self.completeness_score,
            "notes": self.notes,
        }


@dataclass
class ArchitecturalCompletenessReport:
    """Report for architectural completeness."""

    # Module completeness
    modules: Dict[str, ModuleCompleteness]

    # Overall completeness
    overall_score: float

    # Critical gaps
    critical_gaps: List[str]

    # Recommendations
    recommendations: List[str]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "modules": {name: m.to_dict() for name, m in self.modules.items()},
            "overall_score": self.overall_score,
            "critical_gaps": self.critical_gaps,
            "recommendations": self.recommendations,
        }


class ArchitecturalCompletenessChecker:
    """Checks architectural completeness.

    Tracks:
    - Which modules exist
    - Which are tested
    - Which are mathematically complete
    - Which are integrated
    - Which are active
    - Which are validated
    """

    def __init__(self):
        """Initialize completeness checker."""
        self._modules: Dict[str, ModuleCompleteness] = {}

    def register_module(
        self,
        name: str,
        levels: List[CompletenessLevel],
        notes: str = "",
    ) -> None:
        """Register a module with its completeness levels.

        Args:
            name: Module name
            levels: Completeness levels achieved
            notes: Notes
        """
        all_levels = list(CompletenessLevel)
        missing = [l for l in all_levels if l not in levels]

        self._modules[name] = ModuleCompleteness(
            name=name,
            levels=levels,
            missing=missing,
            notes=notes,
        )

    def get_report(self) -> ArchitecturalCompletenessReport:
        """Get completeness report.

        Returns:
            ArchitecturalCompletenessReport
        """
        # Compute overall score
        if not self._modules:
            overall_score = 0.0
        else:
            scores = [m.completeness_score for m in self._modules.values()]
            overall_score = sum(scores) / len(scores)

        # Identify critical gaps
        critical_gaps = []
        for name, module in self._modules.items():
            if CompletenessLevel.MATHEMATICALLY_COMPLETE in module.missing:
                critical_gaps.append(f"{name}: missing mathematical definition")
            if CompletenessLevel.INTEGRATED in module.missing:
                critical_gaps.append(f"{name}: not integrated")
            if CompletenessLevel.ACTIVE in module.missing:
                critical_gaps.append(f"{name}: not active")

        # Generate recommendations
        recommendations = []
        if overall_score < 0.5:
            recommendations.append("Focus on integrating existing modules")
        if any(CompletenessLevel.MATHEMATICALLY_COMPLETE in m.missing for m in self._modules.values()):
            recommendations.append("Complete mathematical definitions")
        if any(CompletenessLevel.VALIDATED in m.missing for m in self._modules.values()):
            recommendations.append("Validate module contributions with metrics")

        return ArchitecturalCompletenessReport(
            modules=self._modules,
            overall_score=overall_score,
            critical_gaps=critical_gaps,
            recommendations=recommendations,
        )

    def get_incomplete_modules(self) -> List[str]:
        """Get list of incomplete modules.

        Returns:
            List of module names
        """
        return [name for name, module in self._modules.items() if not module.is_complete]

    def get_module_score(self, name: str) -> float:
        """Get completeness score for a module.

        Args:
            name: Module name

        Returns:
            Completeness score [0, 1]
        """
        module = self._modules.get(name)
        if module is None:
            return 0.0
        return module.completeness_score
