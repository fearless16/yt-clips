"""Tests for Architectural Completeness Report.

Target: 10 tests
"""

import numpy as np
import pytest

from face_os.architectural_completeness import (
    CompletenessLevel,
    ModuleCompleteness,
    ArchitecturalCompletenessReport,
    ArchitecturalCompletenessChecker,
)


class TestCompletenessLevel:
    """Test CompletenessLevel enum."""

    def test_levels_exist(self):
        """All levels must exist."""
        assert CompletenessLevel.IMPLEMENTED
        assert CompletenessLevel.TESTED
        assert CompletenessLevel.MATHEMATICALLY_COMPLETE
        assert CompletenessLevel.INTEGRATED
        assert CompletenessLevel.ACTIVE
        assert CompletenessLevel.VALIDATED


class TestModuleCompleteness:
    """Test ModuleCompleteness."""

    def test_is_complete(self):
        """Must check if complete."""
        module = ModuleCompleteness(
            name="test",
            levels=list(CompletenessLevel),
            missing=[],
        )
        assert module.is_complete

    def test_is_not_complete(self):
        """Must check if not complete."""
        module = ModuleCompleteness(
            name="test",
            levels=[CompletenessLevel.IMPLEMENTED],
            missing=[CompletenessLevel.TESTED],
        )
        assert not module.is_complete

    def test_completeness_score(self):
        """Must compute score."""
        module = ModuleCompleteness(
            name="test",
            levels=[CompletenessLevel.IMPLEMENTED, CompletenessLevel.TESTED],
            missing=[l for l in CompletenessLevel if l not in [CompletenessLevel.IMPLEMENTED, CompletenessLevel.TESTED]],
        )
        score = module.completeness_score
        assert 0.0 <= score <= 1.0

    def test_to_dict(self):
        """Must convert to dict."""
        module = ModuleCompleteness(
            name="test",
            levels=[CompletenessLevel.IMPLEMENTED],
            missing=[CompletenessLevel.TESTED],
        )
        d = module.to_dict()
        assert d["name"] == "test"
        assert "implemented" in d["levels"]


class TestArchitecturalCompletenessChecker:
    """Test ArchitecturalCompletenessChecker."""

    def test_register_module(self):
        """Must register module."""
        checker = ArchitecturalCompletenessChecker()
        checker.register_module("test", [CompletenessLevel.IMPLEMENTED])
        assert "test" in checker._modules

    def test_get_report(self):
        """Must generate report."""
        checker = ArchitecturalCompletenessChecker()
        checker.register_module("test", [CompletenessLevel.IMPLEMENTED])
        report = checker.get_report()
        assert report.overall_score > 0

    def test_get_incomplete_modules(self):
        """Must identify incomplete modules."""
        checker = ArchitecturalCompletenessChecker()
        checker.register_module("complete", list(CompletenessLevel))
        checker.register_module("incomplete", [CompletenessLevel.IMPLEMENTED])

        incomplete = checker.get_incomplete_modules()
        assert "incomplete" in incomplete
        assert "complete" not in incomplete

    def test_get_module_score(self):
        """Must get module score."""
        checker = ArchitecturalCompletenessChecker()
        checker.register_module("test", [CompletenessLevel.IMPLEMENTED, CompletenessLevel.TESTED])
        score = checker.get_module_score("test")
        assert score > 0

    def test_report_to_dict(self):
        """Report must convert to dict."""
        checker = ArchitecturalCompletenessChecker()
        checker.register_module("test", [CompletenessLevel.IMPLEMENTED])
        report = checker.get_report()
        d = report.to_dict()
        assert "modules" in d
        assert "overall_score" in d

    def test_critical_gaps(self):
        """Must identify critical gaps."""
        checker = ArchitecturalCompletenessChecker()
        checker.register_module("test", [CompletenessLevel.IMPLEMENTED])
        report = checker.get_report()
        assert len(report.critical_gaps) > 0
