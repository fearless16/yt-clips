"""
visibility.py — Parameter-wise Visibility Logger for Face OS.

Phase 0: Contract Lockdown — Mandatory logging format.

Every pass must emit a JSON object with at least:
- pass_id
- frame_id
- before
- after
- delta
- metrics
- status

If this visibility is missing, the change must be rejected.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from face_os.types import (
    PassReport,
    EnergyReport,
    FrameContract,
    RendererReport,
)


class VisibilityLogger:
    """Logs before/after/delta metrics for every pass.

    This is the MANDATORY visibility format for every change.
    If this report is missing, the change must be rejected.
    """

    def __init__(self, output_dir: str = "output/face_os/visibility"):
        self.output_dir = output_dir
        self.reports: List[PassReport] = []
        self.energy_reports: List[EnergyReport] = []
        self.renderer_reports: List[RendererReport] = []
        os.makedirs(output_dir, exist_ok=True)

    def log_pass(
        self,
        pass_id: str,
        frame_id: int,
        before: Dict[str, float],
        after: Dict[str, float],
        metrics: Dict[str, Any],
        status: str = "accepted",
    ) -> PassReport:
        """Log a pass with before/after/delta metrics.

        Args:
            pass_id: Phase and operation identifier (e.g. "phase2_transform_hardening")
            frame_id: Frame index
            before: Metrics before the change
            after: Metrics after the change
            metrics: Full metrics snapshot
            status: accepted / rejected / skipped

        Returns:
            PassReport with computed delta
        """
        report = PassReport(
            pass_id=pass_id,
            frame_id=frame_id,
            before=before,
            after=after,
            metrics=metrics,
            status=status,
        )
        report.compute_delta()
        self.reports.append(report)
        return report

    def log_energy(self, report: EnergyReport) -> None:
        """Log an energy report for a frame."""
        self.energy_reports.append(report)

    def log_renderer(self, report: RendererReport) -> None:
        """Log a renderer report for a frame."""
        self.renderer_reports.append(report)

    def save_all(self, prefix: str = "") -> str:
        """Save all reports to JSON files.

        Args:
            prefix: Optional prefix for filenames

        Returns:
            Path to the main report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix_str = f"{prefix}_" if prefix else ""

        # Save pass reports
        if self.reports:
            pass_file = os.path.join(
                self.output_dir, f"{prefix_str}pass_reports_{timestamp}.json"
            )
            with open(pass_file, "w") as f:
                json.dump(
                    [r.to_dict() for r in self.reports],
                    f, indent=2, default=str,
                )

        # Save energy reports
        if self.energy_reports:
            energy_file = os.path.join(
                self.output_dir, f"{prefix_str}energy_reports_{timestamp}.json"
            )
            with open(energy_file, "w") as f:
                json.dump(
                    [r.to_dict() for r in self.energy_reports],
                    f, indent=2, default=str,
                )

        # Save renderer reports
        if self.renderer_reports:
            renderer_file = os.path.join(
                self.output_dir, f"{prefix_str}renderer_reports_{timestamp}.json"
            )
            with open(renderer_file, "w") as f:
                json.dump(
                    [r.to_dict() for r in self.renderer_reports],
                    f, indent=2, default=str,
                )

        # Save summary
        summary = {
            "timestamp": timestamp,
            "total_passes": len(self.reports),
            "total_energy_reports": len(self.energy_reports),
            "total_renderer_reports": len(self.renderer_reports),
            "pass_summary": self._summarize_passes(),
            "energy_summary": self._summarize_energy(),
        }

        summary_file = os.path.join(
            self.output_dir, f"{prefix_str}summary_{timestamp}.json"
        )
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        return summary_file

    def _summarize_passes(self) -> Dict[str, Any]:
        """Summarize pass reports."""
        if not self.reports:
            return {}

        accepted = sum(1 for r in self.reports if r.status == "accepted")
        rejected = sum(1 for r in self.reports if r.status == "rejected")

        # Collect all delta keys
        delta_keys = set()
        for r in self.reports:
            delta_keys.update(r.delta.keys())

        # Compute mean deltas
        mean_deltas = {}
        for key in delta_keys:
            values = [r.delta[key] for r in self.reports if key in r.delta]
            if values:
                mean_deltas[key] = sum(values) / len(values)

        return {
            "accepted": accepted,
            "rejected": rejected,
            "mean_deltas": mean_deltas,
        }

    def _summarize_energy(self) -> Dict[str, Any]:
        """Summarize energy reports."""
        if not self.energy_reports:
            return {}

        # Compute mean energy terms
        E_geom = [r.terms.E_geom for r in self.energy_reports]
        E_identity = [r.terms.E_identity for r in self.energy_reports]
        E_temporal = [r.terms.E_temporal for r in self.energy_reports]
        E_photometric = [r.terms.E_photometric for r in self.energy_reports]
        E_smoothness = [r.terms.E_smoothness for r in self.energy_reports]
        E_total = [r.terms.E_total for r in self.energy_reports]

        return {
            "mean_E_geom": sum(E_geom) / len(E_geom) if E_geom else 0,
            "mean_E_identity": sum(E_identity) / len(E_identity) if E_identity else 0,
            "mean_E_temporal": sum(E_temporal) / len(E_temporal) if E_temporal else 0,
            "mean_E_photometric": sum(E_photometric) / len(E_photometric) if E_photometric else 0,
            "mean_E_smoothness": sum(E_smoothness) / len(E_smoothness) if E_smoothness else 0,
            "mean_E_total": sum(E_total) / len(E_total) if E_total else 0,
            "min_E_total": min(E_total) if E_total else 0,
            "max_E_total": max(E_total) if E_total else 0,
        }


def validate_pass_report(report: PassReport) -> bool:
    """Validate that a pass report has all required fields.

    Args:
        report: PassReport to validate

    Returns:
        True if all required fields are present
    """
    required = ["pass_id", "frame_id", "status"]
    for field in required:
        if not getattr(report, field, None):
            return False

    # Must have either before/after or metrics
    if not report.before and not report.after and not report.metrics:
        return False

    return True
