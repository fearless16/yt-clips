"""Static Rich telemetry: no refresh loops, hidden errors, or fake progress."""

import logging

from rich import box
from rich.console import Group
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


_STATES = {
    "start": ("›", "RUNNING", "bright_cyan"),
    "ok": ("✓", "COMPLETE", "bright_green"),
    "failed": ("✗", "FAILED", "bright_red"),
    "partial": ("!", "PARTIAL", "yellow"),
    "review": ("!", "REVIEW", "yellow"),
    "empty": ("–", "NO OUTPUT", "yellow"),
    "skipped": ("–", "SKIPPED", "bright_black"),
    "cancelled": ("!", "CANCELLED", "yellow"),
}


def _state(status):
    # Unknown states must not inherit a successful visual treatment.
    return _STATES.get(status, ("?", "UNKNOWN", "yellow"))


def _job_panel(record):
    status = getattr(record, "status", "unknown")
    meta = getattr(record, "metadata", None) or {}
    run_id = getattr(record, "run_id", None) or "unknown"
    if status == "start":
        body = Text("YT // CLIPS", style="bold bright_cyan")
        body.append("   PIPELINE CONTROL\n", style="bold bright_magenta")
        body.append(f"RUN {run_id}  •  {meta.get('mode', 'LOCAL')}\n", style="dim")
        body.append(str(meta.get("source", "")))
        body.append("\nGreen = completed work • amber = attention • red = failure", style="dim")
        return Panel(body, border_style="bright_cyan", box=box.ROUNDED, padding=(1, 2))

    icon, label, style = _state(status)
    title = "JOB COMPLETE" if status == "ok" else {
        "empty": "NO OUTPUT", "review": "REVIEW REQUIRED",
    }.get(status, f"JOB {label}")
    heading = Text(f"{icon} {title}", style=f"bold {style}")
    heading.append(f"   {getattr(record, 'duration_ms', 0) / 1000:.1f}s  //  {run_id}", style="dim")

    counts = Table.grid(padding=(0, 3))
    counts.add_column(style="dim")
    counts.add_column()
    counts.add_row("Discovery", Text(
        f"{meta.get('candidates', 0)} candidates · {meta.get('selected', 0)} selected · "
        f"{meta.get('rejected', 0)} rejected"))
    counts.add_row("Artifacts", Text(
        f"{meta.get('exported', 0)} exports · {meta.get('seo_generated', 0)} SEO · "
        f"{meta.get('thumbnails_generated', 0)} covers + {meta.get('thumbnails_preserved', 0)} preserved"))
    counts.add_row("YouTube", Text(f"{meta.get('uploaded', 0)} uploaded"))
    counts.add_row("Transcript", Text(str(meta.get("transcript_source", "none"))))
    counts.add_row("Selection", Text(str(meta.get("selection_status", "NOT_RUN"))))
    content = [heading, Text(""), counts]

    stages = meta.get("stages", [])
    if stages:
        table = Table(box=box.SIMPLE, expand=True, header_style="bold bright_cyan", padding=(0, 1))
        table.add_column("STAGE", ratio=3)
        table.add_column("STATE", ratio=1)
        table.add_column("TIME", justify="right", ratio=1)
        for stage in stages:
            marker, state_label, color = _state(stage["status"])
            table.add_row(Text(str(stage["phase"])), Text(f"{marker} {state_label}", style=color),
                          f"{stage['duration_ms'] / 1000:.1f}s")
        content.extend([Text(""), table])

    for directory in meta.get("output_dirs", []):
        content.append(Text(f"OUTPUT  {directory}", style="bright_cyan"))
    content.append(Text(f"LOG     {meta.get('log_file', 'logs/pipeline.log')}", style="dim"))
    if meta.get("upload_requested") and meta.get("uploaded", 0) < meta.get("exported", 0):
        content.append(Text("Upload requested, but not every export has a confirmed YouTube upload.", style="yellow"))
    if meta.get("review_required") and meta.get("exported"):
        content.append(Text("Draft completion is not release approval; owner review is still required.", style="yellow"))
    elif meta.get("exported") and not meta.get("uploaded"):
        content.append(Text("Local artifacts only. No YouTube upload performed; not release approval.", style="dim"))
    for failure in meta.get("failures", []):
        content.append(Text(f"! {failure}", style="bright_red"))
    return Panel(Group(*content), border_style=style, box=box.ROUNDED, padding=(1, 2))


class PipelineConsoleHandler(RichHandler):
    """Style structured events; leave the JSON records and message text intact."""

    def render_message(self, record: logging.LogRecord, message: str):
        try:
            return self._render_event(record)
        except (TypeError, ValueError, KeyError, AttributeError):
            # A malformed optional display field must never stop the job or
            # drop its original diagnostic. JSON still receives the record.
            return Text(record.getMessage())

    def _render_event(self, record):
        stage = getattr(record, "stage", None)
        status = getattr(record, "status", None)
        if stage == "pipeline":
            return _job_panel(record)
        if stage and status:
            icon, label, style = _state(status)
            line = Text(f"{icon} {label:<9}", style=f"bold {style}")
            line.append(f" {getattr(record, 'phase', stage)}", style="bold")
            if status != "start":
                line.append(f"  {getattr(record, 'duration_ms', 0) / 1000:.1f}s", style="dim")
            metadata = getattr(record, "metadata", None) or {}
            details = [f"{key}={value}" for key, value in metadata.items()
                       if isinstance(value, (int, float, str)) and value is not None]
            if details:
                line.append("  //  " + " · ".join(details), style="dim")
            return line
        style = "bright_red" if record.levelno >= logging.ERROR else "yellow" if record.levelno >= logging.WARNING else "dim"
        line = Text(f"{record.levelname:<5} ", style=style)
        line.append(record.getMessage(), style=None if record.levelno >= logging.WARNING else "default")
        return line
