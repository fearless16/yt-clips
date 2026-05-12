"""
reports.py — COLORFUL STATE-OF-THE-ART PIPELINE REPORT.
Outputs a Rich terminal dashboard. No file reading needed — just look.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.layout import Layout
from rich.columns import Columns

console = Console()
_report_file: Optional[str] = None


def generate_run_report(
    export_dir: str,
    pipeline_duration_sec: float,
    exported_clips: List[Path],
    uploaded_clips: int = 0,
    failures: List[Dict[str, str]] = None,
) -> str:
    global _report_file
    if failures is None:
        failures = []

    export_dir_path = Path(export_dir)
    _report_file = str(export_dir_path / "run_report.md")

    total_clips = len(exported_clips)
    total_size_mb = sum(p.stat().st_size for p in exported_clips if p.exists()) / 1_048_576 if exported_clips else 0
    avg_clip_mb = total_size_mb / max(1, total_clips)
    speed = pipeline_duration_sec / max(1, total_clips)

    # ─── HEADER ─────────────────────────────────────────────────────────────
    status = "SUCCESS" if not failures else "COMPLETED WITH ERRORS"
    status_color = "green" if not failures else "yellow"
    header = Panel(
        Text(f"PIPELINE COMPLETE — {status}", style=f"bold {status_color}", justify="center"),
        border_style=status_color,
        width=70,
    )
    console.print()
    console.print(header)

    # ─── METRICS TABLE ──────────────────────────────────────────────────────
    metrics = Table(title="📊 Pipeline Metrics", box=box.ROUNDED, header_style="bold cyan")
    metrics.add_column("Metric", style="cyan")
    metrics.add_column("Value", justify="right", style="white")
    metrics.add_row("Total Duration", f"{pipeline_duration_sec:.1f}s ({pipeline_duration_sec/60:.1f} min)")
    metrics.add_row("Shorts Generated", f"[bold green]{total_clips}[/]")
    metrics.add_row("Total Size", f"{total_size_mb:.1f} MB")
    metrics.add_row("Avg Size/Clip", f"{avg_clip_mb:.1f} MB")
    metrics.add_row("Speed", f"{speed:.1f}s per clip")
    metrics.add_row("Uploaded", f"[{'green' if uploaded_clips else 'yellow'}]{uploaded_clips}/{total_clips}[/]")
    upload_pct = (uploaded_clips / max(1, total_clips)) * 100
    metrics.add_row("Upload Rate", f"{upload_pct:.0f}%")
    console.print(metrics)

    # ─── CLIP TABLE ────────────────────────────────────────────────────────
    if exported_clips:
        clip_table = Table(title="🎬 Exported Shorts", box=box.SIMPLE, header_style="bold magenta")
        clip_table.add_column("#", justify="right", style="dim")
        clip_table.add_column("Filename", style="cyan")
        clip_table.add_column("Size", justify="right")
        clip_table.add_column("Title", style="white", max_width=40)

        for i, p in enumerate(exported_clips, 1):
            size = p.stat().st_size / 1_048_576 if p.exists() else 0
            size_str = f"{size:.1f} MB" if p.exists() else "[red]MISSING[/]"
            size_color = "green" if size < 10 else "yellow" if size < 20 else "red"

            title = "N/A"
            meta_path = p.with_name(f"{p.stem}_metadata.json")
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    title = meta.get("title", "N/A")[:40]
                except Exception:
                    pass

            clip_table.add_row(
                str(i),
                p.name,
                f"[{size_color}]{size_str}[/]",
                title,
            )

        console.print()
        console.print(clip_table)

    # ─── FAILURES ───────────────────────────────────────────────────────────
    if failures:
        fail_table = Table(title="❌ Failures", box=box.ROUNDED, header_style="bold red")
        fail_table.add_column("Phase", style="red")
        fail_table.add_column("Error", style="white")
        fail_table.add_column("Clip", style="dim")
        for f in failures:
            fail_table.add_row(
                f.get("phase", "Unknown"),
                f.get("error", "Unknown"),
                f.get("clip_id", "-"),
            )
        console.print()
        console.print(fail_table)

    # ─── SAVE to file (markdown) and return
    md_lines = [
        "# Pipeline Run Report",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Duration:** {pipeline_duration_sec:.1f}s",
        f"**Status:** {status}",
        "",
        "## Metrics",
        f"- Clips: {total_clips}",
        f"- Size: {total_size_mb:.1f} MB",
        f"- Uploaded: {uploaded_clips}/{total_clips}",
        "",
        "## Clips",
    ]
    for p in exported_clips:
        sz = p.stat().st_size / 1_048_576 if p.exists() else 0
        md_lines.append(f"- {p.name} ({sz:.1f} MB)")
    if failures:
        md_lines.append("\n## Failures")
        for f in failures:
            md_lines.append(f"- {f.get('phase')}: {f.get('error')}")

    Path(_report_file).write_text("\n".join(md_lines), encoding="utf-8")
    console.print(f"\n[dim]📄 Report saved: {_report_file}[/]")
    return _report_file
