import json
from pathlib import Path
from typing import Dict, List
import time

def generate_run_report(
    export_dir: str,
    pipeline_duration_sec: float,
    exported_clips: List[Path],
    uploaded_clips: int = 0,
    failures: List[Dict[str, str]] = None
) -> str:
    if failures is None:
        failures = []

    export_dir_path = Path(export_dir)
    report_path = export_dir_path / "run_report.md"

    total_clips = len(exported_clips)
    total_size_mb = sum(p.stat().st_size for p in exported_clips) / 1_048_576 if exported_clips else 0

    seo_data = []
    for clip_path in exported_clips:
        meta_path = clip_path.with_name(f"{clip_path.stem}_metadata.json")
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    seo_data.append((clip_path.name, json.load(f)))
            except Exception:
                pass

    lines = [
        "# YouTube Automation Run Report",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Pipeline Duration:** {pipeline_duration_sec:.1f} seconds",
        f"**Status:** {'⚠️ Completed with Errors' if failures else '✅ Success'}",
        "---",
        "## 📈 Performance Snapshot",
        f"- **Total Clips Generated:** {total_clips}",
        f"- **Total Export Size:** {total_size_mb:.1f} MB",
        f"- **Processing Speed:** {(pipeline_duration_sec / max(1, total_clips)):.1f} sec / clip (avg)",
        f"- **YouTube Uploads:** {uploaded_clips} / {total_clips}",
        "---",
        "## 🎬 Export Report",
    ]

    if not exported_clips:
        lines.append("*No clips were successfully exported in this run.*")
    else:
        for p in exported_clips:
            lines.append(f"- `{p.name}` ({p.stat().st_size / 1_048_576:.1f} MB)")

    lines.append("---")
    lines.append("## 🔍 SEO & Metadata Report")
    if not seo_data:
        lines.append("*No SEO metadata generated.*")
    else:
        for name, meta in seo_data:
            lines.append(f"### {name}")
            lines.append(f"- **Title:** {meta.get('title', 'N/A')}")
            lines.append(f"- **Description:** {(meta.get('description', 'N/A') or 'N/A')[:180]}...")
            lines.append(f"- **Hashtags:** {', '.join(meta.get('hashtags', [])) if meta.get('hashtags') else 'N/A'}")
            lines.append(f"- **Search Terms:** {', '.join(meta.get('search_terms', [])) if meta.get('search_terms') else 'N/A'}")
            lines.append(f"- **Trend Topics:** {', '.join(meta.get('trend_topics', [])) if meta.get('trend_topics') else 'N/A'}")
            lines.append("")

    if failures:
        lines.append("---")
        lines.append("## ❌ Failure Summary")
        for f in failures:
            lines.append(
                f"- **{f.get('phase', 'Unknown')}:** {f.get('error', 'Unknown Error')} "
                f"(Clip: {f.get('clip_id', 'N/A')})"
            )

    report_content = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    return str(report_path)
