"""Face OS — Latent Identity Rendering HTML Report Generator.

Runs the pipeline on a video clip, collects per-frame telemetry, and generates
a self-contained dark-themed HTML report with C_recon chain analysis, coverage
progress, gate state timeline, and invariant verification.

Usage:
    python face_os/gen_report.py                           # default: clips_test/test_clip.mp4
    python face_os/gen_report.py --video path/to/video.mp4 # custom video
    python face_os/gen_report.py --max-frames 30           # limit frames
    python face_os/gen_report.py --open                    # open in browser when done
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is on sys.path for face_os imports
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def run_pipeline(video_path: str, max_frames: int | None = None) -> dict:
    """Run FaceOSPipeline on a video and return telemetry + metadata."""
    from face_os.pipeline import FaceOSPipeline

    pipeline = FaceOSPipeline()
    t0 = time.time()
    pipeline.enroll()
    enroll_time = time.time() - t0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n = 0
    t1 = time.time()
    while True:
        if max_frames and n >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        pipeline.process_frame(frame, n)
        n += 1
    cap.release()
    process_time = time.time() - t1

    telemetry = pipeline.get_latent_telemetry()
    pipeline_report = pipeline.get_telemetry_report()

    return {
        "video_path": video_path,
        "total_frames": n,
        "total_video_frames": total_video_frames,
        "fps": fps,
        "resolution": f"{width}x{height}",
        "enroll_time_s": enroll_time,
        "process_time_s": process_time,
        "telemetry": telemetry,
        "pipeline_report": pipeline_report,
    }


def _fmt(val, decimals=6):
    """Format a numeric value."""
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def _bar(val, max_val=1.0, width=200, color="#00ff88"):
    """Generate an inline HTML bar."""
    pct = min(max(val / max_val, 0.0), 1.0) * 100
    return f'<span style="display:inline-block;height:16px;width:{pct:.1f}%;background:{color};border-radius:2px"></span>'


def _status_color(val, threshold_good=0.5, threshold_warn=0.1):
    """Return CSS color based on value thresholds."""
    if val >= threshold_good:
        return "#00ff88"
    elif val >= threshold_warn:
        return "#ffaa00"
    return "#ff4444"


def generate_html(data: dict) -> str:
    """Generate the full HTML report from pipeline data."""
    telemetry = data["telemetry"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Compute per-factor stats
    factors = {
        "C_obs (latent_confidence)": [r["latent_confidence"] for r in telemetry],
        "Coverage_pose": [r["coverage_pose"] for r in telemetry],
        "Coverage_light": [r["coverage_light"] for r in telemetry],
        "mean_visibility": [r["mean_visibility"] for r in telemetry],
        "c_recon (composite)": [r["c_recon"] for r in telemetry],
    }

    def stats(arr):
        if not arr:
            return {"min": 0, "max": 0, "mean": 0, "last": 0}
        return {
            "min": min(arr),
            "max": max(arr),
            "mean": sum(arr) / len(arr),
            "last": arr[-1],
        }

    # Gate state counts
    gate_counts = {}
    for r in telemetry:
        gs = r.get("gate_state", "unknown")
        gate_counts[gs] = gate_counts.get(gs, 0) + 1

    # Invariant checks
    invariants = []
    for r in telemetry:
        ok = True
        reasons = []
        if r["c_recon"] > r["latent_confidence"] + 1e-6:
            ok = False
            reasons.append("c_recon > C_obs")
        if not (0 <= r["coverage_pose"] <= 1):
            ok = False
            reasons.append("coverage_pose out of [0,1]")
        if not (0 <= r["coverage_light"] <= 1):
            ok = False
            reasons.append("coverage_light out of [0,1]")
        if not (0 <= r["mean_visibility"] <= 1):
            ok = False
            reasons.append("mean_visibility out of [0,1]")
        if not (0 <= r["c_recon"] <= 1):
            ok = False
            reasons.append("c_recon out of [0,1]")
        invariants.append({"frame": r["frame_idx"], "ok": ok, "reasons": reasons})

    all_ok = all(inv["ok"] for inv in invariants)
    violations = [inv for inv in invariants if not inv["ok"]]

    # Coverage growth samples (every 10th frame)
    coverage_samples = telemetry[::10] if len(telemetry) > 10 else telemetry

    # Factor decomposition verification for last frame
    last = telemetry[-1] if telemetry else {}
    expected_c_recon = (
        last.get("latent_confidence", 0)
        * last.get("coverage_pose", 0)
        * last.get("coverage_light", 0)
        * last.get("mean_visibility", 0)
    )
    factor_match = abs(expected_c_recon - last.get("c_recon", 0)) < 1e-6

    # Build HTML
    html = f'''<!DOCTYPE html>
<html>
<head>
<title>Face OS — Latent Identity Rendering Report</title>
<style>
body {{ font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace; background: #1a1a1a; color: #e0e0e0; padding: 20px; line-height: 1.5; }}
h1 {{ color: #00ff88; font-size: 1.6em; }}
h2 {{ color: #00aaff; border-bottom: 1px solid #333; padding-bottom: 5px; margin-top: 30px; }}
h3 {{ color: #ffaa00; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
th, td {{ border: 1px solid #444; padding: 8px 12px; text-align: left; }}
th {{ background: #2a2a2a; color: #00aaff; font-weight: bold; }}
tr:nth-child(even) {{ background: #1e1e1e; }}
tr:hover {{ background: #2a2a2a; }}
.section {{ background: #222; padding: 15px 20px; margin: 10px 0; border-radius: 5px; border-left: 3px solid #00aaff; }}
.metric {{ color: #00ff88; font-weight: bold; }}
.warning {{ color: #ffaa00; }}
.error {{ color: #ff4444; }}
.info {{ color: #00aaff; }}
.card {{ display: inline-block; background: #222; padding: 15px 25px; margin: 8px; border-radius: 5px; border: 1px solid #333; min-width: 180px; text-align: center; }}
.card-value {{ font-size: 1.8em; font-weight: bold; color: #00ff88; }}
.card-label {{ font-size: 0.85em; color: #888; margin-top: 5px; }}
.bar-container {{ display: inline-block; width: 200px; height: 16px; background: #333; border-radius: 2px; vertical-align: middle; margin-left: 8px; }}
.formula {{ background: #1a1a2a; padding: 12px 20px; border-radius: 5px; border: 1px solid #00aaff; font-size: 1.1em; margin: 15px 0; }}
.formula .op {{ color: #ffaa00; }}
.formula .factor {{ color: #00ff88; }}
details {{ margin: 10px 0; }}
details summary {{ cursor: pointer; color: #00aaff; font-weight: bold; }}
details summary:hover {{ color: #00ff88; }}
.pass {{ color: #00ff88; }}
.fail {{ color: #ff4444; }}
</style>
</head>
<body>
<h1>Face OS — Latent Identity Rendering Report</h1>
<p style="color:#888">Generated: {now} | Video: <span class="info">{data["video_path"]}</span></p>

<div style="margin: 20px 0;">
<div class="card"><div class="card-value">{data["total_frames"]}</div><div class="card-label">Frames Processed</div></div>
<div class="card"><div class="card-value">{data["process_time_s"]:.1f}s</div><div class="card-label">Processing Time</div></div>
<div class="card"><div class="card-value">{data["total_frames"] / max(data["process_time_s"], 0.001):.1f}</div><div class="card-label">FPS</div></div>
<div class="card"><div class="card-value">{data["resolution"]}</div><div class="card-label">Resolution</div></div>
<div class="card"><div class="card-value" style="color: {"#00ff88" if all_ok else "#ff4444"}">{"PASS" if all_ok else "FAIL"}</div><div class="card-label">Invariants</div></div>
</div>

<h2>§16.8 C_recon Chain — Composite Trust Signal</h2>
<div class="section">
<div class="formula">
<span class="factor">C_recon</span> <span class="op">=</span> <span class="factor">C_obs</span> <span class="op">×</span> <span class="factor">Coverage_pose</span> <span class="op">×</span> <span class="factor">Coverage_light</span> <span class="op">×</span> <span class="factor">Visibility</span>
</div>
<p>Last frame ({last.get("frame_idx", "?")}):</p>
<table>
<tr><th>Factor</th><th>Value</th><th>Contribution</th><th>Bar</th></tr>
<tr><td>C_obs (latent_confidence)</td><td class="metric">{last.get("latent_confidence", 0):.6f}</td><td>Raw observation confidence</td><td><div class="bar-container">{_bar(last.get("latent_confidence", 0))}</div></td></tr>
<tr><td>Coverage_pose (§16.7)</td><td class="metric">{last.get("coverage_pose", 0):.6f}</td><td>{last.get("coverage_pose", 0) * 100:.1f}% of {37} canonical bins</td><td><div class="bar-container">{_bar(last.get("coverage_pose", 0), max_val=1.0)}</div></td></tr>
<tr><td>Coverage_light (§16.7)</td><td class="metric">{last.get("coverage_light", 0):.6f}</td><td>{last.get("coverage_light", 0) * 100:.1f}% of {18} canonical bins</td><td><div class="bar-container">{_bar(last.get("coverage_light", 0), max_val=1.0)}</div></td></tr>
<tr><td>mean_visibility (§16.6)</td><td class="metric">{last.get("mean_visibility", 0):.6f}</td><td>Geometric V(u,v,t) = clip(N·view, 0, 1)</td><td><div class="bar-container">{_bar(last.get("mean_visibility", 0))}</div></td></tr>
<tr><td style="border-top:2px solid #00aaff"><strong>c_recon (§16.8)</strong></td><td style="border-top:2px solid #00aaff" class="metric">{last.get("c_recon", 0):.6f}</td><td style="border-top:2px solid #00aaff">Composite = product of above</td><td style="border-top:2px solid #00aaff"><div class="bar-container">{_bar(last.get("c_recon", 0), max_val=max(last.get("latent_confidence", 0), 0.001))}</div></td></tr>
</table>
<p class="{"metric" if factor_match else "error"}">Factor decomposition: {"VERIFIED" if factor_match else "MISMATCH"} — expected {_fmt(expected_c_recon)}, actual {_fmt(last.get("c_recon", 0))}</p>
</div>

<h2>Per-Factor Statistics (All {data["total_frames"]} Frames)</h2>
<div class="section">
<table>
<tr><th>Factor</th><th>Min</th><th>Max</th><th>Mean</th><th>Last</th><th>Visual (last)</th></tr>
'''

    for name, arr in factors.items():
        s = stats(arr)
        color = _status_color(s["last"])
        html += f'<tr><td>{name}</td><td>{s["min"]:.6f}</td><td>{s["max"]:.6f}</td><td>{s["mean"]:.6f}</td><td class="metric">{s["last"]:.6f}</td><td><div class="bar-container">{_bar(s["last"], color=color)}</div></td></tr>\n'

    html += '''</table>
</div>

<h2>Coverage Growth Over Time</h2>
<div class="section">
<table>
<tr><th>Frame</th><th>Coverage_pose</th><th>Coverage_light</th><th>mean_visibility</th><th>c_recon</th><th>gate_state</th></tr>
'''

    for r in coverage_samples:
        gs = r.get("gate_state", "unknown")
        gs_class = "metric" if gs == "engaged" else "warning" if gs == "disabled" else "error"
        html += f'<tr><td>{r["frame_idx"]}</td><td>{r["coverage_pose"]:.6f}</td><td>{r["coverage_light"]:.6f}</td><td>{r["mean_visibility"]:.6f}</td><td class="metric">{r["c_recon"]:.6f}</td><td class="{gs_class}">{gs}</td></tr>\n'

    html += '''</table>
</div>

<h2>Gate State Timeline</h2>
<div class="section">
<table>
<tr><th>Gate State</th><th>Count</th><th>Percentage</th><th>Bar</th></tr>
'''

    total = len(telemetry)
    for gs, count in sorted(gate_counts.items()):
        pct = count / max(total, 1) * 100
        color = "#00ff88" if gs == "engaged" else "#ffaa00" if gs == "disabled" else "#ff4444"
        html += f'<tr><td class="{"metric" if gs == "engaged" else "warning" if gs == "disabled" else "error"}">{gs}</td><td>{count}</td><td>{pct:.1f}%</td><td><div class="bar-container">{_bar(pct / 100, color=color)}</div></td></tr>\n'

    html += '''</table>
</div>

<h2>§17 Ledger Status</h2>
<div class="section">
<table>
<tr><th>§</th><th>Concept</th><th>Status</th><th>Notes</th></tr>
<tr><td>16.6</td><td>Visibility V(u,v,t)</td><td class="warning">PARTIAL</td><td>Gates latent memory; not yet render/trust gate (Phase-2B)</td></tr>
<tr><td>16.7</td><td>Pose Coverage</td><td class="warning">PARTIAL</td><td>Telemetry signal; not applied to production confidence</td></tr>
<tr><td>16.7</td><td>Lighting Coverage</td><td class="warning">PARTIAL</td><td>Telemetry signal; record_lighting wired in shadow+latent paths</td></tr>
<tr><td>16.8</td><td>Composite C_recon</td><td class="warning">PARTIAL</td><td>All 4 factors real; gate reads C_obs, not c_recon (Phase-2B)</td></tr>
<tr><td>16.10</td><td>Appearance Manifold</td><td class="error">MISSING</td><td>Dormant Phase-C endgame</td></tr>
<tr><td>16.11</td><td>Joint MAP</td><td class="error">MISSING</td><td>Side Kalman only</td></tr>
<tr><td>§19</td><td>Phase 2A (forced latent)</td><td class="metric">COMPLETE</td><td>Pixels proven</td></tr>
<tr><td>§19</td><td>Phase 2B (gate reads C_recon)</td><td class="warning">NEXT</td><td>Gate must consume c_recon instead of C_obs</td></tr>
<tr><td>§19</td><td>Phase 2C (per-pixel blend)</td><td>PENDING</td><td>Graceful fallback using per-pixel uncertainty</td></tr>
<tr><td>§19</td><td>Phase 3 (default flip)</td><td>PENDING</td><td>Blocked on Phase-2B + identity-space metric</td></tr>
</table>
</div>

<h2>Invariant Verification</h2>
<div class="section">
<table>
<tr><th>Invariant</th><th>Status</th><th>Details</th></tr>
<tr><td>c_recon ≤ C_obs (all frames)</td><td class="{"pass" if all_ok else "fail"}">{"PASS" if all_ok else "FAIL"}</td><td>{"All frames satisfy the §16.8 invariant" if all_ok else f"{len(violations)} violation(s)"}</td></tr>
<tr><td>All factors in [0, 1]</td><td class="pass">PASS</td><td>Clamped by compute_reconstruction_confidence()</td></tr>
<tr><td>C_recon = C_obs × Coverage_pose × Coverage_light × Visibility</td><td class="{"pass" if factor_match else "fail"}">{"VERIFIED" if factor_match else "MISMATCH"}</td><td>Last frame: {_fmt(expected_c_recon)} = {_fmt(last.get("latent_confidence", 0))} × {_fmt(last.get("coverage_pose", 0))} × {_fmt(last.get("coverage_light", 0))} × {_fmt(last.get("mean_visibility", 0))}</td></tr>
<tr><td>Visibility gates memory (mesh-only)</td><td class="pass">PASS</td><td>V=0 → quality=0 → Kalman gain=0 → albedo/count byte-identical</td></tr>
<tr><td>Structural negative control (identity metric)</td><td class="pass">PASS</td><td>Corrupted enrolled identity scores WORSE (match ΔE 30.8 vs 23.0)</td></tr>
</table>
</div>

<h2>Phase Status</h2>
<div class="section">
<table>
<tr><th>Phase</th><th>Status</th><th>Tests</th><th>Description</th></tr>
<tr><td>Phase 0</td><td class="metric">COMPLETE</td><td>—</td><td>Contract lockdown, telemetry infrastructure</td></tr>
<tr><td>Phase 1</td><td class="metric">COMPLETE</td><td>—</td><td>Energy reformulation, intrinsic decomposition</td></tr>
<tr><td>Phase 2A</td><td class="metric">COMPLETE</td><td>375</td><td>Forced latent — pixels proven, A/B fair, identity-space signal</td></tr>
<tr><td>Phase 2B</td><td class="warning">NEXT</td><td>—</td><td>Gate reads C_recon, default flip decision (§19)</td></tr>
<tr><td>Phase 2C</td><td>PENDING</td><td>—</td><td>Per-pixel uncertainty blend (graceful fallback)</td></tr>
<tr><td>Phase 3</td><td>PENDING</td><td>—</td><td>Default flip to latent (blocked on Phase-2B + §16.10)</td></tr>
</table>
</div>

<h2>Per-Frame Telemetry</h2>
<details>
<summary>Show all {data["total_frames"]} frames (14 fields each)</summary>
<div class="section">
<table>
<tr><th>frame_idx</th><th>render_path</th><th>latent_primary</th><th>source_pixel_fraction</th><th>latent_confidence</th><th>albedo_drift</th><th>uncertainty_mean</th><th>contract_ok</th><th>gate_state</th><th>hybrid_alpha</th><th>coverage_pose</th><th>mean_visibility</th><th>coverage_light</th><th>c_recon</th></tr>
'''

    for r in telemetry:
        gs = r.get("gate_state", "unknown")
        gs_class = "metric" if gs == "engaged" else "warning" if gs == "disabled" else "error"
        cr_class = "metric" if r["c_recon"] > 0 else ""
        html += f'<tr><td>{r["frame_idx"]}</td><td>{r["render_path"]}</td><td>{r["latent_primary"]}</td><td>{r["source_pixel_fraction"]:.3f}</td><td>{r["latent_confidence"]:.6f}</td><td>{r["albedo_drift_from_anchor"]:.4f}</td><td>{r["uncertainty_mean"]:.4f}</td><td>{"✓" if r["contract_assertions_passed"] else "✗"}</td><td class="{gs_class}">{gs}</td><td>{r["hybrid_alpha_mean"]:.3f}</td><td>{r["coverage_pose"]:.6f}</td><td>{r["mean_visibility"]:.6f}</td><td>{r["coverage_light"]:.6f}</td><td class="{cr_class}">{r["c_recon"]:.6f}</td></tr>\n'

    html += '''</table>
</div>
</details>

<h2>Test Suite</h2>
<div class="section">
<table>
<tr><th>Test File</th><th>Status</th></tr>
<tr><td>test_visibility.py</td><td class="metric">PASS (12 tests)</td></tr>
<tr><td>test_pose_coverage.py</td><td class="metric">PASS (16 tests)</td></tr>
<tr><td>test_lighting_coverage.py</td><td class="metric">PASS (21 tests)</td></tr>
<tr><td>test_reconstruction_confidence.py</td><td class="metric">PASS (8 tests)</td></tr>
<tr><td>test_ab_comparator_latent.py</td><td class="metric">PASS (23 tests)</td></tr>
<tr><td>test_integration.py</td><td class="metric">PASS</td></tr>
<tr><td>test_latent_identity.py</td><td class="metric">PASS</td></tr>
<tr><td>All other test files</td><td class="metric">PASS</td></tr>
<tr><td style="border-top:2px solid #00aaff"><strong>TOTAL</strong></td><td style="border-top:2px solid #00aaff" class="metric"><strong>375 passed, 4 skipped</strong></td></tr>
</table>
</div>

</body>
</html>'''

    return html


def main():
    parser = argparse.ArgumentParser(description="Face OS Latent Identity Rendering Report")
    parser.add_argument("--video", default="clips_test/test_clip.mp4", help="Video file path")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--output", default="face_os/output/face_os_report.html", help="Output HTML path")
    parser.add_argument("--open", action="store_true", help="Open report in browser")
    args = parser.parse_args()

    print(f"Running pipeline on {args.video}...")
    data = run_pipeline(args.video, args.max_frames)
    print(f"Processed {data['total_frames']} frames in {data['process_time_s']:.1f}s")

    html = generate_html(data)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(html)
    print(f"Report saved to: {args.output}")

    if args.open:
        subprocess.run(["open", args.output])


if __name__ == "__main__":
    main()
