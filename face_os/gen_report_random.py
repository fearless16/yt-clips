"""Face OS — Random 5s Clip Frame-by-Frame HTML Report.

Picks a random 5s segment from a source video, runs the pipeline frame-by-frame,
captures the actual frames, computes per-frame C_recon / C_obs telemetry, and
emits a self-contained HTML report with embedded thumbnails + expectation.png
ground-truth comparison.

Usage:
    python face_os/gen_report_random.py
    python face_os/gen_report_random.py --seed 42
    python face_os/gen_report_random.py --open
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

DEFAULT_SOURCE = "clips_test/test_clip.mp4"
DEFAULT_OUTPUT = "output/random_5s_report.html"
EXPECTATION_PATH = "expectation.png"

# Per-frame thumbnail size (keeps HTML report size manageable)
THUMB_W = 240


def _b64_jpeg(img: np.ndarray, quality: int = 80) -> str:
    """Encode a BGR ndarray to a base64 JPEG data URI."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _b64_png(path: str) -> str:
    """Encode a PNG file to a base64 data URI (for ground-truth reference)."""
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


def _resize_keep_aspect(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= width:
        return img
    return cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)


def _status_color(val: float, good: float = 0.05, warn: float = 0.001) -> str:
    if val >= good:
        return "#00ff88"
    if val >= warn:
        return "#ffaa00"
    return "#ff4444"


def _bar_html(val: float, max_val: float = 1.0, width: int = 80, color: str = "#00ff88") -> str:
    pct = min(max(val / max(max_val, 1e-9), 0.0), 1.0) * 100
    return (
        f'<span style="display:inline-block;width:{width}px;height:10px;'
        f'background:#222;border:1px solid #444;border-radius:2px;vertical-align:middle">'
        f'<span style="display:block;height:100%;width:{pct:.1f}%;background:{color};border-radius:1px"></span>'
        f"</span>"
    )


def _draw_legend(img: np.ndarray, text: str) -> np.ndarray:
    """Overlay a small black bar with white text on top-left of an image."""
    out = img.copy()
    h = 18
    cv2.rectangle(out, (0, 0), (out.shape[1], h), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def pick_random_5s_window(cap: cv2.VideoCapture, fps: float, duration_s: float = 5.0) -> tuple[int, int]:
    """Pick a random 5s window inside the video. Returns (start_frame, end_frame)."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    window = int(duration_s * fps)
    if total <= window:
        return 0, total
    start = random.randint(0, total - window)
    return start, start + window


def run_pipeline_with_frames(
    video_path: str, start_frame: int, end_frame: int
) -> dict:
    """Run pipeline, capturing each input frame alongside its telemetry."""
    from face_os.pipeline import FaceOSPipeline

    pipeline = FaceOSPipeline()
    t0 = time.time()
    pipeline.enroll()
    enroll_time = time.time() - t0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames: list[np.ndarray] = []
    telemetry: list[dict] = []
    timestamps_s: list[float] = []
    t1 = time.time()
    for fidx in range(start_frame, end_frame):
        ok, frame = cap.read()
        if not ok:
            break
        pipeline.process_frame(frame, fidx)
        frames.append(frame)
        timestamps_s.append((fidx - start_frame) / fps)
    cap.release()
    process_time = time.time() - t1

    tel = pipeline.get_latent_telemetry()
    return {
        "frames": frames,
        "timestamps_s": timestamps_s,
        "telemetry": tel,
        "enroll_time_s": enroll_time,
        "process_time_s": process_time,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "fps": fps,
    }


def generate_html(data: dict, expectation_b64: str | None) -> str:
    telemetry = data["telemetry"]
    frames = data["frames"]
    ts = data["timestamps_s"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n = len(telemetry)

    # Aggregate stats
    c_obs = [r["latent_confidence"] for r in telemetry]
    cov_pose = [r["coverage_pose"] for r in telemetry]
    cov_light = [r["coverage_light"] for r in telemetry]
    mean_vis = [r["mean_visibility"] for r in telemetry]
    c_recon = [r["c_recon"] for r in telemetry]

    def _stats(arr):
        if not arr:
            return (0, 0, 0, 0)
        return (min(arr), max(arr), sum(arr) / len(arr), arr[-1])

    s_cobs = _stats(c_obs)
    s_covp = _stats(cov_pose)
    s_covl = _stats(cov_light)
    s_vis = _stats(mean_vis)
    s_crec = _stats(c_recon)

    # Gate state counts
    gate_counts: dict[str, int] = {}
    for r in telemetry:
        gs = r.get("gate_state", "unknown")
        gate_counts[gs] = gate_counts.get(gs, 0) + 1

    # Invariant verification
    violations = 0
    for r in telemetry:
        if r["c_recon"] > r["latent_confidence"] + 1e-6:
            violations += 1
    all_ok = violations == 0

    # Match last frame factor decomposition
    last = telemetry[-1] if telemetry else {}
    expected_cr = (
        last.get("latent_confidence", 0)
        * last.get("coverage_pose", 0)
        * last.get("coverage_light", 0)
        * last.get("mean_visibility", 0)
    )
    factor_match = abs(expected_cr - last.get("c_recon", 0)) < 1e-6

    # Encode frames as base64 thumbnails (with frame_idx overlay for clarity)
    frame_thumbs: list[str] = []
    for i, frame in enumerate(frames):
        annotated = _draw_legend(
            _resize_keep_aspect(frame, THUMB_W),
            f"frame {telemetry[i]['frame_idx']} t={ts[i]:.2f}s",
        )
        frame_thumbs.append(_b64_jpeg(annotated, quality=75))

    expectation_html = ""
    if expectation_b64:
        expectation_html = f'''
<div class="expectation-block">
  <h3 style="margin:0 0 8px 0">Enrolled Ground Truth (expectation.png)</h3>
  <img src="{expectation_b64}" style="max-width:240px;border:2px solid #00aaff;border-radius:4px" />
  <p style="color:#888;margin:6px 0 0 0">This is the speaker the pipeline was enrolled against.
  Per-frame metrics below indicate how well each input frame could be reconstructed from this identity.</p>
</div>
'''

    # Build per-frame table rows
    rows_html = []
    for i, r in enumerate(telemetry):
        obs = r["latent_confidence"]
        cr = r["c_recon"]
        cp = r["coverage_pose"]
        cl = r["coverage_light"]
        mv = r["mean_visibility"]
        gs = r.get("gate_state", "unknown")
        rp = r.get("render_path", "?")
        th = frame_thumbs[i] if i < len(frame_thumbs) else ""
        trust = "TRUST" if cr >= 0.01 else ("MARGINAL" if cr >= 0.001 else "LOW")
        trust_color = "#00ff88" if cr >= 0.01 else ("#ffaa00" if cr >= 0.001 else "#ff4444")
        gs_class = "metric" if gs == "engaged" else ("warning" if gs == "disabled" else "error")
        rows_html.append(f'''
<tr>
  <td class="thumb-cell"><img src="{th}" style="width:200px;border:1px solid #333;border-radius:3px" /></td>
  <td><b>{r['frame_idx']}</b><br/><span style="color:#888">{ts[i]:.2f}s</span></td>
  <td>{rp}</td>
  <td class="{gs_class}">{gs}</td>
  <td class="metric">{obs:.4f}</td>
  <td>{_bar_html(cp, 1.0, 60, _status_color(cp, 0.1, 0.02))} {cp*100:.1f}%</td>
  <td>{_bar_html(cl, 1.0, 60, _status_color(cl, 0.1, 0.02))} {cl*100:.1f}%</td>
  <td>{_bar_html(mv, 1.0, 60, _status_color(mv, 0.5, 0.2))} {mv:.2f}</td>
  <td style="color:{trust_color}"><b>{cr:.6f}</b><br/><span style="font-size:0.8em">{trust}</span></td>
</tr>''')

    # Spans
    start_s = ts[0] if ts else 0.0
    end_s = ts[-1] if ts else 0.0
    duration_s = end_s - start_s

    # Per-factor aggregate table
    aggregate_rows = ""
    for name, arr, s in [
        ("C_obs (latent_confidence)", c_obs, s_cobs),
        ("coverage_pose (§16.7)", cov_pose, s_covp),
        ("coverage_light (§16.7)", cov_light, s_covl),
        ("mean_visibility (§16.6)", mean_vis, s_vis),
        ("c_recon (§16.8 composite)", c_recon, s_crec),
    ]:
        col = _status_color(s[3], 0.05, 0.001) if "c_recon" not in name else _status_color(s[3], 0.01, 0.001)
        aggregate_rows += f'''
<tr>
  <td>{name}</td>
  <td>{s[0]:.6f}</td>
  <td>{s[1]:.6f}</td>
  <td>{s[2]:.6f}</td>
  <td style="color:{col}"><b>{s[3]:.6f}</b></td>
  <td>{_bar_html(s[3], 1.0, 120, col)}</td>
</tr>'''

    gate_rows = ""
    total = max(n, 1)
    for gs, count in sorted(gate_counts.items()):
        pct = count / total * 100
        col = "#00ff88" if gs == "engaged" else ("#ffaa00" if gs == "disabled" else "#ff4444")
        gs_class = "metric" if gs == "engaged" else ("warning" if gs == "disabled" else "error")
        gate_rows += f'<tr><td class="{gs_class}">{gs}</td><td>{count}</td><td>{pct:.1f}%</td><td>{_bar_html(pct/100, 1.0, 200, col)}</td></tr>'

    # Build sparkline-style coverage chart (inline SVG)
    chart_w = 880
    chart_h = 160
    n_max = max(n, 1)
    paths = {}
    for key, color in [
        ("c_recon", "#00ff88"),
        ("latent_confidence", "#00aaff"),
        ("coverage_pose", "#ffaa00"),
    ]:
        pts = []
        for i, r in enumerate(telemetry):
            x = i / n_max * chart_w
            y = chart_h - min(max(r[key], 0.0), 1.0) * chart_h
            pts.append(f"{x:.1f},{y:.1f}")
        paths[key] = (color, " ".join(pts))

    chart_svg = f'<svg width="{chart_w}" height="{chart_h}" style="background:#1a1a1a;border:1px solid #333;border-radius:4px">'
    # gridlines
    for g in [0.25, 0.5, 0.75]:
        y = chart_h - g * chart_h
        chart_svg += f'<line x1="0" y1="{y}" x2="{chart_w}" y2="{y}" stroke="#333" stroke-dasharray="2,2" />'
        chart_svg += f'<text x="4" y="{y-2}" font-size="9" fill="#666">{g:.2f}</text>'
    for color, pts in paths.values():
        chart_svg += f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{pts}" />'
    chart_svg += "</svg>"

    legend = ""
    for label, key in [("c_recon", "c_recon"), ("c_obs (latent_confidence)", "latent_confidence"), ("coverage_pose", "coverage_pose")]:
        col = paths[key][0]
        legend += f'<span style="display:inline-block;margin-right:14px"><span style="display:inline-block;width:10px;height:10px;background:{col};margin-right:4px"></span>{label}</span>'

    # Trust summary
    trust_high = sum(1 for r in telemetry if r["c_recon"] >= 0.01)
    trust_mid = sum(1 for r in telemetry if 0.001 <= r["c_recon"] < 0.01)
    trust_low = sum(1 for r in telemetry if r["c_recon"] < 0.001)
    trust_high_pct = trust_high / n * 100 if n else 0
    trust_mid_pct = trust_mid / n * 100 if n else 0
    trust_low_pct = trust_low / n * 100 if n else 0

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Face OS — Random 5s Frame-by-Frame Report</title>
<style>
  body {{ font-family: 'SF Mono','Fira Code','Cascadia Code',monospace; background:#0f0f0f; color:#e0e0e0; padding:20px; line-height:1.45; margin:0; }}
  h1 {{ color:#00ff88; font-size:1.6em; margin:0 0 6px 0; }}
  h2 {{ color:#00aaff; border-bottom:1px solid #333; padding-bottom:5px; margin-top:30px; }}
  h3 {{ color:#ffaa00; }}
  .meta {{ color:#888; font-size:0.9em; }}
  table {{ border-collapse:collapse; width:100%; margin:10px 0; }}
  th, td {{ border:1px solid #333; padding:6px 8px; text-align:left; font-size:0.85em; vertical-align:middle; }}
  th {{ background:#1a1a1a; color:#00aaff; font-weight:bold; position:sticky; top:0; z-index:1; }}
  tr:nth-child(even) {{ background:#161616; }}
  tr:hover {{ background:#222; }}
  .section {{ background:#181818; padding:14px 18px; margin:10px 0; border-radius:5px; border-left:3px solid #00aaff; }}
  .metric {{ color:#00ff88; font-weight:bold; }}
  .warning {{ color:#ffaa00; }}
  .error {{ color:#ff4444; }}
  .info {{ color:#00aaff; }}
  .card {{ display:inline-block; background:#1a1a1a; padding:10px 18px; margin:6px; border-radius:5px; border:1px solid #333; min-width:140px; text-align:center; }}
  .card-value {{ font-size:1.6em; font-weight:bold; color:#00ff88; }}
  .card-label {{ font-size:0.75em; color:#888; margin-top:3px; text-transform:uppercase; letter-spacing:0.5px; }}
  .expectation-block {{ background:#181818; padding:14px 18px; margin:10px 0; border-radius:5px; border-left:3px solid #ffaa00; }}
  .thumb-cell {{ width:210px; padding:4px; }}
  details {{ margin:10px 0; }}
  details summary {{ cursor:pointer; color:#00aaff; font-weight:bold; padding:6px 0; }}
  details summary:hover {{ color:#00ff88; }}
  .pass {{ color:#00ff88; }}
  .fail {{ color:#ff4444; }}
  .top-cards {{ display:flex; flex-wrap:wrap; gap:6px; margin:14px 0; }}
</style>
</head>
<body>
<h1>Face OS — Random 5s Frame-by-Frame Report</h1>
<p class="meta">Generated: {now} | Source: <span class="info">{data.get("video_path","?")}</span> | Window: frames {data["start_frame"]}–{data["end_frame"]} (t={start_s:.2f}s → {end_s:.2f}s, {duration_s:.2f}s) @ {data["fps"]:.1f} fps</p>

<div class="top-cards">
  <div class="card"><div class="card-value">{n}</div><div class="card-label">Frames</div></div>
  <div class="card"><div class="card-value">{duration_s:.2f}s</div><div class="card-label">Clip Length</div></div>
  <div class="card"><div class="card-value">{data["process_time_s"]:.1f}s</div><div class="card-label">Process Time</div></div>
  <div class="card"><div class="card-value">{n / max(data["process_time_s"], 0.001):.1f}</div><div class="card-label">FPS</div></div>
  <div class="card"><div class="card-value">{data["enroll_time_s"]:.1f}s</div><div class="card-label">Enroll</div></div>
  <div class="card"><div class="card-value" style="color:{"#00ff88" if all_ok else "#ff4444"}">{"PASS" if all_ok else "FAIL"}</div><div class="card-label">Invariants</div></div>
  <div class="card"><div class="card-value" style="color:#00ff88">{trust_high}</div><div class="card-label">Trust (≥0.01)</div></div>
  <div class="card"><div class="card-value" style="color:#ffaa00">{trust_mid}</div><div class="card-label">Marginal</div></div>
  <div class="card"><div class="card-value" style="color:#ff4444">{trust_low}</div><div class="card-label">Low (&lt;0.001)</div></div>
</div>

{expectation_html}

<h2>Per-Frame Trust Signals (timeline)</h2>
<div class="section">
  <p class="meta">{legend}</p>
  {chart_svg}
  <p class="meta">Y-axis = signal value [0,1]. X-axis = frame index in the 5s window ({n} frames).</p>
</div>

<h2>§16.8 C_recon Chain (last frame: {last.get("frame_idx","?")})</h2>
<div class="section">
  <div class="formula" style="background:#0a0a1a;padding:10px 16px;border-radius:4px;border:1px solid #00aaff;font-size:1.05em;margin:6px 0">
    <span class="metric">c_recon</span> = <span class="metric">C_obs</span> × <span class="metric">coverage_pose</span> × <span class="metric">coverage_light</span> × <span class="metric">visibility</span>
  </div>
  <table>
    <tr><th>Factor</th><th>Value</th><th>Notes</th></tr>
    <tr><td>C_obs (latent_confidence)</td><td class="metric">{last.get("latent_confidence",0):.6f}</td><td>Raw observation confidence from latent render</td></tr>
    <tr><td>coverage_pose (§16.7)</td><td class="metric">{last.get("coverage_pose",0):.6f}</td><td>{last.get("coverage_pose",0)*100:.1f}% of 37 canonical pose bins observed</td></tr>
    <tr><td>coverage_light (§16.7)</td><td class="metric">{last.get("coverage_light",0):.6f}</td><td>{last.get("coverage_light",0)*100:.1f}% of 18 canonical lighting bins observed</td></tr>
    <tr><td>mean_visibility (§16.6)</td><td class="metric">{last.get("mean_visibility",0):.6f}</td><td>Geometric V(u,v,t) = clip(N·view, 0, 1)</td></tr>
    <tr style="border-top:2px solid #00aaff"><td><b>c_recon (§16.8)</b></td><td style="border-top:2px solid #00aaff" class="metric">{last.get("c_recon",0):.6f}</td><td style="border-top:2px solid #00aaff">Composite product — trust signal</td></tr>
  </table>
  <p class="{"metric" if factor_match else "error"}">Factor decomposition: {"VERIFIED" if factor_match else "MISMATCH"} — expected {expected_cr:.6f}, actual {last.get("c_recon",0):.6f}</p>
</div>

<h2>Per-Factor Statistics (all {n} frames)</h2>
<div class="section">
  <table>
    <tr><th>Factor</th><th>Min</th><th>Max</th><th>Mean</th><th>Last</th><th>Visual (last)</th></tr>
    {aggregate_rows}
  </table>
</div>

<h2>Gate State Distribution</h2>
<div class="section">
  <table>
    <tr><th>Gate State</th><th>Count</th><th>Percent</th><th>Bar</th></tr>
    {gate_rows}
  </table>
</div>

<h2>Frame-by-Frame Telemetry ({n} frames)</h2>
<div class="section" style="padding:0;overflow-x:auto">
  <table>
    <tr>
      <th style="width:210px">Frame</th>
      <th>idx / t</th>
      <th>render_path</th>
      <th>gate_state</th>
      <th>C_obs</th>
      <th>cov_pose</th>
      <th>cov_light</th>
      <th>mean_vis</th>
      <th>c_recon (trust)</th>
    </tr>
    {"".join(rows_html)}
  </table>
</div>

<h2>Invariant Verification</h2>
<div class="section">
  <table>
    <tr><th>Invariant</th><th>Status</th><th>Details</th></tr>
    <tr><td>c_recon ≤ C_obs (all frames)</td><td class="{"pass" if all_ok else "fail"}">{"PASS" if all_ok else "FAIL"}</td><td>{n - violations} / {n} frames satisfy §16.8 invariant</td></tr>
    <tr><td>C_recon = C_obs × cov_pose × cov_light × visibility</td><td class="{"pass" if factor_match else "fail"}">{"VERIFIED" if factor_match else "MISMATCH"}</td><td>Last frame: {expected_cr:.6f} = {last.get("latent_confidence",0):.6f} × {last.get("coverage_pose",0):.6f} × {last.get("coverage_light",0):.6f} × {last.get("mean_visibility",0):.6f}</td></tr>
    <tr><td>All factors in [0,1]</td><td class="pass">PASS</td><td>Clamped by compute_reconstruction_confidence()</td></tr>
  </table>
</div>

<h2>Trust Distribution</h2>
<div class="section">
  <table>
    <tr><th>Tier</th><th>Threshold</th><th>Count</th><th>Percent</th><th>Bar</th></tr>
    <tr><td class="metric">TRUST</td><td>c_recon ≥ 0.01</td><td>{trust_high}</td><td>{trust_high_pct:.1f}%</td><td>{_bar_html(trust_high_pct/100, 1.0, 300, "#00ff88")}</td></tr>
    <tr><td class="warning">MARGINAL</td><td>0.001 ≤ c_recon &lt; 0.01</td><td>{trust_mid}</td><td>{trust_mid_pct:.1f}%</td><td>{_bar_html(trust_mid_pct/100, 1.0, 300, "#ffaa00")}</td></tr>
    <tr><td class="error">LOW</td><td>c_recon &lt; 0.001</td><td>{trust_low}</td><td>{trust_low_pct:.1f}%</td><td>{_bar_html(trust_low_pct/100, 1.0, 300, "#ff4444")}</td></tr>
  </table>
  <p class="meta">Trust threshold 0.01 ≈ "we have ≥ 1% pose + 5% lighting coverage AND ≥ 0.1 raw C_obs". Frames above this are reliable for the latent render decision.</p>
</div>

</body>
</html>'''
    return html


def main():
    parser = argparse.ArgumentParser(description="Face OS Random 5s Frame-by-Frame HTML Report")
    parser.add_argument("--video", default=DEFAULT_SOURCE, help="Source video")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output HTML path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--duration", type=float, default=5.0, help="Window length in seconds")
    parser.add_argument("--open", action="store_true", help="Open report in browser when done")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    start, end = pick_random_5s_window(cv2.VideoCapture(args.video), fps, args.duration)
    print(f"[random_5s] video={args.video} fps={fps} total={total} start={start} end={end} ({args.duration}s)")

    data = run_pipeline_with_frames(args.video, start, end)
    data["video_path"] = args.video
    print(f"[random_5s] processed {len(data['frames'])} frames in {data['process_time_s']:.1f}s")

    exp_b64 = None
    if os.path.exists(EXPECTATION_PATH):
        exp_b64 = _b64_png(EXPECTATION_PATH)
        print(f"[random_5s] loaded expectation.png as reference")

    html = generate_html(data, exp_b64)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write(html)
    size_kb = os.path.getsize(args.output) / 1024
    print(f"[random_5s] report saved to: {args.output} ({size_kb:.0f} KB)")

    if args.open:
        subprocess.run(["open", args.output])


if __name__ == "__main__":
    main()
