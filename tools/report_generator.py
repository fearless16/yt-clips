import json
import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports" / "face_detection"

with open(REPORTS / "expectation_metrics.json") as f:
    ref = json.load(f)
with open(REPORTS / "detection_summary.json") as f:
    summary = json.load(f)
with open(REPORTS / "per_frame_results.json") as f:
    frames = json.load(f)
with open(REPORTS / "cropped_metrics.json") as f:
    cropped = json.load(f)
with open(REPORTS / "cropped_summary.json") as f:
    csum = json.load(f)

frames_dir = REPORTS / "frames"
cropped_dir = REPORTS / "cropped_faces"

def img_to_b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()

ref_roi_b64 = img_to_b64(REPORTS / "expectation_face_roi.png")

frame_imgs = {}
for f in frames[:10]:
    p = frames_dir / f"frame_{f['frame_idx']:06d}.jpg"
    if p.exists():
        frame_imgs[f['frame_idx']] = img_to_b64(p)

cropped_imgs = {}
for c in cropped:
    if c["face_detected"]:
        cp = cropped_dir / f"face_{c['frame_idx']:06d}.jpg"
        if cp.exists():
            cropped_imgs[c["frame_idx"]] = img_to_b64(cp)

detected_frames = [f for f in frames if f.get("face_detected")]
detected_cropped = [c for c in cropped if c.get("face_detected")]

def delta(val, ref_val, pct=False):
    d = val - ref_val
    cls = "pos" if d > 0 else "neg" if d < 0 else "zero"
    s = f"{d:+.1f}" if not pct else f"{d:+.1f}pp"
    return f'<span class="delta {cls}">{s}</span>'

def gauge(val, low, high):
    if val < low:
        cls, p = "danger", val / low * 50
    elif val > high:
        cls, p = "warning", 50 + min((val - high) / high * 50, 50)
    else:
        cls, p = "good", 25 + (val - low) / (high - low) * 50
    return f'<div class="gauge {cls}" style="width:{min(p,100):.0f}%"></div>'

face_area_pcts = [f["face_area_pct"] for f in detected_frames]
face_brightness = [f["brightness"] for f in detected_frames]
face_contrasts = [f["contrast"] for f in detected_frames]
face_sharpness = [f["sharpness"] for f in detected_frames]
confidences = [f["confidence"] for f in detected_frames]
skin_a = [f["skin_lab"][1] for f in detected_frames]
skin_b = [f["skin_lab"][2] for f in detected_frames]
saturations = [f["saturation"] for f in detected_frames]

def sparkline(vals, h=30, color="#2196F3"):
    if not vals: return ""
    mn, mx = min(vals), max(vals)
    rng = max(mx - mn, 1)
    pts = []
    w = max(len(vals) * 4, 60)
    for i, v in enumerate(vals):
        x = i / (len(vals) - 1) * w if len(vals) > 1 else w / 2
        y = h - (v - mn) / rng * (h - 4) - 2
        pts.append(f"{x:.1f},{y:.1f}")
    return f'<svg width="{w}" height="{h}" class="sparkline"><polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="1.5"/></svg>'

def compute_stats(vals):
    if not vals: return {}
    import numpy as np
    a = np.array(vals)
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "median": float(np.median(a)),
    }

ref_face_area = ref["face_area_pct"]
ref_brightness = ref["brightness"]
ref_contrast = ref["contrast"]
ref_sharpness = ref["sharpness"]
ref_skin_a = ref["skin_lab"][1]
ref_skin_b = ref["skin_lab"][2]
ref_saturation = ref["saturation"]

def stat_row(label, detected_vals, ref_val, unit=""):
    s = compute_stats(detected_vals)
    mean, std = s["mean"], s["std"]
    return f"""<tr>
  <td class="metric-label">{label}</td>
  <td class="ref-val">{ref_val:.1f}{unit}</td>
  <td class="det-mean">{mean:.1f}{unit}</td>
  <td class="det-std">±{std:.1f}{unit}</td>
  <td class="det-range">{s["min"]:.1f} – {s["max"]:.1f}{unit}</td>
  <td class="det-delta">{delta(mean, ref_val)}</td>
  <td class="spark">{sparkline(detected_vals, color="#4CAF50")}</td>
</tr>"""

def cropped_stat_row(label, metric_key, expected_val, fmt=".1f", idx=None):
    if idx is not None:
        vals = [r["roi_metrics"][metric_key][idx] for r in detected_cropped]
    else:
        vals = [r["roi_metrics"][metric_key] for r in detected_cropped]
    s = compute_stats(vals)
    mean, std = s["mean"], s["std"]
    pct = float(f"{(mean - expected_val) / expected_val * 100:.1f}")
    pct_cls = "pos" if pct > 0 else "neg" if pct < 0 else "zero"
    return f"""<tr>
  <td class="metric-label">{label}</td>
  <td class="ref-val">{expected_val:{fmt}}</td>
  <td class="det-mean">{mean:{fmt}}</td>
  <td class="det-std">±{std:{fmt}}</td>
  <td class="det-delta">{delta(mean, expected_val)}</td>
  <td><span class="delta {pct_cls}">({pct:+.1f}%)</span></td>
  <td class="spark">{sparkline(vals, color="#4CAF50")}</td>
</tr>"""

cropped_roi_stats = []
for c in detected_cropped[:6]:
    m = c["roi_metrics"]
    d = c["deltas"]
    cropped_roi_stats.append(f"""<tr>
  <td>{c['frame_idx']}</td>
  <td>{c['timestamp_sec']:.1f}s</td>
  <td>{m['brightness']:.0f}</td>
  <td>{m['contrast']:.0f}</td>
  <td>{m['sharpness']:.0f}</td>
  <td>{m['saturation']:.0f}</td>
  <td>{'↗' if d['brightness_delta'] > 0 else '↘'}{abs(d['brightness_delta']):.0f}</td>
  <td>{'↗' if d['sharpness_delta'] > 0 else '↘'}{abs(d['sharpness_delta']):.0f}</td>
</tr>""")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Face Detection Report — After Cropping Analysis</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f1117; color: #e1e4e8; padding: 20px; }}
.container {{ max-width: 1200px; margin: 0 auto; }}
h1 {{ font-size: 28px; margin-bottom: 4px; background: linear-gradient(135deg, #58a6ff, #bc8cff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
h2 {{ font-size: 20px; color: #8b949e; font-weight: 400; margin-bottom: 24px; }}
h3 {{ font-size: 16px; color: #c9d1d9; margin: 24px 0 12px; padding-bottom: 8px; border-bottom: 1px solid #21262d; }}
.section {{ background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 20px; margin-bottom: 16px; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
.card {{ background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 14px; }}
.card-label {{ font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; }}
.card-value {{ font-size: 24px; font-weight: 600; margin-top: 4px; }}
.card-value.green {{ color: #3fb950; }}
.card-value.yellow {{ color: #d29922; }}
.card-value.red {{ color: #f85149; }}
.card-value.blue {{ color: #58a6ff; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th {{ text-align: left; padding: 8px 10px; color: #8b949e; border-bottom: 1px solid #21262d; font-weight: 500; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }}
td {{ padding: 8px 10px; border-bottom: 1px solid #1c2128; }}
tr:hover td {{ background: #1c2128; }}
.metric-label {{ color: #c9d1d9; font-weight: 500; }}
.ref-val {{ color: #58a6ff; font-family: 'SF Mono', 'Fira Code', monospace; }}
.det-mean {{ color: #e1e4e8; font-family: monospace; }}
.det-std {{ color: #8b949e; font-family: monospace; font-size: 12px; }}
.det-range {{ color: #8b949e; font-family: monospace; font-size: 12px; }}
.det-delta {{ font-family: monospace; }}
.delta {{ font-weight: 600; }}
.delta.pos {{ color: #f85149; }}
.delta.neg {{ color: #3fb950; }}
.delta.zero {{ color: #8b949e; }}
.spark {{ padding: 4px 10px; }}
.sparkline {{ display: block; }}
.sparkline polyline {{ filter: drop-shadow(0 0 3px rgba(33,150,243,0.4)); }}
.frame-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; }}
.frame-card {{ background: #0d1117; border: 1px solid #21262d; border-radius: 6px; overflow: hidden; }}
.frame-card img {{ width: 100%; display: block; }}
.frame-meta {{ padding: 10px; font-size: 12px; }}
.frame-meta .row {{ display: flex; justify-content: space-between; margin-bottom: 3px; }}
.frame-meta .label {{ color: #8b949e; }}
.frame-meta .val {{ color: #e1e4e8; font-family: monospace; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 500; }}
.badge.green {{ background: #1b3a1f; color: #3fb950; }}
.badge.red {{ background: #3d1a1e; color: #f85149; }}
.ref-image-container {{ text-align: center; }}
.ref-image-container img {{ max-width: 100%; max-height: 300px; border-radius: 6px; }}
.roi-compare {{ display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; margin: 12px 0; }}
.roi-compare img {{ height: 160px; border-radius: 6px; }}
.roi-compare .roi-item {{ text-align: center; }}
.roi-compare .roi-label {{ font-size: 11px; color: #8b949e; margin-top: 4px; }}
.gauge {{ height: 6px; border-radius: 3px; }}
.gauge.good {{ background: #3fb950; }}
.gauge.warning {{ background: #d29922; }}
.gauge.danger {{ background: #f85149; }}
.footer {{ text-align: center; color: #484f58; font-size: 12px; margin-top: 32px; padding: 16px; }}
.big-number {{ font-size: 36px; font-weight: 700; }}
.big-number.green {{ color: #3fb950; }}
.big-number.red {{ color: #f85149; }}
.big-number.yellow {{ color: #d29922; }}
.status-banner {{ padding: 12px 16px; border-radius: 6px; margin-bottom: 12px; font-size: 14px; }}
.status-banner.pass {{ background: #1b3a1f; border: 1px solid #2ea043; color: #7ee787; }}
.status-banner.fail {{ background: #3d1a1e; border: 1px solid #f85149; color: #ff7b72; }}
.status-banner.warn {{ background: #3d2e00; border: 1px solid #d29922; color: #e3b341; }}
@media (max-width: 768px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="container">

<h1>🔍 Face Detection Report — After Cropping Analysis</h1>
<h2>{Path(summary['video_path']).name} ({summary['frame_width']}×{summary['frame_height']}, {summary['total_frames']} frames @ {summary['fps']:.0f}fps)</h2>

<!-- ═══════════════════ OVERVIEW ═══════════════════ -->
<div class="section">
  <h3>📊 Overview</h3>
  <div class="grid">
    <div class="card">
      <div class="card-label">Detection Rate</div>
      <div class="card-value green">{summary['detection_rate']*100:.0f}%</div>
      <div style="color:#8b949e;font-size:12px;margin-top:4px">{len(detected_frames)}/{summary['samples']} frames</div>
    </div>
    <div class="card">
      <div class="card-label">Avg Confidence</div>
      <div class="card-value green">{summary['avg_confidence']:.2f}</div>
    </div>
    <div class="card">
      <div class="card-label">Face Area (% frame)</div>
      <div class="card-value yellow">{summary['avg_face_area_pct']:.1f}%</div>
      <div style="color:#8b949e;font-size:12px;margin-top:4px">Ref: {ref_face_area:.1f}%</div>
    </div>
    <div class="card">
      <div class="card-label">Cropped ROI Size μ</div>
      <div class="card-value blue">{csum['avg_brightness']:.0f}±{csum['std_brightness']:.0f} bright</div>
      <div style="color:#8b949e;font-size:12px;margin-top:4px">Sharp: {csum['avg_sharpness']:.0f}±{csum['std_sharpness']:.0f}</div>
    </div>
    <div class="card">
      <div class="card-label">Faces/Frame (avg)</div>
      <div class="card-value yellow">{sum(f['num_faces'] for f in detected_frames)/len(detected_frames):.1f}</div>
    </div>
    <div class="card">
      <div class="card-label">Host Match Rate</div>
      <div class="card-value blue">{summary['frames_with_host_match']}/{summary['samples']}</div>
    </div>
  </div>
</div>

<!-- ═══════════════════ AFTER CROPPING — QUALITY ═══════════════════ -->
<div class="section">
  <h3>✂️ After Cropping — Face ROI Quality vs Expectation</h3>
  <p style="color:#8b949e;font-size:13px;margin-bottom:12px">
    Metrics computed on the <strong>cropped face region only</strong> (same ROI extraction as expectation_face_roi.png).
    This is an apples-to-apples comparison of face quality independent of framing.
  </p>

  <table>
    <tr>
      <th>Metric</th>
      <th>Expectation ROI</th>
      <th>Detected ROI μ</th>
      <th>Detected σ</th>
      <th>Δ (μ−ref)</th>
      <th>% Change</th>
      <th>Distribution</th>
    </tr>
    {cropped_stat_row("Face Brightness", "brightness", ref_brightness)}
    {cropped_stat_row("Face Contrast", "contrast", ref_contrast)}
    {cropped_stat_row("Face Sharpness", "sharpness", ref_sharpness)}
    {cropped_stat_row("Saturation", "saturation", ref_saturation)}
    {cropped_stat_row("Skin L* (lightness)", "skin_lab", ref["skin_lab"][0], ".0f", idx=0)}
    {cropped_stat_row("Skin a* (red-green)", "skin_lab", ref_skin_a, ".0f", idx=1)}
    {cropped_stat_row("Skin b* (yellow-blue)", "skin_lab", ref_skin_b, ".0f", idx=2)}
  </table>

  <div style="margin-top:16px">

  <!-- Key findings cards -->
  <div class="grid">
    <div class="card">
      <div class="card-label">Sharpness Drop</div>
      <div class="card-value red">{abs(csum['avg_deltas']['sharpness_pct']):.0f}%</div>
      <div style="color:#8b949e;font-size:12px;margin-top:4px">
        {csum['expected']['sharpness']:.0f} → {csum['avg_sharpness']:.0f} (video compression / streaming)
      </div>
    </div>
    <div class="card">
      <div class="card-label">Contrast Drop</div>
      <div class="card-value yellow">{abs(csum['avg_deltas']['contrast_pct']):.0f}%</div>
      <div style="color:#8b949e;font-size:12px;margin-top:4px">
        {csum['expected']['contrast']:.0f} → {csum['avg_contrast']:.0f}
      </div>
    </div>
    <div class="card">
      <div class="card-label">Saturation Drop</div>
      <div class="card-value yellow">{abs(csum['avg_deltas']['saturation_pct']):.0f}%</div>
      <div style="color:#8b949e;font-size:12px;margin-top:4px">
        {csum['expected']['saturation']:.0f} → {csum['avg_saturation']:.0f}
      </div>
    </div>
    <div class="card">
      <div class="card-label">Brightness Drop</div>
      <div class="card-value yellow">{abs(csum['avg_deltas']['brightness_pct']):.0f}%</div>
      <div style="color:#8b949e;font-size:12px;margin-top:4px">
        {csum['expected']['brightness']:.0f} → {csum['avg_brightness']:.0f}
      </div>
    </div>
  </div>

  <!-- Status banner -->
  {f"""<div class="status-banner warn">
    ⚠ Face quality degrades significantly from reference expectation after cropping.
    Sharpness drops {abs(csum['avg_deltas']['sharpness_pct']):.0f}% — the video stream is much softer than the reference image.
    Consider sharpening/enhancing the face ROI before downstream processing.
  </div>""" if abs(csum['avg_deltas']['sharpness_pct']) > 50 else ""}
  </div>
</div>

<!-- ═══════════════════ CROPPED ROI GALLERY ═══════════════════ -->
<div class="section">
  <h3>🖼️ Cropped Face ROIs vs Expectation</h3>
  <div class="roi-compare">
    <div class="roi-item">
      <img src="data:image/png;base64,{ref_roi_b64}" alt="Expectation Face ROI">
      <div class="roi-label">Expectation (reference)</div>
    </div>
    {"".join(f'''<div class="roi-item">
      <img src="data:image/jpeg;base64,{cropped_imgs[c['frame_idx']]}" alt="Frame {c['frame_idx']} ROI">
      <div class="roi-label">Frame {c['frame_idx']} ({c['timestamp_sec']:.1f}s)</div>
    </div>''' for c in detected_cropped[:6] if c['frame_idx'] in cropped_imgs)}
  </div>
</div>

<!-- ═══════════════════ CROPPED ROI PER-FRAME ═══════════════════ -->
<div class="section">
  <h3>📋 After Cropping — Per-Face ROI Details</h3>
  <table>
    <tr>
      <th>Frame</th>
      <th>Time</th>
      <th>Brightness</th>
      <th>Contrast</th>
      <th>Sharpness</th>
      <th>Saturation</th>
      <th>Δ Bright</th>
      <th>Δ Sharp</th>
    </tr>
    {''.join(cropped_roi_stats)}
  </table>
  <details style="margin-top:12px">
    <summary style="color:#58a6ff;cursor:pointer;font-size:13px">Show all {len(detected_cropped)} frames →</summary>
    <div style="max-height:400px;overflow-y:auto;margin-top:8px">
    <table>
      <tr>
        <th>Frame</th>
        <th>Time</th>
        <th>Bright</th>
        <th>Contrast</th>
        <th>Sharp</th>
        <th>Sat</th>
        <th>Skin L</th>
        <th>Skin a</th>
        <th>Skin b</th>
        <th>Δ Bright%</th>
        <th>Δ Sharp%</th>
        <th>Δ Sat%</th>
        <th>Light</th>
      </tr>
      {"".join(f'''<tr>
        <td>{c['frame_idx']}</td>
        <td style="color:#8b949e">{c['timestamp_sec']:.1f}s</td>
        <td style="font-family:monospace">{c['roi_metrics']['brightness']:.0f}</td>
        <td style="font-family:monospace">{c['roi_metrics']['contrast']:.0f}</td>
        <td style="font-family:monospace">{c['roi_metrics']['sharpness']:.0f}</td>
        <td style="font-family:monospace">{c['roi_metrics']['saturation']:.0f}</td>
        <td style="font-family:monospace">{c['roi_metrics']['skin_lab'][0]:.0f}</td>
        <td style="font-family:monospace">{c['roi_metrics']['skin_lab'][1]:.0f}</td>
        <td style="font-family:monospace">{c['roi_metrics']['skin_lab'][2]:.0f}</td>
        <td style="font-family:monospace">{c['deltas']['brightness_pct']:+.0f}%</td>
        <td style="font-family:monospace"><span class="delta {'neg' if c['deltas']['sharpness_pct'] < 0 else 'pos'}">{c['deltas']['sharpness_pct']:+.0f}%</span></td>
        <td style="font-family:monospace">{c['deltas']['saturation_pct']:+.0f}%</td>
        <td style="color:#8b949e;font-size:11px">{c['roi_metrics']['light_direction']}</td>
      </tr>''' for c in detected_cropped)}
    </table>
    </div>
  </details>
</div>

<!-- ═══════════════════ ORIGINAL: EXPECTATION VS DETECTION ═══════════════════ -->
<div class="section">
  <h3>🎯 Frame-Level — Expectation vs Detection</h3>
  <table>
    <tr>
      <th>Metric</th>
      <th>Reference</th>
      <th>Detected μ</th>
      <th>Detected σ</th>
      <th>Range</th>
      <th>Δ (μ−ref)</th>
      <th>Distribution</th>
    </tr>
    {stat_row("Face Area (% frame)", face_area_pcts, ref_face_area, "%")}
    {stat_row("Face Brightness", face_brightness, ref_brightness)}
    {stat_row("Face Contrast", face_contrasts, ref_contrast)}
    {stat_row("Face Sharpness", face_sharpness, ref_sharpness)}
    {stat_row("Skin a* (red-green)", skin_a, ref_skin_a)}
    {stat_row("Skin b* (yellow-blue)", skin_b, ref_skin_b)}
    {stat_row("Saturation", saturations, ref_saturation)}
    {stat_row("Detection Confidence", confidences, 1.0)}
  </table>
</div>

<!-- ═══════════════════ REFERENCE FACE ═══════════════════ -->
<div class="section">
  <h3>📸 Reference Face (expectation.png)</h3>
  <div class="ref-image-container">
    <img src="data:image/png;base64,{ref_roi_b64}" alt="Reference Face ROI">
    <div style="margin-top:8px;color:#8b949e;font-size:12px">
      BBox: ({ref['face_bbox'][0]}, {ref['face_bbox'][1]}, {ref['face_bbox'][2]}, {ref['face_bbox'][3]}) ·
      Center: ({ref['face_center_norm'][0]:.3f}, {ref['face_center_norm'][1]:.3f}) ·
      Face: {ref['face_size_pct'][0]:.1f}% × {ref['face_size_pct'][1]:.1f}% of frame ·
      Lighting: {ref['light_direction']} (L/R ratio: {ref['lr_ratio']})
    </div>
  </div>
</div>

<!-- ═══════════════════ FACE POSITION ═══════════════════ -->
<div class="section">
  <h3>📐 Face Position Consistency</h3>
  <table>
    <tr>
      <th>Metric</th>
      <th>Reference</th>
      <th>Detected μ</th>
      <th>Detected σ</th>
      <th>Range</th>
      <th>Δ</th>
    </tr>
    {"".join(f'''<tr>
      <td class="metric-label">{label}</td>
      <td class="ref-val">{ref_val:.3f}</td>
      <td class="det-mean">{sum(get_field(f) for f in detected_frames) / len(detected_frames):.3f}</td>
      <td class="det-std">{__import__('numpy').std([get_field(f) for f in detected_frames]):.3f}</td>
      <td class="det-range">{min(get_field(f) for f in detected_frames):.3f} – {max(get_field(f) for f in detected_frames):.3f}</td>
      <td class="det-delta">{delta(sum(get_field(f) for f in detected_frames) / len(detected_frames), ref_val)}</td>
    </tr>''' for label, get_field, ref_val in [
      ("Face Center X (norm)", lambda f: f["face_center_norm"][0], ref["face_center_norm"][0]),
      ("Face Center Y (norm)", lambda f: f["face_center_norm"][1], ref["face_center_norm"][1]),
      ("Face Top (norm)", lambda f: f["face_top_norm"], ref["face_top_norm"]),
      ("Face Width %", lambda f: f["face_size_pct"][0], ref["face_size_pct"][0]),
      ("Face Height %", lambda f: f["face_size_pct"][1], ref["face_size_pct"][1]),
    ])}
  </table>
</div>

<!-- ═══════════════════ PER-FRAME TABLE ═══════════════════ -->
<div class="section">
  <h3>📋 Per-Frame Detection Details</h3>
  <details>
    <summary style="color:#58a6ff;cursor:pointer;font-size:13px">Show all {len(frames)} frames →</summary>
    <div style="margin-top:8px;max-height:500px;overflow-y:auto">
    <table>
      <tr>
        <th>Frame</th>
        <th>Time</th>
        <th>Face</th>
        <th>BBox</th>
        <th>Center (norm)</th>
        <th>Area%</th>
        <th>Conf</th>
        <th>Bright</th>
        <th>Sharp</th>
        <th>Host</th>
      </tr>
      {"".join(f'''<tr>
        <td>{f['frame_idx']}</td>
        <td style="color:#8b949e">{f['timestamp_sec']:.1f}s</td>
        <td>{"<span class='badge green'>✔</span>" if f['face_detected'] else "<span class='badge red'>✘</span>"}</td>
        <td style="font-family:monospace;font-size:11px;color:#8b949e">{f['face_bbox'][0]},{f['face_bbox'][1]} {f['face_bbox'][2]}×{f['face_bbox'][3]}</td>
        <td style="font-family:monospace;font-size:11px">{f['face_center_norm'][0]:.2f},{f['face_center_norm'][1]:.2f}</td>
        <td style="font-family:monospace">{f['face_area_pct']:.1f}%</td>
        <td style="font-family:monospace">{f['confidence']:.2f}</td>
        <td style="font-family:monospace">{f['brightness']:.0f}</td>
        <td style="font-family:monospace">{f['sharpness']:.0f}</td>
        <td style="font-family:monospace">{f.get('host_match_confidence',0):.2f}</td>
      </tr>''' for f in frames)}
    </table>
    </div>
  </details>
</div>

<!-- ═══════════════════ SAMPLED FRAMES ═══════════════════ -->
<div class="section">
  <h3>🖼️ Sampled Frames</h3>
  <div class="frame-grid">
    {"".join(f'''<div class="frame-card">
      <img src="data:image/jpeg;base64,{frame_imgs[f['frame_idx']]}" alt="Frame {f['frame_idx']}">
      <div class="frame-meta">
        <div class="row"><span class="label">Frame</span><span class="val">{f['frame_idx']}</span></div>
        <div class="row"><span class="label">Time</span><span class="val">{f['timestamp_sec']:.1f}s</span></div>
        <div class="row"><span class="label">Face</span><span class="val">{'✔' if f['face_detected'] else '✘'} (×{f['num_faces']})</span></div>
        <div class="row"><span class="label">Area</span><span class="val">{f['face_area_pct']:.1f}%</span></div>
        <div class="row"><span class="label">Confidence</span><span class="val">{f['confidence']:.2f}</span></div>
      </div>
    </div>''' for f in frames if f['frame_idx'] in frame_imgs)}
  </div>
</div>

<!-- ═══════════════════ QUALITY GAUGE ═══════════════════ -->
<div class="section">
  <h3>📊 Quality Gauge (Detected vs Reference Range)</h3>
  <table>
    <tr><th>Metric</th><th>Detected μ</th><th>Reference</th><th>Acceptable Range</th><th>Gauge</th></tr>
    <tr>
      <td class="metric-label">Face Brightness</td>
      <td class="det-mean">{sum(face_brightness)/len(face_brightness):.0f}</td>
      <td class="ref-val">{ref_brightness:.0f}</td>
      <td style="color:#8b949e">{ref_brightness-15:.0f} – {ref_brightness+15:.0f}</td>
      <td style="width:200px">{gauge(sum(face_brightness)/len(face_brightness), ref_brightness-15, ref_brightness+15)}</td>
    </tr>
    <tr>
      <td class="metric-label">Face Contrast</td>
      <td class="det-mean">{sum(face_contrasts)/len(face_contrasts):.0f}</td>
      <td class="ref-val">{ref_contrast:.0f}</td>
      <td style="color:#8b949e">{ref_contrast-10:.0f} – {ref_contrast+10:.0f}</td>
      <td style="width:200px">{gauge(sum(face_contrasts)/len(face_contrasts), ref_contrast-10, ref_contrast+10)}</td>
    </tr>
    <tr>
      <td class="metric-label">Face Sharpness</td>
      <td class="det-mean">{sum(face_sharpness)/len(face_sharpness):.0f}</td>
      <td class="ref-val">{ref_sharpness:.0f}</td>
      <td style="color:#8b949e">{ref_sharpness*0.3:.0f} – {ref_sharpness*1.5:.0f}</td>
      <td style="width:200px">{gauge(sum(face_sharpness)/len(face_sharpness), ref_sharpness*0.3, ref_sharpness*1.5)}</td>
    </tr>
    <tr>
      <td class="metric-label">Saturation</td>
      <td class="det-mean">{sum(saturations)/len(saturations):.0f}</td>
      <td class="ref-val">{ref_saturation:.0f}</td>
      <td style="color:#8b949e">{ref_saturation-20:.0f} – {ref_saturation+20:.0f}</td>
      <td style="width:200px">{gauge(sum(saturations)/len(saturations), ref_saturation-20, ref_saturation+20)}</td>
    </tr>
    <tr>
      <td class="metric-label">Skin a*</td>
      <td class="det-mean">{sum(skin_a)/len(skin_a):.0f}</td>
      <td class="ref-val">{ref_skin_a:.0f}</td>
      <td style="color:#8b949e">{ref_skin_a-5:.0f} – {ref_skin_a+5:.0f}</td>
      <td style="width:200px">{gauge(sum(skin_a)/len(skin_a), ref_skin_a-5, ref_skin_a+5)}</td>
    </tr>
    <tr>
      <td class="metric-label">Skin b*</td>
      <td class="det-mean">{sum(skin_b)/len(skin_b):.0f}</td>
      <td class="ref-val">{ref_skin_b:.0f}</td>
      <td style="color:#8b949e">{ref_skin_b-5:.0f} – {ref_skin_b+5:.0f}</td>
      <td style="width:200px">{gauge(sum(skin_b)/len(skin_b), ref_skin_b-5, ref_skin_b+5)}</td>
    </tr>
  </table>
</div>

<div class="footer">
  Generated {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')} ·
  DNN Face Detection (OpenCV ResNet-10 SSD) · {summary['samples']} random samples ·
  Face ROIs cropped and compared against expectation_face_roi.png
</div>

</div>
</body>
</html>"""

out_path = REPORTS / "face_detection_report.html"
out_path.parent.mkdir(parents=True, exist_ok=True)
Path(out_path).write_text(html)
print(f"Report saved: {out_path}")
print(f"Size: {len(html)/1024:.0f} KB")
