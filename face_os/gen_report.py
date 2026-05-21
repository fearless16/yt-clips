"""Generate HTML visibility report with frame comparisons."""
import base64
import json
import os

frames_dir = 'output/face_os/visibility/frames'
vis_dir = 'output/face_os/visibility'

with open(f'{vis_dir}/summary.json') as f:
    summary = json.load(f)

with open(f'{vis_dir}/energy_reports.json') as f:
    energy_reports = json.load(f)

html = f'''<!DOCTYPE html>
<html>
<head>
<title>Face OS V2.3.0 — Visibility Report</title>
<style>
body {{ font-family: monospace; background: #1a1a1a; color: #e0e0e0; padding: 20px; }}
h1 {{ color: #00ff88; }}
h2 {{ color: #00aaff; border-bottom: 1px solid #333; padding-bottom: 5px; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
th, td {{ border: 1px solid #444; padding: 8px; text-align: left; }}
th {{ background: #2a2a2a; color: #00aaff; }}
.energy-bar {{ display: inline-block; height: 16px; background: #00ff88; margin-right: 5px; }}
.frame-row {{ display: flex; gap: 10px; margin: 20px 0; }}
.frame-col {{ flex: 1; text-align: center; }}
.frame-col img {{ width: 100%; max-width: 300px; border: 2px solid #444; }}
.metric {{ color: #00ff88; }}
.section {{ background: #222; padding: 15px; margin: 10px 0; border-radius: 5px; }}
</style>
</head>
<body>
<h1>Face OS V2.3.0 — Full Parameter Visibility Report</h1>
<p>Total Frames: {summary['total_frames']} | Processing: {summary['processing_time_s']:.1f}s ({summary['fps']:.1f} fps)</p>

<h2>Energy Terms (mean +/- std)</h2>
<div class="section">
<table>
<tr><th>Term</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th><th>Visual</th></tr>
'''

for term in ['E_geom', 'E_identity', 'E_temporal', 'E_photometric', 'E_smoothness', 'E_total']:
    es = summary['energy_summary'][term]
    bar_width = int(es['mean'] * 3)
    html += f'<tr><td>{term}</td><td class="metric">{es["mean"]:.3f}</td><td>{es["std"]:.3f}</td><td>{es["min"]:.3f}</td><td>{es["max"]:.3f}</td><td><span class="energy-bar" style="width:{bar_width}px"></span></td></tr>\n'

html += '''</table>
</div>

<h2>Geometry Metrics (Frame 0)</h2>
<div class="section">
<table>
<tr><th>Metric</th><th>Value</th></tr>
'''

g = energy_reports[0]['geometry']
for key, val in g.items():
    html += f'<tr><td>{key}</td><td class="metric">{val}</td></tr>\n'

html += '''</table>
</div>

<h2>Identity Metrics (Frame 0)</h2>
<div class="section">
<table>
<tr><th>Metric</th><th>Value</th></tr>
'''

i = energy_reports[0]['identity']
for key, val in i.items():
    html += f'<tr><td>{key}</td><td class="metric">{val}</td></tr>\n'

html += '''</table>
</div>

<h2>Temporal Metrics (Frame 0)</h2>
<div class="section">
<table>
<tr><th>Metric</th><th>Value</th></tr>
'''

t = energy_reports[0]['temporal']
for key, val in t.items():
    html += f'<tr><td>{key}</td><td class="metric">{val}</td></tr>\n'

html += '''</table>
</div>

<h2>Renderer Metrics (Frame 0)</h2>
<div class="section">
<table>
<tr><th>Metric</th><th>Value</th></tr>
'''

r = energy_reports[0]['renderer']
for key, val in r.items():
    html += f'<tr><td>{key}</td><td class="metric">{val}</td></tr>\n'

html += '''</table>
</div>

<h2>Frame Comparison</h2>
'''

for frame_num in [0, 30, 49]:
    html += f'<h3>Frame {frame_num:04d}</h3>\n'
    html += '<div class="frame-row">\n'
    for suffix in ['source', 'identity', 'output']:
        filepath = f'{frames_dir}/frame_{frame_num:04d}_{suffix}.png'
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode()
            html += f'<div class="frame-col"><p>{suffix}</p><img src="data:image/png;base64,{img_data}"></div>\n'
    html += '</div>\n'

html += '''
<h2>Energy per Frame (first 10)</h2>
<div class="section">
<table>
<tr><th>Frame</th><th>E_geom</th><th>E_identity</th><th>E_temporal</th><th>E_photometric</th><th>E_total</th></tr>
'''

for r in energy_reports[0:10]:
    et = r['energy_terms']
    html += f'<tr><td>{r["frame_idx"]}</td><td>{et["E_geom"]:.2f}</td><td>{et["E_identity"]:.2f}</td><td>{et["E_temporal"]:.2f}</td><td>{et["E_photometric"]:.2f}</td><td class="metric">{et["E_total"]:.2f}</td></tr>\n'

html += '''</table>
</div>

</body>
</html>
'''

output_path = f'{vis_dir}/visibility_report.html'
with open(output_path, 'w') as f:
    f.write(html)

print(f'HTML report saved to: {output_path}')
