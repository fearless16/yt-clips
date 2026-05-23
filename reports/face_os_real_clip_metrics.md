# Face OS Real Clip Metrics

**Input:** `clips_test/test_clip.mp4`
**Output:** `output/face_os/test_clip_validated.mp4`
**Frames processed:** 345 / 345
**Source:** 640x360 @ 30.00 fps
**Processing time:** 14.57s (23.68 fps)
**Pipeline failures:** 0

## Verdict

**Project completed by target metrics:** NO

| Gate | Target | Actual | Pass |
|---|---:|---:|:---:|
| Sharpness | >= 274.0 | 2.32 | NO |
| Flicker | < 1.0 | 0.65 | YES |
| Contrast | >= 73.0 | 51.94 | NO |
| Runtime | 0 failures | 0 failures | YES |
| Per-frame telemetry | 345 rows | 345 rows | YES |

## Signal Metrics

| Metric | Input mean | Output mean | Output/Input | Output min | Output max |
|---|---:|---:|---:|---:|---:|
| Sharpness, Laplacian variance | 611.56 | 2.32 | 0.004 | 1.65 | 3.01 |
| High-frequency energy | 134.56 | 2.40 | 0.018 | 0.67 | 4.41 |
| Contrast, grayscale std | 57.64 | 51.94 | 0.901 | 40.20 | 59.98 |
| Luminance mean | 78.70 | 114.31 | 1.452 | 102.71 | 130.97 |
| LAB flicker | n/a | 0.65 | n/a | 0.00 | 7.98 |
| Output inter-frame delta | n/a | 6.95 | n/a | 0.00 | 22.32 |

## Runtime Telemetry

- Render paths: `{'enhancement': 345}`
- Geometry sources: `{'none': 345}`
- Resample counts: `{0: 345}`
- Fallback reasons: `{'face_lost': 345}`
- Physical render rate: `0.0000`
- Alpha fallback rate: `0.0000`
- Intrinsic success rate: `0.0000`
- Average intrinsic confidence: `0.0000`
- Average decomposition error: `0.0000`
- Mesh normal rate: `0.0000`
- Shading normal rate: `0.0000`
- Renderer mode transitions: `0`
- Transform determinant mean/std: `1.0000` / `0.0000`

## Completion Assessment

The project should only be called complete when runtime succeeds and the measured visual gates pass on real clips.
This report uses the explicit D-01 targets supplied for sharpness, flicker, and contrast.
