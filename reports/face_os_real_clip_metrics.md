# Face OS Real Clip Metrics

**Input:** `clips_test/test_clip.mp4`
**Output:** `output/face_os/test_clip_validated.mp4`
**Frames processed:** 345 / 345
**Stopped by max frames:** NO
**Stopped by max seconds:** NO
**Source:** 640x360 @ 30.00 fps
**Processing time:** 547.46s (0.63 fps)
**Pipeline failures:** 0

## Verdict

**Project completed by target metrics:** NO

| Gate | Target | Actual | Pass |
|---|---:|---:|:---:|
| Sharpness | >= 274.0 | 2.59 | NO |
| Flicker | < 1.0 | 0.95 | YES |
| Contrast | >= 73.0 | 56.93 | NO |
| Runtime | 0 failures | 0 failures | YES |
| Per-frame telemetry | 345 rows | 345 rows | YES |

## Signal Metrics

| Metric | Input mean | Output mean | Output/Input | Output min | Output max |
|---|---:|---:|---:|---:|---:|
| Sharpness, Laplacian variance | 611.56 | 2.59 | 0.004 | 2.03 | 4.08 |
| High-frequency energy | 134.56 | 1.64 | 0.012 | 0.62 | 3.99 |
| Contrast, grayscale std | 57.64 | 56.93 | 0.988 | 39.55 | 65.32 |
| Luminance mean | 78.70 | 106.40 | 1.352 | 96.68 | 125.59 |
| LAB flicker | n/a | 0.95 | n/a | 0.03 | 12.09 |
| Output inter-frame delta | n/a | 8.18 | n/a | 0.08 | 27.85 |

## Runtime Telemetry

- Render paths: `{'alpha': 4, 'physical': 315, 'enhancement': 26}`
- Geometry sources: `{'canonical_identity': 4, 'mesh': 315, 'none': 26}`
- Resample counts: `{1: 319, 0: 26}`
- Fallback reasons: `{'renderer_mode_alpha': 4, 'face_lost': 26}`
- Physical render rate: `0.9130`
- Alpha fallback rate: `0.0116`
- Intrinsic success rate: `0.9246`
- Average intrinsic confidence: `0.7576`
- Average decomposition error: `0.0335`
- Mesh normal rate: `1.0000`
- Shading normal rate: `0.0000`
- Renderer mode transitions: `1`
- Transform determinant mean/std: `6.8137` / `2.6198`

## Completion Assessment

The project should only be called complete when runtime succeeds and the measured visual gates pass on real clips.
This report uses the explicit D-01 targets supplied for sharpness, flicker, and contrast.
