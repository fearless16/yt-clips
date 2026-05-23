# Face OS Gap Files From Real Clip Validation

Input clip: `clips_test/test_clip.mp4`

Latest full run report: `reports/face_os_real_clip_metrics.md`

## Fixed In This Pass

| File | Gap | Fix |
|---|---|---|
| `face_os/canonical_map.py` | `IdentityProfile.embeddings` was never populated during enrollment, so `FaceTracker` could not create target tracks. | Build LAB-histogram embeddings from reference detections during `build_identity_profile()`. |
| `face_os/pipeline.py` | Public `process_frame()` did not increment `total_frames`, so telemetry rates were zero during API-based validation. | Increment `total_frames` in `process_frame()`. |
| `face_os/physical_renderer.py` | Mesh-normal rasterization ran at full output resolution, making real physical frames ~13.7s each. | Allow bounded mesh-normal rastering and resize/renormalize normals to output resolution. |
| `face_os/pipeline.py` | Physical path requested full-resolution mesh normal rastering. | Use `_normal_raster_shape()` to cap the normal raster long side at 384px. |
| `tools/run_face_os_clip_metrics.py` | Validation had no progress output or run budget controls. | Add progress logging plus `--max-frames` and `--max-seconds`. |

## Still Not Up To Mark

| File | Remaining gap | Evidence |
|---|---|---|
| `face_os/face_enhance.py` | Post-composite sharpening does not restore enough high-frequency detail after 640x360 to 1080x1920 crop/render output. | Full run sharpness is `2.59` vs target `>=274`; HF energy output/input is `0.012`. |
| `face_os/pipeline.py` | Physical render path still produces a softened output stack despite activating on most frames. | Full run: `315` physical frames, `mesh_normal_rate=1.0`, but sharpness/contrast gates still fail. |
| `face_os/crop_planner.py` | Output scaling from 360p source to 1920p portrait strongly suppresses Laplacian sharpness; no scale-aware restoration exists. | Input sharpness `611.56`; output sharpness `2.59`. |
| `face_os/detect_track.py` | Tracking briefly loses the face mid-clip and relies on recovery rather than predictive continuity. | Full run has `26` `face_lost` enhancement frames. |
| `face_os/physical_renderer.py` | Physical rendering is now much faster but still slow for full-clip validation on laptop CPU. | Full run: `547.46s` for `345` frames, `0.63 fps`. |

## Current Verdict

The project is not complete by the stated real-clip gates. Runtime activation is substantially better than the previous all-lost run, but visual fidelity is still below target.
