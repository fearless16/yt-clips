# Portrait reference — visual-only repair, 2026-11-04

Requested visual: full-width 9:16 host portrait, bold white captions in a rounded
dark translucent rectangle, below the chin. Not a tiny landscape camera panel
floating inside a blurred canvas, and not angular cyan captions over the face.

## Rendering contract

- `export.composition: portrait-caption-v1` uses actual source dimensions and
  the detected host's sample envelope, not just its highest-confidence frame.
- Dominant hosts take the portrait route even if screen-share classification
  was contradictory. A genuinely small presentation facecam is not overridden.
- Sharp video spans all 1080 pixels; the crop remains fixed across the clip.
- `automation/portrait_layout.py` shares the caption rectangle with export.
  At 1080×1920 its visible bounds are x=98, y=1484, width=884, height=224.
  The sampled chin stays at least 64 pixels above it. Hair already missing from
  the source cannot be recovered by cropping. Source movement between samples
  still requires visual review.
- `portrait-glass-v1` uses local font files, browser-measured whole-word lines,
  white text and rounded dark glass. It never shrinks type or drops words to
  disguise overflow. Legacy neon is still available explicitly.
- This particular source ends near the neck. The lower continuation uses
  blurred source pixels, **not a generated torso**. It cannot exactly reproduce
  the supplied photograph's chest/shoulder coverage.

## Actual local preview

`shorts/portrait_review_2026-11-04_155957_a1dcd532/portrait.mp4`

Source range 344.92–362.68 seconds, unchanged. Actual ffprobe: 1080×1920,
17.766667 seconds video / 17.760000 seconds audio; both start at zero. Full
decode and encoded A/V timing checks passed. Four encoded frames were inspected;
18 source-face samples set a fixed crop and caption clearance. This does not
claim full listening or phone playback. Owner visual approval was subsequently
received; see the scoped approval below.

The explicitly reviewed camera bounds exclude the stream header/chat. A small
source `CONTROLS` label on the neck was interpolated out using the recorded
source rectangle, not inferred face cleanup. Original source and draft files
were not changed. The manifest records all geometry, cleanup and the full graph.

**Cut points and cached caption wording are unchanged.** Existing ASR mistakes
and editorial/metadata problems are not fixed by this visual pass. Release
approval remains null. No paid provider calls, pipeline reruns or uploads were used.

### Owner visual approval

The owner accepted this exact preview on 2026-11-04: "this is much better lets
move forward with this one...". Its SHA-256 and the approved visual scope are
recorded separately in `visual_approval.json` beside the preview, preserving the
original `review.json`. This is **not** caption-fidelity, release or upload approval.
The original video remains unchanged. Two local listening excerpts,
`caption-check-1.mp4` and `caption-check-2.mp4`, isolate unresolved caption phrases
without guessing replacement speech. Metadata still requires correction before
release; the old India senior-team framing and unsupported Siraj terms are not
approved by the owner's visual acceptance.

Reproduce locally without any AI/network call:

```powershell
.venv\Scripts\python.exe -m tools.portrait_review shorts/2026-11-04_151858_d9bf65c0/clip1_edit.json --camera-bounds 764 0 1120 1440 --delogo 1194 1396 172 28
```

The tool writes a new directory; it does not overwrite previous media, change
words/cuts, generate SEO, or create a release approval.

## Failure-first checks

`tests/test_portrait_reference.py` first reproduced the recorded screen-share
misrouting, absent caption-safe geometry and old caption design. A further red
test reproduced trimming away the worst sampled chin. The suite includes real
Playwright card rasters and real FFmpeg output-pixel assertions, not only graph
labels or counters. The dry run now supplies decoded-frame/host-model fixtures,
distinguishes landscape source dimensions from portrait output, and independently
requires actual portrait provenance. Restoring the old layout makes acceptance
fail even when three media files are returned.

Validation logs: `logs/portrait-pytest.log`, `logs/portrait-dry-run.log`.
Passing automation remains a technical check, **not visual approval**.
