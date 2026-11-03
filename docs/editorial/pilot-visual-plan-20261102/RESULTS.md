# Aurora implementation result — 2026-11-02

**Visual rebuild implemented and rendered. Release remains REVIEW, not READY.**
This file supersedes the historical proposed-status ledger in `PLAN.md`.

## Open these actual files

- Video: `shorts/aurora-pilot-20261102/aurora-review.mp4`
- Thumbnail: `shorts/aurora-pilot-20261102/aurora-review_thumb.jpg`
- Final frame sheet: `shorts/aurora-pilot-20261102/review/video-contact-sheet.jpg`
- Phone-sized still: `shorts/aurora-pilot-20261102/review/phone-08.0s.jpg`
- Identity/QC/review ledger: `shorts/aurora-pilot-20261102/artifacts.json`
- Exact cleanup inventory: `shorts/aurora-pilot-20261102/cleanup.json`

These are local artifacts, not delivered YouTube assets or publication approval.

## Implemented and checked

| Area | Actual result | Verification limit |
|---|---|---|
| 9:16 composition | 1080x1920, 30fps; source84.22–124.28s, 1.0x; 40.066667s encoded | estimated source boundaries, not listening-approved |
| Camera framing | x450/y0/w960/h1040; fit to ~920x998 at y210; face scale ~0.96 vs old~1.03 | final 12-frame sheet and selected full-size frames inspected; not full motion review |
| Head turn | earlier narrow crop rejected; widened crop keeps hair in the inspected21s frame | no assertion every frame/gesture is perfect |
| Source UI | right chat and lower controls excluded; small upper-left badge locally blurred in x0/y0/w136/h94 | explicit visible non-face blur, not clean-camera footage or invented scenery |
| Background | same-time moving camera source, full-canvas cover fill, sigma64 blur with restrained darkening/blue tint; no solid caption band | graph/readback and differing sampled frames, not subjective full-playback approval |
| Captions | new Aurora renderer, OFL Oxanium800, midnight glass, pale border/cyan edge, stable top anchor | all37 actual cards measured at68px, max capsule637.16x199.88px; no font downsize needed |
| Placement | visible card top1248; about40px below foreground | phone-sized still inspected; actual YouTube UI/device not checked |
| Tail grouping | final `jeetne ki koshish` then `karte hain`, preserving original intervals | avoids a separate0.3s `hain` card; does not certify speech or whole-clip pacing |
| Hinglish | real API request `deepseek-v4-flash`, actual response `deepseek-flash` | original V4 retired; documented V4.1 successor disclosed, not relabeled V4 |
| Re-render provenance | later geometry/grouping renders replay unchanged Flash words after exact source text/start/end equality checks | no new model call, fallback, fabricated timestamps or human correction claimed |
| Thumbnail | real113.6025s questioning expression; `PEHLE OVER MEIN?`; matching geometric/cyan glass identity | 12 real candidate frames compared; full/120px/360px and diagnostic square viewed, no CTR claim |
| Cleanup |100 superseded/scratch files deleted,94.42MiB;68.81MiB of superseded canonical content replaced | exact paths/sizes/hashes in cleanup.json; source/review/corrections retained |

### Fresh identity pins

```text
source MP4  b95b1bfd57a270066b6e63ccd3b2c8baec041fa614ff60e04925ff8e9eb603d5
final MP4   924d210a6372a60340498724054fe9cf9938aca90d8dcc9630b3f3bbcf32e06e
final cover 1b6bd1cdf46052ae29e341e90c9c3458893f948f4ad2c0bdf8af7f0cb537f487
font        2ce01d946e1e1ffc8d7eecfffbda8623bedd63eaf811a20488c4b69af45babb0
```

Font and SIL OFL license are bundled in `assets/fonts/oxanium/`. No remote font
load is needed to reproduce a caption. The old plain-caption CSS is removed;
legacy diagnostic text helpers remain for compatibility, not as the active
Flash-caption renderer/fallback.

## Validation actually performed

- TDD failures reproduced before implementation: exact routing/word mapping,
  renderer/blur layout, cover composition, dry-run Pillow compatibility,
  scoped badge blur and orphan ending auxiliary. Corresponding tests then passed.
- Final mandatory suite: **1,013 passed,47 skipped,18 xfailed**. Existing skip/xfail
  counts unchanged; no new skips or expected failures used to hide this work.
- Final required dry run: **PASS**,2 selected/exported,2 SEO outputs,0 failures,
  0 uploads. Thumbnail composition no longer raises stale Pillow-stub errors.
  Dry-run sandbox removal was checked. Its stubbed images are not visual evidence.
- Actual final encoded QC: **PASS**, full video/audio decode, expected duration,
  1080x1920, encoded A/V start/end timing. Audio: **-16.23LUFS / -2.18dBTP**.
  Measurement is not listening or semantic/caption synchronization verification.
- Final visual evidence actually viewed:12-frame contact sheet; full21s head-turn
  frame;360px8s and40s frames; cover full-size,120px,360px and centered-square120px.
- Source hash freshly rechecked and unchanged. No changes to Face OS, Gaming Clip
  Studio, learning ownership or release-approval mechanism. Nothing committed/pushed.

## Unfinished gates — do not report these as completed

- **Speech fidelity is not solved.** Eight model-flagged source tokens remain,
  including literal `hukhaas`, `rante`, `vaan`, `trant` and inferred spellings.
  The disputed `11-100` source number was preserved, not silently changed to a
  plausible score/run rate. Source ASR is not ground truth.
- Actual listening corrections, reviewed names/numbers/negations and any needed
  corrected-text alignment remain required. Joined Roman forms share the source
  token interval; no individual times for invented subwords are claimed.
- Whole export playback with sound, actual phone/YouTube UI review and named
  owner approval have **not** occurred. Full decode and stills do not replace them.
- The extended ending retains the ASR first-over argument, but semantic
  completeness/clean phonetic boundaries are not listening-certified. Whole-source
  strongest-moment/alternative selection is not certified by this visual revision.
- Thumbnail copy is editorial, not a verbatim quote. Its first-over question must
  be checked against the final corrected cut; a new cut invalidates that review.
- The direct API cannot supply original retired V4 Flash. Official mapping source:
  https://api-docs.deepseek.com/quick_start/pricing (inspected2026-11-02). Both model
  IDs and successor disclosure are stored in the real caption evidence.
- No upload, schedule, publication, real audience outcome or performance claim.

## Resumption rules

Read `PLAN.md`, this file and the current `artifacts.json`. Preserve the new visual
system. Next quality work is **listening correction + supported alignment**, not
another restyle or more ASR guesses. Re-render changed speech, re-run QC, invalidate
old hashes and obtain actual playback/phone/owner review before READY. Retain the
existing listening packet and original transcript provenance. Old generated MP4,
thumbnail and12 candidate JPGs were intentionally removed; their old JSON paths
are historical evidence, not existing files. Do not claim they can still be opened.
