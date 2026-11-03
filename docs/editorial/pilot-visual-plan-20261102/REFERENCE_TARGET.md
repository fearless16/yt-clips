# Creator reference — governing visual target

Date: 2026-11-02. **Supersedes the earlier Aurora look and the 4:5-panel proposal.**

**Latest scope update:** the creator subsequently authorized a best-effort crop
with a dedicated caption area. The new `review/packaging-v3/` review cut follows
that source-limited compromise; missing wider footage no longer blocks producing
this review. It is not a faithful torso match or publication approval. See the
latest section below; earlier “no replacement render” statements are historical.

**Latest package:** `review/packaging-v4/` refines the authorized compromise with
a larger opening topic headline, proven actual-frame cover options and a
listening-first opening comparison. It preserves the v3 crop and approved caption
cards. See the final section; v3 below remains preserved evidence.

User: “this is the ideal … check this out properly and make sure we come near”.

## Reference identity and inspection

- Original: `C:/Users/user/Pictures/Screenshots/Screenshot 2026-11-02 145010.png`
- Byte-identical retained copy:
  `shorts/aurora-pilot-20261102/review/reference-match/creator-reference.png`
- SHA-256: `33985375564e79b3c50153bde0d23bfa4dc525719a885af7d7811b0fbfd825c3`
- Actual dimensions: **611x910**, approximately **2:3**, not 9:16. Keep the requested
  1080x1920 delivery unless the creator explicitly changes it. Do not stretch the
  screenshot or assume its normalized layout transfers unchanged to a taller frame.
- Inspected the entire reference, enlarged caption crop, new caption comparison,
  and new 360px caption proof. These are static image checks only.

## What needs matching

1. One continuous, sharp portrait scene: entire hair/head, shoulders and substantial
   shirt/chest coverage. No floating video window or separately visible blur slab.
2. Approximate reference proportions: hair begins about4% from the top; chin is
   around56% down; head height is about52% of the image. These are visual estimates,
   not face-detector measurements or audience-preference statistics.
3. Compact caption across the lower shirt, below the face. Reference graphic is
   roughly80% of image width and13–14% of height. Its very low placement in this
   screenshot is **not** certified Shorts UI safety; test a real app/device preview.
4. Slanted heavy uppercase lettering: white upper line, luminous cyan lower line.
   Dark translucent stepped/angular panel, double cyan contours, clipped corners,
   short diagonal accents and restrained glow. Not the old rounded glass capsule.
5. Preserve actual Roman Hinglish words and timing. The reference's “AUR AB / YEH
   PLAN HAI…” is a style example, not permission to insert those unspoken words.
6. The reference is brighter/crisper than this source. Do not call synthetic detail,
   fabricated torso or excessive sharpening a faithful restoration.

## Source constraint established, not guessed

Both `input/video.mp4` and `input/recovered/nrl5qQD0E98.mp4` were freshly hashed;
both equal `b95b1bfd57a270066b6e63ccd3b2c8baec041fa614ff60e04925ff8e9eb603d5`.
There is no separate wider camera recording among the usable input video files.

The selected84.22–124.28s footage is already a close-up in a baked landscape live
layout. At source113.6025s, hair-to-chin occupies roughly85% of the usable1040px
camera height. The screenshot's neck/chest coverage simply is not present below
that camera frame. Cropping/zooming cannot produce the missing body.

The stable full-width1170px-high experiment actually gets close to the reference's
head height relative to a1920px delivery canvas, but ends the useful picture at
61% of canvas height. Extending it with blur fails the continuous torso requirement.
Making the close-up fill the entire9:16 canvas instead cuts lateral head coverage
and increases face size. **Do not silently select either failure as a match.**

Needed for a faithful overall composition: the clean/wider camera recording of
this performance, with shoulders/chest in frame. Otherwise a source-limited
composition compromise must be explicitly acknowledged by the creator. No fake
shirt/torso, generative outpainting, face synthesis or aspect-ratio distortion.

## Work completed after receiving the reference

Generated a separate **caption-style proof**, not a finished-video replacement:

- `shorts/aurora-pilot-20261102/review/reference-match/caption-style-comparison.jpg`
- `shorts/aurora-pilot-20261102/review/reference-match/caption-proof.png` (transparent)
- `shorts/aurora-pilot-20261102/review/reference-match/caption-proof-360w.jpg`
- `shorts/aurora-pilot-20261102/review/reference-match/caption-proof.html`
- `shorts/aurora-pilot-20261102/review/reference-match/reference-manifest.json`

The proof uses existing cue7, “jis rate pe India”, without inventing speech or
claiming corrected alignment. Bundled Oxanium with browser-synthesized italic
approximates the reference; exact reference font identity is unknown. Browser
font-loaded/no-overflow checks passed. Inspected the full comparison and360px
proof, and removed an unwanted crossing stroke found in the first inspection.
At the initial proof stage, production caption code was unchanged. The approved
caption implementation below now supersedes that prototype-only status; framing
remains a separate unresolved choice.

Latest creator feedback: the caption text touches the top; lower it slightly,
with the rest looking good. Lowered both proof text lines by10px (top56→66 and
143→153), leaving the frame, font, colors, words and horizontal geometry unchanged.
Re-rendered and inspected the full comparison and360px proof. Browser checks
confirm exactly10px vertical movement, unchanged remaining text geometry, loaded
font, no overflow and no browser errors. Prior proof retained in
`review/reference-match/revisions/before-text-lowered/`; current hashes refreshed.
This feedback accepts the style direction subject to that adjustment, not the
unrendered whole-video composition or unresolved speech/release gates.

The earlier12s tracked4:5 experiment finished before reconciliation of this new
brief. It is1080x1920/30fps,360 frames,12.000s picture/audio, and full A/V decode
passed. Eight extracted frames were inspected. It is **superseded as a target**;
these checks do not establish playback, listening approval, aesthetic acceptance
or conformity to the newly supplied reference. Its manifest/commands remain as
historical evidence, not a recommended final edit.

Canonical MP4 remains unchanged:
`924d210a6372a60340498724054fe9cf9938aca90d8dcc9630b3f3bbcf32e06e`.
Status remains REVIEW with creator-rejected composition and unresolved speech/
phone/owner-approval gates. No full replacement render, upload, commit or push.

## Decision and verification gates

- Caption treatment: **implemented after style approval**, high confidence in observable visual direction;
  no performance prediction. Risk: visual noise on long phrases. Test every actual
  aligned cue for fit/readability; do not reduce words into invented timestamps.
- Composition: **blocked for faithful match on current source**. High confidence
  in missing-coverage diagnosis; wider recording is the concrete prerequisite.
- Expected impact: closer visual fidelity, not a measured retention gain.
- Rollback: canonical media untouched; new caption assets remain separate and the
  old cover keeps its own version. Production caption implementation changed only
  after style approval; no accepted whole-video composition is implied.
- Before release: actual source movement/cut review, corrected speech/alignment,
  finished playback with sound, real phone/Shorts UI check, final hashes and named
  creator approval. Initial tests/dry run were historical; fresh results follow.

## Approved caption implementation — after “works.. continue”

**Completed: renderer integration and all37 real pilot caption assets. Not a
new finished-video export or approval of the rejected crop.**

- `automation/caption_design.py`: `neon-reference-v2`, same approved stepped cyan
  panel, white/cyan italic72px text and10px-lowered baselines. Header width is
  measured from the actual bundled font; long headers widen instead of shrinking
  type. Upper decorative accents are hidden only where an expanded header would
  intersect them. Single-word cues do not invent a second line. Source text is
  HTML-escaped and template replacement is single-pass.
- `automation/caption_engine.py`: uses the new browser measurement, records actual
  frame and per-line bounds, and fails closed on overflow. Existing source cue
  intervals and word grouping are untouched; no new ASR/model requests occurred.
- `automation/cover_design.py`: explicitly retains its own `aurora-v1` identity.
  The old rounded cover is **not** falsely relabeled as the new caption treatment;
  a matching revised cover remains pending the final composition.
- `export.py`: caption log messages no longer mislabel new cards as Aurora.
  Production camera framing/configuration is unchanged and still not accepted.
- `tests/test_futuristic_captions.py`: approved font/offset geometry, word/order
  preservation, escaped content/template-token safety, wider headers, single-word
  cues and refusal of overflow without font shrinking.

Actual assets and evidence:
`shorts/aurora-pilot-20261102/review/reference-match/render-v2/`.

- `card_0.png` through `card_36.png`: all37 transparent1080x320 cards,72px text.
- `caption_design.json`: new measured bounds and exact reused text/timing.
- `all-captions-360w.jpg`: all37 inspected at360px/card, including long headers.
- `index.html`: caption-only audio/timing review using the preserved review MP4
  as its audio source. It is explicitly **not** a new portrait edit. All37 seek
  buttons, boundary/gap visibility, muted playback clock and animation cancellation
  on pause passed browser checks. Seeking enters a cue by1ms to avoid media seek
  rounding onto the previous cue; **no stored timing is changed**.
- `review-page-checks.json`: browser verification and actual40.066667s media duration.
  `review-page-desktop.png`/`review-page-360w.png` were inspected; desktop/mobile
  viewport checks are not a real phone/YouTube UI review or listening approval.
- `render-manifest.json`: artifact/code hashes and exact source-cue provenance.
- Cue7's production-rendered PNG is **pixel-identical** to the creator-approved
  lowered proof (Pillow RGBA difference has no nonzero bounding box). Every one of
  the37 text/start/end tuples equals its prior cue; uncertainty is not erased.

Fresh final-code validation:

1. `.venv\Scripts\python.exe -m pytest tests/ -v --ignore=tests/test_face_detect.py`
   — **1,020 passed,47 skipped,18 xfailed**,35.83s; no failures. The required full
   run includes the Automation and Shorts Intelligence tests.
2. `.venv\Scripts\python.exe dry_run.py https://youtu.be/test --skip-download --skip-transcribe`
   — **PASS**, run170edd90,2 selected/exported,0 failures,0 uploaded. External
   operations are stubbed; this is not a real render or audience outcome. The
   expected embedding-fixture warning disables topic segmentation in this dry run.
3. `git diff --check` — PASS (line-ending notices only).

No canonical media or source was replaced, no speech was corrected by listening,
and no commit/push/upload/scheduling occurred. Face OS and Gaming Clip Studio
remain untouched. Next full edit still needs wider source coverage or an explicit
source-limited framing decision, followed by the existing speech/release gates.

## Best-effort whole package — creator-authorized compromise

The creator requested best-effort cropping, separate caption space, deliberate
thumbnail/hook packaging and better use of the LLMs. This resolves the missing
framing choice without claiming the source contains the reference's torso.

Artifacts: `shorts/aurora-pilot-20261102/review/packaging-v3/`.

- `packaging-review.mp4`: new full40.066667s1080x1920/30fps review cut, not an
  overwrite of the canonical pilot. Full-width1080x1248 picture at the top; manual
  smooth pan of a900x1040 source crop. Source movement samples led to two extra
  head-turn corrections before rendering. Small baked badge-only blur remains.
- Approved37 neon cards in their own navy area. Transparent transport begins
  y1304; its visible frame is y1356–1552, entirely below the y1248 picture end.
  No giant blurred duplicate-face backdrop or invented torso; uniform scale only.
- A brief explicitly labeled `TOPIC: GROUND BHI DEKHO!` line introduces the topic;
  it is not a fabricated source subtitle. Original audio/cut/timing is retained.
- `thumbnail.jpg`: matching source-derived cover reference, real questioning face,
  white/cyan **GROUND BHI / DEKHO!**,120px and360px proofs. A standalone JPG is
  not promised as a Data API-assigned Shorts cover. App frame selection/text still
  needs actual-device review; the strong29.38s frame has burned captions, while
  the15.6s gap provides a clean-text-space alternative.
- `PACKAGING_DECISION.md`: final title/cover decision, rejected alternatives,
  research, publication limits and measurable channel-pilot proposal.
- Pro generated20 title ideas/10 thumbnail directions; Flash challenged them.
  Rejected Pro's unverified-score hooks and unsound A/B assumptions. Exact prompts,
  responses and returned IDs are retained. Neither model saw pixels/heard audio;
  model scores are not audience evidence. OpenCode API403 was disclosed, not hidden
  behind silent fallback. Chosen lead title: **Chhota ground, bada score? Ground bhi
  dekho**; no numeric, current-match or result promise.
- Required unit suite: **1020 passed,47 skipped,18 xfailed**. Required dry run:
  **PASS**, run11395012,2 exported,0 failures,0 uploaded. Five additional artifact
  contracts passed for uniform geometry, all-frame crop-coordinate bounds/smooth
  steps, source identity, unchanged card timing and real gaps. These are not an
  all-frame hair detector or listening review.
- `technical-qc.json`:1202 frames;40.066667s video/40.064s audio; start0; full A/V
  decode passed; all3752 audio packet hashes and timing exactly match preserved
  canonical audio (stream copied, no additional audio encode).

New package remains REVIEW. The speech, complete-cut sound, real-phone/app,
strongest-selection and creator-approval gates are still unresolved. The original
source/canonical media remain preserved. No upload, scheduling, commit or push.

Final handoff: all1202 encoded caption regions checked against37 approved cards
and gaps (32 gap frames; maximum mean RGB error2.6997/255). Seven completed browser
seeks plus a presented8s frame verified before fresh desktop/360px screenshots;
both images inspected, no spinner or horizontal overflow. Muted playback only.
Final rerun:1020 passed,47 skipped,18 xfailed;5 packaging contracts passed;
dry run250635ba PASS,2 exported,0 failures,0 uploaded. Scripts/logs retained in
the package. All8 canonical artifact hashes and both source copies freshly match.
`package-manifest.json` binds the new review artifacts and outstanding gates.
Review video SHA-256: `10b8bafa6bf6f6680d1d27f5a721dd5a02e3c9f5c69a8e2c886abf93e60f2c7a`.
Cover reference SHA-256: `6721254c94ddb0d8e227a356ce103e42ad22469abc16b796fd22fce16db4bcb3`.

## Packaging04 — viewer-first refinement

The creator prioritizes whole-package quality. `review/packaging-v4/` adds a
68px opening headline with separate small TOPIC label, not a new caption theme.
Same original-source render, crop anchors,37 approved PNGs,72px caption type,
10px-lowered text and audio/cut/timing. It does not claim wider torso coverage.

Preferred clean app-cover mockup uses actual encoded frame477 at15.9s. Compared
462/470/477, retained all three originals/lettered proofs and120px tests. Only text
below the image is added. The camera pixels are unchanged before JPG compression.
It is still an app mockup, not an assigned YouTube thumbnail; app fit is unresolved.

The review page compares the current0–8s opening with a proposed5.58–13.58s
argument-first excerpt. This is listening comparison in the untrimmed player,
**not an applied edit or verified phonetic boundary**. Three problem-passage
replays and hash-keyed local/downloadable notes reduce review friction; no automatic
caption correction, alignment, approval, new learning database or publication.

`automation/packaging_design.py` supplies reusable source-faithful proof helpers;
24 new regressions reject stale/captioned cover frames, stretching, hidden copy
controls, unreadable overflow and accidental picture/UI-region changes. Existing
production defaults and unrelated systems remain unchanged.

Actual verification:1202 opening-band frames and1202 visible caption regions
checked;62 opening frames; copied3752 audio packets identical; full A/V decode
PASS. Browser viewports320/360/390/768/1240px no overflow; presented frame before
screenshots; preview-switch/stop/pause handling and draft persistence/export PASS.
Desktop/mobile opener, mobile cover and encoded/120px proofs inspected. Not sound
listening, actual device review or traction evidence.

Required suite:1044 passed,47 skipped,18 xfailed. Dry run7691a373 PASS,2 exports,
0 failures,0 uploads. Package manifest retains hashes and preservation audit.
Still REVIEW: speech, alignment, sound, source alternatives, actual app/UI and
final hash-bound creator approval. No commit, push, upload or scheduling.
