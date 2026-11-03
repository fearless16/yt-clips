# Pilot visual + editorial execution contract

**Latest visual authority:** [REFERENCE_TARGET.md](REFERENCE_TARGET.md). The
creator rejected the Aurora composition and supplied a new ideal reference;
the rounded-capsule/floating-blur directions below are historical, not governing.

Date: 2026-11-02. Status: **VISUAL REBUILD RENDERED / CLEANUP COMPLETE / RELEASE HELD**.

Actual outputs, checks, hashes and remaining gates: **[RESULTS.md](RESULTS.md)**.
That result record supersedes historical proposed statuses below. Speech fidelity
is still unresolved; do not confuse the completed visual rebuild with READY.

## Latest user brief / implementation revision (takes precedence below)

The later instruction explicitly requests a from-scratch futuristic caption
system, proper 9:16 composition, a matching thumbnail and cleanup of superseded
outputs. This supersedes the earlier restriction against a new design, but not
speech fidelity, source preservation or publication gates.

- New design: **Aurora v1**, rebuilt renderer in `automation/caption_design.py`;
  bundled OFL Oxanium variable font, heavy geometric capitals, midnight-glass
  capsule, thin pale outline and restrained ice-cyan edge. Old plain CSS removed.
- Solo composition: same-time camera-source split, full-canvas moving blur,
  genuinely smaller foreground. The initial clean x580/y0/w830/h1040 crop was
  **rejected** after a head-turn sample clipped hair at the left edge; evidence
  `evidence/rejected-narrow-crop-21s.jpg`. Revised source rectangle is
  **x450/y0/w960/h1040**. It excludes chat and bottom controls; its small upper-left
  baked badge is explicitly blurred in camera-local x0/y0/w136/h94, not silently
  declared absent. This is a non-face blur, not face synthesis or a large solid mask.
  Foreground maximum box 972x998 at top210 on 1080x1920; aspect-preserving fit
  produces ~920x998. Actual scale is ~0.96, versus the old ~1.03. Visible card
  starts at y1248, ~40px below it. Wider source coverage keeps head movement from
  being solved by additional zoom. Cover keeps its separate source-derived crop.
- Cover: source-derived questioning expression at 113.6025s, chosen after viewing
  a 12-frame contact sheet, not by sharpness score alone. Proposed hook
  `PEHLE OVER MEIN?` depends on retaining the subsequent first-over argument.
  Font/capsule match captions; hook raised so a diagnostic square crop retains it.
- Live model discovery corrected the original assumption: direct DeepSeek accepts
  `deepseek-v4-flash` but returned `deepseek-flash`. Official pricing documentation
  inspected 2026-11-02 confirms original V4 Flash is retired and this ID routes to
  **DeepSeek-V4.1-Flash-1017**. Original exact-version availability is NOT satisfied.
  `allow_documented_successor: true` explicitly accepts only that documented
  successor response, records both IDs and discloses the difference. Other model
  mismatches still fail. Reference: https://api-docs.deepseek.com/quick_start/pricing
- Model output may propose phonetic/orthographic corrections with uncertainty
  flags; digits/negation have validation guards. These are **not listening-approved
  corrections or freshly forced-aligned speech**. Original ASR intervals remain
  attached per source-token ID, including joined Roman forms.
- ASR diagnostics found zero-duration tokens, including an old final token and a
  new internal one. No timestamps were fabricated. The current render input uses
  valid original medium-ASR segments through114.72s and valid expanded-context
  segments114.72–124.28s. Turbo's incomplete trial is rejected, not promoted as a
  better transcript. Cut84.22–124.28s is an extended **review candidate**, not a
  listened/approved complete-thought claim.
- New outputs/evidence live in `shorts/aurora-pilot-20261102/`. Cleanup is restricted
  to superseded generated pilot media/candidate images and invocation-owned
  scratch after a replacement is verified. Keep source media, original transcripts,
  listening corrections, review records and this plan. Record exact deletions.

The sections below retain the original planning rationale. Where geometry,
model availability or design decisions differ, this revision governs. Final
results/validation will be recorded separately; do not infer completion here.

This is the detailed visual/caption/cover plan for the recovery pilot. The root
`EDITORIAL_QUALITY_PLAN.md` remains the wider engineering checkpoint. This document
does not supersede source evidence, listening corrections or release approvals.
All dimensions below are **proposed 1080x1920 output-pixel targets**, unless marked
observed. They are not measured results of an improved render.

## 1. Locked brief: what must not drift

- Preserve the supplied reference's dark translucent rounded caption capsule,
  thin pale outline and bold, wide, white uppercase lettering. Improve its
  placement, not replace it with floating text, karaoke blocks or a new theme.
- Slightly smaller, calmer foreground; whole hairline, chin and useful expression
  visible. No extreme face crop, automatic punch-ins or stretching.
- Moving blurred source video fills the entire portrait background. No large
  solid caption band, fake duplicate frame or frozen backdrop.
- Caption sits immediately below the sharp foreground, not across the mouth.
  Background behind this card is still moving video, not a dedicated empty slab.
- Hindi meaning in natural Roman-script Hinglish. Existing English cricket terms
  stay natural. No English translation or AI-generated improvements to speech.
- Exact `deepseek-v4-flash` route for contextual caption processing. Neither a
  generic preference nor local Qwen/rule-only fallback satisfies this requirement.
- Complete thought and payoff at 1.0x; a good-looking frame cannot certify a cut.
- Preserve current artifacts and unrelated edits. No Face OS or Gaming Clip
  Studio changes, upload, scheduling, commit or push under this planning task.

## 2. Evidence register and present verdict

Paths below are relative to repository root unless absolute. Pilot directory:
`shorts/recovery-pilot-20261101-172917/`.

| ID | Evidence actually inspected | Finding / limits |
|---|---|---|
| E1 | `C:/Users/user/Pictures/Screenshots/Screenshot 2026-11-02 002241.png` | Authoritative card reference: two-line `JEETNE KI / KOSHISH`, dark rounded capsule, restrained pale border, wide white uppercase. Crop too tight; card crosses lower face. Player/channel UI is not an instruction to burn it into video. Font identity and exact opacity not established. |
| E2 | `evidence/prior-pilot-frame-8s.png` | Previously extracted eight-second pilot frame, viewed again now. Sharp foreground above large solid dark lower area; plain mixed-case caption `kaun se aapake jis`; no reference capsule. Static image only, not current playback or speech verification. |
| E3 | Pilot `pilot-unverified_thumb.jpg` | Actual cover inspected full size. Large solid top/bottom bars, downcast expression, raised hand, partial viewer counter at upper left and `CONTROLS` near bottom. No hook text. |
| E4 | Pilot `pilot-unverified_cover.json` | Selected source time 99.610833s, `text` empty, `review_status` NOT CHECKED. Selection explicitly sharpness/exposure only. Technical score is not emotional quality or audience evidence. |
| E5 | Pilot cover `candidate-00.jpg`, `candidate-08.jpg` | 00 has closed eyes; 08 has a more intent expression. Both retain padding/overlays. Only these two alternatives and the selected image were visually inspected; no claim that all 12 were compared. |
| E6 | `evidence/source-99_610833.png` | Newly extracted full 1920x1080 source frame at selected cover time. Right-side chat/subscriber panel; upper-left stream status; lower-center `CONTROLS` are baked into the source. Face/hand occupy the center-right camera region. Not removable by toggling output overlays. |
| E7 | `evidence/current-cover-120w.png`, `evidence/current-cover-center-square-120.png` | Actual 120px portrait and diagnostic centered-square reductions inspected. Portrait wastes substantial height; face is recognizable but the subject/story is unclear. Square uses the face more efficiently but still has no topic. Square is a stress test, not a promise about YouTube's current crop. |
| E8 | `export.py`, `automation/caption_engine.py`, `utils/ai_client.py`, `config.yaml` | Implementation mismatches documented in section 9. Read-only inspection, no fixes made in this pass. |

**Visual verdict:** current pilot/cover need revision. This is not a claim that
the moment is poor or that a replacement will perform better. Audio fidelity,
full-clip motion, final safe zones and strongest-moment selection remain unverified.

### Identity pins (SHA-256 rechecked during this planning pass)

```text
reference screenshot  307d14745c82ff37590b132cbbf834a02f2e410d0a52543d61b8e3e464ba7619
pilot MP4            e12252293bfa3ead5a3e72f88a2f2d48b73f9b8ff40105b54f7c46bdfc091efa
pilot thumbnail      51d0c03c6d81eb11aa5519b8b8f75576e8005b6cc9130711e644a061dc6746f9
cover JSON           953afdf792de3e983deb782429d6f2a62b8639e44c1700c047778deb07f35133
```

Source identity from prior verified work, **not rehashed this pass**:
`input/recovered/nrl5qQD0E98.mp4`, SHA-256
`b95b1bfd57a270066b6e63ccd3b2c8baec041fa614ff60e04925ff8e9eb603d5`.
Current edit manifest declares 84.22–117.8s, crop x400/y0/w1000/h1080, 1.0x.
Those are provisional inputs, not approved boundaries or final framing.

## 3. Crop: solve the source constraints, not just a zoom slider

### Composition hierarchy

1. Face/eyes and genuine gesture carry the reaction.
2. Compact caption card carries the spoken phrase.
3. Defocused background fills space quietly; it must not compete with either.

Use the same source timestamp for foreground and background throughout. Split
after the source-time edit map is applied so a future trim cannot desynchronize
the two branches. Preserve aspect ratio on both branches.

### Foreground starting trials (not approved crops)

- Start from the declared 1000x1080 source camera crop, **not** a new narrow 9:16
  face extraction. Trial A: fit to width 960 -> height about 1037, centered at
  x60, top about y140. Trial B: width 1000 -> height1080, x40, top about y120.
  Round encoded dimensions/coordinates consistently where the filter requires it.
- Trial A is about 7% narrower than the ~1032px sharp region in E2. This is a
  modest zoom reduction, not a tiny landscape video stranded in a portrait frame.
- Compare identical timestamps for A/B: opening, largest hand/head motion,
  longest caption, payoff and last spoken word. Keep card design identical so
  framing is the only variable. Prefer A unless face/expression becomes too small.
- Target face center roughly 45–65% across the canvas; maintain visible hairline
  and chin with breathing room across motion, not just one still. These are review
  guides, not face-detector measurements already obtained.
- Locked framing is the default. If motion breaks it, first widen/reposition the
  source region; only then consider slow bounded tracking. Never chase every
  detection or add a zoom on each caption change. Disclose unresolved edge clipping.

### Critical constraint: overlays versus less zoom

E6 proves the viewer badge and lower controls are baked in. Widening the crop
rightwards exposes live chat; trimming the source top can clip hair; trimming the
bottom can lose gesture/chin. Enlarging a smaller clean crop back to the same
output width would **increase** face zoom and defeat the brief.

Before locking coordinates, inspect source frames across the whole proposed cut,
map face/head/hand and overlay extents, and record the actual source rectangle.
Try a clean rectangular source region only when it retains the full head/gesture.
If cleaning the region requires a smaller crop, reduce its output width enough to
hold or reduce face scale; review whether the sharp panel then looks too small.
Do not hide this tradeoff behind a generic `face-safe` label.

Preferred outcome: clean camera image with less zoom. If source-baked overlays
cannot be excluded without harming the subject, explicitly document the residual
and keep REVIEW; do not inpaint a face, invent replacement scenery or silently
cover the frame with a large solid mask. A clean original camera recording, if
available, is preferable to destructive cropping; its existence is not assumed.

### Full-canvas background and spacing

- Use a camera-region source branch where feasible so background chat does not
  become legible decoration. Cover-fill 1080x1920, crop excess, then blur.
- Starting blur sigma 25–40 at output resolution; starting brightness reduction
  15–25%. These are trial values. Inspect for duplicated giant facial features,
  residual legible text, distracting hands and blur pumping before choosing.
- Background must remain video all the way to all four edges and below captions.
  Mild darkening is allowed; near-black treatment that recreates apparent bars is not.
- Foreground bottom will initially fall around y1177–1200. Place the visible card
  about 32–48px below it; do not count transparent PNG padding as this gap.
- Keep the foreground simple and sharp. No ornamental frame, heavy glow, extra
  corner logos, score graphics, CTA or fake phone interface in this first revision.

## 4. Caption appearance: restore the reference faithfully

### Proposed reference-matching tokens

| Element | Starting specification | Verification |
|---|---|---|
| Capsule | dark navy/charcoal, alpha about .78–.88; radius 28–36px | recognizably dark/translucent on light and dark moving frames |
| Outline | 1–2px pale gray/white at restrained opacity | visible but not a neon frame |
| Shadow | small soft external shadow only | separation, no muddy haze or extravagant glow |
| Type | installed/licensed wide heavy sans, uppercase presentation | compare actual rendered glyphs to E1; do not claim the reference font is known |
| Size | start 56–64px, line-height 1.05–1.15 | readable at 360px-wide playback, no clipping |
| Width | content-fit, normal max 840px | safe center region; no long full-width strip |
| Padding | x28–36px, y18–24px | compact around actual text, not a 320px visible card |
| Lines | one or two; target 2–4 spoken words, semantic grouping first | no three-line captions or split names/numbers |
| Color | white as reference; no default gold emphasis | no semantic emphasis invented by renderer |

Use `JEETNE KI / KOSHISH` as a **style regression fixture only**. Do not insert it
into this pilot unless the reviewed speech actually says it. Preserve normalized
readable text separately from uppercase display; never stretch the rendered image
to fake a wide font. Audit installed font files and capture real candidates first.

### Position and transparent-canvas trap

- Starting visible card top y1230–1250, width <=840, centered. Two-line card
  should end about y1410–1450. This leaves space for lower UI without dropping
  the card to the bottom edge or covering the mouth.
- Treat x96–924 and y1200–1480 as a conservative **initial text region**, not
  official fixed YouTube safe-zone dimensions. Reconcile with actual phone UI.
- Existing card PNG is 1080x320 and content is vertically centered. `_caption_card_y()`
  currently positions that outer canvas. New placement must account for the
  visible alpha/content bounding box; a correct PNG origin can still leave an
  excessive gap. Test both the canvas and visible capsule bounds.
- Anchor two-line cards consistently so one-line/two-line changes do not bounce
  the whole design. Hold a stable baseline/anchor within a reserved small envelope.
- Use clean cuts between phrases for v1. No typewriter entrance, scale pulse or
  transition that delays reading. No added text during genuine silence.
- Split an overlong phrase at an aligned linguistic boundary before shrinking
  type. Never arbitrarily split a word to make it fit or silently drop a word.

## 5. Caption truth + exact V4 Flash route

### Order of operations

1. Listen to the source with surrounding context; record verbatim correction,
   reviewer/date and uncertainty. Existing listening packet is preparation only.
2. Send bounded corrected/contextual text to **exact `deepseek-v4-flash`** for
   natural Roman Hindi spelling and phrase proposals. Context is not caption text.
   Model may flag an ASR suspicion; it cannot certify what the audio says.
3. Validate the response before accepting it; hold unresolved names, numbers,
   negations or changes of meaning. Return uncertain speech for actual listening.
4. Align corrected spoken tokens to the original audio. Preserve original word
   references for one-to-one Romanization; explicitly map one-to-many/many-to-one
   forms. If correction changes tokens, rerun alignment rather than dividing time
   proportionally. Invalid/unsupported alignments block the render/release route.
5. Group aligned words into readable phrases; render and compare against sound.

### Routing and evidence contract

- Request exact provider/model without changing unrelated SEO/global selection.
  Verify configured endpoint support; no successful V4 Flash call is claimed yet.
- Missing credentials/model, timeout, malformed response or model mismatch ->
  explicit error/REVIEW. No silent Qwen, Pro, generic `deepseek-chat` or rule fallback
  presented as Flash. An explicitly labeled diagnostic fallback cannot satisfy the gate.
- Record requested model, returned model/provider when exposed, timestamp,
  prompt/schema revision, input hash, raw response, validation and accepted text.
  If the provider does not attest the underlying model, report that limitation.
  Never log credentials.
- Structured output must retain stable cue/token IDs, original text, proposed
  Roman text, source references and uncertain spans/reasons. Reject missing/extra
  IDs, unexpected prose, unrelated context leakage and invented timing.
- Keep a small consistent spelling glossary (e.g. `kyunki`, `nahi`, `koshish`),
  not over-formal transliteration. Keep cricket names/terms faithful. No automatic
  numeric normalization until the spoken value is verified.
- Known prior errors `हुखास`, `मैज`, `11-100` and disputed numbers are explicit
  listening cases, not replacements the model may confidently guess.
- Cache key includes source/corrected text and context hash, exact model,
  glossary, prompt/schema version. Old Qwen cache must not masquerade as Flash.

### Timing/readability acceptance

No word removed, invented or assigned outside its source interval. Preserve negation
with the phrase it qualifies; keep a name/score readable as a unit. Investigate
onset/offset errors exceeding roughly 80/120ms as an initial review tolerance,
not a claim of current accuracy. Avoid sub-250ms whole-card flashes by regrouping
where real word timing permits; do not slow speech or fabricate display time.
Record unresolved fast speech instead of claiming every cue meets a generic CPS.

## 6. Cut and moment selection: independent from styling

- Keep 84.22–117.8s as the current candidate, not a locked final cut.
- Listen from at least two seconds before/after; expand if a pronoun, claim or
  payoff depends on missing context. The packet covering 80–135s is available.
- Write one sentence each: viewer's question, essential setup, claim, payoff,
  why the ending resolves it. If these cannot be written faithfully, hold the cut.
- Start on a clean phonetic boundary with enough setup to identify the subject;
  end after the last meaningful word/reaction. Do not truncate for a target duration.
- Inspect opening/closing audio and adjacent frames. A fade is not a remedy for
  a missing syllable; evaluate existing 0.5s video/0.4s audio fade settings so they
  cannot conceal the payoff. Prefer a natural endpoint without a long black tail.
- Preserve 1.0x. Any internal removal needs an explicit source-to-output map and
  continuity check; no jump cuts justified solely by silence length.
- Compare strongest admitted alternatives with context, not a few attractive
  frames. Disclose the 60-candidate budget/omissions and source coverage limits.
  Current structural shortlist evidence does not prove this is the best moment.

## 7. Thumbnail: a separate design and review decision

**Current verdict: revise, not delivery-ready.** Face and hand are usable human
elements, but large solid bands waste the portrait; downcast eyes weaken direct
engagement; baked stream UI distracts; no text/topic tells a new viewer the story.
At 120px the face remains identifiable, but clarity of topic fails. This is a
visual judgment, not a measured CTR prediction.

### Candidate and composition procedure

1. After cut/listening is resolved, inspect all existing candidates and additional
   nearby real source frames if needed. Record source time, expression, eyes,
   hand obstruction, blur, overlay residue and relevance to the actual payoff.
2. Candidate-08 is worth comparing for its intent expression; it is **not selected**.
   Candidate-00's closed eyes are a negative unless that is genuinely the story.
   Current candidate-05's sharpness score alone must not win.
3. Create a clean **source-derived** cover composition, not a screenshot of a
   subtitle mid-phrase. One face, one emotion, one honest hook of <=3 words.
   No fabricated celebrity, match score, trophy, tears or rage expression.
4. No solid bars; use restrained camera-derived blur. Allow a tighter cover than
   playback if it improves small-size recognition without clipping the face.
   Put face on one side and hook on the other only if portrait space permits;
   otherwise stack hook over a clear shoulder/background zone, never the eyes.
5. Use the same heavy font family and dark/white identity as captions; optional
   single saffron accent for the hook. No paragraph, tiny stats, multiple logos
   or live badge on a prerecorded Short. Existing channel conventions do not
   justify a misleading LIVE label.
6. Inspect portrait at 120px and 360px, diagnostic square/center crops and actual
   intended YouTube surface. Keep hook/eyes away from crop edges and UI badges.
   Record each observed result; do not mark mobile PASS from a full-size image.

### Three constrained cover directions (not final hook copy)

| Direction | Subject/emotion | Text/layout | Decision rule |
|---|---|---|---|
| A: argument | authentic intent expression, compare candidate-08 | <=3-word Roman-Hindi question drawn from verified claim; white on dark area | preferred starting direction if the cut contains a clear argument |
| B: tension | genuine raised-hand emphasis | <=3-word stakes hook, one saffron word at most | only if gesture and spoken stakes agree; reject distracting hand obstruction |
| C: clean reaction | best real eyes-open expression | 1–2-word faithful reaction, minimal capsule | use if topic is otherwise immediately identifiable; do not hide lack of context behind vague hype |

Final wording is deliberately **unresolved** until the speech/story is checked.
Do not manufacture a collapse/score/revenge hook from the India–NZ stream label.
Save final cover, source timestamp, copy, crop, hashes and review separately from
the old cover record. A JPEG on disk is not evidence YouTube accepted it: verify
the supported Shorts cover-selection surface later, under separate publication
authorization; do not promise arbitrary custom-thumbnail upload support.

## 8. Ordered execution, deliverables and stop conditions

| Phase | Deliverable | Exit gate / stop |
|---|---|---|
| 0: pin inputs | source identity, current worktree, reference + this plan | mismatch -> re-inspect, preserve existing work |
| 1: speech/cut truth | real corrections, uncertainty log, complete-thought rationale, alternatives | unresolved number/name/negation or clipped payoff -> HOLD |
| 2: caption route TDD | strict Flash call/schema tests, mocked failures, corrected alignment mapping | no silent fallback or fake timing; real route evidence required separately |
| 3: composition TDD | A/B foreground trials, reference capsule, alpha-bound placement tests | no solid band, face/gesture safety and full moving background |
| 4: frame review | same-time A/B, first/last, extreme gesture, long caption, light/dark background | choose geometry from viewed images; don't declare motion/audio checked |
| 5: full pilot | new versioned unapproved MP4 + edit/caption/model/QC evidence | full decode and actual end-to-end audiovisual playback; no unnoticed fade damage |
| 6: cover | compared source frames, faithful hook, new cover and reductions | inspected 120px + surface/crop checks; no technical-score-only selection |
| 7: owner review | phone playback notes + named owner/date bound to final hashes | READY only through existing release gate; otherwise REVIEW |

Implementation targets: `automation/caption_engine.py`, `export.py`, narrow caption
configuration in `config.yaml`, explicit model route in `utils/ai_client.py` only
as needed; tests in `tests/test_caption_engine.py`, `tests/test_export_composition.py`
and routing/alignment tests appropriate to their existing modules. No wholesale
renderer rewrite, no second learning DB or replacement release-approval mechanism.

Use a new versioned pilot output folder; never overwrite `pilot-unverified.*`,
source transcripts or listening corrections. Keep manifest links to original
artifacts and final hashes. Preview frames do not replace a finished video.

## 9. Specific implementation gaps and regression cases

Observed gaps:
- `config.yaml` reserves a caption band and still names `qwen3.8-27b` for captions.
- `export.py` final reserve-band path scales/pads with solid `0x101624`; a blur
  branch for screen shares alone does not fix solo framing. Prevent a later pad
  from overwriting a correctly built blurred background.
- Caption CSS `.holo-glass` is transparent; current font selection prioritizes
  Arial Bold, not an established match to E1. Name `holo_glass` is not visual proof.
- Word-timed path bypasses contextual AI conversion. Fixing only legacy conversion
  leaves the active route wrong.
- `generate_deepseek()` uses `self._model`; generic `prefer_model` does not prove
  an exact Flash request. Avoid global model state leakage to unrelated tasks.
- `_caption_card_y()` reasons about a 320px canvas, not actual visible content.

Required red -> green coverage before code acceptance:
- Solo blur branch covers complete canvas; source/foreground timestamps agree;
  no later solid-pad override. Existing guest/screen-share behavior stays intact.
- In-bounds/aspect-preserving crop and scaling; bad/nonfinite dimensions reject;
  full head + gesture fixtures include motion extremes and baked overlays.
- Capsule has real alpha/background/border; correct uppercase display and explicit
  font load; one/two-line and long-token fixtures fit without shrinking to illegibility.
- Actual visible capsule/card bounds respect face gap and safe-region envelope;
  separate transparent padding from rendered shape. No bouncing or third line.
- Both timed and legacy caption entry points request exact Flash where intended;
  absent credentials, timeout, wrong model, malformed JSON, extra IDs and context
  leakage stay explicit. Tests must assert actual request payload, not config only.
- Romanization/correction token mappings, numerals/negation/names, alignment failure,
  clip offset, 1.0x mapping and cache invalidation. No invented word timestamps.
- Cover must retain source provenance; no sharpness-only approval or stale old
  cover hash after a new selection. Preserve existing release gating.

After code changes, mandatory commands (do not claim new results now):

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v --ignore=tests/test_face_detect.py
.venv\Scripts\python.exe dry_run.py https://youtu.be/test --skip-download --skip-transcribe
```

Run appropriate focused tests first; full suite/dry run must pass before any
requested commit. Mocked model tests cannot establish real endpoint support;
FFmpeg decode cannot establish caption truth or playback quality.

## 10. Anti-drift ledger: update from evidence, not intention

Use statuses **PROPOSED / IMPLEMENTED / VERIFIED / BLOCKED**, each tied to an
artifact and check. Invalidate dependent reviews after changing cut, transcript,
crop, captions or cover. Update this ledger before calling any phase complete.

| Item | Current status | Evidence needed to advance |
|---|---|---|
| Reference and current visuals inspected | VERIFIED (static images only) | E1–E7 above; not full playback |
| Detailed plan persisted | IMPLEMENTED | this file + root plan link; no video change implied |
| Less-zoom crop and moving blur | PROPOSED | actual A/B frames, motion review and chosen geometry |
| Reference-style capsule | PROPOSED | rendered card + composited frames at phone size |
| Exact V4 Flash processing | PROPOSED | validated request/response route evidence, no credentials |
| Corrected speech and alignment | BLOCKED | actual listening corrections + valid aligned token mapping |
| Clean cut / strongest-moment claim | BLOCKED | source-context playback and alternative comparison |
| Improved cover | PROPOSED | actual new source-backed cover + inspected reductions |
| Full export/audio/phone review | BLOCKED | new artifact, complete playback, owner/date/hash |
| Publication / owner approval | BLOCKED | explicit later authorization and existing release gates |

Never claim: listened when only read, corrected when only transliterated, Flash
when only preferred, rendered when only planned, full review from selected frames,
best moment from a capped shortlist, delivered thumbnail from a local JPEG, or
approved from a technical PASS. No fabricated mobile/owner sign-off or performance
guarantees. If a tool cannot perform a required check, record NOT CHECKED and the
concrete missing evidence.

### Next action on resumption

Read this file and the current edit/listening manifests; inspect worktree/input
identity. Resolve speech/cut truth, then implement tested Flash routing and the
two controlled visual trials. Do not ask again for the supplied old reference.

Planning-pass validation: source frame extraction succeeded; listed images and
120px reductions actually viewed; identity pins above freshly checked. This pass
changed documentation and review images only, not runtime code. No new suite,
listening, model call, improved render, owner approval or publication is claimed.
