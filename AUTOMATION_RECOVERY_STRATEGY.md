# Cricket Shorts automation: evidence review and recovery strategy

**Updated: 2026-11-01. Status: implementation under validation; not release-certified.**

## Implementation checkpoint

- Implemented sandboxed dry runs with input-overwrite and subprocess escape guards,
  truthful selected/exported/metadata counts, and actual pipeline event reporting.
- Removed arbiter-rejection backfill and post-render source-time jumpcuts. Added
  fractional timing, discovery/alternative evidence, and source-to-output edit maps.
- Active VidIQ integration is disabled. Native DeepSeek uses the exact model ID;
  SEO accepts native credentials without requiring OpenCode credentials.
- Pilot defaults preserve 1.0x speech, disable generated hook overlays, reserve a
  caption band, use local fonts and require word-timed shortlist ASR.
- Added source-frame cover shortlists, encoded-file decode/audio QC and hash-bound
  named owner approval. Changed artifacts invalidate approval; no publishing was run.
- Restored `input/video.mp4` matches `input/recovered/nrl5qQD0E98.mp4` by SHA-256:
  `b95b1bfd57a270066b6e63ccd3b2c8baec041fa614ff60e04925ff8e9eb603d5`.
  The damaged input remains archived separately.
- Real medium ASR improves coverage but still corrupts Hindi words and numbers.
  Confidence/timing checks do **not** establish faithful captions. Any pilot using
  that text remains an engineering sample, not publishable editorial work.
- Whole-video audiovisual comparison, full listening review, phone approval,
  fresh channel outcomes and a channel-specific model benchmark remain outstanding.
- Validation: broad non-Face-OS suite **894 passed, 47 skipped, 18 xfailed**;
  two subsequently added isolation regressions also pass. The sandbox dry run
  reports **2 selected / 2 exports / 2 valid metadata / 10 events / 0 failures**.
- Engineering pilot: `shorts/recovery-pilot-20261101-172917/pilot-unverified.mp4`.
  Real full decode passes; duration 33.583333 s, measured audio -16.59 LUFS and
  -2.18 dBTP. Sampled frames and source-derived cover were inspected; this is
  not full-motion/listening review. Known bad caption text, heuristic crop and
  source overlays remain review issues. No owner approval or metadata package
  was manufactured to make this sample eligible for publication.

The original audit below records historical findings, not current implementation
status. Its baseline failures and original unsafe dry-run result are retained as evidence.

## Decision

Repair editorial selection, timeline correctness, composition and packaging as one
source-bound workflow. Do not treat more keywords, louder audio, a more expensive
model, or more uploads as a substitute for a watchable moment.

At the original audit checkpoint no application code or configuration had changed.
`face_os/`, its configuration and its tests remain excluded from the work.
Gaming Clip Studio was reviewed read-only; do
not merge its channel, publishing policy, layout or learner into this project.

## 1. Git baseline and validation gate

- Current branch: `feat/hinglish-hook-caption`, HEAD `679e0ae`.
- Local `main` is already at the same commit; it is three commits ahead of the
  locally recorded `origin/main`. No remote fetch or push was performed.
- Existing uncommitted work: eight tracked files plus `AUDIT_SHORTS_SEO.md`,
  `automation/caption_engine.py`, `automation/hinglish_converter.py`, and
  `tests/test_caption_engine.py`. Preserve it; it is not work created by this audit.
- Baseline suite launched with
  `.venv\Scripts\python.exe -m pytest tests/ -v --ignore=tests/test_face_detect.py --ignore=tests/face_os`.
  Final result: **26 failed, 833 passed, 47 skipped, 18 xfailed** in 195.55 s
  (924 collected). Failures include SEO context, SEO failure recovery, Instagram
  argument threading, dynamic cricket parsing and the AI rotation mock contract.
  The root conftest already marks specific historical failures xfail with
  `run=False`; those expected failures were not fixed or newly suppressed here.
- The mandated dry run exited with its own `OVERALL: PASS`, but selected three
  clips, exported **zero**, reported zero failures, and observed zero structured
  records. This is not a passing end-to-end acceptance result.
- **Do not commit/merge this unvalidated working tree.** The repository requires
  failures fixed before committing, while this task explicitly asks for review
  before fixes. The safe next step is approval of the repair plan, isolated
  validation fixes, then an explicitly scoped baseline commit on the current
  branch, a fast-forward of local main, and a fresh recovery branch. Never use
  `git add .`, reset user work, include secrets/media, or push without a request.

### Dry-run side effect requiring attention

`dry_run.py:306-316` scans FFmpeg arguments backwards for any media filename and
writes fake bytes to it. For pipe/null-output commands that can be the **input**.
After the run `input/video.mp4` is 512,000 bytes and FFprobe reports `moov atom not
found`. No pre-run source backup/hash was captured, so this review cannot restore
the prior input or establish its pre-run validity. Reacquire the source into a
fresh path before source/export comparison. Do not run this harness again on
working media; first isolate every test path and prevent writes to input files.
The dry run also rewrites the ignored highlights artifact. Its generated
highlights must not be attributed to the October export batch.

## 2. What was actually inspected

- Current automation, selector, arbiter, export, caption, thumbnail, SEO/provider,
  upload and canonical learner code/configuration.
- All three videos' geometry/duration and measured audio loudness in
  `shorts/2026-10-31_170414_2036b3e6/`.
- Five sampled frames per video (opening, 1 s, 35%, 65%, near ending), plus
  `clip1_thumb.jpg` and `clip2_thumb.jpg`, viewed as images.
- `transcripts/video.json`: 1,729 cues, 14,123 whitespace-delimited tokens,
  55.039–4470.56 source seconds; **zero word-timed cues** and **1,656 overlapping
  adjacent cue pairs**. Overlap is a display-timing fact, not proof of duplicate
  words or actual simultaneous speech.
- Read-only SQL on `shorts_intelligence.db`: 213 catalog entries, 381 snapshots,
  64 production-feature records. Last outcome snapshot: **2026-08-20**.
- Gaming Clip Studio's current `package_video`, `package_planner`,
  `package_renderer`, source/runtime contracts and one recovered batch manifest
  and cover. The manifest reports five deliveries, with incomplete discovery
  coverage explicitly recorded. This is stronger evidence than the README's older
  pending-E2E note, but was not independently replayed or fully played back here.

Review-frame artifacts:
`C:/Users/user/AppData/Local/Temp/opencode/shorts-audit-20261101/clip{1,2,3}-sheet.jpg`.

**Limits:** no listening/full-motion playback, fresh owner Analytics query,
DeepSeek task benchmark, or live thumbnail-delivery verification was performed.
The unavailable/invalid source prevents a reliable frame/audio correspondence
audit. Do not infer the precise cause of current channel reach from this sample.

## 3. Current flow and concrete defects

Current flow:
`ingest -> transcript -> candidate discovery -> heuristic agents -> arbiter ->
export/speed/captions + per-clip SEO -> post-render jumpcuts -> hook audit ->
optional enhancement -> thumbnails -> sync/upload/schedule -> telemetry`.

| Priority | Finding and evidence | Consequence / confidence |
|---|---|---|
| P0 | `export.py:1734-1794` cuts already-rendered media using source-relative transcript intervals; it does not transform these by the export speed. | Timing mismatch when speed != 1; gaps are not waveform-verified silence. High confidence in mechanism, not attribution of every observed cut. |
| P0 | `export.py:1593-1594` prefers formatted `start`/`end` over precise `start_sec`/`end_sec`. | Fractional boundaries can be lost before render. High. |
| P0 | `caption_engine.py` discards word timing, estimates cues by character share, slices boundary text proportionally, and deliberately allows rolling-cue overlap. | This is estimated text placement, not speech alignment. Simultaneous cards are possible; overlap is visible near the sampled end of clip2. High. |
| P0 | Caption PNGs use fixed `overlay=0:1216`, a fixed 1080x320 canvas; configured `from_bottom` is not used there. | Mouth obstruction is visible in all three sampled exports. No face/UI collision solver. High. |
| P0 | `thumbnail.py:69-94` extracts exactly 1 s from captioned output and uses the title's first three words. | Downward-looking portrait, mouth caption, generic `IND VS NZ`, duplicate branding; not moment-specific cover selection. High. |
| P0 | Uncommitted SEO fallback returns source-title inheritance early, with `grounded: true`, generic CTA and a reduced field set. | It bypasses later audit/validation; examined clip2 metadata actually uses this path. A boolean is not evidence. High. |
| P0 | `generate_deepseek()` remaps V4 Pro to `deepseek-chat`; `generate_seo_text()` requires an OpenCode key before trying its DeepSeek entries. | Configured model/provider is not the effective contract; DeepSeek-only SEO can be blocked. High. |
| P1 | Split-on-pause output is immediately passed into `_merge_caption_fragments()`, which joins until punctuation/gap. Candidates are subsequently stamped `complete_thought=True`. | Split intent and semantic completeness are not independently guaranteed. High on code; benchmark impact still needed. |
| P1 | Selector refills arbiter omissions to configured `min_selected=3`, and falls back to heuristic winners if arbiter returns none. | Rejection/abstention can become publication. Quality floor 38 is not validated audience appeal. High. |
| P1 | Orchestrator counts SEO success as `len(exported)`; exporter logs success without inspecting the SEO helper's returned status. | Misleading completion telemetry; tests already exercise failure recovery. High. |
| P0 | Failing recovery tests demonstrate scheduling includes clips without metadata, falsey uploads can be treated as published, malformed/incomplete metadata is not gated, and retry trusts a payload clip ID such as `../escaped`. | Repair publication eligibility, safe artifact identity, atomic persistence and failure reporting before any new unattended release. These were mocked tests, not actual uploads. High. |
| P1 | Hook audit runs after selection/render; when transcript is missing it may use generated description text. | Generated copy can masquerade as speech evidence; findings do not gate release. High. |
| P1 | Screen-share branch says blur+fit but uses cover-fill and crop. | Comment is not implementation; scorecard content can be cropped. High. |
| P1 | Existing solo geometry attempts 30% face-top placement but clamps inside an existing region; other layout/fallback branches bypass this framing. | A nominal padding percentage does not guarantee full hair/head/chin in final pixels. Tight framing is visually confirmed; exact active branch needs a render manifest. |
| P1 | Learner is active although locally stored outcomes stop in August. It computes `engaged/views` as a continued-view proxy. | Stale evidence; that ratio is **not** Studio's feed-based stayed-to-watch percentage. High. |

### Audio measurements: do not blindly increase gain

These are the `input_*` measurements from analyzing the existing encoded files,
not the hypothetical `output_*` values of the analysis filter.

| Export | Duration | Cadence | Integrated LUFS | True peak dBTP | LRA LU |
|---|---:|---:|---:|---:|---:|
| clip1 | 29.617 s | 60 fps | -15.74 | -1.34 | 4.0 |
| clip2 | 10.450 s | 60 fps | -15.35 | -1.28 | 2.6 |
| clip3 | 9.583 s | 60 fps | -16.65 | -2.74 | 2.5 |

Loudness alone does not establish intelligibility, unclipped source audio, good
mic tone, or lip-sync. Current config says 30 fps, while this batch is 60 fps:
save a per-render config snapshot rather than assuming current settings produced
an old artifact. Clip3 is below today's 10 s floor; provenance is needed before
attributing that difference to one function.

## 4. What to borrow from Gaming Clip Studio

Borrow **contracts and validation patterns**, not the whole project:

1. Source cue IDs, transcript hash, retained chronological parts and explicit
   source-to-output mapping; validate once, then use everywhere.
2. Complete setup/payoff from one event; zero/fewer clips is valid. Deduplicate
   events, not merely overlapping windows.
3. Exact-cut audio measurement, two-pass normalization, encoded loudness readback,
   decode/packet coverage checks and a persisted QC manifest.
4. Cover candidates from retained source frames, with timestamps and selection
   evidence; independent cover text that is measured to fit.
5. Fresh batch directories, no overwrite, truthful partial completion, source hash
   checks and separate renderer versus publication responsibilities.

Do **not** copy its hardcoded couple-camera geometry, gaming palette, 25-second
cap, 60 fps normalization, manual-only publishing, game-specific assertions or
ASCII language restrictions wholesale. Its phrase timestamps are still estimated,
not forced word alignment; cover quality is sharpness/luma, not emotion or event
understanding. The inspected cover also repeats the embedded facecam above the
enlarged camera; its large header/graphic zones are not an ideal cricket layout.

Recommendation: keep both apps separate for this recovery. A small shared edit
contract may be extracted later only after each has regression fixtures.

## 5. Target flow: edit decisions before pixels

`source identity + metadata -> normalized transcript + audio/VAD evidence ->
whole-stream event discovery -> semantic shortlist -> local ASR/word alignment ->
reviewable edit decision list -> composition + captions + cover plan ->
single timeline render -> encoded technical/visual QC -> metadata validation ->
reviewed pilot -> existing authorized publication route -> actual outcomes`.

Each proposal must retain: source/video ID and hash; cue IDs; event and angle;
hook/setup/payoff ranges; retained parts; why an omission is safe; speed segments;
confidence and unresolved context; source-to-output mapping; model/provider IDs;
cost/latency; layout version; caption timing quality; metadata evidence; QC result.
Store production/outcome learning only in the existing canonical database, not a
second analytics/learner database. Per-run JSON files are evidence manifests.

### Audience-first release contract (clarified 2026-11-01)

The objective is the strongest self-contained viewing experience we can justify
from the source, not a quota of technically valid clips. "Best part" is an
editorial judgment, not a provable global optimum or a promise of audience reach.
Our defensible claim is: we covered the available source, compared the strongest
distinct moments, preserved their meaning, and reviewed the actual delivery.

1. **Prove discovery coverage.** Inventory the full playable source duration,
   transcript coverage, missing speech and failed discovery blocks. Inspect
   transcript gaps against audio activity; use local ASR where needed. A full
   transcript scan alone does not establish full audiovisual review. Report the
   inspection method and coverage separately. With unresolved material gaps, label
   the shortlist partial and hold automated release rather than claiming the best
   moments of the entire video were found.
2. **Compare moments, not keyword counts.** Keep a ranked shortlist with source
   timestamps, event IDs, the actual spoken premise/payoff, necessary preceding
   context, and reasons for acceptance/rejection. Judge standalone clarity,
   specificity, insight or emotional payoff, opening usefulness and editability.
   Compare the proposed winner with the strongest distinct alternatives. Loudness,
   player-name frequency, historical views and model confidence cannot substitute
   for this comparison. Do not fabricate objective audience-appeal scores.
3. **Hear the cut in context.** Listen to each finalist and its surrounding source
   before locking boundaries. Check clipped syllables, names, numbers, negation,
   sarcasm, pronouns and whether omitted context changes the claim. Preserve the
   setup/reaction needed for the payoff. Expand, repair or reject a weak boundary;
   never mark proportional transcript slicing as verified speech alignment.
4. **Make one honest promise.** The opening, title and cover must describe the same
   retained moment, and the edit must deliver that promise. No invented hook,
   misleading reaction frame, unrelated score, forced CTA or static cover intro.
   A longer complete thought beats a shorter misleading fragment.
5. **Review the final file, not just the plan.** Decode and check the entire output
   for technical failures; inspect motion/layout and listen through the complete
   encoded pilot. Check captions against speech, mouth/head visibility, crop
   stability, readable screens, audio continuity and the last word. Still frames,
   successful FFmpeg exit and loudness measurements alone cannot pass this gate.
6. **Separate evidence from unknowns.** Each package records source ranges,
   alternatives considered, coverage gaps, transcript corrections, exact edit map,
   actual QC results and reviewer/method. Gates have PASS / FAIL / NOT CHECKED;
   missing evidence is never PASS. Automated checks must state their limitations.

Release decisions: **READY** only when every mandatory gate passes; **REVIEW** for
uncertainty or incomplete inspection; **REJECT** for an unsuitable moment or a
failed gate. REVIEW and REJECT are ineligible for scheduling/upload. A batch with
zero READY clips is a legitimate editorial result, but must explicitly report
zero deliverables, reasons and coverage—not pretend export/publication succeeded.

For the controlled pilot, the owner reviews each complete encoded package on a
phone with sound before release. Record approval against the exact media and
metadata hashes; any subsequent edit invalidates that approval. This is a
proposed implementation contract, not a claim that these gates exist today.

### Selection and timing policy

- Discovery reads overlapping bounded blocks across the whole transcript, then
  ranks distinct events globally. Report any unprocessed blocks.
- Require one understandable claim or exchange, necessary context and an actual
  payoff. Reject greetings, mic troubleshooting, score recitation with no point,
  dangling pronouns, unfinished claims and unrelated chat pivots.
- Audio peaks are candidate hints, not votes that shouting equals quality. Quiet
  insight can beat a loud filler phrase.
- Arbiter rejection remains rejection. No forced minimum; allow zero publishable
  clips. Keep complete-thought and source boundaries hard gates.
- Keep natural speech at 1.0x for the first corrected pilot. No universal 15 s
  target. Test complete 10–20 s and 20–35 s moments as cohorts, not forced cuts.
- Prefer a continuous cut. Remove pauses only with speech/VAD evidence and
  semantic safety; preserve breaths, anticipation, sarcasm and reactions.
- Represent all retained parts and speed transforms before rendering; remap both
  captions and audio/video from that same map. Never cut a finished captioned MP4
  as an independent cleanup pass. Use short audio edge ramps where appropriate.

### Transcript curation example from this source

The 84.56–120.32 s neighborhood contains a coherent argument: a 260 chase looked
daunting, but the speaker compares the teams around 12 overs and discusses the
short leg-side boundary. This is a better *review candidate* than beginning at
`But come on yaar thoda` or publishing a disconnected list of scores.

Provisional package: **title:** `260 Ka Darr? India Ki Chase Par Meri Baat`;
**cover:** `260 KA DARR?`; **description:** `IND vs NZ ki chase par meri rai:
sirf target nahi, run rate aur boundary size bhi dekho. Hindi cricket analysis,
not match footage.` No claim here that the spoken boundary lengths or scores are
verified match facts. Exact in/out points require source reacquisition, alignment
and listening. Do not publish this as a finished selection.

The 320–350 s neighborhood mixes chase criticism, score comparison, a viewer's
SKY request and a rhyme. Split by editorial meaning, not sentence punctuation
alone. The 1668–1687 s argument also needs its preceding premise; `theek hai?`
is a poor standalone opening. These are text-level judgments, not playback review.

## 6. Framing and caption design contract

Pilot composition (engineering targets to validate, **not YouTube rules**):

- Full hair/head and chin visible, some shoulders/body; aim for crown headroom
  around 6–10% of output height and eyes around the upper third. Adjust for pose
  rather than forcing a face-box top percentage.
- Constrain the complete head envelope across sampled motion, not a single face
  detection. If the available camera crop cannot provide room, fit it inside a
  designed canvas with a restrained background; do not zoom harder or invent head
  pixels. Maintain identity-safe existing automation detection, not Face OS.
- Stable shot-level framing, scene-change resets and confidence hysteresis. No
  animated zoom just because the detector jitters or a hand enters.
- Separate layouts for solo analysis, two-person debate and scorecard/screen
  explanation. Preserve readable scorecard content; do not stretch two faces.
- Plan a reserved caption band **after** face placement. Expand face/mouth boxes
  by a safety margin; measure the actual rendered text/card and reject collision.
  If there is no safe band, reduce face scale/recompose rather than cover a mouth.
- Account for right-side controls, bottom title/handle and device-specific UI;
  start conservatively with ~10% right and ~18% bottom exclusion, then inspect on
  real phones. These are test masks, not permanent platform guarantees.
- Roman Hinglish should transliterate actual speech, not rewrite it into punchy
  copy. Preserve English, numbers, negation and names; uncertain words get flagged.
- Prefer word-aligned 2–4-word phrases, one or two lines, plain bold type with
  restrained highlight. Avoid the current large opaque/glowing mouth-level card.
- Local bundled fonts; deterministic rendering. Do not silently depend on Google
  Fonts loading before a screenshot. Clamp cues to retained parts with no overlaps.

Acceptance: zero clipped head envelope at representative poses; zero face/mouth
or UI-mask caption collisions; readable 120 px-wide thumbnail and phone preview;
sampled caption timing within ~150 ms of checked speech as a pilot target.

## 7. Thumbnail strategy: a separate editorial product

Generate a clean cover from the **source**, not from an already captioned frame.
Shortlist 12–24 frames inside retained footage, reject blur/eyes-closed/obstructed
face and poor exposure, then review the expression in context. Do not label a
sharpness metric as emotion recognition.

Use one genuine reaction/subject, one idea, at most three words, deep blue/dark
background and white/saffron text. One small brand mark maximum. Leave the face
and mouth clear. No fake match action, stock player photo implying visible match
footage, generic `CRICKET LIVE`, or automatic title-prefix truncation.

Ten conditional cover directions (only use when the spoken story supports it):

| Hook | Genuine expression / composition | Color |
|---|---|---|
| `260 KA DARR?` | Concerned host on right, number-led text left | Navy + saffron |
| `CHASE KYUN TOOTI?` | Disappointed host, clean left question | Navy + white |
| `PLAN GALAT THA?` | Questioning host, no unrelated player | Navy + yellow |
| `YEH TURNING POINT` | Alert host, one verified event cue | Navy + saffron |
| `POWERPLAY MEIN GALTI` | Analytical host, one premise | Navy + white |
| `BOWLING KA PLAN?` | Thinking expression, generous headroom | Navy + yellow |
| `SKY WAPAS?` | Questioning reaction only if this is the actual exchange | Navy + white |
| `RUN RATE DEKHO` | Explaining gesture, one relevant number at most | Navy + saffron |
| `SIRF SCORE NAHI` | Firm expression, text opposite face | Navy + white |
| `YAHAN MATCH PALTA` | Genuine surprise after an established turning point | Navy + yellow |

First three to test after footage verification: `260 KA DARR?` (specific number
and argument), `RUN RATE DEKHO` (analytical promise), `SIRF SCORE NAHI` (curiosity).
All mobile checks remain **pending actual design**, not claimed passes.

Treat cover and opening frame separately. A better cover can help click-driven
surfaces; the first actual seconds still need to win the Shorts feed. Do not insert
a static cover intro to make it selectable.

Current official YouTube Help now documents custom 9:16 Shorts thumbnail upload
in **desktop Studio for verified accounts**. Do not repeat old mobile-only/no-
custom-cover advice. Existing code calls `thumbnails.set`, but generated, API-
accepted and visibly displayed are separate states: test on an authorized pilot,
read back and inspect Studio/surfaces. If not supported for this account/route,
deliver the cover plus a manual Studio step instead of claiming success.

## 8. SEO without VidIQ

### Removal scope for the implementation phase

Disable both `seo.vidiq.enabled` and `required`, remove its mandatory research,
title/keyword floors and fallback dependency from active generation, and prove
zero calls to its endpoint with tests. Remove/relabel VidIQ-only gates and audit
fields. Keep existing historical evidence; do not delete unrelated integration
files as a shortcut. Missing subscription must not block a good clip or trigger
generic inherited source metadata.

### What belongs in SEO

- **Evidence:** retained speech; canonical player/team spelling; clearly labeled
  source metadata; source date; independently verified match facts when needed.
  A VOD description alone is not authoritative confirmation of results or squads.
- **Intent:** one clip-specific idea plus one or two natural primary phrases.
  Own Studio search terms/research are first choice. Autocomplete and Trends are
  qualitative topic hints, **not keyword volume or a guarantee of demand**.
- **Title:** one clear local premise, usually 35–60 characters if it fits, natural
  Hinglish plus recognizable cricket entities. Entity presence is useful, but
  entity-first and mandatory emoji are not universal requirements. Generate a
  small candidate pool and select against the actual edit.
- **Description:** a unique short opening explaining THIS clip, useful context,
  optional source/related-video reference and a concise commentary disclaimer.
  Start with 1–3 useful sentences; longer text only when it adds genuine context.
  There is no evidence-backed 1,500/4,000-character minimum for Shorts reach.
- **Tags:** a few relevant names/aliases/misspellings; no 450-character floor or
  requirement to fill the API budget. YouTube explicitly says tags usually play
  a minimal discovery role.
- **Hashtags:** 2–3 relevant ones if useful, no unrelated trend hitchhiking. Never
  create long tag phrases from every generated search query.
- **Language:** align public copy with Hindi/English audience and actual speech.
  Remove contradictions between the English-only SEO contract and Hinglish AI
  configuration. Benchmark language alternatives instead of declaring a winner.
- **Fields:** `search_terms` is an internal research artifact, not a magic public
  `videos.insert` ranking field. Only valid mapped API fields reach YouTube.
- **Quality:** reject fabricated players/results, misleading `LIVE`, unfulfilled
  promises and generic inherited stream titles. Do not relax grounding to chase
  unrelated high-volume keywords.

The older `AUDIT_SHORTS_SEO.md` is not an authority: its minimum-length/tag-volume
recommendations and assertions about classifier penalties/CTR power words are not
established by the official sources reviewed here. Preserve the document as
history but supersede those recommendations.

### Title lab: twenty structures to evaluate, not fabricated finished metadata

Each placeholder requires clip evidence. Rank on truthfulness, clarity, local
specificity, curiosity and mobile readability; never fabricate numerical scores.

1. `[Target] Ka Darr? [Team] Ki Chase Par Meri Baat`
2. `[Team] Ka Run Rate Theek Tha—Phir Galti Kahan Hui?`
3. `Sirf [Target] Nahi: [Team] Ki Chase Ka Asli Sawaal`
4. `[Player] Ko Yeh Risk Lena Chahiye Tha?`
5. `[Player] Ki Is Baat Se Main Agree Nahi Karta`
6. `[Team] Ne Powerplay Mein Kya Miss Kiya?`
7. `[Bowler] Ke Khilaaf Plan Kya Hona Chahiye Tha?`
8. `Yeh [Team] Ki Batting Ka Problem Hai`
9. `[Player] Ki Wapsi? Meri Seedhi Rai`
10. `[Team] Ko Toss Nahi, Yeh Badalna Hoga`
11. `[Event] Par Mera Turant Reaction`
12. `[Team] Ki Chase Yahan Badli—Meri Analysis`
13. `[Player] Par Chat Ka Sawaal, Mera Jawaab`
14. `[Team] Ki Haar Ko Sirf [Cause] Mat Bolo`
15. `[Player] vs [Player]: Is Comparison Ka Sawaal`
16. `[Verified Number] Dekho, Phir [Team] Ko Judge Karo`
17. `[Team] Ko Yeh Darr Nikaalna Hoga`
18. `[Player] Ke Is Shot Par Meri Baat`
19. `[Team] Ka Plan: Main Kahan Disagree Karta Hoon`
20. `[Match] Mein Sirf Score Dekhna Kaafi Nahi`

For the transcript neighborhood discussed above, prioritize structures 1, 3 and
20 for pilot review. No claim of search-volume backing or measured CTR uplift.

## 9. DeepSeek choice, verified against current provider docs

As of this review, the official API lists:

| Role | Model ID | Why / limitation |
|---|---|---|
| Editorial shortlist and final source-grounded SEO | `deepseek-v4-pro` (DeepSeek-V4-Pro-0813) | Strong candidate for reasoning over context, setup/payoff and claims; text only, not image inspection. Suitability still requires Hindi/cricket benchmark. |
| Cheap first pass, transliteration candidate, sampled-frame review | `deepseek-flash` (DeepSeek-V4.1-Flash-1017) | Newer Flash line, image input supported; not a video/audio decoder or image generator. |
| Word timing, silence, dB and rendering | Local ASR/alignment, FFmpeg, detectors | Deterministic measured evidence; do not ask an LLM to guess decibels or speech times. |

Use the requested/effective exact model ID in logs and manifests. No silent alias
to `deepseek-chat`; no unrelated OpenCode credential prerequisite. Use explicit
thinking configuration: high for final editorial decisions, non-thinking where
appropriate for mechanical extraction. Parse final structured content, validate
schema/IDs/ranges, cap spend/retries, and queue on semantic failure. Do not race
19 models for every task or treat fastest response as best editor.

Official USD per million token rates (cache-miss input / output): Pro off-peak
**$0.66 / $1.98**, peak **$1.32 / $3.96**; Flash off-peak **$0.15 / $0.60**, peak
**$0.30 / $1.20**. Both list 1M context, which is not a reason to process every
frame. A 10k-input/2k-output Pro call is approximately $0.01056 off-peak or $0.02112
peak **before additional reasoning output, retries, image tokens and research**.
Search/fetch each list $0.75 per 1,000 successful requests. Recheck prices at build.

No paid model calls were made in this review. Before switching production, use
20–30 labeled Hindi/English cricket moments, including quiet analysis, loud filler,
sarcasm, rolling captions, negations and player-name errors. Measure complete-
thought precision, unsupported claims, opening usefulness, JSON validity,
transliteration fidelity, cost and latency. Pro is the provisional editorial
choice, not a benchmark-proven winner on this channel.

## 10. Audio and runtime analysis

- Measure source and final edited audio: integrated/short-term loudness, true
  peak, clipping/flat tops, noise floor, dropouts, channel balance and voice versus
  background. A single global RMS threshold cannot establish emotion or silence.
- Start with -16 LUFS integrated and -2 dBTP normalization target, then verify
  encoded output around -16 +/-1 LUFS and <=-1 dBTP. These are project delivery
  targets, **not a mandated YouTube ranking specification**. Preserve dynamics;
  do not repair already-clipped speech by turning it up.
- Two-pass normalization over the exact retained edit, short edge ramps, bounded
  filtering only when evidence warrants it. Review quiet speech and shouting on
  headphones and a phone. Keep natural voice pace during the first pilot.
- Instrument transcript fetch, ASR/alignment, discovery, ranking, visual review,
  render, QC, SEO and thumbnail stages separately: wall time, retries, token cost,
  cache hit, encoder and peak memory. Report p50/p95 from real runs, not guesses.
- Cache reusable transcript/evidence by source hash and model/prompt version;
  coarse discovery first, precise alignment only for shortlisted neighborhoods.
  Pilot target: three reviewed-quality packages from a cached ~75-minute source
  within 15 minutes on this PC, **unbenchmarked**, with stage budgets established
  by measurement. Do not sacrifice semantic correctness to meet this target.

## 11. Outcomes: what the current evidence can and cannot say

The local cricket subset's latest-snapshot median public views is 138. Several
August 23-second discussion uploads show roughly 9–13 seconds average duration
and ~42–58% viewed. These are historical observations, not today's baseline or
proof that one title/edit caused poor performance. Some higher-engaged examples
have a specific player conflict or analytical premise; their ages and formats
differ, so this is a hypothesis, not a causal packaging result.

Refresh the exact owned channel's Shorts shelf and owner outcomes before tuning
the learner. Distinguish missing data from zero. Recheck active learner freshness
and calibration before allowing adjustments; preserve the existing bounded
tie-breaker and never use it to change boundaries or waive completeness.

Track outcomes at comparable ages (24 h, 72 h, 7 d): engaged views, watch duration,
average percentage viewed, feed shown/stayed-to-watch where actually available,
likes/shares, subscribers per 1,000 engaged views and traffic-source split.
Do not call `engaged/views` a swipe rate. Use Studio exports where the API does
not expose a metric, clearly labeling origin and coverage in the canonical store.
Thumbnail CTR belongs to eligible impression surfaces, not all feed playback.

Do not hard-code 'Thursday is dead', mandatory hourly uploads, or time jitter for
'algorithm love'. Evaluate scheduling after quality, using fresh matched cohorts.
Timely post-match moments should not become stale because of an old weekday rule.

## 12. Phased implementation and release gates

| Phase | Work | Required exit evidence |
|---|---|---|
| A: trustworthy baseline | Isolate dry run; fix real failing non-Face-OS tests; preserve existing work; merge local baseline, then create recovery branch | Tests pass without weakening assertions; dry run checks nonzero expected exports and accurate errors; no input writes; scoped commit and clean diff |
| B: timeline/editorial | Source IDs/hash, precise times, edit map, semantic rejection, no forced backfill, alignment; eliminate post-render cleanup cuts | Golden fixtures for rolling captions, speed changes, pauses, end words, negation, duplicate events and zero-output cases |
| C: picture/captions/covers | Shot layout/head envelope, collision-free captions, clean source-frame cover shortlist, bundled fonts | Before/after images plus full playback of at least three distinct cricket moments; zero visual contract violations |
| D: SEO/model | VidIQ-free route, exact DeepSeek IDs, source-grounded copy, sane metadata gates and honest completion status | Zero VidIQ calls; DeepSeek-only credential case; fabricated entity/score rejection; valid API payload; benchmark and cost manifest |
| E: controlled pilot | 5–10 complete packages across several source videos; approve quality before using existing publishing route | No test-triggered upload, live metadata overwrite or scheduled-publication changes; evidence attached to each deliverable |
| F: measured iteration | 20–30 comparable new Shorts, modest pace; change one creative variable per cohort | Age-matched channel comparison; no claims based on a single outlier; feedback only in canonical learner |

Run the repository-mandated non-Face-OS unit suite and isolated dry run before
every code commit; include automation and Shorts Intelligence tests where affected.
Do not change, run or repair `face_os/tests/` for this work.

### Experiments and rollback

| Experiment | Impact / confidence / difficulty | Risk | Test / metric | Rollback |
|---|---|---|---|---|
| Single edit map + semantic gates | High / high mechanism, medium reach / high | Fewer outputs | Golden timeline checks; zero clipped words or unexplained timing changes | Keep source and versioned plans; disable new selector, never restore unsafe cuts |
| Wider headroom + reserved captions | High / high visual, medium reach / medium | Smaller face/text | Phone playback, collision QC, comparable AVD/% viewed | Previous validated layout preset, not known mouth-blocking one |
| Clean 2–3-word moment cover | High packaging / high defect, medium CTR / medium | Wrong expression or promise | 120 px check, Studio delivery, eligible-surface CTR cohort | Prior approved cover; no repeated automated swapping |
| VidIQ-free Pro metadata | Medium / high removal feasibility, medium copy / medium | Hallucinations, cost | Claim audit, credential tests, model benchmark, search-source outcomes | Queue metadata; use approved extractive copy, never generic stream-title inheritance |
| Natural pace vs mild speed | Medium / medium / medium | Slower delivery or unnatural voice | Comparable complete-thought cohorts, AVD and comprehension | 1.0x |
| Fresh outcomes vs stale heuristics | High / high freshness need / medium | Sparse/confounded data | Freshness, missingness, holdout calibration | Shadow-only bounded policy |

A practical pilot win target is zero critical edit/visual defects plus a ~20%
relative improvement in age-matched median engaged views without worse retention
or subscriber conversion. This is an experiment decision threshold, not a promised
result or a YouTube algorithm threshold. Sparse results remain inconclusive.

## 13. Research references and evidence strength

Official sources fetched/researched on 2026-11-01:

1. [YouTube Shorts search/discovery](https://support.google.com/youtube/answer/11914225?hl=en&co=YOUTUBE._YTVideoType%3Dshorts): choice to view, watch duration/% viewed, satisfaction, topic/competition; no minimum cadence.
2. [YouTube tags](https://support.google.com/youtube/answer/146402?hl=en): minimal role outside misspellings; do not stuff description tags.
3. [Description guidance](https://support.google.com/youtube/answer/12948449?hl=en): unique descriptions, useful opening and 1–2 main terms; no mandated minimum length.
4. [Current custom thumbnail guidance](https://support.google.com/youtube/answer/72431?hl=en): desktop Studio Shorts upload, verified account, 9:16 recommendation.
5. [Thumbnail API](https://developers.google.com/youtube/v3/docs/thumbnails/set): upload contract and error states; account/surface verification still needed.
6. [DeepSeek models/pricing](https://api-docs.deepseek.com/quick_start/pricing), [vision](https://api-docs.deepseek.com/guides/vision), [thinking](https://api-docs.deepseek.com/guides/thinking_mode): current IDs/capabilities/rates, not a channel-specific quality benchmark.
7. [FFmpeg filters](https://ffmpeg.org/ffmpeg-filters.html#loudnorm): measured normalization; local encoded measurements are the actual evidence here.

Secondary research, hypothesis generation only:

- [NewTubers: common thumbnail/Shorts editing mistakes](https://www.reddit.com/r/NewTubers/comments/1ptvcoh/what_small_creators_usually_get_wrong_with_shorts/)
- [NewTubers: reviewing failed Shorts](https://www.reddit.com/r/NewTubers/comments/1qfg3b9/went_through_100_shorts_to_figure_out_why_most/)
- [IndiaCricket: commentary complaints](https://www.reddit.com/r/IndiaCricket/comments/1943qdy/annoying_hindi_commentary/)
- [IndiaCricket: analysis versus wordplay](https://www.reddit.com/r/IndiaCricket/comments/1rlk1h2/after_aakash_chopras_tweet_about_a_paid_campaign/)

Repeated themes are relevant analysis over off-topic/shouted filler and readable,
uncluttered packaging. These are self-selected anecdotes, not representative
channel research; ignore claimed universal seconds/percent thresholds. Public
ESPNcricinfo/Aakash Chopra search examples use specific local questions such as
`Was India's bowling better than everyone thought?`; borrow specificity, not
their reach, celebrity authority, imagery or claims of causal success.

**Bottom line:** fix what viewers actually experience and make the pipeline prove
what it produced. Better SEO and a better model help only after a complete,
well-framed, intelligible, correctly captioned moment exists.
