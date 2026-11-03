# Editorial quality implementation plan

Baseline: `e9b5d4a`. Scope: cricket Shorts only; no Face OS changes or publishing.

## Active visual/caption/thumbnail contract (2026-11-02)

**Latest integration audit:** [runtime/prompt/cleanup findings](docs/editorial/INTEGRATION_AUDIT_20261102.md).
39 new regressions; final suite 1,083 passed,47 skipped,18 xfailed. Required,
full-ingestion and fake-delivery dry runs PASS; real synthetic Chromium/FFmpeg
smoke PASS. 459 source/transcript/asset/review files preserved by hash. Face OS
excluded; no publication/commit. V4 remains review-only, not the production layout;
remaining transport, telemetry and editorial limitations are explicitly recorded.

**Latest packaging refinement:** the creator prioritizes whole-package quality.
`shorts/aurora-pilot-20261102/review/packaging-v4/` keeps the authorized best-effort
crop/approved captions, increases the opening topic headline, proves three real
caption-free app-cover frames and adds a listening-first opening comparison.
`automation/packaging_design.py` provides tested source-faithful proof compositors;
it does not change production defaults, publish, approve or learn from generated
scores. See the package's `PACKAGING_DECISION.md` for exact requirements and limits.
Speech and release gates remain unresolved; older status/counts below are history.

Read [the detailed saved execution plan](docs/editorial/pilot-visual-plan-20261102/PLAN.md)
before resuming the pilot. It pins the supplied reference, inspected current
thumbnail/source/frame evidence, proposed crop/card geometry, exact V4 Flash
requirements, temporal-cut checks, thumbnail procedure and anti-drift ledger.
Status: **Aurora visual rebuild rendered; technical validation and cleanup complete;
release held**. See [actual results and remaining gates](docs/editorial/pilot-visual-plan-20261102/RESULTS.md).
Final mandatory validation: **1,013 passed,47 skipped,18 xfailed**; required dry run
PASS. Actual export full decode/audio/timestamp QC PASS. Speech errors and actual
playback/phone/owner review remain unresolved. The later user brief authorizes a from-scratch futuristic
design. Live provider discovery also found V4 Flash retired; its documented V4.1
successor is explicitly disclosed, never represented as original V4. Review images
are preserved alongside the plan. Static inspection is not listening/owner approval.

## Implemented checkpoint

- Steps 1–3 below are implemented with red -> green regression evidence.
  Word-based discovery retains source intervals/confidence and rejects malformed,
  overlapping or out-of-parent-cue evidence. Cue-only cuts remain estimates.
- Discovery reserves source-time alternatives, then fills by score. The arbiter
  compares up to 60 admitted candidates independently of output count, supplies
  per-candidate context, and discloses unexplained/omitted decisions. A cap is a
  cost bound, not proof of whole-source editorial coverage.
- Final overlap checks apply to both AI and heuristic-only selection. Source cue
  IDs, original cue text (explicitly context, not exact-cut captions), boundary
  methods and editorial rejections survive into review evidence.
- RMS audio uses invocation-owned scratch. Dry-run sandboxes clean up by default
  on success/failure; Windows log handles are closed without closing live logs.
  Debug retention: `$env:YT_CLIPS_KEEP_DRY_RUN = "1"` (unset after debugging).
- Final required suite: **970 passed, 47 skipped, 18 xfailed**. Required sandbox
  dry run: **2 selected, 2 exported, 2 valid metadata, 10 events, 0 failures**;
  its sandbox was automatically removed. No skipped/xfail tests were added.
- Real transcript structural smoke check: 1,729 cues -> 777 discovery units ->
  383 eligible windows -> 25 non-overlapping shortlist alternatives. All retain
  cue provenance; all have estimated boundaries because this input has no word
  timing. This is not an audio/AI/editorial benchmark or a new approved pilot.

## Execution order and acceptance gates

1. **Source-safe clipping (TDD).** Preserve original word intervals, confidence
   and cue identity through discovery. Split word-timed blocks only between
   words; never estimate sentence boundaries from token ratios. Keep pause
   boundaries from being immediately merged away. Label cue-only interpolation
   as discovery estimates, never verified speech or semantic completeness.
2. **Defensible selection (TDD).** Keep temporally distributed source-time
   alternatives in a capped shortlist. Compare the whole admitted shortlist,
   not only `max_selected + 5` heuristic winners (60-candidate cost bound with
   explicit omissions). Supply local preceding/following
   context for each candidate. Hold truncated scripts, invalid scores and
   contradictory decisions; deduplicate overlapping final picks. Zero is valid.
   Persist comparison scope, omissions and boundary provenance.
3. **Scratch ownership (TDD).** Use isolated temporary audio per invocation;
   cleanup on extraction failure, decode failure and success. Dry-run sandboxes
   clean up by default; explicit retention is for debugging. Never remove source,
   corrected transcripts, final exports, approval or review evidence as scratch.
4. **Validation.** Demonstrate red -> green regressions, run existing selection,
   clipping and cleanup tests, mandatory full suite and sandbox dry run. Verify
   source hashes and inspect the diff. Report failures; do not mask them.

## Subsequent quality gates (not certified by unit tests)

### Caption listening preparation

Validation after this increment: **981 passed, 47 skipped, 18 xfailed** in the
mandatory full suite; mandatory dry run **PASS**, 2 selected/exported, 0 failures,
sandbox removed. Eleven new packet regressions cover range/ID validation,
escaping, source/cue identity, failure cleanup and preserving existing review work.
Automation + canonical learner + packet subset: **98 passed**.

`automation/caption_review.py` prepares bounded, source-hashed listening cases
from a manifest. It never aligns corrected text, approves captions, changes the
source transcript or connects corrections to publication. Existing packets are
never overwritten. Extraction uses owned scratch and preserves no partial packet
on failure. HTML escapes source cue text; invalid cues are explicitly disclosed.

Real packet: `shorts/recovery-pilot-20261101-172917/caption-listening/index.html`.
Four cases span 80–135 seconds (each includes two seconds of listening context).
All audio files passed full decode, expected duration and SHA-256 checks.
Original transcript cue IDs 1375, 1382, 1392 and 1395 have invalid zero-duration
intervals and are disclosed/excluded, not repaired. The transcript is unchanged.

Open `index.html`, listen to the exact requested range, and record its verbatim
speech in the matching `listening.json` case's `corrected_text`. Record uncertainties
instead of guessing names/numbers; retain NOT CHECKED until actual listening.
Reviewer and listening date start blank. Context cues can overlap and contain
errors: never paste them wholesale into captions. The JSON is review work, not a
validated corrected transcript. Correction import, word alignment and independent
fixture scoring are still pending; this is not yet a reviewed benchmark.

Reproduction (use a new output directory to preserve existing corrections):

```powershell
.venv\Scripts\python.exe -m automation.caption_review input/recovered/nrl5qQD0E98.mp4 transcripts/video.json tests/fixtures/caption_review/recovery-pilot.json NEW_OUTPUT_DIRECTORY
```

- Caption fidelity: listening-reviewed Hindi/Hinglish fixture set, including
  names, negations and scores; correction before alignment, no invented text.
- Composition: whole-clip framing/UI collision fixtures and phone-sized review;
  test varied camera motion and scorecards before changing layout defaults.
- Editorial benchmark: independently annotated full sources; measure strong
  moment recall, context damage, critical-caption errors and correction time.
  Synthetic fixtures demonstrate mechanisms, not audience quality.
- Delivery pilot: compare source alternatives, correct one real Short, play the
  entire export with sound, obtain actual owner/phone review. Keep REVIEW until
  performed; never fabricate listening or approval evidence.
- Outcomes: refresh owned-channel Analytics and compare matched topic/length/age
  cohorts. No generated score counts as an outcome; no second learning database.

## Non-goals

No upload quota, viral guarantee, wholesale renderer replacement, automatic
rewriting of unclear speech, expensive model expansion, or blanket temp deletion.
