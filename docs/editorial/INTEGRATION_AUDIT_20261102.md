# Cricket Shorts integration audit — 2026-11-02

Scope: runtime orchestration, ingestion, selection/export handoffs, caption and
packaging prompts, covers, release boundaries and disposable scratch. **Face OS
and its configuration are excluded.** No commit, push, live API experiment,
learning refresh, real upload, Drive sync or scheduling was performed.

## Findings fixed

| Finding | Change / regression evidence |
|---|---|
| Failed Drive pull or selection could continue with old local inputs/highlights | Required-stage abort; zero selection is a valid no-export result |
| Resume scanned recursively and guessed a batch lexicographically, including nested review videos | Direct-batch discovery only; ambiguity requires `--skip-export --export-dir shorts/BATCH`; nested review/escaping paths rejected |
| Database backup ran without explicit sync, including `skip_sync=True` | Both folder and DB remote sync require `auto_sync and not skip_sync` |
| Failed delivery QC still allowed downstream remote work | Abort before covers, sync, scheduling or upload; retain failed paths for diagnosis and count held releases |
| Thumbnail batch ignored its enable flag and failed-metadata markers | Honor enable flag and shared `read_metadata` validity contract |
| Resuming could overwrite an existing owner-selected cover | Existing covers are preserved; explicit rebuild is required |
| Legacy frame extraction overwrote/deleted adjacent `clip.jpg` | Invocation-owned temporary directory; cleanup on success/failure; bounded FFmpeg timeout |
| Missing cover copy silently produced a blank source cover | Hold before frame extraction; never infer it from the first three title words |
| Central SEO prompt contradicted runtime rules with forced essays, hashtag quotas and unsupported algorithm claims | Shared `prompts.SEO_SYSTEM` used by runtime; concise source-grounded public template and complete escalation schema |
| Runtime forcibly appended an emoji despite its optional-emoji prompt | Keep deliberate title; preserve an explicitly supplied emoji; remove dead forced-emoji helper |
| Supplied cover copy lacked the public-copy checks | Validate one to three visible Roman words; reject empty/control-character/malformed copy and include cover in deterministic entity checks |
| Compatibility ranker prompt excused missing context for exciting clips | Necessary setup/payoff outside the cut is disqualifying; no excitement exception |
| Grounding/analytics prompts blurred evidence and model judgment | Explicit untrusted-data boundary, retained-transcript primacy, uncertainty and advisory-only outcomes guidance |
| Dry-run transcript stub used `end` where upstream API requires `duration` | Correct external response shape; complete-ingest path is now tested, not just skip-transcribe |
| Empty remote transcript could prevent Whisper fallback | Try the next source on missing/empty segments; unavailable only after all routes fail |
| VTT downloader searched/deleted files in shared temp and could parse its own empty placeholder | Owned directory only; no placeholder and no shared glob; parse only nonempty successful outputs; clean all exits |
| Failed local ASR could reuse a previous destination transcript | Transcribe into owned scratch; publish only fresh nonempty output, preserving old destination on failure |
| whisper.cpp reused/deleted fixed raw-output filenames | Unique configured-temp directory per call; preserve existing raw JSON; cleanup success/failure/exception |
| SDK video/thumbnail handles remained open after upload, breaking Windows cleanup | Close streams on success, request-construction failure, execute failure and thumbnail failure; real local handles with stub service tests |

New tests: `tests/automation/test_integration_safety.py`,
`tests/automation/test_transcript_ingestion_safety.py` and
`tests/automation/test_upload_resource_safety.py`. Existing title tests were
updated to the optional-emoji contract, not skipped. New regressions reproduced
failures before fixes. No new skip/xfail rules were introduced.

## Integration map: what is actually connected

1. `automation.cli` → `automation.orchestrator.run`: explicit upload/sync intent;
   required-stage failures stop dependent work; cached-export resume is scoped.
2. `automation.transcript.fetch`: official captions → transcript API → owned VTT
   → local ASR. Empty output cannot establish success. Returned source must still
   undergo listening review; successful acquisition is not speech certification.
3. `automation.clip_selection.pipeline` → selector/arbiter: cricket scope, diverse
   candidates, local before/after context, bounded comparisons and preserved
   boundary provenance. Text-only judgments cannot establish phonetic cut safety.
4. `export.export_all` → word evidence → caption engine → FFmpeg: required-caption
   path does not accept cue-proportional timing. Per-ID Flash romanization retains
   number/negation/model provenance; lexical proposals remain unreviewed. The
   optional legacy cue-only route is still approximate and is NOT a fidelity gate.
5. `automation.seo.seo`: runtime uses the shared system prompt, its richer grounded
   user schema and an escalation preserving `cover_text`. Compatibility templates
   existing in `prompts.py` do not prove that those prompts run in production.
   `ai.system_prompt` is compatibility config, not the active SEO system prompt.
6. `thumbnail`: clean source-derived cover with retained shortlist evidence;
   technical sharpness ranking is not expression/CTR review. Existing covers are
   not rebuilt implicitly. A JPG is not proof of an assigned Shorts thumbnail.
7. Delivery QC → hash-bound owner review → explicitly authorized publication.
   Real-service operations were intercepted during tests; no actual publication.
8. `shorts_intelligence` remains the sole persistent learner. Generated SEO,
   hypothesis scores, rendering success and dry-run outputs are not outcomes.

### Important: V4 is not the production layout

`shorts/aurora-pilot-20261102/review/packaging-v4/` is still a separate, preserved
review package. Its full-width picture, manual pan anchors, y1304 caption
transport, 68px opening headline and frame-477 cover proof are **not** silently
wired into `export_all`. The general renderer currently uses the reserved-band
composition with caption transport y1216. The pinned caption renderer is
`neon-reference-v2`; the stale config style label was corrected without changing
geometry or typography. The source-cover compositor remains separately versioned
`aurora-v1`, not a falsely claimed V4/app-cover integration.

## Validation evidence

Baseline before fixes: **1,044 passed, 47 skipped, 18 xfailed**.
Final: **1,083 passed, 47 skipped, 18 xfailed** in 34.24s;
all **39 new regressions passed**. `git diff --check` passed.

All three sandbox scenarios passed with zero pipeline failures and automatic
cleanup: required dry run `4a269f30`, full ingestion `effa88c8`, and fake
upload/sync/scheduling `c805129e`. Each exported two fixture clips; the first two
uploaded zero, the third reported two **stubbed** uploads. Real remote operations: zero.

Final results are recorded in `reports/integration-audit-20261102/`:

- `full-tests.log`: mandatory suite, excluding `tests/test_face_detect.py`.
- `pipeline-dry-run.log`: mandated skip-download/skip-transcribe scenario.
- `full-ingest-dry-run.log`: valid-ID download/transcript/selection/export scenario.
- `delivery-stub-dry-run.log`: upload/sync/schedule branches with external services
  stubbed. Reported fake uploads must not be read as real publication.
- `real-render-smoke.json`, `real-render-verified.log`: actual Chromium, bundled
  font, neon cards, FFmpeg filter graph/H.264/AAC/full decode/loudness/true-peak QC.
  Synthetic words, source and crop; no live ASR/LLM or speech-accuracy claim.
- `preserved-before.json` and `preservation-check.json`: exact identities of all
  459 pre-existing source/transcript/highlight/asset/export/review files.
- `cleanup-result.json`: exact deleted paths, sizes, hashes and reasons.

The real smoke's initial 3s fixture correctly failed the configured 8s minimum;
the 10s replacement exercised the normal policy without lowering it. The capture
helper's `.6` seek syntax was corrected to `0.6`; the successful render was reused,
not repeated. Logs retain these failed checks. Browser frame was visually
inspected as a synthetic mechanical proof, not an editorial approval.

Python bytecode writing was disabled and pytest's cache provider was disabled
during validation; the Face OS suite was not invoked. Existing skipped/xfail
coverage remains a limitation, not a pass for those tests.

## Cleanup boundaries

Removed **213 allowlisted cache/throwaway-media files, 26,905,057 bytes**. Temp-media
basenames were checked against source code, docs, transcripts and retained review
metadata; none were referenced. Cleanup deletes exact files only, refuses links
and excludes Face OS before traversal. No broad recursive data deletion.
Additionally removed 10 files / 3,013,228 bytes created by this audit's failed
synthetic test and abandoned locked sandbox; original failure logs retained in
`own-test-cleanup.json`. Total reclaimed: **29,918,285 bytes (28.5 MiB)**.

Kept `temp/test_captions.ass`, `temp/whisper_rocm_out.json` and
`temp/whisper_vulkan_out.json`: caption text and raw ASR are evidence, not
automatically junk. Kept all source media, corrections, packaged review originals,
proofs, outputs, models, secrets and the canonical database. Runtime-generated
cover candidate directories are review evidence and were not swept as caches.

## Remaining limits / release decision

**Code validation is not product/editorial readiness.** Current pilot remains
REVIEW: disputed speech/numbers, corrected-text alignment, whole-source strongest
alternative/complete-thought boundaries, complete-cut sound playback, actual
phone/YouTube-app fit and final exact-artifact creator approval remain open.

No live provider, fresh owned-channel Analytics, actual ASR engine quality or
account-specific thumbnail assignment was verified. Prompt hardening cannot prove
semantic fidelity or defeat every prompt injection. General export still needs
a separately reviewed integration decision before adopting V4's manual packaging.
The SEO internal search-query output still follows the approved-seed order rather
than fully preserving a model-proposed ranking; public copy is not stuffed with
that list. This pre-existing behavior was reviewed but not expanded in this
safety/cleanup patch.

Two pre-existing runtime limitations also remain explicit:

- `upload_video` still uses a non-resumable `.execute()` call; configured upload
  retry/deadline values are not enforced by that transport. Closing handles does
  not establish resumable-upload/recovery reliability. Older architecture prose
  overstates this; a focused transport/retry change is still needed.
- `PipelineResult.transcript_source` still reports `none` on the full-ingestion
  path despite successful acquisition. The acquired transcript's own source
  remains available; this is inaccurate summary telemetry, not a failed fetch.
