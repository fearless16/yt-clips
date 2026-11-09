# Transcript-first Shorts curation

**Visual-only update (2026-11-04):** the owner's subsequent portrait reference
supersedes the old crop/neon appearance. See [PORTRAIT_REFERENCE.md](PORTRAIT_REFERENCE.md).
It does not change the unresolved editorial/caption-wording findings below.

Goal: **2–3 distinct, complete local drafts per substantive cricket video**.
This is an editorial target, never permission to fabricate speech, force a weak
clip, weaken caption fidelity, or bypass owner listening/release review.

## Why the previous green checks were insufficient

Run `22e9299c` discovered 24 candidates, passed only 8 to V4 Pro **low**,
selected two estimated cue cuts, and discovered unusable word evidence during
export. There was no repair or reviewed reserve queue. The dry-run sandbox
explicitly disabled `require_word_timing`, hiding that production path.

The saved word evidence also contained zero-duration words and garbled ASR,
not just crossing boundaries. Deleting those words or moving a timestamp until
a validator turns green would not establish faithful speech.

## Current execution contract

1. **Full original-script coverage.** Existing heuristic discovery and scores
   remain evidence. The curator also reads every original timestamped transcript
   cue, before the legacy minimum-score cutoff. Long input is sectioned with
   overlap. Section-budget overflow is explicit review, not silent truncation.
2. **High-thinking proposals.** The configured provider/model is used without
   fallback. Editorial calls explicitly request high thinking; native DeepSeek
   also receives `thinking.type=enabled`. Proposals refer to original cue IDs,
   exact opening/closing phrases, and specific setup/payoff rationales. The
   default route is `deepseek/deepseek-v4-pro`.
3. **Word preflight before rendering.** Only shortlisted ranges are transcribed.
   Exact source phrases locate word boundaries. Malformed/low-confidence fresh
   ASR gets at most one explicit fallback-model/wider-context retry. Supplied
   word evidence is not overwritten. No word deletion, confidence inflation,
   proportional timing, or guessed source-name correction is permitted.
4. **Fresh final editorial review.** Changed cuts invalidate completeness.
   V4 Pro sees the actual retained words and selects up to three drafts plus
   ranked reserves. Attestations must be boolean, source-range-bound, explained,
   and noncontradictory. Overlapping moments cannot fill multiple slots.
   If exact caption/ASR phrases disagree after the explicit ASR retry, an
   additional high-thinking boundary pass may choose a contiguous span by
   actual word IDs. Its result is still UNVERIFIED until the separate final
   review. It cannot rewrite speech, omit internal words or invent timing.
5. **Bounded recovery.** If fewer than two qualify, one additional transcript
   search is allowed, subject to the same global call/alignment budgets. Export
   uses reviewed reserves only when a primary fails; successful siblings are
   retained and not rerendered. Unrecovered batch errors carry actual successful
   paths back to the orchestrator.
6. **Independent downstream gates.** Caption model identity/fidelity, encoded
   A/V checks, delivery QC and hash-bound owner review remain separate. Text
   approval and ASR confidence are not listening or visual verification.

## Artifacts and resumption

- `highlights/video.curation.json`: source identity, source-keyed final provider
  JSON responses, call attempts, holds, selection and reviewed reserves. No
  private model reasoning is used as a final answer.
- `highlights/.speech-cache/speech-<hash>.json`: immutable raw ASR attempts keyed
  by source path/size/mtime, range, model, language and context.
- `highlights/video.yaml`: exact numeric cut points, retained speech, source word
  evidence, primary/reserve role and range-bound completeness.
- Export batch `*_speech.json`, `*_edit.json`, caption evidence and
  `batch_failures.json`: actual per-clip outcomes, including recovered failures.

Changing source identity, transcript/prompt, route or editorial settings prevents
reuse of an unrelated LLM decision. Cached ASR is revalidated before use. These
artifacts are pre-publication evidence, **not channel outcomes or a new learner**.

## Testing: failure first, not green by construction

`tests/automation/test_editorial_recovery.py` and
`tests/automation/test_transcript_curation.py` exercise real alignment validators,
curation, completeness, export recovery and dry-run acceptance. Only external
ASR, provider, rendering and other dependency boundaries are substituted.
The recorded failed run has timestamp/word excerpts under
`tests/fixtures/pvA10GoCS9U_alignment_failures.json`; they remain unapproved.

The dry run now keeps required word captions enabled, exercises native JSON
request construction at the mocked SDK boundary, injects zero-duration ASR that
must take the real repair route, runs real Flash response validation against
mocked HTTP, and checks actual speech/edit sidecars independently of counters.
Unexpected urllib/socket access fails instead of contacting real services.
Fake ASR instances cannot survive sandbox teardown, including first-import runs.

Real-source run `10699ddd` additionally exposed malformed **context-only**
words poisoning otherwise valid cuts, and Hindi relevance normalization gaps.
Those are separate regressions: finite malformed words strictly outside the cut
do not certify or invalidate retained speech; malformed words inside/at the cut
still hold it. Full Hindi player phrases are recognized without rewriting the
script or guessing misheard names. Boundary mismatch cannot silently become an
approved nearest-timestamp edit.

Deliberate breakage checks were executed in isolated Python processes without
editing production files:

- Removing boundary repair made the crossing-word regression fail.
- Downgrading high thinking made the actual SDK-payload regression fail.
- Erasing successful sibling paths made the partial-batch regression fail.

Mutation evidence: `C:\Users\user\AppData\Local\Temp\opencode\editorial-mutation-*.log`.

Required validation:

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v --ignore=tests/test_face_detect.py
.venv\Scripts\python.exe dry_run.py https://youtu.be/test --skip-download --skip-transcribe
```

These checks prove deterministic contracts and recovery, **not that an LLM's
editorial taste, ASR transcript, crop or live source is automatically correct**.
A real-source run plus final media inspection is still required. No test suite
can justify a promise that software will never require another fix.

## Real-source validation: incomplete, not a release success

Run `2b40f2d0` on `pvA10GoCS9U` produced **one draft**, not the target 2–3.
The run correctly ended `PARTIAL` with a nonzero exit code. No upload occurred.

Artifact: `shorts/2026-11-04_151858_d9bf65c0/clip1.mp4`

- Source range: 344.92–362.68 seconds; encoded duration: 17.766667 seconds.
- ffprobe confirms 1080×1920, audio/video start at zero, matching stream lengths.
- Delivery QC reports full-file decode PASS, -15.97 LUFS, -1.41 dBTP. These are
  technical checks, not semantic, listening or visual approval.
- A frame at output 2s was inspected at
  `C:\Users\user\AppData\Local\Temp\opencode\pvA10GoCS9U-final-clip1-check.png`.
  Existing crop/caption styling was not redesigned. The sampled frame retains
  source `CONTROLS` text and awkward literal ASR caption wording (`ER INDIA / EK
  KA`); it is not evidence of release-ready framing or caption fidelity.
- Metadata still needs review: the retained script refers to India A, whereas
  the title says `India vs New Zealand Test`; search terms include Mohammed
  Siraj without support in this retained script. `metadata_valid: true` is not
  proof of source grounding.
- Remaining options were held for invalid/missing cue phrase citations,
  excessive duration, cricket-relevance failures or unresolved word boundaries.
  A high-thinking response did not itself solve those contracts.
- Owner approval remains null. Full listening/phone playback and final
  editorial/caption/title-cover approval were not performed.

The implementation has passed deterministic regression, dry-run and deliberate
breakage checks, but **the end-to-end 2–3-clip objective is not complete**.
Do not describe this run as a finished fix or a publishable batch.
