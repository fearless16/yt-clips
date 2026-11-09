# Stadium / EA07 pipeline integration

Current stock delivery behavior is documented in
[`STOCK_GAMEPLAY_RUNTIME.md`](STOCK_GAMEPLAY_RUNTIME.md#current-delivery-behavior-2026-11-09).
Enabled stock is no longer draft-only; exact-file owner release remains required
for YouTube scheduling. The integration results and temp paths below are historical.

The approved v7 appearance is promoted into production, not just linked from a
private renderer. Existing final MP4s are unchanged. No real uploads, permission
approvals, purchases, installations, commits or pushes are part of this integration.

## Runtime path

```text
run_pipeline.bat / automation.cli
  -> config.yaml default: cricket-gameplay-stadium-v1
  -> orchestrator: rights/range/crop/artwork/font/runtime preflight
  -> discovery / complete-thought selection / word-alignment gates (unchanged)
  -> export_all: explicit profile + manifest forwarded per clip
  -> render_gameplay_clip: re-read library, deterministic EA07 preference
  -> prepare_clip_captions: stadium-reference-v1, actual word-boundary fit
  -> build_stadium_command: approved artwork + equal panels + fill crop + legal strip
  -> full encode/decode/timing/loudness/peak QC + gameplay/edit/caption/speech evidence
  -> SEO / SEO recovery: append immutable gameplay attribution
  -> cover + hash-bound release review (including gameplay, artwork, font and crop)
  -> existing scheduler/uploader, only after release gates clear and upload requested
```

Shared implementation is `automation/gameplay_export.py`; the stadium-specific
builder is `automation/stadium_export.py`. The old portrait and gameplay profiles
remain available explicitly and have their existing regression tests.

## Fixed integration gaps

1. Private-only artwork/captions/geometry are now production-owned assets/modules.
2. The current config/CLI/export/review paths all recognize the stadium profile.
3. Profile overrides are forwarded even when selecting portrait against a gameplay
   default; portrait filters honor per-analysis composition, not just global config.
4. Source-specific gameplay crops are schema/probe validated, not guessed by title.
5. EA07 preference is restricted to approved ranges long enough for the full Short.
6. All SEO seams, including failed-generation retry, retain the complete source credit.
7. Review binds design/font/crop dependencies and cannot lose gameplay checks merely
   because the edit sidecar is missing. Existing final-layout packages remain private.
8. Post-render enhancement cannot silently regrade brand/captions or stale QC hashes.
9. The isolated dry run now supplies clearly synthetic local gameplay/permission
   fixtures and verifies stadium/crop/credit evidence. Fake media durations are
   per-command, fixing a concurrency race exposed by parallel gameplay QC.
10. Legacy tests explicitly choose their intended profile. The production-code
    learning-boundary test prunes dependencies/generated media instead of reading
    an entire `.venv`, which had caused a 120-second test timeout.

## Validation evidence

Initial promotion validation (2026-11-08): **1,370 passed, 4 expected failures** in
85.14s. Required local dry run: **PASS**, 3 stadium exports, 0 failures, 0 uploads.
Scheduled-upload/sync dry run: **PASS**, 3 stadium exports and 3 **fake** uploaded
IDs, 0 failures. All dry-run network/credentials/media are isolated stubs; no real
YouTube submission occurred. `git diff --check` passes. The production PNG is
confirmed not ignored by Git. Both original FINAL.mp4 hashes are unchanged.

- Full repository suite: `temp/stadium-integration-full-tests.log`.
- Required standalone, stubbed-API dry run: `temp/stadium-integration-dry-run.log`.
- Fake scheduled-upload/sync route: `temp/stadium-integration-scheduled-dry-run.log`.
  It uses only isolated synthetic credentials/media and fake API responses.
- Real two-moment production render test in `tests/test_stadium_pipeline.py`:
  actual FFmpeg, browser captions, crop pixels, audio frequency, full ending/QC,
  evidence persistence and artwork-tampering release rejection. Only recordings
  and permissions in this test are synthetic; it grants no real rights.
- Real-source production-builder proof frames:
  `temp/stadium-integration-proof/production-layout-comparison.jpg`.
  Actual production caption/artwork/legal-strip composition inspected visually.
  This is a bounded one-frame-per-clip check, not continuous playback or a cleared
  production batch. It does not convert private recordings into approved assets.
- Live local CLI negative check: `temp/stadium-production-preflight.log`.
  Expected exit 1: missing cleared library. Zero discovery/ASR/exports/uploads.

Runtime/browser validation uses the actual local Windows environment. No universal
zero-bug guarantee or listening/phone/rights approval is inferred from tests.

## Readiness fixes following the audit

- **Resume safety:** before enhancement, the orchestrator now checks each actual
  clip's gameplay/edit sidecars. A portrait override cannot enable grading or face
  mapping of a stadium export. Missing, invalid or unknown-profile edit evidence
  holds enhancement instead of guessing about legacy media. Real portrait exports
  remain supported even when the current global default is stadium. The early
  rejection returns before enhancement, Drive sync or uploads can run.
- **Reproducible caption setup:** `requirements.txt` declares Playwright. On a fresh
  Windows environment, run:

  ```powershell
  .venv\Scripts\python.exe -m pip install -r requirements.txt
  .venv\Scripts\python.exe -m playwright install chromium
  ```

  FFmpeg/ffprobe must be on PATH; licensed Arial Black remains an installed,
  hash-checked dependency, not a redistributed font. Runtime preflight now checks
  the Chromium executable, launch and a local page, closing the browser even on
  page-probe failure. Errors give the explicit installation command. Nothing is
  installed automatically during a pipeline run. Dry runs stub only the browser
  boundary, not rights/range/assets preflight.
- **Early range coverage:** normal discovery runs require at least one approved
  continuous range supporting `clip_selection.output_max_duration` (currently
  38 seconds; legacy configs fall back to `highlight.max_duration`, then 38).
  The check runs before download/transcription/discovery. It never shortens,
  loops or speeds up a clip to fit gameplay. Direct `export_all()` batches check
  each actual candidate duration before analysis/alignment, preserving sibling
  successes and the bounded already-reviewed reserve route. Each render rechecks
  its exact duration. The library CLI uses the same required-duration preflight.

Regression coverage is in `tests/test_stadium_readiness.py`; focused, full-suite
and standalone dry-run validation logs use the `temp/stadium-readiness-*` prefix.
The earlier validation above is historical. Current readiness validation:

- **1,402 passed, 4 expected failures**, 57.17s:
  `temp/stadium-readiness-full-tests.log`.
- Required local dry run: **PASS**, 3 stadium exports, 0 failures, 0 uploads:
  `temp/stadium-readiness-dry-run.log`.
- Scheduled-upload/sync dry run: **PASS**, 3 stadium exports, 3 **fake** uploaded
  IDs, 0 failures: `temp/stadium-readiness-scheduled-dry-run.log`.
- Actual local runtime preflight: approved artwork/font, FFmpeg/ffprobe, Chromium
  executable, launch and local page all **PASS**. No installation was necessary
  on this machine. Browser failure/cleanup and missing dependency paths also have
  explicit regression tests.
- `git diff --check` passes. No real media approvals, uploads, commits or pushes.

These checks validate the code fixes, not permission to use the private recording
or a successful live API production batch. The cleared manifest and final source/
phone/legal-strip review below remain outstanding.

The subsequent [branch audit and conservative cleanup](BRANCH_AUDIT_20261108.md)
adds release-evidence, outgoing-credit, per-candidate analysis, timestamp/filename
and malformed-resume regressions. Its full suite passes **1,428 tests**, with the
same four existing expected failures; it does not remove the production holds.

## Required before a real production run

1. Resolve publisher/background use, recording/mod provenance, third-party assets
   and reviewed ranges. No approved live library is currently available.
2. Create `input/gameplay/library.json` only after genuine review. Follow
   `assets/gameplay/library.example.json`; it is intentionally REVIEW/false by default.
   Include a crop reviewed against that exact file/hash/dimensions.
3. Check local library and active profile:
   `.venv\Scripts\python.exe -m tools.gameplay_library input/gameplay/library.json --seconds 38`.
4. Run the normal pipeline locally first. Default CLI/batch commands now use the
   stadium profile. Existing private final copies are NOT resumable production batches.
5. Review actual source listening/caption names/numbers, complete playback and
   phone/UI readability, especially the new on-screen publisher legal strip.
6. Obtain exact-file release approval with public-facing credits/disclaimer intact;
   then use the existing scheduling workflow. None of these gates were bypassed.
