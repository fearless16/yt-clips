# Existing EA07 stock: actual-video pipeline

## Current delivery behavior (2026-11-09)

At the owner's request, the configured `input/gameplay/library-clean-v2.json`
now uses `usage: STOCK` and asset `status: AVAILABLE`. The stock is no longer
draft-only: normal upload/sync runs pass gameplay preflight, render seeded
continuous excerpts, and may sync the packaged batch to Drive. Only original
commentary audio is retained. No new gameplay download or fresh random selection
on retry is introduced.

This is availability, **not a fabricated rights grant**: the original source,
recording identity, CC label, evidence/media hashes, attribution, crop and
`rights_status: UNRESOLVED` remain intact. Owner release checks (including gameplay
rights/attribution and exact-file listening/visual approval) are still required
by `config.yaml` before YouTube scheduling. Stock review can now reach READY after
those checks are genuinely completed; it no longer demands relabelling the
recording as an APPROVED permission manifest. Upload revalidates enabled stock
and encoded-media hashes, even when release review is explicitly disabled.

Old `LOCAL_REVIEW_ONLY` manifests and existing private packages are not migrated
or silently published. Their hashes remain unchanged. The strict `load_library()`
permission-manifest route remains supported alongside enabled stock.

```powershell
.venv\Scripts\python.exe -m tools.gameplay_library input/gameplay/library-clean-v2.json --seconds 38
```

The sections below record earlier preparation and local-run history; their
draft-only restrictions describe that historical configuration, not enabled stock.

### Delivery validation and cleanup

- Real configured stock: 38-second preflight PASS, including provenance hashes,
  dark-transition screening, reviewed crop, artwork/font and browser launch.
  Evidence: `logs/stock-delivery-live-preflight.log`.
- Isolated scheduled-upload/sync dry run now uses enabled STOCK fixtures rather
  than APPROVED permission fixtures, with fake YouTube **and Drive** credentials.
  It exercises actual packaging/upload code: three synthetic exports, 24 fake
  Drive file uploads, three fake scheduled YouTube IDs, zero failures. No real
  service submission or production credential copy occurs.
  Evidence: `logs/stock-delivery-final-scheduled-dry-run.log`.
- The dry-run also patches the uploader's cached Google API builder, so an
  already-imported uploader cannot escape the fake API boundary in warm processes.
  A regression deliberately installs a failing live builder before the run.
- Drive media and DB-backup upload streams now close on success and failure.
  Retained request references previously left Windows files locked at sandbox
  cleanup. Real local-file regressions cover both create/update and failed requests.
- Targeted stock/Drive-resource regressions: 44 passed. Evidence:
  `logs/stock-delivery-final-focused.log`. Final required suite: **1,516 passed,
  4 existing expected failures**, 94.74 seconds. Required standalone dry run:
  **OVERALL PASS**, three synthetic exports, zero failures/uploads. Evidence:
  `logs/stock-delivery-final-full-tests.log` and `logs/stock-delivery-final-dry-run.log`.
- Deleted 301 obsolete generated caption-card PNG/HTML files and stale stadium
  test/dry-run logs (10,264,768 bytes) from explicitly scoped branch temp folders.
  Kept source/stock footage, exports, review/provenance JSON, proof frames and
  private renderer scripts. Preserved MP4 hashes were checked before/after.
  Also removed this session's failed synthetic sandbox after fixing the lock.
  Exact cleanup inventory: `reports/stock-delivery-cleanup-20261109.json`.
- No inspection, mutation or test execution under `face_os/` was part of this work.
  Live ASR/AI, real Drive delivery, real YouTube scheduling and owner review are
  not proven by these isolated technical checks.

### Publish a newly reviewed batch without rerendering

NEW RUN + upload no longer aborts on enabled stock. It still creates owner-review
drafts and holds scheduling until every required exact-file check is completed.
After genuine review and `automation.release_review approve` for each clip, resume
the **same new source-scoped batch**, not a legacy private sample:

```powershell
.venv\Scripts\python.exe -m automation.cli "https://youtu.be/wRP_KO9vP5g" --source-scoped --skip-download --skip-transcribe --skip-highlight --skip-export --export-dir "shorts/wRP_KO9vP5g/BATCH" --sync --upload --schedule --skip-instagram
```

Replace `BATCH` with the real batch directory. This command is a user-run delivery
step, not part of technical validation, and does not create approval by itself.

The channel owner's existing stock recording is reused; no replacement gameplay
or external commentary is requested. Each source moment seeds a reproducible,
pseudorandom continuous excerpt matching the Short's full duration. The accepted
stadium panels and recording-specific `[240,128,1440,810]` crop stay unchanged.
Only input 0 (the channel owner's actual-video voice) supplies audio. Stock input
1 is video-only and is never audio-mixed. Natural speed and complete-thought,
caption, encoding and delivery checks remain enabled.

`input/gameplay/library.json` now records the existing GSTG EA07 file, its actual
hash, the stored CC-attribution label/recording identity, and sample-screened
continuous ranges `[288,340]` and `[414,470]`. This local file is ignored by Git;
original footage, license metadata and prior accepted finals remain untouched.

## Separate preparation from publication

The previous preflight blocked even local preparation because it required full
publication clearance first. `load_render_library()` now accepts an explicit
`LOCAL_REVIEW_ONLY` manifest with `REVIEW_ONLY` / `UNRESOLVED` assets. It still
checks file/evidence hashes, uploader/source identity, recorded CC label, crop,
ranges, real coverage, browser, artwork and font dependencies. A withdrawn or
changed asset fails. No publisher permission or third-party clearance is invented.

`load_library()` remains strict for publication/release validation. The runner
rejects upload/Drive sync of a local-review library before paid processing; the
uploader independently rejects a REVIEW_ONLY selection even if release review is
disabled. Local review drafts may be created, with rights checks NOT CHECKED;
approval and release validation still use the strict publication loader.

Preflight the local stock:

```powershell
.venv\Scripts\python.exe -m tools.gameplay_library input/gameplay/library.json --seconds 38 --local-review
```

Production publication readiness is not asserted by that command or tests.
Source audio/caption listening, continuous playback, phone checks and exact-file
release review are separate from technical QC. Existing rights questions remain
recorded rather than silently relabelled cleared.

## Current actual-source run

- Source: `https://www.youtube.com/watch?v=wRP_KO9vP5g`.
- Video isolated at `input/wRP_KO9vP5g/video.mp4`; earlier canonical source is not replaced.
- Transcript isolated at `transcripts/wRP_KO9vP5g/video.json`: 1,050 YouTube API cues.
- Selection target: two distinct complete source moments, not two layout variants.
- Selection: `highlights/wRP_KO9vP5g/video.yaml` and source-bound editorial evidence.
- Gameplay range inspection: `reports/stock-range-review-20261108/ranges.jpg`;
  four-second samples, not a claim of continuous playback or rights clearance.
- No auto-upload, Drive sync, Instagram publication, enhancement, commit or push.

Regression coverage: `tests/test_stock_gameplay.py`, plus the existing actual
FFmpeg/browser gameplay/stadium tests. Full tests and standalone dry run are
recorded under `reports/stock-pipeline-*`.

Validation after the final code/test changes:

- Full required suite: **1,446 passed, 4 existing expected failures**, 76.82s;
  `reports/stock-pipeline-validated-tests.log`.
- Required standalone stubbed-API dry run: **OVERALL PASS**, three stadium exports,
  zero failures/uploads; `reports/stock-pipeline-validated-dry-run.log`.
- New stock regressions: 18 passed, including a real two-moment Chromium/FFmpeg
  stadium encode from synthetic video-only stock and distinct source ranges.
  Encoded audio retains the synthetic source's 440 Hz signal at natural speed.
- Actual existing EA07 manifest: local 38-second preflight PASS, including media/
  evidence hashes, crop, continuous coverage, artwork, font and browser runtime.
- Both previously accepted FINAL.mp4 hashes remain unchanged. No commit/push.

These checks do not claim that the new actual-source batch is finished or that
human listening, phone playback, rights review or publication has been completed.

### Actual-run boundary recovery

The first selection attempt returned zero drafts / REVIEW_REQUIRED. Stored
evidence is `reports/stock-pair-wRP_KO9vP5g-selection.json` and
`highlights/wRP_KO9vP5g/video.curation.json`. Seven proposals failed exact ASR
anchors or tentative cue-edge timing checks; four other proposals had missing
verbatim source anchors. No clips were exported/uploaded by that attempt.

Inspection found that fresh fallback ASR could be rejected before the existing
word-ID boundary editor saw it: an invalid word near a tentative edge, an
overlapping caption edge, or oversized tentative duration prevented a narrower
valid span from being proposed. Fresh fallback ASR can now be returned ONLY as
`boundary_review_required` / UNVERIFIED evidence. All raw intervals remain;
supplied malformed/reviewed evidence is still rejected without retranscription.
The editor must select one bounded contiguous word-ID span; actual selected
intervals, confidence, gaps and duration are revalidated, then a separate final
completeness review remains mandatory. No nearest-time guessing, dropping
internal words, timestamp fabrication or automatic PASS was introduced.

Focused recovery/curation/stock tests: 60 passed. The full actual-source runner
was restarted with cached discovery/raw ASR, isolated source paths, target two,
and no upload/sync/enhancement. Runtime result, when complete, is written to
`reports/stock-pair-wRP_KO9vP5g-run.json`.

That retry (`cebb8534`) also held with zero exports after final review rejected
garbled raw Hindi ASR; it did not prove a working actual-source batch. Its complete
curation record is retained in
`reports/stock-pair-wRP_KO9vP5g-cebb8534-curation.json`.

A bounded diagnostic then re-decoded only the already discovered Gill-reaction
and batting-approach moments using eight-second independent Whisper windows.
Both raw results had zero zero-length word intervals, compared with degenerate
intervals in the longer-window attempts. Wording is still ASR, not listening-
verified transcription. Raw diagnostics are under `reports/stock-short-chunk-probe/`.

Production `transcribe_range()` now uses `chunk_length=8` and
`condition_on_previous_text=False`. Speech-cache identity version 2 records those
options, preventing reuse of a stale long-window failure. It does not change
probabilities, invent intervals, drop words, or remove validation. A bounded
two-moment recovery uses the existing word-ID editor and separate final reviewer
before the normal export/SEO/thumbnail/QC runner, not another whole-video scan.

The scoped recovery approved only the distinct batting-approach moment at
1564.26–1586.80s (22.54s), and rejected the Gill-jinx candidate. One candidate is
not relabelled as a two-clip success. Its first actual encode is preserved at
`shorts/wRP_KO9vP5g-stock/2026-11-08_184535_24418546/clip1.mp4` with FAIL QC:
full decode/A–V/duration passed, but measured loudness was -17.04 LUFS, outside
the unchanged -16 ±1 target. No export success or release is inferred for it.

Both gameplay builders now measure the exact retained input-0 source voice and
use two-pass loudnorm with measured I/TP/LRA/threshold/offset. Input-1 gameplay is
still not audio-mapped; no speed/fade is added. The actual source-range AAC probe
then measured -16.35 LUFS / -1.44 dBTP, inside the existing targets; evidence is
`reports/stock-short-chunk-probe/two-pass-voice-measured.json`. A fresh normal
encode/SEO/thumbnail/QC attempt follows from the already reviewed source range.
No QC tolerance was widened. Dry-run synthetic audio measurement fixtures now
supply the five real two-pass fields rather than only I/TP.

## Completed real two-Short package

Final batch:
`shorts/wRP_KO9vP5g-final-stock-pair/2026-11-08_190100_fcc8d357/`.

| File | Source range | Duration | Seeded muted EA07 range | Encoded LUFS / dBTP |
| --- | --- | --- | --- | --- |
| `01-batting-approach.mp4` | 1564.26–1586.80 | 22.54s | 418.063–440.603 | -16.32 / -1.47 |
| `02-kohli-drive.mp4` | 664.76–675.96 | 11.20s | 455.985–467.185 | -16.01 / -1.68 |

- Two distinct nonoverlapping commentary moments; distinct, nonoverlapping seeded
  stock excerpts, with the accepted stadium composition and recorded crop.
- Both final files freshly pass full decode, A/V stream timing, duration,
  loudness and true-peak QC. Actual timeline speed is 1.0, not the detector's
  advisory 1.15x log. Gameplay contributes no audio.
- Source-voice comparison at 1x: median one-second waveform correlations are
  0.999955 and 0.999979. This is signal evidence, not listening approval.
- A 250ms internal silent flush guard before loudnorm, trimmed away AFTER the
  filter, avoids losing a non-round final voice block. It does not append silence
  to the delivered ending. Output audio is explicitly resampled to 48kHz.
  A real dynamic-audio regression checks non-round duration and the final voice
  block; MP4/AAC endpoint rounding is allowed only within 1ms in that test.
- Whisper timestamps/probabilities are converted to native Python floats without
  changing values, fixing the observed NumPy-scalar YAML serialization failure.
- Full automatic SEO failed format checks and exceeded its 300-second export
  deadline. The completed pair uses manually authored, source-grounded
  Roman-Hinglish titles/cover copy and English descriptions, validated with the
  existing SEO quality gate. No provider output was relabelled successful.
- Both have source-derived covers, full gameplay credit/EA disclaimer, and
  hash-bound NOT CHECKED owner-review drafts. No human approvals were fabricated.
- Opening/middle/end frames and both covers inspected in
  `reports/stock-pair-final-review/pair-and-covers.jpg`. The first clip's opening
  like-alert is baked into the original source; it is not a generated overlay.
  Source-score/chat panels and the production legal footer remain visible.

Final media SHA-256:

```text
8533d83cd1022f7b8c7a0f4fa1828a6faf0f6003f69fc8e771fd6de2d22096d6  01-batting-approach.mp4
f2b45e9137d9f89858fa9b47262fd708cd52cdf7bdd8d0b7f0544d3a4824de21  02-kohli-drive.mp4
```

Evidence:

- `reports/stock-pair-final-render.json`: both normal exports, zero failures.
- `reports/stock-pair-final-run.json`: both metadata/covers, zero failures,
  two verified distinct source moments; **zero uploads**.
- `reports/stock-pair-final-review/verification.json`: fresh hashes, per-file
  actual-source/audio checks, exact gameplay ranges and remaining release holds.
- `reports/stock-pair-final-tests.log`: **1,453 passed, 4 existing expected
  failures**, 60.61s. No acceptance tests/expected-failure rules removed.
- `reports/stock-pair-final-dry-run.log`: **OVERALL PASS**, three synthetic stadium
  exports, zero failures/uploads. `git diff --check` passes.
- Both prior canonical FINAL.mp4 hashes rechecked unchanged. No commit/push.

This is a completed **local two-Short package after bounded recovery**, not proof
of a fully unattended new-source run. Automatic discovery/ASR and automatic SEO
had real holds and needed the documented recoveries. Full playback/listening,
caption/name/number fidelity, phone/UI review, publisher/third-party rights and
exact-file release approval remain open. Neither Short has been uploaded or
scheduled; the existing schedule has not been changed.

## Production update after appearance acceptance

The owner accepted the appearance of the finished pair. That feedback is not
converted into rights clearance, listening attestations or publication approval.
The exact completed files, covers and prior evidence are not changed by this
code update.

- `config.yaml` defaults to **two** distinct source moments. Stadium geometry,
  branding, continuous seeded stock selection, muted gameplay, original-source
  voice, 1x playback, two-pass audio and caption/release gates remain unchanged.
- Fresh ASR fallback now keeps the same bounded context (3 seconds with current
  settings), rather than expanding to 6 seconds. Eight-second independent
  decoding remains source/config keyed. Primary speech rejected as garbled by
  the final text reviewer gets at most two scoped fallback re-alignments before
  another whole-transcript discovery. Raw attempts are retained. Supplied word
  evidence, already-tried fallback ASR, weak/incomplete content and prior export
  overlaps cannot be silently rescued. New speech requires fresh boundary and
  final text review; provider-call limits remain enforced.
- A Devanagari title or invalid cover length gets one title/cover-only strict
  JSON correction (60s, low reasoning, 4,096 tokens) before full escalation.
  Descriptions/tags/research are preserved. Grounding, promise-alignment and
  quality checks still decide final acceptance; no generic metadata fallback.
- SEO workers generate into private temporary staging. A lock-protected deadline
  flag prevents late results from overwriting batch metadata, retry markers or
  later owner recovery. Timeout immediately persists per-clip retry evidence.
  A fast provider rejection is now a partial batch failure even if all videos
  encoded. Successful media is preserved. This bounds artifact publication, not
  force-killing an in-flight provider SDK/network request.
- NEW RUN in `run_pipeline.bat` now passes `--source-scoped`: all stages share
  `input/<source-id>`, `transcripts/<source-id>`, `highlights/<source-id>` and
  `shorts/<source-id>/<batch>`. Existing canonical source media is not replaced.
  Stock remains shared at its configured manifest path. Config scope is serial,
  restored on exit/failure, and is also available directly from the CLI.

```powershell
# New local batch; no upload, sync or Instagram publication
.venv\Scripts\python.exe -m automation.cli "https://www.youtube.com/watch?v=VIDEO_ID_HERE" --source-scoped --skip-instagram

# Resume packaging only in an explicit source-scoped batch; no rerender
.venv\Scripts\python.exe -m automation.cli "https://www.youtube.com/watch?v=VIDEO_ID_HERE" --source-scoped --skip-download --skip-transcribe --skip-highlight --skip-export --export-dir "shorts/VIDEO_ID_HERE/BATCH" --skip-instagram
```

Replace the example ID with the video's actual 11-character ID. The special
previously completed pair remains at its recorded final-stock-pair path; it is
not moved into the new default namespace. These tests exercise the automatic
recovery and concurrency boundaries with substituted provider/ASR responses;
they are not proof of a successful live-provider run on every future source.

Validation of this update:

- Required full suite: **1,479 passed, 4 existing expected failures**, 56.80s;
  `reports/stock-runtime-update-tests.log`. New regressions cover cached ASR
  fallback + mandatory fresh review, immutable supplied evidence, bounded failed
  recoveries, title/cover-only repair, late-worker races, retry-marker disk
  failures and serial source-config restoration. Existing launcher/default
  assertions were updated to the source-scoped two-Short contract, not removed.
- Required standalone dry run: **OVERALL PASS**, three synthetic acceptance
  exports/covers/metadata, zero failures/uploads;
  `reports/stock-runtime-update-dry-run.log`. Its explicit three-candidate fixture
  continues testing reserve handling independently of the production pair.
- Existing actual EA07 stock: 38-second local preflight **PASS**. CLI help exposes
  `--source-scoped`; `git diff --check` passes.
- All four current/previous final video hashes are unchanged:
  `reports/stock-runtime-update-final-hashes.json`.
- No new live-source/provider batch, upload, schedule mutation, commit or push
  was performed for this code update. Rights/listening/release holds are unchanged.

## Live-run defects corrected (2026-11-08)

The subsequent live run `9c12c0fc` automatically prepared two new moments, but
frame review found a dark stock transition and internal metadata contained
Suryakumar/Arshdeep search terms unrelated to those moments. That run remains
preserved at `shorts/wRP_KO9vP5g/2026-11-08_203255_d1f8f8e3/` for diagnosis.

### Production fixes

- `automation/gameplay_visual_qc.py` decodes every frame in each declared stock
  crop using FFmpeg blackdetect (160x90 spatial downscale, no temporal sampling;
  dark-pixel threshold 0.08 and dark-picture fraction 0.90). Even 1-2-frame
  transitions are excluded with 0.3-second guards. The selector chooses a single
  full-duration clean interval inside the visual review ranges or fails closed.
  This is a darkness detector, not semantic/branding/rights certification.
- The old ranges contained several 33-100ms transitions missed by four-second
  samples. New default `input/gameplay/library-clean-v2.json` additionally declares
  the visually sampled, dark-screened 356-399s section of the SAME stock file.
  Old `library.json` and the stock recording are unchanged, so prior provenance
  is preserved. The new manifest still says LOCAL_REVIEW_ONLY / UNRESOLVED.
- `automation/delivery_qc.py` rechecks the actual delivered gameplay panel on
  every QC pass. It validates panel geometry and media hash, persists
  `gameplay_visual`, and rejects darkness or unavailable decoding. A later
  orchestrator QC pass cannot erase this check.
- Query generation uses exact-clip entities, including the existing LLM's
  bounded Hindi-name/topic grounding, never all players in the stream description.
  Broad team research cannot crowd out a named clip subject; six/wicket research
  must agree with the clip-grounded topic. Audience aliases retain their wording
  only after canonical grounding. Model ordering is retained within approved
  phrases. BOTH final search fields are entity-checked after all assignments,
  with `search_grounding` retained for audit.

### Corrected batch and verification

- Corrected render run `2b554acc`: both original source moments reused, fresh
  clean stock excerpts, live-provider metadata/covers, zero reported failures.
  Final packaging/QC resume `296b123d` used the stricter final query policy,
  recovered both explicit repair markers, regenerated matching covers, and
  recreated unapproved exact-file review drafts. Its `selection_status: NOT_RUN`
  deliberately reflects reusing the prior reviewed selection, not new discovery.
- Output: `shorts/wRP_KO9vP5g/2026-11-08_205639_27417d20/`.
- `clip1.mp4`: 27.44s, original source 1213.32-1240.76s, muted stock
  369.678-397.118s; SHA-256
  `38174a8f1f131fa7d901b90c0539ce8543439bd602da2be282d8a82bd7b80f91`.
- `clip2.mp4`: 30.28s, original source 2005.30-2035.58s, muted stock
  364.096-394.376s; SHA-256
  `9bfbe6cddf636c2bd1af107d91b3c3783955aaa6ea4d4e66e63eb5436792e8e9`.
- Both encoded QC reports PASS, including complete gameplay-panel darkness
  scans. Fresh scans of the exact selected stock excerpts detect no dark frames.
  Decoded PCM audio hashes are IDENTICAL to the original new batch: voice and
  natural speed did not change during the gameplay repair.
- Final primary terms are `Virat Kohli wicket New Zealand` and `Rohit Sharma six`.
  Final internal fields contain no Suryakumar/Arshdeep contamination; the Rohit
  six is no longer packaged as a wicket. Covers match the new cover_text values.
- Inspected first/8%/middle/92%/near-last frames and both covers. Every hash-bound
  review artifact was rehashed; all four previously accepted finals and the old
  manifest remain unchanged. This is not full listening/phone playback approval.
- Archived replaced unapproved packaging:
  `reports/stock-defect-fix-packaging-before-final/`.
- Evidence: `reports/stock-defect-fix-final-review/verification.json` and
  `frames.jpg`; live logs `reports/stock-defect-fix-live-run.log` and
  `reports/stock-defect-fix-packaging-resume.log`.
- Required tests: **1,494 passed, 4 existing expected failures**, 57.25s;
  `reports/stock-defect-fix-tests.log`. Required standalone dry run: **PASS**;
  `reports/stock-defect-fix-dry-run.log`. Regressions cover brief black frames
  despite HUD, all-black endings, guarded selection, repeated delivery QC,
  source-roster pollution, generic-query crowding and late internal-field writes.

Nothing uploaded, scheduled, committed or pushed. Rights and exact-file owner
listening/caption/phone checks remain open; the corrected files are local drafts.
