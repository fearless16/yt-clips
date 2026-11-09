# Branch audit and conservative cleanup — 2026-11-08

Branch: `plan/cricket-stock-split-variant`. Pending work was preserved; nothing
was committed, pushed or uploaded. This is regression evidence, not a zero-bug
guarantee or permission to publish.

## Reproduced and fixed

- **Release-check downgrade:** an empty/legacy gameplay sidecar or nested gameplay
  evidence could lose the additional rights/framing checks. Presence now keeps
  those obligations, and incomplete selection evidence holds review explicitly.
- **Attribution loss:** the credit seam checked characters rather than UTF-8 bytes
  and skipped the limit on its idempotent path. It now checks the complete text
  every time. Upload also checks the exact outgoing description after Shorts-marker,
  markdown and truncation processing, before authentication or schedule reservation.
  It holds rather than silently removing/rewriting reviewed credit.
- **Batch abort:** a candidate's analysis exception aborted all sibling exports
  before the reserve queue. Candidate-level failures now preserve successful
  siblings and bounded, already-reviewed reserve recovery, with failure evidence.
- **Invalid edit ranges:** negative/nonfinite timestamps, malformed strings and
  booleans could reach analysis or even rendering. They now fail before analysis;
  an invalid timestamp is never silently converted to zero.
- **Unsafe filenames:** highlight IDs could escape their batch or collide with
  Windows filenames. Unsafe/reserved/case-colliding IDs are rejected before writes.
- **Resume crash:** non-object finalization JSON could raise an uncaught attribute
  error. Malformed/unreadable manifests now give a controlled preservation/review hold.

Regression cases: `tests/test_branch_audit.py`. Initial failure-first run reproduced
13 failures; the final tests also cover plain-credit delivery and malformed speech.

## Validation

- Full required suite: **1,428 passed, 4 existing expected failures**.
- Required local pipeline dry run: **PASS**, 3 stadium exports, 0 failures/uploads.
- Scheduled-upload/sync dry run: **PASS**, 3 stadium exports, 3 **fake** IDs,
  0 failures. No real network publication is proven by this check.
- Compiled all **70** pending/untracked Python source and test files: no syntax errors.
- Actual local artwork/font, FFmpeg/ffprobe and Chromium-page preflight: **PASS**.
- `git diff --check`: **PASS**. Both finalized MP4 hashes are unchanged.
- Post-cleanup full rerun: **1,428 passed, 4 expected failures** in 56.22s;
  required dry run **PASS**. Removed caches stayed absent. The real CLI still
  exits with the expected missing-cleared-manifest hold before production work.

Logs and removal inventory: `reports/branch-audit-*`. Tests after cleanup disable
pytest's cache provider and bytecode writing, not tests, to avoid recreating junk.

## Cleanup boundaries

Removed **229 ignored cache files, 5,262,484 bytes (5.02 MiB)** and empty scratch/cache
directories. Each file was untracked, ignored, in a known Python/pytest cache, not
linked outside the workspace, backed up and hash-checked immediately before removal.
Recovery ZIP: `%LOCALAPPDATA%/Temp/opencode/yt-clips-branch-audit-20261108T120250468572Z/cache-recovery.zip`.

Kept all pending source changes, source media, finalized/private renders, raw ASR,
caption/provenance evidence, historical logs, dependencies, secrets, models, learner
database and Face OS/face-related files. In particular, `temp/gameplay-review-20261107/`
is referenced by the finalized package; it is not an indiscriminate cleanup target.

## Remaining limits

The cleared `input/gameplay/library.json` is still absent. Real production must
remain held pending genuine rights/range review and exact-file listening, caption,
phone and release approval. Four pre-existing expected failures remain; passing
tests does not certify those unresolved paths or live-service reliability.
They cover Instagram's overlong-hook repair and invented-hashtag enforcement,
plus the software NV12 pipe and frame-sampling integration cases. No expected
failure was removed or added to make this audit pass.
