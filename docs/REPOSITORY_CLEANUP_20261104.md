# Main integration and conservative cleanup

Started 2026-11-03; completed 2026-11-04. Scope: integrate pending cricket
pipeline work into local `main`, then identify disposable files and unused
modules. **Face OS, its tests, configuration and face-related assets are excluded
from cleanup.** No upload, remote push, token refresh, learning sync or model
download is part of this task.

## Integration

- Original local `main`: `519bfde`; original recovery HEAD: `e9b5d4a`.
- Fetched `origin`; its `main` remained `d59ffc4`, an ancestor of local `main`.
- Committed the 57 already-pending paths as `5d325c0` (caption/packaging work,
  integration-safety fixes, existing test cleanup, fixtures, font/license and
  editorial documentation). Their pre-existing changes were preserved, not
  reset or overwritten.
- Switched to `main` and ran `git merge --ff-only recovery/audience-first-pilot`.
  No conflict resolution, history rewrite or force operation was necessary.
- All other local and fetched remote topic branch tips were already ancestors
  of the integrated HEAD. Existing branches and external worktrees were retained;
  they are not disposable merely because their commits have been merged.
- This is a **local** integration. Publishing to `origin/main` is separate.

## Removed: confirmed unused code

| Path | Evidence / current owner |
| --- | --- |
| `automation/seo/artifacts.py` | No importing module, caller, command entry point or documentation reference. Its two metadata helpers only referenced each other. The active structural metadata contract is `automation/artifacts.py`, imported by SEO, orchestration, covers and release review. |
| `utils/reports.py` | No import or caller of `generate_run_report`, and no CLI entry point. Its terminal/Markdown report was disconnected from the pipeline. The documented standalone report tools and canonical Analytics report remain. |

Evidence combined AST import inspection with tracked-source, launcher and
documentation searches. An absent static import alone was **not** sufficient
for deletion. These are repository-local findings, not proof that unknown
external scripts never imported an old helper. Both deletions remain recoverable
from Git history.

## Removed: disposable local cache files

- **183 files / 3,908,921 bytes (~3.73 MiB)** of ignored Python/pytest caches.
- Exact paths came from a saved inventory. Every removal required an unchanged
  SHA-256, an ignored/untracked path, a cache-directory location, no link escape
  and no `face` in its path. No broad directory deletion was used.
- One `.pytest_cache/v/cache/nodeids` file changed during baseline validation;
  it was deliberately skipped instead of deleting a changed file blindly.
- The file inventory and a recovery ZIP live outside the repository, so cleanup
  does not commit generated cache lists or binary backups.

## Kept: not junk, or insufficient evidence

| Paths | Why retained |
| --- | --- |
| `face_os/`, `face_os_config.yaml`, face tests/tools/models, `photos/`, reference images | Explicitly protected; no Face OS tests were invoked. All 106 protected tracked files matched their starting SHA-256 and had no diff against original local `main`. |
| `utils/token_refresh.py` | Called from `run_pipeline.bat:51` using `python -c`; ordinary Python AST imports miss this caller. The actual OAuth operation was not executed. |
| `automation/seo/analytics_report.py`, `tools/run_report.py` and report helpers | Documented/manual entry points; tool-local imports and script invocation are not ordinary root-package imports. |
| `automation/caption_review.py` | Documented `python -m` listening-review tool, with active regression tests. |
| `automation/packaging_design.py` | Documented/tested proof compositor; not automatically connected to production export. Review tooling is not dead solely because production does not call it. |
| `automation/colab.py`, `automation/kaggle.py`, `automation/scoring/ranker.py`, `automation/instagram/integration.py` | Compatibility/setup/public API candidates with test callers but no direct production import found. Removing supported interfaces or their tests needs a separate deprecation decision; no dead-code claim is made. |
| `automation/worker.py`, `video_analyzer.py` | Documented programmatic worker API and standalone video-analysis command, respectively. |
| `utils/face_restore.py` | No incoming static import found, but face-related code was deliberately excluded rather than deleted. |
| `temp/test_captions.ass`, `temp/whisper_rocm_out.json`, `temp/whisper_vulkan_out.json` | Caption/raw ASR evidence, not automatically disposable scratch. Together approximately 20.8 MiB. |
| `input/`, `output/`, `shorts/`, `transcripts/`, `highlights/`, logs and audit reports | Source, generated deliverables, review evidence or diagnostic history. No age-based or extension-based sweep. |
| Bundled font/license, model weights, secrets, `shorts_intelligence.db` | Runtime dependencies or persistent user state. The canonical outcome learner remains the sole persistent learning database. |

The earlier removal of `tests/test_pipeline_overhaul.py` was already pending
before this task and is documented in `tests/README.md`. This cleanup adds no
test deletion, skip, expected-failure rule or replacement test registry.

## Validation and evidence

Before integration: **1,111 passed, 4 expected failures**, no skipped tests,
plus the required stubbed pipeline dry run PASS (two fixture clips, zero uploads,
zero pipeline failures). The four expected failures are existing Instagram
hook/hashtag and NV12/frame-sampling gaps, not resolved functionality.

Post-cleanup: **1,111 passed, 4 expected failures** in 76.55 seconds; required
stubbed pipeline dry run PASS; `git diff --check` PASS. Logs:

- `%LOCALAPPDATA%/Temp/opencode/yt-clips-cleanup-pytest-20261104.log`
- `%LOCALAPPDATA%/Temp/opencode/yt-clips-cleanup-dry-run-20261104.log`

The unit command is the required suite with only pytest's cache provider disabled;
`PYTHONDONTWRITEBYTECODE=1` prevents regenerating removed bytecode. Neither setting
excludes tests. No Face OS suite was run.

External audit/backup directory:
`%LOCALAPPDATA%/Temp/opencode/yt-clips-merge-20261103/`.
It contains the original pending patch/ZIP, protected-file hashes, inventory,
import analysis, cache-removal manifest/recovery ZIP and protection verification.
Baseline command logs are adjacent, named `yt-clips-merge-*-20261103.log`.

Passing these checks establishes repository regression validation, not live-service
reliability, ASR accuracy, editorial approval or a publication-ready video.
