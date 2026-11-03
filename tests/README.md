# Cricket pipeline tests

Run the current suite from the repository root:

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v --ignore=tests/test_face_detect.py
.venv\Scripts\python.exe dry_run.py https://youtu.be/test --skip-download --skip-transcribe
```

These commands do not run the separate `face_os/` suite. Face OS was excluded
from this cleanup; its files and configuration were not changed.

## Cleanup (completed 2026-11-03)

- Removed the permanently skipped `test_pipeline_overhaul.py`. Its duplicate
  provider, SEO, transcription, upload and legacy face-detection checks belong
  to the dedicated active suites. The still-useful structured phase-logging and
  config-validation checks now run in `test_logger.py` and
  `test_dry_run_fixtures.py`.
- Removed obsolete assertions forbidding the supported native DeepSeek provider,
  fixed provider/model totals, the duplicate Mimo catalog check, and duplicate
  circuit-breaker/token-bucket checks. Current credential-aware routing,
  per-model token buckets, cooldowns, auth/quota errors and restricted SEO
  fallback remain covered.
- Removed the hardware-specific “software must beat DXVA2” timing assertion.
  It ignored command exit status and cannot establish correctness across PCs.
  Functional NV12 conversion/frame-sampling tests remain.
- Removed the deleted tests' entries from the shared expected-failure registry;
  no collection exclusions or replacement skip lists were added.
- Reactivated the Instagram caption/SEO suite. Removed its module-wide skip,
  fake `sys.modules` evidence module and global minimal-config replacement.
  Tests now stub the real call boundaries without affecting other suites.
- Removed the obsolete “lazy AI client not implemented yet” skip branch.

## Known gaps are not obsolete tests

Two existing NV12/frame-sampling checks remain in the shared baseline
expected-failure registry. Their functionality is still relevant; cleanup
does not repair or certify that behavior.

Two previously blanket-skipped Instagram checks now execute as **strict**
expected failures with concrete reasons:

- Hook-length enforcement is bypassed despite the prompt's 55-character rule.
- Supplied hashtags bypass evidence allowlisting in `_validate_and_scrub`.

These are existing implementation gaps, not permission to weaken those safety
requirements. A future fix must make the checks pass and remove the markers;
an unexpected pass already fails the suite. No production policy was changed
to make this cleanup pass.

Keep current release-review, delivery-QC, caption-fidelity, source-boundary,
owned-channel learning, ingestion/resource-safety and remote-side-effect tests.
A test is not disposable just because it fails or requires an optional tool.

Cleanup inventory and validation logs:
`reports/test-suite-cleanup-20261102/` (work started before midnight).

Validation: **1,111 passed, 0 skipped, 4 expected failures**; required pipeline
dry run PASS; `git diff --check` PASS. The previous suite reported 1,083 passed,
47 skipped and 18 expected failures. That comparison reflects deletion,
consolidation and restored execution, not new production functionality.
Production file identities and all 459 preserved media/evidence files are unchanged.
