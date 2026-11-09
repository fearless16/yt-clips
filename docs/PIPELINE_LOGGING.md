# Pipeline console and logs

Run normally with `run_pipeline.bat` or `.venv\Scripts\python.exe -m automation.cli URL`.
The terminal uses a cyan/magenta header, compact stage events, and one final
summary with counts, stage timings, output directories, and the log path.

- **Green ✓**: that stage, or the requested job, completed successfully.
- **Amber !**: partial completion or editorial review required.
- **Red ✗**: failed work.
- **Grey –**: skipped, not completed. An empty result has no success tick.

Local draft completion is **not release approval**. Owner-review and publication
gates remain unchanged. Requested but unconfirmed YouTube uploads cannot receive
an overall success tick. Failures are listed in the final summary, including on
early exits. The Windows launcher closes with a neutral message instead of a
second unconditional success banner.

## Diagnostics

- `--verbose`: show DEBUG detail in the console.
- `--json-logs`: structured JSON console events instead of the dashboard.
- `logging.level` and `logging.console_format` in `config.yaml` configure defaults.
- `logs/pipeline.log` retains JSON-lines diagnostics, including DEBUG records
  and exception tracebacks; quieter console output does not delete diagnostics.
- `run_id` correlates orchestration events; `stage`, `status`, `duration_ms`, and
  `metadata` describe stage outcomes. `PipelineResult.stage_results` retains the
  current run's completed/skipped stage events, not audience outcomes.

The display uses static events, not a refreshing animation or guessed percentage.
Terminal color can be disabled with `NO_COLOR=1`. JSON output has no ANSI styling.
Concurrent export and SEO timings may overlap; they are not added to estimate
total runtime. Total elapsed time includes final telemetry.
