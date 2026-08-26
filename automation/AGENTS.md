# Automation Module

`automation/orchestrator.py` runs ingest, transcript, cricket-only selection,
export, SEO, upload, telemetry, and canonical DB persistence.

Learning ownership belongs exclusively to `shorts_intelligence/`:

- exact channel Shorts shelf is the catalog boundary;
- real YouTube Analytics snapshots are outcomes;
- export metadata is pre-publication evidence, never success;
- policy changes require calibrated positive evidence;
- complete-thought and clip boundaries are hard gates.

The in-memory `DecisionStore` is runtime telemetry only. Do not add another
persistent learner or analytics database.

```bash
.venv\Scripts\python.exe -m pytest tests/test_automation.py -v
.venv\Scripts\python.exe -m pytest tests/shorts_intelligence/ -v
```
