# Automation Module

> **MANDATORY for every agent before touching automation, export, upload,
> scheduler or their config:** read and follow
> `.claude/skills/shorts-automation-guardrails/SKILL.md` (anti-pattern defence,
> north-star metric, change protocol) and its
> `references/cancer-register.md` (known blockers with file:line evidence).
> Cutting work → `.claude/skills/shorts-cut-quality/SKILL.md`.
> Publishing work → `.claude/skills/shorts-publish-path/SKILL.md`.
> Never edit `face_os/` from automation work.

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
