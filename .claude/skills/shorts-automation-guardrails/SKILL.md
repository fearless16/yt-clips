---
name: shorts-automation-guardrails
description: Mandatory operating rules and anti-pattern defence for ANY agent reading, reviewing, or changing the yt-clips Shorts automation (automation/, export.py, upload.py, scheduler.py, thumbnail.py, sync.py, config.yaml clip/upload sections). Load before touching clip selection, cutting, gates, SEO, release review, scheduling or upload. Not for face_os/.
---

# Shorts automation guardrails

## 0. Scope fence
- **Never edit `face_os/` or its tests.** It is a separate system. If a change seems to need it, stop and ask.
- This skill covers ingest → selection → cut → render → SEO → review → schedule → upload.
- Companion skills: `shorts-cut-quality` (how to cut), `shorts-publish-path` (how to ship).
- Known defects with file:line evidence: `references/cancer-register.md`. Read it before you propose work, and update it when you fix or find something.

## 1. North star (what "working" means)
The product is a **published Short that people watch**. No other output counts.
In order:
1. `published_per_week`: clips with a `*_delivery.json` receipt in status `COMPLETE`.
2. Engaged views, "shown in feed" and average % viewed from `shorts_intelligence` (real YouTube Analytics).
3. Everything else is a proxy: scores, gate PASS counts, test counts, "status=ok".

Before any change, write down which of 1–2 it moves and how you will measure it. "Safer", "more rigorous" or "more evidence" does not answer that.

As of 2026-11-27 the baseline is **0 receipts and 0 uploads from real-source runs**. Every review, QC and gate change must be judged against that.

## 2. Anti-patterns and their defences

| # | Anti-pattern | How to detect it | Required defence |
|---|---|---|---|
| A1 | **Gate stacking.** You add another hold or `raise ValueError` "for safety". | `grep -c "raise ValueError" <file>` goes up. The same predicate is checked at more than one stage. | Gate budget: a PR may add a gate only if it removes or merges one, or ships with measured data showing the false-hold rate is under 10% on real sources. Re-checking the same predicate downstream is forbidden. Pass the verdict along instead. |
| A2 | **Batch poisoning.** One clip's failure aborts or fails the whole batch. | `_abort_required_stage` or a `raise` inside a per-clip loop. A batch-level deadline. | Per-clip isolation. A bad clip is quarantined with its reason, and the other clips continue. |
| A3 | **Human gate with no throughput.** An approval needs many hand-written fields, so nobody completes it. | Count `*_review.json` with `approval: null` against approved ones. | A human gate needs one command and under 60 s per clip. It shows evidence and takes yes or no. Machine-verifiable checks must auto-fill. |
| A4 | **Hash-bound deadlock.** An artifact changes and the review or receipt goes stale with no automatic way back. | Messages like "explicit reconciliation required" or "preserve and review", where the only exit is manual file surgery. | Every HOLD state needs a documented, scripted exit, either regenerate or archive. If you add a state you also add its exit. |
| A5 | **Theatre telemetry.** Scores and audits are computed, logged, and never used. | The value is written to an event or log only. Nothing branches on it. | Make it decide something or delete it. Do not add new "advisory" scorers. |
| A6 | **Single-LLM chokepoint.** One provider call is the only path to any output, and its failures are opaque. | Error text is empty (`"}`) or generic. No deterministic fallback. | Log `type(exc).__name__` and `repr(exc)`. Keep a deterministic fallback that sends candidates to the review queue, never to zero. |
| A7 | **Threshold by intuition.** Magic numbers (0.75 confidence, 2 s gap, 300 s deadline) with no measured distribution. | Look for a constant with no benchmark reference next to it. | Measure the distribution on 3 or more real sources first, and record p10/p50/p90 in the register. Then set the threshold. |
| A8 | **Plan and doc inflation.** You write a new strategy or review markdown instead of fixing or measuring. | Root `*.md` count grows. Plans contradict each other. | Update `references/cancer-register.md`. A new top-level plan doc needs the user's approval. |
| A9 | **Overfitting tiny data.** Day-of-week or duration rules come from about 200 Shorts with a median of about 325 views. | Rules like `avoid_days`, or "Thursday is 2.7x worse". | Treat it as a hypothesis. It needs a confidence interval from `shorts_intelligence`, or it stays a tie-breaker only. |
| A10 | **Test-polluted evidence.** Tests write to `logs/pipeline.log`. | Fixture URLs `AbC123xyz98` and `youtu.be/test` dominate the log. | Filter fixture runs out before any log measurement (see §4). Never quote raw log counts. |
| A11 | **Dead config.** Config keys the code never reads. | `grep -rn <key> --include=*.py` matches only dry_run or tests. | Delete the key or wire it up. Never cite config as behaviour without a grep. |
| A12 | **Green tests, red product.** "441 passed" is reported as success. | — | Also report the north-star numbers from §4 in every summary. |
| A13 | **Silent swallow.** `except Exception: return None`, after which a caller invents a generic error. | `upload returned no video ID` hides the real API error. | Carry the original reason all the way to `result.failures`. |

## 3. Change protocol
1. Read `references/cancer-register.md`. Find or add the item you are addressing.
2. Measure before: run the §4 commands and record the numbers in the register.
3. Make the smallest change that moves the north star. No drive-by refactors.
4. Validate per root `AGENTS.md`: unit tests excluding face_detect, then `dry_run.py`. Do not commit if either fails.
5. Measure after, on a real source if one is available. Update the register with before and after.
6. Report the north-star numbers, what changed, and what is still blocked.

## 4. Measurement commands (Git Bash, from repo root)
```bash
# Pipeline funnel on disk
echo "mp4=$(find shorts -name '*.mp4'|wc -l) meta=$(find shorts -name '*_metadata.json'|wc -l) \
review=$(find shorts -name '*_review.json'|wc -l) receipts=$(find shorts -name '*_delivery.json'|wc -l) \
seo_failed=$(find shorts -name '*_seo_failed.json'|wc -l)"

# Hold reasons, REAL sources only (excludes test fixtures)
awk '/\[START\] pipeline/{fake=($0~/youtu.be\/test|AbC123xyz98|example/)}
 !fake && match($0,/(Speech alignment held: [A-Za-z ]{5,45}|Editorial curation held|Release held \([A-Z]+\)|NO_PUBLISHABLE_CANDIDATES|Gameplay coverage held|SEO generation exceeded|delivery_qc [^:]+|editorial_target)/){c[substr($0,RSTART,RLENGTH)]++}
 END{for(k in c)print c[k],k}' logs/pipeline.log | sort -rn

# Real outcomes
.venv/Scripts/python.exe -m shorts_intelligence status
```

## 5. Traps (write them down, do not chase them)
If something looks like a rabbit hole (log archaeology, a 2000-line file read end to end, re-researching settled policy, a fix that needs face_os), note it in `references/cancer-register.md` under "Open questions / traps" and move on. Ask the user at the end of the session.

## 6. Token discipline
- Map before you read: `grep -n "^def \|raise "` first, then read only the function involved.
- Do not read tests unless a test is the thing being changed.
- Send web research to a cheap subagent with a strict budget and evidence labels (OFFICIAL / SECONDARY / ANECDOTAL / UNVERIFIED).
