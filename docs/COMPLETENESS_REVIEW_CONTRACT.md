# Shorts completeness review contract

Discovery is not a semantic review. Punctuation, ASR word timestamps, loudness,
and successful rendering cannot establish that a thought is complete.

The default transcript-first workflow now performs word preflight **before** its
final text review; see [TRANSCRIPT_FIRST_CURATION.md](TRANSCRIPT_FIRST_CURATION.md).
The legacy selector/arbiter route and its historical low-effort JSON budget below
remain documented for compatibility. Editorial curation explicitly uses V4 Pro
high thinking and does not reuse an approval after changing a cut.

## State and ordering

- `UNVERIFIED`: discovered range; `complete_thought: null`.
- `PASS`: explicit source-range-bound review; `complete_thought: true`.
- `FAIL`: confirmed incomplete thought; `complete_thought: false`.
- `NEEDS_REVIEW`: missing, malformed, contradictory or stale review evidence.

The canonical interpretation lives in `shorts_intelligence/completeness.py`.
Explicit legacy booleans remain supported; missing flags never default to true.
Structured evidence takes precedence and cannot fall back to a legacy pass if
its reason, method or source range is invalid.

The runtime path is:

1. Discover and score candidate ranges; preserve existing hard rejections.
2. Learner abstains on unknown completeness (no bonus and no false failure).
3. Quality-qualified candidates reach the LLM with retained script and local
   before/after context. Selection requires an explicit boolean
   `complete_thought: true` and a nonempty rationale explaining setup/payoff.
4. Record `PASS` with method `text_only`, reason and exact source range. This
   allows a **draft**, not publication. No-LLM mode cannot approve unknowns.
5. Apply bounded learner adjustments to reviewed drafts, without changing
   boundaries. Reapplying the policy does not compound the previous bonus.
6. Carry evidence through discovery JSON, highlight YAML, returned highlights,
   exported edit sidecars and canonical production-feature telemetry.
7. Encoded-file QC and hash-bound owner listening/visual/release approval remain
   separate. Text-only completeness does not certify word timing or audio.

## Reporting and recovery

`selection_summary` distinguishes `DRAFTS_SELECTED`,
`NO_PUBLISHABLE_CANDIDATES`, `REVIEW_REQUIRED`, `DISABLED` and
`NON_CRICKET_SOURCE` (the last is a source-level pipeline outcome).
The orchestrator exposes candidate/rejection/review counts. An unavailable or
invalid editorial review producing no drafts is reported as a failure, not a
successful empty editorial selection. Explicit editorial abstention with reasons
is still valid: there is no minimum-output quota.

The November 2026 blocker was the unconditional discovery assignment of `False`
before a learner that interprets `False` as confirmed failure. All candidates
were removed before the arbiter ran. Do not repair it by globally setting true,
lowering quality thresholds or switching to shadow mode.

## Regression coverage

`tests/automation/test_completeness_flow.py` exercises the real discovery,
agents, policy, selector and arbiter with external media/model calls stubbed.
It covers active/shadow mode, unknown/failed states, explicit AI attestation,
stale ranges, malformed input, idempotent bonuses, truthful export telemetry,
and exact YAML edit points. Existing owner-release tests remain authoritative.
`dry_run.py` now enables the real learner against its disposable sandbox DB.

## Structured model responses

The arbiter uses native `response_format: json_object` on the exact configured
provider/model. It requires a completed (`finish_reason: stop`), nonblank final
JSON object and never substitutes reasoning text or repairs a partial decision.
The example in the prompt is valid JSON, not expressions such as `score: 0-100`.
Retries remain bounded to two arbiter attempts and include validation feedback.
Bounded final-answer/error evidence is saved in the editorial sidecar.

For native DeepSeek, low reasoning effort and a 16,384-token output budget leave
room for the final decision; the previous free-text route could exhaust its
8,192-token budget without a usable decision. This new route does not change
ordinary text/SEO generation or silently switch provider/model.

Provider references:
- https://api-docs.deepseek.com/guides/json_mode
- https://api-docs.deepseek.com/guides/thinking_mode

## Validation and live-run evidence (2026-11-04)

- Required suite: **1,148 passed, 4 expected failures**. Required dry run: PASS.
- `4SOmJSCdzrw`, run `451af37d`: 29 candidates, 8 hard rejections,
  8 below threshold, 13 actually reviewed, 11 editorial rejections, **2 selected**.
  Native JSON completed on the first attempt. Export correctly held both
  estimated caption cuts because they straddled source words.
- A source-word revision of the prediction thought (33.235–49.715s) passed fresh
  text review and word-alignment checks. Original ASR and failed run artifacts
  were preserved. One ASR token (`मैज` → `मैच`) was cross-checked against the
  original YouTube caption; no timestamps or confidence were changed.
- Final export run `c57fdfc7` was **held**, not successful: the exact Flash
  caption response failed the source-number-preservation check. No clips were
  exported or uploaded. This remains a caption/listening-review issue; do not
  bypass the numeric-fidelity or owner-release gates.

Artifacts remain under `highlights/4SOmJSCdzrw/`, `shorts/4SOmJSCdzrw/`, and
`logs/shorts-4SOmJSCdzrw-*-result.json`. The final revised source and review plan
are in `highlights/4SOmJSCdzrw/aligned-caption-review-20261104/`.
