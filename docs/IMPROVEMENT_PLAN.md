# yt-clips — Staff Engineering Improvement Plan (Phase 1 Deliverable)

> Scope: **legacy cricket pipeline** (`automation/`, `utils/`, root pipeline scripts).
> `face_os/` is **protected / read-only** and is intentionally excluded from all proposed changes.
> Status: **Analysis complete — awaiting plan approval before implementation.**
> Evidence convention: every claim cites `file:line` and was verified by reading source, not inferred.

---

## 1. Architecture Report

The repo hosts **two independent systems**:

| System | Entry | Purpose | In scope here? |
|---|---|---|---|
| **Face OS** | `pipeline.py`, `face_os/*` | Identity-reconstruction for portrait studio video | ❌ Protected — excluded |
| **Legacy cricket pipeline** | `automation/orchestrator.py` | 16:9 livestream → 9:16 cricket Shorts + SEO + upload | ✅ All work happens here |

### Legacy pipeline execution flow (verified)

`automation/orchestrator.py:run()` runs 9 skippable phases, each lazily imported:

```
Phase 0  Transcript (API)   automation/transcript.py  → youtube-transcript-api → yt-dlp VTT → local whisper
Phase 1  Download           download.py               (yt-dlp + aria2c)
Phase 2  Transcribe         transcribe.py             (faster-whisper, hi)  [skipped if Phase 0 got segments]
Phase 3  Highlight          highlight.py              (audio RMS + transcript score + LLM refine)
Phase 4  Export             export.py + frame_analyzer.py / premium_analyzer.py  → 9:16 crop + encode
Phase 4.5 Enhance (opt)     ref_grade.py / face_mapper.py
Phase 5  SEO                automation/seo/seo.py     (LLM cricket SEO)
Phase 5.5 Thumbnails        thumbnail.py
Phase 6  Sync (opt)         sync.py                   (Google Drive)
Phase 7  Upload (opt)       upload.py                 (YouTube Data API v3, token rotation)
Phase 8  Analytics          automation/seo/analytics.py → seo_learner.py
```

### Face detection reality (myth-busted, with evidence)

The product brief states the system uses **Haar Cascade**. **It does not.** Verified by repo-wide grep:

- `utils/face_detect.py` — the shared `detect_faces()` used by the "cheap" path — uses **MediaPipe FaceDetector (BlazeFace)** loading `face_detector.tflite` (`utils/face_detect.py:10-15, 32-50`). No `cv2.CascadeClassifier`, no `haarcascade_*.xml`, no `detectMultiScale` anywhere outside `face_os/`.
- The only "Haar" mentions are **stale comments**: `premium_analyzer.py:3` and `face_mapper.py:152`, plus the `config.yaml` `premium:` comment ("cheap (Haar Cascade)").
- `premium_analyzer.py` uses **YOLOv8n-face** on GPU (`premium_analyzer.py:237`) with **ByteTrack + Kalman** tracking (`premium_analyzer.py:50, 120-227`) and bezier crop smoothing.

**Implication:** the face-detection task changes from "replace Haar" to "**GPU-accelerate + batch the already-modern detectors, add identity locking, and delete stale Haar references.**"

---

## 2. Dependency Graph (impacted modules only)

```
orchestrator.py
 ├─ automation/transcript.py ──────────────── (youtube-transcript-api, yt-dlp) ──┐
 ├─ transcribe.py ── faster-whisper ── utils/ai_client.py (LLM correction)        │ Transcript text
 ├─ highlight.py ── utils/ai_client.py                                            │ feeds SEO
 ├─ export.py
 │   ├─ frame_analyzer.py ── utils/face_detect.py (MediaPipe) ── utils/face_matcher.py (dlib)
 │   └─ premium_analyzer.py ── ultralytics YOLO (GPU) + ByteTrack/Kalman ── utils/face_detect.py (fallback)
 ├─ automation/seo/seo.py  ◀── HIGHEST PRIORITY
 │   ├─ automation/seo/trends.py ── (Google Trends RSS, YT suggest, Google News RSS, Cricbuzz scrape)
 │   ├─ automation/seo/cricket_context.py (spelling correction + canonical sets [DEAD])
 │   ├─ automation/seo/seo_learner.py ── enhance_seo_prompt() / get_best_model()
 │   └─ utils/ai_client.py ── generate_fastest_first()  [BYPASSES health layer]
 ├─ upload.py ── google-api-python-client (YouTube Data API v3)
 └─ automation/seo/analytics.py ── YouTube Data API v3 (statistics only) ── seo_learner.py

Shared infra:
  utils/ai_client.py    — multi-provider LLM client (groq/deepseek/openrouter/nvidia + ollama)
  utils/resilience.py   — CircuitBreaker + retry_with_backoff  [UNUSED by LLM path]
  utils/logger.py       — structured JSON logger  [under-adopted: 505 print() calls repo-wide]
  automation/_cache.py  — TTLCache singletons
```

**Cross-subsystem coupling that matters:** transcript quality → SEO quality (player names); SEO output `_generated_by_provider/model` + `data/seo_performance.json` → self-learning; analytics → `seo_learner` → `enhance_seo_prompt` → SEO prompt.

---

## 3. Module Findings (evidence-based)

### 3.1 SEO — `automation/seo/seo.py` (HIGHEST PRIORITY) — **violates "no generic fallback"**
- **Forbidden generic tag padding:** `_enforce_limits` appends `safe_defaults = ["cricket highlights", "cricket live match", ...]` whenever `<10` tags exist (`seo.py:826-847`). Runs on **every** path (AI, salvage, transcript). This is exactly the forbidden generic tag generation.
- **Random hardcoded hooks/CTAs:** `_inject_viral_elements` uses `random.choice(VIRAL_HOOKS)` / `random.choice(ENGAGING_CTAS)` (`seo.py:476`, lists at `:62/:75`) — non-clip-specific templated copy.
- **Shorts description bug:** `_attempt_seo_generation` calls `assemble_description()` unconditionally (`seo.py:1089`); `is_shorts` is never threaded in. The AI's <400-char Shorts description gets overwritten by the long-form LIVE/CHAPTERS/Disclaimer template → **malformed Shorts metadata** (core deliverable broken).
- **Degradation, not escalation:** on parse failure → "salvage" branch fabricates metadata but still flags `ai_generated=True` (`seo.py:1132-1198`); on no response → transcript template with hardcoded broadcasters/chapters/disclaimer (`seo.py:1200-1279`). Docstring `seo.py:1069` ("Never uses template fallback") is false.
- **Cricbuzz is match-blind:** `fetch_cricbuzz_live_score` ignores its `query` and scrapes a generic landing page with brittle selectors (`trends.py:164-180, 126`); empty results silently become placeholder text ("Toss details not available", `seo.py:377`).
- **Factual correction is partial/risky:** `correct_cricket_spelling` runs on transcript only (`seo.py:950-951`), not title/AI output; canonical `CRICKET_PLAYERS`/`CRICKET_TEAMS` (`cricket_context.py:60/104`) are **dead code**; false positives like `"head"→"Travis Head"` and hardcoded `"ipl"→"IPL 2026"`.

### 3.2 LLM orchestration — `utils/ai_client.py` — **health layer is dead code on the real path**
- Providers/models in `PROVIDER_MODELS` (`ai_client.py:29-34`); failover chain, token-bucket rate limit (`:80-103`), and circuit breaker (`:112-117`) exist **only in `generate_text`**.
- The actual SEO path uses `generate_fastest_first` (`seo.py:1074` → `ai_client.py:242-296`), which constructs a fresh `AIClient()` per thread and calls providers directly — **never** touching the token bucket or circuit breaker. So rate-limit/health protection is effectively disabled for real traffic.
- **No 429 / Retry-After handling** anywhere despite docstrings claiming "exponential backoff on 429/593" (`seo.py:927, 943`). `utils/resilience.py` has a real backoff helper but is **unused** by the LLM client.
- Some `PROVIDER_MODELS` IDs look speculative (`deepseek/deepseek-v4-flash`, `qwen/qwen3.5-122b-a10b`, `anthropic/claude-sonnet-4`); racer drops slow-but-valid tiers via `timeout=45` + `result(timeout=5)` (`:280-285`).

### 3.3 Transcription — `transcribe.py` / `automation/transcript.py`
- Engine: faster-whisper, config-driven (`transcribe.py:41-50`), `language: hi` (`config.yaml`).
- **Corrections only run on the local-whisper path** (`transcribe.py:133-209`); API/VTT transcripts (the common case) are returned uncorrected (`transcript.py:234-236`).
- Over-broad regex corrupts English: `\bsky\b→SKY`, `\bstark\b→Starc` IGNORECASE (`transcribe.py:135-145`).
- **No validation** of the LLM correction output (indices/length/language); exceptions swallowed (`transcribe.py:196-208`).
- `batch_size` is read but **never passed** to `model.transcribe` → dead config (`transcribe.py:48` vs `:83-89`).
- No diarization, no scoreboard-aware numeric correction.

### 3.4 Self-learning — `seo_learner.py` / `analytics.py` — **mostly heuristic, real signals missing**
- Ingests only `viewCount/likeCount/commentCount` via YouTube **Data** API `statistics` (`analytics.py:142-144, 249-253`). **No** YouTube **Analytics** API → no CTR, impressions, retention, ranking (grep confirms zero usage).
- The scorer's `retention`/`ctr` branches are **dead** because the producer never supplies those keys (`seo_learner.py:213-238`).
- Auto-benchmark picks "best model" from **one synthetic hardcoded prompt** + keyword rubric (`+15 for "kohli"/"virat"`) (`seo_learner.py:318-401`); commits selection at score ≥40 (`:420-423`). Two competing best-model writers (synthetic vs real) with no precedence.
- Pattern-key explosion: numeric features (`title_length`, etc.) in `_stable_pattern_key` (`:47-48`) → near-unique keys → `title_patterns` rarely reaches `min_clips=2` → learning inert.
- `analytics.title` never stored → LLM-insight titles render as "?" (`:633-634`).

### 3.5 YouTube upload — `upload.py` — **API-clean, reliability gaps**
- All request fields are **documented & supported** (snippet: title/description/tags/categoryId; status: privacyStatus/selfDeclaredMadeForKids/containsSyntheticMedia/publishAt). `containsSyntheticMedia` verified real (API revision 2024-10-30). **No unsupported fields.**
- Required defaults present: `selfDeclaredMadeForKids=false` (`upload.py:277`), `containsSyntheticMedia=false` (`:278`).
- Reliability issues: `chunksize=-1` single-shot (`:306`); description truncated by **chars not bytes** (`:271`, API limit is 5000 **bytes**); unbounded retry loop (`:333-342`); no `categoryId` validation (`:273`); synthetic-media flag hardcoded; tag limiter ignores quote overhead (cap 450 vs real 500, `:108-120`).

### 3.6 Observability — `orchestrator.py` / `utils/logger.py`
- Structured JSON logger exists (`utils/logger.py`) but `extra={}` enrichment appears in **only 3 files** (orchestrator 12, ai_client 5, tests 1). **505 `print()`** calls repo-wide.
- Failures are stringified into `result.failures` and replayed at EXIT as plain `log.warning` **without** `stage`, exception type, or traceback (`orchestrator.py:345-346`); `exc_info` is never passed, so the JSON handler's `exc` capture is dead.
- Per-phase timing/`stage` is logged **only on success** (the `done` log sits inside the `try`), so failed phases emit no timing and no stage record.
- `automation/dashboard.py` is an orphaned **RAM/GPU** monitor (not wired to the pipeline; reports resources, not stages/progress/failures).

### 3.7 GPU utilization
- `transcribe.py` requests `cuda/float16` with VRAM logging (`transcribe.py:18-30`) but `batch_size` never reaches the model (no batched inference).
- `utils/face_detect.py` MediaPipe runs **CPU** (no GPU delegate), single-image, `score_threshold=0.5`.
- `premium_analyzer.py` YOLO has **no explicit `device=0`** (relies on ultralytics default) and silently degrades to CPU DNN on load failure (`premium_analyzer.py:237-262`); frame extraction is per-timestamp `ffmpeg` (CPU, un-batched).

---

## 4. Risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | SEO changes alter live upload metadata; bad output → real CTR/revenue loss | High | Golden-output tests + dry-run diff before merge; keep a per-clip metadata diff |
| R2 | Removing fallbacks could make SEO **fail hard** when all LLMs are down | High | Escalation chain (provider→model→stricter prompt) + a "queue for retry" terminal state instead of silent generic output |
| R3 | LLM provider/model IDs may be invalid → whole tiers error | Med | Validate against live model lists; load from config; integration test |
| R4 | Face-detection GPU changes could destabilize crop/tracking | Med | Measure GPU/VRAM + crop stability **before** changing; A/B on `clips_test/test_clip.mp4` |
| R5 | YouTube Analytics API requires extra OAuth scope + token | Med | Add scope, document re-auth; degrade gracefully to Data API if unavailable |
| R6 | Touching shared `utils/` affects `face_os` indirectly | Med | `utils/face_detect.py` etc. are shared — change behind flags, run face_os tests as guard |
| R7 | Network-dependent tests (trends, Cricbuzz, suggest) flaky in CI | Low | Mock external HTTP; mark live tests as opt-in |

---

## 5. Proposed Changes (per branch)

**Branch strategy:** one feature branch per subsystem, off `main`. Foundational branches merge first (observability → LLM orchestration), then the rest can proceed in parallel. Merge each only after its validation passes.

1. **`feat/observability`** (foundational): `run_phase()` context manager that always logs start/end/status/`duration_ms`/`run_id` even on failure; failures logged at site with `exc_info=True` + `stage` + `error_type`; convert `automation/` prints to `log.*`; add retry visibility hook.
2. **`feat/llm-orchestration`**: route `generate_fastest_first` through `_check_and_consume_token` + circuit breaker; real 429/`Retry-After` + 5xx backoff (reuse `utils/resilience.py`); skip providers in cooldown; validate `PROVIDER_MODELS`; tune racer timeouts. Failure → escalate (next tier + stricter JSON prompt + JSON-repair), never degrade.
3. **`feat/seo-quality`** (highest priority): delete `safe_defaults` padding (`seo.py:834`) and `<10`-tag forcing; thread `is_shorts` so Shorts keep the LLM short description; gate/remove random hooks/CTAs (LLM-generate per clip); apply `correct_cricket_spelling` to title + AI output; wire canonical player/team sets for validation + enrichment; make Cricbuzz match-specific or treat empty as a hard signal; de-hardcode `#IPL2026` (derive season from date/live context).
4. **`feat/transcription`**: centralize cricket correction across **all** sources (api/vtt/whisper); roster/config-driven, context-guarded lexicon (drop `sky`/`stark` false positives); post-LLM validation (index-set equality, length/language guard, fallback on mismatch, correction-rate logging); scoreboard-aware numeric hints; fix/wire `batch_size`.
5. **`feat/self-learning`**: add `youtubeAnalytics.v2` query (views, estimatedMinutesWatched, averageViewPercentage→retention, impressions, CTR) → activate the dead scoring branches; exclude numeric fields from pattern key; unify the two best-model writers behind one precedence; store titles; externalize hardcoded weights/lists.
6. **`feat/youtube-upload`**: finite `chunksize` (8–16MB) resumable upload; byte-based description truncation; `categoryId` validation via `videoCategories.list`; bounded retries with deadline; drive `containsSyntheticMedia` from metadata; account for tag quote overhead.
7. **`feat/face-detection-gpu`**: explicit `device=0` for YOLO + batched sampled-frame inference; MediaPipe GPU delegate (or keep as CPU fallback); identity locking using `photos/` + `expectation.png` embeddings; resolve YOLO weight path + fail loudly; delete stale Haar references. **Measure GPU/VRAM + crop stability before & after.**

---

## 6. Validation Results (Phase 1 verifications already performed)

| Claim under test | Method | Result |
|---|---|---|
| "System uses Haar Cascade" | repo-wide grep for Haar/CascadeClassifier/detectMultiScale (excl. face_os) | **FALSE** — only stale comments; real detector is MediaPipe BlazeFace |
| "Generic fallback SEO exists" | read `seo.py` | **TRUE** — `safe_defaults` (`:834`), random hooks/CTAs (`:476`), hardcoded hashtags |
| "Health/rate-limit protects SEO calls" | trace `generate_fastest_first` | **FALSE** — bypasses token bucket + circuit breaker |
| "Self-learning uses CTR/retention" | grep `youtubeAnalytics`/`impressions`/etc. | **FALSE** — zero usage; only Data API statistics |
| "Upload uses unsupported API fields" | read `upload.py` vs API docs | **FALSE** — all fields supported; reliability gaps only |
| "Shorts get a proper short description" | trace `_attempt_seo_generation` | **FALSE** — long template overwrites Shorts description |

---

## 7. Documentation Updates (Phase 2 plan)

- Fix stale Haar references: `config.yaml` (`premium:` comment), `premium_analyzer.py:3`, `face_mapper.py:152`.
- Update `ARCHITECTURE.md` legacy section: detector is MediaPipe (cheap) / YOLOv8-face (premium), not Haar.
- Document actual execution flow (Section 1 above) and **module ownership** map in `automation/AGENTS.md`.
- Reconcile false docstrings in `seo.py:927, 943, 1069` and `ai_client.py` with real behavior.

---

## 8. Final Implementation Plan (sequencing)

```
main
 ├─ feat/observability        (merge 1st — everything else logs through it)
 ├─ feat/llm-orchestration    (merge 2nd — SEO + transcription + learner depend on resilient LLM)
 ├─ feat/seo-quality          (parallel after #2; highest product priority)
 ├─ feat/transcription        (parallel; improves SEO input quality)
 ├─ feat/self-learning        (parallel; needs analytics OAuth scope)
 ├─ feat/youtube-upload       (parallel; independent)
 └─ feat/face-detection-gpu   (parallel; measure-first, shared utils caution)
Then: testing/validation branch — per-module + integration + e2e on a clip from
      https://www.youtube.com/watch?v=4ylLhtICj1I (and clips_test/test_clip.mp4).
```

Each branch: Analyze → Implement → Test → Verify → Update docs → open PR → merge on green.

---

## 9. Open Questions for Approval

1. **Priority order / which branches this session?** Recommend starting with `feat/observability` → `feat/llm-orchestration` → `feat/seo-quality`.
2. **Hard-fail vs queue on total LLM failure** for SEO (no generic fallback): acceptable to **skip+queue** a clip rather than emit generic metadata?
3. **YouTube Analytics API**: OK to add the extra OAuth scope (`yt-analytics.readonly`) and re-auth, required for real CTR/retention learning?
4. **Live external calls in tests** (trends/Cricbuzz/suggest): mock by default, live tests opt-in — agreed?
5. **GPU work** can only be *measured* in a real T4 environment (this sandbox has none). OK to implement + unit-test logic here and validate GPU metrics on Colab/Kaggle?
