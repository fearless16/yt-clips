# Cancer register: Shorts automation

Review date: 2026-11-27. Scope: automation/, export.py, upload.py, scheduler.py and config.yaml. face_os is excluded.
Status key: **CONFIRMED** means reproduced from code and on-disk or log evidence. **PLAUSIBLE** means the code path is real but impact is unmeasured.
Review: no code changed. Rehaul on 2026-11-27: see **Fix log** at the end. Statuses below are updated.

## Funnel evidence at review time
- On disk: 33 mp4, 7 metadata, 7 `_review.json`, **0 `_delivery.json` receipts**.
- All 7 reviews are **UNAPPROVED**, with only 1/10 or 1/12 checks PASS (the auto `full_decode`).
- Real-source runs in `logs/pipeline.log` (fixtures excluded): **0 uploads**.
- Top hold reasons in real runs: `Missing or invalid ASR confidence` 36, `Missing word timestamps inside selected range` 36, `NO_PUBLISHABLE_CANDIDATES` 36, `Editorial curation held` 23, `Gameplay coverage held` 22, `Release held (REVIEW)` 7, `Cut straddles a spoken word` 4.

---

## C1. CRITICAL. The release-review gate is unreachable, so nothing publishes. FIXED 2026-11-27
- `config.yaml:15-16` sets `release_review.required: true`.
- `automation/release_review.py:21-26, 168-183`: release needs 10 hand-written checks (12 for gameplay). Each needs status, evidence and method strings, plus `selection.reason`, `alternatives`, `coverage_gaps == []`, and then an `approve` CLI call.
- `orchestrator.py:775-782` creates the review draft in the same run that then tries to upload (`:812-818`), so the first run always holds.
- Any artifact change (re-render, SEO retry, cover change) breaks the hash match (`release_review.py:196-198`). `create_review` refuses to overwrite an existing review (`:125-126`), so a stale review deadlocks until someone deletes the file by hand.
- Also blocks Instagram fan-out (`export.py:2075-2077`).
- **Fix direction:** one-command review (contact sheet plus play, then y/n per clip). Auto-fill the machine-verifiable checks (decode, duration, aspect, caption-word match, credit present). Archive and recreate automatically when artifacts change. Optionally an `auto` trust tier after N approved clips with no rejects.

## C2. CRITICAL. The speech-alignment gate kills normal cricket speech. FIXED (thresholds still unmeasured)
- `automation/speech_alignment.py:237-250` holds a clip for any of these:
  - a lead or trail gap over 1.5 s
  - **any internal gap over 2 s**
  - mean Whisper probability under 0.75
  - more than 15% of words under 0.5 probability
- `:197-198` (`words_in_range`) hard-fails when any segment in range has no words. Whisper emits wordless segments over crowd noise or music. PLAUSIBLE root cause of the 36 "Missing word timestamps" holds.
- The gate runs twice: in editorial alignment (`:142`) and again at export (`export.py:2015-2017`).
- Source transcripts are YouTube captions with **0 word timings** (all 3 files under `transcripts/` checked). So every clip depends on one local Whisper pass clearing these thresholds.
- **Bias:** wickets and sixes produce crowd roar and commentator pauses over 2 s. The gate selects *against* the best moments.
- **Fix direction:** first measure the probability and gap distribution on 3 real sources (anti-pattern A7). Then treat gaps as edit decisions (trim, or keep as a reaction beat) and low confidence as a caption-review flag, not a cut-killer. Skip wordless non-speech segments unless they fall at a cut edge.

## C3. HIGH. A single LLM is the only path to any clip, and its failures are opaque. FIXED except the theatre scorers (A5)
- `clip_selection/pipeline.py:1026` forces every candidate to UNVERIFIED. `:1064` keeps only `completeness is True`, which only `editorial_planner._finalize` sets (`editorial_planner.py:230`).
- Log: 28 holds have an **empty** error (`Editorial decision unavailable: "}`) because `str(exc)` was empty (`editorial_planner.py:115-121`). 8 more are "Response requires a proposals list".
- Phrase anchors must match exactly once (`speech_alignment.py:48-49`, `editorial_planner.py:172-175`). An LLM that quotes Hinglish in a different spelling than the captions gets its proposal dropped.
- The 7-agent scorer, topic segmenter, RMS and `ClipScorer` (`pipeline.py:899-1021`, `orchestrator.py:482-513`) run on every source but are only "hints" in editorial mode. They cost time and decide nothing (A5).
- **Fix direction:** log the exception type and repr. Add a deterministic fallback that sends the top heuristic windows to the review queue. Make anchor matching tolerant (fuzzy or transliteration-normalised, with the best unique match).

## C4. HIGH. Gate stacking. FIXED 2026-11-27 (second pass)
- The cricket-only predicate runs 4 times: `pipeline.py:865`, `editorial_planner.py:183`, the export prefilter (`export.py`, around line 1898), and `seo/seo.py:1383,1425`.
- The completeness check runs twice: `pipeline.py:1064` and `export.py:1992`.
- `seo/seo.py` has 22 or more `SEOGenerationError` hard blocks, for example `:1990`, which blocks when the title names an entity not spoken in the clip.
- The SEO deadline is batch-wide (`export.py:2146-2159`). One slow call marks every clip still waiting as failed (seen in `wRP_KO9vP5g-stock` batch_failures).
- **Fix direction:** one gate per predicate, at the earliest stage. Downstream stages read the verdict. SEO failure falls back to a safe template title for review, not a hold.

## C5. HIGH. One bad clip aborts the whole batch. FIXED
- `orchestrator.py:723-738`: if any clip fails delivery QC, `_abort_required_stage` returns before thumbnails, review and upload for **all** clips.
- `orchestrator.py:156-168`: `editorial_target` marks the run failed when there are fewer than 2 exports, even when 1 good clip exists.
- **Fix:** quarantine the failing clip and continue with the rest (A2).

## C6. HIGH (policy/reach risk). The default composition is stock gameplay split-screen. OPEN (owner decision)
- `config.yaml:136-137` uses `cricket-gameplay-stadium-v1` with an EA07 stock library.
- YouTube's reused/inauthentic policy targets "mass-produced, generic, repetitive" content and "clips … with little or no narrative" (https://support.google.com/youtube/answer/1311392, OFFICIAL). Unrelated gameplay under a talking track is a well-known low-effort template. Its effect on distribution is UNVERIFIED but it is a monetisation risk.
- Third-party game footage needs licence evidence (`release_review.py:113-117`), which makes C1 harder.
- `Gameplay coverage held` = 22 in real runs (third-largest hold).
- **Fix direction:** make portrait-of-own-stream the default. Run gameplay only as a measured A/B arm through `shorts_intelligence`.

## C7. MEDIUM. Schedule collisions and expired schedules. COLLISIONS FIXED; expired-schedule hold OPEN
- `scheduler.py:198-224` builds slots from "now" without reading existing receipts or scheduled videos. Two runs on the same day stack. A lone clip always lands at 19:00 (`:157-160`).
- `youtube_delivery.py:175-177`: once a persisted `publish_at` passes, the delivery is held permanently ("explicit reconciliation required"). That is likely, given that review is slow (C1).
- `config.yaml:247-250` sets `avoid_days`/`priority_days` from about 200 Shorts with a median of about 325 views. That is overfitting (A9).

## C8. MEDIUM. The thumbnail is on the critical path. FIXED
- `upload.py:391-402`: if the cover fails validation, the whole upload preflight fails.
- Delivery then waits up to 180 s for processing plus 90 s to verify, per clip.
- Whether the Shorts feed shows custom thumbnails is UNVERIFIED. Blocking delivery on it is disproportionate. **Fix:** make the cover best-effort after the video is live.

## C9. MEDIUM. Upload errors are swallowed and quota is unhandled. FIXED
- `upload.py:502-504` returns None. `orchestrator.py:919-920` then raises the generic "upload returned no video ID", and the real API reason is lost (A13).
- There is no `quotaExceeded`/`uploadLimitExceeded` handling anywhere, so the loop keeps calling the API.
- `youtube.upload_max_retries` and `upload_deadline_seconds` are **dead config**: only `dry_run.py:63-64` references them (A11). `resumable=False` (`upload.py:477`).
- `orchestrator.py:800`: the presence of `cookies.txt` counts as YouTube auth, but upload only uses the OAuth token.

## C10. LOW. Tests pollute the production log. CONFIRMED
- `logs/pipeline.log`: 1283 runs for fixture `AbC123xyz98` and 205 for `youtu.be/test`. Raw log stats are meaningless (A10).

## C11. LOW. Duration rules contradict each other. FIXED 2026-11-27
- `highlight.target_duration: 15` against output 10–38 s. Proposals are allowed up to 44 s (`editorial_planner.py:177`), but alignment requires 10–38 s. An LLM-chosen 40 s complete thought is accepted, then held.

## C12. LOW (process). Plan sprawl.
- The repo root has 4 review/strategy docs (about 1400 lines) and `docs/` has 12 more. Earlier sessions added gates and docs, but nothing reached a receipt (A8).

---

## External facts used (evidence labels)
| Fact | Label | Source |
|---|---|---|
| Square or vertical and ≤ 3 min counts as a Short (uploads from 2024-10-15) | OFFICIAL | support.google.com/youtube/answer/15424877 |
| A Short over 1 min with an active Content ID claim is blocked globally | OFFICIAL | same |
| Reused content without narrative is not allowed. Commentary with your own storyline is allowed | OFFICIAL | support.google.com/youtube/answer/1311392 |
| Since 2025-03-31, Shorts views count each play and replay. "Engaged views" is the old metric | SECONDARY | TechCrunch, 9to5Google |
| `#shorts` is not required for classification | UNVERIFIED (absent from docs) | — |
| videos.insert quota cost (research said "1 unit, 100/day"; older docs say 1600 units) | **CONFLICTING. Check your Cloud console** | developers.google.com/youtube/v3/determine_quota_cost |
| Unverified API projects force uploads to private (research says no longer) | **CONFLICTING. Check the `privacyStatus` readback** | developers.google.com/youtube/v3/docs/videos/insert |
| Feed-distribution signals (swipe-away, APV) | UNVERIFIED (no official doc) | — |
| Safe zones on 1080×1920 | ANECDOTAL (no official spec) | third-party tools |

## C-UPLOAD-READBACK (FIXED 2026-11-27, found in job ff59721c)
`automation/youtube_delivery.py` `finish_publication` read the status once, immediately after
`videos.update`, and raised "Publication readback does not yet match the authorized intent".
On the real run Iv07M7UbJwg, both videos (SuVwNwAbfko, HSpXuZw81NQ) were actually processed and private,
with the correct publishAt and thumbnails VERIFIED. Minutes later the readback matched. This was
eventual consistency, not a real failure, but it reported 0 uploaded and left the receipts in PUBLICATION_UNKNOWN.
Both receipts were reconciled to COMPLETE by hand from the live readback.
Fix (branch fix/upload-readback-poll): readback now polls `PUBLICATION_READBACK_ATTEMPTS=5` times,
`PUBLICATION_READBACK_DELAY_S=3.0` apart, before it raises. The update is still sent exactly once, and
the error now names the observed privacy and publishAt. These are not gates (A1): the same predicate
is retried, not added. The 3 s delay is a guess (A7) based on one observation that "minutes later it
matched"; widen it if the raise recurs. Regression: tests/automation/test_publication_readback_lag.py.

## Open questions / traps (ask the user, do not chase)
1. Is the owner willing to review clips manually at all? If not, C1 must become an auto tier and not just a faster form.
2. Is the stock-gameplay composition a deliberate brand choice, or an experiment? (C6)
3. The quota and unverified-project facts conflict (see table). Check the Google Cloud console for the API project's audit status and quota.
4. Were the 36 "Missing or invalid ASR confidence" holds caused by older code paths? Repro on one real source with current code before you trust the count.
5. ~~Does `seo/seo.py` need a register pass?~~ Replaced by a ~330-line grounded packager (fix log, pass 2).
6. Do real Hinglish Whisper `medium` word probabilities sit below 0.75? This needs a measurement run. Do not guess (A7).

---

## Fix log 2026-11-27 (rehaul, no hot patches)
| Item | Change | Where |
|---|---|---|
| C1 | Release review rebuilt (schema v2). Every check is recomputed from disk: decode/QC, metadata, completeness, source range, word-edge cut, gameplay rights. No hand-typed evidence. The owner gives one y/n through `python -m automation.release_review review shorts/ --open`. `release_review.mode: auto` approves only when every machine check is PASS. Stale or legacy reviews are archived to `.review-archive/` and rebuilt each run. Rejections stick until `reset`. | `automation/release_review.py`, `orchestrator.py` (prepare_review), `config.yaml` |
| C2 | `alignment_issues` keeps only cut-integrity holds: invalid, straddled or overlapping words, no speech, >1.5 s dead air at the edges, mean confidence <0.4. Interior pauses and low or missing confidence became `alignment_flags`, shown to the reviewer. Wordless noise segments are skipped, but a wordless segment **with text** still holds. | `automation/speech_alignment.py` |
| C3 | The LLM error keeps its type and repr. Anchors are transliteration-normalised (Devanagari→roman) and fuzzy (≥0.8, unique best). A provider outage now routes the top heuristic windows to the owner as `heuristic_fallback`, which is never auto-approved. | `editorial_planner.py`, `speech_alignment.py` |
| C5 | A QC failure quarantines only that clip. `editorial_target` fails the run only when there are 0 exports. | `orchestrator.py` |
| C7 | Slot assignment avoids `publish_at` values already reserved in `*_delivery.json`. | `scheduler.py` (`occupied_publish_times`) |
| C8 | An invalid cover means the video uploads without one. A thumbnail set or verify error is recorded as `thumbnail.status=FAILED` and publication proceeds. Processing failure still holds. | `upload.py`, `youtube_delivery.py` |
| C9 | `upload_video` raises `UploadRejected(reason, stop_batch)` and no longer returns None. A 4xx on insert clears the pre-insert receipt. quota/auth stops the loop and lists pending clips. 5xx/timeout keeps `VIDEO_UPLOAD_UNKNOWN`. `cookies.txt` no longer counts as upload auth. Dead keys `upload_max_retries`/`upload_deadline_seconds` were removed. | `upload.py`, `orchestrator.py`, `config.yaml`, `dry_run.py` |

**Measured after (read-only, the 7 real exports on disk):** 1/7 is auto-ready, up from 0/7 READY. The other 6 are held correctly by `gameplay_rights_and_attribution`: 4 are a LOCAL_REVIEW_ONLY stock library and 2 are gameplay evidence that changed after render. That is C6, an owner decision, not a bug.
**Tests:** 1520 passed, 0 failed (face_detect excluded). `dry_run.py` OVERALL PASS.

## Fix log 2026-11-27, pass 2 (discovery + SEO rewrite; layout and gameplay untouched)
| Item | Change | Where |
|---|---|---|
| C4 cricket ×4 | One predicate per item, at birth. Source gate runs once. Window gate `is_cricket_window` (own words, ≥2 cricket signals, ≥7% density) runs once in `_build_livestream_windows`, and editorial proposals are checked once in `_proposal`. The export prefilter and the SEO cricket check are deleted. | `clip_selection/pipeline.py`, `export.py`, `seo/seo.py` |
| C4 SEO blocks | `seo.py` rewritten (2494 → ~330 lines). One grounded LLM call per clip and at most one corrective retry. Style (length, emoji, hashtags, tag budget, cover words) is **repaired**, never blocking. The only rejection is a fact problem (a name nobody said, clickbait filler, empty fields). After one retry it becomes `seo_method: template`, which release review marks `NEEDS_OWNER` (never auto). Provider errors keep `type: reason`. `vidiq.py` and the trend research in the SEO path are removed. | `automation/seo/seo.py`, `release_review.py` |
| C4 SEO deadline | The batch-wide deadline is gone. The per-clip budget is 2×`clip_timeout_seconds`+30 s from that clip's submit. A timeout degrades **that clip** to template packaging. Workers return metadata and the export thread writes it, so a late worker cannot write anything. | `export.py` |
| C4 retry | `retry_failed_seo` always resolves a legacy `_seo_failed.json` marker (LLM, or template for the owner, A4). It upgrades template packaging when the provider is back, only on `--skip-export` re-runs, and never touches a clip with a delivery receipt. | `seo/seo.py`, `orchestrator.py` |
| A5 | Removed the 7-agent selector, arbiter, topic segmenter, cricket heuristics, keyword lists, `automation/scoring/*` lock score and the orchestrator hook-audit stage. None of them decided anything once editorial curation was on. `hook_auditor.analyze_clip_hook` is kept, because export's hook overlay (layout) reads it. | `clip_selection/`, `scoring/`, `orchestrator.py` |
| Ranking | Windows are ranked 0–100 by loudness spike against the stream's own robust baseline (median/MAD), reaction language, a hook in the first 3 s, a payoff in the last 3 s, cricket density, source-team match and duration fit, with a lead-in penalty ("toh guys…"). Ranking only orders the editor's suggestions and the outage fallback; it never holds a clip. **Weights are unmeasured priors (A7).** | `clip_selection/pipeline.py` (`score_window`) |
| Dedupe (new bug) | Cross-run dedupe read `highlights/<match>.dedupe_history.yaml`, but **nothing ever wrote it**, so re-runs could re-pick the same moments. Now the orchestrator records exported windows (`record_exported_windows`) and discovery filters against them. | `pipeline.py`, `orchestrator.py` |
| C11 | Clip length comes from one source: `clip_selection.output_min/max_duration`. The editor's proposal ceiling is that max + 2×`alignment_max_boundary_shift`, not a hard-coded +6. Dead keys were deleted: `highlight.{use_ai_refinement, audio_energy_threshold, max_duration, merge_gap, max_clips, fast_speech_wpm, silence_penalty_seconds}`, `clip_selection.{enabled, use_llm_arbiter, min_quality, min_selected, prefer_source_match, source_match_min_candidates, editorial.enabled}`, and 13 `seo.*` keys including `vidiq`. | `config.yaml` |
| Empty run | Zero publishable candidates is an honest `empty` run. `editorial_target` fails only when drafts were selected but none exported. | `orchestrator.py` |
| Dry run | It now exercises the LLM SEO path and fails if any clip falls back to a template. Its acceptance checks were keyed on the deleted `editorial.enabled` key and were re-enabled. The class-level provider rate limiter is reset per sandbox. | `dry_run.py` |

**Measured (read-only discovery on 3 real transcripts, no LLM):** 1,878–2,503 cricket windows reduce to 58–60 diverse candidates per stream. Score p10/p50/p90 is 26/47/68 with audio, and 19/32/41 and 19/29/41 transcript-only. The top picks are real cricket talk (bowling figures, wicket counts, scores). Discovery CPU is 100–135 s per source (see traps).
**Tests:** 1295 passed, 0 failed, 4 xfailed (face_detect excluded). About 240 tests of deleted internals were removed and 12 contract tests added (`test_seo_grounded.py`, `test_discovery_ranking.py`). `dry_run.py` OVERALL PASS.

### Pass 2 review: real-data spelling defect (2026-11-27)
**Found:** Whisper and YouTube emit Hindi in Devanagari. `to_roman` keeps the inherent vowel (क्या→kyaa, धोनी→dhonee, आउट→aauta). The lexicons and the name table held only casual spellings. On 3 real streams:
- The reaction signal matched only "wicket" and "abe" across 178 candidates, so the ranking was effectively energy + density + duration.
- The cricket gate missed romanized names. Dhoni alone occurs 108 times.
- SEO grounding would have flagged a correct "Dhoni" title as invented and routed the clip to template packaging (NEEDS_OWNER).

**Fixed:**
- `CRICKET_SPELLING_CORRECTIONS` now has romanizer spellings for the player names and cricket terms that actually occur, with counts. Look-alikes that are ordinary Hindi words were excluded by hand (bolate, kauna, andara, khaana, puraanaa). A generic fuzzy key was rejected as a trap, because schwa rules collide (puraanaa ≈ Pooran).
- The reaction, lead-in and hook lexicons carry both spellings. Too-common words (gaya, kya, no, what) count only inside phrases. "?" is no longer a hook, because `to_roman` emits it for unmapped characters.
- The seed correction regex is compiled once (`_static_corrections`).
- The editor LLM gets hints for its own section as cue ranges with the audio signals, without repeating the cue text.

**Measured after:**
- Cricket windows: 1,896→5,600, 1,878→4,060 and 2,503→3,087.
- Windows with any reaction hit: 30–62%. Hook text fires on 19–26% of openings, so it discriminates.
- Candidate duration p50: 13–19 s (it was 13–14 s).
- Discovery CPU is unchanged at 89–123 s per source while handling ~2.5× more windows.

**Tests:** 1298 passed. `dry_run.py` PASS.
**Remaining trap:** LLM proposals use `is_cricket_content(text, source_title)`, so the title can rescue a proposal. Windows use the stricter `is_cricket_window`. This leniency is deliberate for the editor, but it is two predicates for one question. Owner decision.

### Pass 3: banter policy, GPU, caption language (2026-11-27)
- **Owner policy:** a cricket stream's Shorts may be non-cricket banter or a live moment.
  The only cricket gate is now the source gate (`detect_highlights`). The per-window gate
  (`is_cricket_window` in `_build_livestream_windows`) and the per-proposal gate
  (`editorial_planner._proposal`, which let the title rescue it) are gone. This closes the
  "two predicates" trap. Window `cricket` and `cricket_density` are ranking labels only.
  The discover/review prompts now ask for banter and live moments on the same payoff bar,
  and reject stream-tech talk and inside references. The SEO prompt no longer forces cricket.
- **Discovery cost:** cricket counts are summed per thought, not re-scanned per window.
  Measured on 3 real streams: windows 24,124 / 18,044 / 20,263, discovery CPU 5 / 1 / 1 s
  (was 123 / 89 / 102 s on about 4-5x fewer windows). Top-6 picks are still mostly cricket,
  because cricket_density (15) and cricket reaction words favour it. Without audio (2 of 3
  streams) banter can only rank on text reactions. The editor LLM reads all sections, so
  banter reaches it through discovery, not through the hints.
- **GPU (Radeon AI PRO R9700, 32 GB):**
  - Before: transcribe.py picked whisper_vulkan first and its pref picked `small`, though
    `large-v3-turbo` is on disk.
  - 5 min of real Hindi audio: vulkan+small 57 s, rocm+turbo 41 s, rocm+turbo -p 2 24 s
    (-p 4 also 24 s); small garbled names.
  - Now ROCm is first and turbo leads both prefs.
  - Still on CPU: `speech_alignment` (faster-whisper medium int8; CTranslate2 has no
    ROCm on Windows), and the gameplay/stadium export (libx264 by design: x264 quality per
    bit over AMF, layout untouched). Logged, not changed.
- **Caption language:** Data API and youtube-transcript-api preferred English tracks. On a
  Hindi stream that is a translation or mis-tagged ASR. Now the configured
  `transcription.language` goes first. All 3 cached real transcripts are `youtube_api`
  Hindi Devanagari with no word timings, so alignment re-times the cut ranges.
- **Open:**
  - shorts_intelligence `domain.py` marks Shorts non_cricket and drops them from training.
    Uploaded banter Shorts would then never be learned from. That is an owner decision
    (learner ownership), not changed here.
  - The banter share among the editor's picks is unmeasured until a real DeepSeek run.

### Real local run + intelligence audit (2026-11-27)
- **Run:** `automation.cli <H8OjNZ8XLbI> --skip-download --skip-transcribe --skip-instagram`
  (no upload), run_id f7b0c7c8.
  - Result: status ok, exported=2, failures=0. Layout cricket-gameplay-stadium-v1, 1080x1920,
    20.5 s and 16.6 s. QC PASS, grounded LLM titles.
  - Both picks were cricket debate (no banter chosen this time).
  - Elapsed 1,459 s, of which selection took 1,367 s:
    - DeepSeek V4 Pro "high thinking": 4 sequential calls, 253 + 174 + 186 + 75 s.
    - CPU speech alignment: about 11 min for 6 ranges.
  - Fixed alignment (config): turbo is now primary. On CPU, 30 s of audio takes medium 122 s
    vs turbo 14.5 s (8 threads made no difference). Medium failed 2 of 3 ranges.
  - Still open: `release_review.mode: auto` approved both clips with visual and listening
    "NOT CHECKED". With --upload they would publish unwatched.
- **shorts_intelligence:**
  - The selection influence is dead (A5). `apply_selection_policy` runs only at discovery,
    where every candidate is UNVERIFIED, and the policy returns 0 for UNVERIFIED.
  - Measured: applied=0 and shadow=0 for every candidate.
  - Even when applied, it has 1 active recommendation (duration under_15, effect +0.028),
    capped at 3 of 100 points, on a score the editor treats as advisory.
  - The model was last fitted 2026-08-20.
  - Value is the store: 213 Shorts and 381 real Analytics snapshots, the only evidence for
    ranking weights and duration.
  - Owner decision: delete the bridge, or use the store to fit ranking weights.

### ASR options research (2026-11-27, cheap-model web research + local measurement)
- **GPU alignment:** whisper.cpp ROCm DTW (turbo) took 11.7 s vs CPU faster-whisper turbo
  14.5 s on 30 s of audio. Word starts disagree p50 0.2 s, p90 2.1 s, with no ground truth.
  No verified Windows gfx1201 CTranslate2 build exists. Decision: alignment stays on CPU
  turbo, about 1.5 min per run.
- **Hinglish STT APIs with word timestamps (vendor-reported; no independent Hinglish WER):**
  - ElevenLabs Scribe batch: $0.20/hr plus $0.05/hr keyterms (up to 1,000 player names),
    15 h per file, words[] with start/end. Ranked first.
  - Deepgram Nova-3 `language=multi`: about $0.31/hr.
  - AssemblyAI Universal-3.5 Pro: about $0.21/hr.
  - Dropped: OpenAI gpt-4o-transcribe (no word timestamps), Sarvam Saaras (no word
    timestamps).
  - Google Chirp 3: word timestamps capped at 20 min per batch request.
  - Vendor table: Whisper large-v3 FLEURS Hindi 33.3% WER.
- **Design note:** an STT whose words carry timings can feed `speech_segments`.
  `align_candidate` then uses the "supplied" path, with no re-transcription and no
  11-minute alignment. Next step is a bake-off on the same 5 min (YouTube captions as the
  reference); it needs API keys from the owner.

### whisper.cpp is the only ASR engine (2026-11-27, branch feat/whisper-cpp-only)
- **Why:** CTranslate2 has no AMD GPU path on Windows, so faster-whisper ran only on CPU. Now
  removed from code, requirements and config (guard: `tests/test_no_faster_whisper.py`).
  `face_os/colab_*.py` still `pip install faster-whisper` in Colab text; face_os is fenced, not touched.
- **Alignment on GPU (reverses the "stays on CPU" decision above):** 5 real ranges of 10-12 s
  (3 streams), ROCm turbo: 2.7-7.5 s each, 0 zero-length and 0 overlapping words (was CPU turbo
  about 14.5 s per 30 s, medium about 122 s and failed 2 of 3 ranges). Fallback model is now
  `small` (only other ggml on disk): zero-length words in 2 of 5 ranges, so it is a weak retry.
- **Trap fixed:** the old whisper.cpp path treated each sub-word token as a word (Devanagari split
  into byte pieces with empty text). Words now come from `-ml 1 -sow` (one segment per whole word).
- **Padding artifact:** whisper.cpp pads to a 30 s window and emitted words at/after the audio end
  (zero-length at the clip end, last word stretched to the window end, e.g. 88.0 s for a 72 s
  context). They lie outside the audio, so `parse_words` drops them and clamps the last end.
  Nothing inside the audio is repaired.
- **Behaviour changes:** no automatic model download (error names the file and URL); a failed
  `transcribe()` now raises the real reason instead of silently switching engine; no per-word
  `--dtw` (t_dtw stayed -1 in a measured run, so not assumed).
- **Open (A7):** whisper.cpp heuristic word times can clump words at one instant (60 s full-source run:
  2 of 107 words zero-length, "के सामने" at 7.68 s; alignment ranges with turbo: 0 of 233 words). Alignment's
  hold/retry policy applies; rate on more sources unmeasured. Segment grouping thresholds (`GROUP_MAX_GAP_SECONDS` 0.5, `GROUP_MAX_WORDS` 25)
  are presentation-only and unmeasured. Full-source `transcribe()` on whisper.cpp with `-ml 1 -sow`
  was not run end to end on a long stream in this change.

### Still open
- Real SEO quality is unmeasured. Run once against DeepSeek on real exports and read the titles. Check whether `find_canonical_entities` misses names in context (e.g. "Arre Kohli ne"). Grounding now also accepts any spoken name part.
- Discovery cost: about 90–120 s CPU per source. The correction regex is now cached. The remaining cost is `find_canonical_entities` per growing window. Caching per thought would fix it. Not chased (trap).
- Ranking weights: calibrate against `shorts_intelligence` outcomes (A7/A9).
- C6 composition/licence: the stock library is not cleared for upload.
- C7b: an expired `publish_at` on an existing receipt still needs manual reconciliation. Needs a `reslot` exit (A4).
- `VIDEO_UPLOAD_UNKNOWN` (ambiguous insert) still has no scripted reconcile. Needs a channel lookup by title/receipt time.
- C10 test-polluted log.
- Measure real Hinglish ASR confidence and gap distributions, then revisit `MIN_MEAN_PROBABILITY=0.4` and `EDGE_GAP_SECONDS=1.5` (A7).
