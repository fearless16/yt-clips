---
name: shorts-cut-quality
description: How to choose and cut cricket Shorts in yt-clips so clips are watchable and actually survive to export. Use when changing clip discovery, editorial_planner, speech_alignment, export cut boundaries, durations, captions timing, or composition. Pairs with shorts-automation-guardrails.
---

# Cutting cricket Shorts properly

Load `shorts-automation-guardrails` first. Known defects live in its `references/cancer-register.md` (C2, C3, C6 and C11 are about cutting).

## What a good cut is (viewer contract)
1. **Second 0–2 carries the hook.** Open on the claim, the reaction or the action, never on a lead-in like "so guys…" or "haan toh…". Lead silence must be at most 0.3 s. Remove the gap; do not reject the clip for it.
2. **One complete thought.** It has the setup that is needed and a payoff that lands inside the clip. Never end mid-sentence or on a topic change.
3. **The payoff is the last beat.** End within about 0.5 s after the payoff word or the reaction peak. Do not trail.
4. **Duration follows the thought**, normally 10–38 s. A strong complete 40–60 s thought is valid, because Shorts allow up to 180 s. Anything over 60 s with a Content ID risk is blocked globally. Do not let proposal limits and alignment limits disagree (C11, fixed: one bound, `clip_selection.output_*_duration`).
5. **Reaction beats are content.** Crowd roar, a commentator's stunned pause, laughter. Gaps over 2 s at these moments are often the best part. Never treat "speech gap" as "defect" without context (C2).

## Discovery rules
- Speech is not the only signal. Rank windows by `transcript claim + audio energy spike + (when available) visual event`. A pure-transcript pipeline misses the moments that are mostly sound. Implemented: `pipeline.score_window`, which ranks only and never holds. Visual events are still missing.
- The cricket check runs **once per item at birth** (source, window, editor proposal). Never re-check it downstream (A1).
- Give the LLM editor cue IDs and exact text. Expect it to transliterate Hinglish differently, so match anchors tolerantly (normalised, best unique match) rather than with exact substrings (C3).
- An LLM outage must not mean zero clips. The deterministic top-N windows go to the **review queue** with `semantic_review: "heuristic"`, never straight to publishing.
- Dedupe across runs by source video ID plus window overlap. Exported windows are recorded by `record_exported_windows`. Before 2026-11-27 nothing wrote the history.

## Boundary rules (word level)
- Cut on word edges taken from local word-timed ASR. Snap to the nearest word gap of at least 80 ms, and add about 0.1 s of pad at the head and 0.25 s at the tail.
- Wordless ASR segments (noise or music) inside the range are **not** an error unless they sit at the cut edge (C2).
- Low ASR confidence is a **caption** problem (flag the words for review). It is not a **cut** problem.
- Check boundaries once, at alignment. Export reuses that verdict. Do not re-run the same gate (A1).

## Render rules
- Output 1080×1920, 30 fps, H.264 plus AAC, and loudness normalised to about −14 LUFS integrated. Measure this; do not just raise gain.
- Captions: 2–4 words per cue, word-timed. Keep them out of the bottom ~25% and the right ~12% (ANECDOTAL safe zones; check on a phone).
- No fade-in. The first frame is content.
- Composition: default to the creator's own stream or webcam in portrait. Stock-gameplay split-screen is an experiment arm, not the default (C6, reused/inauthentic policy risk).

## Before you claim a cut change works
1. Run it on one or more **real** sources (not fixtures).
2. Report candidates → proposals → aligned → selected → exported, plus the top hold reasons (guardrails §4).
3. Watch at least the first 3 s and the last 3 s of every exported clip yourself (extract frames or the audio waveform). If you cannot, say so plainly.
