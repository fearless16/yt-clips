---
name: shorts-publish-path
description: Rules for getting exported yt-clips Shorts reviewed, scheduled and uploaded to YouTube via Data API v3 without deadlocks or silent failures. Use when touching release_review.py, upload.py, youtube_delivery.py, scheduler.py, thumbnail delivery, or youtube/upload_schedule/release_review config. Pairs with shorts-automation-guardrails.
---

# Getting Shorts published

Load `shorts-automation-guardrails` first. Publishing defects are listed in its register: C1, C5, C7, C8 and C9.

## Hard facts (keep the evidence labels when you cite them)
- A video is a Short if it is vertical or square and ≤ 180 s. OFFICIAL. `#shorts` is not documented as required, so treat it as UNVERIFIED and keep it harmless (one tag, no spam).
- A Short over 60 s with an active Content ID claim is blocked worldwide. OFFICIAL. Keep broadcast-audio clips under 60 s.
- The quota cost of `videos.insert`, and whether unverified API projects get forced to private, are **CONFLICTING**. Read the real numbers from the Google Cloud console and from the `privacyStatus` readback after upload. Do not assume either way.
- Do not quote view counts as success. Since 2025-03-31 replays count as views. Use engaged views and "shown in feed" from `shorts_intelligence`.

## Delivery invariants (keep these, they are good)
- Write the receipt **before** `videos.insert`, so an ambiguous timeout never double-uploads (`upload.py:481-485`).
- Upload as private, then read back and flip to public or scheduled (`youtube_delivery.finish_publication`).
- Read before you write, and never reset unrelated status fields.

## Rules that stop the pipeline from strangling itself
1. **Every HOLD needs a scripted exit.** If a receipt or review can go stale (hash change, expired `publish_at`, legacy receipt), there must be one command that archives and regenerates it (A4). Expired schedules are re-slotted automatically while the video is still private.
2. **The human review step takes one command and under 60 s per clip** (implemented): `python -m automation.release_review review shorts/ --open`. Every check is recomputed from disk by `release_review._build`. Never add a hand-typed evidence field. `release_review.mode: auto` releases clean clips. `heuristic_fallback` clips always need the owner (C1).
3. **Isolate per clip.** A QC, SEO or cover failure on one clip never blocks its siblings (C5). SEO has a per-clip budget and no batch deadline. A failure degrades to `seo_method: template`, which needs the owner (C4).
4. **The cover is best-effort** (implemented). A failed custom thumbnail is recorded as `thumbnail.status=FAILED` and never blocks publication. Processing failure still holds (C8).
5. **Carry the real error** (implemented). `upload_video` raises `UploadRejected(reason, stop_batch)` and never returns a bare None. The loop stops on `stop_batch` (quota/auth/429). A 4xx on insert clears the pre-insert receipt (C9).
6. **The schedule knows what is already scheduled.** Slot assignment must read existing receipts and scheduled publish times and keep at least `interval_hours` spacing across runs. No hard-coded 19:00 for lone clips (C7).
7. **Cadence is an experiment, not a belief.** Begin at no more than 2–3 per day. Change cadence only with `shorts_intelligence` evidence. Day-of-week rules from under about 500 Shorts are tie-breakers only (A9).
8. **No delete-and-reupload loops** to "fix" a low performer. Log it as a learning sample instead (ANECDOTAL, but cheap to avoid).

## Pre-upload checklist (machine-checked, per clip)
- [ ] 1080×1920 (or square), ≤ 180 s, and ≤ 60 s if broadcast audio is present
- [ ] `_metadata.json` is valid: title ≤ 100 chars, description ≤ 5000 bytes, tags ≤ 500 chars
- [ ] `categoryId` is assignable in `IN`. Set `defaultLanguage` and `defaultAudioLanguage` to `hi` (currently not set)
- [ ] `selfDeclaredMadeForKids` and `containsSyntheticMedia` are set explicitly
- [ ] The release review is READY, or the clip is in a configured auto-trust tier
- [ ] The schedule slot is in the future and does not collide with existing receipts

## Verify after delivery
Read back `status.privacyStatus`, `status.publishAt` and `processingDetails`. Count `*_delivery.json` with `status: COMPLETE`. Report that number. "Upload returned an ID" alone does not count as success.
