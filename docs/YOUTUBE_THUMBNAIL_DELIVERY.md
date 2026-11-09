# YouTube Shorts thumbnail delivery

The approved cover is an explicit artifact, not a frame guessed during upload.
Nothing in the staging workflow authenticates, uploads or grants release approval.

## Bind an approved cover locally

```powershell
.venv\Scripts\python.exe -m tools.stage_thumbnail shorts/<batch>/portrait.mp4 shorts/<artwork>/thumbnail.jpg --sha256 <approved-sha256>
```

The helper validates and copies the **exact bytes** into the delivery video's
directory as `portrait_thumb.jpg`. It checks the supplied approval digest and
never overwrites a different cover. Repeating the identical binding is safe.
Automatic generation uses the same lock for installation and rechecks the
destination after rendering. If an owner stages a cover meanwhile, that cover
and its evidence are preserved; the discarded render installs neither artifact.
JPEG installation uses a same-filesystem hard link (atomic, no overwrite); a
filesystem without hard-link support reports a failure rather than falling back
to an overwrite or exposing a partial image.
Archive an unwanted existing cover explicitly before restaging, then redo the
hash-bound release review. A staged cover is not release or upload approval.

Required local format for this pipeline: upright RGB JPEG, 9:16, at least 640px
high, nonempty and no larger than the API's 50 MiB cap. A renamed PNG, corrupt
JPEG, landscape cover or EXIF-rotated image fails before authentication.

## Upload guarantees

`upload.upload_video()` still returns a video ID on complete delivery, or `None`
for a failed local preflight/video insert. After a video exists but its cover or
publication is incomplete it raises `DeliveryIncompleteError`, including `video_id` in both the
exception and message. The orchestrator records a failure, not a completed upload
or published event. The MP4 and reviewed metadata are never rewritten by upload.

1. Validate the canonical cover before authentication when thumbnails are enabled.
2. Enforce the existing owner release-review gate; thumbnail changes still
   invalidate any prior approval.
3. Hold a per-video exclusive lock. A crash's stale lock requires manual
   reconciliation; it is not automatically deleted by another worker.
4. Persist schema-v2 insert intent in `<stem>_delivery.json` **before** inserting a
   video. Intent binds normalized privacy, UTC publication time and notification
   preference, in addition to the artifact hashes and configured channel.
5. Insert **privately with no publication schedule**, regardless of the intended
   final visibility. Persist the returned video ID before any thumbnail work.
   Read back the configured channel and privately staged status before proceeding.
6. Wait for `processingDetails.processingStatus == "succeeded"`; processing,
   termination, failure, malformed responses and timeouts cannot pass as ready.
7. Call `thumbnails.set` with the canonical JPEG and explicit `image/jpeg` MIME.
   Validate that the response actually contains a thumbnail resource.
8. Poll the same video's `snippet.thumbnails.shortsAspectRatio` and fetch that
   returned HTTPS YouTube CDN image with a cache-busting query. Redirects and
   arbitrary hosts are refused; reads have socket and byte limits.
9. Compare its 9:16 pixels with the local approved cover after normalized resizing.
   Permit recompression, but reject mismatched artwork. URL equality alone cannot
   prove assignment: YouTube can serve a stale image at the same URL.
10. Persist verified thumbnail evidence, then recheck artifact hashes and owner
    release approval. A disabled thumbnail still requires processing success.
11. Read status before changing visibility. Preserve unrelated mutable status
    fields (embedding, license, audience and disclosure), and write only `status`.
    Persist `PUBLICATION_UNKNOWN` **before** a potentially public API update.
12. Apply the authorized final visibility or schedule, then read it back from
    YouTube. Recheck artifact hashes after the API round trip and mark `COMPLETE`
    only after the matching readback is durably recorded.
    Private deliveries normally need no visibility update. All SDK media handles
    are closed on failure.

A failed cover check leaves a newly inserted video private and unscheduled. A
failed/ambiguous **publication** update may already have made the video public;
the receipt says `PUBLICATION_UNKNOWN`, never promises that it is still private.
A retry first reads the same video's status to reconcile that outcome. It does
not re-upload the video or reapply its already verified thumbnail.

The image check is a conservative **perceptual** comparison, not byte identity or
human editorial approval. It uses normalized 180×320 RGB pixels, overall RMSE ≤12
and worst-of-12 tile RMSE ≤24 (0–255 channels). It may hold an unusually altered
YouTube rendition for manual review rather than silently accept it. It does not
prove placement on every YouTube feed, device or cached surface.

## Receipts and safe retries

The receipt binds media, metadata and cover paths/hashes plus configured channel.
It stores delivery evidence only; it is not a learning or analytics database.

| Status | Meaning | Next invocation |
|---|---|---|
| `VIDEO_UPLOAD_UNKNOWN` | Insert intent recorded, no trustworthy returned ID | Hold. Reconcile with YouTube; never blindly insert again. |
| `THUMBNAIL_PENDING` | Known video ID, processing/cover incomplete | Confirm privately staged state and resume cover work on that ID. |
| `PUBLICATION_PENDING` | Thumbnail done, final visibility not applied | Recheck release evidence, read current status, then apply authorized intent. |
| `PUBLICATION_UNKNOWN` | Publication write may have succeeded | Read back first; finish if it matches, otherwise resume only from confirmed private/unscheduled state. |
| `COMPLETE` | Thumbnail done and final visibility/schedule verified | Return recorded ID only for the same intent. No new upload. |

An explicit privacy, schedule or notification change is rejected before
authentication; it is neither silently ignored nor automatically applied. A
`scheduled` retry without an explicit time reuses the receipt's time rather than
reserving another slot. The orchestrator schedules only new deliveries. Equivalent
timezone representations normalize to the same UTC timestamp.

An expired schedule is held rather than sent as a past `publishAt` value (which
can publish immediately). If a previously attempted schedule has already executed,
a matching public readback after the intended time reconciles it without replay.

An artifact/channel mismatch, corrupt receipt, or legacy metadata containing a
video ID without a bound receipt is held for reconciliation. Do not delete a
receipt or clear the ID simply to retry. If receipt persistence fails after an
insert, the exception still includes the known ID; reconcile the existing video.
Schema-v1 upload receipts are held for manual reconciliation: they do not bind
publication intent, so safe automatic migration cannot infer the owner's choice.

The `COMPLETE` receipt proves the check at delivery time. An idempotent retry does
not re-check subsequent changes made manually in YouTube Studio.

The existing video transfer is still a single non-resumable `execute()` call.
This work does **not** implement its configured retry/deadline/resumability
options. Ambiguous insert failures are now held safely instead of replayed.

## Analytics metadata compatibility

The analytics importer supports both legacy metadata `youtube_video_id` values
and schema-v1/v2 receipts with a known video ID. A receipt must match the configured
channel, canonical sibling paths and SHA-256 of media, loaded metadata bytes, and
cover when present. Malformed, changed, unknown-upload or conflicting bindings
are ignored; invalid receipts cannot fall back to a conflicting legacy ID.
No reviewed metadata is rewritten to restore the join.

A receipt supplies only local context. `YouTubeShortsSource.fetch()` still requires
the exact channel Shorts shelf, verifies the API video's owner and joins real
YouTube Analytics. A pending receipt is not treated as a successful outcome.

## Bounds and tests

`config.yaml` supplies processing timeout (180s), verification timeout (90s), poll
interval (3s) and attempt cap (60). Polls use a monotonic deadline and attempt cap;
individual API requests also use the authenticated client's existing socket
timeout. A phase can exceed its nominal poll deadline by an in-flight request.

`tests/automation/test_youtube_thumbnail_delivery.py` uses real JPEGs with fake
Google/CDN endpoints. It covers failure-first preflight, processing order and
failures, stale CDN content, malformed responses, image recompression, partial
delivery, idempotency, unknown outcomes, receipt failure, locking and staging.
The orchestrator test ensures incomplete covers are not counted/published.
`tests/automation/test_delivery_publication.py` checks private-first staging,
durable-before-publication ordering, status preservation/readback, rejected intent
changes, lost update responses, failed receipt writes, expired schedules and
revoked release approval. Cover concurrency tests include real JPEGs and the
source-cover evidence writer. Importer tests cover legacy compatibility, hash/path
and channel rejection, and the independent Shorts-shelf gate.
The upload-enabled dry run uses actual local JPEG fixture bytes through the
production validator/pixel comparison; only the API/CDN boundaries are stubbed.

Sources checked 2026-11-04:
- https://developers.google.com/youtube/v3/docs/thumbnails/set
- https://developers.google.com/youtube/v3/docs/thumbnails
- https://developers.google.com/youtube/v3/docs/videos/update

No live upload is part of this validation. Channel eligibility/permissions and
actual YouTube delivery still need an explicitly authorized live check.

## Regression-fix evidence

The implementation plan preceded production edits: reproduce the four findings,
introduce private-first publication with a durable intent/state machine, restore
the hash-validated metadata join, serialize no-clobber cover installation, then
validate focused tests, the full suite, both dry runs and approved artifact hashes.

- `logs/delivery-safety-fixes-red.log`: 23 failures before implementation.
- `logs/delivery-safety-fixes-resume-red.log`: the scheduler retry interaction.
- `logs/delivery-safety-fixes-intent-red.log`: reject changed intent before slot reservation.
- `logs/delivery-safety-fixes-cover-evidence-red.log`: preserve an approved cover's evidence.
- `logs/delivery-safety-fixes-post-publication-red-recovered.log`: reject completion
  if reviewed artifacts change during the publication API round trip.
- `logs/delivery-safety-fixes-recovered-focused.log`: focused post-recovery checks.
- `logs/delivery-safety-fixes-final-pytest.log`: final full-suite validation.
- `logs/delivery-safety-fixes-final-dry-run.log` and
  `logs/delivery-safety-fixes-final-dry-upload.log`: isolated stubbed pipeline checks.
- `logs/delivery-safety-fixes-artifacts.json`: approved media/thumbnail hash evidence.

These are local tests and simulations, not evidence of a live upload or owner
release approval. No paid processing, live upload or commit is part of this fix.
