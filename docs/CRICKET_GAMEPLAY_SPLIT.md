# Cricket gameplay split — implementation and delivery plan

Date: 2026-11-07. Branch: `plan/cricket-stock-split-variant`.
Supersedes the **real-sport footage interpretation** in
`CRICKET_STOCK_SPLIT_VARIANT_PLAN.md`; the user's intended footage is video-game
gameplay, such as RC24 or EA Cricket, **not actual match/stock sport footage**.

## Current production integration (2026-11-08)

**Default profile is now `cricket-gameplay-stadium-v1`.** The CLI, normal/quick
launcher and `export_all()` select it from `config.yaml`; it is no longer a
private-temp-only renderer. `portrait-caption-v1` and `cricket-gameplay-split-v1`
remain explicitly selectable for compatibility. Overrides are forwarded through
the orchestrator and respected by the filter builder without changing global config.

Production owns the approved RGBA artwork in
`assets/gameplay/stadium-reference-v1/overlay.png` (explicit Git-ignore exception),
geometry/font fingerprints in `automation/stadium_layout.py`, the compositor in
`automation/stadium_export.py` and the 96px caption ink in
`automation/stadium_caption.py`. No production module imports private scripts or
reads design assets from `temp/` or `shorts/`. Installed licensed Arial Black is
hash checked, not redistributed or silently substituted.

- Equal 992×558 16:9 panels, full commentary frame, same centered plaque/brand.
- Asset-level `crop: [x,y,width,height]` is required for this profile, even-pixel,
  exact 16:9 and inside the actual recording. The accepted EA07 recording uses
  `[240,128,1440,810]`; **do not apply that crop blindly to another recording**.
- Stable source/range-seeded selection prefers usable approved EA07 assets, then
  other explicitly approved games if EA07 cannot cover the whole clip. No matching
  to commentary events, downloading, looping or boundary changes. A retry selects
  the same background; different source moments produce varied seeded selections.
- Captions stay 96px. Wider phrases split only at real source-word boundaries;
  an oversized indivisible word holds rather than shrinking/dropping/faking timing.
- Source audio only, 1x, complete endings, no hook/retime/fade/post-render enhancement.
- Publisher disclaimer appears in a **new production-only bottom 72px legal strip**
  below GAMEPLAY VISUAL from frame one. This differs from the private final pair;
  footer phone/UI placement needs finished-file review. GSTG/other recording credit,
  source URL and crop modifications are retained in metadata and public description.
- Initial SEO, direct regeneration and failed-SEO retries all preserve attribution
  through one idempotent helper. Oversized credits hold instead of truncating them.
- Gameplay, crop, artwork, font, source, permissions, metadata, cover and QC remain
  hash-bound to review. Removing an edit sidecar cannot hide gameplay obligations.
- Final-layout private packages cannot be auto-discovered as resumed production
  batches. Post-render enhancement is rejected before it can alter approved artwork
  or make the gameplay/QC hashes stale. Unknown profiles fail rather than falling back.

**Delivery update (2026-11-09):** the owner enabled the configured
`input/gameplay/library-clean-v2.json` as `STOCK` / `AVAILABLE`. Normal runs can now
prepare and sync seeded stock packages without the earlier draft-only preflight
abort. Provenance, attribution and unresolved rights records remain intact; exact-file
owner release is still required for YouTube scheduling. Legacy private manifests
are unchanged. See `docs/STOCK_GAMEPLAY_RUNTIME.md` for current behavior and history.

Verification and current logs: `docs/STADIUM_PIPELINE_INTEGRATION.md`.

### Private revision/finalization history (predates production promotion)

**Final layout accepted:** owner confirmed "yes, finalize this" after v7.
Canonical final-layout package: `shorts/final-ea07-gameplay-pair/`.
Both FINAL.mp4 files are byte-identical to reviewed v7, freshly hash-verified
against all three evidence surfaces and freshly full-decode checked. The package
records appearance approval in `finalization.json`, preserves original evidence
and credits, and retains `publication_authorized: false` until existing rights,
attribution and editorial/release checks clear. No upload or production-default
change was made; existing schedule remains unchanged.

Latest framing request: `ea07-fill-pair-v7/` excludes the right-side GSTG gameplay
logo and pillarboxes using source crop [240,128,1440,810], scaled to the unchanged
992×558 lower panel. Both are exact 16:9; no stretch or padding. Layout, captions,
commentary/audio and selected v6 gameplay ranges remain unchanged. Source credit,
URL and crop modification remain in metadata/sidecars/CREDITS.txt; public-facing
attribution and existing rights/release gates are still unresolved. No upload or
production-default change. See the v7 README for exact-file verification.

Owner accepted the v5 layout and prefers **EA Cricket 07**, with varied arbitrary
gameplay for aesthetics rather than commentary synchronization. `ea07-pair-v6/`
retains the accepted layout/caption pixels/cuts and verified-identical v5 AAC audio,
replacing only the gameplay panel. Distinct persisted random candidates were
sample-screened; the menu-containing first candidate was rejected. Full EA07
source framing retains pillarboxing and recording watermark. No production
default, randomizer, approved manifest, rights clearance or upload was enabled.
Design acceptance does not authorize exact-file release. See
`shorts/private-gameplay-review-20261107/ea07-pair-v6/README.md`.

Owner-supplied image revision: `reference-pair-v5/` supersedes v4 visually.
It follows the provided stadium/floodlight background, gold wordmark, cyan/gold
angular borders and centered block-italic white/cyan caption plaque. Branding
reads CRICKET WITH PRAJJWAL only: no 2.0 badge or @ handle. Equal 992×558 16:9
planes have identical angular masks. Same original cuts/audio/cue text/timing;
no generated face or gameplay. Previous drafts and release gates remain intact.
See `shorts/private-gameplay-review-20261107/reference-pair-v5/README.md`.

Latest request (2026-11-08): new channel masthead in the empty header, centered
caption text using the caption master, and better use of negative space.
`shorts/private-gameplay-review-20261107/branded-pair-v4/` contains that scoped
revision and its research notes. The actual primary branch is `main`, not
`master`; its bold italic 72px uppercase Oxanium white/cyan neon design is reused
with measured horizontal centering. Caption wording/timing and equal 1056×594
16:9 panels remain unchanged. V3 is preserved, not overwritten. Production
defaults, rights gates and upload authorization are unchanged.

The user rejected layout v2 and clarified the intended design: **matching 16:9
landscape video panels, with the current captions unchanged between them**. The
corrected drafts are in `shorts/private-gameplay-review-20261107/landscape-pair-v3/`.
Both panels are exactly 1056×594; commentary retains the full original 16:9 frame,
gameplay gets an aspect-exact 1664×936 source crop, and the delivered v2 caption
PNGs are reused directly without rewriting or redesign. This remains a private,
scoped renderer, not a production-profile change or upload approval.

Historical (rejected) v2 revision:

The two real-source local drafts have a visually inspected `layout-v2/` revision
under `shorts/private-gameplay-review-20261107/`. Original drafts are preserved.
This is a scoped private renderer, **not a change to the production/default profile**.
Camera pillarboxing and the large blank gameplay gap are removed; gameplay is
about 23% taller; existing captions are reflowed in heavy type with measured bounds;
credits and unresolved-clearance labels remain. Encoded opening/final frames and
all caption midpoints were inspected, including a 360px-wide comparison.
Subjective layout-only scores: batting role 8.2/10; team game 7.8/10. Caption
transcription errors, source-baked reaction effect, listening, actual-phone review,
publisher rights and exact-file owner release remain unresolved. See the revision's
README and hash-bound `visual-review.json`; neither score is audience-outcome evidence.

## Authorized outcome and present blocker

- Use cached original commentary `https://youtu.be/H8OjNZ8XLbI`.
- Target at least **two distinct complete source moments**, preserving hooks,
  setup and punchlines. A different layout of the same moment does not count twice.
- Upload to `@cricketwithprajjwal2.0` using the **existing schedule**, after the
  existing exact-file release review. No Instagram publication for this pilot.
- User confirmed **no gameplay recording is available yet**. Implement and test
  now; real gameplay exports/uploads remain blocked on a cleared recording.
  Synthetic test footage is not a production substitute.

## Rights findings (checked November 7)

Gameplay is copyrighted; permission to publish is not public-domain status or a
guarantee against future Content ID claims. Two layers of rights must be checked:
the publisher/game assets and the person who recorded the footage. A creator's
"no copyright" title or CC label does not supersede the publisher's restrictions.

| Game | Official finding | Pilot decision |
| --- | --- | --- |
| EA Cricket 07 | EA's content policy explicitly excepts passive ads and YouTube/Twitch partner monetization from its commercial-use prohibition; other rules and third-party rights remain | Prefer an **own, legally acquired, unmodded recording**, after checking the actual policy and visible licensed content. The policy is not an unrestricted license |
| Real Cricket 24 | November 1, 2025 terms permit gameplay creation/free viewing but require prior written authorization for commercial use; prominent exact disclaimer at the beginning | **Written publisher authorization required** for monetized/background use; also require recording permission. Do not infer commercial clearance from popular RC24 channels |
| Other cricket games | No blanket permission established | Require written publisher authorization plus recording permission; do not apply EA terms to Big Ant/Nacon/other publishers |

All game audio is omitted, including licensed music and in-game commentary. This
does not by itself clear third-party visual assets, mods, likenesses or trademarks.
EA also prohibits using its footage to promote third-party brands/products/services.
Its ordinary gameplay monetization exception is **not automatic clearance for this
branded background format**. Approval must explicitly cover split-screen background
use and the no-third-party-promotion restriction; obtain written permission if the
intended use is outside or unclear under the general policy. Both permission records
require `split_screen_background: true`; EA's policy route additionally requires
`no_third_party_promotion: true`, supported by the review, not inferred by software.
EA's non-affiliation statement is added on screen and to the description:
"This project is not endorsed by or affiliated with EA or its licensors."
No mods/patch packs in v1. No game installation, purchase or random YouTube download
is authorized/necessary for the compositor implementation.

RC24 mandatory on-screen disclaimer:

> This video is subject to the Game’s Terms of Service and was not endorsed or sponsored by KRAFTON, Inc.

The compositor displays it from frame one when the approved asset is RC24. The
label/disclaimer cannot cure absent permission. Required attribution/disclaimers
are retained in the gameplay sidecar, appended to generated descriptions before
review, and checked again at release; missing credits hold publication.

Official sources:
- https://help.ea.com/en/articles/security-and-rules/ea-content-policy/
  (official search-index text verified; direct fetch timed out/failed in this session;
  save/review the actual policy at acquisition, not this summary alone)
- https://www.nautilusmobile.com/tos-rc24 (full terms fetched)
- https://realcricket.krafton.com/clause/terms_of_service (official successor terms
  also indexed with commercial-authorization restriction)
- https://support.google.com/youtube/answer/138161?hl=en
- https://support.google.com/youtube/answer/1311392?hl=en

Community discussion was checked, not treated as permission:
- https://www.reddit.com/r/NewTubers/comments/mfkayo/is_it_legal_to_use_a_portion_of_others_gameplay/
- https://www.reddit.com/r/NewTubers/comments/1eqx04a/how_can_gaming_channels_monetize_these_type_of/

Claims that overlays/short excerpts automatically establish fair use conflict with
the need for a fact-specific review. Use documented permissions for this pilot.
Copyright permission and YouTube monetization eligibility are separate decisions;
original commentary remains the value, not an endless generic gameplay loop.

## Legacy gameplay profile (still supported)

`cricket-gameplay-split-v1` is explicitly selectable; the current default is the
approved stadium profile above. This section describes the preserved older layout.

```text
1080 × 1920
0–840       Original host/source, valid camera ROI or full-source aspect fit
840–1080    240px futuristic HUD, word-timed Hinglish captions
1080–1920   Muted cricket video-game action, aspect fit, illustrative label
```

- Full original timeline at **1x**, source audio input 0 only. No end fades,
  fabricated words, reordered speech, internal silence trimming or speed-up.
- Fit, never stretch, the original/gameplay panels. Unsafe face ROIs fall back to
  full-source fit; phone framing still needs visual review. Source face metadata
  is not a claim that every frame was examined.
- `gameplay-hud-v1`: bundled local Oxanium, 76px, at most two measured lines;
  navy glass, cyan/violet edges, no all-caps proper-name rewriting. A 320px PNG
  transport contains visible pixels only in its central 240px. Oversized phrases
  hold for shorter aligned cues rather than tiny type or dropped words.
- Stable source/range-based selection of **one approved continuous gameplay range**
  long enough for the complete Short. No automatic loops, cuts mid-delivery,
  downloads, frozen ends or fallback to actual cricket. An insufficient asset holds.
- Validate hashes, both permissions, ranges and actual stream durations. Full-file
  decode/loudness/A–V QC runs before returning the gameplay export.
- Save exact selected gameplay range, original hash, permissions, label/disclaimer,
  geometry and output hash in `CLIP_gameplay.json`; carry it into edit/review evidence.
- Release review adds gameplay rights/attribution and label/framing checks; it
  binds local recording, manifest, publisher evidence and recording permission.
  A withdrawn asset/changed manifest invalidates release readiness.

Implementation:
- `automation/cricket_gameplay.py`: manifest, rights checks, deterministic ranges,
  safe source crop, distinct-source-moment count.
- `automation/gameplay_caption.py`: center HUD and browser-measured fit.
- `automation/gameplay_export.py`: actual FFmpeg composition and technical QC.
- `automation/cli.py`, `automation/orchestrator.py`, `export.py`: isolated opt-in
  routing, early rights hold before paid processing, unchanged scheduling route.
- `automation/release_review.py`: exact gameplay evidence and extra human checks.
- `tools/gameplay_library.py`: offline inspection/hashing, never auto-approval.

## Acquisition → two clips → scheduled upload

1. Supply a clean 2–5 minute landscape capture from a legitimately obtained game
   (longer is helpful, not mandatory). Disable game music; avoid menus, watermark
   compilations, mods and borrowed broadcast inserts. We drop all game audio anyway.
2. Save original recording and permission documents under `input/gameplay/` (ignored
   by Git). Copy `assets/gameplay/library.example.json` there as `library.json`.
   Paths resolve relative to the manifest. The example is deliberately **REVIEW**
   with false permissions; it is not a usable rights approval.
3. Review the actual publisher terms and the recording provenance, clear visible
   third-party assets, fill SHA-256 values and attribution, approve exact continuous
   ranges, and record reviewer/date. Set APPROVED/true only after that review.
   For RC24, use `publisher_permission.kind: written_authorization` and retain the
   written grant covering monetized edited background gameplay.
4. Preflight locally:

   ```powershell
   .venv\Scripts\python.exe -m tools.gameplay_library --hash input/gameplay/ea-cricket-07.mp4
   .venv\Scripts\python.exe -m tools.gameplay_library input/gameplay/library.json --seconds 38
   ```

5. Curate and render the cached source without publication:

   ```powershell
   .venv\Scripts\python.exe -m automation.cli https://youtu.be/H8OjNZ8XLbI --skip-download --skip-transcribe --skip-instagram --composition cricket-gameplay-stadium-v1 --gameplay-manifest input/gameplay/library.json
   ```

   Existing high-thinking discovery, ASR preflight, fresh final review and reviewed
   reserve promotion remain in force. Two moments are a target, not permission to
   relabel failed alignment/completeness as PASS. If fewer than two survive, retain
   successful siblings and report the shortfall. No unrelated dirty work is reset.
6. Review actual opening, full setup and final payoff with sound; compare source,
   captions and metadata. Add mandatory asset credits to each reviewed description.
   Approve exact artifacts through `automation.release_review`; do not fabricate
   phone/listening/owner attestations. Layout approval is not release approval.
7. Resume the **explicit batch** using the established upload/scheduling path:

   ```powershell
   .venv\Scripts\python.exe -m automation.cli https://youtu.be/H8OjNZ8XLbI --skip-download --skip-transcribe --skip-highlight --skip-export --skip-seo --skip-enhancement --skip-instagram --export-dir shorts/EXACT_BATCH --upload --schedule
   ```

   Preserve reviewed covers and metadata. Existing resumable upload receipts,
   channel checks and processing/thumbnail/publication verification apply. Report
   actual YouTube IDs and scheduled times; queue acceptance alone is not delivery.

## Validation and honest scope

Automated fixtures use synthetic colors/audio, never real footage with fabricated
permissions. Tests cover rights layers, modified/restricted/withdrawn assets, hash
changes, deterministic ranges, duration shortage, source ROI safety, real Chromium
caption raster bounds, FFmpeg panel pixels, exact source-audio frequency (not game
audio), full final audio energy, full decode and duplicate-variant counts.

Required full tests and dry run are run before any code commit. No commit/push was
requested. Real source listening, phone approval, recording acquisition and actual
YouTube upload cannot be certified by synthetic tests. The earlier broad plan's
three-source growth pilot, full API wall-deadline/token-budget redesign and controlled
analytics experiment remain separate work; this profile does not claim those are
implemented or promise a two-clip success before the real pilot completes.

### Validation completed in this session

- Required full suite: **1,344 passed, 4 expected failures**, 50.23s.
  Command: `.venv\Scripts\python.exe -m pytest tests/ -v --ignore=tests/test_face_detect.py`.
- Required standalone dry run: **OVERALL PASS**, 3 stubbed portrait exports,
  zero failures, zero uploads. This proves regression coverage, not real gameplay
  or source editorial success.
- Automation plus canonical learner suites: **99 passed**.
- New gameplay suite: **26 tests**, included in the full suite. Its two-moment
  batch uses real Chromium/FFmpeg/encoded QC with **synthetic** media and permission
  fixtures. It is not a rights-cleared production batch.
- Inspected the encoded synthetic frame: top/bottom panels and center HUD visible.
  Audio-frequency/end-energy assertions are automated checks, not human listening.
- Ran the gameplay CLI against the requested cached source **without a manifest**:
  expected early hold, **0 exports, 0 uploads, no editorial/download stage run**.
- Example manifest correctly holds as unreviewed; it never becomes usable by merely
  copying it. Existing source/caption/editorial artifacts were not overwritten.

### Acquisition follow-up — November 7

The user subsequently authorized us to find/acquire gameplay rather than wait
for their recording. Two CC-attribution-labelled recordings were downloaded:

- `input/gameplay/review/i1y1VpOsRaQ/source.mp4`: Cricket 22 nets practice,
  193.883333s, 1080p60, video only.
- `input/gameplay/review/K3jEw-oBPb4/source.mp4`: EA Cricket 07 match gameplay,
  524.766667s, 1080p30, video only.

Both passed full-file FFmpeg decode and sampled contact-sheet inspection.
Saved metadata records actual uploader licence labels; hashes, proposed credits,
visual limitations and unresolved rights are in `input/gameplay/review/README.md`.
A third Cricket 26 section timed out and is NOT counted as acquired. A publisher
permission request is drafted but not sent.

**Delivery state: FOOTAGE_ACQUIRED_FOR_REVIEW; AWAITING_PUBLISHER_CLEARANCE.**
No production Shorts created/uploaded, paid model calls, purchases, commit or push.
Do not treat CC uploader labels as publisher clearance or mark assets APPROVED.
Next obtain/review permission covering this background format and third-party
assets, review provenance and continuous ranges, then curate/render two distinct
drafts, obtain exact-file release review, and use the existing upload schedule.

### Metadata/branding follow-up — private review, not rights clearance

Removing technical tags or visible logos does not remove copyright or prevent
claims. Original recordings, licence evidence and provenance are preserved. The
gameplay has a technical creation date but no embedded copyright tag to remove.
Local review exports omit inherited technical tags and retain explicit game and
recording credits, an unresolved-rights notice and on-screen PRIVATE REVIEW labels.
A bottom-HUD crop is an aesthetic change only; it is not a brand-free guarantee.

Two distinct real-source private drafts have passed fresh native Pro high-thinking
text-completeness review and encoded delivery QC:

- `shorts/private-gameplay-review-20261107/01-define-the-batting-role-REVIEW-ONLY.mp4`
  — original source 4740.280–4750.880 seconds (10.60s).
- `shorts/private-gameplay-review-20261107/02-team-game-not-one-player-REVIEW-ONLY.mp4`
  — original source 4689.599–4702.239 seconds (12.64s).
- Source ranges do not overlap; these are different thoughts, not visual variants
  of the same source moment. Both are 1080×1920 at 30fps.
- Layout: camera-only host crop above, futuristic word-timed center captions,
  muted Cricket 22 gameplay below. Original commentary at 1x, without end fades.
- Source-matched ASR evidence was used; caches from other source fingerprints
  were excluded. The Rohit-opener alternative was rejected as garbled, not
  relabelled PASS. New ASR attempts and a shorter-chunk diagnostic are retained.
- Captions still contain explicit lexical uncertainties. Text review and
  technical QC are NOT source listening, caption-fidelity, phone or release approval.
- Actual encoded stills were inspected: host/caption/gameplay ordering and readable
  caption placement are visible, and the bottom game-HUD/logo strip is excluded.
  Other in-game branding may remain. This is sampled-frame review, not playback.
- The batting-role draft has an unresolved ASR `4-6` / four-sixes ambiguity, and
  both drafts need natural-spelling/name corrections checked against the audio.
  Caption uncertainties are retained in the per-file caption-processing evidence.

These private previews use no production APPROVED manifest and create no release
approval; publisher clearance and all production upload gates remain unchanged.
The initial team draft was re-rendered for the corrected HUD crop; the earlier
file is preserved under the scoped `temp/gameplay-review-20261107/` folder.
Current MP4 hashes match their private-review and full-decode/loudness/A–V QC
sidecars. Inherited creation-time tags are absent, but required attribution and
an explicit unresolved-copyright notice are deliberately retained. No upload,
commit or push occurred. Local one-off scripts/evidence are scoped to that temp
folder; production rights gates were not relaxed.

Follow-up validation: **1,344 passed, 4 expected failures**, 51.73s; standalone
mandatory dry run **OVERALL PASS**, zero uploads. Final logs are
`temp/gameplay-review-20261107/final-pytest.log` and `final-dry-run.log`.
