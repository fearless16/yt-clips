# Crop reassessment after creator rejection — 2026-11-02

**Superseded visual direction:** the creator subsequently supplied an ideal
portrait reference. [REFERENCE_TARGET.md](REFERENCE_TARGET.md) now governs.
The 4:5 experiment below is retained as evidence, not an accepted target.

Status: **COMPARISON STILLS + 12s TEST GENERATED; production settings and canonical render unchanged.**

The creator rejected Aurora's composition and requested more screen occupation,
deliberate thinking and Reddit research. This supersedes any implication that the
earlier static inspection established aesthetic acceptance. RESULTS.md remains a
historical record of technical checks, not evidence that the creator likes it.

## Actual image diagnosis

Inspected current full-size 8s and 21s export frames and retained full source frame
at113.6025s. The sharp foreground is approximately920x998 inside1080x1920:
**44.3% of the canvas area**, with998/1920=52.0% height. The rest is blur, not
additional useful scene coverage. The face is already close inside that rectangle.
The problem is not simply a small face: it is a cramped head inside a floating
near-square window, with excessive surrounding filler and a detached caption card.

The source itself is a close-up. Cropping cannot create missing torso/headroom.
Stretching it vertically is not an acceptable way to occupy the screen.

## Evidence inspected and its limits

Reddit direct page fetches returned no usable body in this client. Findings below
are from live search-indexed post/comment excerpts, not an exhaustive thread read,
viewer study, current competitor audit or verified channel performance benchmark.
No engagement statistics from Reddit are adopted as causal evidence.

1. r/videography, landscape + vertical workflow:
   https://www.reddit.com/r/videography/comments/1iv9pmo/ive_done_it_the_landscape_vertical_solution/
   Excerpts distinguish reframing for the destination from blindly fitting footage;
   one creator explicitly reports vertical crops becoming too close. Other comments
   describe recropping/splitting shots and using vertical guides.
2. r/videography, dual-format deliverables:
   https://www.reddit.com/r/videography/comments/rpuvke/how_do_you_deal_with_productions_that_require/
   Excerpts discuss composition compromises, destination-format guides, pan/scan,
   wider capture and separate capture where necessary. Not a universal crop ratio.
3. r/editors, vertical action/title safety:
   https://www.reddit.com/r/editors/comments/1vz6p9v/what_is_the_action_safe_and_title_safe_area_for/
   Discusses phone-side cropping and keeping eyelines within safe areas. These are
   community reports; do not turn them into a universal mobile-cropping percentage.
4. r/NewTubers, gaming format comparison (different niche; weak supporting evidence):
   https://www.reddit.com/r/NewTubers/comments/1solpl4/what_is_the_best_format_for_gaming_clips/
   A comment prefers cropping to the action over blur filler. This is an opinion,
   not proof that full bleed improves cricket-commentary retention. Reject the
   post's AI-generated performance ranking as audience evidence.
5. Official YouTube help, successfully fetched and read:
   https://support.google.com/youtube/answer/16215842?co=GENIE.Platform%3DAndroid&hl=en
   Its visual-guides section says controls/descriptions and some device edges may
   obscure overlays. Actual app/device preview matters. It supplies no universally
   preferred human-subject occupancy ratio or fixed safe pixel rectangle.

Practical recurring theme: compose for portrait and protect meaningful subject
coverage. **No defensible evidence here establishes a universally loved ratio.**

## Proposed direction for THIS close-up source

- Keep the delivery canvas9:16. Do not confuse delivery ratio with camera-panel ratio.
- First candidate: full-width **4:5 camera region,1080x1350**, starting at the top
  rather than floating with an empty upper margin. It occupies70.3% of the canvas,
  versus44.3% currently. This is our proposed starting point, NOT a Reddit statistic.
- Preserve source aspect ratio. Using1040 source pixels in height would require
  about832 source pixels in width and1.298x uniform scale. Review the full motion
  envelope before selecting x; a static center crop is not automatically valid.
- Captions should sit close to the picture/below the chin, not in a separately
  designed empty section. Retain the font/identity initially; reduce card padding
  if needed rather than launching another wholesale restyle. Caption placement
  must be checked against face movement AND the actual Shorts UI.
- Keep remaining blur restrained. Do not fill the remainder with invented footage,
  copied match footage, extra decorations or fabricated torso/background.
- Compare full-bleed9:16 as an alternative only if it preserves the face. With1040
  source height, true9:16 permits just585px source width. The inspected hair/head
  span is wider, so forcing full bleed risks the exact hair/ear loss already found.
- Do not promise both a much larger screen-filling image and a much smaller face
  from this close-up source: those aims conflict without more source coverage.

## Test before committing another render

Impact: potentially high for visual coherence; no measured retention gain.
Confidence: high in the current wasted-area diagnosis, medium in4:5 as a starting
composition, low in any claim of audience preference before testing.
Risk: tighter lateral crop, softened upscale and more visible source-overlay edges.
Difficulty: moderate; movement-aware framing must precede production changes.

Source-based comparison stills and a short motion test are now generated. The
motion test predates the new creator reference and is superseded as a target.
Reject any option clipping hair/ears/chin or obscuring speech
with captions. Do not overwrite the existing render until a better candidate has
been inspected. No creator approval has been given to the replacement candidates.

For a later authorized channel experiment, compare chose-to-view and retention
across comparable cricket Shorts; topic/timing confound crop effects. Do not upload
duplicate cuts or claim a controlled A/B outcome. Rollback: retain current artifact
and hashes until a replacement passes review. Publication still requires the
existing speech-fidelity and owner-approval gates.

## Source-based comparison — continuation

Evidence directory:
`shorts/aurora-pilot-20261102/review/crop-reassessment/`.

Inspected the 12-source-frame motion sheet, the full source frames at clip15s and
21s, all four rows of the 360px comparison, the full-size tracked15s/21s images,
and the full-bleed diagnostic. These are image inspections, not playback or a
phone/YouTube-app test.

The source inspection **invalidates a static832px-wide crop**. The head moves
across a wider source envelope: a crop protecting the leftward lean cannot also
protect the opposite head turn without reframing. Do not implement the earlier
4:5 dimensions with a fixed x just because the aspect-ratio math fits.

| Candidate | Sharp picture | Canvas area | Evaluation |
|---|---|---|---|
| Creator-rejected baseline | ~920x998, inset/y210 | 44.3% | Reference only; not accepted |
| Stable full-width panel | 1080x1170 at0,0; source450/0/960/1040 | 60.9% | Removes floating margins with less enlargement; more lower blur remains |
| Manually reframed4:5 | 1080x1350 at0,0; source832x1040 | 70.3% | More useful screen occupation; needs movement review and has a tight side margin at15s |
| Full-bleed9:16 diagnostic | source594x1056, uniform resize | 100% | Reject: lateral hair/ear coverage is lost in the inspected frames |

- `crop-comparison-360w.jpg`: columns are rejected / stable full-width / tracked4:5;
  rows are clip8s,15s,21s,29.3825s (source92.22,99.22,105.22,113.6025s).
- `full-bleed-rejection.jpg`: actual-source diagnostic, not a recommended crop.
- `preview-manifest.json`: source identity, per-image crop/caption values, sparse
  manual tracking anchors and actual FFmpeg commands. The anchors are visual
  judgements, **not face-detector output or a certified whole-clip movement bound**.
- Both candidates keep uniform scaling, the same68px bundled Aurora typography
  and the existing caption text/timing. Captions start24px below the sharp panel;
  they do not obscure the mouth in these four inspected stills. No speech correction
  or improved alignment is claimed. A small baked source badge is explicitly
  blurred before reframing; its patch is visible near the top-left during the lean.
- These comparisons show a real layout tradeoff, not an audience-performance win.
  The larger4:5 candidate also makes the face larger. Preserve both alternatives
  for creator review rather than silently declaring a preferred look accepted.

No production code/configuration changed during this continuation. Face OS and
Gaming Clip Studio were not changed. No full replacement render, upload, commit,
push or scheduling was performed.

The 12s test `tracked-4x5-motion-12s.mp4` covers source96.22–108.22s,
1080x1920/30fps. FFprobe readback:360 video frames,12.000s video and audio,
both starting0. Full A/V decode passed. Eight extracted frames were inspected;
no listening, real-phone review or creator approval occurred. A first interrupted
attempt had no valid MP4 header; only that invocation-owned invalid file was
removed after probing it. The successful render and its command/hash are retained.
