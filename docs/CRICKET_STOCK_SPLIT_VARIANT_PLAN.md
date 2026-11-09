# Cricket stock split-screen variant — research and implementation plan

> **Superseded footage interpretation (2026-11-07):** the user clarified that
> “gameplay” means cricket **video games**, not real sport stock footage, and
> authorized implementation plus eventual YouTube upload. See
> [CRICKET_GAMEPLAY_SPLIT.md](CRICKET_GAMEPLAY_SPLIT.md) for the corrected plan,
> implemented opt-in profile, rights findings and remaining delivery blockers.
> The following is the historical planning-only proposal, not current scope.

Date: 2026-11-07. Status: **PLAN ONLY; not implemented or release-approved.**
Branch: `plan/cricket-stock-split-variant`, created from `main` at `00af5ae`.
Existing tracked/untracked user work was carried across unchanged. Nothing staged,
committed, rendered, downloaded, sent to a paid model, or uploaded for this task.

## 1. Product decisions

- Add an opt-in `cricket-stock-split-v1` composition; retain the owner-approved
  `portrait-caption-v1` unchanged as the control and rollback route.
- Top: original host/source video. Center: accurate, word-timed futuristic captions.
  Bottom: licensed **real cricket sport footage**, not broadcast highlights.
  This interprets “cricket game clip” as stock sport footage. Video-game capture
  would need a separate publisher/monetization-rights review; it is not included.
- Two clips means **two distinct source moments**, each with its own hook, setup,
  and payoff. Two compositions of one moment do not satisfy the minimum.
- A substantive source containing at least two viable moments must yield at least
  two technically verified local drafts before the batch is successful. Aim for
  two primaries plus two reviewed reserves; allow a third strong primary.
- No arbitrary source can honestly guarantee two good moments. If bounded recovery
  cannot find them, keep successful drafts, report `PARTIAL`/`REVIEW_REQUIRED` with
  the shortfall and reasons, and require editorial intervention. Never duplicate,
  invent speech, weaken hard gates, or describe one export as a completed batch.
- Original speech remains at natural speed, in source order. The hook and punchline
  are protected editorial ranges, not expendable material to meet a time target.
- This proposal does not authorize publication, replacing the current default,
  paid asset purchases, or disabling the existing owner-release review.

## 2. Current system: evidence and gaps

| Surface | Observed state | Consequence |
| --- | --- | --- |
| `config.yaml` | Portrait composition, glass captions, 10–38s selection limits; editorial target 2, maximum 3 | New design needs its own profile, not a global appearance replacement |
| Editorial route | Native `deepseek/deepseek-v4-pro`, high thinking, 32,768 output-token cap, 300s SDK timeout | Research must concern this API, not local GGUF throughput |
| `automation/editorial_planner.py` | Whole-transcript sections, exact cue citations, word preflight, final review, reserves, one discovery recovery round | Reuse this architecture; diagnose where viable moments are lost |
| Budgets | Maximum 3 sections, 8 provider calls, 8 aligned candidates; retries share the call count | Several sections plus repairs/reviews can consume the budget before useful recovery |
| `automation/orchestrator.py:153–162` | Minimum-output check counts exported files | Variant-aware success must count distinct source moments, not render count |
| `automation/clip_selection/hook_auditor.py` | Keyword/punctuation heuristics; `clip_path` is not decoded | Existing hook score is not proof that encoded opening/audio/punchline survived |
| Orchestrator hook audit | Runs after export, emits telemetry, exceptions only warn; may fall back to description text | Add source-word/encoded-media verification; generated description is not speech evidence |
| Export transitions | Configured 0.5s picture fade and 0.4s audio fade at end | These can obscure the final punch if speech reaches the cut; protect the payoff |

Recorded source failures:

- `docs/TRANSCRIPT_FIRST_CURATION.md` records November 4 run `2b40f2d0` producing
  one draft, with unresolved caption and metadata findings. A visual approval did
  not approve those words or publication.
- Newer November 6 evidence in `logs/pipeline.log` and
  `highlights/video.curation.json`: 81.8-minute source, 53 candidates, 11 proposal
  holds, one review candidate, **zero selected**, target shortfall 2, six calls.
  Holds include missing/invented phrase anchors, cricket-relevance rejection,
  low-confidence/malformed ASR and boundary-repair limits. These are recorded
  symptoms, not proof that every rejection was wrong.
- Triage those candidates against source audio and original cues. Do not “fix”
  them by changing all completeness flags to true or making relevance unconditional.
- Live/test records coexist in `pipeline.log`; zero-duration mocked calls are
  not performance evidence. The November 6 call sequence was checked in context.

## 3. Footage research and rights policy

“Royalty-free” is a license model, **not copyright-free or claim-proof**. Cropping,
muting, mirroring, adding captions or using a few seconds does not grant rights.
Use permission for commercial editing and distribution, with asset-level evidence.

### Preferred sourcing order

| Source | Finding as checked on November 7 | Decision |
| --- | --- | --- |
| Pexels | Free commercial use, editing allowed, attribution not required; third-party/person/brand restrictions remain | Primary free library; manually approve each asset |
| Pixabay | Indexed license summary permits free use/adaptation, with prohibited uses and third-party-rights caveats | Secondary candidate source; direct license fetch was blocked with HTTP 403, so recheck full terms before acquisition |
| Mixkit | Has both Free and Restricted video licenses | Never infer rights from the category page |
| Self-shot cricket | Owner-created net practice/local cricket with suitable permissions | Strong long-term rights provenance and distinctive footage |
| Paid stock | Potential alternative if free action coverage is inadequate | Optional later decision; no purchase approved |

### Concrete candidate pages (not cleared assets)

1. [Pexels — People Playing Cricket, Altaf Shah, asset 12460907](https://www.pexels.com/video/people-playing-cricket-12460907/)
   Page reports 1080×1920, 60fps, 12 seconds, Free license. Actual cricket action;
   portrait-to-bottom-panel reframing needs playback review.
2. [Pexels — Women's cricket practice on sunny day, LayG Traveller, asset 40663292](https://www.pexels.com/video/women-s-cricket-practice-on-sunny-day-40663292/)
   Page reports 3840×2160, 25fps, 5 seconds, Free license. Better landscape geometry,
   but short and repetitive if used alone. Do not imply it depicts the discussed match.
3. [Pexels — People on Sport Pitch, Altaf Shah, asset 12478623](https://www.pexels.com/video/people-on-sport-pitch-12478623/)
   Page reports 1080×1920, 60fps, 10 seconds. Aerial/atmosphere candidate, not a
   substitute for checking that the sport/action is suitable.
4. [Pixabay — Stadium, British, Cricket, Somerset, asset 189845](https://pixabay.com/videos/stadium-british-cricket-somerset-189845/)
   Indexed page reports 3840×2160, 30fps, Pixabay Content License. Stadium B-roll,
   not verified batting/bowling coverage; secondary pending direct rights review.
5. **Reject for free commercial use:**
   [Mixkit — Cricket court in a stadium, asset 12233](https://mixkit.co/free-stock-video/cricket-court-in-a-stadium-12233/).
   Its category page advertises free cricket footage, but the **item page explicitly
   limits the free 720p download to personal use under the Restricted License**.
   The commercial version is paid through Envato. This is a regression-test case.

The pages/technical descriptions were researched; no asset was downloaded or
fully watched. Uploader authority, releases, full action continuity, identifiable
minors, trademarks and any embedded broadcast material remain acquisition checks.
Search-result counts/tags are noisy and are not a count of usable cricket clips.

### Acquisition workflow to implement later

1. Curate approximately 12–20 approved shots: batting, bowling, fielding, nets,
   ball/bat details and limited ground atmosphere. Prioritize usable action over
   a numerical library quota. Prefer landscape HD/4K, clean subjects, no scorebugs,
   broadcaster graphics, music, watermarks, prominent advertising or unclear rights.
2. Store a local media library with a versioned JSON/YAML manifest (not a new
   learning database). Record asset/provider IDs, creator, landing/download URLs,
   acquisition time, exact license/version and saved terms, attribution requirement,
   original SHA-256, technical metadata, approved source ranges, framing/action
   tags, rights/release checks, reviewer and approval status.
3. Keep originals immutable. Record derived/proxy hashes and processing recipe;
   do not accidentally distribute original stock files as repository fixtures.
   Use synthetic fixtures in automated tests. Approve downloads manually initially;
   future automation must use permitted APIs, not uncontrolled scraping.
4. Choose only from approved manifest ranges. Use stable source/moment-based seeds
   so a resumed render reproduces its asset selection and shot order.
5. Match generic action to topic where appropriate without manufacturing evidence.
   Put `Illustrative cricket footage` on the lower panel; add source credits where
   required. A label cannot cure an otherwise misleading player/event association.
   Keep recognizable unrelated players away from allegations or insulting commentary.
6. Mute/drop every stock audio stream. Only original commentary is audible.
   Audit the original source for third-party match audio/music too: stock rights
   do not cure rights problems already present in the top video.
7. If assets are absent, corrupted, restricted or withdrawn, use another approved
   asset or hold this variant. Do not fetch random YouTube “no copyright” uploads,
   substitute IPL/ICC/TV footage, or silently claim the portrait fallback is a split.

YouTube's reused/inauthentic-content policy is distinct from copyright permission.
The original host commentary must supply real value; split-screen formatting alone
does not guarantee monetization or protect against claims. Preserve license evidence
for disputes; a clean private check is useful but not a future rights guarantee.

## 4. Visual specification: `cricket-stock-split-v1`

Initial design hypothesis, pending owner phone/playback approval:

```text
1080 × 1920, 30fps
┌───────────────────────────────────────┐ y=0
│ Original source / host                │
│ Face and expressive delivery readable │ 840px
├───────────────────────────────────────┤ y=840
│  FUTURISTIC WORD-TIMED CAPTIONS        │ 240px
├───────────────────────────────────────┤ y=1080
│ Licensed illustrative cricket action  │
│ No stock audio; no fake match claim    │ 840px
└───────────────────────────────────────┘ y=1920
```

- Top remains original video, not a generated face/body or frozen replacement.
  Use a source-aware camera/scene ROI; fit/pad if crop would cut the head, chin,
  gestures, guest or necessary source information. Do not blindly crop a 16:9
  stream with a tiny facecam. Review multi-person/screen-share cases explicitly.
- Bottom framing must preserve bowler/batter and action trajectory. Default to
  fitting the approved shot if fill-cropping loses the ball or bat. Cut between
  approved continuous shots at natural action boundaries, not arbitrarily mid-delivery.
  Do not reverse cricket action, freeze it to fill time or make a fake live replay.
- Base colors: near-black/navy `#07111F`, white `#F8FAFC`, cyan `#22D3EE` and a
  restrained violet `#8B5CF6` edge accent. Thin HUD/bracket details, subtle glow,
  dark glass/solid-backed text area. Bright readable type, not luminous clutter.
- Reuse the bundled OFL Oxanium font where readable. Start at 72–80px, bold,
  maximum two lines and roughly 2–4 spoken words per cue. Browser-measure actual
  ink bounds; split cues at valid word timestamps, never shrink/drop words to fit.
- Keep text within the center band with provisional horizontal bounds x=90–900;
  inspect current Shorts UI on phones before locking safe zones. These pixels are
  a design proposal, not an official universal YouTube overlay specification.
- Light active-word emphasis may follow existing word timing. No typewriter lag,
  flashing, continuous bouncing, unreadable all-caps proper names or effects that
  hide the opening words. If animated, use short restrained entrances only.
- Accurate Roman Hinglish captions, including negation, numbers, names and jokes.
  A punchline must not appear early as a spoiler. No fabricated “viral” dialogue.
  Keep the contextual-caption route and its model-identity/fidelity checks.
- Reuse caption scheduling/rasterization infrastructure, but add a separate style
  renderer and shared split geometry. Existing 320px caption transport canvases
  must be adapted explicitly; they cannot silently overflow a 240px center band.
- The source speech determines output duration. Bottom shots must cover that whole
  duration; a short stock stream must never terminate the host/audio through an
  unintended `-shortest`. Normalize PTS/fps/SAR and select only source audio.

## 5. Preserve hooks, setup and punches

### Source-bound editorial evidence

For each proposed moment retain exact cue IDs and, after preflight, word IDs for:
`hook_span`, `required_setup_spans`, `payoff_span`, opening/closing phrases and
any necessary before/after context. Every range refers to immutable source identity.

1. Discover across the complete original transcript. Cover quiet insights, debates,
   funny reversals and reactions, not only loud keywords. Record coverage and
   truncation/section limits explicitly. Topic-aware sections may improve locality,
   but must preserve every cue and bridge thoughts crossing section boundaries.
2. Make proposals cite copied cue/word IDs and exact strings. Diagnose the recorded
   invalid-anchor proposals; consider deterministic word-ID selection/copying rather
   than asking the model to retype a paraphrase. Never certify approximate citations.
3. Listen to regression examples rejected as non-cricket despite player/team words;
   fix demonstrated normalization/entity/context defects, not the hard niche gate.
4. Preflight a diverse reserve pool. Repair only from real neighboring speech;
   if ASR is garbled, retry the explicitly approved alignment route, then hold for
   listening/transcript correction with provenance. Do not alter confidence/times
   just to make validation pass.
5. Require a separate final review of the **actual retained words**. Changed cuts
   invalidate completeness and all prior hook/payoff attestations.
6. Protect the whole hook → setup → payoff span. Do not start after the key word,
   remove a meaningful comic pause, cut negation or end before the final phrase.
   Small pre/post handles should follow source audio evidence, not a blanket trim.
7. Keep 10–38s as a preference, not permission to amputate a thought. Propose a
   separately named, reviewed extension up to 60s for this profile when needed.
   All discovery/alignment/export/delivery bounds must agree; no hidden bypass or
   global limit increase. If the full thought still cannot fit, select another moment.
8. No automatic speed-up, internal silence removal, jump-cut reordering or
   model-written replacement hook. Fade-out cannot overlap protected payoff
   speech: disable it for this profile or apply it only after verified speech ends.
9. Compare the rendered opening 0–3s and full ending/payoff against source audio,
   plus word evidence and encoded timing. A transcript score cannot detect an
   encoder cutting a consonant or fading out the final joke. Owner listening stays.

### Two-moment recovery contract

- Desired selection: two primaries, up to two reviewed reserves, optional third.
- Track separate counts: discovered → valid citations → aligned → text-reviewed →
  encoded/QC-passed → owner-approved; also track distinct source moment IDs.
- If a primary fails export/caption QC, promote a reviewed **different** reserve;
  preserve successful siblings. A new cut needs fresh final review.
- If fewer than two remain, use one bounded targeted search of uncovered/unused
  moments and re-preflight/review. Prioritize fixing proven false negatives before
  expanding model calls. Do not resubmit known failed windows unchanged.
- Reserve call/time capacity for final review and recovery. Currently three
  sections across two discovery rounds plus boundary/final reviews can require
  ten calls before retries, exceeding the existing eight-call limit.
- Proposed pilot ceiling: up to 12 editorial attempts, 45 minutes total editorial
  stage wall time (including queue, retries/backoff and alignment), and a declared
  aggregate output-token/spend ceiling. Schedule against remaining budget; stop
  cleanly when exhausted. These are proposed limits, not implemented guarantees.
- If two drafts cannot be delivered, expose the lost-stage reasons and a manual
  review queue. A weak/too-short/non-cricket source is an explicit exception.
  Test sources independently adjudged to contain two usable moments must meet 2/2.

## 6. Thinking-model tokens and timing

### Official documentation checked now

- Native route remains `deepseek-v4-pro`; current pricing docs identify
  `DeepSeek-V4-Pro-0813`. The Flash successor is `deepseek-flash`, currently
  `DeepSeek-V4.1-Flash-1017`; legacy Flash names route to that successor.
- Current docs list a 1M context and maximum 384K (393,216) output tokens. Chat
  Completions defaults are 8K non-thinking, 64K thinking, 128K at max effort.
  These are provider limits/defaults, **not a recommendation to spend them all**.
- Reasoning and final output share the generated-token allowance. A tiny visible
  JSON result can still require thousands of generated reasoning tokens. Keep
  high thinking for editorial judgment; do not silently switch model or effort.
- `finish_reason=length`, empty final content, malformed JSON and `stop` with
  whitespace-only content all require failure handling. Never use reasoning text
  as final editorial JSON or repair a truncated decision into a pass.
- DeepSeek sends blank-line/SSE keep-alives while waiting. The documented ten-minute
  disconnect applies when **inference has not started**, not a ten-minute maximum
  total generation time. Keep-alives are transport liveness, not token progress.
- Official docs do not promise fixed tokens/second, latency or a task token count.
  Reddit/forum figures vary with model version, route, load, caching and prompt.

### Actual local evidence

Fourteen nonzero JSON-call durations in `logs/pipeline.log` on November 4/6 range
from **10.2 to 287.2 seconds**. November 6's six sequential calls alone total
**814.4s (~13.6 minutes)**, excluding ~10 minutes of intervening alignment/other work.
Its boundary review took **284.5s for only 380 final characters**. Thus 45s racing
or 60–120s generic deadlines are inappropriate for this stage; 300s offers little
headroom over observed slow successes. There is no token breakdown in these
strict-JSON logs, so deriving actual generation TPS from them would be invented.

### Proposed initial pilot policy (measure before locking defaults)

| Stage | Effort and token cap | Deadline policy |
| --- | --- | --- |
| Discovery/final review | Same native Pro, high, start at 32,768 combined generated tokens | Up to 900s true per-call wall time, limited by remaining stage budget |
| Boundary review | Same Pro, high; begin with current 32,768 cap for comparable evidence | Same 900s bound; compact source evidence, not amputated context |
| One truncation recovery | Same route/high, at most 65,536 after measured `length`/budget exhaustion | Up to 1,200s only if stage/spend budget leaves room; otherwise explicit hold |
| Captions/packaging | Keep separately configured exact route; benchmark independently | Do not globally inherit editorial deadlines or lower fidelity |

These are caps, not expected usage or provider SLAs. A larger cap gives room for
the final JSON but cannot reserve an exact internal reasoning/final split. An
editorial prompt should request concise final decisions, not an essay.

Illustrative **generation-only** arithmetic, NOT measured API speed:

| Combined output tokens | At 50 tokens/s | At 100 tokens/s |
| --- | --- | --- |
| 8,192 | 2m44s | 1m22s |
| 16,384 | 5m28s | 2m44s |
| 32,768 | 10m55s | 5m28s |
| 65,536 | 21m51s | 10m55s |

Add queue + prompt/prefill + network/validation time. The 65K cap cannot necessarily
be exhausted within a 20-minute deadline, particularly under load; reaching any
bound is a truthful hold, not permission to accept partial JSON.

Implementation details to verify:

- An SDK numeric timeout is not necessarily a total task deadline; repeated
  keep-alive bytes can keep a read alive. Use explicit monotonic wall-clock bounds,
  cancellation and a separately enforced stage deadline. Avoid orphaned requests
  or duplicate paid calls after cancellation. Do not blindly retry an unknown outcome.
- Optionally stream to report first byte, first model delta and first final-answer
  delta. Discard private reasoning text; retain only usage/counters, elapsed times
  and validated final JSON. Do not label keep-alives as “model is making progress.”
- Record exact provider/base host/model returned/fingerprint where available,
  requested effort/token cap, prompt/completion/reasoning/cached tokens, finish
  reason, elapsed, deadline classification, request IDs, retry number and cost.
  No keys, private reasoning or raw source transcripts in generic console logs.
- Provider token usage is authoritative; `characters / 4` is unreliable for Hindi
  and excludes hidden reasoning. Count request evidence with the appropriate
  tokenizer or provider usage; character caps are only transport safety bounds.
- Same-route retries only: bounded backoff for transient network/429/5xx using
  Retry-After. Invalid schema/citations get specific feedback. Authentication and
  retired-model errors fail clearly. Token escalation is not a fix for bad ASR.
- Proposed aggregate editorial output cap 262,144 tokens (~$1.04 peak output cost
  at the checked $3.96/M Pro price), plus separately capped input/spend. Recheck
  prices before the pilot; account for each retry and cache hit/miss. Do not infer
  costs from the legacy character-based logger or promise unlimited retries.

### Reddit/forum interpretation

Research found recurring reports of long reasoning phases, final-answer starvation
at small caps, variable route latency and client deadlines killing otherwise valid
requests. A GitHub issue demonstrates empty final output at a tiny DeepSeek token
budget; other reports distinguish zero-token empty responses from budget exhaustion.
These support instrumented bounded waits and explicit error classes, **not** copying
a reported GPU/API TPS figure into our SLA. Flash anecdotes are not Pro benchmarks;
agentic coding loops are not our one-shot editorial workload. Older preview-model
reports may not describe the current native API. Prefer current docs + our pilot.

Benchmark later on at least 10 representative saved requests across discovery,
boundary and final review, with additional samples before claiming a stable p95.
Record cold/warm/cache/load differences and repeat failure cases. Compare 32K vs
64K only where truncation evidence warrants it; acceptance is better valid decisions,
not more thinking tokens. No benchmark or paid provider call was run in this task.

## 7. Planned implementation map (no changes made)

| Area | Planned responsibility |
| --- | --- |
| New stock manifest/selector module | Rights validation, hashes, approved ranges, stable shot selection and fallback/hold |
| New split layout module | One geometry contract for source/stock panels, caption ink and review overlays |
| New futuristic caption style | Shared word scheduling, local fonts/rasterization, center-band measured fit |
| `export.py` | Two-input composition, explicit original audio, full duration coverage, protected payoff transitions |
| `automation/editorial_planner.py` / `automation/speech_alignment.py` | Demonstrated false-negative repairs, exact citations, protected spans, reserve-aware budgets |
| `automation/clip_selection/hook_auditor.py` | Distinguish advisory hook score from source/encoded integrity evidence |
| `utils/ai_client.py` | Strict JSON usage/timing telemetry, real deadlines, cancellation and bounded retry policy |
| `automation/orchestrator.py` | Distinct-moment minimum, preserve good outputs, variant-aware batch/review status |
| Config / CLI | Explicit profile selection and bounded budget settings; default portrait remains unchanged |
| Existing edit/review sidecars | Composition/version, asset hashes/ranges/license evidence, word spans, all approvals bound to exact output |
| `shorts_intelligence/` | Only if needed, carry approved variant features through existing canonical telemetry; no second learner/database |

The stock manifest and review artifacts are production evidence, not analytics
outcomes. Do not change the learner's bounded adjustments, clip boundaries or
complete-thought gate. Preserve the exact channel Shorts-shelf ownership contract.

## 8. Failure-first validation and acceptance

Implement tests before the corresponding behavior changes:

1. **Rights:** reject the Mixkit Restricted example, missing license/hash,
   unapproved ranges, missing required attribution, corrupted media and withdrawn
   assets. No unapproved network download in a render or dry run.
2. **Layout:** assert decoded pixels for top/source, center/captions and bottom/stock;
   validate aspect ratios, source crop, action framing, safe text bounds, long names,
   guests and missing face detection. Golden frames alone do not prove motion quality.
3. **Duration/audio:** short stock input cannot shorten the main output; stock audio
   must be absent even with multiple audio tracks. Decode the whole encoded file,
   check timestamps/duration/A-V sync, no black/frozen gaps, and existing audio QC.
4. **Hook/payoff:** recorded Hindi/Hinglish fixtures with clipped first syllables,
   end-word cuts, negation, numbers, anticipation pauses, delayed punchlines, garbled
   ASR and thoughts crossing sections. Cuts/fades that touch protected speech fail.
   An exported MP4 plus a keyword score cannot make these tests pass.
5. **Two distinct moments:** eligible source with two complete moments → two QC
   drafts; one render failing → reviewed reserve; same moment in two formats counts
   once; short/weak source → truthful shortfall; successful sibling survives resume.
6. **Model resilience:** delayed 400–800s valid responses, keep-alive-only stalls,
   length truncation, blank `stop`, invalid citations, bad JSON, queue close, 429,
   cancellation, no usage, aggregate budgets and preserved final-review capacity.
   Simulate time in tests; do not make the suite wait 15 minutes per test.
7. **Regression:** portrait default unchanged; caption fidelity/numbers/model identity,
   dedupe, completeness, canonical analytics and owner approval still enforced.
8. **Mutation checks:** bypassing rights, enabling stock audio, restoring an
   unconditional end fade, counting duplicate variants twice, or accepting partial
   JSON must each make a relevant test fail. Counters/stubbed success are insufficient.

Mandatory validation before any future code commit:

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v --ignore=tests/test_face_detect.py
.venv\Scripts\python.exe dry_run.py https://youtu.be/test --skip-download --skip-transcribe
.venv\Scripts\python.exe -m pytest tests/test_automation.py -v
.venv\Scripts\python.exe -m pytest tests/shorts_intelligence/ -v
```

Pilot media acceptance: at least three real substantive sources adjudged to contain
two usable moments, including the recorded problem sources where available. At
least two distinct split drafts per eligible source, complete first/last words and
payoff, accurate captions, asset-level rights evidence, full playback/listening
and phone UI review. If a source is ineligible, record why rather than changing the
test label solely because the pipeline failed. No automatic public upload.

## 9. Hook/layout experiment after technical and owner approval

- Stage A: same protected moment locally rendered as portrait vs split; blind
  owner/reviewer comparison for legibility, distraction and joke comprehension.
- Stage B: evaluate alternative **source-faithful** opening boundaries locally;
  both must keep the whole setup/payoff. Never optimize by spoiling the ending.
- Stage C: after separate publication approval, use distinct matched clips across
  control/treatment cohorts. Keep topic/type/duration/posting windows comparable;
  vary layout first, then hook treatment. Do not publicly upload near-duplicate
  versions of one moment or count them as extra daily content.
- Initial exploratory floor: 10 distinct clips per arm, 7-day observation each;
  extend if traffic is sparse. This is not a statistical-power guarantee. Report
  sample size, exposure and uncertainty; avoid “viral” thresholds copied from Reddit.
- Metrics: stayed-to-watch/swiped-away, early retention where available, average
  view duration/percentage, engaged views and subscribers per 1,000 engaged views;
  use only real available Analytics/Studio evidence. Do not fabricate unavailable
  API metrics or treat raw autoplay views as the sole success criterion.
- Provisional promotion target: +5 percentage points stayed-to-watch with no more
  than 5% relative AVD loss in comparable duration groups, no comprehension decline,
  no material rights/quality issues. Treat as a pilot decision rule, not a platform
  ranking law; more evidence is needed if the uncertainty spans material harm.
- Rollback: select existing portrait profile and stop split publication for any
  rights problem, clipped speech, illegible captions, distracting action or sustained
  retention regression. Preserve artifacts/evidence; do not delete unrelated work.

| Recommendation | Impact | Risk / difficulty | Confidence | Validation / rollback |
| --- | --- | --- | --- | --- |
| Protected hook/payoff spans + two-moment recovery | High | Medium / High | High for integrity, unproven yield | Failure fixtures + real sources; keep holds if unresolved |
| Curated licensed stock library | High | Rights risk / Medium | High on workflow, individual assets pending | Asset review; substitute approved shot or hold |
| Futuristic split visual profile | Potentially high | Distraction / Medium | Medium-low on growth | Local comparison + controlled cohort; portrait rollback |
| Longer instrumented high-thinking calls | High for avoiding premature aborts | Spend/latency / Medium | High on need, medium on exact caps | Usage + latency pilot; bounded hold/resume |

## 10. Execution order after plan approval

1. Adjudicate recorded editorial failures and select rights-reviewed stock library.
2. Establish protected-span/citation/count contracts with red regression tests.
3. Add deadline/token telemetry and recovery scheduling without changing model quality.
4. Build the opt-in compositor and futuristic caption style with deterministic media tests.
5. Run required regression/dry-run checks, then the three-source real pilot.
6. Obtain separate visual, caption/listening, editorial and release approvals.
7. Run the controlled channel experiment only after publication authorization.

No code tests were run for this documentation-only task. No implementation or
end-to-end two-clip success is claimed. Main unresolved work: stock acquisition/
rights clearance, real source listening, measured token budgets, actual renders
and owner approval of the new design.

## Research sources

Official rights/platform sources:

- https://www.pexels.com/license/
- https://help.pexels.com/hc/en-us/articles/360042295214-Can-I-use-the-photos-and-videos-for-a-commercial-project
- https://www.pexels.com/search/videos/cricket%20sport/
- https://pixabay.com/service/license/ (indexed text only; direct fetch blocked)
- https://mixkit.co/license/
- https://mixkit.co/license/modal/videoFree/
- https://mixkit.co/free-stock-video/cricket-court-in-a-stadium-12233/ (item-level restriction)
- https://support.google.com/youtube/answer/1311392?hl=en (monetization/reused/inauthentic content)
- https://support.google.com/youtube/answer/12220281?hl=en (metric definitions)
- https://support.google.com/youtube/answer/12942217?co=YOUTUBE._YTVideoType%3Dshorts

Official model documentation (current fetched Markdown preferred over stale snippets):

- https://api-docs.deepseek.com/guides/thinking_mode.md
- https://api-docs.deepseek.com/api/create-chat-completion
- https://api-docs.deepseek.com/quick_start/pricing.md
- https://api-docs.deepseek.com/quick_start/rate_limit
- https://api-docs.deepseek.com/guides/json_mode

Community/engineering reports (anecdotal, route/version/workload dependent):

- https://github.com/noah-art3mis/veritrace/issues/102 (fetched; shared reasoning/output budget)
- https://github.com/NousResearch/hermes-agent/pull/148439 (indexed; premature reasoning stale-timeouts)
- https://github.com/deepseek-ai/DeepSeek-V3/issues/1453 (indexed; zero-token empty response is a different failure)
- https://www.reddit.com/r/DeepSeek/comments/1wip5u7/v41flash_api_thinking_effort_vs_model_thinking/ (search-indexed; direct fetch returned no usable text)
- https://www.reddit.com/r/opencode/comments/1xdme8b/deepseek_v41_flash_spent_55_of_a_coding_session/ (search-indexed; queue/wall-time versus generation speed)
- https://www.reddit.com/r/NewTubers/comments/1742qth/viral_short_analytics_breakdown_with_screenshots/ (search-indexed; hook/retention hypotheses, not causal proof)
- https://www.reddit.com/r/NewTubers/comments/1fzrtgf/ive_been_working_on_shorts_for_2_years_and_i_just/ (search-indexed; hook/story/payoff and metric hypotheses)

Local authority/evidence: `AGENTS.md`, `automation/AGENTS.md`, `config.yaml`,
`docs/TRANSCRIPT_FIRST_CURATION.md`, `docs/COMPLETENESS_REVIEW_CONTRACT.md`,
`docs/PORTRAIT_REFERENCE.md`, `automation/editorial_planner.py`,
`automation/clip_selection/hook_auditor.py`, `automation/orchestrator.py`,
`utils/ai_client.py`, `export.py`, `logs/pipeline.log`,
`highlights/video.curation.json`. Research and existing model opinions do not
constitute listening, footage clearance, publication approval or channel outcomes.
