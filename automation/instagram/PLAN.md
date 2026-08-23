# Instagram SEO + Upload Workflow — Parallel Agent Battle Plan

**Branch:** `feat/insta-seo-upload` (isolated; merges to `main` only after full approval chain)
**Policy:** REAL trends ONLY. Zero AI-invented hashtags/phrases. Every caption token must trace
to a verified evidence item (Cricbuzz facts, roster, IG hashtag API volume, our learner DB,
YouTube-suggest corpus). Same standard as YT entity grounding — audited, scrubbed, fail-loud.

---

## 1. Goal & Non-Goals

**Goal:** For every exported clip: generate grounded Instagram caption+hashtags from live real
trend data, publish as a Reel, fully resumable per-stage and skippable — then run YouTube SEO/upload
and Instagram SEO/upload **in parallel**, each with an independent failure domain.

**Non-goals (v1):** Stories/carousel posts, DM automation, comment management, multi-account,
scheduling UI, Instagram analytics ingestion into `shorts_intelligence.db` (v2 — schema hook left open).

## 2. Architecture

```
automation/instagram/
├── PLAN.md            ← this file (living doc)
├── graph_client.py    ← thin IG Graph API wrapper (container/publish/status/hashtag/insights)
├── credential.py      ← insta_token.json load+refresh, mirrors setup_auth.py pattern
├── host.py            ← public MP4 URL provider chain (drive primary → fallback), HEAD preflight
├── evidence.py        ← InstaEvidencePack builder + REAL hashtag validation via Graph API
├── caption_engine.py  ← LLM caption writer FROM EVIDENCE ONLY + audit/scrub (reuse entity_grounding)
├── seo.py             ← orchestrator: evidence → caption → audit → insta_metadata.json (staged)
├── uploader.py        ← container → poll FINISHED → publish → verify permalink
├── runner.py          ← marker protocol, stage checkpoints, retry_failed_insta(), skip logic
└── integration seam   ← publish_everywhere() in automation flow (thread fan-out)
```

### Frozen contracts (implementers code against these)

```python
# graph_client.py — Protocol; tests inject FakeGraphClient
class GraphClient(Protocol):
    def create_reels_container(self, video_url: str, caption: str, *,
                               share_to_feed: bool = True,
                               audio_name: str | None = None) -> str: ...   # creation_id
    def container_status(self, creation_id: str) -> dict: ...               # status_code/error
    def publish_container(self, creation_id: str) -> str: ...               # media_id
    def media_permalink(self, media_id: str) -> str: ...
    def hashtag_search(self, q: str) -> dict: ...                           # REAL volume probe
    def hashtag_top_media(self, hashtag_id: str, *, limit: int = 5) -> dict: ...

# runner.py
def process_instagram_for_clip(clip_dir: Path, transcript: str,
                               video_title: str, video_description: str,
                               *, force: bool = False) -> str | None       # media_id | None
def retry_failed_insta(shorts_root: Path = "shorts") -> dict               # consumes markers

# integration (automation flow)
def publish_everywhere(clip_dir, transcript, video_title, video_description,
                       *, skip_youtube=False, skip_instagram=None) -> dict:
    """ThreadPoolExecutor(2). YouTube failure keeps existing marker flow.
       Instagram failure is NEVER fatal to the YouTube ship."""
```

### State machine (per clip, resumable)

```
EVIDENCE → CAPTION → AUDIT → HOST → CONTAINER → PUBLISH → VERIFY → DONE
```

- Checkpoint file: `shorts/<run>/clip_insta_state.json`
  `{stage, attempts, last_error, updated_at}` — resume skips completed stages.
- Failure at any stage → `clip_insta_failed.json` marker (same semantics as
  `clip_seo_failed.json`) → next run's `retry_failed_insta()` picks it up.
- Skip paths (any one suffices): CLI `--skip-instagram`, config `instagram.enabled: false`,
  env `YT_CLIPS_SKIP_INSTAGRAM=1`. Skipped clips get no markers, zero network calls.

### Evidence pack schema (REAL-ONLY policy enforcement point)

```python
{
  "match_facts": [...],           # Cricbuzz verified (fetch_verified_match_context)
  "roster": [...],                # verified players only
  "learner_top_captions": [...],  # OUR shorts_intelligence.db outcomes
  "seed_phrases": [...],          # YT suggest strings — labeled proxy corpus
  "validated_hashtags": [         # EVERY tag MUST come from here
      {"tag": "...", "recent_volume_24h": int, "source": "ig-hashtag-api"}],
  "forbidden": ["invented player names", "generic tags without volume evidence"]
}
```

Caption rules: ≤2200 chars · first 125 chars carry primary keyword + match hook ·
keywords woven from pack only · 3–10 hashtags ALL from validated_hashtags · alt_text set ·
no Devanagari in title-like fields · audit reuses `audit_written_copy_llm` scrub pattern ·
policy violation → corrective LLM repair pass → still dirty → loud raise + failed marker.

### Config additions (`config.yaml`)

```yaml
instagram:
  enabled: false              # flips true only after creds verified + live smoke
  ig_user_id: ""
  token_file: insta_token.json
  host_provider: drive        # drive | r2 | tunnel
  caption_max_chars: 2200
  hashtags_min: 3
  hashtags_max: 10            # hard API cap 30 — research may revise guidance
  upload_timeout_s: 900
  max_posts_per_day: 25
```

Test kill-switches: `YT_CLIPS_INSTA_LIVE=0` (default in conftest — all Graph calls stubbed),
extended `dry_run.py` prints `INSTA(STUB): PASS`.

## 3. Agent Roster (parallel waves)

| Wave | Agent(s) | Job | DoD |
|---|---|---|---|
| 0 | **Architect** (lead) | Freeze contracts above, branch, plan doc | This doc committed |
| 1 | **R1 Researcher** | Exact 2026 Graph API Reels contract: endpoints, params, polling states, errors, rate limits, app perms/modes, video specs | Endpoint table + JSON shapes + citations + confidence |
| 1 | **R2 Researcher** | Public direct-MP4 hosting for `video_url`: Drive direct-link viability vs R2/S3/tunnel/Dropbox; Content-Type/redirect constraints; size ceiling | Primary+fallback provider rec with test evidence |
| 1 | **R3 Researcher (IG SEO expert)** | Caption keyword search mechanics 2026, hashtag count (Meta official vs creator consensus), first-line hook, alt text API support, audio name, Trial Reels | Rule list separating OFFICIAL / consensus / speculation |
| 1 | **R4 Researcher (IG data expert)** | Can a business account get REAL hashtag volume in 2026? ig-hashtag-search current status, top_media signals, own-insights alternatives; enumerate FORBIDDEN practices | Go/no-go verdict + exact query recipe |
| 2 | **I1–I6 Implementers** (TDD, parallel) | I1 graph_client+credential · I2 host.py · I3 evidence.py · I4 caption_engine+seo · I5 uploader+runner · I6 pipeline integration | Failing tests FIRST → impl → scoped pytest green |
| 3 | **Reviewer A + Reviewer B** | Independent 8k-token reviews, never read each other; Severity/Problem/Evidence/Fix format | All Critical/High resolved |
| 3 | **Consensus chair** | Exchange findings, max 2 debate rounds, evidence wins | Merged finding list |
| 4 | **Approver A + Approver B** | Final gate: TDD proof, architecture fit, production readiness (B = IG-domain expert: compliance, rate limits, ToS) | Explicit APPROVE |
| 4 | **QA Engineer** | Stage-fail→resume matrix, kill-mid-upload (no orphan posts), skip-flag zero-network, offline mode, rate-limit soak | Full matrix green |
| 5 | **Integrator** | Full suite + dry_run gates, commit discipline, merge | AGENTS.md gates PASS |

Review protocol = house `elite-engineering-team`: max 3 review cycles, max 2 debate rounds,
evidence wins, nothing merges on one reviewer's word.

## 4. Test Matrix (QA owns)

1. Fail injection at EACH of the 7 stages → resume completes without redoing prior stages
2. Kill process mid-PUBLISH → resume detects orphan container, no duplicate Reel
3. All three skip paths → zero HTTP calls asserted (mock counter == 0)
4. FakeGraphClient contract tests: FINISHED / IN_PROGRESS timeout / ERROR paths
5. Caption audit: invented player + unvalidated hashtag → scrubbed or raised
6. `retry_failed_insta()` end-to-end on fixture markers
7. Parallel fan-out: insta worker raises → youtube result unaffected
8. Rate-limit soak: N containers within budget window → serialized correctly
9. Offline mode: no network → clean failed markers, pipeline ships YT anyway
10. Live smoke (gated by `instagram.enabled: true`): one real Reel → permalink verified → manual review

## 5. Risk Register

| # | Risk | Mitigation |
|---|---|---|
| 1 | `video_url` hosting rejected by IG validator (top risk) | Dual provider chain + HEAD preflight (200, video/mp4) before container create |
| 2 | Hashtag endpoints restricted/deprecated for business accounts | R4 verdict gates evidence design; fallback = learner DB + Cricbuzz facts (still zero AI generics) |
| 3 | Broadcast-footage rights enforcement on IG | Operational note — channel owner's call; researcher documents enforcement norms only |
| 4 | Rate limits / daily post caps | Serial post queue + `max_posts_per_day` config + backoff on error codes |
| 5 | Token expiry mid-run | Refresh flow mirrors setup_auth; stale-callback relaunch pattern documented |

## 6. Prerequisites (channel owner — arrange while researchers run)

- [ ] Instagram **Professional/Business** account linked to a Facebook Page
- [ ] Meta app with `instagram_business_basic` + `instagram_business_content_publish`
- [ ] Account added as admin/tester on that app (dev mode works for own account)
- [ ] Note the IG user id (graph explorer `me?fields=user_id` or Page linked IG account)
- [ ] Drop app id/secret + long-lived token into `setup_auth.py` extension when ready

## 7. Milestones

M1 contracts frozen (done) → M2 research verdicts merged → M3 I1–I6 TDD green →
M4 reviews+approvals passed → M5 QA matrix green + gates PASS → M6 live smoke Reel →
M7 merge `main`, flip `instagram.enabled: true`.
