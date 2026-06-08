# SEO Pipeline — Complete Flow

> Every detail of how YouTube Shorts SEO is generated, uploaded, and learned from.
> File: `automation/seo/` + `automation/orchestrator.py` + `utils/ai_client.py`

---

## Table of Contents

1. [Pipeline Entry Points](#1-pipeline-entry-points)
2. [Trend Context Fetching](#2-trend-context-fetching)
3. [Clip Analysis & Export](#3-clip-analysis--export)
4. [Per-Clip SEO Generation (the core)](#4-per-clip-seo-generation-the-core)
   - 4a. Prompt Construction
   - 4b. Learner Context Injection
   - 4c. The Racer (Fastest-First Parallel Model Racing)
   - 4d. JSON Parsing & Truncation Repair
   - 4e. Quality Gate & Limits
   - 4f. Escalation (Tier 2)
   - 4g. Failure Markers & Retry
5. [Thumbnail Generation](#5-thumbnail-generation)
6. [YouTube Upload](#6-youtube-upload)
   - 6a. Scheduling & Dead-Day Gating
   - 6b. Token Refresh
7. [Stage 9 — Self-Learning](#7-stage-9--self-learning)
   - 7a. Analytics
   - 7b. Self-Learner (PersistentMemory)
   - 7c. SEOLearner (Performance DB)
   - 7d. Automation Learner (PolicyUpdater)
   - 7e. Cricket Learning Engine
   - 7f. Provider Health
8. [SEOLearner Deep Dive](#8-seolearner-deep-dive)
   - 8a. Performance Recording
   - 8b. Feature Extraction
   - 8c. Time-Decay Weighting
   - 8d. Pattern Learning
   - 8e. Auto-Benchmark
   - 8f. Prompt Enhancement
9. [AI Client & The Racer](#9-ai-client--the-racer)
   - 9a. Provider Chain
   - 9b. Token Bucket Rate Limiting
   - 9c. Circuit Breaker & Cooldown
   - 9d. Error Classification
10. [Why Architecture Is This Way](#10-why-architecture-is-this-way)

---

## 1. Pipeline Entry Points

### CLI (`automation/cli.py`)

```
.venv/bin/python -m automation.cli <youtube_url> [flags]
```

Flags relevant to SEO:
- `--upload` — enable YouTube upload after export (`auto_upload=True`)
- `--schedule` — enable scheduled upload with time slots (`auto_schedule=True`)
- `--skip-seo` — skip SEO generation
- `--skip-export` — skip clip export (use existing clips)
- `--mode` — enhancement mode (`ref_grade` or `face_mapper`)
- `--dry-run` — print what would be done without executing

### Orchestrator (`automation/orchestrator.py`)

The `run()` function executes 9 stages. SEO-relevant stages:

| Stage | Function | File |
|-------|----------|------|
| 3-5 | Highlight detection, scoring, ranking | `highlight.py` |
| 6 | Export clips + SEO generation | `export.py`, `automation/seo/seo.py` |
| 7b | Thumbnails | `thumbnail.py` |
| 8a | Drive sync | `sync.py` |
| 8b | YouTube upload | `upload.py` |
| 9a-9e | Self-learning | multiple |

---

## 2. Trend Context Fetching

**File:** `automation/seo/trends.py`

Before any SEO is generated, `get_trending_context()` is called **once** per pipeline run (results cached in `TREND_CACHE` for 300s). This function aggregates **6 sources**:

### Source 1: Google Trends RSS (India)
```python
fetch_google_trends_in()  # https://trends.google.com/trending/rss?geo=IN
```
Why: Shows what India is searching for right now. Cricket is region-specific, so geo=IN is critical.

### Source 2: YouTube Suggest API
```python
fetch_youtube_suggestions("cricket live")
# Hits: https://suggestqueries.google.com/complete/search?client=firefox&ds=yt&q=...
```
Why: YouTube's own autocomplete reveals what users actually type. This is the single best signal for what search terms to target. The seed query "cricket live" is intentionally broad to capture general cricket intent.

### Source 3: Competitor Signals (Google News RSS)
```python
fetch_competitor_signals()
# https://news.google.com/rss/search?q=cricket+live+score+today+IPL&hl=en-IN&gl=IN
```
Why: News headlines reveal what topics are being actively covered by other cricket channels. If everyone is covering a specific match moment, it's trending.

### Source 4: Cricbuzz Live Scores
```python
fetch_cricbuzz_live_score(video_title, match_type)
```
Why: Injects actual match context (score, top scorers, bowling figures) into the SEO prompt. This makes titles factually accurate instead of generic. Protected by a `CircuitBreaker` (3 failures → 60s cooldown).

### Source 5: Own Live Stream URL
```python
fetch_own_live_stream_url(channel_id)
```
Why: If the channel is currently live streaming, the live URL is injected into the prompt so the AI can reference the live stream.

### Source 6: Team Extraction from Video Title
```python
extract_match_teams(video_title)
```
Why: Parses the video title for team names (CSK, MI, RCB, etc.) using `TEAM_MAPPINGS`. Also detects match type (ipl/t20/test/odi).

### Return Value

```python
{
    "topics": ["...", ...],        # 20 unique trending topics
    "scorecard": "Match: ...",     # Cricbuzz scorecard text
    "live_stream_url": "https://...",  # Own live stream (if active)
    "teams": ["CSK", "MI"],        # Extracted from title
}
```

---

## 3. Clip Analysis & Export

**File:** `export.py` (called in orchestrator stage 6)

Before SEO runs, `export_all()` does:

1. **Face detection** via MediaPipe `face_detector.tflite` — detects if the person in frame matches reference photos
2. **Chat detection** — detects whether a chat overlay is present on the right side (common in live stream VODs), excludes it for proper 9:16 crop
3. **Variable speed analysis** — analyzes silence gaps, determines playback speed (e.g., 1.15x if low silence)
4. **Encoding** — uses `h264_nvenc` (GPU) for fast export, parallel workers
5. **Output** — 6 clips go to `shorts/YYYY-MM-DD_HHMMSS/clip{1-6}.mp4`

**SEO does NOT run during export.** Export sets `generate_seo=False`. SEO runs **immediately after** all clips are exported, as a separate step:

```python
# orchestrator.py:267-269
if not skip_seo and result.exported:
    export_dir = str(result.exported[0].parent)
    process_all_seo(highlights_path, export_dir)
```

---

## 4. Per-Clip SEO Generation (the core)

**File:** `automation/seo/seo.py`, function `process_all_seo()`

**Flow:**

```
process_all_seo(highlights.yaml, export_dir)
  │
  ├── Load highlights YAML → {clip_id: {text: "...", ...}}
  ├── Load video_metadata.json (title, live_stream_url)
  ├── Fetch trend context ONCE (cached in TREND_CACHE)
  │
  ├── For EACH clip (sequential, 5s sleep between):
  │     ├── generate_clip_seo(clip_id, transcript, ...)
  │     │     ├── Build prompt with:
  │     │     │     ├── CLIP TRANSCRIPT (timestamped text)
  │     │     │     ├── VIDEO TITLE
  │     │     │     ├── SCORECARD (from Cricbuzz)
  │     │     │     ├── TREND TOPICS (from Google/YouTube)
  │     │     │     ├── TEAMS (extracted from title)
  │     │     │     └── LEARNER CONTEXT (from self_learner.db)
  │     │     │
  │     │     ├── generate_fastest_first()  ⬅ THE RACER
  │     │     │     └── Returns first valid JSON from 5 parallel models
  │     │     │
  │     │     ├── Parse JSON (handles markdown, truncation, Python dict)
  │     │     ├── Validate quality gate
  │     │     ├── Enforce limits (title≤80, desc≤800, tags≤5)
  │     │     │
  │     │     └── If AI fails → escalation_seo() (simpler salvage prompt)
  │     │
  │     ├── Write clip_{id}_metadata.json to disk
  │     ├── Record SEO outcome to SEOLearner
  │     └── Sleep 5s (configurable)
  │
  └── Write seo_results.json (combined)
```

### 4a. Prompt Construction

The prompt template (`_PROMPT_TMPL` in seo.py:207-256) contains:

```
CONTEXT:
  Match: {video_title}
  Scorecard: {scorecard}
  Live Trending: {trend_topics}
  Live Streaming URL: {live_stream_url}
  Teams: {teams}

CLIP TRANSCRIPT: {transcript}

TASK: Generate YouTube Shorts SEO for this specific clip.

You MUST return valid JSON:
{
  "title": "...",
  "description": "...",
  "hashtags": [...],
  "search_terms": [...]
}
```

**Title requirements (Hinglish only):**
- Start with the most important event of THIS CLIP
- Hinglish = Hindi words in Roman alphabet, NEVER Devanagari
- Include the most dramatic moment
- End with relevant emojis
- Max 80 characters
- CRITICAL: be clip-specific, not match-generic

**Description requirements (English):**
- First 2 lines: what happened in this specific clip
- Then: context of the match situation
- Then: player stats/achievements if relevant
- End with CTA
- Shorts: max 300 chars, Regular: max 800 chars

**Search terms (English):**
- Primary: Player name + action (e.g., "virat kohli six wankhede")
- Generic terms like "cricket video" are filtered out

**Hashtags:**
- Primary: #PlayerName, #TeamName
- Include #Shorts
- Max 5

### 4b. Learner Context Injection

**File:** `seo.py:92-135`, function `_get_learner_context()`

Before the prompt is sent to the AI, the `self_learner.db` SQLite database is queried for:

1. **Entity scores** — `entity_scores` table:
   - Top 5 players by score with avg views
   - Low-ROI players to avoid (n>10, avg_views<150)
   - Top 3 teams by score with avg views

2. **Format scores** — `format_scores` table:
   - Top 3 hook types by avg views

This data is appended to the prompt under "CHANNEL INTELLIGENCE (from past N videos):".

Why: Biases the AI toward title formats, players, and teams that have historically performed well on THIS specific channel. Without this, every SEO call is a cold start.

### 4c. The Racer (Fastest-First Parallel Model Racing)

**File:** `utils/ai_client.py:393-447`, method `generate_fastest_first()`

**Why a Racer exists at all:**

Different LLM providers have vastly different latencies:
- Groq: 400-1000ms (fast inference hardware)
- NVIDIA: 5000-45000ms (slower, more loaded)
- OpenCode: varies

Waiting for every model sequentially would be 60-120s per clip. Instead, the Racer fires ALL available models **concurrently** and takes the **first valid JSON response**. The rest are cancelled.

**How it works:**

```python
def generate_fastest_first(self, prompt, system_instruction, prefer_provider, prefer_model):
    # 1. Build model list from all providers with API keys
    all_models = self._all_models()            # [(provider, model), ...]
    runnable = [(p, m) for p, m in all_models
                if self._check_and_consume_token(p, m)]  # Rate limit check

    # 2. Submit ALL models to ThreadPoolExecutor simultaneously
    with ThreadPoolExecutor(max_workers=len(runnable)) as exc:
        fut_map = {exc.submit(make_call, p, m): (p, m) for p, m in runnable}

        # 3. Wait for FIRST_COMPLETED
        done, pending = wait(pending, timeout=remaining,
                             return_when=FIRST_COMPLETED)

        # 4. Take first valid result
        for fut in done:
            text = fut.result()
            if text and "API key missing" not in text:
                # CANCEL all pending futures
                for f in pending:
                    f.cancel()
                return text.strip()

    # 5. If all fail → return ""
```

**Deadline:** The racer uses the longest model-specific timeout from `MODEL_TIMEOUTS` (default 45s, qwen3.7-max gets 180s). If all models time out, the racer returns `""` and the escalation tier kicks in.

**Model order preference:** If `prefer_provider` or `prefer_model` is set, those come first in the list. Otherwise models are randomly shuffled to distribute load.

### 4d. JSON Parsing & Truncation Repair

**File:** `seo.py:534-660`, function `_parse_json_response()`

LLMs frequently return JSON that is:
- Wrapped in markdown code fences (```json ... ```)
- Truncated mid-object (last few fields missing)
- Python dict syntax instead of JSON (single quotes)
- Surrounded by prose ("Here is your SEO: {...} — done!")

The parser tries **4 strategies** in order:

1. **Direct parse** — `json.loads(text)`
2. **Markdown extraction** — regex for ```json {...} ```
3. **Balanced brace search** — finds first `{` and balances `{...}` ignoring prose
4. **Truncation repair** — progressively strips trailing partial values, adds closing braces

```python
def _repair_truncated_json(s):
    # Try closing unclosed braces: { needed } needed
    # Progressively trim last incomplete line
    # Up to 5 attempts with shorter suffixes
```

### 4e. Quality Gate & Limits

**File:** `seo.py:505-531`, function `_validate_seo_quality()`

Before any SEO is accepted, it passes a quality gate:

- Title must exist and be ≥10 chars
- Title must not be a known generic pattern (from `GENERIC_TITLES` set)
- Description must be ≥20 chars
- Title must NOT contain Devanagari script (kills discoverability in YouTube's algorithm which prefers Latin/Hinglish for international reach)

**After the gate, limits are enforced** via `_enforce_limits()`:

| Field | Max |
|-------|-----|
| title | 80 chars |
| description | 800 chars |
| hashtags | 5 |
| search_terms | 10 |

**Generic poison terms are stripped** from search_terms (`GENERIC_POISON_TERMS`):
- "cricket highlights", "cricket live match", "ipl match video", etc.

Why: These generic terms waste SEO budget and make the channel look like every other cricket clip channel. YouTube punishes channels that use the same generic tags as millions of others.

### 4f. Escalation (Tier 2)

**File:** `seo.py:824-857`, function `_escalation_seo()`

If the racer returns nothing valid, a **simpler salvage prompt** is used (`_SALVAGE_TMPL`):

```
Generate YouTube Shorts SEO for this cricket clip.
Match: {video_title}
Clip: {transcript}
Requirements:
- Title: Hinglish, max 80 chars, with emojis
- Description: English, casual, max 500 chars
- Hashtags: max 5, include #Shorts
- Search terms: 3-5 English terms
Return valid JSON ONLY:
{...}
```

This simpler prompt has fewer constraints, making it more likely to succeed when the full racer fails. It's sent via `generate_text()` (single model, not the racer) — which itself has a failover chain.

If **both tiers fail**, a `SEOGenerationError` is raised and a `*_seo_failed.json` marker file is written to disk for later retry.

### 4g. Failure Markers & Retry

**File:** `seo.py:1074-1121`, function `retry_failed_seo()`

When SEO fails for a clip:
1. A `{clip_id}_seo_failed.json` file is written with the transcript and video_title
2. The `_metadata.json` file is deleted if it exists
3. The result dict gets `{"_seo_failed": True}`

On subsequent runs, `retry_failed_seo(output_dir)` scans for `*_seo_failed.json` markers and re-attempts generation. On success, the marker is deleted and a fresh `_metadata.json` is written.

---

## 5. Thumbnail Generation

**File:** `thumbnail.py` (called in orchestrator stage 7b)

After SEO, thumbnails are generated for all exported clips:
```python
process_all_thumbnails(export_dir)
```
Generates `clip{1-6}_thumb.jpg` for each clip. No AI involved — uses ffmpeg to extract a frame.

---

## 6. YouTube Upload

**File:** `upload.py` (called in orchestrator stage 8b)

### 6a. Scheduling & Dead-Day Gating

When `auto_schedule=True`, uploads are spread across time slots:
```python
assign_clips_to_slots(clips, interval_hours=1, clip_scores=...)
```

- Each clip (except the first) gets a scheduled publish time
- **Jitter**: 2-3h random jitter per clip (not multiplicative — that was a fixed bug)
- **Dead-day gating**: If a slot lands on a "dead day" (e.g., Sunday for cricket), the slot is shifted to `next_upload_day()`

### 6b. Token Refresh

**File:** `upload.py:auth` section

```python
if creds and creds.expired and creds.refresh_token:
    creds.refresh(Request())
```

YouTube OAuth tokens naturally expire. The uploader detects expiration and refreshes using `yt_channel_token.json` + `client_secrets.json`. If the refresh fails, the pipeline writes a failure and moves on.

**Shorts validation** (`upload.py:67-100`):
- Must be vertical (width < height)
- Must be ≤ 180s (configurable)
- Uses ffprobe to verify dimensions and duration before uploading

---

## 7. Stage 9 — Self-Learning

**File:** `automation/orchestrator.py:492-670`

Stage 9 runs **after** all exports and uploads. It contains 5 parallel sub-stages:

### 7a. Analytics (`automation/seo/analytics.py`)

```python
a = Analytics(_DECISION_STORE)
summary = a.get_summary()
```

Reads all `ClipEvent`s from the decision store and computes:
- `total_clips` — unique clip IDs
- `total_events` — all events ever recorded
- `published_count` — number of published events
- `avg_score` — average candidate score across all scored clips

Why: Lightweight health check. No external API calls, purely from the local event store.

### 7b. Self-Learner (`self_learner/` module)

```python
from self_learner import Learner, SEOLearner, TrendAnalyzer, RecommendationEngine

learner = Learner()              # PersistentMemory — SQLite KV store
seo_learner = SEOLearner()       # Performance DB — seo_performance.json
trend_analyzer = TrendAnalyzer() # Trend point tracking
rec_engine = RecommendationEngine(seo_learner, trend_analyzer)
```

Records:
- **Pipeline observation**: `learner.observe("pipeline_run", {...})` — stores duration, exported count, failures, etc.
- **Trend metrics**: `trend_analyzer.record_metric(TrendPoint(...))` — tracks duration, exported_count, failures_count, selected_clips over time
- **Anomaly detection**: detects sudden spikes in duration or failures

### 7c. SEOLearner (Performance DB)

See [Section 8](#8-seolearner-deep-dive) below.

### 7d. Automation Learner (PolicyUpdater)

```python
automation_learner = AutomationLearner(_DECISION_STORE, _LEARNED_STATE)
updater = PolicyUpdater(automation_learner)
updater.update_from_events(_DECISION_STORE.get_all_events())
replay_engine = ReplayEngine(...)
if not replay_engine.verify():
    replay_engine.replay()
```

This is the **event-driven policy system**:
- Reads all `ClipEvent`s from the decision store
- Updates learned state (preferred duration, pacing, hook weight, payoff weight)
- Replays if state divergence is detected (ensures consistency)
- Outputs preferences like `{"preferred_duration": "medium", "preferred_pacing": "normal", ...}`

### 7e. Cricket Learning Engine

```python
FormatLearner    — learns which clip formats perform best
EntityLearner    — learns player/team performance
TrendEngine      — tracks topic trends over time
TimingLearner    — learns best upload times
DurationLearner  — learns optimal clip durations
CricketScorer    — recalibrates scoring weights from metrics
CricketRecommendationEngine — generates actionable recommendations
```

This is the **deep cricket-specific learning**. Every 20 `metrics_received` events, the scorer recalibrates its weights.

### 7f. Provider Health

```python
for provider in ("transcript", "download", "transcriber", "llm",
                 "export", "enhancement", "drive", "youtube"):
    stats = _PROVIDER_HEALTH.get_stats(provider)
    # status: HEALTHY / DEGRADED / DOWN
```

Tracks success/failure counts for every external provider used during this run. Logs status for diagnostics.

---

## 8. SEOLearner Deep Dive

**File:** `automation/seo/seo_learner.py` (955 lines)

### 8a. Performance Recording

`record_performance()` is called from `process_all_seo()` after each clip's SEO is generated:

```python
seo_learner.record_seo_outcome(SEOPerformance(
    clip_id=clip_id,
    title=...,
    description=...,
    hashtags=...,
    tags=...,
    is_shorts=True,
    provider=...,
    model=...,
    upload_success=True/False,
))
```

Data is stored in `data/seo_performance.json` (append-only, versioned, max 100 clips).

### 8b. Feature Extraction

```python
def _extract_seo_features(title, description, hashtags):
    return {
        "title_length": len(title),
        "has_pipe_format": "|" in title and "vs" in title,
        "has_power_word": any power word in title,
        "has_emoji": emoji in title,
        "has_player_name": regex name pattern,
        "has_score": match score pattern,
        "has_number": any digit,
        "title_words": word count,
        "description_length": len(description),
        "has_sections": structured description,
        "has_cta": like/subscribe/share,
        "hashtag_count_bin": "0" / "1-3" / "4-5" / "6-10" / "10+",
        "has_shorts_hashtag": #Shorts,
        "has_ipl_hashtag": #IPL,
        "has_team_hashtag": team abbreviation,
        "starts_with_live_emoji": 🔴,
        "has_multiple_pipes": 2+ pipe chars,
    }
```

Why extract these specific features? Each has a hypothesized impact on CTR/performance:
- **Pipe format**: "Team vs Team | Tournament" is a proven cricket Shorts format
- **Power words**: "smashes", "destroys" drive emotional engagement
- **Emoji**: increases visual appeal in the feed
- **Player name**: personalizes the click
- **Hashtag bin**: too many hashtags look spammy

### 8c. Time-Decay Weighting

```python
def _time_decay_weight(timestamp_str):
    days_old = (now - ts).total_seconds() / 86400
    return math.exp(-math.log(2) * days_old / DECAY_HALF_LIFE_DAYS)
```

Default half-life: **30 days**. A clip from 30 days ago has 0.5 weight, from 60 days ago has 0.25 weight. This ensures recent performance matters more than old history.

### 8d. Pattern Learning

Patterns are keyed by a **stable pattern key** from low-cardinality features:

```python
def _stable_pattern_key(features):
    parts = []
    for k in sorted(features.keys()):
        v = features[k]
        if isinstance(v, bool):
            parts.append(f"{k}:{'true' if v else 'false'}")
        elif isinstance(v, str):
            parts.append(f"{k}:{v}")
    return "_".join(parts)
```

**Critical insight:** Numeric features (title_length, description_length) are intentionally excluded from the pattern key. Including them made every single clip a unique pattern, so patterns never reached `MIN_CLIPS_FOR_PATTERN` (2). By using only boolean/categorical features, patterns repeat across clips and learning actually happens.

Patterns track:
- `count` — how many clips match this pattern
- `avg_score` — time-decayed weighted average performance
- `scores` — recent score history (last 50)

### 8e. Auto-Benchmark

```python
def run_auto_benchmark(self):
```

Runs a synthetic benchmark (not real clip data) on a hardcoded prompt about "RCB vs CSK IPL 2026" on every available model. Scores each model on:
- **JSON validity**: 10 pts
- **Title quality**: 10 pts for length, 15 pts for player name match, 5 pts for "IPL"
- **Description quality**: 10 pts for length, 15 pts for no dict-in-string
- **Hashtags**: 10 pts for 3+ hashtags
- **Search terms**: 10 pts for 5+ terms
- **Latency penalty**: -20 if >30s, -10 if >15s
- **Power words**: 5 pts
- **Pipe format**: 5 pts

Max score: 100. Results are stored in `benchmark_history` array.

**Best model selection** uses real performance data first, benchmark only as cold-start:

```python
def _recompute_best_model(self):
    # 1. Real model_performance with count >= MIN_CLIPS_FOR_PATTERN
    # 2. Benchmark cold-start fallback (only if score >= 40)
```

### 8f. Prompt Enhancement

```python
def update_prompt_with_learnings(self, base_prompt):
```

Returns a prompt with added sections:
- ✅ Features that boost performance (delta > +3%)
- ❌ Features that hurt performance (delta < -3%)
- 📈 Trending UP patterns
- 🧠 AI Analysis (if available)
- 📋 Recommendations

This is called by the SEO module to inject channel-specific intelligence into future prompts.

---

## 9. AI Client & The Racer

**File:** `utils/ai_client.py` (609 lines)

### 9a. Provider Chain

Three providers with their models:

| Provider | Models | API Key Env Var |
|----------|--------|----------------|
| `opencode` | mimo-v2.5-pro, kimi-k2.5, glm-5, deepseek-v4-pro, qwen3.7-max, etc. | `OPENCODE_ZEN_API_KEY` |
| `nvidia` | llama-3.3-nemotron-super-49b-v1, meta/llama-3.3-70b-instruct, llama-3.3-nemotron-super-49b-v1.5 | `NVIDIA_API_KEY` |
| `groq` | llama-3.3-70b-versatile, meta-llama/llama-4-scout-17b-16e-instruct | `GROQ_API_KEY` |

Failover chain order: `[current_provider, opencode, nvidia, groq]`

### 9b. Token Bucket Rate Limiting

Each provider has a token bucket:

| Provider | Capacity | Refill Rate | Why |
|----------|----------|-------------|-----|
| opencode | 30 | 0.5/s | Generous, Go subscription |
| nvidia | 30 | 0.5/s | Generous |
| groq | 10 | 0.15/s | Strict (~6K TPM free tier) |

The racer calls `_check_and_consume_token()` before submitting each model. If a bucket is empty, that model is skipped for this race. This prevents overwhelming provider APIs.

### 9c. Circuit Breaker & Cooldown

After **5 consecutive failures**, a provider enters cooldown for 300 seconds:

```python
cls._provider_cooldown_until[provider] = time.time() + 300
```

Different error types get different cooldowns:

| Error | Cooldown |
|-------|----------|
| Auth failure | 3600s (1 hour) |
| Quota exhausted | 3600s (1 hour) |
| Rate limited | Retry-After header or 30s |
| Model not found | 60s |
| Timeout / Server | 15s |

### 9d. Error Classification

```python
class ErrorCategory:
    RATE_LIMIT       # 429
    QUOTA_EXHAUSTED  # billing/payment
    MODEL_NOT_FOUND  # 404
    TIMEOUT          # 504
    SERVER_ERROR     # 5xx
    AUTH_FAILURE     # 401/403
    UNKNOWN          # anything else
```

Classification determines:
1. Whether to retry (transient errors only)
2. How long to wait before retry
3. Whether to skip the provider entirely (auth/quota)

---

## 10. Why Architecture Is This Way

### Why parallel racers instead of a single model call?

SEO generation is **not latency-critical** — it runs inside a batch pipeline, not a user-facing request. However, 6 clips × 5 models sequentially would take 5-10 minutes per clip. By racing, the system gets the fastest valid response (typically Groq at ~500ms) while still having the option to wait for more capable but slower models (NVIDIA nemotron) if Groq fails.

### Why 5 models racing simultaneously?

After empirical testing: 2 models was too few (both might fail), 10+ was wasteful (responses are largely redundant). 5 covers diverse providers (Groq, NVIDIA, OpenCode) and ensures at least one returns within seconds.

### Why Hinglish titles specifically?

YouTube's algorithm for Indian cricket content rewards Hinglish (Hindi words in Roman script) over pure English or pure Devanagari. Hinglish titles get higher CTR because:
1. They're searchable in both English and Hindi queries
2. They're readable by international audiences
3. Devanagari titles get suppressed in non-Indian search results

### Why a separate SEOLearner performance DB instead of using the event store?

SEO performance tracking needs different data shapes than the event store. The event store records discrete events (clip_exported, uploaded, etc.) while the SEOLearner needs aggregated analytics (views, likes, retention) keyed by clip_id, with time-decay weighting and pattern analysis. Keeping them separate avoids polluting the event store with YouTube Analytics data.

### Why quality gate before upload?

YouTube doesn't penalize bad metadata — it just ignores it. But generic SEO (like "Cricket Highlights" as a title) actively hurts channel performance by signaling to YouTube that the content is low-effort. The quality gate prevents any SEO that would damage the channel's authority score.

### Why time-decay weighting in the learner?

A cricket channel's audience changes over time. What worked in IPL 2025 might not work in Champions Trophy 2026. Without decay, old performance would dominate learning. With 30-day half-life, the learner adapts to what's working RIGHT NOW.
