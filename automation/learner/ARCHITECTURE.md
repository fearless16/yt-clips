# Cricket Shorts Self-Learning Engine — Architecture v4.0

## 0. Channel Reality (Ground Truth from 121 Shorts)

Before designing anything, here is what your data actually says:

| Signal | Finding | Implication |
|--------|---------|-------------|
| **Reaction hooks** | avg 552 views (n=7) | 2x baseline — under-used |
| **Fan war/comedy** | avg 434 views (n=7) | 1.5x baseline — under-used |
| **Bumrah** | avg 1285 views (n=3) | 4.4x baseline — massively under-covered |
| **MI** | avg 460 views (n=14) | Best team, already decent coverage |
| **KKR** | avg 68 views (n=4) | Poison — avoid |
| **Questions in title** | avg 182 views | **HURTS** by 43% vs no-question (321) |
| **Hinglish** | avg 313 vs English 249 | +26% lift |
| **Long titles (60+)** | avg 365 vs medium 224 | +63% lift |
| **Jadeja** | 17 videos, avg 233 | Over-covered, diminishing returns |
| **GT** | 28 videos, avg 301 | Over-covered |
| **Prediction hooks** | avg 83 views (n=4) | Worst hook type |

**The current Learner adjusts `hook_weight ± 0.1` — approximately 12 bits of total learning capacity. It cannot see any of this.**

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PIPELINE (orchestrator.py)                    │
│  download → transcribe → highlight → export → seo → sync → upload   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ emit ClipEvents
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     EVENT BUS (append-only)                          │
│  candidate_created → candidate_scored → candidate_ranked →          │
│  selected → rejected → exported → published → metrics_received      │
│  → manual_override → policy_updated → trend_ingested                │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
              ┌────────────┼────────────────┐
              ▼            ▼                ▼
┌──────────────────┐ ┌──────────┐ ┌──────────────────┐
│  DecisionStore   │ │Processed │ │  LearnedState     │
│  (append-only)   │ │EventLog  │ │  Store (derived)  │
│                  │ │(idempot.)│ │                   │
└────────┬─────────┘ └────┬─────┘ └────────┬──────────┘
         │                │                 │
         ▼                ▼                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     LEARNING DISPATCHER                              │
│  Routes events to specialized learners. Checks ProcessedEventLog    │
│  for idempotency. Emits policy_updated when state changes.          │
└───┬──────────┬──────────┬──────────┬──────────┬─────────────────────┘
    │          │          │          │          │
    ▼          ▼          ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│Format  │ │Trend   │ │Entity  │ │Timing  │ │Duration│
│Learner │ │Engine  │ │Learner │ │Learner │ │Learner │
│(40%)   │ │(30%)   │ │(20%)   │ │(10%)   │ │        │
└───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘
    │          │          │          │          │
    └──────────┴──────────┴──────────┴──────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      SCORING ENGINE                                  │
│  FinalScore = Σ(weight_i × score_i) for each candidate              │
│  Weights adapt from learned state. Thompson Sampling for explore.   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   RECOMMENDATION ENGINE                              │
│  Next topic, player, hook, title style, duration, upload time       │
│  Each recommendation includes WHY (traceable to learned state)      │
└─────────────────────────────────────────────────────────────────────┘
```

**Key architectural decisions:**
- **Learning Dispatcher** replaces the monolithic `Learner` + `PolicyUpdater`. One router, five specialists.
- **ProcessedEventLog** is a `set[event_id]` that prevents double-processing. This is the idempotency layer.
- **self_learner (SQLite)** and **automation/learner (in-memory)** merge into one system. The SQLite DB becomes the persistent backing store for `LearnedStateStore`.
- **Trend Engine** is a first-class component, not an afterthought.

---

## 2. Event Schema Design

### 2.1 EventType Enum (Extended)

```python
class EventType(str, Enum):
    # Existing (KEEP)
    candidate_created = "candidate_created"
    candidate_scored = "candidate_scored"
    candidate_ranked = "candidate_ranked"
    selected = "selected"
    rejected = "rejected"
    exported = "exported"
    published = "published"
    metrics_received = "metrics_received"
    manual_override = "manual_override"
    validation_failed = "validation_failed"
    infra_failed = "infra_failed"
    deferred = "deferred"
    policy_updated = "policy_updated"

    # NEW
    trend_ingested = "trend_ingested"
    trend_decayed = "trend_decayed"
```

### 2.2 Payload Schemas (per event type)

Every payload is a JSON dict stored in `payload_json`. Here are the schemas:

**`metrics_received`** — the most important event. This is where real learning happens.

```python
{
    # Core metrics (from YouTube Analytics API)
    "views": 870,
    "likes": 4,
    "comments": 1,
    "shares": 0,
    "impressions": 15000,
    "ctr": 0.058,
    "avg_watch_percentage": 0.72,
    "completion_rate": 0.45,
    "rewatch_rate": 0.08,
    "subscribers_gained": 2,
    "views_per_hour": 36.2,
    "duration_seconds": 29,

    # Content metadata (from pipeline at publish time)
    "publish_hour": 19,
    "publish_dow": 3,
    "topic_tags": ["ipl_2026", "gt_vs_rr"],
    "player_tags": ["jadeja", "sundar"],
    "team_tags": ["gt", "rr"],
    "format_tags": ["ipl", "t20"],
    "hook_type": "shock",
    "title_pattern": {
        "length": 62,
        "has_emoji": True,
        "has_caps": True,
        "has_question": False,
        "has_pipe": True,
        "is_hinglish": True,
        "word_count": 9
    },

    # Derived signals (computed by dispatcher)
    "engagement_rate": 0.57,
    "velocity_score": 36.2,
    "quality_score": 0.64
}
```

**`trend_ingested`** — cricket trend from external source.

```python
{
    "trend_id": "ind_vs_afg_test_2026",
    "source": "google_trends",
    "query": "IND vs AFG",
    "trend_score": 0.95,
    "velocity": 0.82,
    "category": "match",
    "entities": {
        "players": ["shubman_gill", "jasprit_bumrah"],
        "teams": ["india", "afghanistan"],
        "series": "test_cricket",
        "format": "test"
    },
    "expires_at": "2026-06-11T00:00:00Z",
    "half_life_hours": 72
}
```

**`candidate_created`** — clip candidate from highlight detection.

```python
{
    "source_video_id": "5ChdXWQakpA",
    "segment_start": 1245.0,
    "segment_end": 1274.0,
    "duration_seconds": 29,
    "transcript_snippet": "Kohli ne maara six! Crowd goes wild",
    "detected_players": ["kohli"],
    "detected_teams": ["india"],
    "detected_hook_type": "shock",
    "audio_energy": 0.87,
    "speech_rate_wpm": 165
}
```

**`candidate_scored`** — 7-dimension score from highlight.py.

```python
{
    "hook_strength": 8.2,
    "clarity": 7.1,
    "emotional_peak": 9.0,
    "topic_completeness": 6.5,
    "punchline_or_payoff": 7.8,
    "cut_safety": 8.0,
    "replay_value": 7.5,
    "weighted_score": 7.67
}
```

**`selected` / `rejected`** — final selection decision.

```python
{
    "reason": "top_k_selection",
    "rank": 3,
    "competing_candidates": 12,
    "score_delta_from_cutoff": 1.2
}
```

**`published`** — upload confirmation.

```python
{
    "youtube_video_id": "gTx8i2Vpup4",
    "title": "2 in 2 ho sakta hai? PSG jaisa kamaal!",
    "description": "...",
    "hashtags": ["#Shorts", "#IPL2026"],
    "tags": ["ipl", "cricket", "rcb", "gt"],
    "publish_time": "2026-06-02T19:30:00+05:30",
    "scheduled": False
}
```

---

## 3. Database Schema

### 3.1 Persistent Backing Store (SQLite — replaces in-memory `LearnedStateStore`)

```sql
-- Append-only event log (replaces in-memory DecisionStore._events list)
CREATE TABLE IF NOT EXISTS event_log (
    event_id    TEXT PRIMARY KEY,
    clip_id     TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    payload     TEXT NOT NULL,
    processed   INTEGER DEFAULT 0
);
CREATE INDEX idx_event_clip ON event_log(clip_id);
CREATE INDEX idx_event_type ON event_log(event_type);
CREATE INDEX idx_event_ts ON event_log(timestamp);

-- Derived learned state (replaces in-memory LearnedStateStore._state dict)
CREATE TABLE IF NOT EXISTS learned_state (
    state_key       TEXT PRIMARY KEY,
    value_json      TEXT NOT NULL,
    derived_from    TEXT,
    updated_at      TEXT NOT NULL,
    version         INTEGER DEFAULT 1
);

-- Processed event IDs (idempotency guard)
CREATE TABLE IF NOT EXISTS processed_events (
    event_id    TEXT PRIMARY KEY,
    learner     TEXT NOT NULL,
    processed_at TEXT NOT NULL
);
CREATE INDEX idx_proc_learner ON processed_events(learner);

-- Trend store (time-series with decay)
CREATE TABLE IF NOT EXISTS trends (
    trend_id        TEXT PRIMARY KEY,
    query           TEXT NOT NULL,
    source          TEXT NOT NULL,
    category        TEXT NOT NULL,
    initial_score   REAL NOT NULL,
    half_life_hours REAL NOT NULL DEFAULT 72.0,
    created_at      TEXT NOT NULL,
    expires_at      TEXT,
    entities_json   TEXT DEFAULT '{}'
);
CREATE INDEX idx_trend_category ON trends(category);
CREATE INDEX idx_trend_expires ON trends(expires_at);
```

### 3.2 Migration from Current System

The current `self_learner.db` (486 rows in `memories` table) maps directly:

| Current key pattern | New location |
|---------------------|-------------|
| `fact:upload:was_observed:*` | `event_log` (as `published` events) |
| `fact:seo_outcome:has_performance:*` | `event_log` (as `metrics_received` events) |
| `fact:trend:*:has_data_point:*` | `learned_state` (aggregated) |
| `fact:upload:is_pattern:*` | `learned_state` (as `format_patterns`) |

---

## 4. Learned State Design

### 4.1 State Keys and Schemas

All values stored as JSON in `learned_state.value_json`.

**`format_scores`** — hook type and title pattern performance.

```json
{
    "hook_types": {
        "reaction":  {"score": 0.88, "n": 7,  "avg_views": 552, "avg_engagement": 0.18, "last_updated": "2026-06-04"},
        "fan_war":   {"score": 0.82, "n": 7,  "avg_views": 434, "avg_engagement": 0.22, "last_updated": "2026-06-04"},
        "shock":     {"score": 0.55, "n": 28, "avg_views": 205, "avg_engagement": 0.08, "last_updated": "2026-06-04"},
        "live":      {"score": 0.52, "n": 35, "avg_views": 229, "avg_engagement": 0.07, "last_updated": "2026-06-04"},
        "debate":    {"score": 0.50, "n": 14, "avg_views": 205, "avg_engagement": 0.09, "last_updated": "2026-06-04"},
        "question":  {"score": 0.35, "n": 27, "avg_views": 182, "avg_engagement": 0.06, "last_updated": "2026-06-04"},
        "prediction":{"score": 0.15, "n": 4,  "avg_views": 83,  "avg_engagement": 0.04, "last_updated": "2026-06-04"}
    },
    "title_patterns": {
        "long_title":    {"score": 0.78, "n": 55, "avg_views": 365},
        "hinglish":      {"score": 0.72, "n": 77, "avg_views": 313},
        "has_caps":      {"score": 0.58, "n": 89, "avg_views": 298},
        "has_emoji":     {"score": 0.56, "n": 87, "avg_views": 295},
        "has_pipe":      {"score": 0.54, "n": 48, "avg_views": 310},
        "no_question":   {"score": 0.70, "n": 94, "avg_views": 321},
        "has_question":  {"score": 0.35, "n": 27, "avg_views": 182}
    }
}
```

**`entity_scores`** — player, team, series, format performance.

```json
{
    "players": {
        "bumrah":      {"score": 0.95, "n": 3,  "avg_views": 1285, "trend": "rising"},
        "kl_rahul":    {"score": 0.88, "n": 1,  "avg_views": 1100, "trend": "stable"},
        "surya":       {"score": 0.87, "n": 3,  "avg_views": 1096, "trend": "rising"},
        "arshdeep":    {"score": 0.65, "n": 1,  "avg_views": 402,  "trend": "stable"},
        "dhoni":       {"score": 0.55, "n": 3,  "avg_views": 284,  "trend": "stable"},
        "jadeja":      {"score": 0.48, "n": 17, "avg_views": 233,  "trend": "declining"},
        "rohit":       {"score": 0.47, "n": 7,  "avg_views": 231,  "trend": "stable"},
        "kohli":       {"score": 0.38, "n": 5,  "avg_views": 176,  "trend": "stable"}
    },
    "teams": {
        "mi":   {"score": 0.82, "n": 14, "avg_views": 460},
        "pbks": {"score": 0.72, "n": 10, "avg_views": 376},
        "gt":   {"score": 0.58, "n": 28, "avg_views": 301},
        "rr":   {"score": 0.56, "n": 19, "avg_views": 295},
        "csk":  {"score": 0.50, "n": 11, "avg_views": 245},
        "kkr":  {"score": 0.12, "n": 4,  "avg_views": 68}
    },
    "series": {
        "ipl_2026":    {"score": 0.70, "n": 65, "avg_views": 333},
        "test_cricket":{"score": 0.45, "n": 3,  "avg_views": 210}
    },
    "formats": {
        "t20":  {"score": 0.72, "n": 80, "avg_views": 320},
        "odi":  {"score": 0.40, "n": 5,  "avg_views": 180},
        "test": {"score": 0.35, "n": 3,  "avg_views": 210}
    }
}
```

**`timing_scores`** — upload hour and day performance.

```json
{
    "best_hours": {
        "19": {"score": 0.82, "n": 15, "avg_views": 420},
        "20": {"score": 0.78, "n": 12, "avg_views": 390},
        "14": {"score": 0.55, "n": 8,  "avg_views": 250}
    },
    "best_days": {
        "saturday":  {"score": 0.75, "n": 18, "avg_views": 380},
        "sunday":    {"score": 0.72, "n": 16, "avg_views": 360},
        "friday":    {"score": 0.60, "n": 14, "avg_views": 290}
    },
    "match_phases": {
        "post_match":     {"score": 0.80, "n": 25},
        "during_innings": {"score": 0.65, "n": 40},
        "pre_toss":       {"score": 0.45, "n": 5}
    }
}
```

**`duration_scores`** — optimal clip length.

```json
{
    "buckets": {
        "15-24s": {"score": 0.60, "n": 20, "avg_completion": 0.72, "avg_views": 280},
        "25-34s": {"score": 0.75, "n": 45, "avg_completion": 0.58, "avg_views": 310},
        "35-44s": {"score": 0.68, "n": 30, "avg_completion": 0.45, "avg_views": 290},
        "45-59s": {"score": 0.50, "n": 15, "avg_completion": 0.35, "avg_views": 240}
    },
    "sweet_spot": "25-34s"
}
```

**`trend_scores`** — active cricket trends with decay.

```json
{
    "active_trends": {
        "ipl_2026_final": {
            "score": 0.92,
            "initial_score": 0.95,
            "velocity": 0.85,
            "created_at": "2026-06-01",
            "half_life_hours": 72,
            "current_score": 0.78,
            "entities": ["rcb", "gt", "kohli", "gill"]
        },
        "bumrah_vaibhav": {
            "score": 0.88,
            "initial_score": 0.90,
            "velocity": 0.92,
            "created_at": "2026-06-03",
            "half_life_hours": 48,
            "current_score": 0.82,
            "entities": ["bumrah", "vaibhav_suryavanshi"]
        }
    }
}
```

**`scoring_weights`** — adaptive weights for the scoring formula.

```json
{
    "trend_weight": 0.30,
    "format_weight": 0.25,
    "entity_weight": 0.20,
    "historical_weight": 0.15,
    "timing_weight": 0.10,
    "exploration_rate": 0.15,
    "last_recalibrated": "2026-06-04"
}
```

---

## 5. Trend Engine Design

### 5.1 Decay Model

Cricket trends follow a **half-life decay** model. Different trend categories have different half-lives:

| Category | Half-Life | Example |
|----------|-----------|---------|
| Live match moment | 4-8 hours | "Kohli six last ball" |
| Match result | 24-48 hours | "IND vs AFG day 1" |
| Series/tournament | 72-168 hours | "IPL 2026 Final" |
| Selection/controversy | 120-240 hours | "Sanju Samson dropped" |
| Evergreen player | 336+ hours | "Dhoni retirement debate" |

**Decay formula:**

```
current_score = initial_score × 2^(-elapsed_hours / half_life_hours)
```

Example: IND vs AFG with initial_score=0.95, half_life=72h:
- After 24h: 0.95 × 2^(-24/72) = 0.95 × 0.794 = **0.754**
- After 72h: 0.95 × 2^(-72/72) = 0.95 × 0.5 = **0.475**
- After 168h (7 days): 0.95 × 2^(-168/72) = 0.95 × 0.198 = **0.188**
- After 336h (14 days): 0.95 × 2^(-336/72) = 0.95 × 0.039 = **0.037**

### 5.2 Velocity Boost

Trends that are accelerating get a boost:

```
velocity = (current_impressions - previous_impressions) / previous_impressions
velocity_boost = min(velocity × 0.3, 0.25)
effective_score = current_score + velocity_boost
```

### 5.3 Trend Ingestion Sources

| Source | Method | Frequency |
|--------|--------|-----------|
| YouTube Search Suggestions | `youtube-search-suggest` API | Every 4 hours |
| Google Trends (India) | RSS/JSON feed | Every 6 hours |
| Cricket fixtures | ESPN Cricinfo / hardcoded schedule | Daily |
| Channel's own upload performance | `metrics_received` events | Per upload |

### 5.4 Python Class

```python
class TrendEngine:
    def __init__(self, state_store: PersistentStateStore):
        self._state = state_store

    def ingest(self, trend: TrendInput) -> None:
        """Add or update a trend. Idempotent by trend_id."""

    def decay_all(self) -> int:
        """Recompute current_score for all active trends. Returns count updated."""

    def get_active(self, min_score: float = 0.10) -> list[TrendState]:
        """Return trends above threshold, sorted by effective_score desc."""

    def get_trend_for_entities(self, players: list[str],
                                teams: list[str]) -> float:
        """Max trend score matching any of the given entities."""

    def prune_expired(self) -> int:
        """Remove trends where current_score < 0.05."""

    def get_hot_topics(self, limit: int = 5) -> list[dict]:
        """Top trending topics with entity context for recommendation."""
```

---

## 6. Formatting Learner Design (40% weight)

This is the most important learner. It tracks what **presentation patterns** drive views.

### 6.1 What It Tracks

**Hook types** (detected from title + transcript first 3 seconds):

| Hook Type | Detection Pattern | Your Data |
|-----------|-------------------|-----------|
| `reaction` | "reaction", "react", "response" in title | avg 552 views |
| `fan_war` | "fan war", "troll", "meme", "comedy" | avg 434 views |
| `shock` | "unbelievable", "insane", "shocking", "kya" | avg 205 views |
| `live` | "live" in title | avg 229 views |
| `debate` | "was X wrong", "sahi ya galat", "should" | avg 205 views |
| `question` | title ends with "?" | avg 182 views |
| `prediction` | "will", "about to", "hoga" | avg 83 views |

**Title features** (binary/numeric):

| Feature | Your Data Impact |
|---------|-----------------|
| `long_title` (60+ chars) | +63% views |
| `hinglish` | +26% views |
| `no_question` | +76% views |
| `has_pipe` | +13% views |
| `has_caps` | marginal |
| `has_emoji` | marginal |

### 6.2 Update Algorithm: Bayesian EMA

Each formatting pattern maintains a **Bayesian posterior** using Exponential Moving Average:

```python
# On metrics_received for a clip with hook_type="reaction":
alpha = 0.15  # learning rate — higher = faster adaptation

old_score = format_scores["hook_types"]["reaction"]["score"]
new_signal = normalize(views, channel_baseline=290)  # 552/290 = 1.90 → capped at 1.0

format_scores["hook_types"]["reaction"]["score"] = (
    (1 - alpha) * old_score + alpha * new_signal
)
format_scores["hook_types"]["reaction"]["n"] += 1
```

**Why Bayesian EMA, not simple average:**
- Simple average gives equal weight to a 3-month-old video and yesterday's. Cricket content relevance decays.
- EMA with α=0.15 means the last ~7 observations carry 67% of the weight. This matches your upload cadence (~1.5/day).
- The `normalize()` step converts raw views to a 0-1 signal relative to channel baseline.

### 6.3 Anti-Pattern: Feedback Loop Prevention

If the system only recommends "reaction" hooks because they score highest, it never tests other hooks. Solution: **Thompson Sampling** with exploration rate ε=0.15.

```python
def select_hook_type(self, format_scores: dict) -> str:
    if random.random() < self.exploration_rate:
        return random.choice(list(format_scores.keys()))

    # Thompson Sampling: sample from Beta posterior for each hook type
    best_type = None
    best_sample = -1
    for hook_type, data in format_scores.items():
        alpha = data["score"] * data["n"] + 1
        beta = (1 - data["score"]) * data["n"] + 1
        sample = np.random.beta(alpha, beta)
        if sample > best_sample:
            best_sample = sample
            best_type = hook_type
    return best_type
```

---

## 7. Cricket Entity Learner Design (20% weight)

### 7.1 Entity Extraction

From title + transcript, extract:

```python
CRICKET_ENTITIES = {
    "players": {
        "kohli": ["kohli", "virat", "king kohli"],
        "bumrah": ["bumrah", "jasprit"],
        "rohit": ["rohit", "hitman", "ro"],
        "gill": ["gill", "shubman"],
        "dhoni": ["dhoni", "msd", "thala", "mahendra"],
        "jadeja": ["jadeja", "sir jadeja", "jaddu"],
        "surya": ["surya", "suryakumar", "sky"],
        "samson": ["samson", "sanju"],
        "hardik": ["hardik", "pandya"],
        "rahul": ["kl rahul", "rahul"],
        "pant": ["pant", "rishabh"],
        "arshdeep": ["arshdeep"],
        "siraj": ["siraj"],
        "ashwin": ["ashwin"],
        "vaibhav": ["vaibhav", "suryavanshi"],
    },
    "teams": {
        "mi": ["mi", "mumbai indians", "mumbai"],
        "rcb": ["rcb", "royal challengers", "bangalore"],
        "csk": ["csk", "chennai super", "chennai"],
        "gt": ["gt", "gujarat titans", "gujarat"],
        "kkr": ["kkr", "kolkata"],
        "dc": ["dc", "delhi capitals", "delhi"],
        "rr": ["rr", "rajasthan royals", "rajasthan"],
        "pbks": ["pbks", "punjab kings", "punjab"],
        "srh": ["srh", "sunrisers", "hyderabad"],
        "lsg": ["lsg", "lucknow"],
    },
    "series": {
        "ipl": ["ipl", "indian premier league"],
        "world_cup": ["world cup", "wc"],
        "asia_cup": ["asia cup"],
        "test": ["test cricket", "test match", "test series"],
        "odi": ["odi", "one day"],
        "t20i": ["t20i", "t20 international"],
    }
}
```

### 7.2 Scoring Algorithm

Entity scores use **confidence-weighted EMA** — entities with more data points get slower learning rates:

```python
def update_entity_score(self, entity: str, views: int, baseline: int = 290):
    current = self._state.get_entity(entity)
    n = current["n"]

    # Confidence-weighted alpha: more data = slower change
    # n=1: alpha=0.30, n=5: alpha=0.18, n=20: alpha=0.12
    alpha = max(0.10, 0.30 / (1 + n * 0.05))

    signal = min(views / baseline, 3.0) / 3.0  # normalize to 0-1, cap at 3x
    new_score = (1 - alpha) * current["score"] + alpha * signal

    # Trend detection: compare recent 3 vs historical
    recent_avg = current.get("recent_views_avg", views)
    trend = "rising" if views > recent_avg * 1.3 else (
            "declining" if views < recent_avg * 0.7 else "stable")

    self._state.set_entity(entity, {
        "score": new_score,
        "n": n + 1,
        "avg_views": (current["avg_views"] * n + views) / (n + 1),
        "trend": trend,
        "recent_views_avg": (recent_avg * min(n, 3) + views) / (min(n, 3) + 1),
    })
```

### 7.3 Fatigue Penalty

Entities covered too frequently without proportional performance get a fatigue penalty:

```python
def fatigue_penalty(self, entity: str, window_days: int = 14) -> float:
    recent_uploads = self._state.count_recent(entity, window_days)
    avg_performance = self._state.get_entity(entity)["score"]

    if recent_uploads > 10 and avg_performance < 0.50:
        return min(0.30, (recent_uploads - 10) * 0.03)
    return 0.0
```

**Your data example:** Jadeja has 17 videos at avg 233 views (score 0.48). After 10 uploads, fatigue penalty kicks in: `(17-10) × 0.03 = 0.21`. Effective score = `0.48 - 0.21 = 0.27`. System recommends reducing Jadeja coverage.

---

## 8. Timing Learner Design (10% weight)

### 8.1 What It Tracks

```python
class TimingLearner:
    def __init__(self, state_store: PersistentStateStore):
        self._state = state_store

    def record(self, publish_hour: int, publish_dow: int,
               match_phase: str, views: int) -> None:
        """Update hour/day/match_phase scores from a metrics_received event."""

    def best_hour(self, day: int | None = None) -> int:
        """Return optimal upload hour, optionally filtered by day."""

    def best_day(self) -> str:
        """Return optimal day of week."""

    def best_match_phase(self) -> str:
        """Return optimal match phase for upload."""

    def get_schedule_recommendation(self) -> dict:
        """Return {hour, day, match_phase, confidence, reason}."""
```

### 8.2 Update Algorithm

Simple count-weighted average with recency bias:

```python
def record(self, hour, dow, match_phase, views):
    baseline = 290
    signal = min(views / baseline, 3.0) / 3.0

    for key_prefix, value in [("hour", hour), ("dow", dow), ("phase", match_phase)]:
        state_key = f"timing:{key_prefix}:{value}"
        current = self._state.get(state_key, {"score": 0.5, "n": 0})
        n = current["n"]
        alpha = max(0.10, 0.25 / (1 + n * 0.08))
        new_score = (1 - alpha) * current["score"] + alpha * signal
        self._state.set(state_key, {
            "score": new_score,
            "n": n + 1,
            "avg_views": (current.get("avg_views", views) * n + views) / (n + 1),
        })
```

**Why 10% weight, not more:** Upload timing matters but your data shows content quality (hook type, player, team) has 3-5x more impact than timing. A Bumrah reaction video at 3 AM will outperform a Jadeja question video at prime time.

---

## 9. Scoring Formula

### 9.1 Candidate Scoring (for clip selection)

```
FinalScore = (
    0.30 × TrendScore +
    0.25 × FormatScore +
    0.20 × EntityScore +
    0.15 × HistoricalScore +
    0.10 × TimingScore
) × (1 - FatiguePenalty)
```

**Component breakdown:**

| Component | Weight | Source | Calculation |
|-----------|--------|--------|-------------|
| **TrendScore** | 0.30 | TrendEngine | `max(trend_engine.get_trend_for_entities(players, teams), 0.1)` |
| **FormatScore** | 0.25 | FormatLearner | `hook_type_score × title_pattern_score` |
| **EntityScore** | 0.20 | EntityLearner | `max(player_scores) × 0.5 + max(team_scores) × 0.3 + series_score × 0.2` |
| **HistoricalScore** | 0.15 | DecisionStore | `clip_scorer.weighted_score / 10.0` (the 7-dimension score from highlight.py) |
| **TimingScore** | 0.10 | TimingLearner | `timing_learner.get_schedule_recommendation()["confidence"]` |
| **FatiguePenalty** | subtractive | EntityLearner | `max(fatigue_penalty(p) for p in players)` |

### 9.2 Why These Weights

| Weight | Rationale |
|--------|-----------|
| **Trend 0.30** | Your top performers (1700, 1600, 1500 views) are ALL tied to trending moments: Bumrah vs Vaibhav, MI comeback, KL Rahul 150. Trend alignment is the single biggest predictor. |
| **Format 0.25** | Reaction hooks (552 avg) vs prediction hooks (83 avg) = 6.6x difference. How you present content matters almost as much as what's trending. |
| **Entity 0.20** | Bumrah (1285 avg) vs KKR content (68 avg) = 19x difference. Entity selection is powerful but secondary to trend timing. |
| **Historical 0.15** | The 7-dimension highlight score (hook_strength, emotional_peak, etc.) captures intrinsic clip quality. Important but not dominant. |
| **Timing 0.10** | Upload hour/day has measurable but modest impact compared to content decisions. |

### 9.3 Weight Adaptation

Weights are not static. After every 20 `metrics_received` events, the system recalibrates:

```python
def recalibrate_weights(self):
    """Compare predicted scores vs actual performance. Adjust weights."""
    events = self._store.get_events(event_type=EventType.metrics_received)[-20:]
    if len(events) < 20:
        return

    predictions = []
    actuals = []
    for event in events:
        payload = json.loads(event.payload_json)
        predicted = self._compute_score_at_time(event, payload)  # what we predicted
        actual = normalize(payload["views"])  # what actually happened
        predictions.append(predicted)
        actuals.append(actual)

    # Compute per-component correlation with actual performance
    correlations = {}
    for component in ["trend", "format", "entity", "historical", "timing"]:
        component_scores = [p[component] for p in predictions]
        correlations[component] = pearsonr(component_scores, actuals)

    # Re-weight proportional to correlation (with floor of 0.05)
    total = sum(max(c, 0.05) for c in correlations.values())
    new_weights = {k: max(v, 0.05) / total for k, v in correlations.items()}

    # Blend with current weights (don't over-correct)
    for k in new_weights:
        new_weights[k] = 0.7 * self._weights[k] + 0.3 * new_weights[k]

    self._state.set("scoring_weights", new_weights)
```

---

## 10. Recommendation Engine

### 10.1 Output Format

```python
@dataclass
class CricketRecommendation:
    category: str          # "topic", "player", "team", "hook", "title", "duration", "timing"
    priority: str          # "high", "medium", "low"
    recommendation: str    # human-readable action
    reason: str            # WHY — traceable to learned state
    confidence: float      # 0-1
    supporting_data: dict  # raw numbers
    expires_at: str | None # for time-sensitive recommendations
```

### 10.2 Recommendation Sources

```python
class CricketRecommendationEngine:
    def __init__(self, format_learner, trend_engine, entity_learner,
                 timing_learner, duration_learner, state_store):
        ...

    def generate_all(self) -> list[CricketRecommendation]:
        recs = []
        recs.extend(self._topic_recommendations())
        recs.extend(self._player_recommendations())
        recs.extend(self._hook_recommendations())
        recs.extend(self._title_recommendations())
        recs.extend(self._duration_recommendations())
        recs.extend(self._timing_recommendations())
        recs.extend(self._anti_pattern_warnings())
        return sorted(recs, key=lambda r: {"high": 0, "medium": 1, "low": 2}[r.priority])
```

### 10.3 Example Recommendations (from your actual data)

```
[HIGH/topic]     Cover Bumrah more aggressively
  Reason:         avg 1285 views (4.4x baseline) but only 3 videos in 80 days
  Confidence:     0.88
  Data:           {n: 3, avg_views: 1285, baseline: 290}

[HIGH/topic]     Create reaction/fan-war content
  Reason:         Reaction hooks avg 552 views, fan-war avg 434 — both 1.5-2x baseline
  Confidence:     0.82
  Data:           {reaction_n: 7, fan_war_n: 7, baseline: 290}

[HIGH/player]    Reduce Jadeja coverage
  Reason:         17 videos at avg 233 views. Fatigue penalty active (0.21).
  Confidence:     0.75
  Data:           {n: 17, avg_views: 233, fatigue: 0.21, effective_score: 0.27}

[MEDIUM/title]   Stop using question marks in titles
  Reason:         Questions avg 182 views vs 321 without. -43% impact.
  Confidence:     0.80
  Data:           {question_n: 27, no_question_n: 94}

[MEDIUM/title]   Use longer titles (60+ characters)
  Reason:         Long titles avg 365 views vs 224 for medium. +63% lift.
  Confidence:     0.78
  Data:           {long_n: 55, medium_n: 60}

[MEDIUM/team]    Avoid KKR content
  Reason:         avg 68 views across 4 videos. Worst team by far.
  Confidence:     0.70
  Data:           {n: 4, avg_views: 68, baseline: 290}

[LOW/timing]     Upload between 19:00-20:00 IST
  Reason:         Evening uploads show higher initial velocity
  Confidence:     0.55
  Data:           {insufficient_data: True, n: 27}
```

---

## 11. Replay Protection & Idempotency

### 11.1 The Problem

Your current `ReplayEngine.replay()` replays ALL events through `PolicyUpdater`, which calls `Learner.process_feedback()`, which adjusts `hook_weight ± 0.1`. If you replay 100 events, you get 100 adjustments — **even though those events were already processed**. This is replay inflation.

### 11.2 The Solution: ProcessedEventLog

```python
class ProcessedEventLog:
    """Tracks which events have been processed by which learners."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def is_processed(self, event_id: str, learner: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM processed_events WHERE event_id=? AND learner=?",
            (event_id, learner)
        ).fetchone()
        return row is not None

    def mark_processed(self, event_id: str, learner: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO processed_events (event_id, learner, processed_at) VALUES (?, ?, ?)",
            (event_id, learner, datetime.now(timezone.utc).isoformat())
        )
        self._conn.commit()
```

### 11.3 Learning Dispatcher (replaces PolicyUpdater)

```python
class LearningDispatcher:
    """Routes events to specialized learners with idempotency."""

    LEARNERS = ["format", "entity", "timing", "duration", "trend"]

    def __init__(self, format_learner, entity_learner, timing_learner,
                 duration_learner, trend_engine, processed_log):
        self._format = format_learner
        self._entity = entity_learner
        self._timing = timing_learner
        self._duration = duration_learner
        self._trend = trend_engine
        self._log = processed_log

    def dispatch(self, event: ClipEvent) -> int:
        """Process event through all relevant learners. Returns count of learners updated."""
        updated = 0
        payload = json.loads(event.payload_json)

        if event.event_type == EventType.metrics_received:
            for learner_name, learner in [
                ("format", self._format),
                ("entity", self._entity),
                ("timing", self._timing),
                ("duration", self._duration),
            ]:
                if not self._log.is_processed(event.event_id, learner_name):
                    learner.process_metrics(event.clip_id, payload)
                    self._log.mark_processed(event.event_id, learner_name)
                    updated += 1

        elif event.event_type == EventType.trend_ingested:
            if not self._log.is_processed(event.event_id, "trend"):
                self._trend.ingest(TrendInput.from_payload(payload))
                self._log.mark_processed(event.event_id, "trend")
                updated += 1

        elif event.event_type == EventType.manual_override:
            # Manual overrides always process (they're corrections)
            self._format.process_override(event.clip_id, payload)
            self._entity.process_override(event.clip_id, payload)
            updated += 2

        return updated
```

### 11.4 Anti-Pattern Guards

| Anti-Pattern | Prevention |
|-------------|-----------|
| **Double processing** | `ProcessedEventLog.is_processed()` check before every learner call |
| **Replay inflation** | `ReplayEngine` checks `is_processed()` and skips already-processed events |
| **Topic overfitting** | Fatigue penalty in EntityLearner reduces score for over-covered entities |
| **Trend chasing only** | HistoricalScore (0.15) + FormatScore (0.25) ensure non-trend signals matter |
| **Keyword stuffing** | FormatLearner caps title feature scores at 1.0; no compounding |
| **Feedback loops** | Thompson Sampling exploration (ε=0.15) forces testing of non-optimal choices |
| **Old trend domination** | Half-life decay ensures 7-day-old trends score < 0.20 |

---

## 12. ML Algorithm Selection

For 500-5000 clips at personal channel scale, deep learning is overkill. Here's what to use and where:

| Component | Algorithm | Why |
|-----------|-----------|-----|
| **Format scoring** | **Bayesian EMA** (α=0.15) | 7 hook types × ~15 samples each. EMA handles recency. Bayesian gives uncertainty estimates. No need for gradient descent. |
| **Entity scoring** | **Confidence-weighted EMA** | 15 players × ~5 samples each. Confidence weighting prevents overreaction to small samples (Bumrah n=3 shouldn't dominate). |
| **Hook selection** | **Thompson Sampling** | Explore/exploit for hook types. Beta posterior naturally handles uncertainty. With n=7 for reaction hooks, the posterior is wide — system will explore. |
| **Trend decay** | **Exponential half-life** | Deterministic, no training needed. Half-life is a domain parameter, not a learned one. |
| **Weight calibration** | **Pearson correlation** | After every 20 events, correlate each component with actual views. Simple, interpretable, no overfitting risk with n=20. |
| **Duration optimization** | **UCB1** | 4 duration buckets. UCB1 naturally explores under-sampled buckets. Better than Thompson here because buckets are discrete and few. |
| **Future (5000+ clips)** | **LightGBM** | When you have enough data, train a gradient-boosted model on features: `[hook_type, player, team, title_length, is_hinglish, hour, dow, trend_score]` → `views`. LightGBM handles categorical features natively, trains in seconds, and is interpretable via feature importance. |

**NOT recommended:**
- **Neural networks**: 5000 samples with ~10 features is firmly in "tabular ML" territory. LightGBM will outperform any NN.
- **Reinforcement learning**: The action space (pick hook type, pick player) is too small for RL. Thompson Sampling solves the explore/exploit problem with 10 lines of code.
- **Collaborative filtering**: Single channel, single audience. No user-item matrix to factorize.

---

## 13. Python Class Design

### 13.1 Core Classes

```python
# automation/learner/state_store.py
class PersistentStateStore:
    """SQLite-backed learned state store. Replaces in-memory LearnedStateStore."""
    def __init__(self, db_path: str = "self_learner.db")
    def get(self, key: str) -> dict | None
    def set(self, key: str, value: dict, derived_from: str | None = None) -> None
    def get_all(self) -> dict[str, dict]
    def delete(self, key: str) -> None
    def clear(self) -> None

    # Entity-specific helpers
    def get_entity(self, entity_type: str, name: str) -> dict
    def set_entity(self, entity_type: str, name: str, data: dict) -> None
    def get_all_entities(self, entity_type: str) -> dict[str, dict]
    def count_recent(self, entity_type: str, name: str, window_days: int) -> int

# automation/learner/processed_log.py
class ProcessedEventLog:
    """Idempotency guard. Tracks which events were processed by which learners."""
    def __init__(self, conn: sqlite3.Connection)
    def is_processed(self, event_id: str, learner: str) -> bool
    def mark_processed(self, event_id: str, learner: str) -> None
    def clear_learner(self, learner: str) -> int  # for replay

# automation/learner/dispatcher.py
class LearningDispatcher:
    """Routes events to specialized learners. Replaces PolicyUpdater."""
    def __init__(self, format_learner, entity_learner, timing_learner,
                 duration_learner, trend_engine, processed_log)
    def dispatch(self, event: ClipEvent) -> int
    def dispatch_batch(self, events: list[ClipEvent]) -> int

# automation/learner/format_learner.py
class FormatLearner:
    """Tracks hook type and title pattern performance. 40% weight."""
    def __init__(self, state_store: PersistentStateStore)
    def process_metrics(self, clip_id: str, payload: dict) -> None
    def process_override(self, clip_id: str, payload: dict) -> None
    def get_hook_score(self, hook_type: str) -> float
    def get_title_score(self, features: dict) -> float
    def select_hook_type(self) -> str  # Thompson Sampling
    def get_scores(self) -> dict

# automation/learner/entity_learner.py
class EntityLearner:
    """Tracks player, team, series, format performance. 20% weight."""
    def __init__(self, state_store: PersistentStateStore)
    def process_metrics(self, clip_id: str, payload: dict) -> None
    def process_override(self, clip_id: str, payload: dict) -> None
    def get_entity_score(self, players: list[str], teams: list[str],
                         series: str | None = None) -> float
    def fatigue_penalty(self, entity_type: str, name: str) -> float
    def get_top_entities(self, entity_type: str, limit: int = 5) -> list[dict]

# automation/learner/trend_engine.py
class TrendEngine:
    """Cricket trend tracking with half-life decay. 30% weight."""
    def __init__(self, state_store: PersistentStateStore)
    def ingest(self, trend: TrendInput) -> None
    def decay_all(self) -> int
    def get_active(self, min_score: float = 0.10) -> list[dict]
    def get_trend_for_entities(self, players: list[str],
                                teams: list[str]) -> float
    def prune_expired(self) -> int
    def get_hot_topics(self, limit: int = 5) -> list[dict]

# automation/learner/timing_learner.py
class TimingLearner:
    """Upload timing optimization. 10% weight."""
    def __init__(self, state_store: PersistentStateStore)
    def process_metrics(self, clip_id: str, payload: dict) -> None
    def best_hour(self, day: int | None = None) -> int
    def best_day(self) -> str
    def get_schedule_recommendation(self) -> dict

# automation/learner/duration_learner.py
class DurationLearner:
    """Optimal clip duration via UCB1."""
    def __init__(self, state_store: PersistentStateStore)
    def process_metrics(self, clip_id: str, payload: dict) -> None
    def select_duration(self) -> int  # UCB1 selection
    def get_scores(self) -> dict

# automation/learner/scorer.py
class CricketScorer:
    """Combines all learners into a single candidate score."""
    def __init__(self, format_learner, entity_learner, trend_engine,
                 timing_learner, duration_learner, state_store)
    def score_candidate(self, candidate: dict) -> float
    def score_many(self, candidates: list[dict]) -> list[tuple[dict, float]]
    def recalibrate_weights(self) -> None

# automation/learner/recommender.py
class CricketRecommendationEngine:
    """Generates actionable recommendations from all learners."""
    def __init__(self, format_learner, entity_learner, trend_engine,
                 timing_learner, duration_learner, state_store)
    def generate_all(self) -> list[CricketRecommendation]
    def _topic_recommendations(self) -> list[CricketRecommendation]
    def _player_recommendations(self) -> list[CricketRecommendation]
    def _hook_recommendations(self) -> list[CricketRecommendation]
    def _title_recommendations(self) -> list[CricketRecommendation]
    def _duration_recommendations(self) -> list[CricketRecommendation]
    def _timing_recommendations(self) -> list[CricketRecommendation]
    def _anti_pattern_warnings(self) -> list[CricketRecommendation]

# automation/learner/replay.py (UPDATED)
class ReplayEngine:
    """Rebuilds learned state from event store. Now idempotent."""
    def __init__(self, decision_store, state_store, dispatcher)
    def replay(self) -> int  # skips already-processed events
    def replay_force(self, learner: str | None = None) -> int  # clear + replay
    def verify(self) -> bool
```

### 13.2 Data Models

```python
@dataclass(frozen=True)
class TrendInput:
    trend_id: str
    source: str
    query: str
    trend_score: float
    velocity: float
    category: str
    entities: dict
    half_life_hours: float
    expires_at: str | None = None

@dataclass(frozen=True)
class CricketRecommendation:
    category: str
    priority: str
    recommendation: str
    reason: str
    confidence: float
    supporting_data: dict = field(default_factory=dict)
    expires_at: str | None = None
```

---

## 14. Folder Structure

```
automation/
├── learner/
│   ├── __init__.py              # Re-exports all public classes
│   ├── state_store.py           # PersistentStateStore (SQLite-backed)
│   ├── processed_log.py         # ProcessedEventLog (idempotency)
│   ├── dispatcher.py            # LearningDispatcher (replaces PolicyUpdater)
│   ├── format_learner.py        # FormatLearner (hook types + title patterns)
│   ├── entity_learner.py        # EntityLearner (players, teams, series)
│   ├── trend_engine.py          # TrendEngine (decay + ingestion)
│   ├── timing_learner.py        # TimingLearner (hour, day, match phase)
│   ├── duration_learner.py      # DurationLearner (UCB1 bucket selection)
│   ├── scorer.py                # CricketScorer (weighted combination)
│   ├── recommender.py           # CricketRecommendationEngine
│   ├── replay.py                # ReplayEngine (UPDATED — idempotent)
│   ├── learner.py               # KEEP — backward compat wrapper
│   ├── policy_updater.py        # KEEP — backward compat wrapper
│   └── preference_engine.py     # KEEP — backward compat wrapper
│
├── memory/
│   ├── __init__.py
│   ├── event_models.py          # EventType (ADD: trend_ingested, trend_decayed)
│   ├── decision_store.py        # DecisionStore + LearnedStateStore (KEEP)
│   ├── feedback_schema.py       # FeedbackPayload (KEEP)
│   └── memtrack.py              # (KEEP)
│
├── orchestrator.py              # UPDATE: wire new learners in stage 9c
└── ...
```

**Backward compatibility:** The old `Learner`, `PolicyUpdater`, and `PreferenceEngine` classes remain as thin wrappers that delegate to the new system. Existing tests in `test_automation.py` continue to pass.

---

## 15. End-to-End Example Flow

### Scenario: Pipeline processes a Bumrah wicket clip during MI vs CSK

```
1. PIPELINE detects highlight:
   transcript: "Bumrah ne maara yorker! CSK batsman clean bowled!"
   → emit candidate_created {
       detected_players: ["bumrah"],
       detected_teams: ["mi", "csk"],
       detected_hook_type: "shock",
       duration_seconds: 28
     }

2. HIGHLIGHT SCORING:
   → emit candidate_scored {
       hook_strength: 8.5, emotional_peak: 9.2, weighted_score: 8.1
     }

3. CRICKET SCORER evaluates:
   TrendScore:    trend_engine.get_trend_for_entities(["bumrah"], ["mi","csk"])
                  = 0.72 (IPL is trending, Bumrah has active trend)
   FormatScore:   format_learner.get_hook_score("shock") × title_score
                  = 0.55 × 0.78 = 0.43
   EntityScore:   entity_learner.get_entity_score(["bumrah"], ["mi","csk"])
                  = 0.95 × 0.5 + 0.82 × 0.3 + 0.70 × 0.2 = 0.86
   Historical:    8.1 / 10.0 = 0.81
   TimingScore:   timing_learner.get_schedule_recommendation()["confidence"]
                  = 0.65

   FinalScore = 0.30×0.72 + 0.25×0.43 + 0.20×0.86 + 0.15×0.81 + 0.10×0.65
              = 0.216 + 0.108 + 0.172 + 0.122 + 0.065
              = 0.683

   FatiguePenalty: bumrah n=3, no fatigue → 0.0
   Final: 0.683

4. CLIP SELECTED (top-k), EXPORTED, SEO GENERATED:
   Title: "Bumrah Ka TOOFANI Yorker! 🥶 CSK Batsman Clean Bowled | MI vs CSK IPL 2026"
   (long title, hinglish, no question, has caps, has pipe, has emoji)
   → emit published { youtube_video_id: "abc123", publish_hour: 19 }

5. 48 HOURS LATER — metrics_received:
   views: 1450, likes: 52, comments: 8, shares: 12
   avg_watch_percentage: 0.78, completion_rate: 0.52
   views_per_hour: 30.2

6. LEARNING DISPATCHER routes to 4 learners:

   FormatLearner.process_metrics():
     hook_type="shock" → signal = min(1450/290, 3.0)/3.0 = 1.0
     score = 0.85 × 0.55 + 0.15 × 1.0 = 0.618 → 0.62
     title: long+hinglish+no_question → all get boosted

   EntityLearner.process_metrics():
     bumrah: signal = min(1450/290, 3.0)/3.0 = 1.0
       score = (1-0.26) × 0.95 + 0.26 × 1.0 = 0.96
     mi: signal = 1.0, score = (1-0.13) × 0.82 + 0.13 × 1.0 = 0.84
     csk: signal = 1.0, score = (1-0.13) × 0.50 + 0.13 × 1.0 = 0.57

   TimingLearner.process_metrics():
     hour=19: signal = 1.0, score boosted
     dow=thursday: signal = 1.0, score boosted

   DurationLearner.process_metrics():
     bucket="25-34s": signal = 1.0, UCB1 arm updated

7. NEXT RECOMMENDATION CYCLE:
   [HIGH] "Cover Bumrah more — now avg 1368 views across 4 videos (4.7x baseline)"
   [HIGH] "MI content continues to outperform — avg 498 views across 15 videos"
   [MEDIUM] "Shock hooks improving — score rose from 0.55 to 0.62 after latest Bumrah clip"
```

---

## 16. Production Implementation Plan

### Phase 1: Foundation (Week 1)
1. Create `PersistentStateStore` backed by `self_learner.db`
2. Create `ProcessedEventLog`
3. Extend `EventType` with `trend_ingested`, `trend_decayed`
4. Migrate 486 existing facts into new schema
5. Write tests for state store + processed log

### Phase 2: Learners (Week 2)
1. Implement `FormatLearner` with Bayesian EMA + Thompson Sampling
2. Implement `EntityLearner` with confidence-weighted EMA + fatigue penalty
3. Implement `DurationLearner` with UCB1
4. Implement `TimingLearner`
5. Seed all learners from existing 121 SEO outcomes
6. Write tests for each learner

### Phase 3: Trend Engine (Week 3)
1. Implement `TrendEngine` with half-life decay
2. Build trend ingestion from YouTube Search Suggestions
3. Build cricket fixture calendar ingestion
4. Implement `decay_all()` periodic task
5. Write tests for decay math

### Phase 4: Integration (Week 4)
1. Implement `LearningDispatcher`
2. Implement `CricketScorer`
3. Implement `CricketRecommendationEngine`
4. Wire into `orchestrator.py` stage 9c (replace old learner wiring)
5. Update `ReplayEngine` for idempotency
6. Keep old `Learner`/`PolicyUpdater`/`PreferenceEngine` as backward-compat wrappers

### Phase 5: Validation (Week 5)
1. Run dry_run.py to validate full pipeline
2. Compare new scorer rankings vs old on historical 121 clips
3. A/B test: generate recommendations, manually evaluate
4. Monitor for anti-patterns (overfitting, feedback loops)

### Phase 6: LightGBM (Month 3, when n > 500)
1. Export features from learned state
2. Train LightGBM on `[hook_type, player, team, title_features, hour, dow, trend_score]` → `views`
3. Compare LightGBM predictions vs weighted formula
4. If LightGBM outperforms by >15%, use it as `HistoricalScore` component

---

## 17. Future ML Upgrade Path

```
Current (n=121)          Phase 2 (n=500)           Phase 3 (n=2000+)
─────────────────        ──────────────────        ──────────────────
Bayesian EMA       →     Bayesian EMA        →     LightGBM ensemble
Thompson Sampling  →     Thompson Sampling   →     Contextual bandits
UCB1               →     UCB1                →     LightGBM regression
Half-life decay    →     Half-life decay     →     Learned decay (GBM)
Pearson recalib.   →     Pearson recalib.    →     Gradient-based optim.
Entity dict        →     Entity embeddings   →     Entity graph (GNN)
```

**Rule:** Do not upgrade until the simpler model has been validated on real data for at least 50 uploads. Premature complexity is the #1 killer of personal ML projects.
