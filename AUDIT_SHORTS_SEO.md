# Automation Shorts SEO Audit Report — v2

**Date:** 2026-09-19 (revised)
**Scope:** `automation/seo/`, `automation/orchestrator.py`, `automation/clip_selection/`, `shorts_intelligence/`, `config.yaml`
**Objective:** Identify why the automated pipeline is not generating high-performing YouTube Shorts metadata, with focus on VidIQ integration, keyword strategy, and how metadata signals reach YouTube's algorithm.

---

## Executive Summary

The pipeline is architecturally strong: entity grounding, copy auditing, fail-closed VidIQ requirement, live-framing filters, and hallucination scrubbing are all well-engineered. The description strategy of writing long, keyword-rich, algorithm-facing metadata is correct — YouTube's classifier ingests the full description text for topic clustering, entity extraction, and audience matching regardless of whether viewers read it.

**The real problems are:**

1. **VidIQ's output is researched but not respected.** The system instructs the LLM to use a specific VidIQ title (`seo.py:1576`) but then deliberately does NOT override the LLM's title with VidIQ's (`seo.py:1629` — commented out line). VidIQ's data-driven titles get discarded in favor of the LLM's creative rewrite.

2. **The grounding filter is too aggressive.** VidIQ returns high-volume keywords that the grounding system rejects because they contain proper nouns or topic signals not literally present in the transcript (`seo.py:1129-1154`). This starves the metadata of exactly the high-demand phrases VidIQ found.

3. **The "vague filler" filter strips power words that drive CTR.** Words like "shocking", "incredible", "insane", "stunning", "jaw-dropping" are banned from titles (`seo.py:109-114`), but these are the exact words that stop the scroll on the Shorts shelf. The filter confuses "LLM laziness" with "audience psychology."

4. **The fallback description builder produces formulaic template prose** that reads like auto-generated content and may trigger YouTube's low-quality classifier (`seo.py:400-424`).

5. **Primary search terms are force-injected into descriptions** with a "Viewer search context also includes..." suffix (`seo.py:1976`) that is detectable spam.

6. **No trending/real-time signal feeds into hashtags.** Hashtags are LLM-generated from the evidence pack, not from live trending data. This means the 3 hashtag slots miss the current trending conversation.

---

## Table of Contents

1. [VidIQ Integration: Research Used, Output Ignored](#1-vidiq-integration-research-used-output-ignored)
2. [Grounding Filter Over-Rejection](#2-grounding-filter-over-rejection)
3. [Title Generation: Accuracy Over CTR](#3-title-generation-accuracy-over-ctr)
4. [Description Strategy: Right Direction, Wrong Execution](#4-description-strategy-right-direction-wrong-execution)
5. [Search Terms & Keyword Injection](#5-search-terms--keyword-injection)
6. [Tags: Correct, Well-Packed](#6-tags-correct-well-packed)
7. [Hashtags: Missing Trending Signal](#7-hashtags-missing-trending-signal)
8. [Quality Gates: Too Many Rejection Points](#8-quality-gates-too-many-rejection-points)
9. [Shorts Intelligence Feedback Loop](#9-shorts-intelligence-feedback-loop)
10. [Recommendations — Priority Ranked](#10-recommendations--priority-ranked)
11. [File Reference Map](#11-file-reference-map)

---

## 1. VidIQ Integration: Research Used, Output Ignored

### How VidIQ is Connected

The pipeline uses VidIQ MCP (`vidiq.py`) with these tools:
- `vidiq_keyword_research` — seed keyword + related/rising keywords with India volume, competition, overall score
- `vidiq_generate_titles` — scored title suggestions for the clip's topic
- `vidiq_channel_videos` — recent channel Shorts history for novelty checking

Budget: 4-6 calls per clip (`config.yaml:278`), with typical allocation:
- 1 call: channel history (optional, cached 6h)
- 1-2 calls: keyword research
- 1 call: title generation

VidIQ is `required: true` (`config.yaml:275`), meaning the pipeline fails closed if VidIQ returns insufficient grounded data. This is correct.

### The Core VidIQ Problem

**VidIQ data is used as *context* for the LLM but not as the *output*.**

Evidence:

**Title handling (`seo.py:1573-1582`):**
```python
user_prompt += (
    "\n\nEXCLUSIVE SEO SOURCE CONTRACT:\n"
    f"Use this exact vidIQ title: {vidiq_titles[0]}\n"
    ...
)
```
The prompt *tells* the LLM to use the VidIQ title. But then:
```python
# result["title"] = vidiq_titles[0] # REMOVED: Do not override the LLM's language/tone.
```
Line `seo.py:1629` — the code that would enforce VidIQ's title is **commented out**. The LLM is free to — and regularly does — rewrite the title in its own style, losing the data-driven phrasing VidIQ optimized for India search volume.

**Keyword handling (`seo.py:1630-1634`):**
```python
result["search_terms"] = vidiq_keywords
result["primary_search_terms"] = vidiq_keywords[:...]
result["tags"] = vidiq_keywords[:...]
```
Search terms and tags correctly use VidIQ's ranked keywords. But these are API-only fields — the *public-facing* title and description text are LLM-generated without VidIQ's top-ranked phrases prioritized.

**Description handling:**
VidIQ keywords are passed into the prompt as "APPROVED SEARCH SEEDS" (`seo.py:201-202`) for the LLM to "weave naturally into flowing prose." The LLM decides where and how to use them. If the LLM buries the strongest VidIQ keyword in paragraph 3 of the description instead of putting it in the hook (first 150 chars), the algorithm's topic classification misses it.

### Impact

VidIQ's value proposition is data-driven: it knows which phrases have India search volume, which title patterns score well, which keywords have low competition. When the LLM rewrites everything, that data-driven advantage is lost. The LLM substitutes its own "creative" phrasing which is not backed by search demand data.

### What Should Change

1. **VidIQ's highest-scored title should be the default title**, with the LLM only allowed to adjust for grounding failures (wrong entity) or length violations. The current commented-out line (`seo.py:1629`) was the right approach.

2. **VidIQ's #1 India-volume keyword should be mandated in the first 150 chars of the description.** The prompt tells the LLM to "front-load the strongest grounded search phrase" (`seo.py:221`) but there is no enforcement — no post-generation check verifies this actually happened.

3. **VidIQ's title score should be a selection criterion.** Currently, title candidates from VidIQ are ranked (`seo.py:1312-1348`) but after grounding, the ranked order is passed as "references" to the LLM. The highest-scored, grounded VidIQ title should win unless a specific quality gate fails.

---

## 2. Grounding Filter Over-Rejection

### The Grounding Design (Correct in Principle)

The pipeline ensures every keyword, title, and search term is "grounded" — meaning it only references players, teams, and topics that are actually present in the clip's transcript, source video, or OCR evidence. This prevents hallucinated metadata (e.g., tagging a clip about Bumrah with "Virat Kohli century" just because Kohli gets more searches).

This is a genuinely good design decision.

### The Over-Rejection Problem

The grounding filter at `seo.py:1121-1154` is too strict in several ways:

**1. Proper noun rejection (`seo.py:1145-1152`):**
```python
unknown_proper = [
    token for token in re.findall(r"\b[A-Z][A-Za-z'-]{3,}\b", text)
    if token.casefold() not in source_tokens
    and token.casefold() not in safe_modifiers
    and token.casefold() not in entity_tokens
]
if unknown_proper:
    return False
```
Any capitalized word not found in the source text, safe modifiers, or entity catalog causes rejection. VidIQ may return legitimate high-volume keywords containing proper nouns that are contextually related but not literally in the transcript (e.g., "ICC Champions Trophy highlights" when the clip discusses a Champions Trophy match but the transcript says "CT" not "ICC Champions Trophy"). These get rejected.

**2. Topic signal drift rejection (`seo.py:1137-1140`):**
```python
candidate_topic_signals = topic_signals(words)
if candidate_topic_signals - allowed_topic_signals:
    return False
if source_topic_signals and not (candidate_topic_signals & allowed_topic_signals):
    return False
```
If the source discusses "bowling" but VidIQ returns a keyword about "batting average" (because the batting counterpart is also trending), it gets rejected. This is correct for hallucination prevention but overly aggressive for SEO — viewers searching for the broader match context should find this Short.

**3. One-unknown-word tolerance (`seo.py:1153-1154`):**
```python
unexplained = words - source_tokens - safe_modifiers - entity_tokens
return len(unexplained) <= 1
```
Only 1 word in a keyword phrase can be "unexplained" (not found in source or safe list). VidIQ keywords like "Bumrah death over masterclass bowling speed" would fail if "masterclass" and "speed" aren't in the transcript — even though both are legitimate SEO terms for that topic.

### Impact

The system requests VidIQ keywords, pays for the API calls, gets high-quality India-volume data back — and then rejects a significant portion because the grounding filter demands literal transcript overlap. The `min_keywords: 12` floor (`config.yaml` via `seo.py:323-329`) can then cause the entire clip's SEO to fail because not enough keywords survived grounding.

### What Should Change

- **Expand `safe_modifiers`** to include common SEO qualifier words that are always legitimate in cricket context: "masterclass", "speed", "fastest", "record", "career", "stats", "history", "prediction", "preview", "recap", "comeback", etc.
- **Increase the unexplained-word tolerance from 1 to 2** for VidIQ-sourced keywords (they've already passed VidIQ's own relevance filtering).
- **Add a "soft grounding" tier**: keywords that fail strict grounding but come from VidIQ with high India volume (>1000) could be allowed into tags/search_terms but excluded from the title and description hook.

---

## 3. Title Generation: Accuracy Over CTR

### Current Title Pipeline

1. VidIQ generates ~5 scored title candidates (`seo.py:1299-1310`)
2. Candidates are filtered through `grounded_title()` (`seo.py:1156-1169`)
3. Surviving titles are ranked and passed to the LLM as "references"
4. LLM generates its own title (ignoring or loosely following VidIQ's)
5. Title is cleaned: vague filler stripped, live-framing removed, emoji ensured
6. Entity-presence gate: must contain a grounded player or team name
7. Promise alignment: title tokens must overlap with transcript + description opening

### Problem 1: Vague Filler Filter Strips CTR Power Words

`seo.py:109-114`:
```python
_VAGUE_FILLER_RE = re.compile(
    r"\b(?:epic|thrilling|incredible|amazing|awesome|best|shocking|"
    r"stunning|mind[-\s]?blowing|unbelievable|fantastic|sensational|"
    r"breathtaking|jaw[-\s]?dropping|must[-\s]?watch|crazy|insane|"
    r"phenomenal|spectacular|greatest|top|superb|brilliant|classic)\b",
    re.IGNORECASE,
)
```

Every one of these words is stripped from every title. The intent is to remove generic LLM adjectives — but many of these words are proven CTR drivers on YouTube Shorts:

- "SHOCKING Bumrah Yorker" → becomes "Bumrah Yorker" (loses emotional hook)
- "INSANE Kohli Six" → becomes "Kohli Six" (bland)
- "Must-Watch Death Over" → becomes "Death Over" (no urgency)

The problem: the filter doesn't distinguish between **an LLM lazily padding a generic title** ("Epic Amazing Cricket Moment") and **a legitimately charged title** ("Shocking Bumrah Yorker Ends Smith"). The former should be caught; the latter should be kept.

**Competitive analysis context:** Top-performing cricket Shorts channels (Cricket Pakistan, ICC, ESPNcricinfo Shorts, CricTracker) regularly use words from this banned list in their highest-performing titles. These aren't slop — they're audience psychology.

### Problem 2: Entity-First Mandate Kills Curiosity Patterns

`seo.py:1826-1881` enforces that every title must lead with a grounded player or team entity. This means:

- "This Yorker Changed Everything — Bumrah vs Smith" → fails (entity not leading)
- "Why India's Bowling Plan Failed — IND vs SL" → fails (entity not in first position)
- "Bumrah vs Smith — The Yorker That Changed Everything" → passes

Entity-first is good for YouTube Search (people type "Bumrah") but suboptimal for the Shorts shelf where titles compete for attention in a scroll feed. Curiosity-gap patterns ("This Yorker...", "Why India's Plan...") perform well on Shorts because they create a reason to watch.

The entity-presence gate is correct — every title should contain a grounded entity. But mandating it as the *lead* reduces the title pattern vocabulary.

### Problem 3: Emoji Strategy is One-Dimensional

`seo.py:777-787` (`_ensure_title_emoji`):
- If the LLM-generated title has no emoji, a cricket ball emoji is appended
- No content-type-aware emoji selection
- The clip_selection pipeline already classifies content type (wicket, six, debate, controversy, etc.) but this classification is NOT used in the SEO layer for emoji selection

A wicket clip should get a fire/stumps emoji. A six should get a rocket/exploding emoji. A controversy should get a warning emoji. Instead, everything defaults to a generic cricket ball.

---

## 4. Description Strategy: Right Direction, Wrong Execution

### The Strategy is Correct

Writing a long, keyword-rich description (1500-4800 chars) is the right approach for YouTube Shorts. The description is metadata for YouTube's algorithm, not content for the viewer. YouTube's classifier ingests the full description text to:
- Extract entities (players, teams, tournaments)
- Identify topic clusters
- Build audience matching vectors
- Cross-reference with search demand

More structured, keyword-dense metadata gives the algorithm more signal to route the Short to the right audience cohort. The `description_min_chars: 1500` and `description_target_chars: 4000` config values are strategically sound.

### Where Execution Falls Short

#### Issue 1: The "Viewer search context also includes..." Suffix (`seo.py:1973-1981`)

```python
if missing_queries:
    suffix = "\n\nViewer search context also includes " + "; ".join(missing_queries) + "."
```

When primary search terms are not naturally present in the description, the system appends this literal string. This is a **detectable keyword injection pattern**. YouTube's content quality classifier can identify this type of formulaic keyword-stuffing suffix. It doesn't look natural, it doesn't add value, and it's exactly the pattern YouTube warns about.

**Fix:** Instead of appending a "Viewer search context" block, the missing keywords should be woven into the existing description sections, or the LLM should be re-prompted with a correction pass that integrates them naturally.

#### Issue 2: Fallback Description Builder (`seo.py:349-453`)

`_build_vidiq_description()` is used when the LLM is unavailable. It generates template prose:

```python
f"Viewers searching for {chunk[0]} will find detailed coverage here. "
f"This Short also covers aspects related to {chunk[1]} and {chunk[2]}, "
"providing context from the actual match discussion."
```

This is formulaic, repetitive, and reads like auto-generated content. Every 3-keyword chunk gets the same sentence template. YouTube's quality classifier may flag this pattern — especially when the same template appears across 10+ Shorts from the same channel.

**Fix:** The fallback should produce varied, natural-sounding prose. Even simple template rotation (3-4 sentence patterns per chunk) would reduce the fingerprint.

#### Issue 3: Description Sections Are Over-Structured

The 6-section mandate (Hook → Match Details → Analysis → Squad → Disclaimer → CTA) at `seo.py:219-242` is rigid. While structure is good for SEO, the fixed section headers ("🏏 MATCH DETAILS", "🔥 DETAILED CLIP ANALYSIS", "🏟️ SQUAD & PLAYING XI CONTEXT") are identical across every Short. This creates a channel-wide fingerprint that may trigger repetitive-content signals.

**Fix:** Vary section headers or let the LLM choose section structure while enforcing the content coverage (must include match context, must include player names, etc.).

#### Issue 4: Disclaimer Eats ~400 Chars of Keyword Budget

`seo.py:238-239`: The mandatory disclaimer is ~350 chars of non-SEO content. In a 4000-char description, this is ~9% of the keyword budget going to legal boilerplate that adds zero algorithmic value.

**Fix:** Move the disclaimer to the channel's About page, or compress it to 1 line (e.g., "Independent cricket commentary. All trademarks belong to their respective owners.").

---

## 5. Search Terms & Keyword Injection

### Search Terms Are Correctly VidIQ-Driven

When `vidiq.required: true`, search terms are purely VidIQ keywords (`seo.py:1518`, `seo.py:1630`):
```python
approved_queries = vidiq_keywords
result["search_terms"] = vidiq_keywords
```

This is correct. VidIQ's keywords are ranked by India volume and competition, grounded against the transcript, and packed into the search_terms field.

### The Force-Injection Problem

The system enforces that all `primary_search_terms` must appear in the description text (`seo.py:1968-1981`). When they don't, it appends the "Viewer search context also includes..." suffix. This creates a feedback loop:

1. VidIQ returns 12+ keywords
2. Top 4-8 become "primary" search terms
3. LLM writes description but doesn't use all 4-8 phrases verbatim
4. System detects missing phrases
5. System appends them as a keyword dump suffix
6. The suffix violates the system's own "NEVER dump a comma-separated keyword list" rule (`seo.py:180-181`)

The system is fighting itself: the prompt bans keyword lists, but the post-processing adds one.

### Config: min_search_terms: 4 — Reasonable

The minimum of 4 search terms (`config.yaml:257`) is reasonable. VidIQ almost always returns more than 4 grounded terms for cricket topics. The issue is not the minimums but the force-injection mechanism.

---

## 6. Tags: Correct, Well-Packed

### Current Implementation

- Tags are purely VidIQ keywords (`seo.py:1647`): `result["tags"] = vidiq_keywords`
- Packed to 450-500 chars (`config.yaml:265-266`)
- API-only field (not visible to viewers, not in description)
- Grounded, deduplicated, poison-filtered

### Assessment

**This is well-implemented.** Tags are the right place for dense keyword packing because they're invisible to viewers and consumed only by YouTube's API. Using VidIQ's ranked keywords here is optimal.

### One Issue: `min_tag_chars: 450` Gate

The 450-char minimum for tags can cause clips to fail when VidIQ doesn't return enough grounded keywords to fill 450 chars. This is an unnecessarily hard gate for a low-priority signal. Tags matter far less than title + description for algorithmic classification.

**Fix:** Reduce `min_tag_chars` to 200 or remove the minimum entirely. Keep `max_tag_chars: 500` for API compliance. Never fail a clip because of insufficient tag volume.

---

## 7. Hashtags: Missing Trending Signal

### Current Implementation

- 3 hashtag max for Shorts (`config.yaml:272`)
- `#Shorts` always leads (`seo.py:688-691`)
- LLM generates the other 2 based on the evidence pack
- Live-framing and Devanagari hashtags filtered
- Hashtags are appended to the description text for Shorts uploads

### The Problem: No Trending Data Feeds Hashtags

The hashtag generation has no integration with:
- **Google Trends** (real-time trending topics in India)
- **VidIQ trending topics** (what's trending on YouTube right now)
- **Twitter/X trending** (what cricket fans are discussing right now)

The LLM generates hashtags from the evidence pack, which means hashtags are based on the clip's content (correct) but NOT based on what's *currently trending* (missed opportunity).

Example: If a clip discusses Bumrah's bowling during an India vs Australia match, and `#INDvAUS` is trending on YouTube right now, the system doesn't know to use that exact trending hashtag. The LLM might generate `#IndiaVsAustralia` or `#BumrahBowling` — correct but not the *exact* trending form.

### What Should Change

- The trends.py module already fetches Google Trends RSS and YouTube suggest data
- This data should flow into hashtag generation as "trending hashtag candidates"
- The 3 slots should be: `#Shorts` + trending match hashtag (from live data) + topic-specific hashtag

---

## 8. Quality Gates: Too Many Rejection Points

### The Rejection Chain

A clip's SEO can fail at any of these points (all `raise SEOGenerationError`):

| Gate | Location | Condition |
|---|---|---|
| Empty/non-cricket transcript | `seo.py:1420` | Transcript fails `is_cricket_content()` |
| VidIQ returns no titles/keywords | `seo.py:1514-1517` | VidIQ call succeeded but returned empty |
| Writer LLM fails Tier 1 + Tier 2 | `seo.py:2045-2047` | Both generation tiers fail |
| Fallback description too short | `seo.py:1615-1618` | Deterministic description < `description_min_chars` |
| VidIQ keyword floor not met | `seo.py:1651-1657` | VidIQ returned < 12 grounded keywords |
| Copy audit: unsupported entities after scrub | `seo.py:1706-1709` | Search terms fell below floor after entity scrub |
| Title promise lost during scrub | `seo.py:1747-1750` | Title alignment failed after entity removal |
| LLM repair failed | `seo.py:1759-1763` | LLM repair returned nothing |
| Repaired title still misaligned | `seo.py:1769-1772` | Repair didn't fix alignment |
| Ungrounded title players (vidiq_required) | `seo.py:1791-1795` | Title contains player not in transcript |
| LLM repair failed (ungrounded name) | `seo.py:1804-1808` | Repair returned nothing |
| Repaired title still has unknown names | `seo.py:1815-1819` | Repair kept the bad name |
| Title has no grounded entity (vidiq_required) | `seo.py:1852-1855` | Generic title with no player/team |
| LLM repair failed (no entity) | `seo.py:1867-1871` | Repair returned nothing |
| Repaired title still no entity | `seo.py:1873-1876` | Repair didn't add entity |
| Ungrounded entities in API tags | `seo.py:1898-1902` | Tags contain ungrounded players |
| Ungrounded entities in rendered text | `seo.py:1920-1924` | Combined title+desc+tags have extra entities |
| Search term count out of range | `seo.py:1947-1951` | Not 4-26 grounded terms |
| Promise alignment below 0.50 | `seo.py:1993-1997` | Title doesn't match transcript + description |
| Final quality gate | `seo.py:1998-2001` | `_validate_seo_quality()` rejects |

That's **18+ distinct failure points**, each of which causes the entire clip's SEO to fail with no fallback. The system writes a `_seo_failed.json` marker for retry, but the same gates will fail again on retry unless the underlying evidence changes.

### The Cascading Scrub Problem

The copy audit can create cascading failures:

1. LLM writes a good title mentioning "Jadeja" (hallucinated — not in clip)
2. Entity scrub removes "Jadeja" from title, description, tags, search terms
3. Removing "Jadeja" from the title guts the promise tokens
4. Promise alignment score drops below 0.50
5. System tries LLM repair — repair may also fail or hallucinate
6. Entire clip SEO fails

This cascade is especially likely for clips where the LLM anchors the entire metadata around a hallucinated entity. The system correctly identifies the hallucination but then has no graceful recovery path — it depends on the same LLM (that just hallucinated) to generate a corrected version.

### What Should Change

- **After entity scrub, rebuild the title from VidIQ's title pool** instead of asking the LLM to repair. VidIQ titles are already grounded and scored.
- **Reduce `min_keywords: 12` to 6-8.** The current 12-keyword floor (`seo.py:323-329`) means a clip needs VidIQ to return 12+ keywords that ALL pass the strict grounding filter. For niche clip topics, this is too high.
- **Make `min_tag_chars` a soft gate.** Log a warning instead of failing closed when tags are under 450 chars.

---

## 9. Shorts Intelligence Feedback Loop

### What's Tracked (Good)

- Duration buckets (under_15, 15_24, etc.)
- Speed buckets (natural, light, fast)
- Complete-thought gate
- Title length, description length, hashtag count (as features)
- Real YouTube Analytics snapshots as outcomes

### What's Missing (Gap)

The learner tracks structural metadata features (lengths, counts) but not **metadata content patterns**:

| Not Tracked | Why It Matters |
|---|---|
| Which VidIQ keywords appeared in the top-performing Shorts | Identifies which keyword themes drive views |
| Title pattern formula (entity-first vs curiosity-gap vs power-word) | Identifies which title styles perform on this channel |
| Whether VidIQ's original title was used vs LLM-rewritten | Tests whether VidIQ titles outperform LLM titles |
| Hashtag combination performance | Identifies which hashtag patterns drive distribution |
| Description section coverage (which sections were included) | Tests whether certain description sections correlate with performance |
| Promise alignment score vs actual views | Tests whether tighter alignment = better performance |

The `bridge.py` (`shorts_intelligence/bridge.py:90-233`) does extract some metadata features but at a bucketed/count level, not at a content/pattern level.

### Specific Recommendation

Track `metadata_source` (already in the output: `"vidiq_ranked_grounded_writer"` or `"deterministic_grounded_fallback"`) as a feature in the learner. This would tell you whether VidIQ-backed metadata outperforms LLM-generated metadata on actual YouTube outcomes.

---

## 10. Recommendations — Priority Ranked

### P0 — Fix Immediately

#### R-1: Enforce VidIQ Title as Default

**Location:** `seo.py:1629`
**Current:** `# result["title"] = vidiq_titles[0] # REMOVED: Do not override the LLM's language/tone.`
**Fix:** Uncomment this line. VidIQ titles are data-driven (based on actual India search volume and competitive analysis). The LLM's "language/tone" preferences should not override data-driven SEO.

**Safeguard:** Only override when the VidIQ title passes the existing grounding gates (entity presence, promise alignment). If VidIQ's title fails grounding, fall back to the LLM title.

#### R-2: Remove "Viewer search context also includes..." Injection

**Location:** `seo.py:1973-1981`
**Problem:** This is the exact keyword-stuffing pattern the system's own prompt bans.
**Fix:** Remove the suffix injection entirely. If primary keywords aren't in the description, either:
  (a) Accept it — they're already in search_terms and tags, or
  (b) Re-prompt the LLM with a repair pass that weaves them naturally.

#### R-3: Revise Vague Filler Filter for Titles

**Location:** `seo.py:109-114`
**Problem:** Strips CTR-driving power words.
**Fix:** Split the list into two tiers:
  - **Strip always:** "epic", "amazing", "awesome", "fantastic", "superb", "brilliant" (genuinely vague)
  - **Keep if paired with entity:** "shocking", "insane", "stunning", "jaw-dropping", "incredible", "unbelievable", "must-watch" (legitimate emotional hooks when attached to a specific moment)

Example: "Incredible Amazing Cricket Moment" → strip both (no entity, no specificity). "Shocking Bumrah Yorker Dismissal" → keep "Shocking" (entity-specific emotional hook).

### P1 — High Priority

#### R-4: Expand Grounding Safe Modifiers

**Location:** `seo.py:1048-1056`
**Fix:** Add these to `safe_modifiers`:
```python
"masterclass", "speed", "fastest", "slowest", "record", "career",
"stats", "history", "prediction", "preview", "recap", "comeback",
"rivalry", "classic", "legendary", "highlight", "performance",
"strategy", "tactics", "form", "average", "strike", "rate",
```
These are legitimate cricket SEO terms that are not hallucinations even when absent from the specific transcript.

#### R-5: Increase Unexplained Word Tolerance for VidIQ Keywords

**Location:** `seo.py:1153-1154`
**Current:** `return len(unexplained) <= 1`
**Fix:** `return len(unexplained) <= 2` — Allow up to 2 words in a VidIQ keyword phrase to be outside the source tokens. VidIQ's own relevance scoring already filters irrelevant keywords.

#### R-6: Use VidIQ Titles in Entity Scrub Recovery

**Location:** `seo.py:1736-1773`
**Problem:** When entity scrub guts the title, the system asks the same LLM to repair. The LLM may hallucinate again.
**Fix:** After entity scrub, first check if any remaining VidIQ title candidate passes grounding. Use that instead of LLM repair. Only fall back to LLM repair if no VidIQ title survives.

#### R-7: Feed Trending Data into Hashtag Generation

**Location:** `seo.py:668-697` (hashtag enforcement) and prompt template
**Fix:** Query `trends.py` for current trending cricket hashtags (Google Trends + YouTube suggest) and inject these into the prompt as "trending hashtag candidates." Let the LLM choose from trending + clip-specific options.

#### R-8: Vary Fallback Description Templates

**Location:** `seo.py:400-424`
**Problem:** Every 3-keyword chunk uses the same "Viewers searching for X will find detailed coverage here" template.
**Fix:** Create 4-5 sentence pattern variants and rotate. Even simple variation ("This clip explores [X] in the context of [match]", "Fans following [X] will recognize this key moment from [match]", etc.) reduces the template fingerprint.

#### R-9: Reduce `min_keywords` Floor from 12 to 6

**Location:** `seo.py:323-329`, default 12
**Fix:** Change to 6. For niche clip topics (e.g., a specific fielding moment or umpire decision), VidIQ may return fewer than 12 keywords that pass strict grounding. Failing those clips entirely means no Short gets uploaded — a total loss vs a Short with 6-8 strong keywords.

### P2 — Medium Priority

#### R-10: Compress or Relocate Disclaimer

**Location:** `seo.py:238-239`, `seo.py:431-437`
**Fix:** Compress to ~80 chars: "Independent cricket commentary. All trademarks belong to their respective owners."
This frees ~270 chars of description budget for actual keyword-bearing content.

#### R-11: Content-Type-Aware Emoji Selection

**Current:** Default cricket ball emoji appended when LLM forgets one (`seo.py:777-787`)
**Fix:** Map clip content type (already classified in clip_selection) to specific emojis:
  - Wicket → fire/stumps emoji
  - Six/boundary → rocket/explosion
  - Controversy → warning/exclamation
  - Win/celebration → trophy
  - Analysis/debate → thinking face

#### R-12: Vary Description Section Headers

**Current:** Fixed headers across all Shorts ("🏏 MATCH DETAILS", "🔥 DETAILED CLIP ANALYSIS")
**Fix:** Provide 3-4 header variants per section in the prompt and let the LLM choose, or vary programmatically.

#### R-13: Track Metadata Source in Shorts Intelligence

**Fix:** Feed `metadata_source` ("vidiq_ranked_grounded_writer", "deterministic_grounded_fallback", "grounded_writer") and `promise_alignment_score` into the shorts_intelligence learner as tracked features. This enables A/B comparison of VidIQ-driven vs LLM-driven metadata performance.

### P3 — Nice to Have

#### R-14: Reduce `min_tag_chars` from 450 to 0

Tags are low-priority algorithm signals. The 450-char minimum causes unnecessary clip rejections. Remove the floor; keep 500-char API maximum.

#### R-15: Add VidIQ Trending Topics Tool

Use VidIQ's trending/competitor analysis tools in the remaining call budget (typically 1-2 unused calls out of 4-6). Feed real-time trending cricket queries into hashtag and title generation.

#### R-16: A/B Title Testing via YouTube API

Generate 2-3 title variants. Upload with VidIQ's top-scored title. After 48 hours, if views are below channel average, swap to variant 2 via YouTube's video.update API.

---

## 11. File Reference Map

| Component | File | Key Lines |
|---|---|---|
| System prompt | `automation/seo/seo.py` | 174-188 |
| User prompt template | `automation/seo/seo.py` | 190-264 |
| Vague filler filter (strips power words) | `automation/seo/seo.py` | 109-114 |
| AI slop phrases | `automation/seo/seo.py` | 122-128 |
| VidIQ title override (COMMENTED OUT) | `automation/seo/seo.py` | 1629 |
| VidIQ keyword research integration | `automation/seo/seo.py` | 1250-1293 |
| VidIQ title generation integration | `automation/seo/seo.py` | 1295-1348 |
| Grounding filter (over-rejection) | `automation/seo/seo.py` | 1121-1154 |
| Grounded title filter | `automation/seo/seo.py` | 1156-1169 |
| Keyword ranking (India volume priority) | `automation/seo/seo.py` | 1174-1183 |
| Primary search term injection suffix | `automation/seo/seo.py` | 1973-1981 |
| Fallback description builder (template) | `automation/seo/seo.py` | 349-453 |
| Title clean + filler strip | `automation/seo/seo.py` | 557-574 |
| Promise alignment scoring | `automation/seo/seo.py` | 586-595 |
| Entity-presence title gate | `automation/seo/seo.py` | 1826-1881 |
| Copy audit + entity scrub | `automation/seo/seo.py` | 1659-1691 |
| Cascading title repair | `automation/seo/seo.py` | 1736-1773 |
| Quality gate | `automation/seo/seo.py` | 813-876 |
| Enforce limits | `automation/seo/seo.py` | 642-774 |
| Title emoji enforcement | `automation/seo/seo.py` | 777-787 |
| Hashtag capping | `automation/seo/seo.py` | 668-697 |
| VidIQ client | `automation/seo/vidiq.py` | 60-264 |
| Context engine | `automation/seo/context_engine.py` | 1-270 |
| Trend research | `automation/seo/trends.py` | 1-616 |
| Entity grounding | `automation/seo/entity_grounding.py` | 1-245 |
| Cricket context | `automation/seo/cricket_context.py` | 1-512 |
| SEO config | `config.yaml` | 252-280 |
| Shorts Intelligence bridge | `shorts_intelligence/bridge.py` | 90-233 |
| Shorts Intelligence policy | `shorts_intelligence/policy.py` | 1-74 |

---

## Summary of Key Differences from v1 Audit

| v1 (Wrong) | v2 (Corrected) |
|---|---|
| "Description should be 150-400 chars" | Description length strategy (1500-4800 chars) is correct — it's algorithm metadata |
| "Tags have near-zero weight" | Tags are a valid secondary signal for API-side classification |
| "Search terms have negligible Shorts weight" | Search terms feed YouTube's topic clustering; VidIQ-driven terms are valuable |
| "Shorts are not discovered via search" | Shorts appear in YouTube Search results and the metadata enables that |
| Focused on reducing description length | Focused on improving description *quality* — less template, more natural prose |
| "Remove min_tag_chars entirely" | Reduce min_tag_chars (450→0 as soft gate) but keep tags well-packed |

The fundamental shift: **the description is metadata for the algorithm, not content for the viewer.** More high-quality, keyword-rich, naturally-written metadata gives YouTube more signal to classify and distribute the Short correctly.

---

*Report v2 — revised 2026-09-19. No code changes made.*
