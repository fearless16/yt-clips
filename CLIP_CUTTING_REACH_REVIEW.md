# Clip Cutting and Shorts Reach Review

## Executive summary

The current system is a substantial clip-selection and export pipeline, but it is not yet a closed-loop Shorts optimization system. It uses transcript, cricket-context, audio-energy, and agent-score heuristics to find and rank candidate moments. The most important engineering concern found in this review is that transcript splitting can be undone by a subsequent caption-fragment merge. The hook audit also runs after export, so it currently reports opening quality without influencing which moments are selected.

This was a static code and configuration review. No exported clips were visually reviewed and no current YouTube Analytics dataset was queried as part of this review.

## Findings

### 1. Transcript split/merge interaction may hide moments

In `automation/clip_selection/pipeline.py`, `_split_segments_on_pauses()` splits long transcript blocks, including blocks without word timings. `_prepare_complete_thoughts()` subsequently calls `_merge_caption_fragments()`, which joins adjacent fragments unless a gap exceeds one second or sentence punctuation closes the fragment. For punctuation-free speech, this can undo the earlier split. Candidate windows are then built from the resulting thought starts, potentially preventing discovery of strong moments inside a long block.

**Recommended action:** add regression cases for long punctuation-free transcripts, both with and without word timings. Preserve intentional split boundaries through thought preparation and verify candidate start times remain available inside the original long block.

### 2. Hook audit is post-selection

`automation/orchestrator.py` runs the hook audit on already-exported clips. The audit in `automation/clip_selection/hook_auditor.py` cannot affect selection or cut timing. Its textual/audio proxies may flag possible opening issues, but they are not a visual or audience-response assessment.

**Recommended action:** use hook evidence before final selection or create a review/re-cut gate for weak openings. Treat the audit as a heuristic, not proof of audience retention.

### 3. Duration target is not a hard output limit

`config.yaml` permits source clips up to 38 seconds, targets 15 seconds, and caps speed-up at 1.25x. The pipeline's speed factor therefore cannot reduce every selected clip to 15 seconds; a 38-second window can remain roughly 30 seconds. This can be appropriate when the complete thought needs the time, but should not be interpreted as a 15-second export guarantee.

**Recommended action:** clarify whether 15 seconds is a preference or a hard cap. Do not speed up or truncate a complete thought solely to satisfy an assumed algorithmic duration target.

### 4. Predicted scores are not viewer outcomes

Candidate ranking uses transcript, audio energy, cricket relevance, and agent scores. These are discovery proxies, not evidence that viewers chose to watch or enjoyed the final Short. YouTube's Shorts guidance describes personalized recommendations and signals including whether viewers choose to watch or ignore a Short, watch duration, average percentage viewed, and satisfaction signals.

**Recommended action:** evaluate published clips using YouTube Analytics outcomes, including engaged views, “How many chose to view” where available, average view duration/percentage viewed, likes, and subscribers gained. Check whether the choice-to-view signal can be collected by the existing analytics workflow before designing a schema change.

### 5. Human visual review is still needed

Automated transcript and audio checks cannot establish whether a clip makes sense without prior context, whether the emotional payoff lands, or whether the portrait crop and on-screen text remain readable on a phone.

**Recommended action:** sample-review exported clips before publication, especially the opening seconds, context, crop, captions, and ending. This review was not performed in the current audit.

### 6. Publishing needs controlled experiments and a clear next step

Reach can vary with moment/topic demand, audience fit, competing content, packaging, and editing. A high-performing clip does not by itself establish which edit caused the result. Shorts can also find viewers without turning them into returning viewers.

**Recommended action:** test one editing variable at a time on comparable cricket moments. Track outcomes over enough uploads to avoid treating one viral outlier as a rule. Where appropriate, connect a Short to a related Short or longer video and measure subscribers and further viewing.

## Research notes

### Official YouTube guidance

- [Search and discovery tips for Shorts](https://support.google.com/youtube/answer/11914225): recommendations are personalized; performance examples include watch/ignore choice, watch duration, average percentage viewed, and satisfaction; competition and audience interest affect reach. The guidance does not provide a universal viral threshold or minimum posting cadence.
- [Shorts analytics overview](https://support.google.com/youtube/answer/12942217): documents Shorts analytics, including views, Shorts Feed appearances, and “How many chose to view.”
- [Change to how Shorts views are counted](https://support.google.com/youtube/thread/333869549): since March 31, 2025, a public Shorts view can count when playback starts; “engaged views” remain useful for understanding viewers who chose to continue.
- [Understand three-minute Shorts](https://support.google.com/youtube/answer/15424877): eligible square/vertical uploads on standard channels can be Shorts up to three minutes. Eligibility is not a recommendation to extend clip duration.

### Reddit creator discussions

Reddit pages were discovered through search results because direct fetching was blocked. Relevant communities and discussions include [r/NewTubers](https://www.reddit.com/r/NewTubers/) and [r/PartneredYoutube](https://www.reddit.com/r/PartneredYoutube/), including threads on [retention vs. distribution](https://www.reddit.com/r/NewTubers/comments/1idglg5/trying_to_understand_retention_rate_for_shorts/), [the Shorts view-count change](https://www.reddit.com/r/NewTubers/comments/1lfqx29/youtube_changed_the_way_shorts_feed_views_are/), and [Shorts performance variability](https://www.reddit.com/r/PartneredYoutube/comments/1nqbpo5/did_something_change_with_shorts/).

Repeated creator reports say strong swipe/retention metrics do not guarantee continued distribution and that topic/audience fit may matter. These are anecdotal, self-reported observations, often contradictory; they are useful as experiment ideas, not verified ranking rules.

## Prioritized plan

1. Test and correct the transcript split/merge behavior with regression tests.
2. Make hook-quality findings actionable before final selection, while retaining human review for visual/context judgments.
3. Audit analytics collection for choice-to-view and engaged-view outcomes; use existing canonical `shorts_intelligence/` storage rather than creating a second learner/database.
4. Run controlled, channel-specific edit tests and assess outcomes across comparable clips.
5. Add a review check for context, crop, readable captions, and satisfying endings before publication.

## Confidence and limitations

- **High confidence:** the described split/merge ordering exists in the current code; hook audit invocation occurs after exports; duration settings do not guarantee a 15-second output under the configured speed cap.
- **Medium confidence:** these code paths reduce discovery or hook quality in real generated clips. Confirm with representative transcripts and source/export playback.
- **Low confidence:** Reddit theories about specific algorithm changes or fixed swipe/retention thresholds. Do not encode these as selection rules without channel-specific outcome evidence.
