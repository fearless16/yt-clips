## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- For cross-module "how does X relate to Y" questions, prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"` over grep — these traverse the graph's EXTRACTED + INFERRED edges instead of scanning files
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost)

## Personal Creator Workflow Requirements
- This is a personal YouTube automation system, not a SaaS platform.
- Prioritize reliability, maintainability, and repeatability over complexity.
- System should run mostly unattended after setup.
- Failures must be logged clearly with human-readable summaries.
- Every run should generate: export report, SEO report, performance snapshot, failure summary if any.
- Store all reports in structured Google Drive folders.

## Clip Quality Intelligence
- Detect and prioritize: emotional spikes, loud reactions, crowd hype, funny moments, arguments/debates, high-retention hooks in first 3 seconds.
- Avoid: dead air, slow intros, repetitive filler, blurry frames, low-energy segments.

## Retention Optimization
- Prefer clips with: strong first-frame hook, motion in first 2 seconds, speech density, emotional intensity, subtitle-friendly dialogue.
- Penalize low-engagement segments automatically.

## Subtitle System
- Generate clean burned-in subtitles.
- Auto-highlight important words.
- Keep subtitles readable on mobile.
- Avoid subtitle overflow.
- Sync subtitles tightly with speech.

## Thumbnail Expectations
- Thumbnail generation must stay lightweight.
- Use frame selection + enhancement first.
- Avoid expensive AI image generation unless necessary.
- Prefer high-expression human frames.

## Performance Tracking
- Compare: views, retention, CTR, upload timing, title performance, hashtag performance.
- Detect which style performs best over time.
- Generate simple daily insights automatically.

## Token Efficiency Rules
- Never send entire transcripts to AI.
- Summarize before AI calls.
- Batch SEO generation.
- Reuse metadata where possible.
- Cache trend/context data aggressively.

## Output Philosophy
- Viral-looking but not cringe.
- Human-feeling titles.
- Avoid obvious AI-generated SEO patterns.
- Shorts should feel manually edited by a premium creator.
