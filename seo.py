"""seo.py — Backward-compatible stub. All logic moved to automation/seo/seo.py."""
from automation.seo.seo import *
from automation.seo.seo import (
    SEOGenerationError, _attempt_seo_generation, _enforce_limits,
    _default_hashtags, generate_clip_seo, generate_seo_for_exported_clip,
    process_all_seo, retry_failed_seo, SUGGEST_CACHE, TREND_CACHE, log,
)
