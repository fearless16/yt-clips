"""automation.seo — SEO generation, learning, analytics, and trend intelligence."""

from .seo import process_all_seo, generate_seo_for_exported_clip
from .seo_learner import SEOLearner, run_auto_benchmark, get_best_model
from .analytics import generate_daily_insights
from .trends import TEAM_MAPPINGS, fetch_youtube_suggestions, get_trending_context
