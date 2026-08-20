"""Fail-soft local bridge from existing pipelines into shadow mode."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from shorts_intelligence.config import RuntimeConfig
from shorts_intelligence.policy import SelectionPolicy
from shorts_intelligence.store import ShortsStore


def record_exported(
    root_config: dict[str, Any],
    highlights: Iterable[dict[str, Any]],
    exported: Iterable[str | Path],
) -> int:
    """Capture features only for clips that actually reached export."""
    config = RuntimeConfig.from_mapping(root_config)
    if not config.enabled:
        return 0
    paths = [Path(path) for path in exported]
    if not paths:
        return 0
    by_id = {
        str(item.get("id")): item
        for item in highlights
        if isinstance(item, dict) and item.get("id")
    }
    recorded = 0
    with ShortsStore(config.db_path, channel_id=config.channel_id) as store:
        for path in paths:
            item = by_id.get(path.stem)
            if item is None:
                continue
            duration = max(0.0, _float(item.get("end")) - _float(item.get("start")))
            score = _float(item.get("final_score", item.get("score")))
            speed = max(0.0, _float(item.get("speed_factor", 1.0)))
            agent_scores = item.get("agent_scores") or {}
            hook = item.get("hook_type")
            if not hook and isinstance(agent_scores, dict):
                hook = agent_scores.get("hook_expert", {}).get("reasoning")
            features = {
                "duration_bucket": _duration_bucket(duration),
                "selection_score_bucket": _score_bucket(score),
                "speed_bucket": _speed_bucket(speed),
                "complete_thought": "true",
            }
            segments = item.get("intelligence_segments") or []
            if segments:
                features["intelligence_segment"] = str(segments[0])
            if hook:
                features["hook_type"] = str(hook).casefold()
            metadata_path = path.with_name(f"{path.stem}_metadata.json")
            if metadata_path.exists():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    title = str(metadata.get("title", ""))
                    description = str(metadata.get("description", ""))
                    tags = metadata.get("tags") or []
                    hashtags = metadata.get("hashtags") or []
                    features.update({
                        "seo_title_length_bucket": _length_bucket(len(title), (40, 55, 70)),
                        "seo_description_length_bucket": _length_bucket(
                            len(description), (1000, 2500, 3800)
                        ),
                        "seo_tag_count_bucket": _length_bucket(len(tags), (10, 20, 30)),
                        "seo_hashtag_count_bucket": _length_bucket(len(hashtags), (5, 10, 15)),
                    })
                    packaging_version = metadata.get("packaging_version")
                    if packaging_version:
                        features["seo_packaging_version"] = str(packaging_version).casefold()
                    if "promise_alignment_score" in metadata:
                        alignment = round(_float(metadata["promise_alignment_score"]) * 100)
                        features["seo_promise_alignment_bucket"] = _length_bucket(
                            alignment, (50, 75, 90)
                        )
                    primary_queries = metadata.get("primary_search_terms") or []
                    if isinstance(primary_queries, list):
                        features["seo_primary_query_count_bucket"] = _length_bucket(
                            len(primary_queries), (1, 2, 3)
                        )
                    for key in ("provider", "model"):
                        if metadata.get(key):
                            features[f"seo_{key}"] = str(metadata[key]).casefold()
                except (OSError, ValueError, TypeError):
                    pass
            store.record_production(
                clip_id=f"{path.parent.name}/{path.stem}",
                transcript=str(item.get("text", "")),
                features=features,
            )
            recorded += 1
    return recorded


def record_upload(root_config: dict[str, Any], clip_id: str, video_id: str | None) -> bool:
    """Save a local upload link; outcomes join only after shelf sync."""
    config = RuntimeConfig.from_mapping(root_config)
    if not config.enabled or not video_id:
        return False
    with ShortsStore(config.db_path, channel_id=config.channel_id) as store:
        store.record_upload(clip_id, video_id)
    return True


def shadow_status(root_config: dict[str, Any]) -> dict[str, Any]:
    """Return compact local shadow state without network access."""
    config = RuntimeConfig.from_mapping(root_config)
    if not config.enabled:
        return {"mode": "disabled"}
    with ShortsStore(config.db_path, channel_id=config.channel_id) as store:
        model = store.latest_model()
        return {
            "mode": "shadow" if config.shadow_mode else "active",
            "production_records": store.production_count(),
            "catalog": store.summary()["shorts"],
            "model_observations": int(model["observations"]) if model else 0,
        }


def recommendation_context(root_config: dict[str, Any]) -> str:
    """Return evidence-only prompt context; empty means no proven channel signal."""
    config = RuntimeConfig.from_mapping(root_config)
    if not config.enabled:
        return ""
    with ShortsStore(config.db_path, channel_id=config.channel_id) as store:
        model = store.latest_model()
        recommendations = store.active_recommendations(config.max_recommendations)
    if not model or not recommendations:
        return ""
    lines = [
        f"Evidence base: {int(model['observations'])} cricket Shorts from this channel."
    ]
    for item in recommendations:
        lines.append(
            "- {name}={value}: supported positive effect {effect:.3f} "
            "(lower bound {lower:.3f}, n={n}).".format(
                name=item["feature_name"],
                value=item["feature_value"],
                effect=float(item["effect"]),
                lower=float(item["effect_lower_bound"]),
                n=int(item["sample_count"]),
            )
        )
    lines.append("Treat these as soft channel priors; transcript and verified match facts remain authoritative.")
    return "\n".join(lines)


def apply_selection_policy(
    root_config: dict[str, Any],
    candidates: Iterable[dict[str, Any]],
) -> int:
    """Apply only bounded evidence scores; never alter clip boundaries."""
    config = RuntimeConfig.from_mapping(root_config)
    if not config.enabled:
        return 0
    with ShortsStore(config.db_path, channel_id=config.channel_id) as store:
        recommendations = store.active_recommendations()
    policy = SelectionPolicy(
        recommendations,
        shadow_mode=config.shadow_mode,
        max_adjustment_points=config.max_selection_adjustment_points,
    )
    matched = 0
    for candidate in candidates:
        result = policy.evaluate(
            duration_seconds=max(
                0.0,
                _float(candidate.get("end")) - _float(candidate.get("start")),
            ),
            complete_thought=bool(candidate.get("complete_thought", True)),
        )
        candidate["intelligence_shadow_adjustment"] = result.shadow_adjustment
        candidate["intelligence_adjustment"] = result.applied_adjustment
        candidate["intelligence_segments"] = list(result.matched_segments)
        if not result.eligible:
            candidate["should_reject"] = True
            candidate.setdefault("rejection_reasons", []).append(result.reason)
        if result.matched_segments:
            matched += 1
        if result.applied_adjustment:
            candidate["final_score"] = float(candidate.get("final_score", 0.0)) + result.applied_adjustment
    return matched


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _duration_bucket(seconds: float) -> str:
    if seconds < 15:
        return "under_15"
    if seconds < 25:
        return "15_24"
    if seconds < 40:
        return "25_39"
    if seconds < 60:
        return "40_59"
    return "60_plus"


def _score_bucket(score: float) -> str:
    lower = max(0, min(80, int(score // 20) * 20))
    return f"{lower}_{lower + 19}"


def _speed_bucket(speed: float) -> str:
    if speed <= 1.05:
        return "natural"
    if speed <= 1.25:
        return "light"
    if speed <= 1.5:
        return "fast"
    return "very_fast"


def _length_bucket(value: int, limits: tuple[int, int, int]) -> str:
    low, medium, high = limits
    if value < low:
        return f"under_{low}"
    if value < medium:
        return f"{low}_{medium - 1}"
    if value < high:
        return f"{medium}_{high - 1}"
    return f"{high}_plus"
