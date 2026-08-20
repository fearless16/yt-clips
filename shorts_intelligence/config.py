"""Standalone configuration boundary for the Shorts Intelligence plugin."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shorts_intelligence.learner import LearnerConfig
from shorts_intelligence.youtube_source import SourceConfig


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Complete runtime config with no dependency on legacy config helpers."""

    channel_id: str
    handle: str
    db_path: str = "shorts_intelligence.db"
    token_path: str = "yt_analytics_token.json"
    local_metadata_root: str = "shorts"
    enabled: bool = True
    shadow_mode: bool = True
    sync_on_pipeline: bool = False
    max_recommendations: int = 8
    max_selection_adjustment_points: float = 3.0
    learner: LearnerConfig = LearnerConfig()

    @classmethod
    def from_mapping(cls, root: dict[str, Any]) -> "RuntimeConfig":
        """Build from the root YAML mapping and validate required identity."""
        section = root.get("shorts_intelligence") or {}
        youtube = root.get("youtube") or {}
        channel = root.get("channel") or {}
        channel_id = str(
            section.get("channel_id") or youtube.get("channel_id") or channel.get("id") or ""
        ).strip()
        if not channel_id:
            raise ValueError("shorts_intelligence channel_id is required")
        learner_raw = section.get("learner") or {}
        learner = LearnerConfig(
            prior_strength=float(learner_raw.get("prior_strength", 12.0)),
            min_segment_samples=int(learner_raw.get("min_segment_samples", 6)),
            half_life_days=float(learner_raw.get("half_life_days", 120.0)),
            confidence_z=float(learner_raw.get("confidence_z", 1.64)),
        )
        return cls(
            channel_id=channel_id,
            handle=str(section.get("handle") or "@CricketWithPrajjwal2.0"),
            db_path=str(section.get("db_path") or "shorts_intelligence.db"),
            token_path=str(section.get("token_path") or youtube.get("token_path") or "yt_analytics_token.json"),
            local_metadata_root=str(section.get("local_metadata_root") or "shorts"),
            enabled=bool(section.get("enabled", True)),
            shadow_mode=bool(section.get("shadow_mode", True)),
            sync_on_pipeline=bool(section.get("sync_on_pipeline", False)),
            max_recommendations=max(0, int(section.get("max_recommendations", 8))),
            max_selection_adjustment_points=max(
                0.0, float(section.get("max_selection_adjustment_points", 3.0))
            ),
            learner=learner,
        )

    @classmethod
    def from_yaml(cls, path: str | Path = "config.yaml") -> "RuntimeConfig":
        """Load config from YAML lazily so the core remains standard-library-only."""
        import yaml

        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError("config root must be a mapping")
        return cls.from_mapping(payload)

    def source_config(self) -> SourceConfig:
        """Project runtime settings into the YouTube adapter config."""
        return SourceConfig(
            channel_id=self.channel_id,
            handle=self.handle,
            token_path=self.token_path,
            local_metadata_root=self.local_metadata_root,
        )
