from shorts_intelligence.config import RuntimeConfig


def test_runtime_config_is_driven_by_own_section_with_channel_fallback():
    config = RuntimeConfig.from_mapping({
        "channel": {"id": "channel-1"},
        "shorts_intelligence": {
            "enabled": True,
            "db_path": "data/custom.db",
            "handle": "@channel",
            "shadow_mode": True,
            "learner": {
                "prior_strength": 20,
                "min_segment_samples": 9,
                "half_life_days": 60,
            },
        },
    })

    assert config.channel_id == "channel-1"
    assert config.db_path == "data/custom.db"
    assert config.learner.prior_strength == 20
    assert config.learner.min_segment_samples == 9
    assert config.learner.half_life_days == 60


def test_runtime_config_rejects_missing_channel_id():
    try:
        RuntimeConfig.from_mapping({"shorts_intelligence": {"enabled": True}})
    except ValueError as exc:
        assert "channel_id" in str(exc)
    else:
        raise AssertionError("missing channel id was accepted")

