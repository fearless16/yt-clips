def test_learn_command_syncs_canonical_intelligence_without_url(monkeypatch, capsys):
    import automation.seo.analytics as analytics
    from automation import cli

    calls = []
    monkeypatch.setattr(
        analytics,
        "sync_clip_performance_from_youtube",
        lambda **kwargs: calls.append(kwargs) or 7,
    )
    monkeypatch.setattr(
        analytics,
        "generate_daily_insights",
        lambda **kwargs: {"shorts": 213, "cricket": 200, "model_observations": 200},
    )

    assert cli.main(["--learn"]) == 0
    assert calls == [{"config_path": "config.yaml"}]
    output = capsys.readouterr().out
    assert '"snapshots_added":7' in output
    assert '"cricket":200' in output


def test_worker_report_reads_canonical_report(monkeypatch, capsys):
    import worker
    import shorts_intelligence.reporting as reporting

    monkeypatch.setattr(
        reporting,
        "load_report",
        lambda config_path="config.yaml": {
            "store": {"shorts": 213, "cricket": 200, "snapshots": 381},
            "model": {"observations": 200},
            "recommendations": [],
            "evaluation": {"status": "ok", "beats_baseline": True},
            "top_shorts": [],
        },
    )

    worker.print_learnings()

    output = capsys.readouterr().out
    assert "213" in output
    assert "200" in output
    assert "SEOLearner" not in output


def test_disabled_intelligence_never_opens_source_or_database(monkeypatch):
    import automation.seo.analytics as analytics
    from shorts_intelligence.config import RuntimeConfig

    monkeypatch.setattr(
        RuntimeConfig,
        "from_yaml",
        classmethod(lambda cls, path="config.yaml": cls(
            channel_id="channel", handle="@channel", enabled=False
        )),
    )

    assert analytics.sync_clip_performance_from_youtube() == 0
    assert analytics.generate_daily_insights() == {"status": "disabled"}


def test_programmatic_learn_only_uses_canonical_sync(monkeypatch):
    import automation.orchestrator as orchestrator
    import automation.seo.analytics as analytics

    calls = []
    monkeypatch.setattr(orchestrator, "_CONFIG", {"paths": {}, "download": {}})
    monkeypatch.setattr(
        analytics,
        "sync_clip_performance_from_youtube",
        lambda **kwargs: calls.append(kwargs) or 4,
    )

    result = orchestrator.run(url="", learn_only=True)

    assert result.failures == []
    assert calls == [{"config_path": "config.yaml"}]
