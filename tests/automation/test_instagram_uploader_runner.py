"""Tests for automation.instagram.uploader.publish_reel and
automation.instagram.runner (stage checkpoints, marker protocol,
retry_failed_insta).

Code is written against the PLAN.md frozen GraphClient Protocol duck-type;
the local FakeGraphClient stands in until/alongside the real client.
"""

import json
from pathlib import Path

import pytest

from utils.config import Config, _config_cache

MEDIA_ID = "17900000000000001"


@pytest.fixture(autouse=True)
def _no_upload_post_creds(monkeypatch):
    """ai_client's load_dotenv() leaks the real key into the session; tests
    here stub the Graph flow and rely on the provider factory bailing out."""
    monkeypatch.setenv("UPLOAD_POST_API_KEY", "")
PERMALINK = "https://www.instagram.com/reel/fake123/"


class FakeGraphClient:
    """Duck-typed GraphClient per PLAN.md frozen contract."""

    def __init__(self, statuses=None, publish_result=MEDIA_ID,
                 permalink=PERMALINK, create_error=None):
        self.statuses = list(statuses or ["FINISHED"])
        self.publish_result = publish_result
        self.permalink = permalink
        self.create_error = create_error
        self.calls = []

    def _next_status(self):
        if len(self.statuses) > 1:
            item = self.statuses.pop(0)
        else:
            item = self.statuses[0]
        if isinstance(item, dict):
            return item
        return {"status_code": item}

    def create_reels_container(self, *, caption, share_to_feed=True,
                               audio_name=None):
        self.calls.append({"op": "create", "caption": caption,
                           "audio_name": audio_name})
        if self.create_error is not None:
            raise self.create_error
        return "cid_fake_1"

    def upload_video_bytes(self, creation_id, video_path):
        self.calls.append({"op": "upload", "creation_id": creation_id})

    def container_status(self, creation_id):
        self.calls.append({"op": "status", "creation_id": creation_id})
        return self._next_status()

    def publish_container(self, creation_id):
        self.calls.append({"op": "publish", "creation_id": creation_id})
        return self.publish_result

    def media_permalink(self, media_id):
        self.calls.append({"op": "permalink", "media_id": media_id})
        return self.permalink


@pytest.fixture
def insta_enabled(monkeypatch):
    monkeypatch.delenv("YT_CLIPS_SKIP_INSTAGRAM", raising=False)
    monkeypatch.setitem(_config_cache, "config.yaml",
                        Config({"instagram": {"enabled": True}}))


@pytest.fixture
def clean_hooks():
    from automation.instagram import runner
    runner._HOOKS.clear()
    yield runner
    runner._HOOKS.clear()


# ---------------------------------------------------------------------------
# uploader.publish_reel
# ---------------------------------------------------------------------------

class TestPublishReelHappyPath:

    def test_full_flow_returns_media_id_and_permalink(self):
        from automation.instagram.uploader import publish_reel
        client = FakeGraphClient(statuses=["IN_PROGRESS", "FINISHED"])
        out = publish_reel(
            client, Path("clip.mp4"), "Kohli ne maara six!",
            audio_name="Six Moment – CricketWithPrajjwal",
            poll_interval_s=0,
        )
        assert out == {"media_id": MEDIA_ID, "permalink": PERMALINK}
        ops = [c["op"] for c in client.calls]
        assert ops == ["create", "upload", "status", "status",
                       "publish", "permalink"]
        assert client.calls[0]["audio_name"] == \
            "Six Moment – CricketWithPrajjwal"
        assert client.calls[1]["creation_id"] == "cid_fake_1"

    def test_immediately_finished_needs_single_poll(self):
        from automation.instagram.uploader import publish_reel
        client = FakeGraphClient(statuses=["FINISHED"])
        out = publish_reel(client, Path("c.mp4"), "cap", poll_interval_s=0)
        assert out["media_id"] == MEDIA_ID
        assert [c["op"] for c in client.calls].count("status") == 1


class TestPublishAmbiguityRecovery:

    def test_publish_none_then_published_recovers_media_id(self):
        from automation.instagram.uploader import publish_reel
        client = FakeGraphClient(
            statuses=["FINISHED", {"status_code": "PUBLISHED", "id": "179recovered"}],
            publish_result=None,
        )
        out = publish_reel(client, Path("c.mp4"), "cap", poll_interval_s=0)
        assert out["media_id"] == "179recovered"
        assert out["permalink"] == PERMALINK
        assert client.calls[-1]["media_id"] == "179recovered"

    def test_publish_none_and_not_published_raises(self):
        from automation.instagram.uploader import InstagramUploadError
        from automation.instagram.uploader import publish_reel
        client = FakeGraphClient(
            statuses=["FINISHED", "FINISHED"], publish_result=None)
        with pytest.raises(InstagramUploadError) as exc:
            publish_reel(client, Path("c.mp4"), "cap", poll_interval_s=0)
        assert exc.value.stage == "PUBLISH"


class TestPublishPolling:

    def test_transient_errors_then_finished_succeeds(self):
        from automation.instagram.uploader import publish_reel
        client = FakeGraphClient(
            statuses=["ERROR", "ERROR", "IN_PROGRESS", "FINISHED"])
        out = publish_reel(client, Path("c.mp4"), "cap", poll_interval_s=0)
        assert out["media_id"] == MEDIA_ID

    def test_three_consecutive_errors_give_up(self):
        from automation.instagram.uploader import InstagramUploadError
        from automation.instagram.uploader import publish_reel
        client = FakeGraphClient(statuses=["ERROR", "ERROR", "ERROR"])
        with pytest.raises(InstagramUploadError) as exc:
            publish_reel(client, Path("c.mp4"), "cap", poll_interval_s=0)
        assert exc.value.stage == "POLL_CONTAINER"
        assert "publish" not in [c["op"] for c in client.calls]

    def test_error_streak_resets_on_non_error(self):
        from automation.instagram.uploader import InstagramUploadError
        from automation.instagram.uploader import publish_reel
        client = FakeGraphClient(
            statuses=["ERROR", "ERROR", "IN_PROGRESS", "ERROR", "ERROR",
                      "ERROR", "FINISHED"])
        with pytest.raises(InstagramUploadError):
            publish_reel(client, Path("c.mp4"), "cap", poll_interval_s=0)

    def test_unknown_status_is_not_ready(self):
        from automation.instagram.uploader import publish_reel
        client = FakeGraphClient(statuses=["WHO_KNOWS", "EXPIRED_LIKE",
                                           "FINISHED"])
        out = publish_reel(client, Path("c.mp4"), "cap", poll_interval_s=0)
        assert out["media_id"] == MEDIA_ID

    def test_timeout_raises_listing_stage_reached(self):
        from automation.instagram.uploader import InstagramUploadError
        from automation.instagram.uploader import publish_reel
        client = FakeGraphClient(statuses=["IN_PROGRESS"])
        with pytest.raises(InstagramUploadError) as exc:
            publish_reel(client, Path("c.mp4"), "cap",
                         poll_interval_s=0, timeout_s=0.05)
        assert exc.value.stage == "POLL_CONTAINER"
        assert "POLL_CONTAINER" in str(exc.value)


# ---------------------------------------------------------------------------
# runner.process_instagram_for_clip
# ---------------------------------------------------------------------------

class TestRunnerHappyPath:

    def test_full_pipeline_writes_done_state(self, tmp_path, insta_enabled,
                                             clean_hooks):
        runner = clean_hooks
        clip = tmp_path / "clipA"
        clip.mkdir()
        (clip / "clip.mp4").write_bytes(b"\x00\x01")
        seen = {}
        runner.register_stage_hook(
            "run_evidence_stage",
            lambda *a, **k: seen.setdefault("evidence", True))
        runner.register_stage_hook(
            "run_caption_stage",
            lambda *a, **k: "Kohli ne maara six! #indvsaus #cricketreels")
        client = FakeGraphClient(statuses=["FINISHED"])
        runner.register_stage_hook("make_graph_client", lambda: client)

        media_id = runner.process_instagram_for_clip(
            clip, "kohli six transcript", "Ind vs Aus", "desc")

        assert media_id == MEDIA_ID
        assert seen.get("evidence") is True
        assert client.calls[0]["caption"].startswith("Kohli ne maara six!")
        state = json.loads(
            (clip / "clip_insta_state.json").read_text(encoding="utf-8"))
        assert {"stage", "attempts", "last_error", "updated_at"} <= set(state)
        assert state["stage"] == "DONE"
        assert state["attempts"] >= 1
        assert state["last_error"] is None
        assert state["media_id"] == MEDIA_ID
        assert not (clip / "clip_insta_failed.json").exists()

    def test_resume_from_caption_done_skips_evidence_and_caption(
            self, tmp_path, insta_enabled, clean_hooks):
        runner = clean_hooks
        clip = tmp_path / "clipB"
        clip.mkdir()
        (clip / "v.mp4").write_bytes(b"\x00")
        (clip / "clip_insta_state.json").write_text(json.dumps({
            "stage": "CAPTION", "attempts": 2, "last_error": None,
            "updated_at": "2026-08-23T00:00:00+00:00",
            "caption": "Stored cap #ig",
        }), encoding="utf-8")

        def _boom(*a, **k):
            raise AssertionError("completed stage was re-run")

        runner.register_stage_hook("run_evidence_stage", _boom)
        runner.register_stage_hook("run_caption_stage", _boom)
        client = FakeGraphClient(statuses=["FINISHED"])
        runner.register_stage_hook("make_graph_client", lambda: client)

        media_id = runner.process_instagram_for_clip(clip, "tr", "t", "d")
        assert media_id == MEDIA_ID
        assert client.calls[0]["caption"] == "Stored cap #ig"

    def test_resume_from_evidence_only_skips_evidence(self, tmp_path,
                                                      insta_enabled,
                                                      clean_hooks):
        runner = clean_hooks
        clip = tmp_path / "clipE"
        clip.mkdir()
        (clip / "v.mp4").write_bytes(b"\x00")
        (clip / "clip_insta_state.json").write_text(json.dumps({
            "stage": "EVIDENCE", "attempts": 1, "last_error": None,
            "updated_at": "2026-08-23T00:00:00+00:00",
        }), encoding="utf-8")

        runner.register_stage_hook(
            "run_evidence_stage",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("redo!")))
        runner.register_stage_hook(
            "run_caption_stage", lambda *a, **k: "Fresh cap")
        client = FakeGraphClient(statuses=["FINISHED"])
        runner.register_stage_hook("make_graph_client", lambda: client)

        assert runner.process_instagram_for_clip(clip, "tr", "t", "d") == \
            MEDIA_ID
        assert client.calls[0]["caption"] == "Fresh cap"

    def test_done_state_short_circuits_without_new_upload(
            self, tmp_path, insta_enabled, clean_hooks):
        runner = clean_hooks
        clip = tmp_path / "clipD"
        clip.mkdir()
        (clip / "clip_insta_state.json").write_text(json.dumps({
            "stage": "DONE", "attempts": 1, "last_error": None,
            "updated_at": "2026-08-23T00:00:00+00:00",
            "media_id": "179already",
        }), encoding="utf-8")

        def _boom(*a, **k):
            raise AssertionError("client constructed after DONE")

        runner.register_stage_hook("make_graph_client", _boom)
        runner.register_stage_hook("run_caption_stage", _boom)

        assert runner.process_instagram_for_clip(clip, "tr", "t", "d") == \
            "179already"


class TestRunnerFailureMarker:

    def test_caption_failure_writes_marker_and_returns_none(
            self, tmp_path, insta_enabled, clean_hooks):
        runner = clean_hooks
        clip = tmp_path / "clipF"
        clip.mkdir()

        def _explode(*a, **k):
            raise RuntimeError("caption LLM exploded")

        runner.register_stage_hook("run_evidence_stage", lambda *a, **k: {})
        runner.register_stage_hook("run_caption_stage", _explode)
        runner.register_stage_hook(
            "make_graph_client",
            lambda: (_ for _ in ()).throw(AssertionError("no client")))

        result = runner.process_instagram_for_clip(
            clip, "six transcript", "Match Title", "Match Desc")
        assert result is None

        marker = json.loads((clip / "clip_insta_failed.json").read_text(
            encoding="utf-8"))
        assert marker["clip_id"] == "clipF"
        assert marker["transcript"] == "six transcript"
        assert marker["video_title"] == "Match Title"
        assert marker["video_description"] == "Match Desc"
        assert marker["stage"] == "CAPTION"
        assert "caption LLM exploded" in marker["error"]

        state = json.loads((clip / "clip_insta_state.json").read_text(
            encoding="utf-8"))
        assert state["attempts"] >= 1
        assert "caption LLM exploded" in state["last_error"]

    def test_upload_stage_failure_marks_upload_stage(
            self, tmp_path, insta_enabled, clean_hooks):
        runner = clean_hooks
        clip = tmp_path / "clipG"
        clip.mkdir()
        (clip / "v.mp4").write_bytes(b"\x00")
        runner.register_stage_hook("run_evidence_stage", lambda *a, **k: {})
        runner.register_stage_hook(
            "run_caption_stage", lambda *a, **k: "Cap text")
        client = FakeGraphClient(create_error=RuntimeError("rupload 500"))
        runner.register_stage_hook("make_graph_client", lambda: client)

        assert runner.process_instagram_for_clip(clip, "tr", "t", "d") is None
        marker = json.loads((clip / "clip_insta_failed.json").read_text(
            encoding="utf-8"))
        assert marker["stage"] == "UPLOAD"
        assert "rupload 500" in marker["error"]

    def test_exception_never_propagates_missing_video_file(
            self, tmp_path, insta_enabled, clean_hooks):
        runner = clean_hooks
        clip = tmp_path / "clipNoVideo"
        clip.mkdir()
        runner.register_stage_hook("run_evidence_stage", lambda *a, **k: {})
        runner.register_stage_hook(
            "run_caption_stage", lambda *a, **k: "Cap")
        runner.register_stage_hook(
            "make_graph_client", lambda: FakeGraphClient())
        try:
            out = runner.process_instagram_for_clip(clip, "tr", "t", "d")
        except Exception as exc:
            raise AssertionError(f"propagated: {exc}")
        assert out is None
        assert (clip / "clip_insta_failed.json").exists()


class TestRunnerSkipPaths:

    def test_env_skip_returns_none_with_zero_network(
            self, tmp_path, monkeypatch, clean_hooks):
        runner = clean_hooks
        monkeypatch.setenv("YT_CLIPS_SKIP_INSTAGRAM", "1")
        monkeypatch.setitem(_config_cache, "config.yaml",
                            Config({"instagram": {"enabled": True}}))
        clip = tmp_path / "clipSkip"
        clip.mkdir()
        built = []
        runner.register_stage_hook(
            "make_graph_client", lambda: built.append(1) or FakeGraphClient())

        assert runner.process_instagram_for_clip(clip, "tr", "t", "d") is None
        assert built == []
        assert not (clip / "clip_insta_state.json").exists()
        assert not (clip / "clip_insta_failed.json").exists()

    def test_config_disabled_returns_none_with_zero_network(
            self, tmp_path, monkeypatch, clean_hooks):
        runner = clean_hooks
        monkeypatch.delenv("YT_CLIPS_SKIP_INSTAGRAM", raising=False)
        monkeypatch.setitem(_config_cache, "config.yaml", Config({}))
        clip = tmp_path / "clipCfgOff"
        clip.mkdir()
        built = []
        runner.register_stage_hook(
            "make_graph_client", lambda: built.append(1) or FakeGraphClient())

        assert runner.process_instagram_for_clip(clip, "tr", "t", "d") is None
        assert built == []
        assert not (clip / "clip_insta_state.json").exists()

    def test_force_bypasses_skip_paths(self, tmp_path, monkeypatch,
                                       clean_hooks, insta_enabled):
        runner = clean_hooks
        monkeypatch.setenv("YT_CLIPS_SKIP_INSTAGRAM", "1")
        clip = tmp_path / "clipForce"
        clip.mkdir()
        (clip / "v.mp4").write_bytes(b"\x00")
        runner.register_stage_hook("run_evidence_stage", lambda *a, **k: {})
        runner.register_stage_hook(
            "run_caption_stage", lambda *a, **k: "Forced cap")
        client = FakeGraphClient(statuses=["FINISHED"])
        runner.register_stage_hook("make_graph_client", lambda: client)

        assert runner.process_instagram_for_clip(
            clip, "tr", "t", "d", force=True) == MEDIA_ID


# ---------------------------------------------------------------------------
# runner.retry_failed_insta
# ---------------------------------------------------------------------------

class TestRetryFailedInsta:

    @staticmethod
    def _fixture_clip(tmp_path):
        shorts = tmp_path / "shorts"
        clip = shorts / "run1" / "clipR"
        clip.mkdir(parents=True)
        (clip / "v.mp4").write_bytes(b"\x00")
        (clip / "clip_insta_failed.json").write_text(json.dumps({
            "clip_id": "clipR",
            "transcript": "retry transcript",
            "video_title": "Retry Title",
            "video_description": "Retry Desc",
            "stage": "UPLOAD",
            "error": "old failure",
        }), encoding="utf-8")
        return shorts, clip

    def test_retry_end_to_end_recovers_and_deletes_marker(
            self, tmp_path, insta_enabled, clean_hooks):
        runner = clean_hooks
        shorts, clip = self._fixture_clip(tmp_path)
        runner.register_stage_hook("run_evidence_stage", lambda *a, **k: {})
        runner.register_stage_hook(
            "run_caption_stage", lambda *a, **k: "Recovered cap")
        client = FakeGraphClient(statuses=["FINISHED"])
        runner.register_stage_hook("make_graph_client", lambda: client)

        summary = runner.retry_failed_insta(str(shorts))
        assert summary == {"retried": 1, "recovered": 1, "still_failed": 0}
        assert not (clip / "clip_insta_failed.json").exists()
        state = json.loads((clip / "clip_insta_state.json").read_text(
            encoding="utf-8"))
        assert state["stage"] == "DONE"
        assert client.calls[0]["caption"] == "Recovered cap"

    def test_retry_still_failing_keeps_marker_and_counts(
            self, tmp_path, insta_enabled, clean_hooks):
        runner = clean_hooks
        shorts, clip = self._fixture_clip(tmp_path)

        def _explode(*a, **k):
            raise RuntimeError("still broken")

        runner.register_stage_hook("run_evidence_stage", lambda *a, **k: {})
        runner.register_stage_hook("run_caption_stage", _explode)
        runner.register_stage_hook(
            "make_graph_client",
            lambda: (_ for _ in ()).throw(AssertionError("no client")))

        summary = runner.retry_failed_insta(str(shorts))
        assert summary == {"retried": 1, "recovered": 0, "still_failed": 1}
        assert (clip / "clip_insta_failed.json").exists()

    def test_retry_bypasses_skip_flag(self, tmp_path, monkeypatch,
                                      clean_hooks, insta_enabled):
        runner = clean_hooks
        monkeypatch.setenv("YT_CLIPS_SKIP_INSTAGRAM", "1")
        shorts, clip = self._fixture_clip(tmp_path)
        runner.register_stage_hook("run_evidence_stage", lambda *a, **k: {})
        runner.register_stage_hook(
            "run_caption_stage", lambda *a, **k: "Cap anyway")
        client = FakeGraphClient(statuses=["FINISHED"])
        runner.register_stage_hook("make_graph_client", lambda: client)

        summary = runner.retry_failed_insta(str(shorts))
        assert summary["recovered"] == 1

    def test_retry_no_markers_returns_zeros(self, tmp_path):
        from automation.instagram.runner import retry_failed_insta
        root = tmp_path / "empty_shorts"
        root.mkdir()
        assert retry_failed_insta(str(root)) == {
            "retried": 0, "recovered": 0, "still_failed": 0}

    def test_retry_missing_root_returns_zeros(self, tmp_path):
        from automation.instagram.runner import retry_failed_insta
        assert retry_failed_insta(str(tmp_path / "nope")) == {
            "retried": 0, "recovered": 0, "still_failed": 0}


class TestRealClientContract:
    """Reviewer A findings 1+2: the REAL FacebookGraphClient envelope must
    drive publish_reel — no smuggled fake shapes."""

    def _client(self, monkeypatch, envelope_sequence):
        from automation.instagram.graph_client import FacebookGraphClient
        monkeypatch.setenv("YT_CLIPS_INSTA_LIVE", "1")
        client = FacebookGraphClient("tok", "17841400000000")
        calls = {"status": 0}

        permalink_env = {"id": MEDIA_ID, "permalink": PERMALINK}

        def fake_get(path, params=None):
            fields = str((params or {}).get("fields") or "")
            if "status_code" in fields:
                calls["status"] += 1
                return dict(envelope_sequence[min(calls["status"] - 1,
                                                  len(envelope_sequence) - 1)])
            return dict(permalink_env)

        def fake_post(path, data=None):
            return {"id": MEDIA_ID}

        monkeypatch.setattr(client, "_get", fake_get)
        monkeypatch.setattr(client, "_post", fake_post)
        monkeypatch.setattr(client, "upload_video_bytes",
                            lambda cid, p: {"success": True})
        return client

    def test_status_envelope_drives_poll_to_finish(self, tmp_path,
                                                   monkeypatch):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"\x00\x01")
        envs = [
            {"status_code": "IN_PROGRESS", "status": "In Progress: ...",
             "id": None},
            {"status_code": "FINISHED",
             "status": "Finished: ready to be published.",
             "id": "178container1"},
        ]
        client = self._client(monkeypatch, envs)
        from automation.instagram.uploader import publish_reel
        result = publish_reel(client, clip, "hook line", poll_interval_s=0)
        assert result["media_id"] == MEDIA_ID

    def test_published_envelope_recovers_without_republish(
            self, tmp_path, monkeypatch):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"\x00\x01")
        envs = [{"status_code": "PUBLISHED", "status": None,
                 "id": "179alreadylive"}]
        client = self._client(monkeypatch, envs)
        published = []

        real_publish = client.publish_container

        def spy(cid):
            published.append(cid)
            return real_publish(cid)

        monkeypatch.setattr(client, "publish_container", spy)
        from automation.instagram.uploader import publish_reel
        result = publish_reel(client, clip, "hook", poll_interval_s=0)
        assert result["media_id"] == "179alreadylive"
        assert published == []

    def test_expired_envelope_fails_fast(self, tmp_path, monkeypatch):
        import pytest as _pytest
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"\x00\x01")
        client = self._client(
            monkeypatch,
            [{"status_code": "EXPIRED", "status": None, "id": None}])
        from automation.instagram.uploader import (InstagramUploadError,
                                                   publish_reel,
                                                   STAGE_POLL_CONTAINER)
        with _pytest.raises(InstagramUploadError) as excinfo:
            publish_reel(client, clip, "hook", poll_interval_s=0,
                         timeout_s=5)
        assert "EXPIRED" in str(excinfo.value) or \
            excinfo.value.stage == STAGE_POLL_CONTAINER


class TestPublishHook:
    """provider=upload_post path: HOOK_PUBLISH bypasses the Graph flow."""

    @pytest.fixture
    def clean_hooks(self):
        from automation.instagram import runner
        runner._HOOKS.clear()
        yield runner
        runner._HOOKS.clear()

    def test_publish_hook_short_circuits_graph_flow(self, tmp_path,
                                                    insta_enabled,
                                                    clean_hooks):
        runner = clean_hooks
        clip = tmp_path / "clipA"
        clip.mkdir()
        (clip / "clip.mp4").write_bytes(b"\x00\x01")
        calls = {}

        def fake_publish(video_path, caption, audio_name=None,
                         clip_dir=None, state=None):
            calls["caption"] = caption
            calls["video"] = Path(video_path)
            return {"media_id": "179thirdparty",
                    "permalink": "https://instagram.com/reel/X/",
                    "provider": "upload_post"}

        runner.register_stage_hook("run_evidence_stage", lambda *a, **k: None)
        runner.register_stage_hook("run_caption_stage",
                                   lambda *a, **k: "Hook cap")
        runner.register_stage_hook("run_publish_stage_unused", None)
        runner.register_stage_hook(runner.HOOK_PUBLISH, fake_publish)

        media_id = runner.process_instagram_for_clip(
            clip, "transcript", "title", "desc")

        assert media_id == "179thirdparty"
        assert calls["video"].name == "clip.mp4"
        state = json.loads((clip / "clip_insta_state.json").read_text(
            encoding="utf-8"))
        assert state["stage"] == "DONE"
        assert state["provider"] == "upload_post"

    def test_publish_hook_failure_writes_marker(self, tmp_path,
                                                insta_enabled,
                                                clean_hooks):
        runner = clean_hooks
        clip = tmp_path / "clipB"
        clip.mkdir()
        (clip / "clip.mp4").write_bytes(b"\x00\x01")

        def boom(*a, **k):
            raise RuntimeError("quota over")

        runner.register_stage_hook("run_evidence_stage", lambda *a, **k: None)
        runner.register_stage_hook("run_caption_stage",
                                   lambda *a, **k: "cap")
        runner.register_stage_hook(runner.HOOK_PUBLISH, boom)

        assert runner.process_instagram_for_clip(
            clip, "t", "ti", "d") is None
        marker = json.loads((clip / "clip_insta_failed.json").read_text(
            encoding="utf-8"))
        assert "quota over" in marker["error"]

    def test_seo_registers_upload_post_hook_when_provider_set(
            self, monkeypatch):
        import automation.instagram.seo as seo

        class _Cfg(dict):
            def get(self, k, d=None):
                return {"instagram": {
                    "enabled": True,
                    "provider": "upload_post",
                    "upload_post": {"api_key_env": "UP_KEY",
                                    "profile": "chan"},
                }}.get(k, d)

        import automation.instagram.seo as seo
        monkeypatch.setenv("UPLOAD_POST_API_KEY", "secret-key")
        monkeypatch.setattr(seo, "_insta_config",
                            lambda: {"provider": "upload_post",
                                     "upload_post": {
                                         "api_key_env": "UPLOAD_POST_API_KEY",
                                         "profile": "chan"}})
        fn = seo.make_upload_post_publish()
        assert callable(fn)


def test_provider_issue_names_missing_env_key(monkeypatch):
    """provider=upload_post + absent key must say THAT, not blame graph."""
    from automation.instagram import runner

    monkeypatch.setenv("UPLOAD_POST_API_KEY", "")
    issue = runner._provider_config_issue({
        "provider": "upload_post",
        "upload_post": {"api_key_env": "UPLOAD_POST_API_KEY",
                        "profile": "prajjwall"},
    })
    assert issue is not None
    assert "UPLOAD_POST_API_KEY" in issue
    assert "upload_post" in issue


def test_provider_issue_none_when_key_present(monkeypatch):
    from automation.instagram import runner

    monkeypatch.setenv("UPLOAD_POST_API_KEY", "k-test")
    assert runner._provider_config_issue({
        "provider": "upload_post",
        "upload_post": {"api_key_env": "UPLOAD_POST_API_KEY",
                        "profile": "prajjwall"},
    }) is None


def test_provider_issue_skips_graph_provider():
    from automation.instagram import runner

    assert runner._provider_config_issue({"provider": "graph"}) is None


def test_upload_stage_error_reports_provider_issue(tmp_path, monkeypatch):
    """End-to-end: UPLOAD stage with unconfigured provider surfaces the real
    reason instead of the generic graph_client complaint."""
    from automation.instagram import runner as rn

    monkeypatch.setattr(rn, "_skip_requested", lambda: False)
    for name in (rn.HOOK_RUN_EVIDENCE, rn.HOOK_RUN_CAPTION,
                 rn.HOOK_MAKE_CLIENT, rn.HOOK_PUBLISH):
        rn._HOOKS.pop(name, None)
    monkeypatch.setattr(
        rn, "_provider_config_issue",
        lambda cfg=None: "instagram.provider=upload_post but "
                         "UPLOAD_POST_API_KEY is not set")
    state = tmp_path / rn.STATE_FILE
    state.write_text(json.dumps({
        "stage": "CAPTION", "attempts": 1,
        "caption": "ready caption"}), encoding="utf-8")
    (tmp_path / "clip.mp4").write_bytes(b"\x00")

    result = rn.process_instagram_for_clip(
        tmp_path, "tr", "T", "D", force=True)

    assert result is None
    marker = json.loads((tmp_path / rn.FAILED_MARKER).read_text())
    assert "upload_post" in marker["error"]
    rn._HOOKS.clear()


def test_resolve_publish_hook_after_lazy_import(monkeypatch):
    """Resume-at-UPLOAD resolves PUBLISH as the FIRST hook in a fresh
    process: the lazy seo import registers defaults mid-call, and the
    resolver must return the registered hook, not getattr('publish')."""
    from automation.instagram import runner as rn

    monkeypatch.setitem(rn.__dict__, "_HOOKS", {})
    hook = rn.resolve_stage_hook(rn.HOOK_PUBLISH)
    assert hook is not None, (
        "PUBLISH hook unresolved on first call — upload_post resumes "
        "at UPLOAD checkpoint can never publish"
    )


def test_resume_at_upload_reaches_publish(tmp_path, monkeypatch):
    """Full resume path: UPLOAD checkpoint + ready caption must reach the
    publish hook without falling into the graph-client branch."""
    from automation.instagram import runner as rn
    from automation.instagram import upload_post_client as upc

    monkeypatch.setattr(rn, "_skip_requested", lambda: False)
    monkeypatch.setenv("UPLOAD_POST_API_KEY", "k-test")
    rn._HOOKS.clear()
    (tmp_path / "clip.mp4").write_bytes(b"\x00")
    (tmp_path / rn.STATE_FILE).write_text(json.dumps({
        "stage": "UPLOAD", "attempts": 1,
        "caption": "ready"}), encoding="utf-8")

    seen = []

    def fake_publish(self, video_path, caption, **kw):
        seen.append(caption)
        return {"media_id": "M1", "permalink": "u"}

    monkeypatch.setattr(upc.UploadPostClient, "publish_reel", fake_publish)

    result = rn.process_instagram_for_clip(
        tmp_path, "tr", "T", "D", force=True)

    assert result == "M1" and seen == ["ready"]
    rn._HOOKS.clear()
