"""Tests for automation.instagram.integration.publish_everywhere (I5).

Both platform workers are stubbed; no network. Covers: parallel success
shape, independent failure domains, skip-path resolution order, the
I2/I3 build_insta_evidence_pack signature reconciliation regression,
and full-suite guards.
"""

import json
import sys
import threading
import types
from pathlib import Path

import pytest
import yaml

import automation.seo.seo  # noqa: F401 — bind module-level cfg before patches
import upload  # noqa: F401
from automation.instagram import runner as _runner_mod  # noqa: F401
from utils.config import Config, _config_cache

MEDIA_ID = "17900000000000002"
VIDEO_ID = "yt_vid_integration_1"


def _merge_config(monkeypatch, enabled):
    try:
        from utils.config import load_config as _load

        base = dict(_load())
    except Exception:
        base = {}
    base["instagram"] = {"enabled": bool(enabled)}
    monkeypatch.setitem(_config_cache, "config.yaml", Config(base))


@pytest.fixture
def insta_enabled(monkeypatch):
    monkeypatch.delenv("YT_CLIPS_SKIP_INSTAGRAM", raising=False)
    _merge_config(monkeypatch, True)


@pytest.fixture
def clip_dir(tmp_path):
    clip = tmp_path / "run1" / "clipA"
    clip.mkdir(parents=True)
    (clip / "clipA.mp4").write_bytes(b"\x00\x01")
    return clip


def _stub_youtube(monkeypatch, *, video_id=VIDEO_ID, seo_result=None,
                  fail=False, calls=None):
    import automation.seo.seo as seo_mod
    import upload as upload_mod

    seen = {"seo": [], "upload": []}

    def fake_generate_seo(clip_id=None, transcript=None, output_dir=None,
                          video_title="", video_description="", **kwargs):
        seen["seo"].append({
            "clip_id": clip_id,
            "transcript": transcript,
            "output_dir": output_dir,
            "video_title": video_title,
            "video_description": video_description,
        })
        if fail:
            return {"_seo_failed": True}
        if seo_result is not None:
            return dict(seo_result)
        metadata_path = Path(output_dir) / f"{clip_id}_metadata.json"
        metadata_path.write_text(json.dumps({"title": "t"}),
                                 encoding="utf-8")
        return {"title": "t"}

    def fake_upload_video(video_path, metadata_path, privacy="public",
                          publish_at=None):
        seen["upload"].append({
            "video_path": video_path,
            "metadata_path": metadata_path,
            "privacy": privacy,
        })
        if isinstance(video_id, Exception):
            raise video_id
        return video_id

    monkeypatch.setattr(seo_mod, "generate_seo_for_exported_clip",
                        fake_generate_seo)
    monkeypatch.setattr(upload_mod, "upload_video", fake_upload_video)

    def _assert_not_called(*a, **k):
        raise AssertionError("upload_video called after SEO failure")

    if fail:
        monkeypatch.setattr(upload_mod, "upload_video", _assert_not_called)
    if calls is not None:
        calls.update(seen)
    return seen


def _stub_instagram(monkeypatch, *, media_id=MEDIA_ID, exc=None,
                    counter=None):
    from automation.instagram import runner

    calls = []

    def fake_process(clip_dir, transcript, video_title, video_description,
                     **kwargs):
        calls.append({
            "clip_dir": str(clip_dir),
            "transcript": transcript,
            "video_title": video_title,
            "video_description": video_description,
        })
        if counter is not None:
            counter.append(1)
        if exc is not None:
            raise exc
        return media_id

    monkeypatch.setattr(runner, "process_instagram_for_clip", fake_process)
    return calls


# ---------------------------------------------------------------------------
# Parallel success + result shape
# ---------------------------------------------------------------------------

class TestParallelSuccess:

    def test_success_dict_shape_and_threading(self, tmp_path, clip_dir,
                                              monkeypatch, insta_enabled):
        from automation.instagram.integration import publish_everywhere

        started_yt = threading.Event()
        started_ig = threading.Event()

        import automation.seo.seo as seo_mod
        import upload as upload_mod
        from automation.instagram import runner

        def gated_seo(**kwargs):
            started_yt.set()
            assert started_ig.wait(10), "instagram worker never ran in parallel"
            return {"title": "t"}

        def gated_insta(*args, **kwargs):
            started_ig.set()
            assert started_yt.wait(10), "youtube worker never ran in parallel"
            return MEDIA_ID

        monkeypatch.setattr(seo_mod, "generate_seo_for_exported_clip",
                            gated_seo)

        def gated_upload(video_path, metadata_path, privacy="public",
                         publish_at=None):
            return VIDEO_ID

        monkeypatch.setattr(upload_mod, "upload_video", gated_upload)
        monkeypatch.setattr(runner, "process_instagram_for_clip",
                            gated_insta)

        result = publish_everywhere(clip_dir, "transcript text",
                                    "Match Title", "Match Desc")

        assert set(result) == {"youtube", "instagram"}
        assert result["instagram"] == MEDIA_ID
        assert result["youtube"]["video_id"] == VIDEO_ID
        assert result["youtube"]["metadata_path"].endswith(
            "clipA_metadata.json")

    def test_workers_receive_threaded_arguments(self, tmp_path, clip_dir,
                                                monkeypatch, insta_enabled):
        from automation.instagram.integration import publish_everywhere

        seen = _stub_youtube(monkeypatch)
        ig_calls = _stub_instagram(monkeypatch)

        result = publish_everywhere(clip_dir, "tr text", "Title X",
                                    "Desc Y")

        assert result["youtube"]["video_id"] == VIDEO_ID
        assert seen["seo"] == [{
            "clip_id": "clipA",
            "transcript": "tr text",
            "output_dir": str(clip_dir),
            "video_title": "Title X",
            "video_description": "Desc Y",
        }]
        assert seen["upload"][0]["video_path"] == \
            str(clip_dir / "clipA.mp4")
        assert seen["upload"][0]["metadata_path"] == \
            str(clip_dir / "clipA_metadata.json")
        assert seen["upload"][0]["privacy"] == "public"
        assert ig_calls[0]["clip_dir"] == str(clip_dir)
        assert ig_calls[0]["transcript"] == "tr text"
        assert ig_calls[0]["video_title"] == "Title X"
        assert ig_calls[0]["video_description"] == "Desc Y"

    def test_privacy_kwarg_forwarded(self, tmp_path, clip_dir, monkeypatch,
                                     insta_enabled):
        from automation.instagram.integration import publish_everywhere

        seen = _stub_youtube(monkeypatch)
        _stub_instagram(monkeypatch)
        publish_everywhere(clip_dir, "tr", "T", "D", privacy="unlisted")
        assert seen["upload"][0]["privacy"] == "unlisted"


# ---------------------------------------------------------------------------
# Independent failure domains
# ---------------------------------------------------------------------------

class TestFailureIsolation:

    def test_instagram_raise_keeps_youtube_intact(
            self, tmp_path, clip_dir, monkeypatch, insta_enabled):
        from automation.instagram.integration import publish_everywhere

        _stub_youtube(monkeypatch)
        _stub_instagram(monkeypatch, exc=RuntimeError("IG down"))

        result = publish_everywhere(clip_dir, "tr", "T", "D")

        assert result["youtube"]["video_id"] == VIDEO_ID
        assert "error" in result["instagram"]
        assert "IG down" in result["instagram"]["error"]

    def test_youtube_raise_keeps_instagram_intact(
            self, tmp_path, clip_dir, monkeypatch, insta_enabled):
        from automation.instagram.integration import publish_everywhere

        _stub_youtube(monkeypatch, video_id=RuntimeError("YT quota"))
        _stub_instagram(monkeypatch, media_id="179survivor")

        result = publish_everywhere(clip_dir, "tr", "T", "D")

        assert result["youtube"]["error"]
        assert "YT quota" in result["youtube"]["error"]
        assert result["instagram"] == "179survivor"

    def test_seo_failed_marker_captured_without_upload_call(
            self, tmp_path, clip_dir, monkeypatch, insta_enabled):
        from automation.instagram.integration import publish_everywhere

        _stub_youtube(monkeypatch, fail=True)
        _stub_instagram(monkeypatch, media_id="179ok")

        result = publish_everywhere(clip_dir, "tr", "T", "D")

        assert "error" in result["youtube"]
        assert result["instagram"] == "179ok"

    def test_both_fail_isolated_entries(
            self, tmp_path, clip_dir, monkeypatch, insta_enabled):
        from automation.instagram.integration import publish_everywhere

        _stub_youtube(monkeypatch, video_id=RuntimeError("boom-yt"))
        _stub_instagram(monkeypatch, exc=ValueError("boom-ig"))

        result = publish_everywhere(clip_dir, "tr", "T", "D")

        assert "boom-yt" in result["youtube"]["error"]
        assert "boom-ig" in result["instagram"]["error"]

    def test_missing_mp4_captured_as_youtube_error(
            self, tmp_path, monkeypatch, insta_enabled):
        from automation.instagram.integration import publish_everywhere

        empty = tmp_path / "run2" / "clipEmpty"
        empty.mkdir(parents=True)
        _stub_instagram(monkeypatch, media_id="179still-ok")

        result = publish_everywhere(empty, "tr", "T", "D")

        assert ".mp4" in result["youtube"]["error"]
        assert result["instagram"] == "179still-ok"


# ---------------------------------------------------------------------------
# Skip resolution: explicit arg > env > config > False
# ---------------------------------------------------------------------------

class TestSkipResolution:

    def test_explicit_skip_true_never_executes_insta_worker(
            self, tmp_path, clip_dir, monkeypatch, insta_enabled):
        from automation.instagram.integration import publish_everywhere

        _stub_youtube(monkeypatch)
        counter = []
        _stub_instagram(monkeypatch, counter=counter)

        result = publish_everywhere(clip_dir, "tr", "T", "D",
                                    skip_instagram=True)

        assert result["instagram"] is None
        assert result["youtube"]["video_id"] == VIDEO_ID
        assert counter == []

    def test_env_skip_never_executes_insta_worker(
            self, tmp_path, clip_dir, monkeypatch):
        from automation.instagram.integration import publish_everywhere

        monkeypatch.setenv("YT_CLIPS_SKIP_INSTAGRAM", "1")
        _merge_config(monkeypatch, True)
        _stub_youtube(monkeypatch)
        counter = []
        _stub_instagram(monkeypatch, counter=counter)

        result = publish_everywhere(clip_dir, "tr", "T", "D")

        assert result["instagram"] is None
        assert counter == []
        assert result["youtube"]["video_id"] == VIDEO_ID

    def test_config_disabled_never_executes_insta_worker(
            self, tmp_path, clip_dir, monkeypatch):
        from automation.instagram.integration import publish_everywhere

        monkeypatch.delenv("YT_CLIPS_SKIP_INSTAGRAM", raising=False)
        _merge_config(monkeypatch, False)
        _stub_youtube(monkeypatch)
        counter = []
        _stub_instagram(monkeypatch, counter=counter)

        result = publish_everywhere(clip_dir, "tr", "T", "D")

        assert result["instagram"] is None
        assert counter == []

    def test_config_enabled_runs_insta_worker(
            self, tmp_path, clip_dir, monkeypatch, insta_enabled):
        from automation.instagram.integration import publish_everywhere

        _stub_youtube(monkeypatch)
        counter = []
        _stub_instagram(monkeypatch, counter=counter)

        result = publish_everywhere(clip_dir, "tr", "T", "D")

        assert result["instagram"] == MEDIA_ID
        assert len(counter) == 1

    def test_explicit_false_overrides_env(
            self, tmp_path, clip_dir, monkeypatch, insta_enabled):
        from automation.instagram.integration import publish_everywhere

        monkeypatch.setenv("YT_CLIPS_SKIP_INSTAGRAM", "1")
        _stub_youtube(monkeypatch)
        counter = []
        _stub_instagram(monkeypatch, counter=counter)

        result = publish_everywhere(clip_dir, "tr", "T", "D",
                                    skip_instagram=False)

        assert result["instagram"] == MEDIA_ID
        assert len(counter) == 1

    def test_skip_youtube_leaves_instagram_alone(
            self, tmp_path, clip_dir, monkeypatch, insta_enabled):
        from automation.instagram.integration import publish_everywhere

        yt_calls = _stub_youtube(monkeypatch)
        _stub_instagram(monkeypatch, media_id="179only")

        result = publish_everywhere(clip_dir, "tr", "T", "D",
                                    skip_youtube=True)

        assert result["youtube"] is None
        assert result["instagram"] == "179only"
        assert yt_calls["seo"] == [] and yt_calls["upload"] == []

    def test_both_skipped_no_workers_at_all(
            self, tmp_path, clip_dir, monkeypatch, insta_enabled):
        from automation.instagram.integration import publish_everywhere

        yt_calls = _stub_youtube(monkeypatch)
        counter = []
        _stub_instagram(monkeypatch, counter=counter)

        result = publish_everywhere(clip_dir, "tr", "T", "D",
                                    skip_youtube=True,
                                    skip_instagram=True)

        assert result == {"youtube": None, "instagram": None}
        assert yt_calls["seo"] == [] and yt_calls["upload"] == []
        assert counter == []


# ---------------------------------------------------------------------------
# I2/I3 signature reconciliation regression
# ---------------------------------------------------------------------------

PACK = {
    "match_facts": ["India vs Sri Lanka, 2nd T20I"],
    "roster": ["Bumrah"],
    "learner_top_captions": [],
    "seed_phrases": [
        "ind vs sl highlights",
        "bumrah magic",
        "pallekele t20",
        "cricket reels india",
        "team india squad",
    ],
    "validated_hashtags": [],
    "forbidden": ["invented player names"],
}


@pytest.fixture
def capturing_evidence(monkeypatch):
    mod = types.ModuleType("automation.instagram.evidence")
    captured = {}

    def build_insta_evidence_pack(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return json.loads(json.dumps(PACK))

    mod.build_insta_evidence_pack = build_insta_evidence_pack
    monkeypatch.setitem(sys.modules, "automation.instagram.evidence", mod)
    return captured


class TestSignatureReconciliation:

    def test_seo_build_evidence_passes_keywords_in_right_slots(
            self, tmp_path, monkeypatch, capturing_evidence):
        from automation.instagram import caption_engine, seo
        monkeypatch.setattr(caption_engine, "_get_ai", lambda: None)

        seo.run_evidence_stage(str(tmp_path), "TRANSCRIPT_BODY",
                               "TITLE_HEAD", "DESC_TAIL")

        assert capturing_evidence["args"] == ()
        kwargs = capturing_evidence["kwargs"]
        assert kwargs["video_title"] == "TITLE_HEAD"
        assert kwargs["video_description"] == "DESC_TAIL"
        assert kwargs["transcript"] == "TRANSCRIPT_BODY"
        assert "clip_dir" not in kwargs

    def test_real_evidence_pack_routes_keyword_args_correctly(
            self, monkeypatch):
        import automation.instagram.evidence as ev

        calls = {}

        def fake_fetch(q):
            calls.setdefault("query", q)
            return {"facts": ["fact"], "player_names": ["Bumrah"]}

        monkeypatch.setattr(
            ev, "_research_query",
            lambda t, d, tr: f"Q::{t}::{d}::{tr}")
        monkeypatch.setattr(ev, "fetch_verified_match_context", fake_fetch)
        monkeypatch.setattr(ev, "fetch_youtube_suggestions",
                            lambda q: [])

        pack = ev.build_insta_evidence_pack(
            video_title="TITLE_X",
            video_description="DESC_Y",
            transcript="TRANS_Z",
        )

        assert calls["query"] == "Q::TITLE_X::DESC_Y::TRANS_Z"
        assert pack["match_facts"] == ["fact"]
        assert "Bumrah" in pack["roster"]

    def test_runner_caption_stage_end_to_end_reconciled(
            self, tmp_path, monkeypatch, capturing_evidence):
        from automation.instagram import caption_engine, seo

        body = ("Pallekele ke death overs mein Bumrah ne match pher diya. "
                "Samson ne bhi ek sharp catch pakda. "
                "Aaj ka game changer kaun raha?")
        caption_text = "Bumrah ne liye 3 wickets! IND v SL\n" + body
        assert 150 <= len(caption_text) <= 250

        class FakeAI:
            def generate_text(self, prompt, system_instruction=None,
                              **kwargs):
                return json.dumps({
                    "caption": caption_text,
                    "hashtags": ["#indvsslhighlights", "#bumrahmagic",
                                 "#pallekelet20", "#cricketreelsindia",
                                 "#teamindiasquad"],
                    "audio_name": "Bumrah – CricketWithPrajjwal",
                })

        def _clean_audit(*a, **k):
            return {"unsupported_entities": [], "supported_topics": []}

        monkeypatch.setattr(caption_engine, "_get_ai", lambda: FakeAI())
        monkeypatch.setattr(caption_engine, "audit_written_copy_llm",
                            _clean_audit)

        caption = seo.run_caption_stage(str(tmp_path), "TRANSCRIPT_BODY",
                                        "TITLE_HEAD", "DESC_TAIL")

        assert caption and caption.startswith("Bumrah")
        kwargs = capturing_evidence["kwargs"]
        assert kwargs["transcript"] == "TRANSCRIPT_BODY"
        assert kwargs["video_title"] == "TITLE_HEAD"


# ---------------------------------------------------------------------------
# Full-suite guards
# ---------------------------------------------------------------------------

class _RecordingClient:
    def __init__(self):
        self.calls = []

    def create_reels_container(self, *, caption, share_to_feed=True,
                               audio_name=None):
        self.calls.append({"caption": caption, "audio_name": audio_name})
        return "cid_1"

    def upload_video_bytes(self, creation_id, video_path):
        pass

    def container_status(self, creation_id):
        return {"status": "FINISHED"}

    def publish_container(self, creation_id):
        return MEDIA_ID

    def media_permalink(self, media_id):
        return "https://www.instagram.com/reel/x/"


class TestRunnerAudioNameThreading:

    def test_publish_reel_receives_audio_name_from_insta_metadata(
            self, tmp_path, monkeypatch, insta_enabled):
        from automation.instagram import runner

        (tmp_path / "v.mp4").write_bytes(b"\x00")
        (tmp_path / "insta_metadata.json").write_text(json.dumps({
            "caption": "cap from metadata",
            "audio_name": "Six Moment – CricketWithPrajjwal",
        }), encoding="utf-8")
        client = _RecordingClient()
        monkeypatch.setattr(runner, "_HOOKS", {
            runner.HOOK_RUN_EVIDENCE: lambda *a, **k: {},
            runner.HOOK_RUN_CAPTION: lambda *a, **k: "cap from metadata",
            runner.HOOK_MAKE_CLIENT: lambda: client,
        })

        assert runner.process_instagram_for_clip(
            tmp_path, "tr", "t", "d") == MEDIA_ID
        assert client.calls[0]["audio_name"] == \
            "Six Moment – CricketWithPrajjwal"

    def test_missing_metadata_passes_audio_name_none(
            self, tmp_path, monkeypatch, insta_enabled):
        from automation.instagram import runner

        (tmp_path / "v.mp4").write_bytes(b"\x00")
        client = _RecordingClient()
        monkeypatch.setattr(runner, "_HOOKS", {
            runner.HOOK_RUN_EVIDENCE: lambda *a, **k: {},
            runner.HOOK_RUN_CAPTION: lambda *a, **k: "cap",
            runner.HOOK_MAKE_CLIENT: lambda: client,
        })

        assert runner.process_instagram_for_clip(
            tmp_path, "tr", "t", "d") == MEDIA_ID
        assert client.calls[0]["audio_name"] is None


class TestSuiteGuards:

    def test_config_yaml_ships_instagram_disabled_facebook_login(self):
        data = yaml.safe_load(
            Path("config.yaml").read_text(encoding="utf-8")) or {}
        insta = data.get("instagram") or {}
        assert insta.get("enabled") is False
        assert insta.get("api_flavor") == "facebook_login"
        assert insta.get("hashtags_count") == 5
        assert insta.get("caption_target_chars") == 250
