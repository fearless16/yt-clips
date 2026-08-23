"""TDD tests for automation/instagram graph_client + credential.

Covers:
- GraphClient Protocol conformance (FakeGraphClient in-memory)
- FacebookGraphClient param assembly/parsing (monkeypatched session)
- Kill-switch env YT_CLIPS_INSTA_LIVE
- GraphAPIError code/subcode/message parsing
- credential.py token load/save/validity
"""
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Resp:
    def __init__(self, payload=None, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _client(**kw):
    from automation.instagram.graph_client import FacebookGraphClient
    defaults = dict(token="TOK", ig_user_id="17841400000", timeout=5.0)
    defaults.update(kw)
    client = FacebookGraphClient(**defaults)
    return client


def _stub(client, method, resp):
    mock = MagicMock(return_value=resp)
    setattr(client.session, method, mock)
    return mock


# ---------------------------------------------------------------------------
# FakeGraphClient — in-memory Protocol implementation
# ---------------------------------------------------------------------------

@dataclass
class _Container:
    caption: str
    share_to_feed: bool
    audio_name: str | None
    trial_params: dict | None
    uploaded: int | None = None
    state: str = "IN_PROGRESS"
    media_id: str | None = None


class FakeGraphClient:
    """In-memory GraphClient for uploader/runner tests. No network."""

    STATUS_TEXT = {
        "IN_PROGRESS": "Video in progress",
        "FINISHED": "Video finished",
        "PUBLISHED": "Post published",
        "ERROR": "Error",
        "EXPIRED": "Expired",
    }

    def __init__(self):
        self.containers: dict[str, _Container] = {}
        self.calls: list[tuple] = []
        self._counter = 0

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter}"

    def create_reels_container(self, *, caption: str, share_to_feed: bool = True,
                               audio_name: str | None = None,
                               trial_params: dict | None = None) -> str:
        self.calls.append(("create_reels_container", caption))
        cid = self._next_id("creation")
        self.containers[cid] = _Container(caption, share_to_feed,
                                          audio_name, trial_params)
        return cid

    def upload_video_bytes(self, creation_id: str, video_path: Path) -> dict:
        self.calls.append(("upload_video_bytes", creation_id))
        cont = self.containers[creation_id]
        size = Path(video_path).stat().st_size
        if size <= 0:
            cont.state = "ERROR"
            return {"success": False}
        cont.uploaded = size
        cont.state = "FINISHED"
        return {"success": True, "id": creation_id}

    def container_status(self, creation_id: str) -> dict:
        self.calls.append(("container_status", creation_id))
        cont = self.containers.get(creation_id)
        if cont is None:
            return {"status_code": "EXPIRED", "status": "Expired"}
        return {"status_code": cont.state,
                "status": self.STATUS_TEXT[cont.state]}

    def publish_container(self, creation_id: str) -> str | None:
        self.calls.append(("publish_container", creation_id))
        cont = self.containers.get(creation_id)
        if cont is None:
            raise ValueError(f"unknown container {creation_id}")
        if cont.state != "FINISHED":
            return None
        cont.media_id = self._next_id("media")
        cont.state = "PUBLISHED"
        return cont.media_id

    def media_permalink(self, media_id: str) -> str:
        self.calls.append(("media_permalink", media_id))
        for cont in self.containers.values():
            if cont.media_id == media_id:
                return f"https://www.instagram.com/reel/{media_id}/"
        raise ValueError(f"unknown media {media_id}")

    def hashtag_search(self, q: str) -> dict:
        self.calls.append(("hashtag_search", q))
        return {"data": [{"id": f"tag_{q}", "name": q}]}

    def hashtag_recent_velocity(self, hashtag_id: str) -> int:
        self.calls.append(("hashtag_recent_velocity", hashtag_id))
        return 42

    def hashtag_top_engagement(self, hashtag_id: str) -> float:
        self.calls.append(("hashtag_top_engagement", hashtag_id))
        return 123.5


class TestFakeGraphClientProtocol:

    def test_satisfies_graph_client_protocol(self):
        from automation.instagram.graph_client import GraphClient
        fake = FakeGraphClient()
        assert isinstance(fake, GraphClient)

    def test_happy_path_create_upload_status_publish(self):
        fake = FakeGraphClient()
        cid = fake.create_reels_container(caption="Bumrah yorker! #cricket")
        assert fake.upload_video_bytes(cid, _mk_video(100))["success"] is True
        status = fake.container_status(cid)
        assert status["status_code"] == "FINISHED"
        media_id = fake.publish_container(cid)
        assert media_id == "media_2"
        assert fake.container_status(cid)["status_code"] == "PUBLISHED"
        assert fake.media_permalink(media_id).startswith("https://www.instagram.com/reel/")

    def test_publish_before_finished_is_ambiguous_none(self):
        fake = FakeGraphClient()
        cid = fake.create_reels_container(caption="x")
        assert fake.publish_container(cid) is None

    def test_publish_unknown_container_raises(self):
        fake = FakeGraphClient()
        with pytest.raises(ValueError):
            fake.publish_container("nope")

    def test_zero_byte_upload_marks_error(self):
        fake = FakeGraphClient()
        cid = fake.create_reels_container(caption="x")
        assert fake.upload_video_bytes(cid, _mk_video(0))["success"] is False
        assert fake.container_status(cid)["status_code"] == "ERROR"

    def test_unknown_container_status_expires(self):
        fake = FakeGraphClient()
        assert fake.container_status("ghost")["status_code"] == "EXPIRED"


def _mk_video(size: int) -> Path:
    import tempfile
    f = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    f.write(b"0" * size)
    f.close()
    p = Path(f.name)
    p.write_bytes(b"0" * size)
    return p


# ---------------------------------------------------------------------------
# Kill-switch (module-level env gate)
# ---------------------------------------------------------------------------

class TestKillSwitch:

    def test_disabled_by_default_raises_runtime_error(self, monkeypatch):
        monkeypatch.delenv("YT_CLIPS_INSTA_LIVE", raising=False)
        client = _client()
        with pytest.raises(RuntimeError):
            client.create_reels_container(caption="c")

    def test_explicit_zero_raises_runtime_error(self, monkeypatch):
        monkeypatch.setenv("YT_CLIPS_INSTA_LIVE", "0")
        client = _client()
        with pytest.raises(RuntimeError):
            client.container_status("cid")

    def test_enabled_allows_calls(self, monkeypatch):
        monkeypatch.setenv("YT_CLIPS_INSTA_LIVE", "1")
        client = _client()
        _stub(client, "post", _Resp({"id": "c1"}))
        assert client.create_reels_container(caption="c") == "c1"


# ---------------------------------------------------------------------------
# FacebookGraphClient — request assembly / response parsing
# ---------------------------------------------------------------------------

class TestCreateReelsContainer:

    def test_posts_resumable_reel_payload_and_returns_creation_id(
            self, monkeypatch):
        monkeypatch.setenv("YT_CLIPS_INSTA_LIVE", "1")
        client = _client()
        post = _stub(client, "post", _Resp({"id": "creation_7"}))

        cid = client.create_reels_container(caption="Kohli finisher!")

        assert cid == "creation_7"
        url = post.call_args.args[0]
        assert url == ("https://graph.facebook.com/v21.0/"
                       "17841400000/media")
        data = post.call_args.kwargs["data"]
        assert data["media_type"] == "REELS"
        assert data["upload_type"] == "resumable"
        assert data["caption"] == "Kohli finisher!"
        assert data["share_to_feed"] == "true"
        assert "video_url" not in data
        assert "audio_name" not in data
        assert "trial_params" not in data
        kw = post.call_args.kwargs
        assert kw["timeout"] == 5.0

    def test_optional_params_serialized(self, monkeypatch):
        monkeypatch.setenv("YT_CLIPS_INSTA_LIVE", "1")
        client = _client()
        post = _stub(client, "post", _Resp({"id": "c2"}))

        client.create_reels_container(caption="c", share_to_feed=False,
                                      audio_name="Moment – CricketWithPrajjwal",
                                      trial_params={"mode": "MANUAL"})

        data = post.call_args.kwargs["data"]
        assert data["share_to_feed"] == "false"
        assert data["audio_name"] == "Moment – CricketWithPrajjwal"
        assert json.loads(data["trial_params"]) == {"mode": "MANUAL"}

    def test_missing_id_in_response_raises(self, monkeypatch):
        monkeypatch.setenv("YT_CLIPS_INSTA_LIVE", "1")
        client = _client()
        _stub(client, "post", _Resp({}))
        with pytest.raises((KeyError, ValueError)):
            client.create_reels_container(caption="c")


class TestUploadVideoBytes:

    def test_rupload_headers_and_body(self, monkeypatch, tmp_path):
        monkeypatch.setenv("YT_CLIPS_INSTA_LIVE", "1")
        client = _client()
        post = _stub(client, "post", _Resp({"success": True, "id": "c1"}))
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"MP4DATA!!")

        result = client.upload_video_bytes("c1", video)

        assert result["success"] is True
        url = post.call_args.args[0]
        assert url == ("https://rupload.facebook.com/"
                       "ig-api-upload/v21.0/c1")
        headers = post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "OAuth TOK"
        assert headers["offset"] == "0"
        assert headers["file_size"] == "9"
        assert headers["Content-Type"] == "video/mp4"
        assert post.call_args.kwargs["data"] == b"MP4DATA!!"

    def test_missing_file_raises(self, monkeypatch, tmp_path):
        monkeypatch.setenv("YT_CLIPS_INSTA_LIVE", "1")
        client = _client()
        with pytest.raises(FileNotFoundError):
            client.upload_video_bytes("c1", tmp_path / "nope.mp4")

    def test_error_json_raises_graph_api_error(self, monkeypatch, tmp_path):
        monkeypatch.setenv("YT_CLIPS_INSTA_LIVE", "1")
        client = _client()
        _stub(client, "post", _Resp(
            {"error": {"message": "bad video", "code": 2207026,
                       "error_subcode": 2207026}}, status_code=400))
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"x")
        with pytest.raises(Exception) as excinfo:
            client.upload_video_bytes("c1", video)
        assert excinfo.value.code == 2207026


class TestContainerStatus:

    def test_gets_fields_and_returns_dict(self, monkeypatch):
        monkeypatch.setenv("YT_CLIPS_INSTA_LIVE", "1")
        client = _client()
        get = _stub(client, "get", _Resp(
            {"status_code": "FINISHED", "status": "Video finished",
             "id": "179media"}))

        status = client.container_status("c9")

        assert status == {"status_code": "FINISHED",
                          "status": "Video finished",
                          "id": "179media"}
        url = get.call_args.args[0]
        assert url == "https://graph.facebook.com/v21.0/c9"
        assert get.call_args.kwargs["params"]["fields"] == \
            "status_code,status,id"

    def test_error_response_raises_with_parsed_fields(self, monkeypatch):
        monkeypatch.setenv("YT_CLIPS_INSTA_LIVE", "1")
        client = _client()
        _stub(client, "get", _Resp(
            {"error": {"message": "Expired", "code": 2207076,
                       "error_subcode": 2108006,
                       "fbtrace_id": "Az1"}}, status_code=400))
        with pytest.raises(Exception) as excinfo:
            client.container_status("dead")
        err = excinfo.value
        assert err.message == "Expired"
        assert err.code == 2207076
        assert err.subcode == 2108006
        assert err.http_status == 400
        assert err.fbtrace_id == "Az1"

    def test_http_error_without_json_raises(self, monkeypatch):
        monkeypatch.setenv("YT_CLIPS_INSTA_LIVE", "1")
        client = _client()
        _stub(client, "get", _Resp(None, status_code=502))
        with pytest.raises(Exception) as excinfo:
            client.container_status("c")
        assert excinfo.value.http_status == 502
        assert excinfo.value.code is None


class TestPublishContainer:

    def test_publish_success_returns_media_id(self, monkeypatch):
        monkeypatch.setenv("YT_CLIPS_INSTA_LIVE", "1")
        client = _client()
        post = _stub(client, "post", _Resp({"id": "media_55"}))

        media_id = client.publish_container("c1")

        assert media_id == "media_55"
        url = post.call_args.args[0]
        assert url == ("https://graph.facebook.com/v21.0/"
                       "17841400000/media_publish")
        assert post.call_args.kwargs["data"] == {"creation_id": "c1"}

    def test_publish_failure_returns_none_ambiguous(self, monkeypatch):
        monkeypatch.setenv("YT_CLIPS_INSTA_LIVE", "1")
        client = _client()
        _stub(client, "post", _Resp(
            {"error": {"message": "boom", "code": 500}}, status_code=500))
        assert client.publish_container("c1") is None

    def test_publish_timeout_returns_none_ambiguous(self, monkeypatch):
        monkeypatch.setenv("YT_CLIPS_INSTA_LIVE", "1")
        client = _client()
        import requests as _rq
        _stub(client, "post", _Resp(_rq.exceptions.Timeout("t")))
        assert client.publish_container("c1") is None


class TestMediaPermalink:

    def test_returns_permalink(self, monkeypatch):
        monkeypatch.setenv("YT_CLIPS_INSTA_LIVE", "1")
        client = _client()
        get = _stub(client, "get", _Resp(
            {"permalink": "https://www.instagram.com/reel/abc/"}))
        assert client.media_permalink("abc") == \
            "https://www.instagram.com/reel/abc/"
        assert get.call_args.kwargs["params"]["fields"] == "permalink"


class TestHashtagEndpoints:

    def test_hashtag_search_sends_user_id_and_q(self, monkeypatch):
        monkeypatch.setenv("YT_CLIPS_INSTA_LIVE", "1")
        client = _client()
        get = _stub(client, "get", _Resp({"data": [{"id": "1784"}]}))

        out = client.hashtag_search("cricket")

        assert out["data"][0]["id"] == "1784"
        url = get.call_args.args[0]
        assert url == "https://graph.facebook.com/v21.0/ig_hashtag_search"
        params = get.call_args.kwargs["params"]
        assert params == {"user_id": "17841400000", "q": "cricket"}

    def test_recent_velocity_counts_24h_items_across_pages(self, monkeypatch):
        monkeypatch.setenv("YT_CLIPS_INSTA_LIVE", "1")
        client = _client(max_hashtag_pages=3)
        now = time.time()

        def ts(delta_s):
            return time.strftime("%Y-%m-%dT%H:%M:%S+0000",
                                 time.gmtime(now - delta_s))

        page1 = {"data": [{"id": "m1", "timestamp": ts(3600)},
                          {"id": "m2", "timestamp": ts(60)}],
                 "paging": {"cursors": {"after": "CUR"}}}
        page2 = {"data": [{"id": "m3", "timestamp": ts(86400 * 2)}],
                 "paging": {}}
        get = _stub(client, "get", None)
        get.side_effect = [_Resp(page1), _Resp(page2)]

        velocity = client.hashtag_recent_velocity("tag_1")

        assert velocity == 2
        second_params = get.call_args_list[1].kwargs["params"]
        assert second_params["after"] == "CUR"
        assert second_params["user_id"] == "17841400000"
        assert get.call_count == 2

    def test_recent_velocity_stops_at_page_cap(self, monkeypatch):
        monkeypatch.setenv("YT_CLIPS_INSTA_LIVE", "1")
        client = _client(max_hashtag_pages=2)
        fresh = time.strftime(
            "%Y-%m-%dT%H:%M:%S+0000", time.gmtime(time.time() - 30))
        page = {"data": [{"id": "m", "timestamp": fresh}],
                "paging": {"cursors": {"after": "X"}}}
        get = _stub(client, "get", None)
        get.side_effect = [_Resp(page), _Resp(dict(page))]
        assert client.hashtag_recent_velocity("t") == 2
        assert get.call_count == 2

    def test_top_engagement_median_likes_plus_comments(self, monkeypatch):
        monkeypatch.setenv("YT_CLIPS_INSTA_LIVE", "1")
        client = _client()
        get = _stub(client, "get", _Resp(
            {"data": [{"like_count": 10, "comment_count": 0},
                      {"like_count": 20, "comment_count": 4},
                      {"like_count": 30, "comment_count": 6}]}))

        median = client.hashtag_top_engagement("tag_1")

        assert median == 24.0
        params = get.call_args.kwargs["params"]
        assert params["fields"] == "like_count,comment_count"
        assert "top_media" in get.call_args.args[0]

    def test_top_engagement_empty_data_returns_zero(self, monkeypatch):
        monkeypatch.setenv("YT_CLIPS_INSTA_LIVE", "1")
        client = _client()
        _stub(client, "get", _Resp({"data": []}))
        assert client.hashtag_top_engagement("tag_x") == 0.0


# ---------------------------------------------------------------------------
# credential.py
# ---------------------------------------------------------------------------

class TestCredential:

    def _fn(self, name):
        import automation.instagram.credential as cred
        return getattr(cred, name)

    def test_save_load_roundtrip(self, tmp_path):
        save_token = self._fn("save_token")
        load_token = self._fn("load_token")
        path = tmp_path / "insta_token.json"
        save_token({"access_token": "AT", "expires_at": 99.0,
                    "ig_user_id": "IG1"}, path)
        loaded = load_token(path)
        assert loaded == {"access_token": "AT", "expires_at": 99.0,
                          "ig_user_id": "IG1"}

    def test_load_missing_returns_none(self, tmp_path):
        load_token = self._fn("load_token")
        assert load_token(tmp_path / "absent.json") is None

    def test_load_corrupt_returns_none(self, tmp_path):
        load_token = self._fn("load_token")
        path = tmp_path / "insta_token.json"
        path.write_text("{not json", encoding="utf-8")
        assert load_token(path) is None

    def test_default_path_is_repo_root_insta_token_json(self, tmp_path,
                                                        monkeypatch):
        load_token = self._fn("load_token")
        save_token = self._fn("save_token")
        monkeypatch.chdir(tmp_path)
        save_token({"access_token": "a", "expires_at": 1, "ig_user_id": "i"})
        assert (tmp_path / "insta_token.json").exists()
        assert load_token()["access_token"] == "a"

    def test_is_valid_true_when_future_expiry(self):
        is_valid = self._fn("is_valid")
        assert is_valid({"access_token": "t",
                         "expires_at": time.time() + 3600}) is True

    def test_is_valid_false_when_expired_or_missing_token(self):
        is_valid = self._fn("is_valid")
        assert is_valid({"access_token": "t",
                         "expires_at": time.time() - 10}) is False
        assert is_valid({"access_token": "", "expires_at": 9e15}) is False
        assert is_valid({}) is False
        assert is_valid(None) is False

    def test_refresh_if_needed_raises_not_implemented(self):
        refresh_if_needed = self._fn("refresh_if_needed")
        with pytest.raises(NotImplementedError, match="setup_auth"):
            refresh_if_needed()
