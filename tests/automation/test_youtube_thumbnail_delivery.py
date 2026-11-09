"""Failure-first delivery checks: real image bytes, fake API and CDN only."""
import io
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image, ImageDraw

import upload


VIDEO_ID = "fixture-video"
THUMB_URL = "https://i.ytimg.com/vi/fixture-video/custom.jpg"


def artwork(size=(360, 640), quality=94):
    image = Image.new("RGB", size, (11, 15, 23))
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 15, size[0]-10, size[1]//5), fill=(240, 30, 125))
    draw.ellipse((30, size[1]//4, size[0]-30, size[1]*3//4), fill=(211, 169, 124))
    draw.rectangle((55, size[1]*4//5, size[0]-55, size[1]*9//10), fill="white")
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=quality)
    return output.getvalue()


@pytest.fixture
def delivery(tmp_path, monkeypatch):
    monkeypatch.setitem(upload.cfg, "release_review", {"required": False})
    monkeypatch.setitem(upload.cfg, "thumbnail", {"enabled": True})
    monkeypatch.setitem(upload.cfg, "youtube", {
        "category_id": "17", "thumbnail_processing_timeout_seconds": 2,
        "thumbnail_verify_timeout_seconds": 2, "thumbnail_poll_interval_seconds": .01,
        "thumbnail_max_poll_attempts": 3,
    })
    monkeypatch.setattr(upload, "_validate_shorts_video", lambda *_: True)
    monkeypatch.setattr(upload, "_assignable_category_id", lambda *_a, **_k: "17")
    monkeypatch.setattr("time.sleep", lambda *_: None)
    video = tmp_path / "portrait.mp4"
    video.write_bytes(b"fixture media; ffprobe separately tested")
    meta = tmp_path / "portrait_metadata.json"
    meta.write_text(json.dumps({"title": "Actual cricket quote", "description": "Grounded context"}))
    cover = tmp_path / "portrait_thumb.jpg"
    cover.write_bytes(artwork())
    service = Mock()
    service.videos().insert().execute.return_value = {"id": VIDEO_ID}
    service.thumbnails().set().execute.return_value = {"items": [{
        "shortsAspectRatio": {"url": THUMB_URL, "width": 360, "height": 640},
    }]}
    events = []
    statuses = ["succeeded"]

    def listed(**kwargs):
        events.append(kwargs["part"])
        if "status" in kwargs["part"]:
            data = {"items": [{"id": VIDEO_ID, "status": {"privacyStatus": "private"},
                               "snippet": {"channelId": upload.cfg["youtube"].get("channel_id")}}]}
        elif "processingDetails" in kwargs["part"]:
            status = statuses.pop(0) if len(statuses) > 1 else statuses[0]
            data = {"items": [{"id": VIDEO_ID, "processingDetails": {"processingStatus": status}}]}
        else:
            data = {"items": [{"id": VIDEO_ID, "snippet": {"thumbnails": {
                "shortsAspectRatio": {"url": THUMB_URL, "width": 360, "height": 640},
            }}}]}
        return Mock(execute=Mock(return_value=data))

    service.videos().list.side_effect = listed
    service.thumbnails().set.side_effect = lambda **_kw: (
        events.append("thumbnail-set") or Mock(execute=Mock(return_value={"items": [{
            "shortsAspectRatio": {"url": THUMB_URL, "width": 360, "height": 640},
        }]})))
    auth = Mock(return_value=service)
    monkeypatch.setattr(upload, "get_authenticated_service", auth)
    response = Mock()
    response.headers = {"Content-Type": "image/jpeg"}
    response.iter_content.side_effect = lambda **_kw: iter([artwork(quality=82)])
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    fetch = Mock(return_value=response)
    monkeypatch.setattr("requests.get", fetch)
    service.reset_mock()
    return video, meta, cover, service, auth, response, events, statuses


def run_upload(delivery):
    return upload.upload_video(str(delivery[0]), str(delivery[1]), privacy="private")


def receipt(delivery):
    return json.loads(delivery[0].with_name("portrait_delivery.json").read_text())


@pytest.mark.parametrize("bad_cover", ["missing", "empty", "corrupt", "landscape", "wrong-format"])
def test_cover_is_required_and_decoded_before_auth(delivery, bad_cover):
    cover = delivery[2]
    if bad_cover == "missing":
        cover.unlink()
    elif bad_cover == "empty":
        cover.write_bytes(b"")
    elif bad_cover == "corrupt":
        cover.write_bytes(b"not an image")
    elif bad_cover == "landscape":
        cover.write_bytes(artwork((640, 360)))
    else:
        Image.new("RGB", (360, 640)).save(cover, format="PNG")
    assert run_upload(delivery) is None
    delivery[4].assert_not_called()


def test_waits_for_processing_then_checks_served_pixels(delivery):
    delivery[7][:] = ["processing", "processing", "succeeded"]
    original_meta = delivery[1].read_bytes()
    assert run_upload(delivery) == VIDEO_ID
    events = delivery[6]
    assert events[:5] == ["status,snippet"] + ["processingDetails"]*3 + ["thumbnail-set"]
    assert "snippet" in events[5:]
    proof = receipt(delivery)
    assert proof["status"] == "COMPLETE"
    assert proof["thumbnail"]["status"] == "VERIFIED"
    assert proof["thumbnail"]["verification"]["pixel_rmse"] < 10
    assert delivery[1].read_bytes() == original_meta, "Upload must not invalidate reviewed metadata"


@pytest.mark.parametrize("status", ["processing", "failed", "terminated", "garbage"])
def test_processing_failure_is_incomplete_and_never_sets_cover(delivery, status):
    delivery[7][:] = [status]
    with pytest.raises(RuntimeError, match="(?i)processing"):
        run_upload(delivery)
    # The mock was configured without calling set, so this is a real call count.
    delivery[3].thumbnails().set.assert_not_called()
    proof = receipt(delivery)
    assert proof["video_id"] == VIDEO_ID and proof["status"] != "COMPLETE"


def test_thumbnail_error_resumes_same_video_without_insert(delivery):
    delivery[3].thumbnails().set.side_effect = RuntimeError("fixture rejected thumbnail")
    with pytest.raises(RuntimeError, match="thumbnail") as error:
        run_upload(delivery)
    assert error.value.video_id == VIDEO_ID
    assert receipt(delivery)["status"] == "THUMBNAIL_PENDING"
    delivery[3].thumbnails().set.side_effect = lambda **_kw: Mock(execute=Mock(return_value={
        "items": [{"shortsAspectRatio": {"url": THUMB_URL}}]}))
    assert run_upload(delivery) == VIDEO_ID
    assert delivery[3].videos().insert.call_count == 1


def test_same_cdn_url_with_wrong_pixels_is_not_verified(delivery):
    bad = io.BytesIO()
    Image.new("RGB", (360, 640), "lime").save(bad, format="JPEG")
    delivery[5].iter_content.side_effect = lambda **_kw: iter([bad.getvalue()])
    with pytest.raises(RuntimeError, match="(?i)verif|match"):
        run_upload(delivery)
    assert receipt(delivery)["status"] == "THUMBNAIL_PENDING"


def test_malformed_set_response_cannot_claim_success(delivery):
    delivery[3].thumbnails().set.side_effect = lambda **_kw: Mock(execute=Mock(return_value={"items": []}))
    with pytest.raises(RuntimeError, match="(?i)thumbnail"):
        run_upload(delivery)
    assert receipt(delivery)["status"] == "THUMBNAIL_PENDING"


def test_insert_timeout_blocks_blind_video_replay(delivery):
    delivery[3].videos().insert.return_value.execute.side_effect = TimeoutError("outcome unknown")
    run_upload(delivery)
    with pytest.raises(RuntimeError, match="(?i)unknown|reconcil"):
        run_upload(delivery)
    assert delivery[3].videos().insert.call_count == 1


def test_changed_artifact_cannot_resume_or_reupload(delivery):
    assert run_upload(delivery) == VIDEO_ID
    delivery[1].write_text(json.dumps({"title": "Different promise", "description": "Changed"}))
    with pytest.raises(RuntimeError, match="(?i)changed|differ"):
        run_upload(delivery)
    assert delivery[3].videos().insert.call_count == 1


def test_verified_receipt_is_idempotent(delivery):
    assert run_upload(delivery) == VIDEO_ID
    assert run_upload(delivery) == VIDEO_ID
    assert delivery[3].videos().insert.call_count == 1
    assert delivery[3].thumbnails().set.call_count == 1


def test_explicit_staging_binds_exact_bytes_without_overwrite(tmp_path):
    from automation.youtube_delivery import stage_thumbnail, sha256

    video = tmp_path / "portrait.mp4"
    video.write_bytes(b"video")
    source = tmp_path / "custom.jpg"
    source.write_bytes(artwork())
    staged = stage_thumbnail(video, source, expected_sha256=sha256(source))
    assert staged == tmp_path / "portrait_thumb.jpg"
    assert staged.read_bytes() == source.read_bytes()
    assert stage_thumbnail(video, source, expected_sha256=sha256(source)) == staged
    source.write_bytes(artwork(quality=80))
    with pytest.raises(FileExistsError):
        stage_thumbnail(video, source, expected_sha256=sha256(source))


def test_staging_rejects_changed_approval(tmp_path):
    from automation.youtube_delivery import stage_thumbnail

    video, source = tmp_path/"clip.mp4", tmp_path/"custom.jpg"
    video.write_bytes(b"video")
    source.write_bytes(artwork())
    with pytest.raises(ValueError, match="(?i)hash|digest"):
        stage_thumbnail(video, source, expected_sha256="0"*64)
    assert not (tmp_path/"clip_thumb.jpg").exists()


def test_no_upload_if_receipt_cannot_be_persisted(delivery, monkeypatch):
    monkeypatch.setattr("automation.youtube_delivery.atomic_json", Mock(side_effect=OSError("disk full")))
    with pytest.raises(RuntimeError, match="persist"):
        run_upload(delivery)
    delivery[3].videos().insert.assert_not_called()


def test_saved_video_id_survives_thumbnail_failure_without_metadata_mutation(delivery):
    original = delivery[1].read_bytes()
    delivery[3].thumbnails().set.side_effect = RuntimeError("rejected")
    with pytest.raises(RuntimeError):
        run_upload(delivery)
    assert receipt(delivery)["video_id"] == VIDEO_ID
    assert receipt(delivery)["artifacts"]["metadata"]["sha256"]
    assert delivery[1].read_bytes() == original
    assert not delivery[0].with_name("portrait_delivery.lock").exists()


def test_concurrent_upload_is_held_before_auth(delivery):
    from automation.youtube_delivery import delivery_lock

    with delivery_lock(delivery[0]):
        with pytest.raises(RuntimeError, match="locked"):
            run_upload(delivery)
    delivery[4].assert_not_called()


def test_readback_for_another_video_is_not_verified(delivery):
    delivery[3].videos().list.side_effect = lambda **_kw: Mock(execute=Mock(return_value={
        "items": [{"id": "different", "processingDetails": {"processingStatus": "succeeded"}}]}))
    with pytest.raises(RuntimeError, match="mismatched"):
        run_upload(delivery)
    delivery[3].thumbnails().set.assert_not_called()


def test_missing_custom_shorts_readback_is_not_success(delivery):
    original = delivery[3].videos().list.side_effect

    def missing(**kwargs):
        if kwargs["part"] == "snippet":
            return Mock(execute=Mock(return_value={"items": [{"id": VIDEO_ID, "snippet": {"thumbnails": {}}}]}))
        return original(**kwargs)

    delivery[3].videos().list.side_effect = missing
    with pytest.raises(RuntimeError, match="not visible"):
        run_upload(delivery)
    assert receipt(delivery)["status"] != "COMPLETE"


@pytest.mark.parametrize("url", ["http://i.ytimg.com/a.jpg", "https://attacker.test/a.jpg",
                               "https://i.ytimg.com.attacker.test/a.jpg"])
def test_cdn_readback_cannot_redirect_verification_to_arbitrary_hosts(url):
    from automation.youtube_delivery import download_thumbnail

    with pytest.raises(ValueError, match="host"):
        download_thumbnail(url, 1)


def test_pixel_check_accepts_resize_recompression_but_rejects_changed_hook(tmp_path):
    from automation.youtube_delivery import compare_thumbnail_pixels

    cover = tmp_path/"cover.jpg"
    cover.write_bytes(artwork((1080, 1920)))
    with Image.open(cover) as image:
        served = image.resize((360, 640), Image.Resampling.LANCZOS)
    encoded = io.BytesIO()
    served.save(encoded, format="JPEG", quality=78)
    assert compare_thumbnail_pixels(cover, encoded.getvalue())["pixel_rmse"] < 10
    ImageDraw.Draw(served).rectangle((55, 512, 305, 576), fill=(11, 15, 23))
    wrong = io.BytesIO()
    served.save(wrong, format="JPEG", quality=78)
    with pytest.raises(ValueError, match="do not match"):
        compare_thumbnail_pixels(cover, wrong.getvalue())
