"""Private-first publication and retry regressions; fake Google/CDN only."""
import copy
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

import upload
from tests.automation.test_youtube_thumbnail_delivery import (
    VIDEO_ID, delivery, receipt, run_upload,
)


@pytest.fixture
def publication(delivery, monkeypatch):
    service, events = delivery[3], delivery[6]
    original_list = service.videos().list.side_effect
    remote = {"status": {"privacyStatus": "private", "embeddable": False,
                         "license": "youtube", "publicStatsViewable": False,
                         "selfDeclaredMadeForKids": False, "containsSyntheticMedia": True},
              "channel_id": "fixture-channel"}
    monkeypatch.setitem(upload.cfg["youtube"], "channel_id", remote["channel_id"])

    def listed(**kwargs):
        if "status" not in kwargs["part"]:
            return original_list(**kwargs)
        events.append("status-read")
        return Mock(execute=Mock(return_value={"items": [{"id": VIDEO_ID,
            "status": copy.deepcopy(remote["status"]),
            "snippet": {"channelId": remote["channel_id"]}}]}))

    def updated(**kwargs):
        def execute():
            events.append("status-update")
            remote["receipt_at_update"] = receipt(delivery)
            remote["status"] = copy.deepcopy(kwargs["body"]["status"])
            return {"id": VIDEO_ID, "status": copy.deepcopy(remote["status"])}
        return Mock(execute=Mock(side_effect=execute))

    service.videos().list.side_effect = listed
    service.videos().update.side_effect = updated
    from automation import youtube_delivery
    compare = youtube_delivery.compare_thumbnail_pixels

    def compared(*args):
        result = compare(*args)
        events.append("pixels-verified")
        return result

    monkeypatch.setattr(youtube_delivery, "compare_thumbnail_pixels", compared)
    return delivery, remote


def upload_as(publication, privacy="public", publish_at=None):
    d = publication[0]
    return upload.upload_video(str(d[0]), str(d[1]), privacy=privacy, publish_at=publish_at)


def future_slot(days=2):
    return (datetime.now(timezone.utc) + timedelta(days=days)).replace(
        microsecond=0).isoformat().replace("+00:00", "Z")


@pytest.mark.parametrize("privacy", ["public", "unlisted", "scheduled"])
def test_incomplete_cover_cannot_publish_or_schedule(publication, privacy):
    d, remote = publication
    d[3].thumbnails().set.side_effect = RuntimeError("fixture thumbnail rejection")
    with pytest.raises(upload.DeliveryIncompleteError):
        upload_as(publication, privacy, future_slot() if privacy == "scheduled" else None)
    inserted = d[3].videos().insert.call_args.kwargs["body"]["status"]
    assert inserted["privacyStatus"] == "private"
    assert "publishAt" not in inserted
    d[3].videos().update.assert_not_called()
    assert remote["status"]["privacyStatus"] == "private"


@pytest.mark.parametrize("privacy", ["public", "unlisted"])
def test_publication_happens_only_after_durable_verified_cover(publication, privacy):
    d, remote = publication
    original = d[1].read_bytes()
    assert upload_as(publication, privacy) == VIDEO_ID
    assert d[6].index("pixels-verified") < d[6].index("status-update")
    at_update = remote["receipt_at_update"]
    assert at_update["status"] == "PUBLICATION_UNKNOWN"
    assert at_update["thumbnail"]["status"] == "VERIFIED"
    assert at_update["intent"]["privacy"] == privacy
    update = d[3].videos().update.call_args.kwargs
    assert update["part"] == "status"
    assert update["body"]["status"] == {
        "privacyStatus": privacy, "embeddable": False, "license": "youtube",
        "publicStatsViewable": False, "selfDeclaredMadeForKids": False,
        "containsSyntheticMedia": True,
    }
    assert receipt(d)["publication"]["status"] == "VERIFIED"
    assert receipt(d)["status"] == "COMPLETE"
    assert d[1].read_bytes() == original


def test_schedule_is_applied_only_after_cover_verification(publication):
    d, remote = publication
    slot = future_slot()
    assert upload_as(publication, "scheduled", slot) == VIDEO_ID
    assert "publishAt" not in d[3].videos().insert.call_args.kwargs["body"]["status"]
    assert remote["status"]["publishAt"] == slot
    assert remote["status"]["privacyStatus"] == "private"
    assert receipt(d)["intent"] == {
        "privacy": "scheduled", "publish_at": slot, "notify_subscribers": True}


@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("change", ["privacy", "schedule", "notify"])
def test_changed_intent_is_rejected_before_auth_or_mutation(publication, monkeypatch, pending, change):
    d, _ = publication
    slot = future_slot()
    if pending:
        d[3].thumbnails().set.side_effect = RuntimeError("fixture pending")
        with pytest.raises(upload.DeliveryIncompleteError):
            upload_as(publication, "scheduled", slot)
    else:
        assert upload_as(publication, "scheduled", slot) == VIDEO_ID
    before = d[0].with_name("portrait_delivery.json").read_bytes()
    d[4].reset_mock()
    d[3].reset_mock()
    if change == "notify":
        monkeypatch.setitem(upload.cfg["youtube"], "notify_subscribers", False)
    with pytest.raises(upload.DeliveryIncompleteError, match="(?i)intent|privacy|schedule|settings"):
        upload_as(publication, "private" if change == "privacy" else "scheduled",
                  None if change == "privacy" else future_slot(3) if change == "schedule" else slot)
    d[4].assert_not_called()
    d[3].videos().insert.assert_not_called()
    d[3].videos().update.assert_not_called()
    assert d[0].with_name("portrait_delivery.json").read_bytes() == before


def test_implicit_schedule_reuses_saved_time_instead_of_rescheduling(publication, monkeypatch):
    d, _ = publication
    slot = future_slot()
    scheduler = Mock(return_value=slot)
    monkeypatch.setattr("scheduler.get_next_upload_time", scheduler)
    assert upload_as(publication, "scheduled") == VIDEO_ID
    assert upload_as(publication, "scheduled") == VIDEO_ID
    assert receipt(d)["intent"]["publish_at"] == slot
    scheduler.assert_called_once()
    assert d[3].videos().insert.call_count == 1


def test_changed_private_delivery_does_not_reserve_a_new_schedule_before_rejection(publication, monkeypatch):
    assert upload_as(publication, "private") == VIDEO_ID
    scheduler = Mock(return_value=future_slot())
    monkeypatch.setattr("scheduler.get_next_upload_time", scheduler)
    with pytest.raises(upload.DeliveryIncompleteError, match="(?i)intent|privacy|settings"):
        upload_as(publication, "scheduled")
    scheduler.assert_not_called()


def test_old_receipt_without_publication_intent_is_held(publication):
    d, _ = publication
    from automation.youtube_delivery import artifact_hashes
    legacy = {"schema_version": 1, "channel_id": "fixture-channel",
              "artifacts": artifact_hashes(d[0], d[1], d[2]), "video_id": VIDEO_ID,
              "status": "COMPLETE", "thumbnail": {"status": "VERIFIED"}}
    d[0].with_name("portrait_delivery.json").write_text(json.dumps(legacy))
    with pytest.raises(upload.DeliveryIncompleteError, match="(?i)legacy|intent|reconcil"):
        upload_as(publication)
    d[4].assert_not_called()


def test_update_timeout_reconciles_applied_publication_without_replay(publication):
    d, remote = publication
    normal = d[3].videos().update.side_effect

    def timed_out(**kwargs):
        request = normal(**kwargs)
        apply = request.execute.side_effect
        def execute():
            apply()
            raise TimeoutError("response lost after publication")
        request.execute.side_effect = execute
        return request

    d[3].videos().update.side_effect = timed_out
    with pytest.raises(upload.DeliveryIncompleteError, match="(?i)publication"):
        upload_as(publication)
    assert receipt(d)["status"] == "PUBLICATION_UNKNOWN"
    assert receipt(d)["thumbnail"]["status"] == "VERIFIED"
    d[3].videos().update.side_effect = normal
    assert upload_as(publication) == VIDEO_ID
    assert d[3].videos().insert.call_count == 1
    assert d[3].videos().update.call_count == 1
    assert d[3].thumbnails().set.call_count == 1


def test_failed_publication_resumes_on_existing_private_video(publication):
    d, _ = publication
    normal = d[3].videos().update.side_effect
    d[3].videos().update.side_effect = RuntimeError("fixture update refused")
    with pytest.raises(upload.DeliveryIncompleteError, match="(?i)publication"):
        upload_as(publication)
    d[3].videos().update.side_effect = normal
    assert upload_as(publication) == VIDEO_ID
    assert d[3].videos().insert.call_count == 1
    assert d[3].thumbnails().set.call_count == 1


def test_update_ack_without_matching_readback_cannot_complete(publication):
    d, _ = publication
    d[3].videos().update.side_effect = lambda **_: Mock(execute=Mock(return_value={"id": VIDEO_ID}))
    with pytest.raises(upload.DeliveryIncompleteError, match="(?i)publication|readback"):
        upload_as(publication)
    assert receipt(d)["status"] != "COMPLETE"


def test_wrong_channel_readback_blocks_visibility_changes(publication):
    d, remote = publication
    remote["channel_id"] = "other-channel"
    with pytest.raises(upload.DeliveryIncompleteError, match="(?i)channel"):
        upload_as(publication)
    d[3].videos().update.assert_not_called()


def test_artifact_change_before_publication_never_updates_visibility(publication, monkeypatch):
    d, _ = publication
    from automation import youtube_delivery
    normal = youtube_delivery.compare_thumbnail_pixels
    def changed(*args):
        result = normal(*args)
        d[1].write_text('{"title":"Unapproved change"}')
        return result
    monkeypatch.setattr(youtube_delivery, "compare_thumbnail_pixels", changed)
    with pytest.raises(upload.DeliveryIncompleteError, match="(?i)changed"):
        upload_as(publication)
    d[3].videos().update.assert_not_called()


def test_artifact_change_during_publication_cannot_claim_complete_delivery(publication):
    d, _ = publication
    normal = d[3].videos().update.side_effect
    def changed(**kwargs):
        request = normal(**kwargs)
        apply = request.execute.side_effect
        def execute():
            result = apply()
            d[1].write_text('{"title":"Changed while publication was in flight"}')
            return result
        request.execute.side_effect = execute
        return request
    d[3].videos().update.side_effect = changed
    with pytest.raises(upload.DeliveryIncompleteError, match="(?i)changed"):
        upload_as(publication)
    assert receipt(d)["status"] == "PUBLICATION_UNKNOWN"


def test_elapsed_schedule_is_not_applied_as_immediate_publication(publication, monkeypatch):
    d, _ = publication
    from automation import youtube_delivery
    slot = future_slot()
    normal = youtube_delivery.compare_thumbnail_pixels
    def elapsed(*args):
        result = normal(*args)
        monkeypatch.setattr(youtube_delivery, "_utc_now", lambda: datetime.now(timezone.utc) + timedelta(days=3), raising=False)
        return result
    monkeypatch.setattr(youtube_delivery, "compare_thumbnail_pixels", elapsed)
    with pytest.raises(upload.DeliveryIncompleteError, match="(?i)schedule|past|expired"):
        upload_as(publication, "scheduled", slot)
    d[3].videos().update.assert_not_called()


def test_publication_intent_must_be_durable_before_visibility_update(publication, monkeypatch):
    d, _ = publication
    from automation import youtube_delivery
    persist = youtube_delivery.atomic_json
    def save(path, payload):
        if payload.get("status") == "PUBLICATION_UNKNOWN":
            raise OSError("fixture disk full before publication")
        persist(path, payload)
    monkeypatch.setattr(youtube_delivery, "atomic_json", save)
    with pytest.raises(upload.DeliveryIncompleteError, match="persist"):
        upload_as(publication)
    d[3].videos().update.assert_not_called()
    assert receipt(d)["status"] == "PUBLICATION_PENDING"


def test_lost_complete_receipt_resumes_by_status_without_republishing(publication, monkeypatch):
    d, _ = publication
    from automation import youtube_delivery
    persist = youtube_delivery.atomic_json
    def save(path, payload):
        if payload.get("status") == "COMPLETE":
            raise OSError("fixture completion receipt lost")
        persist(path, payload)
    monkeypatch.setattr(youtube_delivery, "atomic_json", save)
    with pytest.raises(upload.DeliveryIncompleteError, match="persist"):
        upload_as(publication)
    monkeypatch.setattr(youtube_delivery, "atomic_json", persist)
    assert upload_as(publication) == VIDEO_ID
    assert d[3].videos().update.call_count == 1
    assert d[3].thumbnails().set.call_count == 1


def test_release_approval_is_rechecked_before_publication(publication, monkeypatch):
    d, _ = publication
    monkeypatch.setitem(upload.cfg, "release_review", {"required": True})
    gate = Mock(side_effect=[None, ValueError("release revoked during verification")])
    monkeypatch.setattr("automation.release_review.require_ready", gate)
    with pytest.raises(upload.DeliveryIncompleteError, match="revoked"):
        upload_as(publication)
    assert gate.call_count == 2
    d[3].videos().update.assert_not_called()


def test_ambiguous_applied_schedule_that_has_since_published_is_not_replayed(publication, monkeypatch):
    d, remote = publication
    from automation import youtube_delivery
    slot = future_slot()
    normal = d[3].videos().update.side_effect
    def lost(**kwargs):
        request = normal(**kwargs)
        apply = request.execute.side_effect
        def execute():
            apply()
            raise TimeoutError("schedule response lost")
        request.execute.side_effect = execute
        return request
    d[3].videos().update.side_effect = lost
    with pytest.raises(upload.DeliveryIncompleteError):
        upload_as(publication, "scheduled", slot)
    remote["status"] = {"privacyStatus": "public"}
    monkeypatch.setattr(youtube_delivery, "_utc_now", lambda: datetime.now(timezone.utc) + timedelta(days=3))
    assert upload_as(publication, "scheduled", slot) == VIDEO_ID
    assert d[3].videos().update.call_count == 1
    assert d[3].videos().insert.call_count == 1
