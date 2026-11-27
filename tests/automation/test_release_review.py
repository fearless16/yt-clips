import json

import pytest

from automation.artifacts import atomic_json
from automation.release_review import (approve, create_review, prepare_review, reject, release_status,
                                       reset, review_path, sha256)


@pytest.fixture
def package(tmp_path):
    clip = tmp_path / "clip1.mp4"
    source = tmp_path / "source.mp4"
    clip.write_bytes(b"encoded clip fixture")
    source.write_bytes(b"source fixture")
    clip.with_name("clip1_thumb.jpg").write_bytes(b"cover fixture")
    atomic_json(clip.with_name("clip1_metadata.json"),
                {"title": "Specific cricket premise", "description": "Actual retained clip context"})
    atomic_json(clip.with_name("clip1_qc.json"), {"status": "PASS", "media_sha256": sha256(clip)})
    edit(clip, source, "text_only")
    return clip, source


def edit(clip, source, method):
    atomic_json(clip.with_name("clip1_edit.json"), {
        "schema_version": 1, "source_path": str(source), "source_start": 10.25, "source_end": 30.5,
        "completeness": {"status": "PASS", "method": method, "reason": "setup and payoff retained",
                         "source_range": [10.25, 30.5]}})


def test_manual_mode_needs_exactly_one_owner_decision(package):
    clip, source = package
    assert release_status(clip)[0] == "REVIEW"
    draft = prepare_review(clip, source)
    assert all(c["status"] == "PASS" for c in draft["checks"].values())
    assert release_status(clip) == ("REVIEW", ["Awaiting owner decision"])
    approve(clip, "Owner")
    assert release_status(clip) == ("READY", [])
    with pytest.raises(FileExistsError):
        create_review(clip, source)


def test_auto_tier_releases_clean_clips_without_hand_typed_evidence(package):
    clip, source = package
    payload = prepare_review(clip, source, mode="auto")
    assert payload["approval"]["mode"] == "auto"
    assert release_status(clip) == ("READY", [])


def test_heuristic_fallback_clip_always_needs_the_owner(package):
    clip, source = package
    edit(clip, source, "heuristic_fallback")
    payload = prepare_review(clip, source, mode="auto")
    assert payload["approval"] is None
    assert payload["checks"]["complete_meaning"]["status"] == "NEEDS_OWNER"
    status, reasons = release_status(clip)
    assert status == "REVIEW" and "complete_meaning" in reasons[0]
    approve(clip, "Owner")
    assert release_status(clip)[0] == "READY"


@pytest.mark.parametrize("artifact", ["clip1.mp4", "source.mp4", "clip1_thumb.jpg", "clip1_metadata.json"])
def test_artifact_change_invalidates_then_self_heals(package, artifact):
    clip, source = package
    prepare_review(clip, source)
    approve(clip, "Owner")
    path = clip.parent / artifact
    path.write_bytes(path.read_bytes() + b"modified")
    if artifact == "clip1.mp4":
        atomic_json(clip.with_name("clip1_qc.json"), {"status": "PASS", "media_sha256": sha256(clip)})
    elif artifact == "clip1_metadata.json":
        atomic_json(path, {"title": "Changed promise", "description": "Changed"})
    assert release_status(clip)[0] == "REVIEW"
    # No manual file surgery: the next run archives and rebuilds (A4).
    assert prepare_review(clip, source, mode="auto")["approval"]["mode"] == "auto"
    assert release_status(clip)[0] == "READY"
    assert list((clip.parent / ".review-archive").glob("clip1_review.*.json"))


def test_hand_edited_review_cannot_fake_a_pass(package):
    clip, source = package
    atomic_json(clip.with_name("clip1_qc.json"), {"status": "FAIL", "media_sha256": sha256(clip)})
    payload = prepare_review(clip, source, mode="auto")
    assert payload["approval"] is None
    payload["checks"]["full_decode"]["status"] = "PASS"
    atomic_json(review_path(clip), payload)
    with pytest.raises(ValueError, match="Cannot approve"):
        approve(clip, "Owner")
    assert release_status(clip)[0] == "REVIEW"


def test_owner_rejection_sticks_until_reset(package):
    clip, source = package
    prepare_review(clip, source, mode="auto")
    reject(clip, "Owner", "weak payoff")
    clip.write_bytes(clip.read_bytes() + b"re-render")
    atomic_json(clip.with_name("clip1_qc.json"), {"status": "PASS", "media_sha256": sha256(clip)})
    prepare_review(clip, source, mode="auto")
    assert release_status(clip)[0] == "REJECT"
    reset(clip)
    assert release_status(clip)[0] == "REVIEW"


def test_legacy_review_is_archived_and_rebuilt(package):
    clip, source = package
    atomic_json(review_path(clip), {"schema_version": 1, "checks": {}, "approval": None})
    assert release_status(clip)[0] == "REVIEW"
    assert prepare_review(clip, source)["schema_version"] == 2


def test_failed_seo_marker_invalidates_existing_approval(package):
    clip, source = package
    prepare_review(clip, source)
    approve(clip, "Owner")
    atomic_json(clip.with_name("clip1_seo_failed.json"), {"error": "latest regeneration failed"})
    assert release_status(clip)[0] == "REVIEW"


def test_upload_is_held_before_authentication(package, monkeypatch):
    import upload

    clip, _ = package
    monkeypatch.setitem(upload.cfg, "release_review", {"required": True})
    monkeypatch.setattr(upload, "get_authenticated_service",
                        lambda: pytest.fail("Authentication must not happen for an unreviewed clip"))
    with pytest.raises(upload.UploadRejected, match="Release held"):
        upload.upload_video(str(clip), str(clip.with_name("clip1_metadata.json")))
