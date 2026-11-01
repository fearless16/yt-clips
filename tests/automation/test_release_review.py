import json
from pathlib import Path

import pytest

from automation.artifacts import atomic_json
from automation.release_review import CHECKS, approve, create_review, release_status, review_path


@pytest.fixture
def package(tmp_path):
    clip = tmp_path / "clip1.mp4"
    source = tmp_path / "source.mp4"
    clip.write_bytes(b"encoded clip fixture")
    source.write_bytes(b"source fixture")
    clip.with_name("clip1_thumb.jpg").write_bytes(b"cover fixture")
    atomic_json(clip.with_name("clip1_metadata.json"), {"title": "Specific cricket premise", "description": "Actual retained clip context"})
    from automation.release_review import sha256
    atomic_json(clip.with_name("clip1_qc.json"), {"status": "PASS", "media_sha256": sha256(clip)})
    return clip, source


def completed_review(clip, source):
    payload = create_review(clip, source)
    for name in CHECKS:
        payload["checks"][name] = {"status": "PASS", "evidence": f"Fixture attestation for {name}", "method": "test, not real QC"}
    payload["selection"] = {"source_ranges": [[10.25, 30.5]], "reason": "Complete argument with payoff",
                            "alternatives": [{"source_range": [50, 70], "reason": "Weaker standalone context"}], "coverage_gaps": []}
    atomic_json(review_path(clip), payload)
    return payload


def test_missing_or_unchecked_review_is_held(package):
    clip, source = package
    assert release_status(clip)[0] == "REVIEW"
    draft = create_review(clip, source)
    assert draft["checks"]["full_decode"]["status"] == "PASS"
    assert all(c["status"] == "NOT CHECKED" for name, c in draft["checks"].items() if name != "full_decode")
    with pytest.raises(ValueError, match="Review incomplete"):
        approve(clip, "Owner")
    with pytest.raises(FileExistsError):
        create_review(clip, source)


def test_complete_review_can_be_approved_without_upload(package):
    clip, source = package
    completed_review(clip, source)
    approve(clip, "Owner")
    assert release_status(clip) == ("READY", [])


@pytest.mark.parametrize("artifact", ["clip1.mp4", "source.mp4", "clip1_thumb.jpg", "clip1_metadata.json", "clip1_review.json"])
def test_any_artifact_or_review_change_invalidates_approval(package, artifact):
    clip, source = package
    completed_review(clip, source)
    approve(clip, "Owner")
    path = clip.parent / artifact
    if artifact.endswith("review.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["selection"]["reason"] = "Changed after approval"
        atomic_json(path, payload)
    else:
        path.write_bytes(path.read_bytes() + b"modified")
    assert release_status(clip)[0] == "REVIEW"


def test_failed_gate_or_gap_cannot_be_approved(package):
    clip, source = package
    payload = completed_review(clip, source)
    payload["checks"]["complete_meaning"]["status"] = "FAIL"
    payload["selection"]["coverage_gaps"] = [[0, 10]]
    atomic_json(review_path(clip), payload)
    assert release_status(clip)[0] == "REJECT"
    with pytest.raises(ValueError):
        approve(clip, "Owner")


def test_failed_seo_marker_invalidates_existing_approval(package):
    clip, source = package
    completed_review(clip, source)
    approve(clip, "Owner")
    atomic_json(clip.with_name("clip1_seo_failed.json"), {"error": "latest regeneration failed"})
    assert release_status(clip)[0] == "REVIEW"


def test_upload_is_held_before_authentication(package, monkeypatch):
    import upload

    clip, source = package
    monkeypatch.setitem(upload.cfg, "release_review", {"required": True})
    def forbidden():
        pytest.fail("Authentication must not happen for an unreviewed clip")
    monkeypatch.setattr(upload, "get_authenticated_service", forbidden)
    assert upload.upload_video(str(clip), str(clip.with_name("clip1_metadata.json"))) is None
