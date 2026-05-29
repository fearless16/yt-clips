"""Tests for the GPU/identity face-detection logic (feat/face-detection-gpu).

These cover the GPU-independent paths (device/weight resolution, batched-detect
fallback, fail-loud, identity-reference loading). The actual YOLO GPU inference
is validated on a real T4 (ultralytics/torch are Colab-only and not installed
in this sandbox)."""
import numpy as np
import pytest
from unittest.mock import patch

import premium_analyzer as pa
from premium_analyzer import FaceDetector, PremiumAnalyzer


def test_resolve_device_without_torch_is_cpu():
    # No torch in this env -> auto/cpu both resolve to cpu (never crashes).
    assert FaceDetector._resolve_device("auto") == "cpu"
    assert FaceDetector._resolve_device("cpu") == "cpu"


def test_resolve_weights_repo_root_and_bare():
    # A real repo-root file resolves to an absolute existing path...
    resolved = FaceDetector._resolve_weights("config.yaml")
    assert resolved.endswith("config.yaml")
    from pathlib import Path
    assert Path(resolved).is_absolute() and Path(resolved).exists()
    # ...a missing name is returned bare (ultralytics may auto-download known ones).
    assert FaceDetector._resolve_weights("definitely_missing_weights.pt") == "definitely_missing_weights.pt"


def test_detect_batch_falls_back_per_frame_without_yolo():
    fd = FaceDetector()  # backend == "dnn" here (no ultralytics)
    assert fd.backend == "dnn"
    frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(3)]
    sentinel = (np.empty((0, 4)), np.empty((0,)))
    with patch.object(fd, "detect", return_value=sentinel) as mock_detect:
        out = fd.detect_batch(frames)
    assert len(out) == 3
    assert mock_detect.call_count == 3


def test_detect_batch_empty_input():
    assert FaceDetector().detect_batch([]) == []


def test_yolo_require_raises_when_unavailable():
    """yolo_require=true must fail loudly instead of silently using CPU DNN."""
    patched = {**pa.cfg, "premium": {**pa.cfg.get("premium", {}), "yolo_require": True}}
    with patch.object(pa, "cfg", patched), patch.object(pa, "HAS_YOLO", False):
        with pytest.raises(RuntimeError):
            FaceDetector()


def test_identity_refs_resolve_photos_and_expectation():
    """Identity locking references (photos/ + expectation.png) load from repo root."""
    imgs = PremiumAnalyzer._load_identity_images(["photos", "expectation.png"])
    # photos/ has p1.png, p2.png and expectation.png exists at the repo root.
    assert len(imgs) >= 3
    assert all(isinstance(i, np.ndarray) for i in imgs)


def test_identity_refs_ignore_missing():
    assert PremiumAnalyzer._load_identity_images(["nope_dir", ""]) == []


def test_no_haar_cascade_anywhere_in_legacy_path():
    """Regression guard: the legacy path must not reintroduce Haar Cascade."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    for name in ("premium_analyzer.py", "frame_analyzer.py", "face_mapper.py", "utils/face_detect.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert "CascadeClassifier" not in text
        assert "haarcascade" not in text.lower()
