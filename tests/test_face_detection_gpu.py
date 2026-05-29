"""
test_face_detection_gpu.py — Unit tests for the GPU/identity face-detection
improvements in premium_analyzer.py (feat/face-detection-gpu).

These tests run WITHOUT a GPU and without ultralytics/torch installed: they
exercise the device/weight resolution, the identity-reference loader, and the
batched-detect API's CPU fallback path. Real GPU throughput is validated
separately on a T4 (Colab/Kaggle).
"""
import numpy as np
import pytest

import premium_analyzer as pa


# ─── device resolution ────────────────────────────────────────────────────

def test_resolve_device_cpu_explicit():
    assert pa.FaceDetector._resolve_device("cpu") == "cpu"


def test_resolve_device_auto_without_torch_is_cpu(monkeypatch):
    # When torch is unavailable, auto must resolve to cpu (never crash).
    monkeypatch.setattr(pa, "HAS_TORCH", False, raising=False)
    assert pa.FaceDetector._resolve_device("auto") == "cpu"


def test_resolve_device_auto_with_cuda(monkeypatch):
    monkeypatch.setattr(pa, "HAS_TORCH", True, raising=False)

    class _FakeCuda:
        @staticmethod
        def is_available():
            return True

    class _FakeTorch:
        cuda = _FakeCuda()

    monkeypatch.setattr(pa, "torch", _FakeTorch(), raising=False)
    assert pa.FaceDetector._resolve_device("auto") == "cuda:0"


def test_resolve_device_cuda_requested_but_unavailable_falls_back(monkeypatch):
    monkeypatch.setattr(pa, "HAS_TORCH", True, raising=False)

    class _FakeCuda:
        @staticmethod
        def is_available():
            return False

    class _FakeTorch:
        cuda = _FakeCuda()

    monkeypatch.setattr(pa, "torch", _FakeTorch(), raising=False)
    # Asking for cuda when none exists must degrade to cpu, not crash.
    assert pa.FaceDetector._resolve_device("cuda") == "cpu"


# ─── weight path resolution ────────────────────────────────────────────────

def test_resolve_weights_repo_root():
    # Bare known name resolves to repo-root candidate path or the bare name
    # (both acceptable — ultralytics can auto-download known names).
    out = pa.FaceDetector._resolve_weights("yolov8n-face.pt")
    assert out.endswith("yolov8n-face.pt")


def test_resolve_weights_absolute_missing_returns_name(tmp_path):
    missing = tmp_path / "no_such.pt"
    # Absolute but missing -> returns the bare name (don't fabricate a path).
    assert pa.FaceDetector._resolve_weights(str(missing)).endswith("no_such.pt")


# ─── fail-loud contract ─────────────────────────────────────────────────────

def test_yolo_require_raises_without_ultralytics(monkeypatch):
    """premium.yolo_require=true must raise when YOLO is unavailable instead of
    silently degrading to the CPU MediaPipe fallback."""
    monkeypatch.setattr(pa, "HAS_YOLO", False, raising=False)
    monkeypatch.setitem(pa.cfg, "premium", {"yolo_require": True})
    with pytest.raises(RuntimeError):
        pa.FaceDetector()


def test_default_facedetector_degrades_to_dnn(monkeypatch):
    """Without YOLO and without require, FaceDetector falls back to the
    MediaPipe DNN backend (no crash)."""
    monkeypatch.setattr(pa, "HAS_YOLO", False, raising=False)
    monkeypatch.setitem(pa.cfg, "premium", {"yolo_require": False})
    fd = pa.FaceDetector()
    assert fd.backend == "dnn"


# ─── batched detect fallback ────────────────────────────────────────────────

def test_detect_batch_empty_returns_empty():
    fd = pa.FaceDetector.__new__(pa.FaceDetector)
    fd.backend = "dnn"
    fd.yolo_model = None
    assert fd.detect_batch([]) == []


def test_detect_batch_uses_per_frame_when_no_yolo(monkeypatch):
    fd = pa.FaceDetector.__new__(pa.FaceDetector)
    fd.backend = "dnn"
    fd.yolo_model = None
    calls = {"n": 0}

    def fake_detect(frame):
        calls["n"] += 1
        return (np.empty((0, 4)), np.empty((0,)))

    fd.detect = fake_detect  # type: ignore
    frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(3)]
    out = fd.detect_batch(frames)
    assert len(out) == 3
    assert calls["n"] == 3  # one per-frame detect call each


# ─── identity reference loader ──────────────────────────────────────────────

def test_load_identity_images_from_repo_assets():
    # The repo ships photos/ (7 imgs) + expectation.png.
    imgs = pa.PremiumAnalyzer._load_identity_images(["photos", "expectation.png"])
    assert len(imgs) >= 1


def test_load_identity_images_ignores_missing():
    assert pa.PremiumAnalyzer._load_identity_images(["does_not_exist_xyz.png"]) == []


def test_load_identity_images_dedups(tmp_path):
    # Same file referenced twice should load once.
    import cv2
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    f = tmp_path / "ref.png"
    cv2.imwrite(str(f), img)
    out = pa.PremiumAnalyzer._load_identity_images([str(f), str(f)])
    assert len(out) == 1
