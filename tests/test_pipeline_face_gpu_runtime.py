"""GPU face detector regressions for the clip pipeline (not FaceOS)."""

import numpy as np
import pytest


def _empty_scrfd_outputs():
    return [
        np.zeros((1, 12800, 1), dtype=np.float32),
        np.zeros((1, 3200, 1), dtype=np.float32),
        np.zeros((1, 800, 1), dtype=np.float32),
        np.zeros((1, 12800, 4), dtype=np.float32),
        np.zeros((1, 3200, 4), dtype=np.float32),
        np.zeros((1, 800, 4), dtype=np.float32),
        np.zeros((1, 12800, 10), dtype=np.float32),
        np.zeros((1, 3200, 10), dtype=np.float32),
        np.zeros((1, 800, 10), dtype=np.float32),
    ]


def test_scrfd_decodes_both_anchors_and_scales_distances_by_stride():
    from utils.face_detect import _decode_scrfd

    outputs = _empty_scrfd_outputs()
    # Second anchor at the first stride-8 cell. SCRFD outputs distances in
    # feature-map units, so [1,1,1,1] must be multiplied by stride 8.
    outputs[0][0, 1, 0] = 0.95
    outputs[3][0, 1] = [1.0, 1.0, 1.0, 1.0]

    assert _decode_scrfd(outputs, 640, 1.0, 0.5) == [(0, 0, 12, 12)]


def test_runtime_reports_directml_as_active_face_provider():
    from utils.face_detect import get_backend_info

    info = get_backend_info()
    assert info["active_provider"] == "DmlExecutionProvider"
    assert info["gpu_enabled"] is True


def test_gpu_provider_selection_never_silently_chooses_cpu():
    from utils.face_detect import _select_gpu_provider

    assert _select_gpu_provider(["CPUExecutionProvider", "CUDAExecutionProvider"]) == (
        "CUDAExecutionProvider"
    )
    assert _select_gpu_provider(["CPUExecutionProvider", "DmlExecutionProvider"]) == (
        "DmlExecutionProvider"
    )
    with pytest.raises(RuntimeError, match="GPU execution provider"):
        _select_gpu_provider(["CPUExecutionProvider"])


def test_clip_pipeline_deduplicates_identical_identity_references():
    from premium_analyzer import PremiumAnalyzer

    images = PremiumAnalyzer._load_identity_images(
        ["photos/p3.png", "expectation.png"]
    )
    assert len(images) == 1
