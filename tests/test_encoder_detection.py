"""
Tests for GPU-aware encoder detection (export.py).

AMD machines should try h264_amf first, skip nvenc/qsv/vaapi.
NVIDIA machines should try h264_nvenc first.
Intel machines should try h264_qsv first.
Fallback to libx264 always.
"""

import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest


# ── Helpers ──────────────────────────────────────────────────────────────────

def _mock_ffmpeg_encoders(output: str):
    """Return a mock for subprocess.run that returns given ffmpeg -encoders output."""
    def _run(cmd, **kwargs):
        if isinstance(cmd, list) and "-encoders" in cmd:
            return MagicMock(returncode=0, stdout=output, stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")
    return _run


# ── Test: AMD GPU → h264_amf first ──────────────────────────────────────────

class TestEncoderDetectionAMD:
    def test_amd_gpu_prefers_amf_encoder(self):
        """On AMD GPU, h264_amf should be tried before nvenc/qsv/vaapi."""
        import export

        with patch.object(export, "_BEST_ENCODER", None):
            encoders_listing = (
                " V....D h264_amf             AMD AMF H.264 Encoder\n"
                " V....D h264_nvenc           NVIDIA NVENC H.264 encoder\n"
                " V..... h264_qsv             H.264 / AVC (Intel Quick Sync)\n"
            )
            with patch.object(export, "_detect_gpu_vendor", return_value="amd"):
                with patch("subprocess.run", side_effect=_mock_ffmpeg_encoders(encoders_listing)):
                    with patch.object(export, "_smoke_test_encoder", return_value=True) as mock_smoke:
                        result = export._get_best_encoder()
                        first_call = mock_smoke.call_args_list[0][0][0]
                        assert first_call == "h264_amf", f"Expected h264_amf first, got {first_call}"

    def test_amd_gpu_skips_nvenc_if_amf_works(self):
        """When h264_amf smoke test passes, nvenc should not be tested."""
        import export

        with patch.object(export, "_BEST_ENCODER", None):
            encoders_listing = (
                " V....D h264_amf             AMD AMF H.264 Encoder\n"
                " V....D h264_nvenc           NVIDIA NVENC H.264 encoder\n"
            )
            with patch.object(export, "_detect_gpu_vendor", return_value="amd"):
                with patch("subprocess.run", side_effect=_mock_ffmpeg_encoders(encoders_listing)):
                    with patch.object(export, "_smoke_test_encoder", return_value=True) as mock_smoke:
                        result = export._get_best_encoder()
                        assert result == "h264_amf"
                        assert mock_smoke.call_count == 1

    def test_amd_amf_fails_falls_to_nvenc(self):
        """When h264_amf fails smoke test, should try next candidate."""
        import export

        with patch.object(export, "_BEST_ENCODER", None):
            encoders_listing = (
                " V....D h264_amf             AMD AMF H.264 Encoder\n"
                " V....D h264_nvenc           NVIDIA NVENC H.264 encoder\n"
            )
            with patch.object(export, "_detect_gpu_vendor", return_value="amd"):
                with patch("subprocess.run", side_effect=_mock_ffmpeg_encoders(encoders_listing)):
                    with patch.object(export, "_smoke_test_encoder", side_effect=[False, True]) as mock_smoke:
                        result = export._get_best_encoder()
                        assert result == "h264_nvenc"
                        assert mock_smoke.call_count == 2

    def test_amd_no_hardware_encoder_falls_to_libx264(self):
        """When all hardware encoders fail, should return libx264."""
        import export

        with patch.object(export, "_BEST_ENCODER", None):
            encoders_listing = " V....D h264_amf             AMD AMF H.264 Encoder\n"
            with patch("subprocess.run", side_effect=_mock_ffmpeg_encoders(encoders_listing)):
                with patch.object(export, "_smoke_test_encoder", return_value=False):
                    result = export._get_best_encoder()
                    assert result == "libx264"


# ── Test: NVIDIA GPU → nvenc first ─────────────────────────────────────────

class TestEncoderDetectionNVIDIA:
    def test_nvidia_gpu_prefers_nvenc(self):
        """On NVIDIA GPU, h264_nvenc should be tried first."""
        import export

        with patch.object(export, "_BEST_ENCODER", None):
            encoders_listing = (
                " V....D h264_nvenc           NVIDIA NVENC H.264 encoder\n"
                " V....D h264_amf             AMD AMF H.264 Encoder\n"
            )
            with patch("subprocess.run", side_effect=_mock_ffmpeg_encoders(encoders_listing)):
                with patch.object(export, "_smoke_test_encoder", return_value=True) as mock_smoke:
                    result = export._get_best_encoder()
                    first_call = mock_smoke.call_args_list[0][0][0]
                    assert first_call == "h264_nvenc"


# ── Test: Intel GPU → qsv first ────────────────────────────────────────────

class TestEncoderDetectionIntel:
    def test_intel_gpu_prefers_qsv(self):
        """On Intel GPU, h264_qsv should be tried first."""
        import export

        with patch.object(export, "_BEST_ENCODER", None):
            encoders_listing = (
                " V..... h264_qsv             H.264 / AVC (Intel Quick Sync)\n"
                " V....D h264_nvenc           NVIDIA NVENC H.264 encoder\n"
            )
            with patch.object(export, "_detect_gpu_vendor", return_value="intel"):
                with patch("subprocess.run", side_effect=_mock_ffmpeg_encoders(encoders_listing)):
                    with patch.object(export, "_smoke_test_encoder", return_value=True) as mock_smoke:
                        result = export._get_best_encoder()
                        first_call = mock_smoke.call_args_list[0][0][0]
                        assert first_call == "h264_qsv"


# ── Test: GPU vendor detection ──────────────────────────────────────────────

class TestGPUDetection:
    def test_amd_detected_via_wmic(self):
        """AMD GPU detected from Win32_VideoController output."""
        import export
        mock_output = "AMD Radeon AI PRO R9700"
        vendor = export._detect_gpu_vendor(wmic_output=mock_output)
        assert vendor == "amd"

    def test_nvidia_detected_via_wmic(self):
        """NVIDIA GPU detected from Win32_VideoController output."""
        import export
        mock_output = "NVIDIA GeForce RTX 4090"
        vendor = export._detect_gpu_vendor(wmic_output=mock_output)
        assert vendor == "nvidia"

    def test_intel_detected_via_wmic(self):
        """Intel GPU detected from Win32_VideoController output."""
        import export
        mock_output = "Intel UHD Graphics 770"
        vendor = export._detect_gpu_vendor(wmic_output=mock_output)
        assert vendor == "intel"

    def test_unknown_gpu_returns_none(self):
        """Unknown GPU returns None."""
        import export
        vendor = export._detect_gpu_vendor(wmic_output="Some Random GPU")
        assert vendor is None


# ── Test: Encoder ordering per vendor ───────────────────────────────────────

class TestEncoderOrdering:
    def test_amd_candidates_order(self):
        """AMD candidates: amf first in vendor order dict."""
        import export
        assert export._VENDOR_ENCODER_ORDER["amd"][0] == "h264_amf"
        # amf must come before nvenc in AMD order
        amd_order = export._VENDOR_ENCODER_ORDER["amd"]
        assert amd_order.index("h264_amf") < amd_order.index("h264_nvenc")

    def test_nvidia_candidates_order(self):
        """NVIDIA candidates: nvenc first."""
        import export
        assert export._VENDOR_ENCODER_ORDER["nvidia"][0] == "h264_nvenc"

    def test_intel_candidates_order(self):
        """Intel candidates: qsv first."""
        import export
        assert export._VENDOR_ENCODER_ORDER["intel"][0] == "h264_qsv"

    def test_unknown_vendor_uses_default_order(self):
        """Unknown vendor falls back to default order with nvenc first."""
        import export
        assert export._DEFAULT_ENCODER_ORDER[0] == "h264_nvenc"


# ── Test: Smoke test caching ────────────────────────────────────────────────

class TestEncoderCaching:
    def test_second_call_returns_cached(self):
        """Second call to _get_best_encoder returns cached result without re-detecting."""
        import export

        with patch.object(export, "_BEST_ENCODER", "h264_amf"):
            with patch("subprocess.run") as mock_run:
                with patch.object(export, "_smoke_test_encoder") as mock_smoke:
                    result = export._get_best_encoder()
                    assert result == "h264_amf"
                    # No subprocess or smoke test calls
                    mock_run.assert_not_called()
                    mock_smoke.assert_not_called()


# ── Test: AMF-specific smoke test ───────────────────────────────────────────

class TestAMFSmokeTest:
    def test_amf_encoder_gets_quality_preset(self):
        """h264_amf should get a quality preset in smoke test, not crash."""
        import export
        # Verify the smoke test function handles amf encoder
        with patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="", stderr="")):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("pathlib.Path.unlink"):
                    # Should not raise
                    result = export._smoke_test_encoder("h264_amf")
                    assert isinstance(result, bool)
