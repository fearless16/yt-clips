"""
test_export.py — Premium-grade tests for export.py pipeline.

Covers: _normalize_speed, _parse_fps, _sanitize_strategy,
        _build_enhance_stack, encoder detection,
        export_clip, export_all.
"""
import subprocess
import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from export import (
    _get_best_encoder,
    _smoke_test_encoder,
    export_clip,
    _sanitize_strategy,
    _normalize_speed,
    _parse_fps,
    _build_enhance_stack,
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _safe_analysis(**overrides):
    """Build a valid analysis dict with safe defaults, applying overrides."""
    strategy = {
        "use_solo_frame": True,
        "use_vertical_stack": False,
        "guest_cam_on": False,
        "guest_cam_off": False,
        "is_screen_share": False,
        "has_black_panel": False,
        "black_panel_side": None,
        "active_crop": None,
        "speed_factor": 1.0,
        "apply_lighting_fix": False,
        "lighting_filter": "",
        "skip_silence": False,
        "should_drop": False,
    }
    strategy.update(overrides)
    return {"export_strategy": strategy}


# ─── _normalize_speed ─────────────────────────────────────────────────────────

class TestNormalizeSpeed:
    def test_valid_values(self):
        assert _normalize_speed(1.5) == 1.5

    def test_clamp_low(self):
        assert _normalize_speed(0.1) == pytest.approx(0.25)

    def test_clamp_high(self):
        assert _normalize_speed(10.0) == pytest.approx(4.0)


# ─── _parse_fps ───────────────────────────────────────────────────────────────

class TestParseFps:
    def test_fractional(self):
        assert _parse_fps("30000/1001") == pytest.approx(29.97, abs=0.01)

    def test_division_by_zero(self):
        assert _parse_fps("30/0") == pytest.approx(30.0)


# ─── _sanitize_strategy ──────────────────────────────────────────────────────

class TestSanitizeStrategy:
    def test_new_fields_present(self):
        clean = _sanitize_strategy({})
        assert "guest_cam_on" in clean
        assert "is_screen_share" in clean

    def test_invalid_crop_becomes_none(self):
        raw = {"active_crop": "not_a_dict"}
        clean = _sanitize_strategy(raw)
        assert clean["active_crop"] is None


# ─── _build_enhance_stack ─────────────────────────────────────────────────────

class TestBuildEnhanceStack:
    def test_solo_mode_filter(self):
        analysis = _safe_analysis(use_solo_frame=True)
        fc = _build_enhance_stack(analysis)
        assert "crop" in fc
        assert "1080:1920" in fc # Target size

    def test_guest_cam_on_filter(self):
        analysis = _safe_analysis(guest_cam_on=True)
        fc = _build_enhance_stack(analysis)
        assert "vstack" in fc
        assert "split=2" in fc

    def test_screen_share_filter(self):
        """Screen share stays on a 9:16 Shorts canvas."""
        analysis = _safe_analysis(is_screen_share=True)
        fc = _build_enhance_stack(analysis)
        assert "scale=1080:1920" in fc
        assert "gblur" in fc
        assert "overlay=(W-w)/2:(H-h)/2" in fc
        assert "vstack" not in fc

    def test_lighting_filter_appended(self):
        analysis = _safe_analysis(
            use_solo_frame=True, apply_lighting_fix=True,
            lighting_filter="eq=gamma=1.3:contrast=1.1"
        )
        fc = _build_enhance_stack(analysis)
        assert "eq=gamma=1.3" in fc


# ─── Encoder Detection ───────────────────────────────────────────────────────

class TestEncoderDetection:
    def test_returns_valid_encoder(self):
        enc = _get_best_encoder()
        assert enc in {"h264_nvenc", "h264_videotoolbox", "h264_qsv", "h264_vaapi", "libx264"}


# ─── export_clip ──────────────────────────────────────────────────────────────

class TestExportClip:
    @pytest.fixture
    def test_video(self, tmp_path):
        p = tmp_path / "test.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=320x180:rate=30",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
             "-c:v", "libx264", "-c:a", "aac", "-shortest", str(p)],
            capture_output=True, check=True
        )
        return str(p)

    def test_produces_file(self, test_video, tmp_path):
        out = str(tmp_path / "out.mp4")
        res = export_clip(
            test_video, 0, 1, out,
            clip_id="unit_test", analysis=_safe_analysis()
        )
        assert res is not None
        assert Path(res).exists()

    def test_should_drop_returns_none(self, test_video, tmp_path):
        out = str(tmp_path / "dropped.mp4")
        res = export_clip(
            test_video, 0, 1, out,
            clip_id="drop_test",
            analysis=_safe_analysis(should_drop=True)
        )
        assert res is None


# ─── export_all ───────────────────────────────────────────────────────────────

class TestExportAll:
    def test_pre_filters_dropped_clips(self, monkeypatch, tmp_path):
        import export
        
        # Mock analyze_clip to drop one and keep one
        def mock_analyze(vp, s, e, **kwargs):
            cid = kwargs.get("clip_id")
            if cid == "drop_me":
                return _safe_analysis(should_drop=True)
            return _safe_analysis(should_drop=False)
            
        monkeypatch.setattr(export, "analyze_clip", mock_analyze)
        
        # Mock export_clip to do nothing but return success
        monkeypatch.setattr(export, "export_clip", lambda vp, s, e, out, *a, **kw: out)
        
        # Mock SEO context
        monkeypatch.setattr("trends.get_trending_context", lambda *a, **kw: {})
        
        highlights = {
            "keep_me": {"start": 0, "end": 1},
            "drop_me": {"start": 1, "end": 2}
        }
        
        video = tmp_path / "test.mp4"
        video.touch()
        
        # We need to mock _load_transcript_segments as well
        monkeypatch.setattr(export, "_load_transcript_segments", lambda x: [])

        # Mock generate_seo_for_exported_clip
        monkeypatch.setattr("seo.generate_seo_for_exported_clip", lambda *a, **kw: {})

        # Run export_all
        results = export.export_all(highlights, str(video), generate_seo=True)
        
        # Only "keep_me" should have been exported
        assert len(results) == 1
        assert results[0].name == "keep_me.mp4"
        
        # Verify drop_me was not exported
        assert not any(r.name == "drop_me.mp4" for r in results)
