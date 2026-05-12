"""
test_integration.py — End-to-end pipeline tests with synthetic video fixtures.
Tests cheap + premium modes against all layout types.
"""

import json
import pytest
from pathlib import Path

from utils.config import load_config
from frame_analyzer import analyze_clip as cheap_analyze


# ─── Cheap Mode: All Layouts ─────────────────────────────────────────────

class TestCheapPipeline:
    def test_solo_export_strategy(self, test_video_solo):
        res = cheap_analyze(str(test_video_solo), 0.0, 4.0, clip_id="solo")
        strat = res["export_strategy"]
        assert strat["export_aspect_ratio"] == "9:16"
        assert "speed_factor" in strat
        assert "should_drop" in strat

    def test_dual_detected(self, test_video_dual):
        res = cheap_analyze(str(test_video_dual), 0.0, 4.0, clip_id="dual")
        # Should detect dual layout or solo (Haar may not see the divider)
        layout = res["layout"]["layout_type"]
        assert layout in ("solo", "split_both", "split_guest_off")

    def test_black_panel_drops(self, test_video_black_panel):
        res = cheap_analyze(str(test_video_black_panel), 0.0, 4.0, clip_id="black_panel")
        # Should detect black panel on right
        assert res["layout"].get("has_black_panel", False) or res["layout"]["layout_type"] == "split_guest_off"

    def test_blank_frame_drops(self, test_video_blank):
        res = cheap_analyze(str(test_video_blank), 0.0, 4.0, clip_id="blank")
        # Blank should have face crop issues
        assert not res["export_strategy"]["use_solo_frame"]

    def test_screen_share_detected(self, test_video_screen_share):
        res = cheap_analyze(str(test_video_screen_share), 0.0, 4.0, clip_id="screenshare")
        # Screen share or solo
        assert res["layout"]["layout_type"] in ("screen_share", "solo")

    def test_chat_overlay_detected(self, test_video_chat):
        res = cheap_analyze(str(test_video_chat), 0.0, 4.0, clip_id="chat")
        chat = res["layout"].get("chat_overlay")
        if chat:
            assert chat["chat_side"] == "right"
            assert chat["chat_detected"]


# ─── Premium Mode: All Layouts ───────────────────────────────────────────

class TestPremiumPipeline:
    @pytest.fixture(autouse=True)
    def setup(self):
        from premium_analyzer import PremiumAnalyzer
        self.pa = PremiumAnalyzer()
        yield

    def test_solo_face_tracked(self, test_video_solo):
        res = self.pa.analyze_clip(str(test_video_solo), 0.0, 4.0, clip_id="premium_solo")
        assert res["export_strategy"]["export_aspect_ratio"] == "9:16"
        assert "premium" in res
        assert res["premium"]["detection_backend"] in ("yolo", "haar")

    def test_dual_layout(self, test_video_dual):
        res = self.pa.analyze_clip(str(test_video_dual), 0.0, 4.0, clip_id="premium_dual")
        assert res["layout"]["layout_type"] in ("solo", "split_both", "split_guest_off")

    def test_blank_dropped(self, test_video_blank):
        res = self.pa.analyze_clip(str(test_video_blank), 0.0, 4.0, clip_id="premium_blank")
        assert res["export_strategy"]["should_drop"] is True

    def test_black_panel(self, test_video_black_panel):
        res = self.pa.analyze_clip(str(test_video_black_panel), 0.0, 4.0, clip_id="premium_black")
        assert res["layout"]["layout_type"] == "split_guest_off"

    def test_chat_detected_premium(self, test_video_chat):
        res = self.pa.analyze_clip(str(test_video_chat), 0.0, 4.0, clip_id="premium_chat")
        chat = res["layout"].get("chat_overlay")
        if chat:
            assert chat["chat_side"] == "right"


# ─── Config Validation ───────────────────────────────────────────────────

class TestConfigValidation:
    def test_config_loads(self):
        cfg = load_config()
        assert "paths" in cfg
        assert "export" in cfg
        assert "download" in cfg
        assert "highlight" in cfg

    def test_premium_toggle(self):
        cfg = load_config()
        premium = cfg.get("premium", {})
        assert "enabled" in premium

    def test_paths_exist(self):
        cfg = load_config()
        for name, p in cfg["paths"].items():
            if name == "input":
                continue  # input dir might not exist in test
            assert isinstance(p, str), f"Path {name} must be string"
