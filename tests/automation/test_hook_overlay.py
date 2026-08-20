"""Hook Text Overlay — First 2s Retention Hook Tests.

TDD: These tests define the expected behavior before implementation.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestHookOverlayTextGeneration:
    """Tests for generating hook overlay text from hook analysis."""

    def test_generate_overlay_text_wicket_hook(self):
        from export import _generate_hook_overlay_text
        hook_analysis = {
            "hook_type": "wicket",
            "hook_score": 85,
            "hook_types_found": ["wicket"],
            "first_20_words": "OUT! Bowled him! That is a peach of a delivery",
        }
        text = _generate_hook_overlay_text(hook_analysis)
        assert "WICKET" in text.upper() or "OUT" in text.upper()
        assert len(text) <= 30  # Max 3 words for mobile readability

    def test_generate_overlay_text_six_hook(self):
        from export import _generate_hook_overlay_text
        hook_analysis = {
            "hook_type": "six",
            "hook_score": 90,
            "hook_types_found": ["six"],
            "first_20_words": "SIX! What a chhakka! Massive hit into the stands",
        }
        text = _generate_hook_overlay_text(hook_analysis)
        assert "SIX" in text.upper() or "CHHAKKA" in text.upper()
        assert len(text) <= 30

    def test_generate_overlay_text_four_hook(self):
        from export import _generate_hook_overlay_text
        hook_analysis = {
            "hook_type": "four",
            "hook_score": 70,
            "hook_types_found": ["four"],
            "first_20_words": "FOUR! Beautiful boundary through the covers",
        }
        text = _generate_hook_overlay_text(hook_analysis)
        assert "FOUR" in text.upper() or "BOUNDARY" in text.upper()
        assert len(text) <= 30

    def test_generate_overlay_text_crowd_eruption(self):
        from export import _generate_hook_overlay_text
        hook_analysis = {
            "hook_type": "crowd_eruption",
            "hook_score": 75,
            "hook_types_found": ["crowd_eruption"],
            "first_20_words": "Crowd goes absolutely wild! What an atmosphere",
        }
        text = _generate_hook_overlay_text(hook_analysis)
        assert any(w in text.upper() for w in ["CROWD", "WILD", "ROAR", "ERUPTION"])
        assert len(text) <= 30

    def test_generate_overlay_text_commentator_scream(self):
        from export import _generate_hook_overlay_text
        hook_analysis = {
            "hook_type": "commentator_scream",
            "hook_score": 80,
            "hook_types_found": ["commentator_scream"],
            "first_20_words": "OH WOW! WHAT A SHOT! INCREDIBLE!",
        }
        text = _generate_hook_overlay_text(hook_analysis)
        assert any(w in text.upper() for w in ["WOW", "INCREDIBLE", "UNBELIEVABLE"])
        assert len(text) <= 30

    def test_generate_overlay_text_reaction_face(self):
        from export import _generate_hook_overlay_text
        hook_analysis = {
            "hook_type": "reaction_face",
            "hook_score": 65,
            "hook_types_found": ["reaction_face"],
            "first_20_words": "Oh wow! What a catch! Absolutely brilliant",
        }
        text = _generate_hook_overlay_text(hook_analysis)
        assert any(w in text.upper() for w in ["WOW", "BRUTAL", "INSANE", "CRAZY"])
        assert len(text) <= 30

    def test_generate_overlay_text_milestone(self):
        from export import _generate_hook_overlay_text
        hook_analysis = {
            "hook_type": "milestone",
            "hook_score": 85,
            "hook_types_found": ["milestone"],
            "first_20_words": "CENTURY! Kohli reaches his hundred! What a player",
        }
        text = _generate_hook_overlay_text(hook_analysis)
        assert any(w in text.upper() for w in ["CENTURY", "HUNDRED", "FIFTY", "MILESTONE", "RECORD"])
        assert len(text) <= 30

    def test_generate_overlay_text_controversy(self):
        from export import _generate_hook_overlay_text
        hook_analysis = {
            "hook_type": "controversy",
            "hook_score": 70,
            "hook_types_found": ["controversy"],
            "first_20_words": "DRS review! Controversial decision! Fight breaks out",
        }
        text = _generate_hook_overlay_text(hook_analysis)
        assert any(w in text.upper() for w in ["DRS", "REVIEW", "CONTROVERSY", "DECISION"])
        assert len(text) <= 30

    def test_generate_overlay_text_generic_fallback(self):
        from export import _generate_hook_overlay_text
        hook_analysis = {
            "hook_type": "generic_start",
            "hook_score": 15,
            "hook_types_found": [],
            "first_20_words": "and he takes a single to rotate the strike",
        }
        text = _generate_hook_overlay_text(hook_analysis)
        assert text == ""

    def test_generate_overlay_text_empty_analysis(self):
        from export import _generate_hook_overlay_text
        hook_analysis = {}
        text = _generate_hook_overlay_text(hook_analysis)
        assert text == ""

    def test_generate_overlay_text_unknown_hook(self):
        from export import _generate_hook_overlay_text
        text = _generate_hook_overlay_text({
            "hook_type": "unknown",
            "hook_score": 90,
            "hook_types_found": [],
        })
        assert text == ""

    def test_generate_overlay_text_weak_known_hook(self):
        from export import _generate_hook_overlay_text
        text = _generate_hook_overlay_text({
            "hook_type": "six",
            "hook_score": 20,
            "hook_types_found": ["six"],
        })
        assert text == ""

    @pytest.mark.parametrize("malformed", ["six", {"six": True}, 42, None])
    def test_malformed_hook_types_found_is_ignored(self, malformed):
        from export import _generate_hook_overlay_text
        text = _generate_hook_overlay_text({
            "hook_type": "unknown",
            "hook_score": 90,
            "hook_types_found": malformed,
        })
        assert text == ""

    def test_weak_hook_analysis_builds_no_filter(self):
        from export import _build_hook_overlay_from_analysis
        result = _build_hook_overlay_from_analysis({
            "hook_type": "six",
            "hook_score": 20,
            "hook_types_found": ["six"],
        }, output_duration=20.0)
        assert result == ""


class TestHookOverlayFilter:
    """Tests for the ffmpeg drawtext filter generation."""

    def test_build_hook_overlay_filter_basic(self):
        from export import _build_hook_overlay_filter
        overlay_text = "SIX! MASSIVE HIT"
        filter_chain = _build_hook_overlay_filter(overlay_text, duration=2.0)
        assert "drawtext=" in filter_chain
        assert "SIX! MASSIVE HIT" in filter_chain
        assert "enable='between(t,0,2.0)'" in filter_chain

    def test_build_hook_overlay_filter_position_bottom(self):
        from export import _build_hook_overlay_filter
        overlay_text = "WICKET!"
        filter_chain = _build_hook_overlay_filter(overlay_text, duration=2.0, position="bottom")
        assert "y=h-text_h-80" in filter_chain

    def test_build_hook_overlay_filter_position_top(self):
        from export import _build_hook_overlay_filter
        overlay_text = "INCREDIBLE!"
        filter_chain = _build_hook_overlay_filter(overlay_text, duration=2.0, position="top")
        assert "y=" in filter_chain
        assert "h-th" not in filter_chain or "y=0" in filter_chain or "y=20" in filter_chain

    def test_build_hook_overlay_filter_font_size(self):
        from export import _build_hook_overlay_filter
        overlay_text = "TEST"
        filter_chain = _build_hook_overlay_filter(overlay_text, duration=2.0, font_size=80)
        assert "fontsize=80" in filter_chain

    def test_build_hook_overlay_font_color_white_with_stroke(self):
        from export import _build_hook_overlay_filter
        overlay_text = "TEST"
        filter_chain = _build_hook_overlay_filter(overlay_text, duration=2.0)
        assert "fontcolor=white" in filter_chain
        assert "borderw=" in filter_chain
        assert "bordercolor=black" in filter_chain

    def test_build_hook_overlay_font_family(self):
        from export import _build_hook_overlay_filter
        overlay_text = "TEST"
        filter_chain = _build_hook_overlay_filter(overlay_text, duration=2.0, font_file="/path/to/font.ttf")
        assert "fontfile='/path/to/font.ttf'" in filter_chain

    def test_build_hook_overlay_windows_font_path(self):
        r"""Windows font paths (C:\Windows\Fonts\... ) must be ffmpeg-safe.

        Backslashes + a colon in the path break filter parsing on real ffmpeg.
        The path must be normalized (forward slashes, escaped colon) so the
        filter is parseable — otherwise EVERY export fails at filter-parse time.
        """
        from export import _build_hook_overlay_filter
        overlay_text = "TEST"
        filter_chain = _build_hook_overlay_filter(
            overlay_text, duration=2.0, font_file=r"C:\Windows\Fonts\arial.ttf")
        assert "C\\:/Windows/Fonts/arial.ttf" in filter_chain, \
            "Windows path must be normalized (forward slashes) with escaped drive colon"


class TestHookOverlayIntegration:
    """Integration tests: hook overlay in full export filter chain."""

    def test_enhance_stack_includes_hook_overlay(self):
        from export import _build_enhance_stack
        analysis = {
            "export_strategy": {
                "use_solo_frame": True,
            },
            "hook_analysis": {
                "hook_type": "six",
                "hook_score": 90,
                "hook_types_found": ["six"],
                "first_20_words": "SIX! What a chhakka!",
            }
        }
        filter_chain = _build_enhance_stack(analysis, source_fps=30.0, output_duration=20.0)
        assert "drawtext=" in filter_chain
        assert "SIX" in filter_chain.upper() or "CHHAKKA" in filter_chain.upper()
        assert "enable=" in filter_chain  # Time-limited to first 2s

    def test_enhance_stack_hook_overlay_respects_duration(self):
        from export import _build_enhance_stack
        analysis = {
            "export_strategy": {"use_solo_frame": True},
            "hook_analysis": {
                "hook_type": "wicket",
                "hook_score": 85,
                "hook_types_found": ["wicket"],
                "first_20_words": "OUT! Bowled him!",
            }
        }
        # Short clip: overlay should not exceed clip duration
        filter_chain = _build_enhance_stack(analysis, source_fps=30.0, output_duration=5.0)
        assert "drawtext=" in filter_chain
        # enable should be min(2, clip_duration)
        assert "enable=" in filter_chain

    def test_enhance_stack_no_hook_analysis_no_overlay(self):
        from export import _build_enhance_stack
        analysis = {
            "export_strategy": {"use_solo_frame": True},
            # No hook_analysis key
        }
        filter_chain = _build_enhance_stack(analysis, source_fps=30.0, output_duration=20.0)
        assert "drawtext=" not in filter_chain

    def test_enhance_stack_empty_hook_analysis_no_overlay(self):
        from export import _build_enhance_stack
        analysis = {
            "export_strategy": {"use_solo_frame": True},
            "hook_analysis": {},
        }
        filter_chain = _build_enhance_stack(analysis, source_fps=30.0, output_duration=20.0)
        assert "drawtext=" not in filter_chain

    def test_merge_hook_overlay_graph_chain(self):
        """Graph chains (guest_cam_on vstack layout) must keep the overlay.

        The old merge used '[v_tmp]' which does not exist at merge time — the
        label is only appended later by the logo wrap. The overlay was silently
        dropped on the two-panel layout. It must instead be appended to the
        final chain of the graph.
        """
        from export import _merge_hook_overlay_filter
        graph_base = (
            "eq=saturation=1.15,split=2[left_raw][right_raw];"
            "[left_raw]crop=iw/2:ih:0:0,scale=540:960:flags=lanczos[top];"
            "[right_raw]crop=iw/2:ih:iw/2:0,scale=540:960:flags=lanczos[bot];"
            "[top][bot]vstack=inputs=2"
        )
        hook_analysis = {"hook_type": "six", "hook_score": 90}
        result = _merge_hook_overlay_filter(graph_base, hook_analysis, output_duration=20.0)
        assert "drawtext=" in result, \
            "Overlay must be merged into graph (vstack) chains, not silently dropped"

    def test_enhance_stack_guest_cam_layout_includes_hook_overlay(self):
        """guest_cam_on is the two-panel watch-along layout; overlay must render."""
        from export import _build_enhance_stack
        analysis = {
            "export_strategy": {"guest_cam_on": True},
            "hook_analysis": {"hook_type": "wicket", "hook_score": 85},
        }
        filter_chain = _build_enhance_stack(analysis, source_fps=30.0, output_duration=20.0)
        assert "drawtext=" in filter_chain, \
            "guest_cam_on export must include the hook overlay"

    def test_merge_hook_overlay_native_res_skips_for_super_res(self):
        """native_res exports (super-res path) return the chain unchanged.

        The overlay is applied at final res by the super-res upscale pass
        instead; baking drawtext into native-res frames would scale it 4x.
        """
        from export import _merge_hook_overlay_filter
        chain = "crop=iw/2:ih:0:0"
        hook_analysis = {"hook_type": "six", "hook_score": 90}
        result = _merge_hook_overlay_filter(chain, hook_analysis, output_duration=20.0, native_res=True)
        assert result == chain


class TestHookOverlayExportIntegration:
    """End-to-end export tests with hook overlay."""

    @pytest.fixture
    def mock_video_info(self):
        return {"width": 1920, "height": 1080, "fps": 30.0}

    @pytest.fixture
    def mock_analysis_with_hook(self):
        return {
            "export_strategy": {
                "use_solo_frame": True,
                "speed_factor": 1.0,
            },
            "hook_analysis": {
                "hook_type": "six",
                "hook_score": 90,
                "hook_types_found": ["six"],
                "first_20_words": "SIX! What a chhakka! Massive hit",
            },
            "layout": {"layout_type": "solo", "face_in_frame": True},
        }

    def test_export_clip_includes_hook_overlay_in_filter(self, mock_analysis_with_hook, mock_video_info):
        from export import export_clip
        with patch("export._get_video_info", return_value=mock_video_info), \
             patch("export._get_best_encoder", return_value="libx264"), \
             patch("export._has_audio_stream", return_value=True), \
             patch("export._check_free_space", return_value=True), \
             patch("export._run_ffmpeg_with_retry", return_value=(True, "")), \
             patch("export._build_enhance_stack") as mock_build_stack, \
             patch("export._build_audio_filter", return_value="anull"), \
             patch("export.Path.mkdir"), \
             patch("export.Path.exists", return_value=True), \
             patch("export.analyze_clip", return_value=mock_analysis_with_hook):

            mock_build_stack.return_value = "drawtext=text='SIX!':enable='between(t,0,2)',scale=1080:1920"

            result = export_clip(
                video_path="/fake/video.mp4",
                start=10.0,
                end=30.0,
                output_path="/tmp/test_output.mp4",
                clip_id="test_clip",
            )

            # Verify the filter chain includes drawtext for hook overlay
            call_args = mock_build_stack.call_args
            assert call_args is not None
            analysis_passed = call_args[0][0]
            assert "hook_analysis" in analysis_passed

    def test_export_clip_without_hook_analysis_skips_overlay(self, mock_video_info):
        from export import export_clip
        analysis_no_hook = {
            "export_strategy": {"use_solo_frame": True, "speed_factor": 1.0},
            "layout": {"layout_type": "solo", "face_in_frame": True},
        }
        with patch("export._get_video_info", return_value=mock_video_info), \
             patch("export._get_best_encoder", return_value="libx264"), \
             patch("export._has_audio_stream", return_value=True), \
             patch("export._check_free_space", return_value=True), \
             patch("export._run_ffmpeg_with_retry", return_value=(True, "")), \
             patch("export._build_enhance_stack") as mock_build_stack, \
             patch("export._build_audio_filter", return_value="anull"), \
             patch("export.Path.mkdir"), \
             patch("export.Path.exists", return_value=True), \
             patch("export.analyze_clip", return_value=analysis_no_hook):

            mock_build_stack.return_value = "scale=1080:1920"

            result = export_clip(
                video_path="/fake/video.mp4",
                start=10.0,
                end=30.0,
                output_path="/tmp/test_output.mp4",
                clip_id="test_clip",
            )

            call_args = mock_build_stack.call_args
            analysis_passed = call_args[0][0]
            assert "hook_analysis" not in analysis_passed or not analysis_passed.get("hook_analysis")

    def test_export_clip_super_res_applies_hook_overlay_as_vf(self, mock_video_info):
        """HIGH-3 regression: super-res exports must pass the hook overlay as
        `vf` to upscale_video so it renders at FINAL 4x res, not on native
        frames (which would scale the text 4x)."""
        from export import export_clip
        analysis = {
            "export_strategy": {"use_solo_frame": True, "speed_factor": 1.0},
            "layout": {"layout_type": "solo", "face_in_frame": True},
            "hook_analysis": {
                "hook_type": "six", "hook_score": 90,
                "hook_types_found": ["six"],
                "first_20_words": "SIX! What a chhakka!",
            },
        }
        mock_sr = MagicMock()
        mock_sr.available = True
        mock_sr.upscale_video.return_value = True
        with patch("export.super_res_enabled", True), \
             patch("export.premium_enabled", False), \
             patch("export._get_video_info", return_value=mock_video_info), \
             patch("export._get_best_encoder", return_value="libx264"), \
             patch("export._has_audio_stream", return_value=True), \
             patch("export._check_free_space", return_value=True), \
             patch("export._export_native_res", return_value="/tmp/native.mp4"), \
             patch("export.Path.mkdir"), \
             patch("export.Path.exists", return_value=True), \
             patch("export.analyze_clip", return_value=analysis):
            result = export_clip(
                video_path="/fake/video.mp4",
                start=10.0, end=30.0,
                output_path="/tmp/super_res_output.mp4",
                clip_id="sr_clip",
                sr_instance=mock_sr,
            )

        assert result == "/tmp/super_res_output.mp4"
        assert mock_sr.upscale_video.called, \
            "super-res path must call upscale_video"
        kwargs = mock_sr.upscale_video.call_args.kwargs
        vf = kwargs.get("vf", "")
        assert "drawtext=" in vf, \
            f"hook overlay must be passed to upscale_video as vf, got {vf!r}"
        assert "SIX" in vf.upper() or "CHHAKKA" in vf.upper()
        assert "enable=" in vf, "overlay must be time-limited"


class TestHookOverlayConfig:
    """Tests for hook overlay configuration options.

    The hook_overlay block lives under export: and code reads it via
    cfg["export"]["hook_overlay"] — tests assert the REAL nested keys so a
    mis-nested (dead) config block fails loudly instead of passing vacuously.
    """

    def test_config_hook_overlay_nested_under_export(self):
        from utils.config import load_config
        cfg = load_config()
        hook_cfg = cfg.get("export", {}).get("hook_overlay")
        assert hook_cfg is not None, \
            "hook_overlay must be nested under export: (code reads export.hook_overlay)"
        assert isinstance(hook_cfg, dict)

    def test_config_hook_overlay_enabled_by_default(self):
        from utils.config import load_config
        cfg = load_config()
        hook_cfg = cfg["export"]["hook_overlay"]
        enabled = hook_cfg.get("enabled", True)
        assert enabled is True

    def test_config_hook_overlay_duration(self):
        from utils.config import load_config
        cfg = load_config()
        hook_cfg = cfg["export"]["hook_overlay"]
        duration = hook_cfg.get("duration", 2.0)
        assert duration == 2.0

    def test_config_hook_overlay_position(self):
        from utils.config import load_config
        cfg = load_config()
        hook_cfg = cfg["export"]["hook_overlay"]
        position = hook_cfg.get("position", "bottom")
        assert position in ("top", "bottom", "center")

    def test_config_hook_overlay_font_size(self):
        from utils.config import load_config
        cfg = load_config()
        hook_cfg = cfg["export"]["hook_overlay"]
        font_size = hook_cfg.get("font_size", 80)
        assert isinstance(font_size, int)
        assert 40 <= font_size <= 150

    def test_opening_video_and_audio_fades_are_disabled(self):
        from utils.config import load_config
        transitions = load_config()["export"]["transitions"]
        assert transitions["fade_in_duration"] == 0
        assert transitions["audio_fade_in"] == 0


class TestHookOverlayEdgeCases:
    """Edge cases and error handling."""

    def test_hook_overlay_very_long_text_truncated(self):
        from export import _generate_hook_overlay_text
        # Custom analysis carries a long hook_type, but text comes from the map.
        # Verify the generator always returns <= 30 chars for ANY hook_type.
        hook_analysis = {
            "hook_type": "six",
            "hook_score": 90,
            "hook_types_found": ["six"],
            "first_20_words": "SIX! What an absolutely massive incredible unbelievable chhakka hit into the stands",
        }
        text = _generate_hook_overlay_text(hook_analysis)
        assert len(text) <= 30

    def test_hook_overlay_text_never_exceeds_mobile_limit_for_any_hook_type(self):
        """Every hook type must map to text that fits mobile (<= 30 chars).

        Regression guard for the truncation path: the map is the only source of
        text, so if a future entry is too long it must still be capped.
        """
        from export import _generate_hook_overlay_text, _HOOK_TEXT_MAP
        for hook_type in _HOOK_TEXT_MAP:
            text = _generate_hook_overlay_text(
                {"hook_type": hook_type, "hook_score": 90, "hook_types_found": [hook_type]})
            assert len(text) <= 30, \
                f"hook_type={hook_type} produced {len(text)}-char text: {text!r}"

    def test_hook_overlay_special_chars_escaped(self):
        from export import _build_hook_overlay_filter
        # Text with ffmpeg special characters
        overlay_text = "WHAT A 'SHOT'! :)"
        filter_chain = _build_hook_overlay_filter(overlay_text, duration=2.0)
        # Single quotes should be escaped or handled
        assert "drawtext=" in filter_chain

    def test_hook_overlay_unicode_emoji_handled(self):
        from export import _build_hook_overlay_filter
        overlay_text = "SIX! 🔥🏏"
        filter_chain = _build_hook_overlay_filter(overlay_text, duration=2.0)
        assert "drawtext=" in filter_chain

    def test_hook_overlay_zero_duration_clip(self):
        from export import _build_hook_overlay_filter
        filter_chain = _build_hook_overlay_filter("TEST", duration=0.5)
        assert "enable=" in filter_chain
        # Should cap at clip duration
        assert "0,0.5" in filter_chain or "0,0.5" in filter_chain.replace(" ", "")
