import export


def _solo_crop(**crop):
    return {
        "export_strategy": {
            "use_solo_frame": True,
            "active_crop": crop,
        }
    }


def test_scrfd_metrics_survive_strategy_sanitization():
    crop = {
        "x": 640, "y": 0, "width": 607, "height": 1080,
        "face_x": 710, "face_y": 160, "face_w": 500, "face_h": 500,
    }

    assert export._sanitize_strategy(_solo_crop(**crop)["export_strategy"])["active_crop"] == crop


def test_solo_face_crop_fills_vertical_canvas_without_blurred_background():
    graph = export._build_enhance_stack(_solo_crop(
        x=640, y=0, width=607, height=1080,
        face_x=710, face_y=160, face_w=500, face_h=500,
    ), use_logo=False)

    assert "crop=606:1076:641:0" in graph
    assert "force_original_aspect_ratio=increase" not in graph
    assert "gblur=" not in graph
    assert "colorchannelmixer=" not in graph
    assert "scale=1080:1920" in graph


def test_tiny_or_missing_face_metrics_degrade_to_safe_vertical_composition():
    tiny = export._build_enhance_stack(_solo_crop(
        x=100, y=0, width=600, height=1080,
        face_x=300, face_y=200, face_w=2, face_h=2,
    ), use_logo=False)
    missing = export._build_enhance_stack(_solo_crop(
        x=100, y=0, width=600, height=1080,
    ), use_logo=False)

    for graph in (tiny, missing):
        assert "scale=1080:1920" in graph
        assert "crop=600:1066:100:7" in graph
        assert "force_original_aspect_ratio" not in graph
        assert "nan" not in graph.casefold()
        assert "inf" not in graph.casefold()


def test_solo_face_native_mode_uses_planned_source_crop_without_background():
    graph = export._build_enhance_stack(_solo_crop(
        x=640, y=0, width=607, height=1080,
        face_x=710, face_y=160, face_w=500, face_h=500,
    ), use_logo=False, native_res=True)

    assert "crop=606:1076:641:0" in graph
    assert "gblur=" not in graph
    assert "scale=1080:1920" not in graph


def test_quality_chain_has_one_mild_sharpen_and_no_aggressive_duplicate(monkeypatch):
    monkeypatch.setitem(export.cfg["export"], "denoise_luma_spatial", 0.8)
    monkeypatch.setitem(export.cfg["export"], "denoise_chroma_spatial", 0.6)
    monkeypatch.setitem(export.cfg["export"], "denoise_luma_temporal", 1.2)
    monkeypatch.setitem(export.cfg["export"], "denoise_chroma_temporal", 0.9)
    monkeypatch.setitem(export.cfg["export"], "sharpen_amount", 0.25)

    graph = export._build_enhance_stack({}, use_logo=False)

    assert graph.count("unsharp=") == 1
    assert "unsharp=3:3:0.25:3:3:0.0" in graph
    assert "hqdn3d=0.8:0.6:1.2:0.9" in graph
    assert "unsharp=5:5:1.0" not in graph
    assert "deband=" not in graph
    assert "saturation=1.15" not in graph


def test_delivery_quality_uses_square_pixels_and_generation_safe_x264_settings():
    graph = export._build_enhance_stack({}, use_logo=False)

    assert "setsar=1" in graph
    assert export.cfg["export"]["crf"] <= 18
    assert export.cfg["export"]["encoder_preset"] == "medium"


def test_face_crop_centers_face_and_uses_upper_composition_line_when_possible():
    graph = export._build_enhance_stack(_solo_crop(
        x=200, y=100, width=400, height=1000,
        face_x=350, face_y=400, face_w=100, face_h=120,
    ), use_logo=False)

    assert "crop=400:710:200:187" in graph
    assert "gblur=" not in graph


def test_non_finite_face_metrics_are_rejected_without_crashing():
    strategy = export._sanitize_strategy({
        "use_solo_frame": True,
        "active_crop": {
            "x": 0, "y": 0, "width": 607, "height": 1080,
            "face_x": float("inf"), "face_y": 1,
            "face_w": 100, "face_h": 100,
        },
    })

    assert "face_x" not in strategy["active_crop"]
