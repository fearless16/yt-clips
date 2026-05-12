import pytest
from pathlib import Path
from frame_analyzer import analyze_clip
from export import export_clip

def test_full_analysis_to_export_flow(test_video, tmp_path):
    """Check that analysis-driven decisions (like lighting) flow correctly into export."""
    # 1. Run analysis
    analysis = analyze_clip(str(test_video), 0.0, 1.0)
    assert "export_strategy" in analysis
    
    # 2. Force safe strategy to prevent false DROPPING on synthetic video
    analysis["export_strategy"]["should_drop"] = False
    analysis["export_strategy"]["is_multi_active_frame"] = False
    analysis["export_strategy"]["apply_lighting_fix"] = True
    analysis["export_strategy"]["lighting_filter"] = "eq=gamma=1.5"
    
    # 3. Export with this analysis
    output_path = tmp_path / "flow_test.mp4"
    import export
    orig_min = export.cfg["export"].get("min_output_bytes")
    export.cfg["export"]["min_output_bytes"] = 100
    try:
        res = export_clip(str(test_video), 0.0, 1.0, str(output_path),
                          clip_id="flow_test", analysis=analysis)
        assert res is not None
        assert Path(res).exists()
        assert Path(res).stat().st_size > 0
    finally:
        export.cfg["export"]["min_output_bytes"] = orig_min


def test_export_with_custom_strategy(test_video, tmp_path):
    """Test exporting with specific manual strategy overrides."""
    output_path = tmp_path / "custom_strategy.mp4"
    import export
    orig_min = export.cfg["export"].get("min_output_bytes")
    export.cfg["export"]["min_output_bytes"] = 100
    
    try:
        custom_analysis = {
            "export_strategy": {
                "use_solo_frame": True,
                "speed_factor": 2.0,
                "apply_lighting_fix": True,
                "lighting_filter": "eq=gamma=2.0",
                "skip_silence": False
            }
        }
        
        res = export_clip(str(test_video), 0.0, 1.0, str(output_path), analysis=custom_analysis)
        assert res is not None
        assert Path(res).exists()
    finally:
        export.cfg["export"]["min_output_bytes"] = orig_min

def test_lighting_correction_logic():
    from frame_analyzer import analyze_lighting
    # Very dark frames
    samples = [{"avg": 10}, {"avg": 15}, {"avg": 12}]
    res = analyze_lighting(samples)
    assert res["needs_correction"] is True
    assert "gamma=1.3" in res["lighting_filter"]
    
    # Very bright frames
    samples = [{"avg": 240}, {"avg": 250}, {"avg": 245}]
    res = analyze_lighting(samples)
    assert res["needs_correction"] is True
    assert "gamma=0.8" in res["lighting_filter"]
