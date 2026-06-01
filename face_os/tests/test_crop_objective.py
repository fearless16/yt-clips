import dataclasses

import numpy as np

from face_os.crop_objective import (
    CropObjectiveConfig,
    CropObjectiveState,
    ObjectiveCropPlanner,
    plan_objective_crop,
)
from face_os.types import CropPlan, Landmarks


def _config(**overrides):
    base = CropObjectiveConfig(
        output_size=(1080, 1920),
        target_face_center=(0.5, 0.42),
        target_face_width_ratio=0.25,
        target_headroom_ratio=0.22,
        min_headroom_px=12,
        max_velocity_px=50,
        max_acceleration_px=25,
    )
    return dataclasses.replace(base, **overrides)


def test_objective_crop_preserves_fixed_output_aspect_ratio():
    plan, _ = plan_objective_crop(
        source_shape=(1080, 1920),
        bbox=(850, 320, 160, 220),
        config=_config(),
    )

    assert plan.dst_w == 1080
    assert plan.dst_h == 1920
    assert plan.src_w * plan.dst_h == plan.src_h * plan.dst_w


def test_objective_crop_contains_face_and_protects_headroom():
    landmarks = Landmarks(
        points=np.array(
            [
                [860.0, 92.0],
                [910.0, 104.0],
                [940.0, 150.0],
            ],
            dtype=np.float32,
        )
    )
    bbox = (840, 120, 150, 210)

    plan, report = plan_objective_crop(
        source_shape=(1080, 1920),
        bbox=bbox,
        landmarks=landmarks,
        config=_config(),
    )

    fx, fy, fw, fh = bbox
    assert plan.src_x <= fx
    assert plan.src_y <= fy
    assert plan.src_x + plan.src_w >= fx + fw
    assert plan.src_y + plan.src_h >= fy + fh
    assert plan.src_y <= 92 - 12
    assert report["head_cutoff_penalty"] == 0.0


def test_objective_crop_respects_bounded_velocity_and_acceleration():
    state = CropObjectiveState(
        previous_plan=CropPlan(
            src_x=0,
            src_y=0,
            src_w=603,
            src_h=1072,
            dst_w=1080,
            dst_h=1920,
        ),
        previous_velocity=(10.0, 0.0),
    )
    config = _config(max_velocity_px=35, max_acceleration_px=20)

    plan, report = plan_objective_crop(
        source_shape=(1080, 1920),
        bbox=(380, 200, 120, 180),
        state=state,
        config=config,
    )

    dx = plan.src_x
    dy = plan.src_y
    assert -35 <= dx <= 35
    assert 10 - 20 <= dx <= 10 + 20
    assert -20 <= dy <= 20
    assert plan.src_x <= 380
    assert plan.src_x + plan.src_w >= 500
    assert report["temporal_motion_penalty"] >= 0.0


def test_objective_crop_repeated_calls_are_deterministic():
    kwargs = {
        "source_shape": (1080, 1920),
        "bbox": (760, 250, 180, 240),
        "config": _config(),
        "update_state": False,
    }

    plan_a, report_a = plan_objective_crop(**kwargs)
    plan_b, report_b = plan_objective_crop(**kwargs)

    assert plan_a == plan_b
    assert report_a == report_b


def test_objective_crop_energy_report_has_named_terms():
    planner = ObjectiveCropPlanner(config=_config())

    _, report = planner.plan_crop(
        source_shape=(1080, 1920),
        bbox=(760, 250, 180, 240),
    )

    assert {
        "face_alignment_error",
        "head_cutoff_penalty",
        "temporal_motion_penalty",
        "composition_error",
        "total_energy",
        "weighted_total_energy",
    }.issubset(report.keys())
