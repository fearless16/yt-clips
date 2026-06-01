import inspect

import numpy as np

from face_os.temporal_solve import BidirectionalSolver, FrameQuality, TemporalRepairEngine


def _quality(frame_idx, sharpness, detection=1.0, pose=(0.0, 0.0, 0.0)):
    return FrameQuality(
        frame_idx=frame_idx,
        sharpness=sharpness,
        motion_blur=0.0,
        pose=pose,
        detection_confidence=detection,
    )


def test_solve_frame_returns_current_face_bytes_with_better_future_frame():
    solver = BidirectionalSolver(lookback_frames=2, lookahead_frames=5)
    current = np.full((4, 4, 3), 23, dtype=np.uint8)
    future = np.full((4, 4, 3), 231, dtype=np.uint8)

    solver.add_frame(
        0,
        current,
        np.full((4, 4), 0.10, dtype=np.float32),
        _quality(0, sharpness=0.0),
    )
    solver.add_frame(
        2,
        future,
        np.full((4, 4), 0.95, dtype=np.float32),
        _quality(2, sharpness=1.0),
    )
    solver.identify_hq_frames()

    solved_face, solved_confidence = solver.solve_frame(0, (4, 4))

    assert solved_face is current
    assert solved_face.tobytes() == current.tobytes()
    assert solved_face.tobytes() != future.tobytes()
    assert float(np.mean(solved_confidence)) > 0.10


def test_temporal_repair_engine_solve_all_keeps_original_faces_unchanged():
    engine = TemporalRepairEngine(lookback=2, lookahead=5)
    current = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    future = np.full((4, 4, 3), 255, dtype=np.uint8)
    before = current.copy()

    engine.collect_frame(
        0,
        current,
        np.full((4, 4), 0.05, dtype=np.float32),
        sharpness=0.0,
        pose=(0.0, 0.0, 0.0),
        detection_confidence=1.0,
    )
    engine.collect_frame(
        3,
        future,
        np.full((4, 4), 1.0, dtype=np.float32),
        sharpness=1.0,
        pose=(0.0, 0.0, 0.0),
        detection_confidence=1.0,
    )

    solved = engine.solve()

    assert solved[0][0] is current
    assert np.array_equal(solved[0][0], before)
    assert float(np.mean(solved[0][1])) > 0.05


def test_solve_frame_does_not_contain_rgb_repair_path():
    source = inspect.getsource(BidirectionalSolver.solve_frame)
    module_source = inspect.getsource(inspect.getmodule(BidirectionalSolver))

    assert "GaussianBlur" not in module_source
    assert "cv2." not in module_source
    assert "ref_face" not in source
    assert "repaired" not in source
    assert "current_low" not in source
    assert "ref_high" not in source


def test_missing_frame_returns_zero_face_and_zero_confidence():
    solver = BidirectionalSolver()

    solved_face, solved_confidence = solver.solve_frame(42, (3, 5))

    assert solved_face.shape == (3, 5, 3)
    assert solved_face.dtype == np.uint8
    assert np.count_nonzero(solved_face) == 0
    assert solved_confidence.shape == (3, 5)
    assert solved_confidence.dtype == np.float32
    assert np.count_nonzero(solved_confidence) == 0


def test_confidence_propagation_is_deterministic_and_updates_metrics():
    solver = BidirectionalSolver(lookback_frames=4, lookahead_frames=4)
    current = np.full((4, 4, 3), 7, dtype=np.uint8)
    future = np.full((4, 4, 3), 200, dtype=np.uint8)

    solver.add_frame(
        10,
        current,
        np.full((4, 4), 0.20, dtype=np.float32),
        _quality(10, sharpness=0.2, pose=(2.0, 0.0, 0.0)),
    )
    solver.add_frame(
        11,
        future,
        np.full((4, 4), 0.90, dtype=np.float32),
        _quality(11, sharpness=1.0, pose=(3.0, 0.0, 0.0)),
    )

    _, confidence_a = solver.solve_frame(10, (4, 4))
    metrics_a = solver.get_temporal_metrics(10)
    motion_a = solver.get_motion_field(10).copy()
    _, confidence_b = solver.solve_frame(10, (4, 4))
    metrics_b = solver.get_temporal_metrics(10)
    motion_b = solver.get_motion_field(10)

    assert np.array_equal(confidence_a, confidence_b)
    assert metrics_a == metrics_b
    assert np.array_equal(motion_a, motion_b)
    assert metrics_a["rgb_texture_propagated"] == 0.0
    assert 0.0 <= metrics_a["drift_score"] <= 1.0
    assert 0.0 <= metrics_a["continuity_score"] <= 1.0
    assert motion_b.shape == (4, 4, 2)
