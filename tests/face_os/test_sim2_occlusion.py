"""Tests for I-06: SIM(2) Occlusion Validation.

Validates predict_with_velocity() under adversarial temporal conditions:
- Occlusion: face lost for N frames, predict recovery
- Dropped detections: skip every Kth frame
- Rapid motion: fast head turn
- Long horizon: 1000+ frame drift test
"""

import numpy as np
import pytest

from face_os.lie_group import SIM2Transform, geodesic_distance_sim2
from face_os.state_evolution import StateEvolution


def _make_transform(theta=0.0, tx=0.0, ty=0.0, scale=1.0) -> SIM2Transform:
    return SIM2Transform(theta=theta, tx=tx, ty=ty, scale=scale)


def _constant_velocity_sequence(n_frames: int, dt: float = 0.01,
                                  dtheta: float = 0.02,
                                  dtx: float = 1.0,
                                  dty: float = 0.5,
                                  dscale: float = 0.001) -> list:
    """Generate a constant-velocity sequence of SIM(2) transforms."""
    transforms = []
    T = SIM2Transform.identity()
    for i in range(n_frames):
        T = _make_transform(
            theta=i * dtheta,
            tx=i * dtx,
            ty=i * dty,
            scale=1.0 + i * dscale,
        )
        transforms.append(T)
    return transforms


class TestOcclusionRecovery:
    """Predict recovery after N-frame occlusion."""

    def test_predict_after_1_frame_occlusion(self):
        """Prediction after 1-frame gap should be close to actual."""
        ev = StateEvolution()
        transforms = _constant_velocity_sequence(10, dtheta=0.02, dtx=2.0)

        # Use frames 0,1 as last known, predict frame 2
        T_prev = transforms[0]
        T_curr = transforms[1]
        T_predicted = ev.predict_with_velocity(T_prev, T_curr)
        T_actual = transforms[2]

        dist = geodesic_distance_sim2(T_predicted, T_actual)
        assert dist < 5.0, f"1-frame prediction error: {dist:.3f}"

    def test_predict_after_5_frame_occlusion(self):
        """Prediction after 5-frame gap should still be reasonable."""
        ev = StateEvolution()
        transforms = _constant_velocity_sequence(20, dtheta=0.02, dtx=2.0)

        # Last known: frames 0,1. Extrapolate 5 steps with constant velocity
        T_prev = transforms[0]
        T_curr = transforms[1]
        velocity = T_curr.log() - T_prev.log()

        # Extrapolate 5 steps from T_curr using same velocity
        T_predicted = T_curr
        for _ in range(5):
            T_step = SIM2Transform.exp(velocity)
            T_predicted = T_predicted.compose(T_step)

        T_actual = transforms[6]
        dist = geodesic_distance_sim2(T_predicted, T_actual)
        # Error grows with occlusion length
        assert dist < 50.0, f"5-frame prediction error: {dist:.3f}"

    def test_predict_after_10_frame_occlusion(self):
        """Prediction after 10-frame gap — error grows but stays bounded."""
        ev = StateEvolution()
        transforms = _constant_velocity_sequence(30, dtheta=0.02, dtx=2.0)

        T_prev = transforms[0]
        T_curr = transforms[1]
        velocity = T_curr.log() - T_prev.log()

        T_predicted = T_curr
        for _ in range(10):
            T_step = SIM2Transform.exp(velocity)
            T_predicted = T_predicted.compose(T_step)

        T_actual = transforms[11]
        dist = geodesic_distance_sim2(T_predicted, T_actual)
        assert dist < 200.0, f"10-frame prediction error: {dist:.3f}"


class TestDroppedDetections:
    """Handle skipped frames gracefully."""

    def test_skip_every_2nd_frame(self):
        """Predict using every 2nd frame — should track motion."""
        ev = StateEvolution()
        transforms = _constant_velocity_sequence(20, dtheta=0.03, dtx=3.0)

        errors = []
        for i in range(0, 16, 2):
            T_prev = transforms[i]
            T_curr = transforms[i + 1]
            T_predicted = ev.predict_with_velocity(T_prev, T_curr)
            T_actual = transforms[i + 2]
            dist = geodesic_distance_sim2(T_predicted, T_actual)
            errors.append(dist)

        avg_error = np.mean(errors)
        assert avg_error < 5.0, f"Skip-2 avg error: {avg_error:.3f}"

    def test_skip_every_5th_frame(self):
        """Predict using every 5th frame — larger gaps."""
        ev = StateEvolution()
        transforms = _constant_velocity_sequence(50, dtheta=0.02, dtx=1.5)

        errors = []
        for i in range(0, 40, 5):
            T_prev = transforms[i]
            T_curr = transforms[i + 1]
            T_predicted = ev.predict_with_velocity(T_prev, T_curr)
            # Skip 4 frames
            for _ in range(4):
                T_predicted = ev.predict_with_velocity(T_curr, T_predicted)
            T_actual = transforms[i + 5]
            dist = geodesic_distance_sim2(T_predicted, T_actual)
            errors.append(dist)

        avg_error = np.mean(errors)
        assert avg_error < 30.0, f"Skip-5 avg error: {avg_error:.3f}"


class TestRapidMotion:
    """Fast head turns and sudden stops."""

    def test_fast_rotation(self):
        """Large rotation per frame — prediction should track."""
        ev = StateEvolution()
        # 10 degrees per frame = 300 deg/s at 30fps
        transforms = _constant_velocity_sequence(10, dtheta=0.175, dtx=0.0)

        T_prev = transforms[0]
        T_curr = transforms[1]
        T_predicted = ev.predict_with_velocity(T_prev, T_curr)
        T_actual = transforms[2]

        dist = geodesic_distance_sim2(T_predicted, T_actual)
        assert dist < 5.0, f"Fast rotation prediction error: {dist:.3f}"

    def test_sudden_stop(self):
        """After constant motion, sudden stop — prediction overshoots."""
        ev = StateEvolution()
        # Moving for 10 frames, then stop
        transforms = _constant_velocity_sequence(10, dtheta=0.05, dtx=5.0)
        # Frame 10: same as frame 9 (sudden stop)
        stopped = transforms[-1]
        transforms.append(stopped)

        T_prev = transforms[8]
        T_curr = transforms[9]
        T_predicted = ev.predict_with_velocity(T_prev, T_curr)

        # Prediction should overshoot (velocity was nonzero)
        predicted_v = T_predicted.log()
        actual_v = stopped.log()
        # The predicted transform should be different from actual
        # (overshoot is expected)
        assert not np.allclose(predicted_v, actual_v, atol=0.01), \
            "Prediction should overshoot on sudden stop"

    def test_acceleration(self):
        """Increasing velocity — prediction underestimates."""
        ev = StateEvolution()
        # Accelerating: each frame has more motion
        transforms = []
        T = SIM2Transform.identity()
        for i in range(10):
            T = _make_transform(theta=i * i * 0.01, tx=i * i * 0.5)
            transforms.append(T)

        T_prev = transforms[7]
        T_curr = transforms[8]
        T_predicted = ev.predict_with_velocity(T_prev, T_curr)
        T_actual = transforms[9]

        # With acceleration, prediction error should be moderate
        dist = geodesic_distance_sim2(T_predicted, T_actual)
        assert dist < 20.0, f"Acceleration prediction error: {dist:.3f}"


class TestLongHorizon:
    """1000+ frame drift test."""

    def test_1000_frame_constant_velocity_drift(self):
        """Over 1000 frames of constant velocity, drift should be bounded."""
        ev = StateEvolution()
        transforms = _constant_velocity_sequence(1005, dtheta=0.001, dtx=0.5, dty=0.2)

        # Simulate sequential prediction: predict next, then use actual as new T_curr
        total_drift = 0.0
        for i in range(2, 1000):
            T_prev = transforms[i - 1]
            T_curr = transforms[i]
            T_predicted = ev.predict_with_velocity(T_prev, T_curr)
            T_actual = transforms[i + 1]
            dist = geodesic_distance_sim2(T_predicted, T_actual)
            total_drift += dist

        avg_drift = total_drift / 998
        assert avg_drift < 1.0, f"1000-frame avg drift: {avg_drift:.3f}"

    def test_500_frame_with_scale_variation(self):
        """Scale changes over 500 frames — drift stays bounded."""
        ev = StateEvolution()
        transforms = _constant_velocity_sequence(
            505, dtheta=0.002, dtx=0.3, dty=0.1, dscale=0.0001,
        )

        total_drift = 0.0
        for i in range(2, 500):
            T_prev = transforms[i - 1]
            T_curr = transforms[i]
            T_predicted = ev.predict_with_velocity(T_prev, T_curr)
            T_actual = transforms[i + 1]
            dist = geodesic_distance_sim2(T_predicted, T_actual)
            total_drift += dist

        avg_drift = total_drift / 498
        assert avg_drift < 1.0, f"500-frame scale drift: {avg_drift:.3f}"


class TestPredictionProperties:
    """Mathematical properties of predict_with_velocity."""

    def test_identity_velocity_gives_identity_prediction(self):
        """If T_prev == T_curr, predicted should be identity (zero velocity)."""
        ev = StateEvolution()
        T = _make_transform(theta=0.5, tx=10.0, ty=5.0, scale=1.2)
        T_predicted = ev.predict_with_velocity(T, T)

        # Zero velocity → T_predicted = T * exp(0) = T
        dist = geodesic_distance_sim2(T_predicted, T)
        assert dist < 1e-6, f"Zero velocity prediction: {dist}"

    def test_constant_velocity_perfect_prediction(self):
        """Constant velocity sequence (left-multiplication) → prediction exact."""
        ev = StateEvolution()
        # Build sequence where T_n = T_{n-1} * exp(v) (left-multiplication)
        v = np.array([0.02, 1.0, 0.5, 0.001])  # [theta, tx, ty, log(scale)]
        T_step = SIM2Transform.exp(v)

        T0 = SIM2Transform.identity()
        T1 = T0.compose(T_step)
        T2_actual = T1.compose(T_step)

        T2_predicted = ev.predict_with_velocity(T0, T1)

        dist = geodesic_distance_sim2(T2_predicted, T2_actual)
        assert dist < 1e-6, f"Constant velocity prediction error: {dist}"

    def test_prediction_preserves_group_structure(self):
        """Predicted transform must be a valid SIM(2) member."""
        ev = StateEvolution()
        T_prev = _make_transform(theta=0.3, tx=5.0, ty=3.0, scale=0.9)
        T_curr = _make_transform(theta=0.5, tx=8.0, ty=4.0, scale=1.1)

        T_predicted = ev.predict_with_velocity(T_prev, T_curr)

        # Check it's a valid SIM2Transform
        assert isinstance(T_predicted, SIM2Transform)
        assert T_predicted.scale > 0, f"Scale must be positive: {T_predicted.scale}"
        # Check det(R) > 0
        M = T_predicted.to_matrix()
        det = np.linalg.det(M[:2, :2])
        assert det > 0, f"Determinant must be positive: {det}"
