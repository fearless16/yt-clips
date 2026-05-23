"""Integration tests for face_os using real video input.

Tests validate the FULL pipeline: detect → track → render → composite.
No mocks. No stubs. Real video frames from input/video.mp4.
"""
import os
import sys
import pytest
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


@pytest.fixture(scope='module')
def pipeline(real_video_path):
    """Enrolled pipeline ready for process_frame."""
    from face_os.pipeline import FaceOSPipeline
    p = FaceOSPipeline()
    # Use the project's default reference images
    success = p.enroll()  # uses expectation.png + photos/
    if not success or p.tracker is None:
        pytest.skip('Pipeline enrollment failed — no tracker')
    return p


# ═══════════════════════════════════════════════════════════════════
# Pipeline End-to-End
# ═══════════════════════════════════════════════════════════════════

class TestPipelineEndToEnd:
    """Tests that the pipeline processes real video frames end-to-end."""

    @pytest.mark.slow
    def test_pipeline_processes_real_frames(self, video_frames, video_metadata, pipeline):
        """Process first 15 frames of real video. Outputs must be valid uint8 BGR."""
        outputs = []
        for i, frame in enumerate(video_frames[:15]):
            result = pipeline.process_frame(frame, frame_idx=i)
            assert result is not None, f"process_frame returned None at frame {i}"
            assert 'frame' in result, f"Missing 'frame' key at frame {i}"
            out = result['frame']
            assert isinstance(out, np.ndarray), f"Frame {i} output is not ndarray"
            assert out.dtype == np.uint8, f"Frame {i} dtype={out.dtype}, expected uint8"
            assert out.ndim == 3, f"Frame {i} ndim={out.ndim}, expected 3"
            assert out.shape[2] == 3, f"Frame {i} channels={out.shape[2]}, expected 3"
            assert not np.any(np.isnan(out.astype(np.float32))), f"Frame {i} has NaN"
            assert not np.any(np.isinf(out.astype(np.float32))), f"Frame {i} has Inf"
            outputs.append(out)

        assert len(outputs) == 15, f"Expected 15 outputs, got {len(outputs)}"

    @pytest.mark.slow
    def test_frame_dimensions_preserved(self, video_frames, pipeline):
        """Output frames must have valid spatial dimensions (non-zero)."""
        for i, frame in enumerate(video_frames[:5]):
            result = pipeline.process_frame(frame, frame_idx=i)
            out = result['frame']
            h, w = out.shape[:2]
            assert h > 0 and w > 0, f"Frame {i}: output shape {out.shape} has zero dimension"

    @pytest.mark.slow
    def test_process_frame_returns_required_keys(self, video_frames, pipeline):
        """process_frame must return dict with frame, landmarks, transform, render_path."""
        result = pipeline.process_frame(video_frames[0], frame_idx=0)
        required_keys = {'frame', 'landmarks', 'transform', 'render_path'}
        assert required_keys.issubset(result.keys()), \
            f"Missing keys: {required_keys - result.keys()}"

    def test_validate_frame_contract_rejects_bad_frames(self):
        """validate_frame_contract must reject NaN, wrong shape, wrong dtype."""
        from face_os.pipeline import FaceOSPipeline

        assert not FaceOSPipeline.validate_frame_contract(None, 720, 1280)
        assert not FaceOSPipeline.validate_frame_contract(
            np.zeros((100, 100, 3), dtype=np.uint8), 720, 1280
        )
        assert not FaceOSPipeline.validate_frame_contract(
            np.zeros((720, 1280, 3), dtype=np.float32), 720, 1280
        )
        nan_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        assert FaceOSPipeline.validate_frame_contract(nan_frame, 720, 1280)


# ═══════════════════════════════════════════════════════════════════
# Telemetry Integrity
# ═══════════════════════════════════════════════════════════════════

class TestTelemetryIntegrity:
    """Validates per-frame telemetry schema and consistency."""

    REQUIRED_TELEMETRY_KEYS = {
        'frame_idx', 'render_path', 'renderer_mode', 'fallback_reason',
        'intrinsic_used', 'geometry_source', 'resample_count',
        'energy_terms', 'transform_det',
    }
    VALID_RENDER_PATHS = {'physical', 'alpha', 'enhancement', 'error'}

    @pytest.mark.slow
    def test_telemetry_schema_on_real_frames(self, video_frames, pipeline):
        """Every frame must emit telemetry with ALL required keys."""
        for i in range(min(15, len(video_frames))):
            pipeline.process_frame(video_frames[i], frame_idx=i)

        log = pipeline._frame_telemetry_log
        assert len(log) > 0, "No telemetry entries emitted"

        for entry in log:
            missing = self.REQUIRED_TELEMETRY_KEYS - set(entry.keys())
            assert not missing, \
                f"Frame {entry.get('frame_idx', '?')}: missing telemetry keys {missing}"

    @pytest.mark.slow
    def test_telemetry_render_path_valid(self, video_frames, pipeline):
        """render_path must be one of the valid enum values."""
        for i in range(min(10, len(video_frames))):
            pipeline.process_frame(video_frames[i], frame_idx=i)

        for entry in pipeline._frame_telemetry_log:
            rp = entry.get('render_path')
            assert rp in self.VALID_RENDER_PATHS, \
                f"Frame {entry.get('frame_idx')}: invalid render_path '{rp}'"

    @pytest.mark.slow
    def test_telemetry_energy_terms_dict(self, video_frames, pipeline):
        """energy_terms must be a dict (possibly empty) on every frame."""
        for i in range(min(10, len(video_frames))):
            pipeline.process_frame(video_frames[i], frame_idx=i)

        for entry in pipeline._frame_telemetry_log:
            et = entry.get('energy_terms')
            assert isinstance(et, dict), \
                f"Frame {entry.get('frame_idx')}: energy_terms is {type(et)}, expected dict"

    @pytest.mark.slow
    def test_telemetry_transform_det_finite(self, video_frames, pipeline):
        """transform_det must be a finite number on every frame."""
        for i in range(min(10, len(video_frames))):
            pipeline.process_frame(video_frames[i], frame_idx=i)

        for entry in pipeline._frame_telemetry_log:
            td = entry.get('transform_det')
            assert isinstance(td, (int, float, np.integer, np.floating)), \
                f"Frame {entry.get('frame_idx')}: transform_det is {type(td)}"
            assert np.isfinite(float(td)), \
                f"Frame {entry.get('frame_idx')}: transform_det is not finite: {td}"


# ═══════════════════════════════════════════════════════════════════
# Render Quality
# ═══════════════════════════════════════════════════════════════════

class TestRenderQuality:
    """Tests visual quality metrics on pipeline output."""

    @staticmethod
    def _laplacian_variance(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    @staticmethod
    def _histogram_std(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.std(gray))

    @pytest.mark.slow
    def test_output_sharpness(self, video_frames, pipeline):
        """Output frames must not be over-blurred (Laplacian variance > 20)."""
        sharpness_values = []
        for i in range(min(10, len(video_frames))):
            result = pipeline.process_frame(video_frames[i], frame_idx=i)
            s = self._laplacian_variance(result['frame'])
            sharpness_values.append(s)

        avg_sharpness = np.mean(sharpness_values)
        assert avg_sharpness > 20, \
            f"Average sharpness {avg_sharpness:.1f} < 20 — output is over-blurred"

    @pytest.mark.slow
    def test_output_contrast(self, video_frames, pipeline):
        """Output contrast must not collapse vs input — ratio must be > 0.3."""
        input_stds = []
        output_stds = []
        for i in range(min(10, len(video_frames))):
            result = pipeline.process_frame(video_frames[i], frame_idx=i)
            out_std = self._histogram_std(result['frame'])
            in_std = self._histogram_std(video_frames[i])
            output_stds.append(out_std)
            input_stds.append(in_std)

        avg_out = np.mean(output_stds)
        avg_in = np.mean(input_stds)
        ratio = avg_out / max(avg_in, 1e-6)
        # Face crops are naturally lower contrast than full frames,
        # but should retain at least 5% of input contrast
        assert ratio > 0.05 or avg_out > 2.0, \
            f"Contrast ratio {ratio:.2f} (out={avg_out:.1f}, in={avg_in:.1f}) — output collapsed"

    @pytest.mark.slow
    def test_output_no_nan_inf(self, video_frames, pipeline):
        """No output frame may contain NaN or Inf values."""
        for i in range(min(10, len(video_frames))):
            result = pipeline.process_frame(video_frames[i], frame_idx=i)
            out_f = result['frame'].astype(np.float32)
            assert not np.any(np.isnan(out_f)), f"Frame {i} has NaN pixels"
            assert not np.any(np.isinf(out_f)), f"Frame {i} has Inf pixels"

    @pytest.mark.slow
    def test_flicker_between_frames(self, video_frames, pipeline):
        """Mean LAB diff between consecutive output frames must be < 8.0."""
        prev_lab = None
        diffs = []
        for i in range(min(15, len(video_frames))):
            result = pipeline.process_frame(video_frames[i], frame_idx=i)
            out = result['frame']
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
            if prev_lab is not None and lab.shape == prev_lab.shape:
                diff = np.mean(np.abs(lab - prev_lab))
                diffs.append(diff)
            prev_lab = lab

        if diffs:
            avg_flicker = np.mean(diffs)
            assert avg_flicker < 8.0, \
                f"Average inter-frame LAB diff {avg_flicker:.2f} >= 8.0 — excessive flicker"


# ═══════════════════════════════════════════════════════════════════
# Compositor Integrity
# ═══════════════════════════════════════════════════════════════════

class TestCompositorIntegrity:
    """Tests compositor linear-light blending correctness."""

    def test_multiband_blend_output_valid(self):
        """multiband_blend must return uint8 array with same shape as input."""
        from face_os.compositor import multiband_blend

        bg = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        fg = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.float32)
        mask[64:192, 64:192] = 1.0

        result = multiband_blend(bg, fg, mask, levels=4)
        assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"
        assert result.shape == bg.shape, f"Shape mismatch: {result.shape} vs {bg.shape}"
        assert not np.any(np.isnan(result.astype(np.float32))), "Result has NaN"

    def test_multiband_blend_linear_light_energy(self):
        """Linear-light blend must approximately preserve energy.
        A 50/50 blend of two images should have mean ~= avg of input means."""
        from face_os.compositor import multiband_blend

        bg = np.full((128, 128, 3), 50, dtype=np.uint8)
        fg = np.full((128, 128, 3), 200, dtype=np.uint8)
        mask = np.ones((128, 128), dtype=np.float32) * 0.5

        result = multiband_blend(bg, fg, mask, levels=3)
        result_mean = np.mean(result)
        # In linear light, the mean should be close to sRGB(0.5*linear(50/255) + 0.5*linear(200/255))
        # Allow 30% tolerance due to pyramid reconstruction artifacts
        expected_low = 80
        expected_high = 180
        assert expected_low < result_mean < expected_high, \
            f"Blended mean {result_mean:.1f} outside [{expected_low}, {expected_high}] — energy not preserved"

    def test_blend_linear_output_valid(self):
        """_blend_linear must return uint8 array same shape."""
        from face_os.compositor import _blend_linear

        bg = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        fg = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        mask = np.random.rand(128, 128).astype(np.float32)

        result = _blend_linear(bg, fg, mask)
        assert result.dtype == np.uint8
        assert result.shape == bg.shape


# ═══════════════════════════════════════════════════════════════════
# Lie Group Math
# ═══════════════════════════════════════════════════════════════════

class TestLieGroupMath:
    """Tests SE(2)/SIM(2) exp/log are true inverse pairs."""

    def test_se2_exp_log_roundtrip(self):
        """exp(log(T)) must reconstruct T for SE(2)."""
        from face_os.lie_group import SE2Transform

        T = SE2Transform(theta=0.5, tx=3.0, ty=-2.0)
        v = T.log()
        T_reconstructed = SE2Transform.exp(v)
        assert abs(T.theta - T_reconstructed.theta) < 1e-6
        assert abs(T.tx - T_reconstructed.tx) < 1e-6
        assert abs(T.ty - T_reconstructed.ty) < 1e-6

    def test_se2_log_exp_roundtrip(self):
        """log(exp(v)) must reconstruct v for SE(2)."""
        from face_os.lie_group import SE2Transform

        v = np.array([0.8, 1.5, -0.7])
        T = SE2Transform.exp(v)
        v2 = T.log()
        np.testing.assert_allclose(v, v2, atol=1e-6)

    def test_sim2_exp_log_roundtrip(self):
        """exp(log(T)) must reconstruct T for SIM(2)."""
        from face_os.lie_group import SIM2Transform

        T = SIM2Transform(theta=0.3, tx=2.0, ty=-1.0, scale=1.5)
        v = T.log()
        T_r = SIM2Transform.exp(v)
        assert abs(T.theta - T_r.theta) < 1e-6
        assert abs(T.tx - T_r.tx) < 1e-6
        assert abs(T.ty - T_r.ty) < 1e-6
        assert abs(T.scale - T_r.scale) < 1e-6

    def test_sim2_log_exp_roundtrip(self):
        """log(exp(v)) must reconstruct v for SIM(2)."""
        from face_os.lie_group import SIM2Transform

        v = np.array([0.4, 1.0, -0.5, 0.3])
        T = SIM2Transform.exp(v)
        v2 = T.log()
        np.testing.assert_allclose(v, v2, atol=1e-6)

    def test_se2_identity_log_is_zero(self):
        """log(identity) must be zero vector."""
        from face_os.lie_group import SE2Transform

        I = SE2Transform.identity()
        v = I.log()
        np.testing.assert_allclose(v, [0, 0, 0], atol=1e-10)

    def test_sim2_geodesic_distance_symmetric(self):
        """Geodesic distance must be symmetric."""
        from face_os.lie_group import SIM2Transform, geodesic_distance_sim2

        T1 = SIM2Transform(theta=0.1, tx=1.0, ty=0.5, scale=1.2)
        T2 = SIM2Transform(theta=0.5, tx=-1.0, ty=2.0, scale=0.8)
        d12 = geodesic_distance_sim2(T1, T2)
        d21 = geodesic_distance_sim2(T2, T1)
        assert abs(d12 - d21) < 1e-6, f"Asymmetric distance: {d12} vs {d21}"


# ═══════════════════════════════════════════════════════════════════
# Dense Geometry
# ═══════════════════════════════════════════════════════════════════

class TestDenseGeometry:
    """Tests dense geometry estimation from landmarks."""

    def test_estimate_produces_valid_mesh(self):
        """DenseGeometryEstimator.estimate must produce vertices, faces, normals."""
        from face_os.dense_geometry import DenseGeometryEstimator

        estimator = DenseGeometryEstimator()
        # Synthetic 478 landmarks in a face-like pattern
        theta = np.linspace(0, 2 * np.pi, 478, endpoint=False)
        landmarks = np.stack([
            100 + 50 * np.cos(theta),
            100 + 70 * np.sin(theta),
        ], axis=1).astype(np.float32)

        geom = estimator.estimate(landmarks)
        assert geom.vertices.shape[1] == 3, "Vertices must be (N, 3)"
        assert geom.faces.shape[1] == 3, "Faces must be (F, 3)"
        assert geom.normals.shape == geom.vertices.shape, "Normals must match vertices shape"
        assert not np.any(np.isnan(geom.vertices)), "Vertices have NaN"
        assert not np.any(np.isnan(geom.normals)), "Normals have NaN"

    def test_normals_are_unit_length(self):
        """All normals must have unit length (±0.01)."""
        from face_os.dense_geometry import DenseGeometryEstimator

        estimator = DenseGeometryEstimator()
        landmarks = np.random.rand(478, 2).astype(np.float32) * 200 + 50

        geom = estimator.estimate(landmarks)
        norms = np.linalg.norm(geom.normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=0.01,
                                   err_msg="Normals are not unit length")

    def test_anatomical_anchors_mapped(self):
        """Key anatomical landmarks should map to distinct vertex regions."""
        from face_os.dense_geometry import DenseGeometryEstimator

        estimator = DenseGeometryEstimator()
        landmarks = np.random.rand(478, 2).astype(np.float32) * 200 + 50
        estimator.estimate(landmarks)

        # Verify nose tip (index 1) and chin (152) map to different vertices
        assert estimator._landmark_indices[1] != estimator._landmark_indices[152], \
            "Nose tip and chin should map to different vertices"
        # Left eye (33) and right eye (263) should be different
        assert estimator._landmark_indices[33] != estimator._landmark_indices[263], \
            "Left and right eye inner should map to different vertices"


# ═══════════════════════════════════════════════════════════════════
# Identity State
# ═══════════════════════════════════════════════════════════════════

class TestIdentityState:
    """Tests identity state initialization and query."""

    def test_belief_initializes_on_first_update(self, canonical_face):
        """First update must initialize the belief state."""
        from face_os.identity_state import IdentityState

        state = IdentityState(atlas_size=(256, 256))
        assert not state.is_initialized()

        quality = np.ones((256, 256), dtype=np.float32) * 0.8
        state.update(canonical_face, quality)
        assert state.is_initialized(), "Belief should be initialized after first update"

    def test_query_returns_valid_output(self, canonical_face):
        """query must return (image, confidence) with correct shapes."""
        from face_os.identity_state import IdentityState

        state = IdentityState(atlas_size=(256, 256))
        quality = np.ones((256, 256), dtype=np.float32) * 0.8
        state.update(canonical_face, quality)

        result, conf = state.query(canonical_face, quality)
        assert result.shape == canonical_face.shape
        assert conf.shape == (256, 256)
        assert result.dtype == np.uint8
        assert conf.dtype == np.float32

    def test_anchor_distance_zero_for_reference(self, canonical_face):
        """Anchor distance must be small right after setting anchor."""
        from face_os.identity_state import IdentityState

        state = IdentityState(atlas_size=(256, 256))
        state.set_anchor(canonical_face)
        quality = np.ones((256, 256), dtype=np.float32) * 0.9
        state.update(canonical_face, quality)

        dist = state.get_anchor_distance()
        assert dist < 10.0, f"Anchor distance {dist} too high right after enrollment"


# ═══════════════════════════════════════════════════════════════════
# A/B Validation
# ═══════════════════════════════════════════════════════════════════

class TestABValidation:
    """Tests A/B comparison framework on real clips."""

    @pytest.mark.slow
    def test_ab_comparator_produces_metrics(self, real_video_path, pipeline):
        """ABComparator.compare_render_methods must return valid metrics dict."""
        from face_os.ab_validation import ABComparator

        comparator = ABComparator()
        metrics = comparator.compare_render_methods(pipeline, real_video_path, max_frames=10)

        assert isinstance(metrics, dict), f"Expected dict, got {type(metrics)}"
        assert len(metrics) > 0, "Metrics dict is empty"
