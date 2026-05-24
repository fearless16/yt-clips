"""Integration tests for face_os pipeline with real video.

Validates: detect → track → render → composite → verify.
No mocks. Real video from input/video.mp4.
"""
import os
import sys
import pytest
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


@pytest.fixture(scope='module')
def pipeline():
    """Enrolled pipeline ready for process_frame."""
    from face_os.pipeline import FaceOSPipeline
    p = FaceOSPipeline()
    success = p.enroll()
    if not success or p.tracker is None:
        pytest.skip('Pipeline enrollment failed')
    return p


@pytest.fixture(scope='module')
def processed_frames(pipeline, real_video_path):
    """Process first 30 frames and collect results."""
    cap = cv2.VideoCapture(real_video_path)
    results = []
    for i in range(30):
        ret, frame = cap.read()
        if not ret:
            break
        result = pipeline.process_frame(frame, frame_idx=i)
        results.append({
            'idx': i,
            'result': result,
            'frame': result['frame'] if result and 'frame' in result else None,
            'render_path': result.get('render_path', 'unknown') if result else 'none',
        })
    cap.release()
    return results


# ═══════════════════════════════════════════════════════════════════
# Output Validity
# ═══════════════════════════════════════════════════════════════════

class TestPipelineOutputValidity:
    """All output frames must be valid uint8 BGR images."""

    def test_all_frames_produce_output(self, processed_frames):
        for r in processed_frames:
            assert r['result'] is not None, f"Frame {r['idx']} returned None"
            assert r['frame'] is not None, f"Frame {r['idx']} missing 'frame' key"

    def test_output_dtype_and_shape(self, processed_frames):
        for r in processed_frames:
            f = r['frame']
            if f is None:
                continue
            assert f.dtype == np.uint8, f"Frame {r['idx']} dtype={f.dtype}"
            assert f.ndim == 3 and f.shape[2] == 3, f"Frame {r['idx']} shape={f.shape}"

    def test_no_nan_or_inf(self, processed_frames):
        for r in processed_frames:
            f = r['frame']
            if f is None:
                continue
            ff = f.astype(np.float32)
            assert not np.any(np.isnan(ff)), f"Frame {r['idx']} has NaN"
            assert not np.any(np.isinf(ff)), f"Frame {r['idx']} has Inf"


# ═══════════════════════════════════════════════════════════════════
# Physical Renderer Brightness (KEY regression test)
# ═══════════════════════════════════════════════════════════════════

class TestPhysicalRendererBrightness:
    """Physical renderer frames must NOT be black.

    This is the KEY regression test for the double-attenuation bug.
    Before fix: physical frames had mean ~1.0/255.
    After fix: physical frames should have mean >20/255.
    """

    def test_physical_frames_not_black(self, processed_frames):
        """Physical-path frames must have mean brightness > 20/255."""
        physical_frames = [r for r in processed_frames if r['render_path'] == 'physical']
        if not physical_frames:
            pytest.skip('No physical-path frames in first 30 frames')
        for r in physical_frames:
            mean_val = float(np.mean(r['frame']))
            assert mean_val > 20.0, (
                f"Frame {r['idx']} (physical) mean={mean_val:.1f}/255 — "
                f"still too dark, double-attenuation bug may persist"
            )

    def test_physical_alpha_brightness_ratio(self, processed_frames):
        """Physical frames should be within 3x brightness of alpha frames."""
        physical = [float(np.mean(r['frame'])) for r in processed_frames
                    if r['render_path'] == 'physical' and r['frame'] is not None]
        alpha = [float(np.mean(r['frame'])) for r in processed_frames
                 if r['render_path'] == 'alpha' and r['frame'] is not None]
        if not physical or not alpha:
            pytest.skip('Need both physical and alpha frames')
        phys_mean = np.mean(physical)
        alpha_mean = np.mean(alpha)
        if alpha_mean > 0:
            ratio = phys_mean / alpha_mean
            assert ratio > 0.3, (
                f"Physical/alpha brightness ratio={ratio:.2f} — "
                f"physical frames too dark relative to alpha"
            )

    def test_no_frame_below_threshold(self, processed_frames):
        """No output frame should have mean < 5/255 (completely black)."""
        for r in processed_frames:
            if r['frame'] is None:
                continue
            mean_val = float(np.mean(r['frame']))
            assert mean_val > 5.0, (
                f"Frame {r['idx']} ({r['render_path']}) is nearly black: "
                f"mean={mean_val:.1f}/255"
            )


# ═══════════════════════════════════════════════════════════════════
# Face Detection on Output
# ═══════════════════════════════════════════════════════════════════

class TestFaceDetectionOnOutput:
    """A face detector must find faces in output frames."""

    @pytest.fixture(scope='class')
    def detector(self):
        import dlib
        return dlib.get_frontal_face_detector()

    def test_face_detected_in_physical_frames(self, processed_frames, detector):
        """At least 70% of physical frames should have a detectable face."""
        physical_frames = [r for r in processed_frames
                          if r['render_path'] == 'physical' and r['frame'] is not None]
        if not physical_frames:
            pytest.skip('No physical frames')
        detected = 0
        for r in physical_frames:
            gray = cv2.cvtColor(r['frame'], cv2.COLOR_BGR2GRAY)
            faces = detector(gray, 0)
            if len(faces) > 0:
                detected += 1
        rate = detected / len(physical_frames)
        assert rate >= 0.7, (
            f"Face detection rate on physical frames: {rate:.0%} "
            f"({detected}/{len(physical_frames)}) — below 70% threshold"
        )


# ═══════════════════════════════════════════════════════════════════
# Energy Conservation (Unit-level renderer test)
# ═══════════════════════════════════════════════════════════════════

class TestEnergyConservation:
    """Rendered output energy should be reasonable relative to input."""

    def test_renderer_energy_ratio(self):
        """Direct test of PhysicalRenderer energy on synthetic input."""
        from face_os.physical_renderer import PhysicalRenderer, LightingModel

        renderer = PhysicalRenderer()
        h, w = 128, 128
        # Realistic albedo (skin-tone)
        albedo = np.full((h, w, 3), 0.6, dtype=np.float32)
        # Realistic shading
        shading = np.full((h, w, 1), 0.25, dtype=np.float32)
        # Normal map (frontal face)
        normal_map = np.zeros((h, w, 3), dtype=np.float32)
        normal_map[:, :, 2] = 1.0
        # Lighting pre-scaled by shading mean (as pipeline does)
        lighting = LightingModel(
            ambient=0.15,
            diffuse_intensity=0.85,
        )
        result = renderer.render(
            albedo=albedo, normal_map=normal_map,
            shading=shading, lighting=lighting,
        )
        rendered_mean = float(np.mean(result.rendered))
        # Input energy: albedo * shading = 0.6 * 0.25 = 0.15
        input_energy = float(np.mean(albedo * shading))
        # Rendered should be within 0.3x to 3x of input energy
        assert rendered_mean > input_energy * 0.3, (
            f"Rendered mean={rendered_mean:.4f} < 0.3 * input_energy={input_energy:.4f} — "
            f"energy collapse detected"
        )
        assert rendered_mean < input_energy * 3.0, (
            f"Rendered mean={rendered_mean:.4f} > 3x input_energy={input_energy:.4f} — "
            f"energy explosion detected"
        )

    def test_renderer_not_darker_than_ten_percent(self):
        """Rendered output must not be less than 10% of albedo*shading."""
        from face_os.physical_renderer import PhysicalRenderer, LightingModel

        renderer = PhysicalRenderer()
        for shading_val in [0.1, 0.25, 0.5, 0.8]:
            h, w = 64, 64
            albedo = np.full((h, w, 3), 0.5, dtype=np.float32)
            shading = np.full((h, w, 1), shading_val, dtype=np.float32)
            normal_map = np.zeros((h, w, 3), dtype=np.float32)
            normal_map[:, :, 2] = 1.0
            lighting = LightingModel(
                ambient=0.15,
                diffuse_intensity=0.85,
            )
            result = renderer.render(
                albedo=albedo, normal_map=normal_map,
                shading=shading, lighting=lighting,
            )
            rendered_mean = float(np.mean(result.rendered))
            expected = float(np.mean(albedo)) * shading_val
            assert rendered_mean > expected * 0.1, (
                f"shading={shading_val}: rendered={rendered_mean:.4f} < "
                f"10% of expected={expected:.4f}"
            )


# ═══════════════════════════════════════════════════════════════════
# Process Frame Contract
# ═══════════════════════════════════════════════════════════════════

class TestProcessFrameContract:
    """process_frame returns required keys."""

    def test_returns_required_keys(self, processed_frames):
        if not processed_frames:
            pytest.skip('No processed frames')
        r = processed_frames[0]['result']
        required = {'frame', 'landmarks', 'transform', 'render_path'}
        assert required.issubset(r.keys()), f"Missing: {required - r.keys()}"

    def test_render_path_is_valid(self, processed_frames):
        valid_paths = {'physical', 'alpha', 'enhancement', 'passthrough', 'none'}
        for r in processed_frames:
            if r['result'] is None:
                continue
            path = r['result'].get('render_path', 'unknown')
            assert path in valid_paths, f"Frame {r['idx']} has invalid render_path='{path}'"
