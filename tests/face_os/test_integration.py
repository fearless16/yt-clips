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


# ═══════════════════════════════════════════════════════════════════
# Task 1.5 — Legacy-frame telemetry honesty (D-05 Phase 0)
#
# Fast, unit-level checks that exercise the real telemetry emitter
# (FaceOSPipeline._emit_frame_telemetry) with synthetic args — no real
# video, no enrollment — so they stay in the non-slow subset.
#
# Validates: Requirements 7.1, 7.2, 8.1, 8.2, 8.3, 8.4
# ═══════════════════════════════════════════════════════════════════

LATENT_TELEMETRY_KEYS = {
    "frame_idx",
    "render_path",
    "latent_primary",
    "source_pixel_fraction",
    "latent_confidence",
    "albedo_drift_from_anchor",
    "uncertainty_mean",
    "contract_assertions_passed",
}

FRAME_TELEMETRY_KEYS = {
    "render_path",
    "renderer_mode",
    "fallback_reason",
    "intrinsic_used",
    "geometry_source",
    "resample_count",
    "energy_terms",
    "transform_det",
}


class _FakeIntrinsic:
    """Minimal stand-in for IntrinsicComponents carrying albedo_uncertainty.

    Used to drive the current-frame uncertainty_mean wiring in
    _emit_frame_telemetry without running a full decomposition.
    """

    def __init__(self, uncertainty_value, hw=(16, 16)):
        self.albedo_uncertainty = np.full(hw, uncertainty_value, dtype=np.float32)


@pytest.fixture
def fresh_pipeline():
    """A fresh, un-enrolled pipeline for fast telemetry-emitter unit tests."""
    from face_os.pipeline import FaceOSPipeline
    return FaceOSPipeline()


class TestLatentTelemetryHonesty:
    """Phase 0 legacy frames must report the truth, never stale data."""

    def test_every_latent_record_carries_full_schema(self, fresh_pipeline):
        """Each per-frame latent record exposes exactly the 8-field schema."""
        p = fresh_pipeline
        for i in range(3):
            p._emit_frame_telemetry(
                i, None, None, {"E_temporal": 0.1 * i}, 0, 0,
                render_path="alpha", intrinsic_used=False,
            )
        latent_log = p.get_latent_telemetry()
        assert len(latent_log) == 3
        for rec in latent_log:
            assert set(rec.keys()) == LATENT_TELEMETRY_KEYS, (
                f"latent record keys {set(rec.keys())} != {LATENT_TELEMETRY_KEYS}"
            )

    def test_frame_record_carries_full_schema_and_nested_latent(self, fresh_pipeline):
        """Each frame telemetry record carries the D-08 schema + nested latent."""
        p = fresh_pipeline
        p._emit_frame_telemetry(
            0, None, None, {"E_temporal": 0.2}, 0, 0,
            render_path="physical", intrinsic_used=True,
        )
        frame_log = p.get_frame_telemetry()
        assert len(frame_log) == 1
        rec = frame_log[0]
        assert FRAME_TELEMETRY_KEYS.issubset(rec.keys()), (
            f"frame record missing {FRAME_TELEMETRY_KEYS - set(rec.keys())}"
        )
        # nested latent sub-dict mirrors the dedicated log and is full-schema
        assert "latent" in rec
        assert set(rec["latent"].keys()) == LATENT_TELEMETRY_KEYS
        assert rec["latent"] == p.get_latent_telemetry()[-1]

    def test_legacy_frames_report_latent_not_primary(self, fresh_pipeline):
        """Phase 0 legacy frames: latent_primary False, source_pixel_fraction 1.0."""
        p = fresh_pipeline
        p._emit_frame_telemetry(
            0, None, _FakeIntrinsic(0.3), {"E_temporal": 0.5}, 0, 0,
            render_path="physical", intrinsic_used=True,
        )
        latent = p.get_latent_telemetry()[-1]
        assert latent["latent_primary"] is False
        assert latent["source_pixel_fraction"] == 1.0

    def test_alpha_path_reports_intrinsic_not_used(self, fresh_pipeline):
        """An alpha-path frame (no intrinsic) reports intrinsic_used=False."""
        p = fresh_pipeline
        p._emit_frame_telemetry(
            0, "low_confidence", None, {"E_temporal": 0.0}, 0, 0,
            render_path="alpha", intrinsic_used=False,
        )
        rec = p.get_frame_telemetry()[-1]
        assert rec["render_path"] == "alpha"
        assert rec["intrinsic_used"] is False

    def test_enhancement_path_reports_intrinsic_not_used(self, fresh_pipeline):
        """An enhancement-path frame reports intrinsic_used=False."""
        p = fresh_pipeline
        p._emit_frame_telemetry(
            0, "no_face", None, {}, 0, 0,
            render_path="enhancement", intrinsic_used=False,
        )
        rec = p.get_frame_telemetry()[-1]
        assert rec["render_path"] == "enhancement"
        assert rec["intrinsic_used"] is False

    def test_intrinsic_used_true_when_components_present(self, fresh_pipeline):
        """When intrinsic components are present, intrinsic_used defaults True."""
        p = fresh_pipeline
        # intrinsic_used left as None -> derived from intrinsic_components presence
        p._emit_frame_telemetry(
            0, None, _FakeIntrinsic(0.2), {"E_temporal": 0.4}, 0, 0,
            render_path="physical",
        )
        rec = p.get_frame_telemetry()[-1]
        assert rec["intrinsic_used"] is True

    def test_energy_terms_reflect_current_frame_no_carryover(self, fresh_pipeline):
        """Distinct energy_terms across two emits must not carry over.

        Requirement 8.3: each frame reports energy computed for THAT frame.
        """
        p = fresh_pipeline
        energy_a = {"E_temporal": 0.11, "E_geom": 0.22}
        energy_b = {"E_temporal": 0.77, "E_geom": 0.88}
        p._emit_frame_telemetry(0, None, None, energy_a, 0, 0,
                                render_path="physical", intrinsic_used=True)
        p._emit_frame_telemetry(1, None, None, energy_b, 0, 0,
                                render_path="physical", intrinsic_used=True)
        frame_log = p.get_frame_telemetry()
        assert frame_log[0]["energy_terms"] == energy_a
        assert frame_log[1]["energy_terms"] == energy_b
        # no carryover: frame 1's energy is not frame 0's
        assert frame_log[1]["energy_terms"] != frame_log[0]["energy_terms"]

    def test_uncertainty_mean_is_current_frame_only(self, fresh_pipeline):
        """uncertainty_mean is built from THIS frame's intrinsic_components only.

        Frame 0 has high uncertainty intrinsics; frame 1 has none (defaults to
        1.0). The frame-1 value must not inherit frame-0's mean.
        """
        p = fresh_pipeline
        p._emit_frame_telemetry(0, None, _FakeIntrinsic(0.25), {"E_temporal": 0.0}, 0, 0,
                                render_path="physical", intrinsic_used=True)
        p._emit_frame_telemetry(1, None, None, {"E_temporal": 0.0}, 0, 0,
                                render_path="alpha", intrinsic_used=False)
        latent_log = p.get_latent_telemetry()
        assert latent_log[0]["uncertainty_mean"] == pytest.approx(0.25, abs=1e-6)
        # frame 1 had no intrinsics -> default 1.0, NOT carried over from frame 0
        assert latent_log[1]["uncertainty_mean"] == 1.0

    def test_contract_assertions_passed_is_per_frame(self, fresh_pipeline):
        """contract_assertions_passed reflects the outcome for that frame (8.4)."""
        p = fresh_pipeline
        p._emit_frame_telemetry(0, None, None, {}, 0, 0,
                                render_path="physical", intrinsic_used=True,
                                contract_assertions_passed=True)
        p._emit_frame_telemetry(1, None, None, {}, 0, 0,
                                render_path="physical", intrinsic_used=True,
                                contract_assertions_passed=False)
        latent_log = p.get_latent_telemetry()
        assert latent_log[0]["contract_assertions_passed"] is True
        assert latent_log[1]["contract_assertions_passed"] is False
        # mirrored into the frame record's nested latent sub-dict
        assert p.get_frame_telemetry()[1]["latent"]["contract_assertions_passed"] is False

    def test_latent_and_frame_logs_stay_aligned_per_frame(self, fresh_pipeline):
        """One latent record per frame, aligned 1:1 with the frame telemetry log."""
        p = fresh_pipeline
        for i in range(5):
            p._emit_frame_telemetry(i, None, None, {"E_temporal": float(i)}, 0, 0,
                                    render_path="physical", intrinsic_used=True)
        frame_log = p.get_frame_telemetry()
        latent_log = p.get_latent_telemetry()
        assert len(frame_log) == len(latent_log) == 5
        for i, (f, l) in enumerate(zip(frame_log, latent_log)):
            assert f["frame_idx"] == i
            assert l["frame_idx"] == i


# ═══════════════════════════════════════════════════════════════════
# Task 2.6 — Shadow-mode latent wiring (runtime truth)
#
# These prove the latent machinery actually RUNS at runtime (not just in
# isolated unit tests). In shadow mode the render path stays legacy, so
# latent_primary MUST remain False and source_pixel_fraction MUST remain 1.0
# (honest: the face is still source-derived). What changes: latent_confidence
# becomes REAL (read from the estimator's latent), no longer hardcoded 0.0.
# ═══════════════════════════════════════════════════════════════════


class TestLatentConfidenceWiring:
    """latent_confidence must reflect the owned latent, never a hardcoded 0.0."""

    def test_latent_confidence_reads_pipeline_state(self, fresh_pipeline):
        """_emit_frame_telemetry reports the pipeline's current latent confidence."""
        p = fresh_pipeline
        # Simulate the frame loop having computed a real shadow-mode confidence.
        p._last_latent_confidence = 0.42
        p._emit_frame_telemetry(
            0, None, _FakeIntrinsic(0.3), {"E_temporal": 0.1}, 0, 0,
            render_path="physical", intrinsic_used=True,
        )
        latent = p.get_latent_telemetry()[-1]
        assert latent["latent_confidence"] == pytest.approx(0.42, abs=1e-6)
        # Shadow mode invariants stay honest.
        assert latent["latent_primary"] is False
        assert latent["source_pixel_fraction"] == 1.0

    def test_latent_confidence_defaults_zero_without_state(self, fresh_pipeline):
        """With no latent activity, latent_confidence is 0.0 (un-set default)."""
        p = fresh_pipeline
        p._emit_frame_telemetry(
            0, None, None, {"E_temporal": 0.0}, 0, 0,
            render_path="alpha", intrinsic_used=False,
        )
        assert p.get_latent_telemetry()[-1]["latent_confidence"] == 0.0


class TestGeometrySubsystemWired:
    """A-7: GeometryEstimator must have a real runtime instance on the pipeline."""

    def test_pipeline_instantiates_geometry_estimator(self, fresh_pipeline):
        from face_os.subsystems.geometry_estimator import GeometryEstimator
        assert hasattr(fresh_pipeline, "_geometry_estimator")
        assert isinstance(fresh_pipeline._geometry_estimator, GeometryEstimator)


class TestRenderSourceFlag:
    """D-05 Phase 2: render_source selects legacy vs latent render path.

    Default MUST be 'legacy' so existing behavior is untouched until a caller
    opts in (A/B). The flag is the runtime switch that lets the latent drive
    pixels.
    """

    def test_render_source_defaults_to_legacy(self, fresh_pipeline):
        assert hasattr(fresh_pipeline, "render_source")
        assert fresh_pipeline.render_source == "legacy"

    def test_render_source_is_settable(self, fresh_pipeline):
        fresh_pipeline.render_source = "latent"
        assert fresh_pipeline.render_source == "latent"


class TestLatentPrimaryTelemetry:
    """When the latent actually drives the face, telemetry must say so:
    latent_primary True and source_pixel_fraction < 1.0. The emitter must accept
    these as explicit per-frame values (legacy default stays False / 1.0)."""

    def test_emit_can_flag_latent_primary(self, fresh_pipeline):
        p = fresh_pipeline
        p._last_latent_confidence = 0.26
        p._emit_frame_telemetry(
            0, None, _FakeIntrinsic(0.74), {"E_temporal": 0.1}, 0, 0,
            render_path="latent", intrinsic_used=True,
            latent_primary=True, source_pixel_fraction=0.0,
        )
        latent = p.get_latent_telemetry()[-1]
        assert latent["render_path"] == "latent"
        assert latent["latent_primary"] is True
        assert latent["source_pixel_fraction"] == 0.0
        assert latent["latent_confidence"] == pytest.approx(0.26, abs=1e-6)

    def test_emit_defaults_remain_legacy_honest(self, fresh_pipeline):
        """Without the new args, legacy frames still report False / 1.0."""
        p = fresh_pipeline
        p._emit_frame_telemetry(
            0, None, _FakeIntrinsic(0.3), {"E_temporal": 0.1}, 0, 0,
            render_path="physical", intrinsic_used=True,
        )
        latent = p.get_latent_telemetry()[-1]
        assert latent["latent_primary"] is False
        assert latent["source_pixel_fraction"] == 1.0


class TestSourcePixelFractionLeak:
    """Phase 2B FIX B — source_pixel_fraction must be the SPEC metric, not a
    proxy. Spec (design.md:545, requirements.md:32): the fraction of pixels
    INSIDE the face mask whose rendered color still matches the SOURCE crop
    within tolerance — i.e. how much of the synthesized face is actually a
    surviving paste of the source (target < 0.02 on the latent path).

    The previous implementation emitted ``1 - mean(feathered_mask)`` over the
    WHOLE crop (the background fraction ≈0.80), which says nothing about leak.
    These tests pin the honest definition: a pure function over (rendered,
    source, mask).
    """

    def _mask(self, h=16, w=16):
        m = np.zeros((h, w), dtype=np.float32)
        m[4:12, 4:12] = 1.0  # solid interior block
        return m

    def test_identical_to_source_is_full_leak(self, fresh_pipeline):
        """If the rendered face equals the source inside the mask, every
        interior pixel is a leak → fraction ≈ 1.0."""
        src = np.full((16, 16, 3), 120, np.uint8)
        rendered = src.copy()  # pure paste — total leak
        frac = fresh_pipeline._source_pixel_fraction(rendered, src, self._mask())
        assert frac == pytest.approx(1.0, abs=1e-6)

    def test_fully_different_is_zero_leak(self, fresh_pipeline):
        """If the rendered face differs strongly from source everywhere inside
        the mask, no pixel is traceable to source → fraction ≈ 0.0."""
        src = np.full((16, 16, 3), 40, np.uint8)
        rendered = np.full((16, 16, 3), 210, np.uint8)  # nowhere near source
        frac = fresh_pipeline._source_pixel_fraction(rendered, src, self._mask())
        assert frac == pytest.approx(0.0, abs=1e-6)

    def test_only_counts_mask_interior(self, fresh_pipeline):
        """Differences OUTSIDE the mask must not change the leak — leak is a
        face-interior property. Here interior matches source (leak=1) while the
        background differs; result must still be 1.0."""
        src = np.full((16, 16, 3), 90, np.uint8)
        rendered = np.full((16, 16, 3), 255, np.uint8)  # background differs
        rendered[4:12, 4:12] = 90  # interior == source
        frac = fresh_pipeline._source_pixel_fraction(rendered, src, self._mask())
        assert frac == pytest.approx(1.0, abs=1e-6)

    def test_half_interior_leaks(self, fresh_pipeline):
        """Half the interior matches source, half does not → fraction ≈ 0.5."""
        src = np.full((16, 16, 3), 100, np.uint8)
        rendered = np.full((16, 16, 3), 240, np.uint8)
        rendered[4:8, 4:12] = 100  # top half of interior == source
        frac = fresh_pipeline._source_pixel_fraction(rendered, src, self._mask())
        assert frac == pytest.approx(0.5, abs=1e-6)

    def test_empty_mask_is_safe_zero(self, fresh_pipeline):
        """No interior pixels → no leak measurable → 0.0 (never NaN/raise)."""
        src = np.full((16, 16, 3), 100, np.uint8)
        rendered = src.copy()
        empty = np.zeros((16, 16), np.float32)
        frac = fresh_pipeline._source_pixel_fraction(rendered, src, empty)
        assert frac == 0.0


# Resolve the user-specified test clip; fall back to input/video.mp4 (identical).
def _shadow_test_clip():
    here = os.path.dirname(__file__)
    candidates = [
        os.path.abspath(os.path.join(here, '..', '..', 'clips_test', 'test_clip.mp4')),
        os.path.abspath(os.path.join(here, '..', '..', 'input', 'video.mp4')),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


@pytest.mark.slow
@pytest.mark.timeout(600)
class TestLatentShadowModeOnRealVideo:
    """Runtime truth: the latent must actually populate on real video.

    The mission demands telemetry PROVE the latent runs — green unit tests are
    not enough. Shadow mode keeps the render legacy, but the latent must become
    initialized and report a real, non-zero confidence across the clip.

    The clip is processed ONCE (class-scoped fixture) and shared across all
    assertions — one pipeline run, not four (keeps RAM/CPU bounded). Only a few
    frames are needed: the latent initializes on the first detected face.
    """

    @pytest.fixture(scope="class")
    def shadow_run(self):
        from face_os.pipeline import FaceOSPipeline
        clip = _shadow_test_clip()
        if clip is None:
            pytest.skip('No test clip available (clips_test/test_clip.mp4)')
        p = FaceOSPipeline()
        if not p.enroll() or p.tracker is None:
            pytest.skip('Pipeline enrollment failed')
        cap = cv2.VideoCapture(clip)
        try:
            for i in range(6):
                ret, frame = cap.read()
                if not ret:
                    break
                p.process_frame(frame, frame_idx=i)
        finally:
            cap.release()
        return p

    def test_latent_becomes_initialized(self, shadow_run):
        """update_latent runs every frame -> the owned latent initializes."""
        latent = shadow_run._identity_estimator.latent()
        assert latent.initialized is True, (
            "Latent never initialized — update_latent is not wired into the frame loop"
        )
        assert latent.albedo is not None
        assert latent.albedo.dtype == np.float32
        assert float(latent.albedo.min()) >= 0.0
        assert float(latent.albedo.max()) <= 1.0
        assert not np.any(np.isnan(latent.albedo))

    def test_latent_confidence_is_real_in_telemetry(self, shadow_run):
        """At least one frame reports a real (non-zero) latent_confidence."""
        latent_log = shadow_run.get_latent_telemetry()
        assert len(latent_log) > 0
        confidences = [r["latent_confidence"] for r in latent_log]
        assert max(confidences) > 0.0, (
            "latent_confidence stayed 0.0 across all frames — telemetry is still "
            "hardcoded, latent not driving telemetry"
        )
        # Telemetry confidence must agree with the actual latent state.
        assert max(confidences) == pytest.approx(
            shadow_run._identity_estimator.latent().mean_confidence(), abs=0.2
        )

    def test_shadow_mode_keeps_render_legacy(self, shadow_run):
        """Shadow mode: render stays source-derived (no premature latent flip)."""
        latent_log = shadow_run.get_latent_telemetry()
        for r in latent_log:
            assert r["latent_primary"] is False, f"unexpected latent_primary in {r}"
            assert r["source_pixel_fraction"] == 1.0, f"unexpected source fraction in {r}"

    def test_existing_render_paths_unchanged(self, shadow_run):
        """The legacy render still produces frames (no regression from wiring)."""
        frame_log = shadow_run.get_frame_telemetry()
        assert len(frame_log) > 0
        paths = {r["render_path"] for r in frame_log}
        assert paths.issubset(
            {"physical", "alpha", "enhancement", "passthrough", "none", "error"}
        )


@pytest.mark.slow
@pytest.mark.timeout(600)
class TestLatentRenderModeOnRealVideo:
    """Phase 2 runtime truth: with render_source='latent', the latent must
    actually DRIVE the rendered pixels on real video — not just populate
    telemetry (shadow). This is the proof that the latent render path is live.

    Phase 2A policy (forced latent for A/B): the flag forces the latent path
    whenever the latent is initialized, with NO confidence gate yet, so we can
    measure real A/B quality without a gate hiding the result. The relative-to-
    floor production gate is a follow-up.

    Clip processed ONCE (class-scoped) to keep RAM/CPU bounded.
    """

    @pytest.fixture(scope="class")
    def latent_run(self):
        from face_os.pipeline import FaceOSPipeline
        clip = _shadow_test_clip()
        if clip is None:
            pytest.skip('No test clip available (clips_test/test_clip.mp4)')
        p = FaceOSPipeline()
        if not p.enroll() or p.tracker is None:
            pytest.skip('Pipeline enrollment failed')
        p.render_source = 'latent'  # Phase 2A: force the latent render path
        outputs = []
        cap = cv2.VideoCapture(clip)
        try:
            for i in range(6):
                ret, frame = cap.read()
                if not ret:
                    break
                result = p.process_frame(frame, frame_idx=i)
                outputs.append(result.get('frame') if isinstance(result, dict) else None)
        finally:
            cap.release()
        return p, outputs

    def test_latent_drives_pixels_at_least_once(self, latent_run):
        """At least one frame must report latent_primary=True — the latent
        actually rendered the face, source did not."""
        p, _ = latent_run
        latent_log = p.get_latent_telemetry()
        assert len(latent_log) > 0
        primaries = [r for r in latent_log if r["latent_primary"] is True]
        assert len(primaries) > 0, (
            "render_source='latent' but NO frame reported latent_primary=True — "
            "the latent never drove the render (branch not wired or always fell back)"
        )

    def test_latent_render_path_reported(self, latent_run):
        """Frames the latent drives must report render_path='latent'."""
        p, _ = latent_run
        latent_log = p.get_latent_telemetry()
        latent_frames = [r for r in latent_log if r["latent_primary"]]
        for r in latent_frames:
            assert r["render_path"] == "latent", f"latent frame mislabeled: {r}"

    def test_latent_render_reduces_source_fraction(self, latent_run):
        """A latent-driven face is NOT the source crop: source_pixel_fraction
        must drop below 1.0 (the legacy paste-then-relight value)."""
        p, _ = latent_run
        latent_log = p.get_latent_telemetry()
        primaries = [r for r in latent_log if r["latent_primary"]]
        assert primaries, "no latent-primary frames to check"
        for r in primaries:
            assert r["source_pixel_fraction"] < 1.0, (
                f"latent-primary frame still fully source-derived: {r}"
            )

    def test_latent_render_still_produces_valid_frames(self, latent_run):
        """The latent render path must still produce well-formed output frames
        (no crash, correct dtype/shape) — runtime truth, not just telemetry."""
        _, outputs = latent_run
        produced = [o for o in outputs if o is not None]
        assert len(produced) > 0, "latent render produced no output frames"
        for o in produced:
            assert isinstance(o, np.ndarray)
            assert o.ndim == 3 and o.shape[2] == 3
            assert not np.any(np.isnan(o.astype(np.float32)))
