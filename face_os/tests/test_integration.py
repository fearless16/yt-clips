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

from face_os.pipeline import FaceOSPipeline  # noqa: E402  (after sys.path setup)


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
        # 'latent' is a first-class render path per design.md Data Models
        # (render_path in {'latent','physical_legacy','alpha','enhancement'});
        # the pipeline emits it at pipeline.py:2100 whenever the latent branch
        # engages (render_source='latent'). The allow-list must include it
        # regardless of which path is the current default.
        valid_paths = {'physical', 'latent', 'alpha', 'enhancement', 'passthrough', 'none'}
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
    "gate_state",
    "hybrid_alpha_mean",
    "coverage_pose",
    "mean_visibility",
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
        """Each per-frame latent record exposes exactly the 10-field schema."""
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

    def test_coverage_pose_zero_without_patch_memory(self, fresh_pipeline):
        """§16.7: a fresh pipeline has no patch_memory yet (created at enroll),
        so coverage_pose reports 0.0 rather than crashing."""
        p = fresh_pipeline
        assert p.patch_memory is None
        p._emit_frame_telemetry(
            0, None, None, {"E_temporal": 0.0}, 0, 0,
            render_path="alpha", intrinsic_used=False,
        )
        assert p.get_latent_telemetry()[-1]["coverage_pose"] == 0.0

    def test_coverage_pose_reflects_live_patch_memory(self, fresh_pipeline):
        """§16.7: coverage_pose in telemetry == patch_memory.coverage_pose().

        Populate two distinct directional bins (F + R20) and assert the emitted
        signal is the real 2/37 union ratio, not a stale or hardcoded value.
        """
        import numpy as np
        from face_os.patch_memory import PatchMemory

        p = fresh_pipeline
        pm = PatchMemory()
        face = np.ones((64, 64, 3), dtype=np.float32) * 0.5
        pm.initialize(face, np.full((64, 64), 0.3, dtype=np.float32))
        pm.update(face, np.full((64, 64), 0.6, dtype=np.float32), pose=(0.0, 0.0, 0.0))
        pm.update(face, np.full((64, 64), 0.8, dtype=np.float32), pose=(20.0, 0.0, 0.0))
        p.patch_memory = pm

        p._emit_frame_telemetry(
            0, None, None, {"E_temporal": 0.0}, 0, 0,
            render_path="latent", intrinsic_used=True,
        )
        rec = p.get_latent_telemetry()[-1]
        assert rec["coverage_pose"] == pytest.approx(2.0 / 37.0, abs=1e-9)
        assert rec["coverage_pose"] == pytest.approx(pm.coverage_pose(), abs=1e-12)

    def test_mean_visibility_defaults_one_without_estimator(self, fresh_pipeline):
        """§16.6: a fresh pipeline has no _identity_estimator (created at enroll),
        so mean_visibility reports 1.0 (no occlusion evidence ⇒ no penalty)."""
        p = fresh_pipeline
        assert p._identity_estimator is None
        p._emit_frame_telemetry(
            0, None, None, {"E_temporal": 0.0}, 0, 0,
            render_path="alpha", intrinsic_used=False,
        )
        assert p.get_latent_telemetry()[-1]["mean_visibility"] == 1.0

    def test_mean_visibility_reflects_live_estimator(self, fresh_pipeline):
        """§16.6: mean_visibility in telemetry == estimator.last_mean_visibility,
        the geometric visibility recorded by the latent's last update."""
        p = fresh_pipeline

        class _Est:
            last_mean_visibility = 0.42

        p._identity_estimator = _Est()
        p._emit_frame_telemetry(
            0, None, None, {"E_temporal": 0.0}, 0, 0,
            render_path="latent", intrinsic_used=True,
        )
        assert p.get_latent_telemetry()[-1]["mean_visibility"] == pytest.approx(0.42, abs=1e-9)

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
        # Per design.md:483 / requirements.md:126: default stays 'legacy' UNTIL
        # the latent path is proven non-regressing on real video (A/B gate).
        # That proof is not yet established, so legacy is the arch-correct default.
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


class TestHybridBlend:
    """Phase 2B per-pixel uncertainty HYBRID (design.md:665, requirements.md
    10.4): WHILE latent confidence is low across the face, blend the latent
    render with the observation BY UNCERTAINTY. Measured runtime truth drove the
    design — on real video latent uncertainty is BROAD (mean ~0.65, ~44% of
    interior > 0.7), and the observation == the source crop, so a raw blend
    toward the observation reintroduces per-pixel source color and the leak
    metric explodes (measured 0.33 ≫ 0.02 guard). Two design choices keep the
    no-source-leak contract intact (both measured):

      1. blend toward LOWPASS(observation) only — smooth illumination/chroma
         crosses, source HIGH FREQUENCY never returns per-pixel (same anti-leak
         mechanism _observation_shading already proves). Drops leak ~20×.
      2. cap blend strength: ``alpha = 1 - uncertainty*blend_max`` with
         blend_max=0.5, so the LATENT retains >=50% authority on EVERY pixel
         (a fully-uncertain pixel still keeps half the synthesized identity).
         Measured worst-case leak 0.0089 < 0.02 (2.2x margin).

    These pin the helpers as pure functions; the wired runtime is proven on real
    video in TestLatentRenderModeOnRealVideo.
    """

    # ── _hybrid_blend_alpha: per-pixel LATENT weight from uncertainty ──────────
    def test_alpha_confident_is_full_latent(self):
        """Zero uncertainty -> alpha 1.0 (pure latent, no observation)."""
        unc = np.zeros((8, 8), np.float32)
        a = FaceOSPipeline._hybrid_blend_alpha(unc, blend_max=0.5)
        assert np.allclose(a, 1.0)

    def test_alpha_fully_uncertain_keeps_latent_majority(self):
        """Max uncertainty -> alpha = 1 - blend_max = 0.5: the latent NEVER
        drops below 50% authority, even where it knows nothing."""
        unc = np.ones((8, 8), np.float32)
        a = FaceOSPipeline._hybrid_blend_alpha(unc, blend_max=0.5)
        assert np.allclose(a, 0.5)

    def test_alpha_monotonic_in_uncertainty(self):
        """alpha must decrease monotonically as uncertainty rises (blend BY
        uncertainty — more uncertain => more observation)."""
        unc = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]], np.float32)
        a = FaceOSPipeline._hybrid_blend_alpha(unc, blend_max=0.5)
        diffs = np.diff(a[0])
        assert np.all(diffs < 0), f"alpha not strictly decreasing: {a[0]}"

    def test_alpha_blend_max_zero_disables_blend(self):
        """blend_max=0 -> alpha 1.0 everywhere (hybrid off, pure latent)."""
        unc = np.random.default_rng(0).random((8, 8)).astype(np.float32)
        a = FaceOSPipeline._hybrid_blend_alpha(unc, blend_max=0.0)
        assert np.allclose(a, 1.0)

    def test_alpha_bounded(self):
        """alpha stays in [1-blend_max, 1] for any uncertainty in [0,1]."""
        unc = np.random.default_rng(1).random((16, 16)).astype(np.float32)
        a = FaceOSPipeline._hybrid_blend_alpha(unc, blend_max=0.5)
        assert a.min() >= 0.5 - 1e-6 and a.max() <= 1.0 + 1e-6

    # ── _hybrid_face: blend latent toward LOWPASS(obs) by uncertainty ──────────
    def _mask(self, h=32, w=32):
        m = np.zeros((h, w), np.float32)
        m[8:24, 8:24] = 1.0
        return m

    def test_hybrid_confident_region_is_pure_latent(self):
        """Where uncertainty is 0, the output must equal the latent face exactly
        (the latent fully owns confident pixels)."""
        latent = np.full((32, 32, 3), 200, np.uint8)
        obs = np.full((32, 32, 3), 80, np.uint8)
        unc = np.zeros((32, 32), np.float32)
        out = FaceOSPipeline._hybrid_face(latent, obs, unc, self._mask(), blend_max=0.5)
        assert np.array_equal(out, latent), "confident region drifted from latent"

    def test_hybrid_uncertain_region_moves_toward_observation(self):
        """Where uncertainty is 1, the output must move toward the (low-freq)
        observation but the latent retains >=50% (alpha=0.5). Flat fields, so
        lowpass(obs)=obs; output ~ halfway."""
        latent = np.full((32, 32, 3), 200, np.uint8)
        obs = np.full((32, 32, 3), 100, np.uint8)
        unc = np.ones((32, 32), np.float32)
        out = FaceOSPipeline._hybrid_face(latent, obs, unc, self._mask(), blend_max=0.5)
        m = self._mask() > 0.5
        interior = out[m].astype(np.float32).mean()
        # alpha=0.5 -> 0.5*200 + 0.5*100 = 150 (latent kept half)
        assert 145 <= interior <= 155, f"uncertain interior mean {interior:.1f} != ~150"
        assert interior > 100 + 1e-3, "latent did not retain its >=50% authority"

    def test_hybrid_does_not_inject_source_high_frequency(self):
        """THE anti-leak guard: blending toward LOWPASS(obs) must NOT carry the
        observation's HIGH FREQUENCY into the output. Flat latent + HF-noisy obs
        at full uncertainty: the output must stay ~flat (source HF stripped by
        the low-pass), proving source detail does not leak per-pixel."""
        h = w = 64
        latent = np.full((h, w, 3), 150, np.uint8)  # flat, no HF
        rng = np.random.default_rng(7)
        noise = rng.normal(0, 60, (h, w, 1)).astype(np.float32)
        obs = np.clip(150 + noise, 0, 255).astype(np.uint8)
        obs = np.repeat(obs, 3, axis=2)
        unc = np.ones((h, w), np.float32)
        mask = np.ones((h, w), np.float32)
        out = FaceOSPipeline._hybrid_face(latent, obs, unc, mask, blend_max=0.5).astype(np.float32)
        lp = cv2.GaussianBlur(out, (0, 0), 3.0)
        hf_energy = float(np.mean(np.abs(out - lp)))
        assert hf_energy < 5.0, (
            f"output carries source high-frequency (HF energy {hf_energy:.2f}/255) "
            f"— low-pass anti-leak failed, source detail leaked"
        )

    def test_hybrid_leaves_outside_mask_as_latent(self):
        """Outside the face mask the output must be the untouched latent (the
        later composite owns the background; the hybrid only touches interior)."""
        latent = np.full((32, 32, 3), 200, np.uint8)
        obs = np.full((32, 32, 3), 50, np.uint8)
        unc = np.ones((32, 32), np.float32)
        out = FaceOSPipeline._hybrid_face(latent, obs, unc, self._mask(), blend_max=0.5)
        outside = self._mask() <= 0.5
        assert np.array_equal(out[outside], latent[outside]), "background drifted"

    def test_hybrid_contract(self):
        """Output preserves the frame contract: same shape, uint8, finite, [0,255]."""
        latent = np.full((32, 32, 3), 120, np.uint8)
        obs = np.full((32, 32, 3), 200, np.uint8)
        unc = np.full((32, 32), 0.5, np.float32)
        out = FaceOSPipeline._hybrid_face(latent, obs, unc, self._mask(), blend_max=0.5)
        assert out.shape == latent.shape
        assert out.dtype == np.uint8
        assert np.all(np.isfinite(out))


class TestObservationShading:
    """Phase 2B FIX — the renderer's brightness comes from the SHADING field
    (physical_renderer.py:374-386 normalizes the LightingModel amplitude away
    and energy-conserves to ``mean(albedo*shading)``). A neutral unit shading
    therefore pins the output to the latent ALBEDO brightness (~0.84), scene-
    independent and flat — the measured 2.1×-too-bright collapse.

    The architecture-faithful shading the renderer needs is ``S = L / A`` where
    ``L`` is the OBSERVED scene luminance and ``A`` is the latent albedo (so
    ``A*S = L`` reconstructs scene exposure). The latent supplies ``A`` (still
    lighting-invariant — no illumination stored); ``L`` is read from the current
    observation. The field is LOW-PASSED so only smooth illumination crosses
    into the render (no source high-frequency / identity leak).

    These pin the helper as a pure function.
    """

    def _albedo(self, val=0.8, h=32, w=32):
        return np.full((h, w, 3), val, np.float32)

    def test_reconstructs_scene_exposure(self, fresh_pipeline):
        """albedo * shading must reconstruct the observed luminance (low-freq):
        a dim scene (luminance ≈ 0.3) under a bright albedo (0.8) yields a
        shading that pulls the render DOWN to the scene, not the albedo."""
        h = w = 32
        albedo = self._albedo(0.8, h, w)
        observed = np.full((h, w, 3), int(0.3 * 255), np.uint8)  # uniform dim scene
        mask = np.ones((h, w), np.float32)
        shading = fresh_pipeline._observation_shading(observed, albedo, mask)
        assert shading.shape[:2] == (h, w)
        # albedo(0.8) * shading must ≈ observed luminance(0.3) => shading ≈ 0.375
        recon = float(np.mean(0.8 * shading))
        assert recon == pytest.approx(0.3, abs=0.05), (
            f"albedo*shading={recon:.3f} must reconstruct scene luminance 0.3"
        )

    def test_carries_spatial_illumination(self, fresh_pipeline):
        """A spatial luminance gradient in the observation must survive in the
        shading (so the render is not flat) — std must be clearly non-zero."""
        h = w = 32
        albedo = self._albedo(0.8, h, w)
        grad = np.tile(np.linspace(40, 220, w, dtype=np.float32), (h, 1))
        observed = np.repeat(grad[:, :, None], 3, axis=2).astype(np.uint8)
        mask = np.ones((h, w), np.float32)
        shading = fresh_pipeline._observation_shading(observed, albedo, mask)
        assert float(np.std(shading)) > 0.05, "shading collapsed flat — no illumination"

    def test_lowpass_rejects_source_high_frequency(self, fresh_pipeline):
        """Per-pixel salt-and-pepper detail in the observation must NOT survive
        in the shading (else the source crop leaks back via shading). The
        residual after a further low-pass must be tiny."""
        h = w = 64
        albedo = self._albedo(0.8, h, w)
        rng = np.random.default_rng(0)
        base = np.full((h, w), 120.0, np.float32)
        noise = rng.normal(0, 50, (h, w)).astype(np.float32)  # high-freq detail
        observed = np.clip(base + noise, 0, 255).astype(np.uint8)
        observed = np.repeat(observed[:, :, None], 3, axis=2)
        mask = np.ones((h, w), np.float32)
        shading = fresh_pipeline._observation_shading(observed, albedo, mask)
        import cv2 as _cv2
        lp = _cv2.GaussianBlur(shading, (0, 0), 3.0)
        hf_energy = float(np.mean(np.abs(shading - lp)))
        assert hf_energy < 0.02, (
            f"shading carries source high-frequency ({hf_energy:.3f}) — leak risk"
        )

    def test_albedo_zero_is_safe(self, fresh_pipeline):
        """Division by a near-zero albedo must not produce NaN/inf (eps floor)."""
        h = w = 16
        albedo = np.zeros((h, w, 3), np.float32)  # degenerate
        observed = np.full((h, w, 3), 100, np.uint8)
        mask = np.ones((h, w), np.float32)
        shading = fresh_pipeline._observation_shading(observed, albedo, mask)
        assert np.all(np.isfinite(shading)), "shading has NaN/inf on zero albedo"

    def test_render_matches_observed_exposure(self, fresh_pipeline):
        """EXPOSURE ANCHOR (measured fix). The render forms 709-luma(albedo * S),
        and ``S`` used the simple channel-mean albedo; the low-pass of ``L / A``
        does NOT preserve the masked mean when the warped ENROLLED albedo's
        structure / 709-weighting differ from the observation. Measured on real
        video, the latent render therefore landed ~1.17-1.20x too bright vs the
        observed face. The anchor rescales ``S`` by one per-frame scalar so the
        masked-mean render luminance equals the masked-mean OBSERVED luminance —
        ground-truth-anchored, not a magic constant, and flicker-safe (the
        observed mean is temporally smooth).
        """
        h = w = 64
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        # Irregular (circular) mask — runtime-faithful: exercises the prefill +
        # mask-boundary low-pass the all-ones fixtures above do not.
        mask = (((xx - 32) ** 2 + (yy - 32) ** 2) < 26 ** 2).astype(np.float32)
        m = mask > 0.5
        # Structured warm enrolled albedo (bright top -> dark bottom).
        vert = 0.55 + 0.40 * (1.0 - yy / h)
        albedo = np.clip(
            np.stack([0.80 * vert, 0.88 * vert, 0.93 * vert], axis=2), 0, 1
        ).astype(np.float32)
        # Observed scene ANTI-correlated (bright bottom) so the raw lowpass(L/A)
        # cancellation breaks and absolute exposure drifts.
        ill = 0.18 + 0.34 * (yy / h)
        skin = np.array([0.42, 0.52, 0.74], np.float32)
        observed = np.clip(ill[:, :, None] * skin, 0, 1).astype(np.float32)
        observed_u8 = (observed * 255).astype(np.uint8)

        def luma709(x):
            return 0.2126 * x[..., 2] + 0.7152 * x[..., 1] + 0.0722 * x[..., 0]

        obs_mean = float(luma709(observed)[m].mean())
        shading = fresh_pipeline._observation_shading(observed_u8, albedo, mask)
        alb709 = luma709(albedo)
        render_mean = float((alb709 * shading)[m].mean())
        rel_err = abs(render_mean - obs_mean) / obs_mean
        assert rel_err < 0.02, (
            f"anchored render masked-mean {render_mean:.4f} must reconstruct the "
            f"observed masked-mean {obs_mean:.4f} (rel err {rel_err:.1%}); the "
            f"exposure anchor is inactive — latent render would be mis-exposed"
        )

        # NON-VACUOUS GUARD: the un-anchored lowpass(L / mean_channel(A)) field
        # (the pre-fix behaviour) must be STRICTLY worse at reproducing the
        # observed exposure, proving the anchor is doing real work.
        naive = luma709(observed) / np.maximum(np.mean(albedo, axis=2), 1e-3)
        naive = cv2.GaussianBlur(naive, (0, 0), max(4.0, h / 12.0))
        naive_err = abs(float((alb709 * naive)[m].mean()) - obs_mean) / obs_mean
        assert naive_err > rel_err, (
            f"anchor did not improve exposure match over raw lowpass(L/A) "
            f"(anchored {rel_err:.1%} vs un-anchored {naive_err:.1%})"
        )


class TestLatentGate:
    """Phase 2B production gate — decides PER FRAME whether the latent is
    trustworthy enough to DRIVE the render, or whether to fall back to legacy.

    RELATIVE-TO-FLOOR by design (measured runtime truth): on real video the
    latent confidence lives in a tiny band [0.2335 seed -> 0.2567 plateau],
    rises ~0.006/frame for a few frames, then sits flat at the Kalman fixed
    point. An ABSOLUTE threshold (e.g. 0.5) would NEVER fire — the latent could
    never engage. So the gate measures confidence RELATIVE to the enrollment
    floor and watches dC/dt for instability:

      - not initialized          -> ('uninitialized')  never engage
      - sharp confidence drop     -> ('confidence_spike') instability, fall back
        (|C_prev - C_t| >= spike_drop; independent of the floor)
      - confidence at/below floor -> ('below_floor') no evidence earned past
        enrollment; fall back
      - otherwise                 -> ('engaged') the latent drives the face

    The PLATEAU (dC/dt = 0, above floor) MUST engage — it is the measured steady
    state. Real video is monotone (never drops, never dips below seed), so the
    REFUSAL paths cannot be exercised honestly on the clip; they are pinned here
    as a pure function with controlled synthetic trajectories. The engage path +
    telemetry wiring is proven on real video in TestLatentRenderModeOnRealVideo.
    """

    def test_uninitialized_never_engages(self):
        engage, state = FaceOSPipeline._evaluate_latent_gate(
            initialized=False, confidence=0.30, confidence_prev=0.30,
            confidence_floor=0.2335,
        )
        assert engage is False
        assert state == "uninitialized"

    def test_plateau_engages(self):
        """THE critical case: the measured real-video steady state (flat at the
        Kalman fixed point, above the floor) MUST drive the render."""
        engage, state = FaceOSPipeline._evaluate_latent_gate(
            initialized=True, confidence=0.2567, confidence_prev=0.2567,
            confidence_floor=0.2335,
        )
        assert engage is True, "plateau (dC/dt=0, above floor) must engage"
        assert state == "engaged"

    def test_rising_confidence_engages(self):
        engage, state = FaceOSPipeline._evaluate_latent_gate(
            initialized=True, confidence=0.2458, confidence_prev=0.2401,
            confidence_floor=0.2335,
        )
        assert engage is True
        assert state == "engaged"

    def test_at_floor_does_not_engage(self):
        """Confidence stuck at the enrollment seed = no real-video evidence
        absorbed; the latent has earned nothing, so fall back."""
        engage, state = FaceOSPipeline._evaluate_latent_gate(
            initialized=True, confidence=0.2335, confidence_prev=0.2335,
            confidence_floor=0.2335,
        )
        assert engage is False
        assert state == "below_floor"

    def test_below_floor_does_not_engage(self):
        """Stable (no spike) but below floor+margin -> below_floor fallback.
        Floor set artificially high to isolate the floor check from the spike
        check."""
        engage, state = FaceOSPipeline._evaluate_latent_gate(
            initialized=True, confidence=0.30, confidence_prev=0.30,
            confidence_floor=0.40,
        )
        assert engage is False
        assert state == "below_floor"

    def test_confidence_spike_falls_back(self):
        """A sharp drop (>= spike_drop) means the latent destabilized this frame
        -> fall back, EVEN IF the post-drop value is still above the floor (so
        the spike check is independent of the floor check)."""
        engage, state = FaceOSPipeline._evaluate_latent_gate(
            initialized=True, confidence=0.30, confidence_prev=0.40,
            confidence_floor=0.2335,
        )
        assert engage is False
        assert state == "confidence_spike"

    def test_spike_takes_precedence_over_below_floor(self):
        """When a drop is BOTH a spike and lands below the floor, the more
        specific instability signal wins the telemetry label."""
        engage, state = FaceOSPipeline._evaluate_latent_gate(
            initialized=True, confidence=0.15, confidence_prev=0.2567,
            confidence_floor=0.2335,
        )
        assert engage is False
        assert state == "confidence_spike"

    def test_small_dip_within_tolerance_still_engages(self):
        """Normal per-frame jitter (real deltas <= ~0.006) is far below
        spike_drop, so a tiny dip while above floor still engages."""
        engage, state = FaceOSPipeline._evaluate_latent_gate(
            initialized=True, confidence=0.2520, confidence_prev=0.2567,
            confidence_floor=0.2335,
        )
        assert engage is True
        assert state == "engaged"


class TestPhysicalGate:
    """Task 5.2 / A-8 / A-9: the PHYSICAL (legacy) render gate as a pure function.

    The H-03 gate at pipeline.py:2043-2052 decided ``physical_possible`` from two
    MAGIC constants — ``E_geom > 0.8`` and ``E_photometric < 0.1`` — compared
    against Z-SCORE-normalized energy terms (EnergyScaler default
    normalization_method='zscore'), NOT raw values. ``E_geom`` is a pose-magnitude
    proxy (``(|yaw|+|pitch|+|roll|)/180`` z-scored): HIGH = extreme pose this frame
    vs its running history. ``E_photometric`` is the intrinsic-decomposition
    QUALITY z-scored (high=good): LOW = degenerate decomposition.

    Task 5.2 makes the gate (a) name+justify those constants as parameters and
    (b) READ the latent's epistemic uncertainty as a first-class input (closing
    A-8, where Kalman uncertainty was computed but unused by rendering). The
    uncertainty scalar is ``1 - latent.mean_confidence()`` = mean of the same
    ``albedo_uncertainty`` field that ``query_uncertainty`` exposes per pixel —
    chosen over calling ``query_uncertainty(geom_state)`` because it needs no
    geometry/warp and cannot crash when geometry is absent.

    The new ``latent_uncertainty_max`` veto is INITIALIZED-GUARDED by the caller
    (passes None pre-enrollment, where query_uncertainty would be all-ones), and
    its threshold (0.95) sits ABOVE the measured operating point (real-video mean
    U: seed ~0.77, plateau ~0.74, even a spike ~0.8), so it is inert in normal
    operation and fires only on near-total identity collapse (U->1). Energy vetoes
    keep precedence so existing telemetry reason labels are unchanged.
    """

    def test_physical_gate_allows_nominal(self):
        """Stable pose (low E_geom z), good decomposition (high E_photometric z),
        no latent signal -> physical render allowed, no fallback reason."""
        allow, reason = FaceOSPipeline._evaluate_physical_gate(
            energy_terms={'E_geom': 0.1, 'E_photometric': 1.5},
        )
        assert allow is True
        assert reason is None

    def test_physical_gate_vetoes_extreme_geom(self):
        """E_geom z-score above geom_extreme_z (extreme pose) -> veto with the
        EXACT legacy reason string."""
        allow, reason = FaceOSPipeline._evaluate_physical_gate(
            energy_terms={'E_geom': 0.9, 'E_photometric': 1.5},
        )
        assert allow is False
        assert reason == 'energy_geom_extreme'

    def test_physical_gate_vetoes_low_photometric(self):
        """E_photometric z-score below photometric_low_z (degenerate decomp) ->
        veto with the EXACT legacy reason string."""
        allow, reason = FaceOSPipeline._evaluate_physical_gate(
            energy_terms={'E_geom': 0.1, 'E_photometric': 0.05},
        )
        assert allow is False
        assert reason == 'energy_photometric_low'

    def test_physical_gate_geom_takes_precedence(self):
        """When BOTH energy vetoes fire, geom wins (preserves the original
        ``elif`` precedence so telemetry labels stay stable)."""
        allow, reason = FaceOSPipeline._evaluate_physical_gate(
            energy_terms={'E_geom': 0.9, 'E_photometric': 0.05},
        )
        assert allow is False
        assert reason == 'energy_geom_extreme'

    def test_physical_gate_empty_energy_allows(self):
        """REGRESSION LOCK: the original gate ran its checks only under
        ``if physical_possible and energy_terms`` — empty energy_terms => NO
        energy veto. With no latent signal that means allow."""
        allow, reason = FaceOSPipeline._evaluate_physical_gate(energy_terms={})
        assert allow is True
        assert reason is None

    def test_physical_gate_missing_photometric_key_vetoes(self):
        """REGRESSION LOCK of a subtle original behavior: with a NON-empty
        energy_terms that lacks 'E_photometric', the original did
        ``E_photometric = energy_terms.get('E_photometric', 0.0)`` then
        ``0.0 < 0.1`` -> VETO. The refactor must preserve this exactly."""
        allow, reason = FaceOSPipeline._evaluate_physical_gate(
            energy_terms={'E_geom': 0.1},
        )
        assert allow is False
        assert reason == 'energy_photometric_low'

    def test_physical_gate_uninitialized_uncertainty_inert(self):
        """INITIALIZED-GUARD: pre-enrollment the caller passes
        latent_uncertainty_mean=None (query_uncertainty would be all-ones). The
        new veto must NOT fire, so legacy-only runs are byte-for-byte unchanged."""
        allow, reason = FaceOSPipeline._evaluate_physical_gate(
            energy_terms={'E_geom': 0.1, 'E_photometric': 1.5},
            latent_uncertainty_mean=None,
        )
        assert allow is True
        assert reason is None

    def test_physical_gate_vetoes_extreme_latent_uncertainty(self):
        """ANTI-DECORATIVE: the read input must actually CONTROL the gate. With
        nominal energy (which alone would allow), a near-total identity collapse
        (U=0.99 >= latent_uncertainty_max) must flip the decision to fallback."""
        allow, reason = FaceOSPipeline._evaluate_physical_gate(
            energy_terms={'E_geom': 0.1, 'E_photometric': 1.5},
            latent_uncertainty_mean=0.99,
        )
        assert allow is False
        assert reason == 'latent_uncertainty_high'

    def test_physical_gate_real_video_uncertainty_inert(self):
        """NON-REGRESSION at the MEASURED operating point: real-video latent
        mean-U tops out ~0.8 (seed 1-0.2335=0.766, plateau 1-0.2567=0.743). At
        the worst measured U the new veto must stay silent so the physical path
        is unchanged on the real clip."""
        allow, reason = FaceOSPipeline._evaluate_physical_gate(
            energy_terms={'E_geom': 0.1, 'E_photometric': 1.5},
            latent_uncertainty_mean=0.77,
        )
        assert allow is True
        assert reason is None

    def test_physical_gate_energy_veto_precedes_uncertainty(self):
        """Energy vetoes keep precedence over the uncertainty veto so the
        existing reason vocabulary is preserved when both would fire."""
        allow, reason = FaceOSPipeline._evaluate_physical_gate(
            energy_terms={'E_geom': 0.9, 'E_photometric': 1.5},
            latent_uncertainty_mean=0.99,
        )
        assert allow is False
        assert reason == 'energy_geom_extreme'


# ═══════════════════════════════════════════════════════════════════
# Task 3.11 — FAST latent-path drive + subsystem-boundary tests
#
# The full pipeline frame loop needs real MediaPipe detection, so it can only
# run @slow on real video. These tests instead DIRECT-DRIVE `_render_with_latent`
# on synthetic landmarks so the latent render path engages in the FAST subset.
# `_render_with_latent` returns None (silent legacy fallback) on ANY guard miss
# or swallowed exception, so `result is not None` is the LOAD-BEARING guard
# against the documented "green test hides a broken-runtime fallback" trap that
# bit this path once before (a GeometryState NameError stayed green at 228).
# ═══════════════════════════════════════════════════════════════════

# MediaPipe-style 5 alignment anchors and their canonical (256-atlas) positions
# (face_os/canonical_map.py:30-38). Placing the synthetic anchors ON the
# canonical positions makes compute_alignment yield a stable ~identity transform
# and a centered face mask — no real detection required.
_ANCHOR_IDX = [1, 33, 263, 61, 291]
_ANCHOR_XY = [(128.0, 145.0), (150.0, 105.0), (106.0, 105.0),
              (142.0, 185.0), (114.0, 185.0)]


def _synthetic_face_landmarks():
    """478-point Landmarks whose 5 alignment anchors sit on the canonical
    positions (stable transform, centered mask; no MediaPipe)."""
    from face_os.types import Landmarks
    pts = np.full((478, 2), 128.0, dtype=np.float32)
    for idx, (x, y) in zip(_ANCHOR_IDX, _ANCHOR_XY):
        pts[idx] = (x, y)
    return Landmarks(points=pts)


def _init_latent_estimator(atlas_size=(256, 256), seed=0):
    """An IdentityEstimator with an INITIALIZED latent, seeded from a random
    canonical face via update_latent (no enrollment / real video). Built on a
    bare mock state so it NEVER reaches a real IdentityState's internals."""
    from face_os.subsystems.identity_estimator import IdentityEstimator
    from face_os.types import GeometryState

    class _MockState:  # IdentityEstimator only needs an object handle here
        pass

    est = IdentityEstimator(_MockState(), atlas_size=atlas_size)
    geom = GeometryState(
        pose=(0.0, 0.0, 0.0),
        canonical_transform=np.eye(3, dtype=np.float32),
        inverse_transform=np.eye(3, dtype=np.float32),
    )
    rng = np.random.RandomState(seed)
    canonical_face = rng.randint(40, 220, (*atlas_size, 3)).astype(np.uint8)
    quality = np.ones(atlas_size, dtype=np.float32) * 0.8
    est.update_latent(canonical_face, geom, quality)
    assert est.latent().initialized, "latent failed to initialize in test setup"
    return est


def _make_latent_pipeline(crop_size=256):
    """A pipeline wired to drive _render_with_latent on synthetic input:
    real FaceRenderer + an initialized IdentityEstimator, render_source='latent'.
    The source crop is deliberately DIFFERENT from the latent albedo so a real
    latent render yields a low (non-coincidental) source-pixel-fraction.
    """
    from face_os.physical_renderer import PhysicalRenderer
    from face_os.subsystems.renderer import FaceRenderer
    from face_os.types import CropPlan

    p = FaceOSPipeline()
    p.render_source = 'latent'
    p._face_renderer = FaceRenderer(PhysicalRenderer())
    p._identity_estimator = _init_latent_estimator(atlas_size=(256, 256))

    landmarks = _synthetic_face_landmarks()
    crop_plan = CropPlan(src_x=0, src_y=0, src_w=crop_size, src_h=crop_size,
                         dst_w=crop_size, dst_h=crop_size)
    yy, xx = np.mgrid[0:crop_size, 0:crop_size].astype(np.float32)
    grad = ((xx + yy) / (2 * crop_size) * 180 + 40).astype(np.uint8)
    cropped = np.stack([grad, np.roll(grad, 20, 0), np.roll(grad, 40, 1)], axis=-1)
    return p, cropped, landmarks, crop_plan


class _BoundaryProbe:
    """Stands in for pipeline.identity_state; RAISES if the latent render path
    dereferences a forbidden legacy attr (proves subsystem boundary, Req 4.1/7.6)."""
    FORBIDDEN = ("_anchor_albedo", "_intrinsic_decomposer", "_gate")

    def __init__(self):
        self.touched = []

    def __getattr__(self, name):
        # Only invoked for attrs not in __dict__ (so `touched` never recurses).
        if name in _BoundaryProbe.FORBIDDEN:
            self.touched.append(name)
            raise AssertionError(
                f"latent render path accessed forbidden identity_state.{name}"
            )
        return None


class TestLatentDrivesRender:
    """FAST: the latent render path actually drives synthetic-frame pixels and
    reports latent_primary telemetry — without real video."""

    def test_render_with_latent_engages_not_fallback(self):
        """LOAD-BEARING: _render_with_latent must RETURN A FRAME, not silently
        fall back to None. A None here means a guard or swallowed exception
        killed the latent path (the green-test-hides-broken-runtime trap)."""
        p, cropped, landmarks, crop_plan = _make_latent_pipeline()
        out = p._render_with_latent(cropped, landmarks, crop_plan, frame_idx=0)
        assert out is not None, (
            "latent path silently fell back to legacy (returned None) — the "
            "latent did not drive the render"
        )
        assert out.shape == cropped.shape
        assert out.dtype == np.uint8

    def test_render_with_latent_reports_low_source_fraction(self):
        """Runtime truth: the composited face is NOT the source crop. The leak
        metric (_last_source_pixel_fraction) is measured by the real path and
        must drop well below the legacy default of 1.0."""
        p, cropped, landmarks, crop_plan = _make_latent_pipeline()
        out = p._render_with_latent(cropped, landmarks, crop_plan, frame_idx=0)
        assert out is not None
        frac = p._last_source_pixel_fraction
        assert 0.0 <= frac < 1.0, frac  # measured, not the 1.0 default
        assert frac < 0.5, f"source_pixel_fraction {frac} too high — latent not driving"

    def test_latent_primary_telemetry_after_drive(self):
        """Wiring the real render's measured fraction through _emit_frame_telemetry
        (exactly as the pipeline's latent branch does, pipeline.py:2100-2106)
        yields a latent record with latent_primary=True and the measured fraction."""
        p, cropped, landmarks, crop_plan = _make_latent_pipeline()
        out = p._render_with_latent(cropped, landmarks, crop_plan, frame_idx=0)
        assert out is not None
        p._emit_frame_telemetry(
            0, None, None, {"E_temporal": 0.0}, 0, 0,
            render_path="latent", intrinsic_used=True,
            latent_primary=True,
            source_pixel_fraction=float(p._last_source_pixel_fraction),
        )
        rec = p.get_latent_telemetry()[-1]
        assert rec["latent_primary"] is True
        assert rec["render_path"] == "latent"
        assert rec["source_pixel_fraction"] == pytest.approx(p._last_source_pixel_fraction)


class TestSubsystemBoundaries:
    """FAST: the latent render path must not reach into the legacy identity-state
    internals. It owns identity via the IdentityEstimator subsystem, not the
    pipeline's IdentityState (_anchor_albedo / _intrinsic_decomposer / _gate)."""

    def test_latent_path_does_not_touch_legacy_identity_internals(self):
        p, cropped, landmarks, crop_plan = _make_latent_pipeline()
        probe = _BoundaryProbe()
        p.identity_state = probe
        out = p._render_with_latent(cropped, landmarks, crop_plan, frame_idx=0)
        # A tripped probe raises -> swallowed by _render_with_latent's broad
        # except -> out is None. So result-is-not-None AND an empty touch-list
        # both prove the latent path never reached the legacy internals.
        assert out is not None, (
            "latent path fell back to None — likely tripped the boundary probe "
            f"(touched={probe.touched})"
        )
        assert probe.touched == [], (
            f"latent path accessed forbidden identity_state attrs: {probe.touched}"
        )


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
        p._capture_latent_debug = True  # stash pre-composite face + mask + source
        outputs, debug = [], []
        cap = cv2.VideoCapture(clip)
        try:
            for i in range(6):
                ret, frame = cap.read()
                if not ret:
                    break
                p._last_latent_debug = None
                result = p.process_frame(frame, frame_idx=i)
                outputs.append(result.get('frame') if isinstance(result, dict) else None)
                debug.append(p._last_latent_debug)
        finally:
            cap.release()
        return p, outputs, debug

    @staticmethod
    def _mask_interior_stats(dbg):
        """(rendered_mean, rendered_std, source_mean) in grayscale 0-255 over the
        REAL crop_mask interior (not the diluted landmark bbox)."""
        rf = cv2.cvtColor(dbg["rendered_face"], cv2.COLOR_BGR2GRAY).astype(np.float64)
        sc = dbg["source_crop"]
        sg = cv2.cvtColor(sc, cv2.COLOR_BGR2GRAY).astype(np.float64)
        if sg.shape != rf.shape:
            sg = cv2.resize(sg, (rf.shape[1], rf.shape[0]))
        mi = dbg["crop_mask"] > 0.5
        if int(mi.sum()) < 16:
            return None
        return float(rf[mi].mean()), float(rf[mi].std()), float(sg[mi].mean())

    def test_latent_drives_pixels_at_least_once(self, latent_run):
        """At least one frame must report latent_primary=True — the latent
        actually rendered the face, source did not."""
        p, _, _ = latent_run
        latent_log = p.get_latent_telemetry()
        assert len(latent_log) > 0
        primaries = [r for r in latent_log if r["latent_primary"] is True]
        assert len(primaries) > 0, (
            "render_source='latent' but NO frame reported latent_primary=True — "
            "the latent never drove the render (branch not wired or always fell back)"
        )

    def test_latent_render_path_reported(self, latent_run):
        """Frames the latent drives must report render_path='latent'."""
        p, _, _ = latent_run
        latent_log = p.get_latent_telemetry()
        latent_frames = [r for r in latent_log if r["latent_primary"]]
        for r in latent_frames:
            assert r["render_path"] == "latent", f"latent frame mislabeled: {r}"

    def test_latent_render_reduces_source_fraction(self, latent_run):
        """A latent-driven face is NOT the source crop: source_pixel_fraction
        must drop below the spec no-leak threshold (0.02) — measured over the
        face-mask INTERIOR (design.md:545), not the whole crop."""
        p, _, _ = latent_run
        latent_log = p.get_latent_telemetry()
        primaries = [r for r in latent_log if r["latent_primary"]]
        assert primaries, "no latent-primary frames to check"
        for r in primaries:
            assert r["source_pixel_fraction"] < 0.02, (
                f"latent-primary frame leaks source inside the mask: {r}"
            )

    def test_latent_render_still_produces_valid_frames(self, latent_run):
        """The latent render path must still produce well-formed output frames
        (no crash, correct dtype/shape) — runtime truth, not just telemetry."""
        _, outputs, _ = latent_run
        produced = [o for o in outputs if o is not None]
        assert len(produced) > 0, "latent render produced no output frames"
        for o in produced:
            assert isinstance(o, np.ndarray)
            assert o.ndim == 3 and o.shape[2] == 3
            assert not np.any(np.isnan(o.astype(np.float32)))

    def test_latent_render_matches_scene_exposure(self, latent_run):
        """RUNTIME TRUTH (brightness): the latent face must be rendered under the
        CURRENT scene exposure, not its own albedo brightness. The 2.1×-too-bright
        collapse (mask mean ~194 vs scene ~93) was invisible to every plumbing
        test — this pins it. The rendered mask-interior mean must be within ±40%
        of the source scene mean over the mask."""
        _, _, debug = latent_run
        stats = [self._mask_interior_stats(d) for d in debug if d is not None]
        stats = [s for s in stats if s is not None]
        assert stats, "no latent debug captured — capture hook not firing"
        for lat_mean, _std, src_mean in stats:
            ratio = lat_mean / max(src_mean, 1e-6)
            assert 0.6 <= ratio <= 1.4, (
                f"latent face brightness {lat_mean:.1f} vs scene {src_mean:.1f} "
                f"(ratio {ratio:.2f}) — outside scene exposure; lighting collapsed "
                f"to albedo brightness (the 2.1× bug)"
            )

    def test_latent_render_is_not_flat(self, latent_run):
        """RUNTIME TRUTH (structure): the latent face must carry real spatial
        structure on EVERY latent frame, not collapse to a flat blob. The
        lighting-collapse bug drove mask-interior std to ~1.3 on mesh-normal
        frames (while a mean across frames was dragged up by other frames — so
        this asserts PER FRAME, the only honest guard). A structured face clears
        a floor the flat collapse (std ~1.3) never could."""
        _, _, debug = latent_run
        stds = [s[1] for s in (self._mask_interior_stats(d) for d in debug if d is not None) if s is not None]
        assert stds, "no latent debug captured — capture hook not firing"
        worst = float(min(stds))
        assert worst > 5.0, (
            f"a latent frame has mask-interior std {worst:.1f} — too flat; the "
            f"render collapsed to near-uniform (lighting/shading not driving "
            f"structure). per-frame stds={[round(s,1) for s in stds]}"
        )

    def test_gate_state_couples_to_render(self, latent_run):
        """RUNTIME TRUTH (Phase 2B gate): the gate DECISION must actually CONTROL
        the render — biconditional ``gate_state=='engaged' iff latent_primary``.
        This is the anti-"decorative telemetry" guard: a gate that emitted a
        label but never changed which path rendered would pass a flat label
        check yet be a no-op. Here, every engaged frame MUST have driven the
        latent (primary + render_path='latent'), and every non-engaged frame
        MUST have fallen back (primary False, path != 'latent'). Also pins the
        gate_state label vocabulary."""
        valid = {"engaged", "below_floor", "confidence_spike", "uninitialized", "disabled"}
        p, _, _ = latent_run
        log = p.get_latent_telemetry()
        assert log, "no latent telemetry captured"
        for r in log:
            gs = r["gate_state"]
            assert gs in valid, f"unknown gate_state {gs!r} (frame {r['frame_idx']})"
            if gs == "engaged":
                assert r["latent_primary"] is True and r["render_path"] == "latent", (
                    f"frame {r['frame_idx']} gate ENGAGED but did not drive the "
                    f"latent (primary={r['latent_primary']}, path={r['render_path']}) "
                    f"— gate decision is decorative, not controlling the render"
                )
            else:
                assert r["latent_primary"] is False and r["render_path"] != "latent", (
                    f"frame {r['frame_idx']} gate {gs!r} (not engaged) but the "
                    f"latent still drove the render (primary={r['latent_primary']}, "
                    f"path={r['render_path']}) — gate refusal did not fall back"
                )

    def test_gate_engages_on_real_video(self, latent_run):
        """RUNTIME TRUTH (Phase 2B gate not a no-op): the relative-to-floor gate
        must ENGAGE at least once on real video — otherwise the latent never
        drives and Phase 2B is a total regression (the absolute-threshold trap
        this design exists to avoid: confidence plateaus at ~0.2567, so any fixed
        threshold >~0.26 would refuse forever). The plateau frames (top
        confidence, dC/dt=0) are precisely the ones that must engage."""
        p, _, _ = latent_run
        log = p.get_latent_telemetry()
        engaged = [r for r in log if r["gate_state"] == "engaged"]
        assert engaged, (
            "Phase 2B gate NEVER engaged on real video — the latent never drove "
            "the face. gate_states="
            f"{[r['gate_state'] for r in log]}, confidences="
            f"{[round(r['latent_confidence'], 4) for r in log]}"
        )

    def test_hybrid_blend_engages_and_respects_cap(self, latent_run):
        """RUNTIME TRUTH (Phase 2B per-pixel hybrid): on real video the latent is
        broadly uncertain (measured interior U_mean ~0.65), so the uncertainty
        hybrid MUST actually engage — and it MUST respect the blend_max cap so
        the latent never loses majority authority.

        Per engaged latent-primary frame, hybrid_alpha_mean (mean per-pixel
        LATENT weight) must satisfy ``1-blend_max <= alpha_mean < 1.0``:
          - ``>= 1-blend_max`` proves the CAP held (latent kept >=50% everywhere;
            the source-leak metric stays bounded — pinned separately by
            test_latent_render_reduces_source_fraction on the same composite).
          - ``< 1.0`` proves the hybrid is NOT a no-op (observation crossed where
            uncertain).
        At least one frame must show CLEAR blending (alpha_mean < 0.9) so a
        single near-confident pixel can't masquerade as engagement."""
        p, _, _ = latent_run
        floor = 1.0 - p._hybrid_blend_max
        log = p.get_latent_telemetry()
        engaged = [r for r in log if r["gate_state"] == "engaged"]
        assert engaged, "no engaged frames — cannot check hybrid"
        alphas = []
        for r in engaged:
            a = r["hybrid_alpha_mean"]
            alphas.append(a)
            assert floor - 1e-6 <= a < 1.0, (
                f"frame {r['frame_idx']} hybrid_alpha_mean {a:.4f} outside "
                f"[{floor:.2f}, 1.0): cap breached (latent lost majority) or "
                f"hybrid was a no-op"
            )
        assert min(alphas) < 0.9, (
            f"hybrid never meaningfully engaged on real video (min alpha_mean "
            f"{min(alphas):.4f} >= 0.9) — the observation never crossed despite "
            f"broad latent uncertainty. alphas={[round(a, 4) for a in alphas]}"
        )


@pytest.mark.slow
@pytest.mark.timeout(600)
class TestLatentQualityOnRealVideo:
    """D-05 task 4.5: runtime-truth slow test asserting latent quality targets
    on the real test clip (clips_test/test_clip.mp4 — the 1.2 GB master video is
    intentionally NOT used; this clip is representative and bounded)."""

    def test_latent_primary_and_source_fraction(self):
        """latent_primary=True and source_pixel_fraction < 0.02 for ≥90% of face frames."""
        from face_os.pipeline import FaceOSPipeline
        clip = _shadow_test_clip()
        if clip is None:
            pytest.skip("No test clip available (clips_test/test_clip.mp4)")
        pipeline = FaceOSPipeline(use_bidirectional=False)
        pipeline.render_source = 'latent'
        pipeline.enroll()

        import cv2
        cap = cv2.VideoCapture(clip)
        latent_frames = 0
        face_frames = 0
        total_frames = 0
        fractions = []

        # Bounded frame budget keeps the latent path within the timeout; the
        # quality targets are stable well before this many face frames.
        clean_frames = 0  # latent_primary AND leak < 0.02 (the spec conjunction)
        while total_frames < 60:
            ret, frame = cap.read()
            if not ret:
                break
            result = pipeline.process_frame(frame, frame_idx=total_frames)
            if result and result.get('frame') is not None:
                total_frames += 1
                telem = pipeline._frame_telemetry_log[-1] if pipeline._frame_telemetry_log else {}
                latent = telem.get('latent', {})
                # Only count frames where face is detected (not LOST_FACE)
                if pipeline._face_state != 'LOST_FACE':
                    face_frames += 1
                    is_primary = latent.get('latent_primary', False)
                    frac = latent.get('source_pixel_fraction', 1.0)
                    if is_primary:
                        latent_frames += 1
                        fractions.append(frac)  # leak is only defined on driven frames
                        if frac < 0.02:
                            clean_frames += 1

        cap.release()
        del pipeline

        if face_frames < 10:
            pytest.skip("too few face frames processed")

        # requirements.md:125 — the spec criterion is FRAME-COUNT, not a mean:
        # "latent_primary true AND source_pixel_fraction below 0.02 for at least
        # 90 percent of physical frames". A mean is the wrong statistic — a single
        # legacy frame (source_pixel_fraction=1.0 by definition) would dominate it.
        latent_pct = latent_frames / face_frames
        clean_pct = clean_frames / face_frames
        leak_dist = (
            f"n={len(fractions)} mean={np.mean(fractions):.4f} "
            f"p90={np.percentile(fractions, 90):.4f} max={np.max(fractions):.4f}"
            if fractions else "n=0"
        )
        assert latent_pct >= 0.90, f"latent_primary only on {latent_pct:.1%} of face frames"
        assert clean_pct >= 0.90, (
            f"no-leak held on only {clean_pct:.1%} of physical frames "
            f"(spec: >= 90% with source_pixel_fraction < 0.02). leak dist: {leak_dist}"
        )
