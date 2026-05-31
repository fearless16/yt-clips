"""Layer 2b: PhysicalRenderer unit tests.

Validates the physically-inspired renderer (D-01: signal preservation,
D-02: physical quality).

Mathematical invariants:
  - Lambertian diffuse: proportional to max(N·L, 0)
  - Blinn-Phong specular: peak at half-vector
  - Energy conservation: ECR ∈ [0.5, 1.5]
  - Detail residual: zero-mean HF band
  - Output: linear-light float32 in [0, 1]
"""
import cv2
import numpy as np
import pytest

from face_os.physical_renderer import (
    PhysicalRenderer,
    PhysicalRenderConfig,
    PhysicalRenderOutput,
    LightingModel,
    RenderReport,
)
from face_os.intrinsic_decomposition import IntrinsicComponents


def _unit(v):
    """Normalize a 3-vector (test helper for light directions)."""
    v = np.asarray(v, np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


@pytest.fixture
def renderer():
    """Default PhysicalRenderer."""
    return PhysicalRenderer()


@pytest.fixture
def render_result(renderer, synthetic_albedo, synthetic_normals, synthetic_shading):
    """Rendered output from synthetic inputs."""
    return renderer.render(
        albedo=synthetic_albedo,
        normal_map=synthetic_normals,
        shading=synthetic_shading,
        observed=synthetic_albedo,  # Use albedo as observed for HF reference
        frame_idx=0,
    )


# ═══════════════════════════════════════════════════════════════════
# Output Contract
# ═══════════════════════════════════════════════════════════════════

class TestRenderOutputContract:
    """Render output must have all components with correct shapes."""

    def test_output_type(self, render_result):
        """Output is PhysicalRenderOutput."""
        assert isinstance(render_result, PhysicalRenderOutput)

    def test_rendered_shape_matches_albedo(self, render_result, synthetic_albedo):
        """Rendered image shape matches albedo input."""
        assert render_result.rendered.shape == synthetic_albedo.shape

    def test_rendered_dtype_float32(self, render_result):
        """Output is float32 (linear-light, not uint8)."""
        assert render_result.rendered.dtype == np.float32

    def test_rendered_in_unit_range(self, render_result):
        """Output pixels in [0, 1] (clamped)."""
        assert float(np.min(render_result.rendered)) >= 0.0
        assert float(np.max(render_result.rendered)) <= 1.0

    def test_all_components_present(self, render_result):
        """Diffuse, specular, ambient, detail components all non-None."""
        assert render_result.diffuse_component is not None
        assert render_result.specular_component is not None
        assert render_result.ambient_component is not None
        assert render_result.detail_component is not None

    def test_components_same_shape(self, render_result):
        """All components have same spatial dimensions."""
        shape = render_result.rendered.shape
        assert render_result.diffuse_component.shape == shape
        assert render_result.specular_component.shape == shape
        assert render_result.ambient_component.shape == shape
        assert render_result.detail_component.shape == shape


# ═══════════════════════════════════════════════════════════════════
# Lighting Model
# ═══════════════════════════════════════════════════════════════════

class TestLightingComponents:
    """Individual lighting components must follow physical laws."""

    def test_ambient_proportional_to_albedo(self, renderer, synthetic_albedo, synthetic_normals, synthetic_shading):
        """Ambient ∝ albedo × lighting.ambient."""
        lighting = LightingModel(ambient=0.20, diffuse_intensity=0.0, specular_intensity=0.0)
        result = renderer.render(
            albedo=synthetic_albedo,
            normal_map=synthetic_normals,
            shading=synthetic_shading,
            lighting=lighting,
        )
        # With zero diffuse and specular, ambient dominates
        ambient_mean = float(np.mean(result.ambient_component))
        expected = float(np.mean(synthetic_albedo)) * 0.20
        assert ambient_mean > 0, "Ambient should be positive"
        # Ambient should scale with albedo
        assert abs(ambient_mean - expected) / max(expected, 1e-6) < 0.5

    def test_diffuse_lambertian_cos_law(self, renderer, synthetic_normals, synthetic_shading):
        """Diffuse should be proportional to max(N·L, 0) (Lambertian)."""
        h, w = 128, 128
        albedo = np.ones((h, w, 3), dtype=np.float32) * 0.5
        # Light from straight ahead: L = [0, 0, 1]
        lighting = LightingModel(
            ambient=0.0,
            diffuse_direction=np.array([0.0, 0.0, 1.0]),
            diffuse_intensity=1.0,
            specular_intensity=0.0,
        )
        result = renderer.render(
            albedo=albedo,
            normal_map=synthetic_normals,
            shading=np.ones((h, w, 1), dtype=np.float32),
            lighting=lighting,
        )
        # With frontal normals and frontal light, N·L should be high
        diffuse_mean = float(np.mean(result.diffuse_component))
        assert diffuse_mean > 0.1, f"Diffuse mean={diffuse_mean:.4f} too low for frontal lighting"

    def test_specular_peak_exists(self, renderer, synthetic_normals, synthetic_shading):
        """Specular should have a visible peak (not flat zero)."""
        h, w = 128, 128
        albedo = np.ones((h, w, 3), dtype=np.float32) * 0.5
        lighting = LightingModel(
            ambient=0.0, diffuse_intensity=0.0,
            specular_intensity=1.0, specular_power=32.0,
        )
        result = renderer.render(
            albedo=albedo,
            normal_map=synthetic_normals,
            shading=np.ones((h, w, 1), dtype=np.float32),
            lighting=lighting,
        )
        spec_max = float(np.max(result.specular_component))
        spec_mean = float(np.mean(result.specular_component))
        assert spec_max > spec_mean * 1.5, "Specular should have concentrated peaks"


# ═══════════════════════════════════════════════════════════════════
# Energy Conservation
# ═══════════════════════════════════════════════════════════════════

class TestEnergyConservation:
    """Rendered energy must be bounded relative to input energy."""

    def test_ecr_in_range(self, render_result, synthetic_albedo, synthetic_shading):
        """ECR = rendered_energy / (albedo × shading) should be in [0.3, 3.0]."""
        rendered_energy = float(np.mean(render_result.rendered))
        input_energy = float(np.mean(synthetic_albedo * synthetic_shading))
        if input_energy < 1e-6:
            pytest.skip("Input energy too low")
        ecr = rendered_energy / input_energy
        assert 0.3 < ecr < 3.0, f"ECR={ecr:.3f} outside [0.3, 3.0]"

    @pytest.mark.parametrize("shading_val", [0.1, 0.25, 0.5, 0.8])
    def test_ecr_stable_across_shading(self, renderer, synthetic_albedo, synthetic_normals, shading_val):
        """ECR should be stable across different shading levels."""
        h, w = synthetic_albedo.shape[:2]
        shading = np.full((h, w, 1), shading_val, dtype=np.float32)
        result = renderer.render(
            albedo=synthetic_albedo,
            normal_map=synthetic_normals,
            shading=shading,
        )
        rendered_energy = float(np.mean(result.rendered))
        input_energy = float(np.mean(synthetic_albedo)) * shading_val
        assert rendered_energy > input_energy * 0.1, (
            f"shading={shading_val}: rendered={rendered_energy:.4f} < "
            f"10% of input={input_energy:.4f}"
        )

    def test_shading_modulation_brightness(self, renderer, synthetic_albedo, synthetic_normals):
        """Higher shading → brighter output."""
        h, w = synthetic_albedo.shape[:2]
        results = []
        for sv in [0.1, 0.5, 0.9]:
            shading = np.full((h, w, 1), sv, dtype=np.float32)
            result = renderer.render(
                albedo=synthetic_albedo,
                normal_map=synthetic_normals,
                shading=shading,
            )
            results.append(float(np.mean(result.rendered)))
        # Brightness should monotonically increase with shading
        assert results[0] < results[1] < results[2], (
            f"Brightness not monotonic with shading: {results}"
        )


# ═══════════════════════════════════════════════════════════════════
# Detail Component
# ═══════════════════════════════════════════════════════════════════

class TestDetailComponent:
    """Detail residual must be zero-mean HF band."""

    def test_detail_approximately_zero_mean(self, render_result):
        """Detail should be approximately zero-mean (HF band, no DC)."""
        detail_mean = float(np.mean(render_result.detail_component))
        assert abs(detail_mean) < 0.05, f"Detail mean={detail_mean:.4f} not near zero"

    def test_detail_energy_bounded(self, render_result):
        """Detail energy should be small relative to rendered energy."""
        detail_energy = float(np.mean(np.abs(render_result.detail_component)))
        rendered_energy = float(np.mean(render_result.rendered))
        if rendered_energy < 1e-6:
            pytest.skip("Rendered energy too low")
        ratio = detail_energy / rendered_energy
        assert ratio < 0.5, f"Detail energy ratio={ratio:.3f} too high"


# ═══════════════════════════════════════════════════════════════════
# HF Retention & Render Metrics
# ═══════════════════════════════════════════════════════════════════

class TestRenderMetrics:
    """Render output metrics must be populated and reasonable."""

    def test_hf_retention_positive(self, render_result):
        """HF retention should be > 0."""
        assert render_result.high_frequency_retention >= 0.0

    def test_rendering_error_bounded(self, render_result):
        """Rendering error should be < 0.5."""
        assert render_result.rendering_error < 0.5, (
            f"Rendering error={render_result.rendering_error:.3f}"
        )

    def test_render_time_positive(self, render_result):
        """Render time should be measured."""
        assert render_result.render_time_ms > 0

    def test_render_report_populated(self, renderer, synthetic_albedo, synthetic_normals, synthetic_shading):
        """RenderReport is populated when frame_idx is given."""
        renderer.render(
            albedo=synthetic_albedo,
            normal_map=synthetic_normals,
            shading=synthetic_shading,
            frame_idx=42,
        )
        report = renderer._last_report
        assert report is not None
        assert report.frame_idx == 42
        assert report.render_time_ms > 0
        d = report.to_dict()
        assert 'rendering_error' in d
        assert 'mean_diffuse' in d


# ═══════════════════════════════════════════════════════════════════
# Latent-path identity microdetail (design.md:388-401)
# ═══════════════════════════════════════════════════════════════════

class TestIntrinsicDetailResidual:
    """`render_with_intrinsic` is the latent render path (renderer.py:141).

    design.md:390-401 mandates the render core consume the carried identity HF:
        detail  <- components.detail_residual   // identity HF (NOT source HF)
        Y_face  <- base + detail_strength * detail

    The latent albedo arrives bilinearly warped from canonical UV, so its own
    high-pass is near-zero; the stored best-observation `microdetail` (carried in
    `detail_residual`) is the ONLY identity HF source on this path. If the render
    core ignores it, the output is blurry (the measured 4.3x sharpness loss).
    """

    @staticmethod
    def _smooth_albedo(h=128, w=128, seed=0):
        rng = np.random.default_rng(seed)
        base = rng.uniform(0.4, 0.7, (h, w, 3)).astype(np.float32)
        # Emulate the HF-smoothing of a bilinear canonical->crop warp.
        return cv2.GaussianBlur(base, (0, 0), 6.0).astype(np.float32)

    @staticmethod
    def _components(albedo, detail):
        h, w = albedo.shape[:2]
        normal = np.zeros((h, w, 3), np.float32)
        normal[..., 2] = 1.0
        return IntrinsicComponents(
            albedo=albedo,
            shading=np.ones((h, w, 1), np.float32),
            specular=np.zeros((h, w, 3), np.float32),
            normal_map=normal,
            confidence=np.ones((h, w, 1), np.float32),
            reconstruction_error=0.0,
            detail_residual=detail,
        )

    @staticmethod
    def _lapvar(img):
        gray = img.mean(axis=2).astype(np.float32)
        return float(np.var(cv2.Laplacian(gray, cv2.CV_32F, ksize=3)))

    def test_detail_residual_changes_output(self, renderer):
        """Stored identity microdetail MUST reach the render output (design.md:390)."""
        albedo = self._smooth_albedo()
        rng = np.random.default_rng(1)
        hf = (rng.standard_normal(albedo.shape).astype(np.float32) * 0.08)
        hf -= hf.mean(axis=(0, 1), keepdims=True)

        out_zero = renderer.render_with_intrinsic(
            self._components(albedo, np.zeros_like(albedo))
        ).rendered
        out_hf = renderer.render_with_intrinsic(
            self._components(albedo, hf)
        ).rendered

        delta = float(np.max(np.abs(out_hf - out_zero)))
        assert delta > 1e-3, (
            "render_with_intrinsic ignores components.detail_residual: identity "
            f"microdetail has zero effect on output (max delta={delta:.2e}). "
            "Latent-path HF is dropped (drift from design.md:390-401)."
        )

    def test_detail_residual_increases_sharpness(self, renderer):
        """Carrying identity HF raises rendered HF energy vs. a zero residual."""
        albedo = self._smooth_albedo(seed=2)
        rng = np.random.default_rng(3)
        hf = (rng.standard_normal(albedo.shape).astype(np.float32) * 0.08)
        hf -= hf.mean(axis=(0, 1), keepdims=True)

        out_zero = renderer.render_with_intrinsic(
            self._components(albedo, np.zeros_like(albedo))
        ).rendered
        out_hf = renderer.render_with_intrinsic(
            self._components(albedo, hf)
        ).rendered

        assert self._lapvar(out_hf) > self._lapvar(out_zero) * 1.5, (
            "Identity microdetail must measurably sharpen the latent render."
        )


class TestLatentExposureStability:
    """arch.md §16.2 + §18 (MEASURED): on the latent path, rendered EXPOSURE must
    not float with the normal source.

    Measured flicker root cause: when mesh normals are unavailable the renderer
    falls back to the generic `face_prior` hemisphere and `estimate_lighting`
    fits a different (oblique) light. With mesh normals (frontal N≈(0,0,1),
    light≈(0,0,1)) N·L≈1 → bright render (~85); with the sphere + oblique light
    half the face clamps at N·L≤0 and `clip` removes peak energy → ~30% darker
    (~60). The identity did not change — only the GEOMETRY SOURCE did — so this is
    a Principle-4 / §16.2 violation: a 50× flicker spike at each mesh↔face_prior
    flip.

    Invariant (the renderer's own stated intent, physical_renderer.py:382-394):
    'shading controls absolute brightness'. On the latent path the rendered mean
    exposure MUST equal mean(albedo × shading) — which is lighting-invariant and
    temporally continuous — regardless of normal map or light direction.
    """

    @staticmethod
    def _albedo(h=128, w=128, seed=7):
        rng = np.random.default_rng(seed)
        return rng.uniform(0.55, 0.80, (h, w, 3)).astype(np.float32)

    @staticmethod
    def _shading(h=128, w=128):
        # BRIGHT shading (mean ~0.85). This is the regime where the measured bug
        # lives: the peaky face_prior+oblique render lands in the energy-
        # conservation DEAD ZONE (ratio ∈ [0.5, 0.95], left uncorrected), so its
        # exposure floats below the mesh render. Dim shading (≤0.42) clamps both
        # to ratio 0.95 and hides the bug — verified empirically.
        Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
        s = 0.78 + 0.15 * (X / w)
        return s[:, :, np.newaxis].astype(np.float32)

    @staticmethod
    def _frontal_normals(h=128, w=128):
        n = np.zeros((h, w, 3), np.float32)
        n[..., 2] = 1.0
        return n

    @staticmethod
    def _sphere_normals(h=128, w=128):
        # The exact generic face-prior hemisphere the renderer falls back to.
        from face_os.intrinsic_decomposition import IntrinsicDecomposer
        return IntrinsicDecomposer._get_cached_face_prior(h, w)

    @staticmethod
    def _components(albedo, normals, shading, detail):
        h, w = albedo.shape[:2]
        return IntrinsicComponents(
            albedo=albedo,
            shading=shading,
            specular=np.zeros((h, w, 3), np.float32),
            normal_map=normals,
            confidence=np.ones((h, w, 1), np.float32),
            reconstruction_error=0.0,
            detail_residual=detail,  # not None => latent path
        )

    def test_exposure_invariant_to_normal_source(self, renderer):
        """A mesh↔face_prior normal-source switch must NOT move rendered exposure."""
        albedo = self._albedo()
        shading = self._shading()
        detail = np.zeros_like(albedo)  # isolate exposure (no HF contribution)

        frontal = LightingModel(
            ambient=0.10,
            diffuse_direction=np.array([0.0, 0.0, 1.0], np.float64),
            diffuse_intensity=0.80,
            specular_intensity=0.0,
        )
        oblique = LightingModel(
            ambient=0.10,
            diffuse_direction=_unit(np.array([0.22, -0.82, 0.53], np.float64)),
            diffuse_intensity=0.80,
            specular_intensity=0.0,
        )

        out_mesh = renderer.render_with_intrinsic(
            self._components(albedo, self._frontal_normals(), shading, detail),
            lighting=frontal,
        ).rendered
        out_prior = renderer.render_with_intrinsic(
            self._components(albedo, self._sphere_normals(), shading, detail),
            lighting=oblique,
        ).rendered

        mean_mesh = float(np.mean(out_mesh))
        mean_prior = float(np.mean(out_prior))
        rel = abs(mean_mesh - mean_prior) / max(mean_mesh, 1e-6)
        assert rel < 0.05, (
            f"Latent render exposure floats with normal source: mesh mean="
            f"{mean_mesh:.4f} vs face_prior mean={mean_prior:.4f} "
            f"(relative step={rel:.1%} ≥ 5%). This is the measured flicker spike."
        )

    def test_latent_exposure_matches_albedo_shading_target(self, renderer):
        """Latent render mean exposure ≈ mean(albedo × shading) (design intent)."""
        albedo = self._albedo(seed=11)
        shading = self._shading()
        detail = np.zeros_like(albedo)
        target = float(np.mean(albedo * np.repeat(shading, 3, axis=2)))

        oblique = LightingModel(
            ambient=0.10,
            diffuse_direction=_unit(np.array([0.22, -0.82, 0.53], np.float64)),
            diffuse_intensity=0.80,
            specular_intensity=0.0,
        )
        out = renderer.render_with_intrinsic(
            self._components(albedo, self._sphere_normals(), shading, detail),
            lighting=oblique,
        ).rendered

        mean_out = float(np.mean(out))
        rel = abs(mean_out - target) / max(target, 1e-6)
        assert rel < 0.08, (
            f"Latent render exposure {mean_out:.4f} does not match the "
            f"albedo×shading target {target:.4f} (off by {rel:.1%}); shading must "
            f"control absolute brightness on the latent path."
        )
