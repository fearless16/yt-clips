"""Phase 3A: Intrinsic Decomposition Tests.

Tests for decomposing face images into intrinsic components:
- Albedo (identity-intrinsic, lighting-invariant)
- Shading (lighting-dependent, identity-invariant)
- Specular (view-dependent)
- Normal map (surface orientation)
- Confidence (decomposition quality)

Target: 25 tests
"""

import numpy as np
import pytest

from face_os.intrinsic_decomposition import (
    IntrinsicComponents,
    IntrinsicDecomposer,
    DecompositionConfig,
    DecompositionReport,
)


class TestIntrinsicComponents:
    """Test IntrinsicComponents dataclass."""

    def test_albedo_shape(self):
        """Albedo must be (H, W, 3)."""
        albedo = np.random.rand(256, 256, 3).astype(np.float32)
        shading = np.ones((256, 256, 1), dtype=np.float32)
        specular = np.zeros((256, 256, 3), dtype=np.float32)
        normal_map = np.zeros((256, 256, 3), dtype=np.float32)
        confidence = np.ones((256, 256, 1), dtype=np.float32)

        comp = IntrinsicComponents(
            albedo=albedo,
            shading=shading,
            specular=specular,
            normal_map=normal_map,
            confidence=confidence,
            reconstruction_error=0.0,
        )
        assert comp.albedo.shape == (256, 256, 3)

    def test_shading_shape(self):
        """Shading must be (H, W, 1)."""
        albedo = np.random.rand(256, 256, 3).astype(np.float32)
        shading = np.ones((256, 256, 1), dtype=np.float32)
        specular = np.zeros((256, 256, 3), dtype=np.float32)
        normal_map = np.zeros((256, 256, 3), dtype=np.float32)
        confidence = np.ones((256, 256, 1), dtype=np.float32)

        comp = IntrinsicComponents(
            albedo=albedo,
            shading=shading,
            specular=specular,
            normal_map=normal_map,
            confidence=confidence,
            reconstruction_error=0.0,
        )
        assert comp.shading.shape == (256, 256, 1)

    def test_albedo_range(self):
        """Albedo must be in [0, 1]."""
        albedo = np.random.rand(256, 256, 3).astype(np.float32)
        shading = np.ones((256, 256, 1), dtype=np.float32)
        specular = np.zeros((256, 256, 3), dtype=np.float32)
        normal_map = np.zeros((256, 256, 3), dtype=np.float32)
        confidence = np.ones((256, 256, 1), dtype=np.float32)

        comp = IntrinsicComponents(
            albedo=albedo,
            shading=shading,
            specular=specular,
            normal_map=normal_map,
            confidence=confidence,
            reconstruction_error=0.0,
        )
        assert np.all(comp.albedo >= 0) and np.all(comp.albedo <= 1)

    def test_shading_range(self):
        """Shading must be in [0, 1]."""
        albedo = np.random.rand(256, 256, 3).astype(np.float32)
        shading = np.random.rand(256, 256, 1).astype(np.float32)
        specular = np.zeros((256, 256, 3), dtype=np.float32)
        normal_map = np.zeros((256, 256, 3), dtype=np.float32)
        confidence = np.ones((256, 256, 1), dtype=np.float32)

        comp = IntrinsicComponents(
            albedo=albedo,
            shading=shading,
            specular=specular,
            normal_map=normal_map,
            confidence=confidence,
            reconstruction_error=0.0,
        )
        assert np.all(comp.shading >= 0) and np.all(comp.shading <= 1)

    def test_specular_non_negative(self):
        """Specular must be non-negative."""
        albedo = np.random.rand(256, 256, 3).astype(np.float32)
        shading = np.ones((256, 256, 1), dtype=np.float32)
        specular = np.abs(np.random.rand(256, 256, 3)).astype(np.float32)
        normal_map = np.zeros((256, 256, 3), dtype=np.float32)
        confidence = np.ones((256, 256, 1), dtype=np.float32)

        comp = IntrinsicComponents(
            albedo=albedo,
            shading=shading,
            specular=specular,
            normal_map=normal_map,
            confidence=confidence,
            reconstruction_error=0.0,
        )
        assert np.all(comp.specular >= 0)

    def test_normal_map_normalized(self):
        """Normal map vectors must be unit length."""
        normal_map = np.random.rand(256, 256, 3).astype(np.float32)
        norms = np.linalg.norm(normal_map, axis=2, keepdims=True)
        normal_map = normal_map / (norms + 1e-8)

        albedo = np.random.rand(256, 256, 3).astype(np.float32)
        shading = np.ones((256, 256, 1), dtype=np.float32)
        specular = np.zeros((256, 256, 3), dtype=np.float32)
        confidence = np.ones((256, 256, 1), dtype=np.float32)

        comp = IntrinsicComponents(
            albedo=albedo,
            shading=shading,
            specular=specular,
            normal_map=normal_map,
            confidence=confidence,
            reconstruction_error=0.0,
        )
        norms = np.linalg.norm(comp.normal_map, axis=2)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_confidence_bounded(self):
        """Confidence must be in [0, 1]."""
        albedo = np.random.rand(256, 256, 3).astype(np.float32)
        shading = np.ones((256, 256, 1), dtype=np.float32)
        specular = np.zeros((256, 256, 3), dtype=np.float32)
        normal_map = np.zeros((256, 256, 3), dtype=np.float32)
        confidence = np.random.rand(256, 256, 1).astype(np.float32)

        comp = IntrinsicComponents(
            albedo=albedo,
            shading=shading,
            specular=specular,
            normal_map=normal_map,
            confidence=confidence,
            reconstruction_error=0.0,
        )
        assert np.all(comp.confidence >= 0) and np.all(comp.confidence <= 1)


class TestDecompositionConfig:
    """Test DecompositionConfig."""

    def test_default_config(self):
        """Default config must have reasonable values."""
        config = DecompositionConfig()
        assert config.albedo_smoothness > 0
        assert config.shading_smoothness > 0
        assert config.specular_threshold > 0
        assert 0 < config.confidence_threshold < 1

    def test_config_validation(self):
        """Config values must be valid."""
        config = DecompositionConfig(
            albedo_smoothness=0.5,
            shading_smoothness=0.3,
            specular_threshold=0.1,
            confidence_threshold=0.5,
        )
        assert config.albedo_smoothness == 0.5
        assert config.shading_smoothness == 0.3
        assert config.specular_threshold == 0.1
        assert config.confidence_threshold == 0.5


class TestIntrinsicDecomposer:
    """Test IntrinsicDecomposer."""

    def test_decompose_shape(self):
        """Decomposition must preserve input shape."""
        decomposer = IntrinsicDecomposer()
        image = np.random.rand(256, 256, 3).astype(np.float32)
        result = decomposer.decompose(image)
        assert result.albedo.shape == (256, 256, 3)
        assert result.shading.shape == (256, 256, 1)
        assert result.specular.shape == (256, 256, 3)

    def test_decompose_valid_range(self):
        """Decomposition outputs must be in valid range."""
        decomposer = IntrinsicDecomposer()
        image = np.random.rand(256, 256, 3).astype(np.float32)
        result = decomposer.decompose(image)
        assert np.all(result.albedo >= 0) and np.all(result.albedo <= 1)
        assert np.all(result.shading >= 0) and np.all(result.shading <= 1)
        assert np.all(result.specular >= 0)

    def test_reconstruction_closeness(self):
        """Y ≈ A * S + specular must hold."""
        decomposer = IntrinsicDecomposer()
        image = np.random.rand(256, 256, 3).astype(np.float32)
        result = decomposer.decompose(image)
        
        # Reconstruct: Y_recon = A * S + specular
        S_3ch = np.repeat(result.shading, 3, axis=2)
        reconstructed = result.albedo * S_3ch + result.specular
        
        # Allow some tolerance due to regularization
        error = np.mean(np.abs(image - reconstructed))
        assert error < 0.3  # 30% tolerance

    def test_albedo_smoothness(self):
        """Albedo must be spatially smooth (no high-frequency noise)."""
        decomposer = IntrinsicDecomposer()
        image = np.random.rand(256, 256, 3).astype(np.float32)
        result = decomposer.decompose(image)
        
        # Check smoothness via Laplacian
        from scipy.ndimage import laplace
        laplacian = laplace(result.albedo)
        high_freq_energy = np.mean(np.abs(laplacian))
        assert high_freq_energy < 0.5  # Smooth albedo

    def test_shading_smoothness(self):
        """Shading must be spatially smooth."""
        decomposer = IntrinsicDecomposer()
        image = np.random.rand(256, 256, 3).astype(np.float32)
        result = decomposer.decompose(image)
        
        from scipy.ndimage import laplace
        laplacian = laplace(result.shading)
        high_freq_energy = np.mean(np.abs(laplacian))
        assert high_freq_energy < 0.5  # Smooth shading

    def test_specular_sparsity(self):
        """Specular must be sparse (few bright highlights)."""
        decomposer = IntrinsicDecomposer()
        image = np.random.rand(256, 256, 3).astype(np.float32)
        result = decomposer.decompose(image)
        
        # Specular should be mostly zero
        specular_magnitude = np.linalg.norm(result.specular, axis=2)
        sparse_ratio = np.mean(specular_magnitude < 0.01)
        assert sparse_ratio > 0.5  # At least 50% should be near zero

    def test_albedo_invariance(self):
        """Albedo must be similar for same identity under different lighting."""
        decomposer = IntrinsicDecomposer()
        
        # Create two images with same "identity" but different "lighting"
        base = np.random.rand(256, 256, 3).astype(np.float32)
        bright = np.clip(base * 1.5, 0, 1)
        dark = np.clip(base * 0.5, 0, 1)
        
        result_bright = decomposer.decompose(bright)
        result_dark = decomposer.decompose(dark)
        
        # Albedo should be more similar than input images
        input_diff = np.mean(np.abs(bright - dark))
        albedo_diff = np.mean(np.abs(result_bright.albedo - result_dark.albedo))
        
        # Albedo difference should be smaller than input difference
        assert albedo_diff < input_diff

    def test_confidence_high_for_uniform(self):
        """Confidence must be high for uniform images."""
        decomposer = IntrinsicDecomposer()
        image = np.ones((256, 256, 3), dtype=np.float32) * 0.5
        result = decomposer.decompose(image)
        assert np.mean(result.confidence) > 0.5

    def test_confidence_low_for_noisy(self):
        """Confidence must be lower for noisy images."""
        decomposer = IntrinsicDecomposer()
        clean = np.ones((256, 256, 3), dtype=np.float32) * 0.5
        noisy = clean + np.random.randn(256, 256, 3).astype(np.float32) * 0.3
        noisy = np.clip(noisy, 0, 1)
        
        result_clean = decomposer.decompose(clean)
        result_noisy = decomposer.decompose(noisy)
        
        # Noisy should have lower confidence
        assert np.mean(result_noisy.confidence) < np.mean(result_clean.confidence)

    def test_normal_map_shape(self):
        """Normal map must have correct shape."""
        decomposer = IntrinsicDecomposer()
        image = np.random.rand(256, 256, 3).astype(np.float32)
        result = decomposer.decompose(image)
        assert result.normal_map.shape == (256, 256, 3)

    def test_normal_map_magnitude(self):
        """Normal map vectors must have unit magnitude."""
        decomposer = IntrinsicDecomposer()
        image = np.random.rand(256, 256, 3).astype(np.float32)
        result = decomposer.decompose(image)
        norms = np.linalg.norm(result.normal_map, axis=2)
        assert np.allclose(norms, 1.0, atol=1e-3)


class TestDecompositionReport:
    """Test DecompositionReport."""

    def test_report_to_dict(self):
        """Report must convert to dict."""
        albedo = np.random.rand(256, 256, 3).astype(np.float32)
        shading = np.ones((256, 256, 1), dtype=np.float32)
        specular = np.zeros((256, 256, 3), dtype=np.float32)
        normal_map = np.zeros((256, 256, 3), dtype=np.float32)
        confidence = np.ones((256, 256, 1), dtype=np.float32)

        comp = IntrinsicComponents(
            albedo=albedo,
            shading=shading,
            specular=specular,
            normal_map=normal_map,
            confidence=confidence,
            reconstruction_error=0.05,
        )

        report = DecompositionReport(
            frame_idx=0,
            components=comp,
            albedo_stability=0.9,
            shading_smoothness=0.8,
            specular_sparsity=0.95,
            decomposition_time_ms=10.0,
        )

        d = report.to_dict()
        assert d["frame_idx"] == 0
        assert d["albedo_stability"] == 0.9
        assert d["reconstruction_error"] == 0.05

    def test_report_has_metrics(self):
        """Report must have all required metrics."""
        albedo = np.random.rand(256, 256, 3).astype(np.float32)
        shading = np.ones((256, 256, 1), dtype=np.float32)
        specular = np.zeros((256, 256, 3), dtype=np.float32)
        normal_map = np.zeros((256, 256, 3), dtype=np.float32)
        confidence = np.ones((256, 256, 1), dtype=np.float32)

        comp = IntrinsicComponents(
            albedo=albedo,
            shading=shading,
            specular=specular,
            normal_map=normal_map,
            confidence=confidence,
            reconstruction_error=0.05,
        )

        report = DecompositionReport(
            frame_idx=0,
            components=comp,
            albedo_stability=0.9,
            shading_smoothness=0.8,
            specular_sparsity=0.95,
            decomposition_time_ms=10.0,
        )

        assert hasattr(report, "frame_idx")
        assert hasattr(report, "components")
        assert hasattr(report, "albedo_stability")
        assert hasattr(report, "shading_smoothness")
        assert hasattr(report, "specular_sparsity")
        assert hasattr(report, "decomposition_time_ms")


class TestRetinexDecomposition:
    """Test Retinex-inspired decomposition."""

    def test_log_domain_decomposition(self):
        """Retinex must work in log domain."""
        decomposer = IntrinsicDecomposer()
        image = np.random.rand(256, 256, 3).astype(np.float32) * 0.5 + 0.1
        result = decomposer.decompose(image)
        
        # Check that decomposition is valid
        assert result.albedo.shape == (256, 256, 3)
        assert result.shading.shape == (256, 256, 1)

    def test_illumination_smooth(self):
        """Illumination (shading) must be smooth."""
        decomposer = IntrinsicDecomposer()
        image = np.random.rand(256, 256, 3).astype(np.float32)
        result = decomposer.decompose(image)
        
        # Shading should be smooth
        from scipy.ndimage import laplace
        lap = laplace(result.shading)
        assert np.mean(np.abs(lap)) < 1.0

    def test_albedo_preserves_edges(self):
        """Albedo must preserve texture edges."""
        decomposer = IntrinsicDecomposer()
        
        # Create image with clear edge
        image = np.zeros((256, 256, 3), dtype=np.float32)
        image[:128, :, :] = 0.8
        image[128:, :, :] = 0.2
        
        result = decomposer.decompose(image)
        
        # Albedo should have edge at row 128
        edge_albedo = np.abs(result.albedo[127, :, 0] - result.albedo[128, :, 0])
        assert np.max(edge_albedo) > 0.1  # Edge preserved

    def test_multiple_calls_deterministic(self):
        """Multiple calls must be deterministic."""
        decomposer = IntrinsicDecomposer()
        image = np.random.rand(256, 256, 3).astype(np.float32)
        
        result1 = decomposer.decompose(image)
        result2 = decomposer.decompose(image)
        
        np.testing.assert_array_equal(result1.albedo, result2.albedo)
        np.testing.assert_array_equal(result1.shading, result2.shading)
