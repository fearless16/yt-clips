import numpy as np

from face_os.mesh_mask import (
    build_semantic_mesh_mask,
    default_triangles,
    mask_from_sdf,
)
from face_os.types import SemanticMeshMask


def _square_mesh() -> np.ndarray:
    return np.array(
        [
            [16.0, 16.0, 0.0],
            [48.0, 16.0, 0.0],
            [48.0, 48.0, 0.0],
            [16.0, 48.0, 0.0],
        ],
        dtype=np.float32,
    )


def _synthetic_face_mesh(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = 478
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radius = np.sqrt(rng.uniform(0.05, 1.0, n))
    x = 64.0 + 34.0 * radius * np.cos(theta)
    y = 64.0 + 46.0 * radius * np.sin(theta)
    z = 4.0 * np.cos(theta)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def test_mask_is_lighting_invariant_by_construction():
    mesh = _square_mesh()
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    dark_frame = np.zeros((64, 64, 3), dtype=np.uint8)
    bright_frame = np.full((64, 64, 3), 255, dtype=np.uint8)
    assert dark_frame.mean() != bright_frame.mean()

    dark_mask = build_semantic_mesh_mask(mesh, dark_frame.shape, triangles=triangles)
    bright_mask = build_semantic_mesh_mask(mesh, bright_frame.shape, triangles=triangles)

    np.testing.assert_array_equal(dark_mask.mask, bright_mask.mask)
    np.testing.assert_array_equal(dark_mask.sdf, bright_mask.sdf)
    assert dark_mask.coverage == bright_mask.coverage


def test_sdf_edge_is_finite_signed_and_feathered_without_blur():
    mesh = _square_mesh()
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    result = build_semantic_mesh_mask(mesh, (64, 64), triangles=triangles, feather_px=4.0)

    assert isinstance(result, SemanticMeshMask)
    assert result.sdf is not None
    assert np.all(np.isfinite(result.sdf))
    assert np.all(np.isfinite(result.mask))
    assert result.sdf[32, 32] > 0.0
    assert result.sdf[4, 4] < 0.0
    assert result.mask[32, 32] == 1.0
    assert result.mask[4, 4] == 0.0
    assert 0.0 < result.mask[15, 32] < 1.0

    reconstructed = mask_from_sdf(result.sdf, feather_px=4.0)
    np.testing.assert_array_equal(result.mask, reconstructed)


def test_supplied_triangle_inversion_is_reported_but_still_rasterized():
    mesh = _square_mesh()
    triangles = np.array([[0, 2, 1], [0, 2, 3]], dtype=np.int32)

    result = build_semantic_mesh_mask(mesh, (64, 64), triangles=triangles, feather_px=0.0)

    assert result.triangle_count == 2
    assert result.inverted_triangles == 1
    assert result.mask[32, 32] == 1.0
    assert result.source.endswith("supplied_triangles")


def test_mask_bounds_coverage_regions_and_no_nan():
    mesh = _square_mesh()
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    result = build_semantic_mesh_mask(mesh, (64, 64, 3), triangles=triangles, feather_px=3.0)

    assert result.mask.shape == (64, 64)
    assert result.mask.dtype == np.float32
    assert float(np.min(result.mask)) >= 0.0
    assert float(np.max(result.mask)) <= 1.0
    assert 0.20 < result.coverage < 0.35
    assert set(result.regions) >= {"face", "skin"}
    for region in result.regions.values():
        assert region.shape == (64, 64)
        assert region.dtype == np.float32
        assert np.all(np.isfinite(region))


def test_default_mesh_output_is_deterministic():
    mesh = _synthetic_face_mesh()

    first = build_semantic_mesh_mask(mesh, (128, 128), feather_px=5.0)
    second = build_semantic_mesh_mask(mesh, (128, 128), feather_px=5.0)

    assert first.triangle_count == second.triangle_count
    assert first.inverted_triangles == second.inverted_triangles
    assert first.source == second.source
    np.testing.assert_array_equal(first.mask, second.mask)
    np.testing.assert_array_equal(first.sdf, second.sdf)
    assert np.isfinite(first.coverage)
    assert 0.0 < first.coverage < 1.0
    assert "mediapipe" in first.source or "fallback_convex_hull_fan" in first.source


def test_default_fallback_source_is_explicit(monkeypatch):
    import face_os.mesh_mask as mesh_mask

    monkeypatch.setattr(
        mesh_mask,
        "_mediapipe_tessellation_triangles",
        lambda: (np.empty((0, 3), dtype=np.int32), "fallback_convex_hull_fan"),
    )

    mesh = _synthetic_face_mesh()
    triangles, source, normalize_winding = default_triangles(mesh)
    result = build_semantic_mesh_mask(mesh, (128, 128), feather_px=0.0)

    assert len(triangles) >= 3
    assert source == "fallback_convex_hull_fan"
    assert normalize_winding is True
    assert "fallback_convex_hull_fan" in result.source
    assert result.triangle_count == len(triangles)

