"""Integration tests for the PostProcessingPipeline."""

import pytest
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import trimesh
from MODEL_GENERATOR_V2.configs.base_config import PostProcessingConfig
from MODEL_GENERATOR_V2.postprocessing.pipeline import PostProcessingPipeline


@pytest.fixture
def noisy_sphere():
    """Create a noisy icosphere simulating raw generation output."""
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    noise = np.random.normal(0, 0.02, mesh.vertices.shape)
    mesh.vertices += noise
    return mesh


@pytest.fixture
def high_poly_sphere():
    """High-polygon sphere for decimation testing."""
    return trimesh.creation.icosphere(subdivisions=5, radius=1.0)


class TestPostProcessingPipeline:
    """Integration tests for the full post-processing pipeline."""

    def test_default_pipeline(self, noisy_sphere):
        """Run pipeline with default config."""
        pipeline = PostProcessingPipeline()
        result = pipeline(noisy_sphere, verbose=False)
        assert isinstance(result, trimesh.Trimesh)
        assert len(result.vertices) > 0
        assert len(result.faces) > 0

    def test_fast_preset(self, noisy_sphere):
        """Run with FAST preset parameters."""
        config = PostProcessingConfig(
            enable_repair=True,
            enable_smoothing=True,
            smoothing_method="taubin",
            smoothing_iterations=3,
            enable_subdivision=False,
            enable_decimation=True,
            target_faces=50000,
        )
        pipeline = PostProcessingPipeline(config)
        result = pipeline(noisy_sphere, verbose=False)
        assert isinstance(result, trimesh.Trimesh)

    def test_ultra_preset(self, noisy_sphere):
        """Run with ULTRA preset parameters."""
        config = PostProcessingConfig(
            enable_repair=True,
            enable_smoothing=True,
            smoothing_method="taubin",
            smoothing_iterations=10,
            enable_subdivision=False,  # Skip to avoid huge meshes
            enable_decimation=True,
            target_faces=200000,
        )
        pipeline = PostProcessingPipeline(config)
        result = pipeline(noisy_sphere, verbose=False)
        assert isinstance(result, trimesh.Trimesh)

    def test_no_postprocessing(self, noisy_sphere):
        """All steps disabled — output should equal input."""
        config = PostProcessingConfig(
            enable_repair=False,
            enable_smoothing=False,
            enable_subdivision=False,
            enable_decimation=False,
            enable_validation=False,
        )
        pipeline = PostProcessingPipeline(config)
        result = pipeline(noisy_sphere, verbose=False)
        assert len(result.vertices) == len(noisy_sphere.vertices)

    def test_decimation_reduces_faces(self, high_poly_sphere):
        """Decimation should reduce face count."""
        config = PostProcessingConfig(
            enable_repair=False,
            enable_smoothing=False,
            enable_subdivision=False,
            enable_decimation=True,
            target_faces=5000,
            enable_validation=False,
        )
        pipeline = PostProcessingPipeline(config)
        result = pipeline(high_poly_sphere, verbose=False)
        assert len(result.faces) < len(high_poly_sphere.faces)

    def test_smoothing_only(self, noisy_sphere):
        """Run only smoothing."""
        config = PostProcessingConfig(
            enable_repair=False,
            enable_smoothing=True,
            smoothing_iterations=5,
            enable_subdivision=False,
            enable_decimation=False,
            enable_validation=False,
        )
        pipeline = PostProcessingPipeline(config)
        result = pipeline(noisy_sphere, verbose=False)
        # Vertex count should be preserved
        assert len(result.vertices) == len(noisy_sphere.vertices)
