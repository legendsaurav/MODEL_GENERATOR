"""Unit tests for mesh smoothing algorithms."""

import pytest
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import trimesh
from MODEL_GENERATOR_V2.postprocessing.smoothing import MeshSmoother


@pytest.fixture
def smoother():
    return MeshSmoother()


@pytest.fixture
def noisy_icosphere():
    """Create an icosphere with added vertex noise."""
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    noise = np.random.normal(0, 0.02, mesh.vertices.shape)
    mesh.vertices += noise
    return mesh


class TestMeshSmoother:
    """Tests for MeshSmoother."""

    def test_init_defaults(self):
        s = MeshSmoother()
        assert s.default_method == "taubin"
        assert s.default_iterations == 5

    def test_init_custom(self):
        s = MeshSmoother(default_method="hc", default_iterations=10)
        assert s.default_method == "hc"
        assert s.default_iterations == 10

    def test_init_invalid_method(self):
        with pytest.raises(ValueError):
            MeshSmoother(default_method="invalid")

    def test_taubin_preserves_vertex_count(self, smoother, noisy_icosphere):
        result = smoother.taubin_smooth(noisy_icosphere, iterations=3)
        assert len(result.vertices) == len(noisy_icosphere.vertices)

    def test_taubin_reduces_noise(self, smoother, noisy_icosphere):
        """Smoothing should reduce vertex distance variance."""
        original_std = np.std(np.linalg.norm(noisy_icosphere.vertices, axis=1))
        result = smoother.taubin_smooth(noisy_icosphere, iterations=5)
        smoothed_std = np.std(np.linalg.norm(result.vertices, axis=1))
        # Smoothed mesh should have more uniform vertex distances
        assert smoothed_std <= original_std

    def test_hc_preserves_vertex_count(self, smoother, noisy_icosphere):
        result = smoother.hc_laplacian_smooth(noisy_icosphere, iterations=2)
        assert len(result.vertices) == len(noisy_icosphere.vertices)

    def test_callable_taubin(self, smoother, noisy_icosphere):
        result = smoother(noisy_icosphere, method="taubin", iterations=3)
        assert isinstance(result, trimesh.Trimesh)
        assert len(result.vertices) > 0

    def test_callable_hc(self, smoother, noisy_icosphere):
        result = smoother(noisy_icosphere, method="hc", iterations=2)
        assert isinstance(result, trimesh.Trimesh)

    def test_callable_default(self, smoother, noisy_icosphere):
        result = smoother(noisy_icosphere)
        assert isinstance(result, trimesh.Trimesh)

    def test_mesh_integrity(self, smoother, noisy_icosphere):
        """Smoothed mesh should still have valid faces."""
        result = smoother(noisy_icosphere, iterations=3)
        assert len(result.faces) == len(noisy_icosphere.faces)
