"""Integration tests for the repair → smooth → decimate chain."""

import pytest
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import trimesh
from MODEL_GENERATOR_V2.postprocessing.mesh_repair import MeshRepairer
from MODEL_GENERATOR_V2.postprocessing.smoothing import MeshSmoother
from MODEL_GENERATOR_V2.postprocessing.decimation import QuadricDecimator
from MODEL_GENERATOR_V2.postprocessing.validation import MeshValidator


@pytest.fixture
def messy_mesh():
    """Create a mesh with noise and extra disconnected components."""
    # Main body
    main = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    noise = np.random.normal(0, 0.03, main.vertices.shape)
    main.vertices += noise
    # Small floater
    floater = trimesh.creation.icosphere(subdivisions=0, radius=0.02)
    floater.apply_translation([5, 5, 5])
    return trimesh.util.concatenate([main, floater])


class TestRepairSmoothDecimateChain:
    """Test the sequential repair → smooth → decimate pipeline."""

    def test_full_chain(self, messy_mesh):
        original_faces = len(messy_mesh.faces)

        # Step 1: Repair
        repairer = MeshRepairer()
        repaired = repairer(messy_mesh)
        # Floater should be removed
        assert len(repaired.faces) < original_faces

        # Step 2: Smooth
        smoother = MeshSmoother()
        smoothed = smoother(repaired, method="taubin", iterations=3)
        assert len(smoothed.vertices) == len(repaired.vertices)

        # Step 3: Decimate
        decimator = QuadricDecimator(target_faces=5000)
        decimated = decimator(smoothed)
        assert len(decimated.faces) <= 5000

        # Step 4: Validate
        validator = MeshValidator()
        metrics = validator(decimated)
        assert metrics["vertex_count"] > 0
        assert metrics["face_count"] > 0

    def test_quality_improves(self, messy_mesh):
        """Verify mesh quality metrics improve through the chain."""
        validator = MeshValidator()

        before_metrics = validator(messy_mesh)

        repairer = MeshRepairer()
        repaired = repairer(messy_mesh)

        smoother = MeshSmoother()
        smoothed = smoother(repaired, iterations=5)

        after_metrics = validator(smoothed)

        # Degenerate faces should not increase
        assert (
            after_metrics["degenerate_face_count"]
            <= before_metrics["degenerate_face_count"]
        )
