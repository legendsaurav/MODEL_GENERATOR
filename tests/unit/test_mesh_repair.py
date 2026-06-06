"""Unit tests for mesh repair operations."""

import pytest
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import trimesh
from MODEL_GENERATOR_V2.postprocessing.mesh_repair import MeshRepairer


@pytest.fixture
def repairer():
    return MeshRepairer()


@pytest.fixture
def clean_icosphere():
    """A clean icosphere mesh for baseline testing."""
    return trimesh.creation.icosphere(subdivisions=3, radius=1.0)


@pytest.fixture
def clean_box():
    """A clean box mesh."""
    return trimesh.creation.box(extents=(1, 1, 1))


@pytest.fixture
def mesh_with_degenerate_faces():
    """Create a mesh with some degenerate (zero-area) faces."""
    mesh = trimesh.creation.icosphere(subdivisions=2)
    verts = mesh.vertices.copy()
    faces = mesh.faces.copy()
    # Add a degenerate face (all same vertex)
    degen_face = np.array([[0, 0, 0]])
    faces = np.vstack([faces, degen_face])
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


class TestMeshRepairer:
    """Tests for MeshRepairer operations."""

    def test_init_defaults(self):
        repairer = MeshRepairer()
        assert repairer.min_component_face_ratio == 0.005
        assert repairer.max_hole_edges == 20

    def test_repair_clean_mesh(self, repairer, clean_icosphere):
        """Repair on a clean mesh should not break it."""
        result = repairer(clean_icosphere)
        assert len(result.vertices) > 0
        assert len(result.faces) > 0

    def test_fix_normals(self, repairer, clean_icosphere):
        result = repairer.fix_normals(clean_icosphere)
        assert len(result.vertices) == len(clean_icosphere.vertices)

    def test_remove_isolated_components(self, repairer):
        """Create a mesh with a small floating piece and remove it."""
        # Main mesh
        main = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        # Small floater far away
        floater = trimesh.creation.icosphere(subdivisions=0, radius=0.01)
        floater.apply_translation([10, 10, 10])
        # Combine
        combined = trimesh.util.concatenate([main, floater])
        result = repairer.remove_isolated_components(combined)
        # Should have fewer faces (floater removed)
        assert len(result.faces) < len(combined.faces)

    def test_full_pipeline(self, repairer, clean_box):
        """Full repair pipeline on a box mesh."""
        result = repairer(clean_box)
        assert isinstance(result, trimesh.Trimesh)
        assert len(result.vertices) > 0
        assert len(result.faces) > 0

    def test_custom_parameters(self):
        repairer = MeshRepairer(
            min_component_face_ratio=0.01,
            max_hole_edges=50,
        )
        assert repairer.min_component_face_ratio == 0.01
        assert repairer.max_hole_edges == 50
