"""Unit tests for mesh exporters."""

import os
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import trimesh
from MODEL_GENERATOR_V2.exporters import (
    get_exporter,
    GLBExporter,
    OBJExporter,
    STLExporter,
    PLYExporter,
)


@pytest.fixture
def sample_mesh():
    """A simple icosphere for export testing."""
    return trimesh.creation.icosphere(subdivisions=2, radius=1.0)


@pytest.fixture
def export_dir(tmp_path):
    """Temporary directory for export outputs."""
    return str(tmp_path)


class TestGetExporter:
    """Tests for the get_exporter factory function."""

    def test_glb(self):
        exp = get_exporter("glb")
        assert isinstance(exp, GLBExporter)

    def test_obj(self):
        exp = get_exporter("obj")
        assert isinstance(exp, OBJExporter)

    def test_stl(self):
        exp = get_exporter("stl")
        assert isinstance(exp, STLExporter)

    def test_ply(self):
        exp = get_exporter("ply")
        assert isinstance(exp, PLYExporter)

    def test_gltf_alias(self):
        exp = get_exporter("gltf")
        assert isinstance(exp, GLBExporter)

    def test_case_insensitive(self):
        exp = get_exporter("GLB")
        assert isinstance(exp, GLBExporter)

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="Unsupported format"):
            get_exporter("fbx")


class TestGLBExporter:
    """Tests for GLB export."""

    def test_export_creates_file(self, sample_mesh, export_dir):
        exp = GLBExporter()
        path = os.path.join(export_dir, "test.glb")
        result = exp.export(sample_mesh, path)
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0

    def test_auto_extension(self, sample_mesh, export_dir):
        exp = GLBExporter()
        path = os.path.join(export_dir, "test")
        result = exp.export(sample_mesh, path)
        assert result.endswith(".glb")


class TestOBJExporter:
    """Tests for OBJ export."""

    def test_export_creates_file(self, sample_mesh, export_dir):
        exp = OBJExporter()
        path = os.path.join(export_dir, "test.obj")
        result = exp.export(sample_mesh, path)
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0

    def test_obj_contains_vertices(self, sample_mesh, export_dir):
        exp = OBJExporter()
        path = os.path.join(export_dir, "test.obj")
        exp.export(sample_mesh, path)
        with open(path, "r") as f:
            content = f.read()
        assert "v " in content
        assert "f " in content


class TestSTLExporter:
    """Tests for STL export."""

    def test_export_binary(self, sample_mesh, export_dir):
        exp = STLExporter()
        path = os.path.join(export_dir, "test.stl")
        result = exp.export(sample_mesh, path, binary=True)
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0

    def test_export_creates_file(self, sample_mesh, export_dir):
        exp = STLExporter()
        path = os.path.join(export_dir, "test.stl")
        result = exp.export(sample_mesh, path)
        assert os.path.exists(result)


class TestPLYExporter:
    """Tests for PLY export."""

    def test_export_binary(self, sample_mesh, export_dir):
        exp = PLYExporter()
        path = os.path.join(export_dir, "test.ply")
        result = exp.export(sample_mesh, path, binary=True)
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0

    def test_export_creates_file(self, sample_mesh, export_dir):
        exp = PLYExporter()
        path = os.path.join(export_dir, "test.ply")
        result = exp.export(sample_mesh, path)
        assert os.path.exists(result)


class TestExporterValidation:
    """Tests for exporter input validation."""

    def test_none_mesh_raises(self, export_dir):
        exp = GLBExporter()
        with pytest.raises(ValueError, match="Invalid mesh"):
            exp.export(None, os.path.join(export_dir, "test.glb"))

    def test_empty_mesh_raises(self, export_dir):
        empty = trimesh.Trimesh()
        exp = GLBExporter()
        with pytest.raises(ValueError, match="Invalid mesh"):
            exp.export(empty, os.path.join(export_dir, "test.glb"))
