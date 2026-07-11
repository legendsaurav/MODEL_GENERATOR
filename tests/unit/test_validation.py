"""Unit tests for mesh validation."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import trimesh
from MODEL_GENERATOR_V2.postprocessing.validation import (
    MeshValidator,
    MeshQualityReport,
)


@pytest.fixture
def validator():
    return MeshValidator()


@pytest.fixture
def watertight_sphere():
    """A clean watertight icosphere."""
    return trimesh.creation.icosphere(subdivisions=3, radius=1.0)


@pytest.fixture
def watertight_box():
    """A clean watertight box."""
    return trimesh.creation.box(extents=(1, 1, 1))


class TestMeshValidator:
    """Tests for MeshValidator."""

    def test_watertight_sphere(self, validator, watertight_sphere):
        assert validator.check_watertight(watertight_sphere) is True

    def test_watertight_box(self, validator, watertight_box):
        assert validator.check_watertight(watertight_box) is True

    def test_manifold_sphere(self, validator, watertight_sphere):
        is_manifold, details = validator.check_manifold(watertight_sphere)
        assert is_manifold is True
        assert details["non_manifold_edges"] == 0

    def test_normals_consistent(self, validator, watertight_sphere):
        assert validator.check_normals_consistent(watertight_sphere) is True

    def test_degenerate_face_count_clean(self, validator, watertight_sphere):
        count = validator.count_degenerate_faces(watertight_sphere)
        assert count == 0

    def test_quality_metrics(self, validator, watertight_sphere):
        report = validator.compute_quality_metrics(watertight_sphere)
        assert isinstance(report, MeshQualityReport)
        assert report.vertex_count > 0
        assert report.face_count > 0
        assert report.is_watertight is True
        assert report.surface_area > 0
        assert report.volume > 0

    def test_quality_metrics_box(self, validator, watertight_box):
        report = validator.compute_quality_metrics(watertight_box)
        # Box volume should be ~1.0
        assert abs(report.volume - 1.0) < 0.01
        # Box surface area should be ~6.0
        assert abs(report.surface_area - 6.0) < 0.01

    def test_generate_report_string(self, validator, watertight_sphere):
        report_str = validator.generate_report(watertight_sphere)
        assert isinstance(report_str, str)
        assert "MESH QUALITY REPORT" in report_str
        assert "Vertices" in report_str
        assert "Faces" in report_str

    def test_callable_returns_dict(self, validator, watertight_sphere):
        result = validator(watertight_sphere)
        assert isinstance(result, dict)
        assert "vertex_count" in result
        assert "face_count" in result
        assert "is_watertight" in result
        assert "is_manifold" in result

    def test_euler_number_sphere(self, validator, watertight_sphere):
        report = validator.compute_quality_metrics(watertight_sphere)
        # Euler number of a sphere = 2
        assert report.euler_number == 2

    def test_bounding_box(self, validator, watertight_box):
        report = validator.compute_quality_metrics(watertight_box)
        # Box centered at origin, extents (1,1,1)
        for i in range(3):
            assert abs(report.bounding_box_min[i] - (-0.5)) < 0.01
            assert abs(report.bounding_box_max[i] - 0.5) < 0.01
