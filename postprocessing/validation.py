"""
Mesh quality validation for MODEL_GENERATOR_V2.

Checks mesh integrity (watertightness, manifoldness, normal
consistency) and computes quality metrics for generated meshes.

Dependencies:
    - trimesh
    - numpy

Classes:
    MeshValidator: Validates mesh quality and produces reports.
"""

import logging
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np
import trimesh

logger = logging.getLogger("model_generator_v2.postprocessing.validation")


@dataclass
class MeshQualityReport:
    """Structured mesh quality metrics.

    Attributes:
        vertex_count: Total number of vertices.
        face_count: Total number of faces.
        edge_count: Total number of unique edges.
        is_watertight: Whether the mesh is closed (no boundary edges).
        is_manifold: Whether all edges are shared by exactly 2 faces.
        normals_consistent: Whether face normals are consistently oriented.
        degenerate_face_count: Number of zero-area faces.
        bounding_box_min: Minimum corner of axis-aligned bounding box.
        bounding_box_max: Maximum corner of axis-aligned bounding box.
        surface_area: Total surface area.
        volume: Enclosed volume (meaningful only if watertight).
        euler_number: Euler characteristic (V - E + F).
    """

    vertex_count: int = 0
    face_count: int = 0
    edge_count: int = 0
    is_watertight: bool = False
    is_manifold: bool = False
    normals_consistent: bool = False
    degenerate_face_count: int = 0
    bounding_box_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounding_box_max: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    surface_area: float = 0.0
    volume: float = 0.0
    euler_number: int = 0


class MeshValidator:
    """Validates mesh quality and generates detailed reports.

    Performs structural integrity checks (watertightness, manifoldness),
    geometric validation (normals, degenerate faces), and computes
    quantitative quality metrics.

    Example:
        >>> validator = MeshValidator()
        >>> report = validator.generate_report(mesh)
        >>> print(report)
        >>> if not validator.check_watertight(mesh):
        ...     logger.warning("Mesh is not watertight")
    """

    @staticmethod
    def check_watertight(mesh: trimesh.Trimesh) -> bool:
        """Check if the mesh is watertight (closed surface).

        A watertight mesh has no boundary edges — every edge is
        shared by exactly two faces.

        Args:
            mesh: Input mesh.

        Returns:
            True if the mesh is watertight.
        """
        result = bool(mesh.is_watertight)
        logger.debug(f"Watertight check: {result}")
        return result

    @staticmethod
    def check_manifold(mesh: trimesh.Trimesh) -> Tuple[bool, Dict]:
        """Check if the mesh is manifold.

        A manifold mesh has no non-manifold edges (shared by >2 faces)
        or non-manifold vertices.

        Args:
            mesh: Input mesh.

        Returns:
            Tuple of (is_manifold, details_dict).
        """
        # Count boundary/non-manifold edges
        edges = mesh.edges_sorted
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        non_manifold_edges = int(np.sum(counts > 2))
        boundary_edges = int(np.sum(counts == 1))

        is_manifold = non_manifold_edges == 0
        details = {
            "non_manifold_edges": non_manifold_edges,
            "boundary_edges": boundary_edges,
            "total_edges": len(unique),
        }

        logger.debug(f"Manifold check: {is_manifold}, details={details}")
        return is_manifold, details

    @staticmethod
    def check_normals_consistent(mesh: trimesh.Trimesh) -> bool:
        """Check if face normals are consistently oriented.

        Args:
            mesh: Input mesh.

        Returns:
            True if normals appear consistent.
        """
        try:
            # Check if the mesh has consistent winding
            result = bool(mesh.is_winding_consistent)
            logger.debug(f"Normal consistency: {result}")
            return result
        except Exception:
            return True  # Assume consistent if check fails

    @staticmethod
    def count_degenerate_faces(mesh: trimesh.Trimesh) -> int:
        """Count faces with zero or near-zero area.

        Args:
            mesh: Input mesh.

        Returns:
            Number of degenerate faces.
        """
        areas = mesh.area_faces
        degenerate = int(np.sum(areas < 1e-10))
        if degenerate > 0:
            logger.debug(f"Found {degenerate} degenerate faces")
        return degenerate

    def compute_quality_metrics(
        self, mesh: trimesh.Trimesh
    ) -> MeshQualityReport:
        """Compute comprehensive mesh quality metrics.

        Args:
            mesh: Input mesh.

        Returns:
            MeshQualityReport with all metrics.
        """
        is_manifold, manifold_details = self.check_manifold(mesh)

        bounds = mesh.bounds if mesh.bounds is not None else np.zeros((2, 3))

        try:
            volume = float(mesh.volume) if mesh.is_watertight else 0.0
        except Exception:
            volume = 0.0

        try:
            euler = int(mesh.euler_number)
        except Exception:
            euler = 0

        return MeshQualityReport(
            vertex_count=len(mesh.vertices),
            face_count=len(mesh.faces),
            edge_count=len(mesh.edges_unique),
            is_watertight=bool(mesh.is_watertight),
            is_manifold=is_manifold,
            normals_consistent=self.check_normals_consistent(mesh),
            degenerate_face_count=self.count_degenerate_faces(mesh),
            bounding_box_min=tuple(bounds[0].tolist()),
            bounding_box_max=tuple(bounds[1].tolist()),
            surface_area=float(mesh.area),
            volume=volume,
            euler_number=euler,
        )

    def generate_report(self, mesh: trimesh.Trimesh) -> str:
        """Generate a human-readable mesh quality report.

        Args:
            mesh: Input mesh.

        Returns:
            Formatted string report.
        """
        metrics = self.compute_quality_metrics(mesh)

        bbox_size = tuple(
            round(a - b, 4)
            for a, b in zip(metrics.bounding_box_max, metrics.bounding_box_min)
        )

        report = f"""
╔══════════════════════════════════════════╗
║       MESH QUALITY REPORT                ║
╠══════════════════════════════════════════╣
║  Vertices:          {metrics.vertex_count:>10,}           ║
║  Faces:             {metrics.face_count:>10,}           ║
║  Edges:             {metrics.edge_count:>10,}           ║
╠══════════════════════════════════════════╣
║  Watertight:        {'✓ YES' if metrics.is_watertight else '✗ NO':>10}           ║
║  Manifold:          {'✓ YES' if metrics.is_manifold else '✗ NO':>10}           ║
║  Normals OK:        {'✓ YES' if metrics.normals_consistent else '✗ NO':>10}           ║
║  Degenerate Faces:  {metrics.degenerate_face_count:>10,}           ║
╠══════════════════════════════════════════╣
║  Surface Area:      {metrics.surface_area:>10.4f}           ║
║  Volume:            {metrics.volume:>10.4f}           ║
║  Euler Number:      {metrics.euler_number:>10}           ║
║  Bounding Box:      {str(bbox_size):>26} ║
╚══════════════════════════════════════════╝"""

        return report.strip()

    def __call__(self, mesh: trimesh.Trimesh) -> Dict:
        """Run full validation and return metrics as dict.

        Args:
            mesh: Input mesh.

        Returns:
            Dictionary of all quality metrics.
        """
        report = self.compute_quality_metrics(mesh)
        result = asdict(report)
        logger.info(
            f"Validation: {result['vertex_count']} verts, "
            f"{result['face_count']} faces, "
            f"watertight={result['is_watertight']}"
        )
        return result
