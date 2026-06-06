"""
Quadric edge collapse decimation for MODEL_GENERATOR_V2.

Reduces mesh face count while preserving geometric quality using
the Quadric Error Metric (QEM) decimation algorithm.

Dependencies:
    - trimesh
    - pymeshlab

Classes:
    QuadricDecimator: Reduces mesh complexity via QEM decimation.
"""

import logging
import tempfile
from typing import Optional

import trimesh

logger = logging.getLogger("model_generator_v2.postprocessing.decimation")


def _trimesh_to_meshset(mesh: trimesh.Trimesh):
    """Convert trimesh → pymeshlab MeshSet."""
    import pymeshlab
    ms = pymeshlab.MeshSet()
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        mesh.export(f.name)
        ms.load_new_mesh(f.name)
    return ms


def _meshset_to_trimesh(ms) -> trimesh.Trimesh:
    """Convert pymeshlab MeshSet → trimesh."""
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        ms.save_current_mesh(f.name)
        return trimesh.load(f.name, process=False)


class QuadricDecimator:
    """Mesh decimation using Quadric Edge Collapse.

    Reduces face count while minimizing geometric distortion
    by collapsing edges with the lowest quadric error metric.
    Preserves boundary edges, normals, and topology.

    Args:
        target_faces: Default target face count.
        quality_threshold: Quality threshold for edge collapses (0-1).
            Higher = better quality but less aggressive reduction.
        preserve_boundary: Keep boundary edges intact.
        preserve_normal: Prevent normal flips during collapse.
        preserve_topology: Prevent topological changes.

    Example:
        >>> decimator = QuadricDecimator(target_faces=100000)
        >>> reduced_mesh = decimator(high_poly_mesh)
    """

    def __init__(
        self,
        target_faces: int = 100000,
        quality_threshold: float = 1.0,
        preserve_boundary: bool = True,
        preserve_normal: bool = True,
        preserve_topology: bool = True,
    ) -> None:
        self.target_faces = target_faces
        self.quality_threshold = quality_threshold
        self.preserve_boundary = preserve_boundary
        self.preserve_normal = preserve_normal
        self.preserve_topology = preserve_topology

    def decimate(
        self,
        mesh: trimesh.Trimesh,
        target_faces: Optional[int] = None,
        quality_threshold: Optional[float] = None,
        preserve_boundary: Optional[bool] = None,
        preserve_topology: Optional[bool] = None,
    ) -> trimesh.Trimesh:
        """Decimate mesh to a target face count.

        Args:
            mesh: Input mesh.
            target_faces: Target number of faces.
            quality_threshold: Edge collapse quality threshold.
            preserve_boundary: Whether to preserve boundaries.
            preserve_topology: Whether to preserve topology.

        Returns:
            Decimated mesh. Returns original if already below target.
        """
        target = target_faces or self.target_faces
        quality = quality_threshold or self.quality_threshold
        boundary = (
            preserve_boundary if preserve_boundary is not None
            else self.preserve_boundary
        )
        topology = (
            preserve_topology if preserve_topology is not None
            else self.preserve_topology
        )

        current_faces = len(mesh.faces)
        if current_faces <= target:
            logger.info(
                f"Mesh already has {current_faces} faces "
                f"(target: {target}), skipping decimation"
            )
            return mesh

        try:
            ms = _trimesh_to_meshset(mesh)
            ms.apply_filter(
                "meshing_decimation_quadric_edge_collapse",
                targetfacenum=target,
                qualitythr=quality,
                preserveboundary=boundary,
                preservenormal=self.preserve_normal,
                preservetopology=topology,
                autoclean=True,
            )
            result = _meshset_to_trimesh(ms)
            logger.info(
                f"Decimated: {current_faces} → {len(result.faces)} faces "
                f"(target: {target})"
            )
            return result
        except Exception as e:
            logger.warning(f"Decimation failed: {e}")
            return mesh

    def adaptive_decimate(
        self,
        mesh: trimesh.Trimesh,
        target_ratio: float = 0.5,
    ) -> trimesh.Trimesh:
        """Decimate to a fraction of current face count.

        Args:
            mesh: Input mesh.
            target_ratio: Fraction of faces to keep (0-1).

        Returns:
            Decimated mesh.

        Raises:
            ValueError: If ratio is not in (0, 1].
        """
        if not 0 < target_ratio <= 1:
            raise ValueError(f"target_ratio must be in (0, 1], got {target_ratio}")

        target = max(100, int(len(mesh.faces) * target_ratio))
        return self.decimate(mesh, target_faces=target)

    def __call__(
        self,
        mesh: trimesh.Trimesh,
        target_faces: Optional[int] = None,
    ) -> trimesh.Trimesh:
        """Decimate mesh to target face count.

        Args:
            mesh: Input mesh.
            target_faces: Target face count (uses default if None).

        Returns:
            Decimated mesh.
        """
        return self.decimate(mesh, target_faces=target_faces)
