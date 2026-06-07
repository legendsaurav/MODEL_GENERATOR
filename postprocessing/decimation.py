"""
Quadric edge collapse decimation for MODEL_GENERATOR_V2.

Reduces mesh face count while preserving geometric quality using
the Quadric Error Metric (QEM) decimation algorithm.

Uses trimesh as the PRIMARY backend. Pymeshlab is used if available.

Dependencies:
    - trimesh
    - numpy
    - pymeshlab (optional)

Classes:
    QuadricDecimator: Reduces mesh complexity via QEM decimation.
"""

import logging
from typing import Optional

import numpy as np
import trimesh

logger = logging.getLogger("model_generator_v2.postprocessing.decimation")

# ── Pymeshlab availability check ───────────────────────────────────────────
_PYMESHLAB_AVAILABLE = False
try:
    import pymeshlab
    _test_ms = pymeshlab.MeshSet()
    _PYMESHLAB_AVAILABLE = True
    del _test_ms
except Exception:
    logger.info("pymeshlab not fully functional — using trimesh-only decimation")


class QuadricDecimator:
    """Mesh decimation using Quadric Edge Collapse.

    Reduces face count while minimizing geometric distortion.
    Uses pymeshlab when available, falls back to trimesh.

    Args:
        target_faces: Default target face count.
        quality_threshold: Quality threshold for edge collapses (0-1).
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

        # Try pymeshlab (best quality decimation)
        if _PYMESHLAB_AVAILABLE:
            try:
                ms = pymeshlab.MeshSet()
                pm = pymeshlab.Mesh(
                    vertex_matrix=mesh.vertices.astype(np.float64),
                    face_matrix=mesh.faces.astype(np.int32),
                )
                ms.add_mesh(pm)
                ms.apply_filter(
                    "meshing_decimation_quadric_edge_collapse",
                    targetfacenum=target,
                    qualitythr=quality,
                    preserveboundary=boundary,
                    preservenormal=self.preserve_normal,
                    preservetopology=topology,
                    autoclean=True,
                )
                result_mesh = ms.current_mesh()
                verts = result_mesh.vertex_matrix()
                faces = result_mesh.face_matrix()

                if len(verts) > 0 and len(faces) > 0:
                    result = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                    logger.info(
                        f"Decimated: {current_faces} → {len(result.faces)} faces "
                        f"(target: {target}, pymeshlab)"
                    )
                    return result
            except Exception as e:
                logger.debug(f"pymeshlab decimation failed: {e}")

        # Trimesh fallback: simplify_quadric_decimation
        try:
            result = mesh.simplify_quadric_decimation(target)
            logger.info(
                f"Decimated: {current_faces} → {len(result.faces)} faces "
                f"(target: {target}, trimesh)"
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
