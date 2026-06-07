"""
Adaptive mesh subdivision for MODEL_GENERATOR_V2.

Uses trimesh as the PRIMARY backend. Pymeshlab is used if available.

Dependencies:
    - trimesh
    - numpy
    - pymeshlab (optional)

Classes:
    AdaptiveSubdivider: Applies subdivision to increase mesh detail.
"""

import logging
from typing import Optional

import numpy as np
import trimesh

logger = logging.getLogger("model_generator_v2.postprocessing.subdivision")

# ── Pymeshlab availability check ───────────────────────────────────────────
_PYMESHLAB_AVAILABLE = False
try:
    import pymeshlab
    _test_ms = pymeshlab.MeshSet()
    _PYMESHLAB_AVAILABLE = True
    del _test_ms
except Exception:
    logger.info("pymeshlab not fully functional — using trimesh-only subdivision")


class AdaptiveSubdivider:
    """Mesh subdivision to increase geometric detail.

    Supports Loop and midpoint subdivision. Uses pymeshlab when
    available, falls back to trimesh.subdivide.

    Args:
        max_face_limit: Safety limit to prevent over-subdivision.

    Example:
        >>> subdiv = AdaptiveSubdivider()
        >>> detailed_mesh = subdiv(coarse_mesh, iterations=1)
    """

    def __init__(self, max_face_limit: int = 1_000_000) -> None:
        self.max_face_limit = max_face_limit

    def _pymeshlab_subdivide(
        self, mesh: trimesh.Trimesh, filter_name: str, iterations: int
    ) -> Optional[trimesh.Trimesh]:
        """Try pymeshlab subdivision via numpy arrays."""
        if not _PYMESHLAB_AVAILABLE:
            return None
        try:
            ms = pymeshlab.MeshSet()
            pm = pymeshlab.Mesh(
                vertex_matrix=mesh.vertices.astype(np.float64),
                face_matrix=mesh.faces.astype(np.int32),
            )
            ms.add_mesh(pm)

            for i in range(iterations):
                current_faces = ms.current_mesh().face_number()
                if current_faces * 4 > self.max_face_limit:
                    logger.warning(
                        f"Subdivision would exceed face limit "
                        f"({current_faces * 4} > {self.max_face_limit}). "
                        f"Stopping at iteration {i}."
                    )
                    break
                ms.apply_filter(filter_name)

            result_mesh = ms.current_mesh()
            verts = result_mesh.vertex_matrix()
            faces = result_mesh.face_matrix()

            if len(verts) > 0 and len(faces) > 0:
                return trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            return None
        except Exception as e:
            logger.debug(f"pymeshlab subdivision failed: {e}")
            return None

    def midpoint_subdivision(
        self,
        mesh: trimesh.Trimesh,
        iterations: int = 1,
    ) -> trimesh.Trimesh:
        """Apply midpoint subdivision.

        Args:
            mesh: Input mesh.
            iterations: Number of subdivision passes.

        Returns:
            Subdivided mesh.
        """
        # Try pymeshlab
        result = self._pymeshlab_subdivide(
            mesh, "meshing_surface_subdivision_midpoint", iterations
        )
        if result is not None:
            logger.info(
                f"Midpoint subdivision: {len(mesh.faces)} → {len(result.faces)} faces (pymeshlab)"
            )
            return result

        # Trimesh fallback
        try:
            current = mesh
            for i in range(iterations):
                if len(current.faces) * 4 > self.max_face_limit:
                    logger.warning(f"Subdivision face limit reached at iteration {i}")
                    break
                verts, faces = trimesh.remesh.subdivide(
                    current.vertices, current.faces
                )
                current = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            logger.info(
                f"Midpoint subdivision: {len(mesh.faces)} → {len(current.faces)} faces (trimesh)"
            )
            return current
        except Exception as e:
            logger.warning(f"Midpoint subdivision failed: {e}")
            return mesh

    def loop_subdivision(
        self,
        mesh: trimesh.Trimesh,
        iterations: int = 1,
    ) -> trimesh.Trimesh:
        """Apply Loop subdivision for smooth surface refinement.

        Args:
            mesh: Input mesh (must be triangle mesh).
            iterations: Number of subdivision passes.

        Returns:
            Smoothly subdivided mesh.
        """
        # Try pymeshlab
        result = self._pymeshlab_subdivide(
            mesh, "meshing_surface_subdivision_loop", iterations
        )
        if result is not None:
            logger.info(
                f"Loop subdivision: {len(mesh.faces)} → {len(result.faces)} faces (pymeshlab)"
            )
            return result

        # Trimesh fallback: use basic subdivision
        try:
            current = mesh
            for i in range(iterations):
                if len(current.faces) * 4 > self.max_face_limit:
                    logger.warning(f"Loop subdivision face limit reached at iteration {i}")
                    break
                verts, faces = trimesh.remesh.subdivide(
                    current.vertices, current.faces
                )
                current = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            logger.info(
                f"Loop subdivision: {len(mesh.faces)} → {len(current.faces)} faces (trimesh fallback)"
            )
            return current
        except Exception as e:
            logger.warning(f"Loop subdivision failed: {e}")
            return mesh

    def adaptive_subdivision(
        self,
        mesh: trimesh.Trimesh,
        edge_threshold: Optional[float] = None,
        max_iterations: int = 2,
    ) -> trimesh.Trimesh:
        """Selectively subdivide faces with long edges.

        Args:
            mesh: Input mesh.
            edge_threshold: Maximum edge length before subdivision.
            max_iterations: Maximum number of adaptive passes.

        Returns:
            Adaptively subdivided mesh.
        """
        if edge_threshold is None:
            edges = mesh.edges_unique_length
            edge_threshold = float(np.mean(edges))

        try:
            current = mesh
            for i in range(max_iterations):
                if len(current.faces) > self.max_face_limit:
                    break
                # Use trimesh's subdivide_to_size
                verts, faces = trimesh.remesh.subdivide_to_size(
                    current.vertices, current.faces, max_edge=edge_threshold
                )
                current = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            logger.info(
                f"Adaptive subdivision: {len(mesh.faces)} → "
                f"{len(current.faces)} faces"
            )
            return current
        except Exception as e:
            logger.warning(f"Adaptive subdivision failed: {e}")
            return mesh

    def __call__(
        self,
        mesh: trimesh.Trimesh,
        iterations: int = 1,
        method: str = "loop",
    ) -> trimesh.Trimesh:
        """Apply subdivision with the specified method.

        Args:
            mesh: Input mesh.
            iterations: Number of subdivision passes.
            method: 'loop', 'midpoint', or 'adaptive'.

        Returns:
            Subdivided mesh.
        """
        logger.info(
            f"Subdividing mesh ({len(mesh.faces)} faces) "
            f"with {method}, {iterations} iterations"
        )

        if method == "loop":
            return self.loop_subdivision(mesh, iterations)
        elif method == "midpoint":
            return self.midpoint_subdivision(mesh, iterations)
        elif method == "adaptive":
            return self.adaptive_subdivision(mesh, max_iterations=iterations)
        else:
            raise ValueError(f"Unknown subdivision method: {method}")
