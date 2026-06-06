"""
Adaptive mesh subdivision for MODEL_GENERATOR_V2.

Increases mesh resolution by subdividing faces using Loop or
midpoint subdivision schemes. Adaptive mode only subdivides
faces with edges longer than a threshold.

Dependencies:
    - trimesh
    - pymeshlab

Classes:
    AdaptiveSubdivider: Applies subdivision to increase mesh detail.
"""

import logging
import tempfile
from typing import Optional

import numpy as np
import trimesh

logger = logging.getLogger("model_generator_v2.postprocessing.subdivision")


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


class AdaptiveSubdivider:
    """Mesh subdivision to increase geometric detail.

    Supports three subdivision schemes:
    - **Loop**: Smooth subdivision that converges to C2 surfaces.
      Produces high-quality results but approximately 4× faces per step.
    - **Midpoint**: Simple midpoint splitting. Faster but less smooth.
    - **Adaptive**: Only subdivides edges exceeding a length threshold,
      adding detail only where needed.

    Args:
        max_face_limit: Safety limit to prevent over-subdivision.

    Example:
        >>> subdiv = AdaptiveSubdivider()
        >>> detailed_mesh = subdiv(coarse_mesh, iterations=1)
    """

    def __init__(self, max_face_limit: int = 1_000_000) -> None:
        self.max_face_limit = max_face_limit

    def midpoint_subdivision(
        self,
        mesh: trimesh.Trimesh,
        iterations: int = 1,
    ) -> trimesh.Trimesh:
        """Apply midpoint subdivision.

        Splits each edge at its midpoint, creating 4 faces per
        original face per iteration.

        Args:
            mesh: Input mesh.
            iterations: Number of subdivision passes.

        Returns:
            Subdivided mesh.
        """
        try:
            ms = _trimesh_to_meshset(mesh)
            for i in range(iterations):
                current_faces = ms.current_mesh().face_number()
                if current_faces * 4 > self.max_face_limit:
                    logger.warning(
                        f"Subdivision would exceed face limit "
                        f"({current_faces * 4} > {self.max_face_limit}). "
                        f"Stopping at iteration {i}."
                    )
                    break
                ms.apply_filter("meshing_surface_subdivision_midpoint")
            result = _meshset_to_trimesh(ms)
            logger.info(
                f"Midpoint subdivision: {len(mesh.faces)} → {len(result.faces)} faces"
            )
            return result
        except Exception as e:
            logger.warning(f"Midpoint subdivision failed: {e}")
            return mesh

    def loop_subdivision(
        self,
        mesh: trimesh.Trimesh,
        iterations: int = 1,
    ) -> trimesh.Trimesh:
        """Apply Loop subdivision for smooth surface refinement.

        Loop subdivision produces smoother results than midpoint
        by computing weighted averages of neighboring vertices.

        Args:
            mesh: Input mesh (must be triangle mesh).
            iterations: Number of subdivision passes.

        Returns:
            Smoothly subdivided mesh.
        """
        try:
            ms = _trimesh_to_meshset(mesh)
            for i in range(iterations):
                current_faces = ms.current_mesh().face_number()
                if current_faces * 4 > self.max_face_limit:
                    logger.warning(
                        f"Loop subdivision would exceed limit. "
                        f"Stopping at iteration {i}."
                    )
                    break
                ms.apply_filter("meshing_surface_subdivision_loop")
            result = _meshset_to_trimesh(ms)
            logger.info(
                f"Loop subdivision: {len(mesh.faces)} → {len(result.faces)} faces"
            )
            return result
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

        Only subdivides triangles containing edges longer than
        the threshold, concentrating detail where it's needed.

        Args:
            mesh: Input mesh.
            edge_threshold: Maximum edge length before subdivision.
                If None, uses the mean edge length.
            max_iterations: Maximum number of adaptive passes.

        Returns:
            Adaptively subdivided mesh.
        """
        if edge_threshold is None:
            edges = mesh.edges_unique_length
            edge_threshold = float(np.mean(edges))
            logger.info(f"Auto edge threshold: {edge_threshold:.6f}")

        try:
            ms = _trimesh_to_meshset(mesh)
            for i in range(max_iterations):
                current_faces = ms.current_mesh().face_number()
                if current_faces > self.max_face_limit:
                    logger.warning(f"Face limit reached at iteration {i}")
                    break

                # Select faces with long edges
                ms.apply_filter(
                    "compute_selection_by_edge_length",
                    threshold=edge_threshold,
                )

                # Subdivide selected faces using midpoint
                try:
                    ms.apply_filter("meshing_surface_subdivision_midpoint",
                                    selectedonly=True)
                except Exception:
                    # Some versions don't support selectedonly
                    ms.apply_filter("meshing_surface_subdivision_midpoint")

            result = _meshset_to_trimesh(ms)
            logger.info(
                f"Adaptive subdivision: {len(mesh.faces)} → "
                f"{len(result.faces)} faces ({max_iterations} passes)"
            )
            return result
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
