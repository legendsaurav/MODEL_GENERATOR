"""
Mesh repair operations for MODEL_GENERATOR_V2.

Comprehensive mesh cleaning: removes degenerate/duplicate faces,
merges close vertices, removes isolated components (floaters),
fixes normals, and fills small holes.

Dependencies:
    - trimesh
    - pymeshlab
    - numpy

Classes:
    MeshRepairer: All-in-one mesh repair pipeline.

Functions:
    trimesh_to_meshset: Convert trimesh → pymeshlab MeshSet.
    meshset_to_trimesh: Convert pymeshlab MeshSet → trimesh.
"""

import logging
import tempfile
from typing import Optional

import numpy as np
import trimesh

logger = logging.getLogger("model_generator_v2.postprocessing.mesh_repair")


def trimesh_to_meshset(mesh: trimesh.Trimesh):
    """Convert a trimesh.Trimesh to a pymeshlab.MeshSet.

    Uses a temporary file for reliable conversion.

    Args:
        mesh: Input trimesh object.

    Returns:
        pymeshlab.MeshSet containing the mesh.
    """
    import pymeshlab
    ms = pymeshlab.MeshSet()
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        mesh.export(f.name)
        ms.load_new_mesh(f.name)
    return ms


def meshset_to_trimesh(ms) -> trimesh.Trimesh:
    """Convert a pymeshlab.MeshSet to a trimesh.Trimesh.

    Args:
        ms: pymeshlab MeshSet with at least one mesh.

    Returns:
        trimesh.Trimesh object.
    """
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        ms.save_current_mesh(f.name)
        result = trimesh.load(f.name, process=False)
    return result


class MeshRepairer:
    """Comprehensive mesh repair operations.

    Applies a series of cleaning filters to fix common mesh defects
    produced by Marching Cubes extraction, including degenerate geometry,
    duplicate elements, inconsistent normals, and small holes.

    Args:
        min_component_face_ratio: Minimum face ratio for keeping
            connected components (smaller ones are removed).
        max_hole_edges: Maximum edge count for holes to fill.
        merge_tolerance: Vertex merge distance threshold.

    Example:
        >>> repairer = MeshRepairer()
        >>> clean_mesh = repairer(raw_mesh)
    """

    def __init__(
        self,
        min_component_face_ratio: float = 0.005,
        max_hole_edges: int = 20,
        merge_tolerance: float = 1e-6,
    ) -> None:
        self.min_component_face_ratio = min_component_face_ratio
        self.max_hole_edges = max_hole_edges
        self.merge_tolerance = merge_tolerance

    def remove_degenerate_faces(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Remove zero-area and collapsed faces.

        Args:
            mesh: Input mesh.

        Returns:
            Mesh with degenerate faces removed.
        """
        try:
            ms = trimesh_to_meshset(mesh)
            ms.apply_filter("meshing_remove_null_faces")
            result = meshset_to_trimesh(ms)
            removed = len(mesh.faces) - len(result.faces)
            if removed > 0:
                logger.info(f"Removed {removed} degenerate faces")
            return result
        except Exception as e:
            logger.warning(f"Degenerate face removal failed: {e}")
            return mesh

    def remove_duplicate_faces(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Remove duplicate face definitions.

        Args:
            mesh: Input mesh.

        Returns:
            Mesh with duplicate faces removed.
        """
        try:
            ms = trimesh_to_meshset(mesh)
            ms.apply_filter("meshing_remove_duplicate_faces")
            result = meshset_to_trimesh(ms)
            removed = len(mesh.faces) - len(result.faces)
            if removed > 0:
                logger.info(f"Removed {removed} duplicate faces")
            return result
        except Exception as e:
            logger.warning(f"Duplicate face removal failed: {e}")
            return mesh

    def merge_duplicate_vertices(
        self, mesh: trimesh.Trimesh, tolerance: Optional[float] = None
    ) -> trimesh.Trimesh:
        """Merge vertices closer than tolerance distance.

        Args:
            mesh: Input mesh.
            tolerance: Distance threshold for merging.

        Returns:
            Mesh with merged vertices.
        """
        try:
            ms = trimesh_to_meshset(mesh)
            ms.apply_filter(
                "meshing_merge_close_vertices",
                threshold=pymeshlab.PercentageValue(
                    (tolerance or self.merge_tolerance) * 100
                ),
            )
            result = meshset_to_trimesh(ms)
            merged = len(mesh.vertices) - len(result.vertices)
            if merged > 0:
                logger.info(f"Merged {merged} duplicate vertices")
            return result
        except Exception as e:
            logger.warning(f"Vertex merging failed: {e}")
            return mesh

    def remove_isolated_components(
        self, mesh: trimesh.Trimesh, min_face_ratio: Optional[float] = None
    ) -> trimesh.Trimesh:
        """Remove small disconnected components (floaters).

        Keeps only components with more than min_face_ratio of total faces.

        Args:
            mesh: Input mesh.
            min_face_ratio: Minimum face count ratio to keep.

        Returns:
            Mesh with small components removed.
        """
        ratio = min_face_ratio or self.min_component_face_ratio
        try:
            ms = trimesh_to_meshset(mesh)
            ms.apply_filter(
                "compute_selection_by_small_disconnected_components_per_face",
                nbfaceratio=ratio,
            )
            ms.apply_filter(
                "compute_selection_transfer_face_to_vertex",
                inclusive=False,
            )
            ms.apply_filter("meshing_remove_selected_vertices_and_faces")
            result = meshset_to_trimesh(ms)
            removed = len(mesh.faces) - len(result.faces)
            if removed > 0:
                logger.info(f"Removed {removed} faces from small components")
            return result
        except Exception as e:
            logger.warning(f"Component removal failed: {e}")
            return mesh

    def fix_normals(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Fix inconsistent face and vertex normals.

        Re-orients faces for consistent normal direction and
        recomputes vertex normals.

        Args:
            mesh: Input mesh.

        Returns:
            Mesh with corrected normals.
        """
        try:
            ms = trimesh_to_meshset(mesh)
            ms.apply_filter("meshing_re_orient_faces_coherentely")
            ms.apply_filter("compute_normal_per_vertex")
            ms.apply_filter("compute_normal_per_face")
            result = meshset_to_trimesh(ms)
            logger.info("Normals fixed and recomputed")
            return result
        except Exception as e:
            logger.warning(f"Normal fixing failed: {e}")
            # Fallback to trimesh normal fixing
            mesh.fix_normals()
            return mesh

    def fill_small_holes(
        self, mesh: trimesh.Trimesh, max_hole_edges: Optional[int] = None
    ) -> trimesh.Trimesh:
        """Fill holes with fewer edges than the threshold.

        Args:
            mesh: Input mesh.
            max_hole_edges: Maximum edges for a hole to be filled.

        Returns:
            Mesh with small holes filled.
        """
        max_edges = max_hole_edges or self.max_hole_edges
        try:
            ms = trimesh_to_meshset(mesh)
            ms.apply_filter(
                "meshing_close_holes",
                maxholesize=max_edges,
            )
            result = meshset_to_trimesh(ms)
            new_faces = len(result.faces) - len(mesh.faces)
            if new_faces > 0:
                logger.info(f"Filled holes: added {new_faces} faces")
            return result
        except Exception as e:
            logger.warning(f"Hole filling failed: {e}")
            return mesh

    def __call__(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Run all repair operations in sequence.

        Order: degenerate → duplicate faces → merge vertices
        → isolated components → fix normals → fill holes.

        Args:
            mesh: Input mesh.

        Returns:
            Fully repaired mesh.
        """
        original_v = len(mesh.vertices)
        original_f = len(mesh.faces)
        logger.info(
            f"Starting mesh repair: {original_v} vertices, {original_f} faces"
        )

        mesh = self.remove_degenerate_faces(mesh)
        mesh = self.remove_duplicate_faces(mesh)
        mesh = self.merge_duplicate_vertices(mesh)
        mesh = self.remove_isolated_components(mesh)
        mesh = self.fix_normals(mesh)
        mesh = self.fill_small_holes(mesh)

        logger.info(
            f"Repair complete: {len(mesh.vertices)} vertices, "
            f"{len(mesh.faces)} faces"
        )
        return mesh


# Needed for merge_duplicate_vertices
try:
    import pymeshlab
except ImportError:
    pymeshlab = None
    logger.warning("pymeshlab not available — some repairs will be limited")
