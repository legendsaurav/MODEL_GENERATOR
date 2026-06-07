"""
Mesh repair operations for MODEL_GENERATOR_V2.

Comprehensive mesh cleaning: removes degenerate/duplicate faces,
merges close vertices, removes isolated components (floaters),
fixes normals, and fills small holes.

Uses trimesh as the PRIMARY backend (no external dependencies).
Optionally uses pymeshlab for enhanced repair if available and working.

Dependencies:
    - trimesh
    - numpy
    - pymeshlab (optional, for enhanced repair)

Classes:
    MeshRepairer: All-in-one mesh repair pipeline.
"""

import logging
import os
import tempfile
from typing import Optional

import numpy as np
import trimesh

logger = logging.getLogger("model_generator_v2.postprocessing.mesh_repair")

# ── Pymeshlab availability check ───────────────────────────────────────────
_PYMESHLAB_AVAILABLE = False
try:
    import pymeshlab
    # Test if the critical I/O plugin actually works
    _test_ms = pymeshlab.MeshSet()
    _PYMESHLAB_AVAILABLE = True
    del _test_ms
except Exception:
    logger.info("pymeshlab not fully functional — using trimesh-only repair")


def _pymeshlab_roundtrip(mesh: trimesh.Trimesh, filters: list) -> Optional[trimesh.Trimesh]:
    """Apply pymeshlab filters via numpy arrays (no temp files).

    Args:
        mesh: Input trimesh.
        filters: List of (filter_name, kwargs) tuples.

    Returns:
        Processed mesh, or None if pymeshlab fails.
    """
    if not _PYMESHLAB_AVAILABLE:
        return None
    try:
        ms = pymeshlab.MeshSet()
        # Load directly from numpy arrays — no temp file needed
        pm = pymeshlab.Mesh(
            vertex_matrix=mesh.vertices.astype(np.float64),
            face_matrix=mesh.faces.astype(np.int32),
        )
        ms.add_mesh(pm)

        for filter_name, kwargs in filters:
            ms.apply_filter(filter_name, **kwargs)

        # Extract result from numpy arrays
        result_mesh = ms.current_mesh()
        verts = result_mesh.vertex_matrix()
        faces = result_mesh.face_matrix()

        if len(verts) == 0 or len(faces) == 0:
            return None

        return trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            process=False,
        )
    except Exception as e:
        logger.debug(f"pymeshlab filter failed: {e}")
        return None


class MeshRepairer:
    """Comprehensive mesh repair operations.

    Uses trimesh as the primary backend. All operations work without
    pymeshlab. When pymeshlab is available and working, enhanced
    operations are used for better results.

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
        # Try pymeshlab first
        result = _pymeshlab_roundtrip(mesh, [("meshing_remove_null_faces", {})])
        if result is not None:
            removed = len(mesh.faces) - len(result.faces)
            if removed > 0:
                logger.info(f"Removed {removed} degenerate faces (pymeshlab)")
            return result

        # Trimesh fallback
        try:
            mask = mesh.nondegenerate_faces()
            mesh.update_faces(mask)
            mesh.remove_unreferenced_vertices()
            logger.info(f"Removed degenerate faces (trimesh), kept {len(mesh.faces)}")
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
        result = _pymeshlab_roundtrip(mesh, [("meshing_remove_duplicate_faces", {})])
        if result is not None:
            removed = len(mesh.faces) - len(result.faces)
            if removed > 0:
                logger.info(f"Removed {removed} duplicate faces (pymeshlab)")
            return result

        # Trimesh fallback
        try:
            unique = mesh.unique_faces()
            mesh.update_faces(unique)
            mesh.remove_unreferenced_vertices()
            logger.info(f"Removed duplicate faces (trimesh), kept {len(mesh.faces)}")
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
        # Trimesh merge is reliable and fast
        try:
            original_v = len(mesh.vertices)
            mesh.merge_vertices()
            merged = original_v - len(mesh.vertices)
            if merged > 0:
                logger.info(f"Merged {merged} duplicate vertices")
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

        # Try pymeshlab first
        result = _pymeshlab_roundtrip(mesh, [
            ("compute_selection_by_small_disconnected_components_per_face",
             {"nbfaceratio": ratio}),
            ("meshing_remove_selected_vertices_and_faces", {}),
        ])
        if result is not None:
            removed = len(mesh.faces) - len(result.faces)
            if removed > 0:
                logger.info(f"Removed {removed} faces from small components (pymeshlab)")
            return result

        # Trimesh fallback: split into components, keep large ones
        try:
            components = mesh.split(only_watertight=False)
            if len(components) <= 1:
                return mesh

            total_faces = len(mesh.faces)
            min_faces = int(total_faces * ratio)
            large_components = [
                c for c in components if len(c.faces) >= min_faces
            ]

            if not large_components:
                # Keep the largest component
                large_components = [max(components, key=lambda c: len(c.faces))]

            if len(large_components) < len(components):
                mesh = trimesh.util.concatenate(large_components)
                logger.info(
                    f"Removed {len(components) - len(large_components)} "
                    f"small components (trimesh)"
                )
        except Exception as e:
            logger.warning(f"Component removal failed: {e}")
        return mesh

    def fix_normals(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Fix inconsistent face and vertex normals.

        Args:
            mesh: Input mesh.

        Returns:
            Mesh with corrected normals.
        """
        try:
            mesh.fix_normals()
            logger.info("Normals fixed (trimesh)")
        except Exception as e:
            logger.warning(f"Normal fixing failed: {e}")
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

        # Try pymeshlab
        result = _pymeshlab_roundtrip(mesh, [
            ("meshing_close_holes", {"maxholesize": max_edges}),
        ])
        if result is not None:
            new_faces = len(result.faces) - len(mesh.faces)
            if new_faces > 0:
                logger.info(f"Filled holes: added {new_faces} faces (pymeshlab)")
            return result

        # Trimesh fallback
        try:
            mesh.fill_holes()
            logger.info("Filled holes (trimesh)")
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
