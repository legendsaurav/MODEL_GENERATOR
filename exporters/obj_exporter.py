"""
Wavefront OBJ mesh exporter for MODEL_GENERATOR_V2.

Exports meshes in ASCII OBJ format with optional vertex normals.
OBJ is the most widely supported 3D format across DCC tools.

Dependencies:
    - trimesh

Classes:
    OBJExporter: Exports meshes to OBJ format.
"""

import logging
from typing import List

import trimesh

from .base_exporter import MeshExporter

logger = logging.getLogger("model_generator_v2.exporters.obj")


class OBJExporter(MeshExporter):
    """Exports meshes to Wavefront OBJ format.

    OBJ is a plain-text format supported by virtually every 3D
    application. Includes vertices, faces, and optionally
    vertex normals.

    Example:
        >>> exporter = OBJExporter()
        >>> path = exporter.export(mesh, 'output.obj')
    """

    @property
    def supported_extensions(self) -> List[str]:
        return [".obj"]

    def export(
        self,
        mesh: trimesh.Trimesh,
        output_path: str,
        include_normals: bool = True,
        **kwargs,
    ) -> str:
        """Export mesh to OBJ format.

        Args:
            mesh: The mesh to export.
            output_path: Target file path.
            include_normals: Whether to include vertex normals.
            **kwargs: Additional options.

        Returns:
            Absolute path to the exported file.

        Raises:
            ValueError: If mesh is invalid.
        """
        if not self.validate_mesh(mesh):
            raise ValueError("Invalid mesh for export")

        output_path = self.ensure_extension(output_path, ".obj")
        self.ensure_directory(output_path)

        if include_normals:
            _ = mesh.vertex_normals

        mesh.export(output_path, file_type="obj", include_normals=include_normals)
        logger.info(
            f"Exported OBJ: {output_path} "
            f"({len(mesh.vertices)} verts, {len(mesh.faces)} faces)"
        )
        return output_path
