"""
Stanford PLY mesh exporter for MODEL_GENERATOR_V2.

Exports meshes in binary or ASCII PLY format with optional
vertex normals. PLY supports per-vertex properties and is
popular in academic/research settings.

Dependencies:
    - trimesh

Classes:
    PLYExporter: Exports meshes to PLY format.
"""

import logging
from typing import List

import trimesh

from .base_exporter import MeshExporter

logger = logging.getLogger("model_generator_v2.exporters.ply")


class PLYExporter(MeshExporter):
    """Exports meshes to Stanford PLY format.

    PLY is a flexible format that supports per-vertex properties
    (normals, colors, custom attributes). Binary mode is compact;
    ASCII mode is human-readable.

    Example:
        >>> exporter = PLYExporter()
        >>> path = exporter.export(mesh, 'output.ply', binary=True)
    """

    @property
    def supported_extensions(self) -> List[str]:
        return [".ply"]

    def export(
        self,
        mesh: trimesh.Trimesh,
        output_path: str,
        binary: bool = True,
        include_normals: bool = True,
        **kwargs,
    ) -> str:
        """Export mesh to PLY format.

        Args:
            mesh: The mesh to export.
            output_path: Target file path.
            binary: Use binary PLY (True) or ASCII PLY (False).
            include_normals: Include vertex normals in output.
            **kwargs: Additional options.

        Returns:
            Absolute path to the exported file.

        Raises:
            ValueError: If mesh is invalid.
        """
        if not self.validate_mesh(mesh):
            raise ValueError("Invalid mesh for export")

        output_path = self.ensure_extension(output_path, ".ply")
        self.ensure_directory(output_path)

        if include_normals:
            _ = mesh.vertex_normals

        encoding = "binary_little_endian" if binary else "ascii"
        mesh.export(output_path, file_type="ply", encoding=encoding)

        logger.info(
            f"Exported PLY ({encoding}): {output_path} "
            f"({len(mesh.vertices)} verts, {len(mesh.faces)} faces)"
        )
        return output_path
