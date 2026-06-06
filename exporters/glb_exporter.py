"""
GLB/GLTF mesh exporter for MODEL_GENERATOR_V2.

Exports meshes in binary GLB format — a compact, single-file 3D
format widely supported by web viewers and game engines.

Dependencies:
    - trimesh

Classes:
    GLBExporter: Exports meshes to GLB format.
"""

import logging
from typing import List

import trimesh

from .base_exporter import MeshExporter

logger = logging.getLogger("model_generator_v2.exporters.glb")


class GLBExporter(MeshExporter):
    """Exports meshes to binary GLB (GL Transmission Format).

    GLB is a compact binary format that bundles geometry and metadata
    in a single file. Ideal for web applications and real-time
    rendering pipelines.

    Example:
        >>> exporter = GLBExporter()
        >>> path = exporter.export(mesh, 'output.glb')
    """

    @property
    def supported_extensions(self) -> List[str]:
        return [".glb", ".gltf"]

    def export(
        self,
        mesh: trimesh.Trimesh,
        output_path: str,
        include_normals: bool = True,
        **kwargs,
    ) -> str:
        """Export mesh to GLB format.

        Args:
            mesh: The mesh to export.
            output_path: Target file path.
            include_normals: Whether to include vertex normals.
            **kwargs: Additional options passed to trimesh.

        Returns:
            Absolute path to the exported file.

        Raises:
            ValueError: If mesh is invalid.
        """
        if not self.validate_mesh(mesh):
            raise ValueError("Invalid mesh for export")

        output_path = self.ensure_extension(output_path, ".glb")
        self.ensure_directory(output_path)

        if include_normals:
            # Force normal computation before export
            _ = mesh.vertex_normals

        mesh.export(output_path, file_type="glb")
        logger.info(
            f"Exported GLB: {output_path} "
            f"({len(mesh.vertices)} verts, {len(mesh.faces)} faces)"
        )
        return output_path
