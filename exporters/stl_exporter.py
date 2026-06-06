"""
STL mesh exporter for MODEL_GENERATOR_V2.

Exports meshes in binary or ASCII STL format — the standard
format for 3D printing workflows.

Dependencies:
    - trimesh

Classes:
    STLExporter: Exports meshes to STL format.
"""

import logging
from typing import List

import trimesh

from .base_exporter import MeshExporter

logger = logging.getLogger("model_generator_v2.exporters.stl")


class STLExporter(MeshExporter):
    """Exports meshes to STL (Stereolithography) format.

    STL stores triangle geometry using face normals. Binary STL
    is compact and fast to parse; ASCII STL is human-readable.
    Standard format for 3D printing.

    Example:
        >>> exporter = STLExporter()
        >>> path = exporter.export(mesh, 'output.stl', binary=True)
    """

    @property
    def supported_extensions(self) -> List[str]:
        return [".stl"]

    def export(
        self,
        mesh: trimesh.Trimesh,
        output_path: str,
        binary: bool = True,
        **kwargs,
    ) -> str:
        """Export mesh to STL format.

        Args:
            mesh: The mesh to export.
            output_path: Target file path.
            binary: Use binary STL (True) or ASCII STL (False).
            **kwargs: Additional options.

        Returns:
            Absolute path to the exported file.

        Raises:
            ValueError: If mesh is invalid.
        """
        if not self.validate_mesh(mesh):
            raise ValueError("Invalid mesh for export")

        output_path = self.ensure_extension(output_path, ".stl")
        self.ensure_directory(output_path)

        # trimesh STL export
        stl_data = trimesh.exchange.stl.export_stl(mesh)
        if binary:
            with open(output_path, "wb") as f:
                f.write(stl_data)
        else:
            # ASCII STL fallback
            mesh.export(output_path, file_type="stl_ascii")

        logger.info(
            f"Exported STL ({'binary' if binary else 'ascii'}): "
            f"{output_path} ({len(mesh.faces)} faces)"
        )
        return output_path
