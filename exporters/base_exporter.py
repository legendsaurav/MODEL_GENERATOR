"""
Abstract base class for mesh exporters.

Defines the common interface that all format-specific exporters
must implement.

Dependencies:
    - trimesh
    - abc (stdlib)

Classes:
    MeshExporter: Abstract base exporter.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import List

import trimesh

logger = logging.getLogger("model_generator_v2.exporters")


class MeshExporter(ABC):
    """Abstract base class for mesh format exporters.

    All format-specific exporters (GLB, OBJ, STL, PLY) must
    inherit from this class and implement the export() method.

    Subclasses should also define the supported_extensions property.
    """

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """List of file extensions this exporter supports.

        Returns:
            List of extension strings (e.g., ['.glb', '.gltf']).
        """
        ...

    @abstractmethod
    def export(
        self, mesh: trimesh.Trimesh, output_path: str, **kwargs
    ) -> str:
        """Export a mesh to a file.

        Args:
            mesh: The mesh to export.
            output_path: Target file path.
            **kwargs: Format-specific options.

        Returns:
            The absolute path to the exported file.
        """
        ...

    def validate_mesh(self, mesh: trimesh.Trimesh) -> bool:
        """Check that the mesh is valid for export.

        Args:
            mesh: The mesh to validate.

        Returns:
            True if the mesh has vertices and faces.
        """
        if mesh is None:
            logger.error("Cannot export None mesh")
            return False
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            logger.error("Cannot export empty mesh")
            return False
        return True

    def ensure_directory(self, path: str) -> None:
        """Create the output directory if it doesn't exist.

        Args:
            path: File path whose parent directory should exist.
        """
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)

    def ensure_extension(self, path: str, ext: str) -> str:
        """Ensure the file path has the correct extension.

        Args:
            path: Original file path.
            ext: Required extension (e.g., '.glb').

        Returns:
            Path with correct extension.
        """
        if not path.lower().endswith(ext.lower()):
            path = path + ext
        return path
