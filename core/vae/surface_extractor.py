"""
Surface extraction via Marching Cubes for MODEL_GENERATOR_V2.

Adapted from Hunyuan3D-2.1 autoencoders SurfaceExtractors.
Converts SDF (Signed Distance Function) volumetric grids into
triangle meshes using the Marching Cubes algorithm.

Dependencies:
    - torch
    - numpy
    - trimesh
    - skimage.measure (for marching_cubes)

Classes:
    SurfaceExtractor: Extracts triangle meshes from SDF grids.
    Latent2MeshOutput: Dataclass holding extracted mesh data.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import trimesh

from ...utils.logging import get_logger

logger = get_logger("model_generator_v2.core.vae.surface_extractor")


@dataclass
class Latent2MeshOutput:
    """Container for mesh data extracted from latent SDF predictions.

    Attributes:
        mesh_v: Vertex positions array [V, 3].
        mesh_f: Face indices array [F, 3].
        mesh_v_normals: Optional vertex normals [V, 3].
        sdf_grid: Optional raw SDF grid for debugging.
    """

    mesh_v: np.ndarray
    mesh_f: np.ndarray
    mesh_v_normals: Optional[np.ndarray] = None
    sdf_grid: Optional[np.ndarray] = None

    def to_trimesh(self) -> trimesh.Trimesh:
        """Convert to a trimesh.Trimesh object.

        Returns:
            A trimesh.Trimesh with vertices, faces, and optional normals.
        """
        mesh = trimesh.Trimesh(
            vertices=self.mesh_v,
            faces=self.mesh_f,
            process=False,
        )
        if self.mesh_v_normals is not None:
            # trimesh exposes vertex_normals as a settable property at runtime;
            # the stubs type it as a method, hence the targeted ignore.
            mesh.vertex_normals = self.mesh_v_normals  # type: ignore[method-assign]
        return mesh


class SurfaceExtractor:
    """Extracts triangle mesh surfaces from SDF volumetric grids.

    Uses the Marching Cubes algorithm to find the iso-surface
    (SDF = 0) within a 3D scalar field, producing vertices and
    faces for the resulting mesh.

    Args:
        iso_value: The iso-surface threshold (0.0 for SDF zero-crossing).
        padding: Fractional padding around the grid bounds.
        resolution_override: If set, overrides the grid resolution.

    Example:
        >>> extractor = SurfaceExtractor()
        >>> sdf_grid = vae.decode(latents)  # [R, R, R]
        >>> mesh_output = extractor(sdf_grid)
        >>> trimesh_mesh = mesh_output.to_trimesh()
    """

    def __init__(
        self,
        iso_value: float = 0.0,
        padding: float = 0.1,
        resolution_override: Optional[int] = None,
    ) -> None:
        self.iso_value = iso_value
        self.padding = padding
        self.resolution_override = resolution_override

    def __call__(
        self,
        sdf_grid: np.ndarray,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Optional[Latent2MeshOutput]:
        """Extract mesh from an SDF grid using Marching Cubes.

        Args:
            sdf_grid: 3D numpy array of SDF values [X, Y, Z].
            bounds: Optional (min_bound, max_bound) arrays for
                    mapping grid indices to world coordinates.

        Returns:
            Latent2MeshOutput with mesh data, or None if extraction fails.
        """
        return self.extract(sdf_grid, bounds)

    def extract(
        self,
        sdf_grid: np.ndarray,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Optional[Latent2MeshOutput]:
        """Core extraction method using scikit-image Marching Cubes.

        Args:
            sdf_grid: 3D SDF values [X, Y, Z].
            bounds: Optional world-space bounds.

        Returns:
            Latent2MeshOutput or None.
        """
        try:
            from skimage.measure import marching_cubes
        except ImportError:
            logger.error(
                "scikit-image required for marching cubes. "
                "Install: pip install scikit-image"
            )
            return None

        if sdf_grid.ndim != 3:
            raise ValueError(
                f"SDF grid must be 3D, got shape {sdf_grid.shape}"
            )

        resolution = sdf_grid.shape[0]
        logger.info(
            f"Running Marching Cubes on {resolution}³ grid "
            f"(iso_value={self.iso_value})"
        )

        try:
            vertices, faces, normals, _ = marching_cubes(
                sdf_grid,
                level=self.iso_value,
                spacing=(1.0 / resolution,) * 3,
            )
        except (ValueError, RuntimeError) as e:
            logger.error(f"Marching Cubes failed: {e}")
            return None

        if len(vertices) == 0 or len(faces) == 0:
            logger.warning("Marching Cubes produced empty mesh")
            return None

        # Map to world coordinates
        if bounds is not None:
            min_bound, max_bound = bounds
            extent = max_bound - min_bound
            vertices = vertices * extent + min_bound
        else:
            # Default: center at origin, unit scale
            vertices = vertices - 0.5

        logger.info(
            f"Extracted mesh: {len(vertices)} vertices, {len(faces)} faces"
        )

        return Latent2MeshOutput(
            mesh_v=vertices.astype(np.float32),
            mesh_f=faces.astype(np.int64),
            mesh_v_normals=normals.astype(np.float32),
            sdf_grid=sdf_grid,
        )

    def extract_from_torch(
        self,
        sdf_tensor: torch.Tensor,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Optional[Latent2MeshOutput]:
        """Extract mesh from a torch SDF tensor.

        Convenience method that handles tensor → numpy conversion.

        Args:
            sdf_tensor: 3D torch tensor of SDF values.
            bounds: Optional world-space bounds.

        Returns:
            Latent2MeshOutput or None.
        """
        sdf_np = sdf_tensor.detach().cpu().float().numpy()
        if sdf_np.ndim == 4:
            # Remove batch dimension [1, X, Y, Z] → [X, Y, Z]
            sdf_np = sdf_np.squeeze(0)
        if sdf_np.ndim == 4:
            # Remove channel dimension [1, X, Y, Z] → [X, Y, Z]
            sdf_np = sdf_np.squeeze(0)
        return self.extract(sdf_np, bounds)


def export_to_trimesh(
    mesh_output: "Latent2MeshOutput | list[Latent2MeshOutput]",
) -> "trimesh.Trimesh | list[trimesh.Trimesh | None]":
    """Convert Latent2MeshOutput(s) to trimesh object(s).

    Adapted from Hunyuan3D-2.1 pipelines.export_to_trimesh().
    Reverses face winding order to match Hunyuan3D convention.

    Args:
        mesh_output: Single or list of Latent2MeshOutput.

    Returns:
        Corresponding trimesh.Trimesh object(s).
    """
    if isinstance(mesh_output, list):
        results: "list[trimesh.Trimesh | None]" = []
        for m in mesh_output:
            if m is None:
                results.append(None)
            else:
                m.mesh_f = m.mesh_f[:, ::-1]
                results.append(
                    trimesh.Trimesh(m.mesh_v, m.mesh_f, process=False)
                )
        return results
    else:
        mesh_output.mesh_f = mesh_output.mesh_f[:, ::-1]
        return trimesh.Trimesh(
            mesh_output.mesh_v, mesh_output.mesh_f, process=False
        )
