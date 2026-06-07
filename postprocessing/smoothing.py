"""
Mesh smoothing algorithms for MODEL_GENERATOR_V2.

Implements HC (Humphrey's Classes) Laplacian smoothing and Taubin
(lambda-mu) smoothing to reduce surface noise while preserving
volume and geometric features.

Uses trimesh as the PRIMARY backend. Pymeshlab is used if available
for enhanced Taubin smoothing.

Dependencies:
    - trimesh
    - numpy
    - pymeshlab (optional)

Classes:
    MeshSmoother: Applies HC or Taubin smoothing to meshes.
"""

import logging
from typing import Optional

import numpy as np
import trimesh

logger = logging.getLogger("model_generator_v2.postprocessing.smoothing")

# ── Pymeshlab availability check ───────────────────────────────────────────
_PYMESHLAB_AVAILABLE = False
try:
    import pymeshlab
    _test_ms = pymeshlab.MeshSet()
    _PYMESHLAB_AVAILABLE = True
    del _test_ms
except Exception:
    logger.info("pymeshlab not fully functional — using trimesh-only smoothing")


def _pymeshlab_smooth(mesh: trimesh.Trimesh, filter_name: str, **kwargs) -> Optional[trimesh.Trimesh]:
    """Apply a pymeshlab smoothing filter via numpy arrays.

    Args:
        mesh: Input trimesh.
        filter_name: pymeshlab filter name.
        **kwargs: Filter parameters.

    Returns:
        Smoothed mesh, or None if pymeshlab fails.
    """
    if not _PYMESHLAB_AVAILABLE:
        return None
    try:
        ms = pymeshlab.MeshSet()
        pm = pymeshlab.Mesh(
            vertex_matrix=mesh.vertices.astype(np.float64),
            face_matrix=mesh.faces.astype(np.int32),
        )
        ms.add_mesh(pm)
        ms.apply_filter(filter_name, **kwargs)

        result_mesh = ms.current_mesh()
        verts = result_mesh.vertex_matrix()
        faces = result_mesh.face_matrix()

        if len(verts) == 0 or len(faces) == 0:
            return None

        return trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    except Exception as e:
        logger.debug(f"pymeshlab smoothing failed: {e}")
        return None


class MeshSmoother:
    """Surface smoothing using Taubin or HC Laplacian methods.

    Both algorithms are volume-preserving alternatives to naive
    Laplacian smoothing, which tends to shrink meshes.

    - **Taubin**: Alternates between positive (lambda) and negative (mu)
      smoothing passes, cancelling the shrinkage effect. Best for
      general-purpose smoothing.

    - **HC Laplacian**: Humphrey's Classes smoothing applies a
      correction step after each Laplacian pass to push vertices
      back toward their original positions. Best for preserving
      sharp features.

    Args:
        default_method: Default smoothing method ('taubin' or 'hc').
        default_iterations: Default number of smoothing passes.

    Example:
        >>> smoother = MeshSmoother()
        >>> smooth_mesh = smoother(noisy_mesh, method='taubin', iterations=5)
    """

    def __init__(
        self,
        default_method: str = "taubin",
        default_iterations: int = 5,
    ) -> None:
        if default_method not in ("taubin", "hc"):
            raise ValueError(
                f"Method must be 'taubin' or 'hc', got '{default_method}'"
            )
        self.default_method = default_method
        self.default_iterations = default_iterations

    def taubin_smooth(
        self,
        mesh: trimesh.Trimesh,
        iterations: int = 5,
        lambda_factor: float = 0.5,
        mu_factor: float = -0.53,
    ) -> trimesh.Trimesh:
        """Apply Taubin lambda-mu smoothing.

        Args:
            mesh: Input mesh.
            iterations: Number of smoothing passes.
            lambda_factor: Positive smoothing strength (0-1).
            mu_factor: Negative smoothing strength (should be < -lambda).

        Returns:
            Smoothed mesh with preserved vertex count.
        """
        # Try pymeshlab
        result = _pymeshlab_smooth(
            mesh,
            "apply_coord_taubin_smoothing",
            stepsmoothnum=iterations,
            lambda_=lambda_factor,
            mu=mu_factor,
        )
        if result is not None:
            logger.info(
                f"Taubin smoothing: {iterations} iterations, "
                f"λ={lambda_factor}, μ={mu_factor} (pymeshlab)"
            )
            return result

        # Trimesh fallback: standard Laplacian smoothing
        try:
            trimesh.smoothing.filter_laplacian(
                mesh, iterations=iterations
            )
            logger.info(
                f"Laplacian smoothing: {iterations} iterations (trimesh fallback)"
            )
        except Exception as e:
            logger.warning(f"Taubin smoothing failed: {e}")
        return mesh

    def hc_laplacian_smooth(
        self,
        mesh: trimesh.Trimesh,
        iterations: int = 3,
    ) -> trimesh.Trimesh:
        """Apply HC (Humphrey's Classes) Laplacian smoothing.

        Args:
            mesh: Input mesh.
            iterations: Number of HC smoothing passes.

        Returns:
            Smoothed mesh with preserved vertex count.
        """
        # Try pymeshlab
        if _PYMESHLAB_AVAILABLE:
            try:
                ms = pymeshlab.MeshSet()
                pm = pymeshlab.Mesh(
                    vertex_matrix=mesh.vertices.astype(np.float64),
                    face_matrix=mesh.faces.astype(np.int32),
                )
                ms.add_mesh(pm)
                for _ in range(iterations):
                    ms.apply_filter("apply_coord_hc_laplacian_smoothing")

                result_mesh = ms.current_mesh()
                verts = result_mesh.vertex_matrix()
                faces = result_mesh.face_matrix()

                if len(verts) > 0 and len(faces) > 0:
                    logger.info(f"HC Laplacian smoothing: {iterations} iters (pymeshlab)")
                    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            except Exception as e:
                logger.debug(f"pymeshlab HC smoothing failed: {e}")

        # Trimesh fallback: Humphrey smoothing
        try:
            trimesh.smoothing.filter_humphrey(
                mesh, iterations=iterations
            )
            logger.info(f"Humphrey smoothing: {iterations} iterations (trimesh fallback)")
        except Exception as e:
            # Last fallback: basic Laplacian
            try:
                trimesh.smoothing.filter_laplacian(mesh, iterations=iterations)
                logger.info(f"Laplacian smoothing: {iterations} iterations (trimesh fallback)")
            except Exception as e2:
                logger.warning(f"HC smoothing failed: {e2}")
        return mesh

    def __call__(
        self,
        mesh: trimesh.Trimesh,
        method: str = None,
        iterations: int = None,
    ) -> trimesh.Trimesh:
        """Apply smoothing with the specified method.

        Args:
            mesh: Input mesh.
            method: 'taubin' or 'hc' (defaults to self.default_method).
            iterations: Number of passes (defaults to self.default_iterations).

        Returns:
            Smoothed mesh.
        """
        method = method or self.default_method
        iterations = iterations or self.default_iterations

        logger.info(
            f"Smoothing mesh ({len(mesh.vertices)} vertices) "
            f"with {method}, {iterations} iterations"
        )

        if method == "taubin":
            return self.taubin_smooth(mesh, iterations)
        elif method == "hc":
            return self.hc_laplacian_smooth(mesh, iterations)
        else:
            raise ValueError(f"Unknown method: {method}")
