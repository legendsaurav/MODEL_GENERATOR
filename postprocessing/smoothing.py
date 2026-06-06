"""
Mesh smoothing algorithms for MODEL_GENERATOR_V2.

Implements HC (Humphrey's Classes) Laplacian smoothing and Taubin
(lambda-mu) smoothing to reduce surface noise while preserving
volume and geometric features.

Dependencies:
    - trimesh
    - pymeshlab

Classes:
    MeshSmoother: Applies HC or Taubin smoothing to meshes.
"""

import logging
import tempfile

import trimesh

logger = logging.getLogger("model_generator_v2.postprocessing.smoothing")


def _trimesh_to_meshset(mesh: trimesh.Trimesh):
    """Convert trimesh → pymeshlab MeshSet via temp file."""
    import pymeshlab
    ms = pymeshlab.MeshSet()
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        mesh.export(f.name)
        ms.load_new_mesh(f.name)
    return ms


def _meshset_to_trimesh(ms) -> trimesh.Trimesh:
    """Convert pymeshlab MeshSet → trimesh via temp file."""
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        ms.save_current_mesh(f.name)
        return trimesh.load(f.name, process=False)


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

        Alternates positive and negative Laplacian smoothing passes
        to smooth the surface without shrinkage.

        Args:
            mesh: Input mesh.
            iterations: Number of smoothing passes (each pass = lambda + mu).
            lambda_factor: Positive smoothing strength (0-1).
            mu_factor: Negative smoothing strength (should be < -lambda).

        Returns:
            Smoothed mesh with preserved vertex count.
        """
        try:
            ms = _trimesh_to_meshset(mesh)
            ms.apply_filter(
                "apply_coord_taubin_smoothing",
                stepsmoothnum=iterations,
                lambda_=lambda_factor,
                mu=mu_factor,
            )
            result = _meshset_to_trimesh(ms)
            logger.info(
                f"Taubin smoothing: {iterations} iterations, "
                f"λ={lambda_factor}, μ={mu_factor}"
            )
            return result
        except Exception as e:
            logger.warning(f"Taubin smoothing failed: {e}")
            return mesh

    def hc_laplacian_smooth(
        self,
        mesh: trimesh.Trimesh,
        iterations: int = 3,
    ) -> trimesh.Trimesh:
        """Apply HC (Humphrey's Classes) Laplacian smoothing.

        Applies Laplacian smoothing with a correction step that
        pushes vertices back toward their original positions,
        reducing shrinkage while smoothing noise.

        Args:
            mesh: Input mesh.
            iterations: Number of HC smoothing passes.

        Returns:
            Smoothed mesh with preserved vertex count.
        """
        try:
            ms = _trimesh_to_meshset(mesh)
            ms.apply_filter(
                "apply_coord_hc_laplacian_smoothing",
            )
            # HC filter doesn't have a iterations param, apply multiple times
            for _ in range(iterations - 1):
                ms.apply_filter("apply_coord_hc_laplacian_smoothing")
            result = _meshset_to_trimesh(ms)
            logger.info(f"HC Laplacian smoothing: {iterations} iterations")
            return result
        except Exception as e:
            logger.warning(f"HC smoothing failed: {e}")
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
