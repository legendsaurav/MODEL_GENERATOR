"""
Post-processing pipeline orchestrator for MODEL_GENERATOR_V2.

Chains all mesh post-processing operations in the correct order:
repair → smooth → subdivide → decimate → validate.

Dependencies:
    - trimesh

Classes:
    PostProcessingPipeline: Orchestrates the full post-processing chain.
"""

import logging
import time
from typing import Dict, Optional

import trimesh

from ..configs.base_config import PostProcessingConfig
from .mesh_repair import MeshRepairer
from .smoothing import MeshSmoother
from .subdivision import AdaptiveSubdivider
from .decimation import QuadricDecimator
from .validation import MeshValidator

logger = logging.getLogger("model_generator_v2.postprocessing.pipeline")


class PostProcessingPipeline:
    """Orchestrates the complete mesh post-processing chain.

    Applies a configurable sequence of mesh operations:
        1. Mesh repair (degenerate faces, duplicates, normals, holes)
        2. Floater/isolated component removal
        3. Surface smoothing (Taubin or HC Laplacian)
        4. Adaptive subdivision (if enabled)
        5. Quadric decimation (to target face count)
        6. Final normal recomputation
        7. Quality validation

    Each stage is independently configurable via PostProcessingConfig.
    Timing is logged for each stage.

    Args:
        config: Post-processing configuration. Uses defaults if None.

    Example:
        >>> from configs.presets import get_preset_config
        >>> config = get_preset_config('ultra').postprocessing
        >>> pipeline = PostProcessingPipeline(config)
        >>> clean_mesh = pipeline(raw_mesh)
    """

    def __init__(
        self, config: Optional[PostProcessingConfig] = None
    ) -> None:
        self.config = config or PostProcessingConfig()

        self.repairer = MeshRepairer(
            min_component_face_ratio=self.config.min_component_face_ratio,
            max_hole_edges=self.config.max_hole_edges,
        )
        self.smoother = MeshSmoother(
            default_method=self.config.smoothing_method,
            default_iterations=self.config.smoothing_iterations,
        )
        self.subdivider = AdaptiveSubdivider()
        self.decimator = QuadricDecimator(
            target_faces=self.config.target_faces,
            quality_threshold=self.config.decimation_quality,
            preserve_boundary=self.config.preserve_boundary,
        )
        self.validator = MeshValidator()

        logger.info(
            f"PostProcessingPipeline initialized: "
            f"repair={self.config.enable_repair}, "
            f"smooth={self.config.enable_smoothing} ({self.config.smoothing_method}), "
            f"subdivide={self.config.enable_subdivision}, "
            f"decimate={self.config.enable_decimation} "
            f"(target={self.config.target_faces})"
        )

    def __call__(
        self,
        mesh: trimesh.Trimesh,
        verbose: bool = True,
    ) -> trimesh.Trimesh:
        """Run the full post-processing pipeline.

        Args:
            mesh: Input raw mesh from generation.
            verbose: Whether to log detailed progress.

        Returns:
            Processed and optimized mesh.
        """
        total_start = time.perf_counter()
        timings: Dict[str, float] = {}

        logger.info(
            f"Starting post-processing: "
            f"{len(mesh.vertices)} vertices, {len(mesh.faces)} faces"
        )

        # Stage 1: Mesh repair
        if self.config.enable_repair:
            t0 = time.perf_counter()
            mesh = self.repairer(mesh)
            timings["repair"] = time.perf_counter() - t0
            if verbose:
                logger.info(f"  [1/7] Repair: {timings['repair']:.2f}s")

        # Stage 2: Smoothing
        if self.config.enable_smoothing:
            t0 = time.perf_counter()
            mesh = self.smoother(
                mesh,
                method=self.config.smoothing_method,
                iterations=self.config.smoothing_iterations,
            )
            timings["smoothing"] = time.perf_counter() - t0
            if verbose:
                logger.info(f"  [2/7] Smoothing: {timings['smoothing']:.2f}s")

        # Stage 3: Subdivision
        if self.config.enable_subdivision and self.config.subdivision_iterations > 0:
            t0 = time.perf_counter()
            mesh = self.subdivider(
                mesh,
                iterations=self.config.subdivision_iterations,
                method="loop",
            )
            timings["subdivision"] = time.perf_counter() - t0
            if verbose:
                logger.info(
                    f"  [3/7] Subdivision: {timings['subdivision']:.2f}s"
                )

        # Stage 4: Decimation
        if self.config.enable_decimation:
            t0 = time.perf_counter()
            mesh = self.decimator(
                mesh, target_faces=self.config.target_faces
            )
            timings["decimation"] = time.perf_counter() - t0
            if verbose:
                logger.info(
                    f"  [4/7] Decimation: {timings['decimation']:.2f}s"
                )

        # Stage 5: Final normal recomputation
        t0 = time.perf_counter()
        try:
            mesh.fix_normals()
        except Exception:
            pass
        timings["normals"] = time.perf_counter() - t0
        if verbose:
            logger.info(f"  [5/7] Normals: {timings['normals']:.2f}s")

        # Stage 6: Validation
        if self.config.enable_validation:
            t0 = time.perf_counter()
            self.validator(mesh)
            timings["validation"] = time.perf_counter() - t0
            if verbose:
                logger.info(
                    f"  [6/7] Validation: {timings['validation']:.2f}s"
                )
                report = self.validator.generate_report(mesh)
                logger.info(f"\n{report}")

        total_time = time.perf_counter() - total_start
        logger.info(
            f"Post-processing complete: {total_time:.2f}s total, "
            f"{len(mesh.vertices)} vertices, {len(mesh.faces)} faces"
        )

        return mesh
