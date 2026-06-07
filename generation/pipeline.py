"""
Main geometry generation pipeline for MODEL_GENERATOR_V2.

Uses the official Hunyuan3D `hy3dgen` package for correct weight loading
and inference. Falls back to a custom implementation if unavailable.

The official pipeline guarantees correct:
- DINOv2 conditioning
- DiT weight loading
- ShapeVAE weight loading
- Flow-matching diffusion
- SDF evaluation + Marching Cubes

Dependencies:
    - torch
    - PIL
    - trimesh
    - hy3dgen (pip install hy3dgen)

Classes:
    GeometryPipeline: End-to-end image-to-mesh pipeline.
"""

from pathlib import Path
from typing import Optional, Union

import torch
import trimesh
import numpy as np
from PIL import Image

from ..configs.base_config import GenerationConfig, PipelineConfig
from ..configs.presets import get_preset_config
from ..preprocessing.background_removal import BackgroundRemover
from ..preprocessing.image_processor import ImageProcessor
from ..utils.logging import get_logger
from ..utils.timer import synchronize_timer
from ..utils.memory import MemoryOptimizer
from ..utils.device import DeviceManager

logger = get_logger("model_generator_v2.generation.pipeline")


class GeometryPipeline:
    """End-to-end single-image-to-3D-mesh generation pipeline.

    Wraps the official Hunyuan3D-2 `Hunyuan3DDiTFlowMatchingPipeline`
    to guarantee correct weight loading and high-quality mesh output.

    Pipeline stages:
        1. Background removal (rembg) — handled by our preprocessor
        2. Image conditioning (DINOv2)  — handled by official pipeline
        3. Diffusion denoising (DiT)    — handled by official pipeline
        4. VAE decoding (SDF grid)      — handled by official pipeline
        5. Surface extraction (MC)      — handled by official pipeline

    Example:
        >>> pipeline = GeometryPipeline.from_pretrained(
        ...     'tencent/Hunyuan3D-2.1',
        ...     preset='ultra',
        ... )
        >>> mesh = pipeline('input.png')
        >>> mesh.export('output.glb')
    """

    def __init__(
        self,
        shape_pipeline,
        background_remover: Optional[BackgroundRemover] = None,
        image_processor: Optional[ImageProcessor] = None,
        config: Optional[GenerationConfig] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
        pipeline_type: str = "official",
    ) -> None:
        self._shape_pipeline = shape_pipeline
        self.background_remover = background_remover or BackgroundRemover()
        self.image_processor = image_processor or ImageProcessor()
        self.config = config or GenerationConfig()
        self._pipeline_type = pipeline_type

        dm = DeviceManager()
        self.device = device or dm.get_optimal_device()
        self.dtype = dtype

        logger.info(
            f"GeometryPipeline initialized: "
            f"device={self.device}, dtype={self.dtype}, "
            f"backend={self._pipeline_type}"
        )

    @classmethod
    @synchronize_timer("Pipeline Initialization")
    def from_pretrained(
        cls,
        model_path: str = "tencent/Hunyuan3D-2",
        preset: Optional[str] = None,
        config: Optional[PipelineConfig] = None,
        device: str = "auto",
        dtype_str: str = "float16",
        enable_cpu_offload: bool = False,
    ) -> "GeometryPipeline":
        """Load a pretrained pipeline.

        Tries loading strategies in order:
        1. Official hy3dgen Hunyuan3DDiTFlowMatchingPipeline
        2. Custom model loader (fallback — not weight-compatible)

        Args:
            model_path: HuggingFace model ID or local directory.
            preset: Quality preset name ('fast', 'balanced', 'ultra').
            config: Explicit PipelineConfig (overrides preset).
            device: Device string ('auto', 'cuda', 'cpu').
            dtype_str: Dtype string ('float16' or 'float32').
            enable_cpu_offload: Whether to use CPU offloading.

        Returns:
            Initialized GeometryPipeline ready for inference.
        """
        # Resolve config
        if config is None and preset:
            config = get_preset_config(preset)
        if config is None:
            config = PipelineConfig()

        gen_config = config.generation
        dtype = torch.float16 if dtype_str == "float16" else torch.float32

        # Resolve device
        dm = DeviceManager()
        actual_device = dm.get_optimal_device(
            device if device != "auto" else None
        )

        # ── Determine correct subfolder for model variant ──────────────
        subfolder = None
        if "2.1" in model_path or "2-1" in model_path:
            subfolder = "hunyuan3d-dit-v2-1"
        elif "2.0" in model_path or "2-0" in model_path:
            subfolder = "hunyuan3d-dit-v2-0"
        # If no version detected, let hy3dgen figure it out

        # ── Strategy 1: Official hy3dgen ───────────────────────────────
        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

            logger.info(
                f"Loading official Hunyuan3DDiTFlowMatchingPipeline "
                f"from '{model_path}' (subfolder={subfolder})..."
            )

            load_kwargs = {
                "pretrained_model_or_path": model_path,
                "torch_dtype": dtype,
            }
            if subfolder:
                load_kwargs["subfolder"] = subfolder

            pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                **load_kwargs
            )

            if str(actual_device).startswith("cuda"):
                pipe = pipe.to(actual_device)

            logger.info("Official hy3dgen pipeline loaded successfully ✓")

            return cls(
                shape_pipeline=pipe,
                config=gen_config,
                device=actual_device,
                dtype=dtype,
                pipeline_type="hy3dgen",
            )

        except ImportError:
            logger.warning(
                "hy3dgen not installed. Install with: "
                "pip install hy3dgen"
            )
        except Exception as e:
            logger.warning(f"hy3dgen loading failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())

        # ── Strategy 2: Custom model loader (FALLBACK) ─────────────────
        logger.warning(
            "⚠ Falling back to custom model loader. "
            "Weight compatibility is NOT guaranteed. "
            "For correct results, install hy3dgen: pip install hy3dgen"
        )

        from .model_loader import ModelLoader
        from ..core.conditioner import ImageConditioner
        from ..core.dit_model import Hunyuan3DDiT
        from ..core.scheduler import FlowMatchingScheduler
        from ..core.vae import ShapeVAE
        from .diffusion_runner import DiffusionRunner

        loader = ModelLoader(
            device=device,
            dtype=dtype,
            enable_cpu_offload=enable_cpu_offload,
        )
        models = loader.load_from_pretrained(model_path)
        models["conditioner"].load_model(actual_device, dtype)

        custom_pipe = _CustomPipelineWrapper(
            model=models["model"],
            vae=models["vae"],
            conditioner=models["conditioner"],
            scheduler=models["scheduler"],
            device=actual_device,
            dtype=dtype,
        )

        return cls(
            shape_pipeline=custom_pipe,
            config=gen_config,
            device=actual_device,
            dtype=dtype,
            pipeline_type="custom (WARNING: weights may not match)",
        )

    @synchronize_timer("Full Generation")
    def __call__(
        self,
        image: Union[str, Path, Image.Image],
        num_inference_steps: Optional[int] = None,
        octree_resolution: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        remove_background: bool = True,
        show_progress: bool = True,
    ) -> Optional[trimesh.Trimesh]:
        """Generate a 3D mesh from a single input image.

        Args:
            image: Input image (path or PIL Image).
            num_inference_steps: Override for diffusion steps.
            octree_resolution: Override for SDF grid resolution.
            guidance_scale: Override for guidance scale.
            seed: Random seed for reproducibility.
            remove_background: Whether to run background removal.
            show_progress: Whether to show progress bars.

        Returns:
            A trimesh.Trimesh object, or None on failure.
        """
        steps = num_inference_steps or self.config.num_inference_steps
        resolution = octree_resolution or self.config.octree_resolution
        guidance = guidance_scale or self.config.guidance_scale
        seed_val = seed if seed is not None else self.config.seed

        logger.info(
            f"Generating mesh: steps={steps}, "
            f"resolution={resolution}, guidance={guidance}"
        )

        # Load image
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        # Ensure RGBA for the official pipeline (it handles its own bg removal)
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        # Background removal (our own for consistency)
        if remove_background:
            with synchronize_timer("Background Removal"):
                image = self.background_remover(
                    image, target_size=self.image_processor.target_size
                )

        # ── Generate with the loaded pipeline ──────────────────────────
        try:
            if self._pipeline_type == "hy3dgen":
                mesh = self._generate_hy3dgen(
                    image, steps, resolution, guidance, seed_val
                )
            else:
                mesh = self._generate_custom(
                    image, steps, resolution, guidance, seed_val, show_progress
                )

            if mesh is not None:
                logger.info(
                    f"Generated raw mesh: "
                    f"{len(mesh.vertices)} vertices, "
                    f"{len(mesh.faces)} faces"
                )
                MemoryOptimizer.clear_cache()
                return mesh
            else:
                logger.error("Mesh generation failed — no surface extracted")
                return None

        except Exception as e:
            logger.error(f"Pipeline generation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _generate_hy3dgen(
        self, image, steps, resolution, guidance, seed_val
    ) -> Optional[trimesh.Trimesh]:
        """Generate using the official hy3dgen pipeline.

        The official pipeline returns a list of trimesh objects.
        """
        # Set up generator for seed
        gen_device = self.device if str(self.device).startswith("cuda") else "cpu"
        generator = None
        if seed_val is not None:
            generator = torch.Generator(device=gen_device)
            generator.manual_seed(seed_val)

        # Call official pipeline
        # Returns list of trimesh objects when output_type='trimesh'
        call_kwargs = {
            "image": image,
            "num_inference_steps": steps,
            "octree_resolution": resolution,
            "guidance_scale": guidance,
            "output_type": "trimesh",
        }
        if generator is not None:
            call_kwargs["generator"] = generator

        result = self._shape_pipeline(**call_kwargs)

        # Extract mesh from result
        if isinstance(result, list) and len(result) > 0:
            mesh = result[0]
        elif isinstance(result, trimesh.Trimesh):
            mesh = result
        elif hasattr(result, 'meshes'):
            mesh = result.meshes[0] if result.meshes else None
        else:
            mesh = result

        # Ensure it's a trimesh.Trimesh
        if mesh is not None and not isinstance(mesh, trimesh.Trimesh):
            if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
                mesh = trimesh.Trimesh(
                    vertices=np.array(mesh.vertices),
                    faces=np.array(mesh.faces),
                    process=False,
                )
            else:
                logger.error(f"Unknown result type: {type(mesh)}")
                return None

        return mesh

    def _generate_custom(
        self, image, steps, resolution, guidance, seed_val, show_progress
    ) -> Optional[trimesh.Trimesh]:
        """Generate using the custom model wrapper (fallback)."""
        return self._shape_pipeline(
            image=image,
            num_inference_steps=steps,
            octree_resolution=resolution,
            guidance_scale=guidance,
            seed=seed_val,
            show_progress=show_progress,
        )


class _CustomPipelineWrapper:
    """Wraps our custom DiT/VAE models into a callable pipeline.

    This is the FALLBACK when hy3dgen is not available.
    Weight compatibility is NOT guaranteed — output quality may be poor.
    """

    def __init__(self, model, vae, conditioner, scheduler, device, dtype):
        self.model = model
        self.vae = vae
        self.conditioner = conditioner
        self.scheduler = scheduler
        self.device = device
        self.dtype = dtype
        self.image_processor = ImageProcessor()

        from .diffusion_runner import DiffusionRunner
        self.diffusion_runner = DiffusionRunner(model, scheduler)

    def __call__(
        self,
        image,
        num_inference_steps=50,
        octree_resolution=384,
        guidance_scale=7.5,
        seed=None,
        show_progress=True,
    ):
        # Preprocess image
        with synchronize_timer("Image Preprocessing"):
            pixel_values = self.image_processor.to_tensor(
                image, device=self.device, dtype=self.dtype
            )

        # Conditioning
        with synchronize_timer("Image Conditioning"):
            condition_tokens = self.conditioner.encode_image(pixel_values)
            pooled = self.conditioner.get_pooled_embedding(pixel_values)

        MemoryOptimizer.log_memory_usage("After conditioning")

        # Prepare noise
        num_latent_tokens = 2048
        latent_dim = self.model.in_channels
        noise = self.scheduler.sample_noise(
            shape=(1, num_latent_tokens, latent_dim),
            device=self.device,
            dtype=self.dtype,
            seed=seed,
        )

        # Diffusion
        latent_tokens = self.diffusion_runner.run(
            noise=noise,
            condition_tokens=condition_tokens,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            pooled_projection=pooled,
            device=self.device,
            dtype=self.dtype,
            show_progress=show_progress,
        )

        MemoryOptimizer.log_memory_usage("After diffusion")

        # VAE decode
        mesh = self.vae.latents2mesh(
            latent_tokens=latent_tokens,
            octree_resolution=octree_resolution,
        )

        return mesh
