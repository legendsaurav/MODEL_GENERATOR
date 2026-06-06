"""
Main geometry generation pipeline for MODEL_GENERATOR_V2.

Orchestrates the complete image-to-mesh workflow:
Image → Background Removal → Image Conditioning → Shape Diffusion
→ Latent Generation → VAE Decode → Surface Extraction → Mesh

Adapted from Hunyuan3D-2.1 Hunyuan3DDiTFlowMatchingPipeline with
all texture-related code removed.

Dependencies:
    - torch
    - PIL
    - trimesh

Classes:
    GeometryPipeline: End-to-end image-to-mesh pipeline.
"""

from pathlib import Path
from typing import List, Optional, Union

import torch
import trimesh
from PIL import Image

from ..configs.base_config import GenerationConfig, PipelineConfig
from ..configs.presets import get_preset_config
from ..core.conditioner import ImageConditioner
from ..core.dit_model import Hunyuan3DDiT
from ..core.scheduler import FlowMatchingScheduler
from ..core.vae import ShapeVAE
from ..preprocessing.background_removal import BackgroundRemover
from ..preprocessing.image_processor import ImageProcessor
from .diffusion_runner import DiffusionRunner
from .model_loader import ModelLoader
from ..utils.logging import get_logger
from ..utils.timer import synchronize_timer
from ..utils.memory import MemoryOptimizer
from ..utils.device import DeviceManager

logger = get_logger("model_generator_v2.generation.pipeline")


class GeometryPipeline:
    """End-to-end single-image-to-3D-mesh generation pipeline.

    Combines all pipeline stages into a single callable that takes
    an input image and produces a clean triangle mesh.

    The pipeline stages are:
        1. Background removal (rembg)
        2. Image preprocessing (resize, normalize)
        3. Image conditioning (DINOv2 feature extraction)
        4. Diffusion denoising (DiT flow-matching)
        5. VAE decoding (latents → SDF grid)
        6. Surface extraction (Marching Cubes)

    Post-processing and export are handled separately by
    PostProcessingPipeline and exporters.

    Example:
        >>> pipeline = GeometryPipeline.from_pretrained(
        ...     'tencent/Hunyuan3D-2',
        ...     preset='ultra',
        ... )
        >>> mesh = pipeline('input.png')
        >>> mesh.export('output.glb')
    """

    def __init__(
        self,
        model: Hunyuan3DDiT,
        vae: ShapeVAE,
        conditioner: ImageConditioner,
        scheduler: FlowMatchingScheduler,
        background_remover: Optional[BackgroundRemover] = None,
        image_processor: Optional[ImageProcessor] = None,
        config: Optional[GenerationConfig] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.model = model
        self.vae = vae
        self.conditioner = conditioner
        self.scheduler = scheduler
        self.background_remover = background_remover or BackgroundRemover()
        self.image_processor = image_processor or ImageProcessor()
        self.config = config or GenerationConfig()

        dm = DeviceManager()
        self.device = device or dm.get_optimal_device()
        self.dtype = dtype

        self.diffusion_runner = DiffusionRunner(model, scheduler)

        logger.info(
            f"GeometryPipeline initialized: "
            f"device={self.device}, dtype={self.dtype}"
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
        """Load a pretrained pipeline from HuggingFace or local path.

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

        # Resolve dtype
        dtype = (
            torch.float16 if dtype_str == "float16" else torch.float32
        )

        # Load models
        loader = ModelLoader(
            device=device,
            dtype=dtype,
            enable_cpu_offload=enable_cpu_offload,
        )

        models = loader.load_from_pretrained(model_path)

        # Load conditioner's DINOv2 backbone
        dm = DeviceManager()
        actual_device = dm.get_optimal_device(
            device if device != "auto" else None
        )
        models["conditioner"].load_model(actual_device, dtype)

        return cls(
            model=models["model"],
            vae=models["vae"],
            conditioner=models["conditioner"],
            scheduler=models["scheduler"],
            config=gen_config,
            device=actual_device,
            dtype=dtype,
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
        # Resolve parameters (explicit > config)
        steps = num_inference_steps or self.config.num_inference_steps
        resolution = octree_resolution or self.config.octree_resolution
        guidance = guidance_scale or self.config.guidance_scale
        seed = seed if seed is not None else self.config.seed

        logger.info(
            f"Generating mesh: steps={steps}, "
            f"resolution={resolution}, guidance={guidance}"
        )

        # Stage 1: Background removal
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        if remove_background:
            with synchronize_timer("Background Removal"):
                image = self.background_remover(
                    image, target_size=self.image_processor.target_size
                )

        # Stage 2: Image preprocessing
        with synchronize_timer("Image Preprocessing"):
            pixel_values = self.image_processor.to_tensor(
                image, device=self.device, dtype=self.dtype
            )

        # Stage 3: Image conditioning
        with synchronize_timer("Image Conditioning"):
            condition_tokens = self.conditioner.encode_image(pixel_values)
            pooled = self.conditioner.get_pooled_embedding(pixel_values)

        MemoryOptimizer.log_memory_usage("After conditioning")

        # Stage 4: Prepare noise
        num_latent_tokens = 2048  # Hunyuan3D default
        latent_dim = self.model.in_channels
        noise = self.scheduler.sample_noise(
            shape=(1, num_latent_tokens, latent_dim),
            device=self.device,
            dtype=self.dtype,
            seed=seed,
        )

        # Stage 5: Diffusion denoising
        latent_tokens = self.diffusion_runner.run(
            noise=noise,
            condition_tokens=condition_tokens,
            num_inference_steps=steps,
            guidance_scale=guidance,
            pooled_projection=pooled,
            device=self.device,
            dtype=self.dtype,
            show_progress=show_progress,
        )

        MemoryOptimizer.log_memory_usage("After diffusion")

        # Stage 6: VAE decode → mesh
        mesh = self.vae.latents2mesh(
            latent_tokens=latent_tokens,
            octree_resolution=resolution,
        )

        if mesh is None:
            logger.error("Mesh generation failed — no surface extracted")
            return None

        logger.info(
            f"Generated raw mesh: "
            f"{len(mesh.vertices)} vertices, "
            f"{len(mesh.faces)} faces"
        )

        MemoryOptimizer.clear_cache()
        return mesh
