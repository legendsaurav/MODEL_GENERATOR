"""
Base configuration dataclasses for MODEL_GENERATOR_V2.

Defines all configuration structures used throughout the pipeline:
generation parameters, post-processing settings, export options,
and the unified pipeline config.

Dependencies:
    - dataclasses (stdlib)
    - typing (stdlib)

Classes:
    GenerationConfig: Diffusion model inference parameters.
    PostProcessingConfig: Mesh post-processing parameters.
    ExportConfig: Mesh export format and path settings.
    PipelineConfig: Unified config combining all sub-configs.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import json


@dataclass
class GenerationConfig:
    """Configuration for the shape diffusion generation stage.

    Controls the diffusion process quality, resolution, and
    computational settings.

    Attributes:
        num_inference_steps: Number of diffusion denoising steps.
            Higher values produce better quality but take longer.
        octree_resolution: Resolution of the octree grid for SDF
            evaluation. Controls geometric detail level.
        guidance_scale: Classifier-free guidance scale. Higher values
            produce outputs more aligned to the input image.
        seed: Random seed for reproducibility. None = random.
        dtype: Torch dtype string ('float16' or 'float32').
        device: Target device string ('cuda', 'cpu', 'auto').
        model_path: HuggingFace model ID or local path to weights.
        use_safetensors: Whether to load safetensors format.
        enable_cpu_offload: Whether to offload models to CPU when idle.
    """

    num_inference_steps: int = 50
    octree_resolution: int = 384
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    dtype: str = "float16"
    device: str = "auto"
    model_path: str = "tencent/Hunyuan3D-2"
    use_safetensors: bool = True
    enable_cpu_offload: bool = False

    def validate(self) -> None:
        """Validate configuration values are within acceptable ranges.

        Raises:
            ValueError: If any parameter is out of range.
        """
        if self.num_inference_steps < 1 or self.num_inference_steps > 500:
            raise ValueError(
                f"num_inference_steps must be 1-500, got {self.num_inference_steps}"
            )
        if self.octree_resolution < 64 or self.octree_resolution > 1024:
            raise ValueError(
                f"octree_resolution must be 64-1024, got {self.octree_resolution}"
            )
        if self.guidance_scale < 0:
            raise ValueError(
                f"guidance_scale must be >= 0, got {self.guidance_scale}"
            )
        if self.dtype not in ("float16", "float32"):
            raise ValueError(
                f"dtype must be 'float16' or 'float32', got {self.dtype}"
            )


@dataclass
class PostProcessingConfig:
    """Configuration for the mesh post-processing pipeline.

    Controls which post-processing steps are enabled and their
    parameters. Steps execute in order: repair → smooth →
    subdivide → decimate → validate.

    Attributes:
        enable_repair: Run mesh repair (degenerate faces, holes, normals).
        enable_smoothing: Run surface smoothing.
        smoothing_method: Smoothing algorithm ('taubin' or 'hc').
        smoothing_iterations: Number of smoothing passes.
        enable_subdivision: Run adaptive subdivision.
        subdivision_iterations: Maximum subdivision passes.
        enable_decimation: Run quadric edge collapse decimation.
        target_faces: Target face count after decimation.
        decimation_quality: Quality threshold for decimation (0-1).
        enable_validation: Run mesh validation checks.
        min_component_face_ratio: Min face ratio for connected components
            (smaller components are removed as floaters).
        max_hole_edges: Maximum edges in a hole to fill.
        preserve_boundary: Preserve mesh boundary during decimation.
    """

    enable_repair: bool = True
    enable_smoothing: bool = True
    smoothing_method: str = "taubin"
    smoothing_iterations: int = 5
    enable_subdivision: bool = False
    subdivision_iterations: int = 1
    enable_decimation: bool = True
    target_faces: int = 100000
    decimation_quality: float = 1.0
    enable_validation: bool = True
    min_component_face_ratio: float = 0.005
    max_hole_edges: int = 20
    preserve_boundary: bool = True

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if self.smoothing_method not in ("taubin", "hc"):
            raise ValueError(
                f"smoothing_method must be 'taubin' or 'hc', "
                f"got '{self.smoothing_method}'"
            )
        if self.smoothing_iterations < 0:
            raise ValueError("smoothing_iterations must be >= 0")
        if self.target_faces < 100:
            raise ValueError("target_faces must be >= 100")
        if not 0 <= self.decimation_quality <= 1:
            raise ValueError("decimation_quality must be 0-1")


@dataclass
class ExportConfig:
    """Configuration for mesh export.

    Attributes:
        formats: List of export formats to produce.
        output_dir: Directory for output files.
        filename_prefix: Prefix for output filenames.
        include_normals: Include vertex normals in exports.
        binary: Use binary format where supported (STL, PLY).
    """

    formats: List[str] = field(default_factory=lambda: ["glb"])
    output_dir: str = "./outputs"
    filename_prefix: str = "mesh"
    include_normals: bool = True
    binary: bool = True

    SUPPORTED_FORMATS = {"glb", "obj", "stl", "ply"}

    def validate(self) -> None:
        """Validate export configuration.

        Raises:
            ValueError: If any format is unsupported.
        """
        for fmt in self.formats:
            if fmt.lower() not in self.SUPPORTED_FORMATS:
                raise ValueError(
                    f"Unsupported format '{fmt}'. "
                    f"Supported: {self.SUPPORTED_FORMATS}"
                )


@dataclass
class PipelineConfig:
    """Unified configuration combining all pipeline sub-configs.

    This is the top-level config passed to the generation pipeline.

    Attributes:
        generation: Diffusion generation parameters.
        postprocessing: Mesh post-processing parameters.
        export: Export format and path settings.
    """

    generation: GenerationConfig = field(default_factory=GenerationConfig)
    postprocessing: PostProcessingConfig = field(
        default_factory=PostProcessingConfig
    )
    export: ExportConfig = field(default_factory=ExportConfig)

    def validate(self) -> None:
        """Validate all sub-configurations.

        Raises:
            ValueError: If any sub-config has invalid values.
        """
        self.generation.validate()
        self.postprocessing.validate()
        self.export.validate()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full config to a dictionary.

        Returns:
            Nested dictionary representation.
        """
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize the full config to a JSON string.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Deserialize a PipelineConfig from a dictionary.

        Args:
            data: Nested dictionary with config values.

        Returns:
            PipelineConfig instance.
        """
        gen = GenerationConfig(**data.get("generation", {}))
        post = PostProcessingConfig(**data.get("postprocessing", {}))
        export = ExportConfig(**data.get("export", {}))
        return cls(generation=gen, postprocessing=post, export=export)
