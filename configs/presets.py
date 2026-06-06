"""
Quality presets for MODEL_GENERATOR_V2.

Provides three predefined quality tiers (FAST, BALANCED, ULTRA) that
configure the entire pipeline — from diffusion steps to post-processing
intensity to export settings.

Dependencies:
    - base_config (local)

Classes:
    Preset: Enum-like class for preset names.

Functions:
    get_preset_config: Factory function returning a PipelineConfig for a preset.
"""

from enum import Enum
from typing import Dict

from .base_config import (
    GenerationConfig,
    PostProcessingConfig,
    ExportConfig,
    PipelineConfig,
)


class Preset(str, Enum):
    """Quality preset identifiers.

    Attributes:
        FAST: Low-quality, fast generation (~15s on A100).
        BALANCED: Medium-quality, moderate speed (~45s on A100).
        ULTRA: Maximum quality, slower generation (~120s on A100).
    """

    FAST = "fast"
    BALANCED = "balanced"
    ULTRA = "ultra"


# --------------------------------------------------------------------------- #
#  Preset definitions                                                         #
# --------------------------------------------------------------------------- #

_PRESET_CONFIGS: Dict[Preset, PipelineConfig] = {
    Preset.FAST: PipelineConfig(
        generation=GenerationConfig(
            num_inference_steps=25,
            octree_resolution=256,
            guidance_scale=7.5,
            dtype="float16",
        ),
        postprocessing=PostProcessingConfig(
            enable_repair=True,
            enable_smoothing=True,
            smoothing_method="taubin",
            smoothing_iterations=3,
            enable_subdivision=False,
            subdivision_iterations=0,
            enable_decimation=True,
            target_faces=50000,
            enable_validation=True,
            max_hole_edges=10,
        ),
        export=ExportConfig(formats=["glb"]),
    ),
    Preset.BALANCED: PipelineConfig(
        generation=GenerationConfig(
            num_inference_steps=50,
            octree_resolution=384,
            guidance_scale=7.5,
            dtype="float16",
        ),
        postprocessing=PostProcessingConfig(
            enable_repair=True,
            enable_smoothing=True,
            smoothing_method="taubin",
            smoothing_iterations=5,
            enable_subdivision=True,
            subdivision_iterations=1,
            enable_decimation=True,
            target_faces=100000,
            enable_validation=True,
            max_hole_edges=20,
        ),
        export=ExportConfig(formats=["glb"]),
    ),
    Preset.ULTRA: PipelineConfig(
        generation=GenerationConfig(
            num_inference_steps=100,
            octree_resolution=512,
            guidance_scale=7.5,
            dtype="float16",
        ),
        postprocessing=PostProcessingConfig(
            enable_repair=True,
            enable_smoothing=True,
            smoothing_method="taubin",
            smoothing_iterations=10,
            enable_subdivision=True,
            subdivision_iterations=2,
            enable_decimation=True,
            target_faces=200000,
            enable_validation=True,
            max_hole_edges=50,
        ),
        export=ExportConfig(formats=["glb", "obj"]),
    ),
}


def get_preset_config(preset: str | Preset) -> PipelineConfig:
    """Get a full PipelineConfig for a named quality preset.

    Args:
        preset: Preset name string ('fast', 'balanced', 'ultra')
                or a Preset enum value.

    Returns:
        A PipelineConfig instance with all parameters set for the
        requested quality tier.

    Raises:
        ValueError: If the preset name is not recognized.

    Example:
        >>> config = get_preset_config('ultra')
        >>> config.generation.num_inference_steps
        100
    """
    if isinstance(preset, str):
        try:
            preset = Preset(preset.lower())
        except ValueError:
            valid = [p.value for p in Preset]
            raise ValueError(
                f"Unknown preset '{preset}'. Valid presets: {valid}"
            )

    if preset not in _PRESET_CONFIGS:
        raise ValueError(f"No configuration defined for preset: {preset}")

    import copy
    return copy.deepcopy(_PRESET_CONFIGS[preset])


def list_presets() -> list[str]:
    """List all available preset names.

    Returns:
        List of preset name strings.
    """
    return [p.value for p in Preset]
