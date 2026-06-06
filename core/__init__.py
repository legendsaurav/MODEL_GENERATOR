# MODEL_GENERATOR_V2 - Core ML Package
# Contains the DiT model, VAE, conditioner, and scheduler components
# adapted from Hunyuan3D-2.1 for geometry-only generation.

from .conditioner import ImageConditioner
from .dit_model import Hunyuan3DDiT
from .scheduler import FlowMatchingScheduler

__all__ = [
    "ImageConditioner",
    "Hunyuan3DDiT",
    "FlowMatchingScheduler",
]
