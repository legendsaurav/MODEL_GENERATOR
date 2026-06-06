# MODEL_GENERATOR_V2 - VAE Package
# Shape Variational Autoencoder components adapted from Hunyuan3D-2.1.
# Handles encoding 3D surfaces to latents and decoding latents to meshes.

from .shape_vae import ShapeVAE, Latent2MeshOutput
from .surface_extractor import SurfaceExtractor
from .attention_blocks import FourierEmbedder

__all__ = [
    "ShapeVAE",
    "Latent2MeshOutput",
    "SurfaceExtractor",
    "FourierEmbedder",
]
