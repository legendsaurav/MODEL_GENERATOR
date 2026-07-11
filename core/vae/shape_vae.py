"""
Shape Variational Autoencoder (ShapeVAE) for MODEL_GENERATOR_V2.

Adapted from Hunyuan3D-2.1 shapegen/models/autoencoders.
The ShapeVAE decodes latent token sequences into 3D Signed Distance
Function (SDF) grids, which are then converted to meshes via
Marching Cubes.

Architecture:
    - Decoder takes latent tokens + Fourier-encoded 3D query coordinates
    - Cross-attention between query positions and latent tokens
    - Predicts SDF value at each query position
    - Octree-based evaluation for efficient high-resolution grids

Dependencies:
    - torch
    - numpy
    - trimesh

Classes:
    ShapeVAEDecoder: Transformer decoder predicting SDF from latents.
    ShapeVAE: Full VAE with decode() and latents2mesh() methods.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import trimesh

from .attention_blocks import (
    FourierEmbedder,
    TransformerBlock,
)
from .surface_extractor import SurfaceExtractor
from ...utils.logging import get_logger
from ...utils.timer import synchronize_timer

logger = get_logger("model_generator_v2.core.vae.shape_vae")


class ShapeVAEDecoder(nn.Module):
    """Transformer decoder that predicts SDF values from latent tokens.

    Given a set of 3D query coordinates (Fourier-encoded) and a
    sequence of latent tokens from the DiT, predicts the signed
    distance value at each query point.

    Args:
        latent_dim: Dimension of input latent tokens from DiT.
        embed_dim: Internal embedding dimension.
        num_heads: Number of attention heads.
        depth: Number of transformer blocks.
        num_freqs: Number of Fourier frequencies for coordinate encoding.
        include_pi: Whether Fourier frequencies include π multiplier.
        use_ln_post: Whether to apply LayerNorm after final block.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        embed_dim: int = 1024,
        num_heads: int = 16,
        depth: int = 12,
        num_freqs: int = 8,
        include_pi: bool = True,
        use_ln_post: bool = True,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.embed_dim = embed_dim

        # Fourier encoding for 3D query coordinates
        self.fourier = FourierEmbedder(
            num_freqs=num_freqs,
            input_dim=3,
            include_pi=include_pi,
        )

        # Project Fourier features to embed_dim
        self.query_proj = nn.Linear(self.fourier.out_dim, embed_dim)

        # Project latent tokens to embed_dim
        self.latent_proj = nn.Linear(latent_dim, embed_dim)

        # Transformer blocks with cross-attention to latent tokens
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    context_dim=embed_dim,
                    num_heads=num_heads,
                )
                for _ in range(depth)
            ]
        )

        # Optional post-LayerNorm
        self.ln_post = nn.LayerNorm(embed_dim) if use_ln_post else nn.Identity()

        # SDF prediction head: embed_dim → 1
        self.sdf_head = nn.Linear(embed_dim, 1)

    def forward(
        self,
        query_coords: torch.Tensor,
        latent_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Predict SDF values at query coordinates.

        Args:
            query_coords: 3D coordinates [B, N_query, 3].
            latent_tokens: Latent tokens from DiT [B, N_lat, latent_dim].

        Returns:
            SDF predictions [B, N_query, 1].
        """
        # Encode query positions
        query_features = self.fourier(query_coords)
        query_features = self.query_proj(query_features)

        # Project latent context
        context = self.latent_proj(latent_tokens)

        # Transformer blocks with cross-attention
        x = query_features
        for block in self.blocks:
            x = block(x, context=context)

        x = self.ln_post(x)
        sdf = self.sdf_head(x)

        return sdf


class ShapeVAE(nn.Module):
    """Shape Variational Autoencoder for 3D geometry generation.

    Decodes DiT-generated latent tokens into SDF grids and extracts
    triangle meshes via Marching Cubes.

    The decode path:
        latent_tokens → ShapeVAEDecoder → SDF values
        SDF values → octree grid → Marching Cubes → triangle mesh

    Args:
        latent_dim: Dimension of latent tokens.
        embed_dim: Decoder embedding dimension.
        num_heads: Number of attention heads.
        depth: Number of decoder transformer blocks.
        num_freqs: Number of Fourier frequencies.
        include_pi: Whether to include π in Fourier frequencies.
        use_ln_post: Whether to use post-LayerNorm.

    Example:
        >>> vae = ShapeVAE(latent_dim=64)
        >>> mesh = vae.latents2mesh(latent_tokens, octree_resolution=384)
    """

    def __init__(
        self,
        latent_dim: int = 64,
        embed_dim: int = 1024,
        num_heads: int = 16,
        depth: int = 12,
        num_freqs: int = 8,
        include_pi: bool = True,
        use_ln_post: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.decoder = ShapeVAEDecoder(
            latent_dim=latent_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            num_freqs=num_freqs,
            include_pi=include_pi,
            use_ln_post=use_ln_post,
        )

        self.surface_extractor = SurfaceExtractor()

        logger.info(
            f"ShapeVAE initialized: latent_dim={latent_dim}, "
            f"embed_dim={embed_dim}, depth={depth}"
        )

    @synchronize_timer("VAE Decode")
    @torch.no_grad()
    def decode_to_sdf_grid(
        self,
        latent_tokens: torch.Tensor,
        resolution: int = 384,
        batch_size: int = 2 ** 14,
    ) -> np.ndarray:
        """Decode latent tokens to a full SDF grid.

        Evaluates the SDF at every point in a regular 3D grid by
        batching queries to avoid OOM on high-resolution grids.

        Args:
            latent_tokens: Latent tokens from DiT [1, N, latent_dim].
            resolution: Grid resolution (R). Produces R³ query points.
            batch_size: Number of query points per batch.

        Returns:
            SDF grid as numpy array [R, R, R].
        """
        device = latent_tokens.device
        dtype = latent_tokens.dtype

        # Create regular 3D grid coordinates
        coords_1d = torch.linspace(-0.5, 0.5, resolution, device=device, dtype=dtype)
        grid_x, grid_y, grid_z = torch.meshgrid(
            coords_1d, coords_1d, coords_1d, indexing="ij"
        )
        query_points = torch.stack(
            [grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=-1
        )  # [R³, 3]

        total_points = query_points.shape[0]
        sdf_values = torch.zeros(total_points, device=device, dtype=dtype)

        logger.info(
            f"Evaluating SDF on {resolution}³ grid "
            f"({total_points:,} points, batch_size={batch_size:,})"
        )

        # Batch evaluation
        for start in range(0, total_points, batch_size):
            end = min(start + batch_size, total_points)
            batch_coords = query_points[start:end].unsqueeze(0)  # [1, B, 3]

            sdf_batch = self.decoder(batch_coords, latent_tokens)
            sdf_values[start:end] = sdf_batch.squeeze(0).squeeze(-1)

        # Reshape to 3D grid
        sdf_grid = sdf_values.reshape(resolution, resolution, resolution)
        return sdf_grid.cpu().float().numpy()

    @synchronize_timer("Latents to Mesh")
    def latents2mesh(
        self,
        latent_tokens: torch.Tensor,
        octree_resolution: int = 384,
        batch_size: int = 2 ** 14,
    ) -> Optional[trimesh.Trimesh]:
        """Full pipeline: latent tokens → SDF grid → triangle mesh.

        Args:
            latent_tokens: DiT output latent tokens [1, N, latent_dim].
            octree_resolution: Resolution for SDF grid evaluation.
            batch_size: Query batch size for memory efficiency.

        Returns:
            A trimesh.Trimesh object, or None if extraction fails.
        """
        sdf_grid = self.decode_to_sdf_grid(
            latent_tokens, octree_resolution, batch_size
        )

        mesh_output = self.surface_extractor(sdf_grid)
        if mesh_output is None:
            return None

        # Reverse face winding to match Hunyuan3D convention
        mesh_output.mesh_f = mesh_output.mesh_f[:, ::-1]
        mesh = trimesh.Trimesh(
            vertices=mesh_output.mesh_v,
            faces=mesh_output.mesh_f,
            process=False,
        )

        logger.info(
            f"Generated mesh: {len(mesh.vertices)} vertices, "
            f"{len(mesh.faces)} faces"
        )
        return mesh
