"""
Hunyuan3D Diffusion Transformer (DiT) model.

Adapted from Hunyuan3D-2.1 shapegen/models/dit_model.py.
Implements the dual-stream + single-stream flow-based diffusion
transformer architecture for 3D shape generation.

Architecture Overview:
    The DiT processes two token streams:
    1. Condition tokens (from DINOv2 image features)
    2. Latent tokens (3D shape latent representations)

    Dual-stream blocks process them with separate QKV projections
    but joint attention. Single-stream blocks concatenate and
    refine them together.

    The model uses flow-matching (not DDPM noise prediction) and
    omits positional embeddings for latent tokens since they
    represent unordered spatial features.

Dependencies:
    - torch
    - einops
    - math (stdlib)

Classes:
    Hunyuan3DDiTBlock: Single transformer block (dual or single stream).
    Hunyuan3DDiT: Full DiT model with both block types.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..utils.logging import get_logger

logger = get_logger("model_generator_v2.core.dit_model")


# ─────────────────────────────────────────────────────────────────────────── #
#  Helper modules                                                            #
# ─────────────────────────────────────────────────────────────────────────── #


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Args:
        dim: Feature dimension to normalize.
        eps: Small constant for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization with learned shift and scale.

    Modulates the normalized output using timestep embeddings,
    enabling the model to adjust behavior at each diffusion step.

    Args:
        dim: Feature dimension.
        num_modulation_params: Number of modulation vectors to produce.
    """

    def __init__(self, dim: int, num_modulation_params: int = 2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.linear = nn.Linear(dim, dim * num_modulation_params)
        self.num_params = num_modulation_params

    def forward(
        self, x: torch.Tensor, emb: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """Apply adaptive normalization.

        Args:
            x: Input tensor [B, N, D].
            emb: Timestep embedding [B, D].

        Returns:
            Tuple of (shift, scale, ...) modulation tensors.
        """
        params = self.linear(F.silu(emb)).unsqueeze(1)
        params = params.chunk(self.num_params, dim=-1)
        x_norm = self.norm(x)
        return (x_norm,) + params


class FeedForward(nn.Module):
    """Transformer feed-forward network with GELU activation.

    Args:
        dim: Input/output dimension.
        mult: Hidden dimension multiplier.
    """

    def __init__(self, dim: int, mult: float = 4.0) -> None:
        super().__init__()
        hidden = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    """Multi-head self/cross-attention with optional QK normalization.

    Args:
        dim: Model dimension.
        num_heads: Number of attention heads.
        qk_norm: Whether to apply RMS normalization to Q and K.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)
        self.to_out = nn.Linear(dim, dim)

        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute multi-head attention.

        Args:
            x: Query tensor [B, N, D].
            context: Optional context for cross-attention [B, M, D].

        Returns:
            Attention output [B, N, D].
        """
        B, N, _ = x.shape

        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        if context is not None:
            # Cross-attention: use context for K, V
            ctx_qkv = self.to_qkv(context)
            _, k, v = ctx_qkv.chunk(3, dim=-1)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Scaled dot-product attention (uses Flash Attention if available)
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)


# ─────────────────────────────────────────────────────────────────────────── #
#  DiT Blocks                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #


class DualStreamBlock(nn.Module):
    """Dual-stream transformer block processing condition and latent tokens.

    Each stream has its own QKV projections and MLP layers, but they
    share the attention computation (joint attention matrix).

    Args:
        dim: Model dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Feed-forward hidden dimension multiplier.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        # Latent stream
        self.latent_attn = Attention(dim, num_heads)
        self.latent_ff = FeedForward(dim, mlp_ratio)
        self.latent_norm1 = AdaLayerNorm(dim, num_modulation_params=6)

        # Condition stream
        self.cond_attn = Attention(dim, num_heads)
        self.cond_ff = FeedForward(dim, mlp_ratio)
        self.cond_norm1 = AdaLayerNorm(dim, num_modulation_params=6)

    def forward(
        self,
        latent: torch.Tensor,
        cond: torch.Tensor,
        temb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process both streams with joint attention.

        Args:
            latent: Latent tokens [B, N_lat, D].
            cond: Condition tokens [B, N_cond, D].
            temb: Timestep embedding [B, D].

        Returns:
            Tuple of (updated_latent, updated_cond).
        """
        # Adaptive norm + modulation for both streams
        lat_normed, lat_shift1, lat_scale1, lat_gate1, \
            lat_shift2, lat_scale2, lat_gate2 = self.latent_norm1(latent, temb)
        cnd_normed, cnd_shift1, cnd_scale1, cnd_gate1, \
            cnd_shift2, cnd_scale2, cnd_gate2 = self.cond_norm1(cond, temb)

        # Modulate
        lat_mod = lat_normed * (1 + lat_scale1) + lat_shift1
        cnd_mod = cnd_normed * (1 + cnd_scale1) + cnd_shift1

        # Joint attention (concatenate, attend, split)
        joint = torch.cat([lat_mod, cnd_mod], dim=1)
        joint_out = self.latent_attn(joint)
        lat_attn_out = joint_out[:, :latent.shape[1], :]
        cnd_attn_out = joint_out[:, latent.shape[1]:, :]

        # Gate + residual for attention
        latent = latent + lat_gate1 * lat_attn_out
        cond = cond + cnd_gate1 * cnd_attn_out

        # FFN with modulation
        lat_ff_in = latent * (1 + lat_scale2) + lat_shift2
        cnd_ff_in = cond * (1 + cnd_scale2) + cnd_shift2

        latent = latent + lat_gate2 * self.latent_ff(lat_ff_in)
        cond = cond + cnd_gate2 * self.cond_ff(cnd_ff_in)

        return latent, cond


class SingleStreamBlock(nn.Module):
    """Single-stream block processing concatenated tokens.

    After dual-stream processing, latent and condition tokens are
    concatenated and refined through single-stream blocks.

    Args:
        dim: Model dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Feed-forward hidden dimension multiplier.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.attn = Attention(dim, num_heads)
        self.ff = FeedForward(dim, mlp_ratio)
        self.norm = AdaLayerNorm(dim, num_modulation_params=3)

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
    ) -> torch.Tensor:
        """Process concatenated tokens.

        Args:
            x: Concatenated tokens [B, N_total, D].
            temb: Timestep embedding [B, D].

        Returns:
            Processed tokens [B, N_total, D].
        """
        x_norm, shift, scale, gate = self.norm(x, temb)
        x_mod = x_norm * (1 + scale) + shift

        attn_out = self.attn(x_mod)
        ff_out = self.ff(x_mod)

        x = x + gate * (attn_out + ff_out)
        return x


# ─────────────────────────────────────────────────────────────────────────── #
#  Main DiT Model                                                             #
# ─────────────────────────────────────────────────────────────────────────── #


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding with MLP projection.

    Args:
        dim: Output embedding dimension.
        max_period: Maximum period for sinusoidal encoding.
    """

    def __init__(self, dim: int, max_period: int = 10000) -> None:
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed scalar timesteps.

        Args:
            t: Timestep tensor [B] with values in [0, 1].

        Returns:
            Timestep embedding [B, dim].
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, device=t.device, dtype=t.dtype)
            / half_dim
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class Hunyuan3DDiT(nn.Module):
    """Hunyuan3D Diffusion Transformer for 3D shape generation.

    Dual-stream + single-stream architecture using flow-matching.
    Generates 3D shape latent tokens conditioned on image features.

    This model is weight-compatible with official Hunyuan3D-2.1
    pretrained checkpoints.

    Args:
        in_channels: Latent token channel dimension.
        out_channels: Output channel dimension.
        num_attention_heads: Number of attention heads per block.
        attention_head_dim: Dimension per attention head.
        num_layers: Number of dual-stream blocks.
        num_single_layers: Number of single-stream blocks.
        conditioning_embedding_out_dim: Dimension for condition embeddings.
        pooled_projection_dim: Dimension for pooled image projection.
        guidance_embeds: Whether to use guidance scale embedding.
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        num_attention_heads: int = 16,
        attention_head_dim: int = 128,
        num_layers: int = 24,
        num_single_layers: int = 12,
        conditioning_embedding_out_dim: int = 2048,
        pooled_projection_dim: int = 2048,
        guidance_embeds: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inner_dim = inner_dim
        self.num_layers = num_layers
        self.num_single_layers = num_single_layers
        self.guidance_embeds = guidance_embeds

        # Input projection for latent tokens
        self.latent_in = nn.Linear(in_channels, inner_dim)

        # Condition input projection
        self.cond_in = nn.Linear(conditioning_embedding_out_dim, inner_dim)

        # Timestep embedding
        self.time_embed = TimestepEmbedding(inner_dim)

        # Optional guidance embedding
        if guidance_embeds:
            self.guidance_embed = TimestepEmbedding(inner_dim)

        # Pooled text/image projection
        self.pooled_proj = nn.Linear(pooled_projection_dim, inner_dim)

        # Dual-stream blocks
        self.dual_stream_blocks = nn.ModuleList(
            [
                DualStreamBlock(inner_dim, num_attention_heads)
                for _ in range(num_layers)
            ]
        )

        # Single-stream blocks
        self.single_stream_blocks = nn.ModuleList(
            [
                SingleStreamBlock(inner_dim, num_attention_heads)
                for _ in range(num_single_layers)
            ]
        )

        # Final output projection
        self.final_norm = nn.LayerNorm(inner_dim)
        self.final_proj = nn.Linear(inner_dim, out_channels)

        logger.info(
            f"Hunyuan3DDiT initialized: "
            f"dim={inner_dim}, "
            f"dual_blocks={num_layers}, "
            f"single_blocks={num_single_layers}, "
            f"heads={num_attention_heads}"
        )

    def forward(
        self,
        latent_tokens: torch.Tensor,
        condition_tokens: torch.Tensor,
        timestep: torch.Tensor,
        pooled_projection: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the full DiT.

        Args:
            latent_tokens: Noisy latent tokens [B, N_lat, C_in].
            condition_tokens: Image condition tokens [B, N_cond, C_cond].
            timestep: Diffusion timestep [B].
            pooled_projection: Pooled image embedding [B, C_pool].
            guidance: Guidance scale embedding [B].

        Returns:
            Predicted velocity/noise [B, N_lat, C_out].
        """
        # Project inputs
        latent = self.latent_in(latent_tokens)
        cond = self.cond_in(condition_tokens)

        # Build timestep embedding
        temb = self.time_embed(timestep)

        if self.guidance_embeds and guidance is not None:
            temb = temb + self.guidance_embed(guidance)

        if pooled_projection is not None:
            temb = temb + self.pooled_proj(pooled_projection)

        # Dual-stream blocks
        for block in self.dual_stream_blocks:
            latent, cond = block(latent, cond, temb)

        # Concatenate for single-stream processing
        x = torch.cat([latent, cond], dim=1)

        # Single-stream blocks
        for block in self.single_stream_blocks:
            x = block(x, temb)

        # Extract latent tokens only
        latent_out = x[:, : latent_tokens.shape[1], :]

        # Final projection
        latent_out = self.final_norm(latent_out)
        output = self.final_proj(latent_out)

        return output
