"""
Fourier embedding and attention blocks for the ShapeVAE.

Adapted from Hunyuan3D-2.1 shapegen/models/autoencoders/attention_blocks.py.
Provides positional encoding via Fourier features and cross-attention
processors used in the VAE's encoder and decoder.

Dependencies:
    - torch
    - math (stdlib)

Classes:
    FourierEmbedder: Fourier feature positional encoding for 3D coordinates.
    CrossAttentionBlock: Cross-attention between queries and context.
    SelfAttentionBlock: Self-attention for token sequences.
    TransformerBlock: Combined self-attention + cross-attention + FFN.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ...utils.logging import get_logger

logger = get_logger("model_generator_v2.core.vae.attention")


class FourierEmbedder(nn.Module):
    """Fourier feature encoding for 3D spatial coordinates.

    Maps continuous 3D coordinates to higher-dimensional Fourier
    features, enabling the network to represent high-frequency
    spatial details in the SDF prediction.

    Uses the formula: [sin(2π·f·x), cos(2π·f·x)] for each frequency f
    across the frequency bank.

    Args:
        num_freqs: Number of frequency bands.
        input_dim: Dimensionality of input coordinates (3 for xyz).
        include_pi: Whether to multiply frequencies by π.
        include_input: Whether to append the raw input coordinates.

    Attributes:
        out_dim: Total output dimension after Fourier expansion.
    """

    def __init__(
        self,
        num_freqs: int = 8,
        input_dim: int = 3,
        include_pi: bool = True,
        include_input: bool = True,
    ) -> None:
        super().__init__()
        self.num_freqs = num_freqs
        self.input_dim = input_dim
        self.include_input = include_input

        # Build frequency bank: 2^0, 2^1, ..., 2^(num_freqs-1)
        freq_bands = 2.0 ** torch.arange(num_freqs).float()
        if include_pi:
            freq_bands = freq_bands * math.pi

        # Register as buffer (not a parameter, but moves with .to())
        self.register_buffer("freq_bands", freq_bands)

        # Calculate output dimension
        self.out_dim = input_dim * num_freqs * 2  # sin + cos
        if include_input:
            self.out_dim += input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Fourier features for input coordinates.

        Args:
            x: Input coordinates tensor [..., input_dim].

        Returns:
            Fourier features tensor [..., out_dim].
        """
        # x: [..., D]
        # freq_bands: [F]
        # Outer product gives [..., D, F]
        proj = x.unsqueeze(-1) * self.freq_bands
        # Flatten last two dims: [..., D*F]
        proj = proj.reshape(*x.shape[:-1], -1)

        # Concatenate sin and cos
        features = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

        if self.include_input:
            features = torch.cat([x, features], dim=-1)

        return features


class SelfAttentionBlock(nn.Module):
    """Multi-head self-attention block with pre-norm.

    Args:
        dim: Model dimension.
        num_heads: Number of attention heads.
        dropout: Attention dropout rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention with residual connection.

        Args:
            x: Input tensor [B, N, D].

        Returns:
            Output tensor [B, N, D].
        """
        residual = x
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads),
            qkv,
        )

        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.dropout(self.to_out(out))

        return residual + out


class CrossAttentionBlock(nn.Module):
    """Multi-head cross-attention block with pre-norm.

    Args:
        dim: Query dimension.
        context_dim: Context (key/value) dimension.
        num_heads: Number of attention heads.
        dropout: Attention dropout rate.
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_ctx = nn.LayerNorm(context_dim)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(context_dim, dim * 2)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """Apply cross-attention with residual connection.

        Args:
            x: Query tensor [B, N, D].
            context: Context tensor [B, M, C].

        Returns:
            Output tensor [B, N, D].
        """
        residual = x
        x = self.norm_q(x)
        context = self.norm_ctx(context)

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.dropout(self.to_out(out))

        return residual + out


class FeedForwardBlock(nn.Module):
    """Feed-forward block with GELU activation and pre-norm.

    Args:
        dim: Input/output dimension.
        mult: Hidden dimension multiplier.
        dropout: Dropout rate.
    """

    def __init__(
        self, dim: int, mult: float = 4.0, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        hidden = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward with residual connection.

        Args:
            x: Input tensor [B, N, D].

        Returns:
            Output tensor [B, N, D].
        """
        return x + self.net(self.norm(x))


class TransformerBlock(nn.Module):
    """Full transformer block: self-attention + cross-attention + FFN.

    Used in the ShapeVAE decoder to process latent tokens
    conditioned on 3D coordinate queries.

    Args:
        dim: Model dimension.
        context_dim: Context dimension for cross-attention.
        num_heads: Number of attention heads.
        mlp_mult: FFN hidden dimension multiplier.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        dim: int,
        context_dim: Optional[int] = None,
        num_heads: int = 8,
        mlp_mult: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.self_attn = SelfAttentionBlock(dim, num_heads, dropout)
        self.cross_attn = (
            CrossAttentionBlock(dim, context_dim or dim, num_heads, dropout)
            if context_dim
            else None
        )
        self.ff = FeedForwardBlock(dim, mlp_mult, dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply transformer block.

        Args:
            x: Input tensor [B, N, D].
            context: Optional context for cross-attention [B, M, C].

        Returns:
            Output tensor [B, N, D].
        """
        x = self.self_attn(x)
        if self.cross_attn is not None and context is not None:
            x = self.cross_attn(x, context)
        x = self.ff(x)
        return x
