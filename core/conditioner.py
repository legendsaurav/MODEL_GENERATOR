"""
DINOv2-based image conditioner for shape generation.

Adapted from Hunyuan3D-2.1 shapegen/models/conditioner.py.
Encodes a single input image into condition embeddings that guide
the DiT diffusion model during shape generation.

The conditioner uses DINOv2-Giant as a frozen image encoder,
extracting semantic features at 518×518 resolution. These features
are projected to the DiT's conditioning dimension via a learned
linear layer.

Dependencies:
    - torch
    - transformers (DINOv2)
    - PIL

Classes:
    ImageConditioner: Encodes images into DiT-compatible condition tokens.
"""

import torch
import torch.nn as nn
from typing import Optional, Union
from PIL import Image

from ..utils.logging import get_logger

logger = get_logger("model_generator_v2.core.conditioner")


class ImageConditioner(nn.Module):
    """DINOv2-based image feature extractor for shape conditioning.

    Encodes an input image through a frozen DINOv2-Giant backbone
    to produce semantic embeddings that condition the diffusion
    transformer's shape generation process.

    The architecture uses DINOv2's patch tokens (excluding CLS) as
    a sequence of spatial features, then projects them to match
    the DiT's expected conditioning dimension.

    Args:
        model_name: HuggingFace model identifier for DINOv2.
        freeze: Whether to freeze the DINOv2 backbone weights.
        output_dim: Projection output dimension matching DiT conditioning.
        input_resolution: Input image resolution (DINOv2 uses 518).

    Attributes:
        backbone: The frozen DINOv2 model.
        projection: Linear layer projecting DINOv2 features to output_dim.
        input_resolution: Expected input image size.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-giant",
        freeze: bool = True,
        output_dim: int = 2048,
        input_resolution: int = 518,
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self._model_name = model_name
        self._freeze = freeze

        # Lazy initialization — actual model loading happens in load_model()
        self.backbone: Optional[nn.Module] = None
        self.projection: Optional[nn.Linear] = None
        self._hidden_dim: int = 0

    def load_model(self, device: torch.device, dtype: torch.dtype) -> None:
        """Load the DINOv2 backbone and create the projection layer.

        This is called during pipeline initialization, not in __init__,
        to support deferred loading and CPU offloading.

        Args:
            device: Device to load the model onto.
            dtype: Data type for model parameters.
        """
        from transformers import AutoModel, AutoConfig

        logger.info(f"Loading DINOv2 backbone: {self._model_name}")

        config = AutoConfig.from_pretrained(self._model_name)
        self._hidden_dim = config.hidden_size

        self.backbone = AutoModel.from_pretrained(
            self._model_name,
            torch_dtype=dtype,
        )

        if self._freeze:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.projection = nn.Linear(
            self._hidden_dim, self.output_dim, bias=True
        )

        self.to(device=device, dtype=dtype)
        logger.info(
            f"Conditioner loaded: DINOv2 hidden_dim={self._hidden_dim} "
            f"→ output_dim={self.output_dim}"
        )

    def load_state_dict_partial(
        self, state_dict: dict, strict: bool = False
    ) -> None:
        """Load conditioner weights, handling projection layer mapping.

        Supports loading from Hunyuan3D checkpoint format where
        keys may be prefixed differently.

        Args:
            state_dict: State dictionary with conditioner weights.
            strict: Whether to enforce strict key matching.
        """
        # Filter to only our keys
        own_keys = set(self.state_dict().keys())
        filtered = {}
        for key, value in state_dict.items():
            clean_key = key.replace("conditioner.", "")
            if clean_key in own_keys:
                filtered[clean_key] = value

        if filtered:
            missing, unexpected = self.load_state_dict(
                filtered, strict=False
            )
            if missing:
                logger.warning(f"Missing conditioner keys: {missing[:5]}...")
        else:
            logger.warning("No matching conditioner keys found in checkpoint")

    @torch.no_grad()
    def encode_image(
        self, pixel_values: torch.Tensor
    ) -> torch.Tensor:
        """Encode preprocessed image pixels into condition embeddings.

        Args:
            pixel_values: Preprocessed image tensor of shape
                [B, 3, H, W] normalized for DINOv2.

        Returns:
            Condition embedding tensor of shape
            [B, num_patches, output_dim].
        """
        if self.backbone is None:
            raise RuntimeError(
                "Conditioner not loaded. Call load_model() first."
            )

        # DINOv2 forward — extract patch tokens
        outputs = self.backbone(pixel_values=pixel_values)
        # Use last_hidden_state, excluding CLS token (index 0)
        features = outputs.last_hidden_state[:, 1:, :]

        # Project to DiT conditioning dimension
        condition = self.projection(features)

        return condition

    def forward(
        self, pixel_values: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass — alias for encode_image.

        Args:
            pixel_values: Preprocessed image tensor [B, 3, H, W].

        Returns:
            Condition embeddings [B, num_patches, output_dim].
        """
        return self.encode_image(pixel_values)

    def get_pooled_embedding(
        self, pixel_values: torch.Tensor
    ) -> torch.Tensor:
        """Get a single pooled embedding vector per image.

        Used for the DiT's pooled projection input (timestep
        conditioning).

        Args:
            pixel_values: Preprocessed image tensor [B, 3, H, W].

        Returns:
            Pooled embedding tensor of shape [B, output_dim].
        """
        if self.backbone is None:
            raise RuntimeError(
                "Conditioner not loaded. Call load_model() first."
            )

        outputs = self.backbone(pixel_values=pixel_values)
        # Use CLS token for pooled representation
        cls_token = outputs.last_hidden_state[:, 0, :]
        pooled = self.projection(cls_token)

        return pooled
