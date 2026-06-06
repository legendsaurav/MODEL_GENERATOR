"""
Image preprocessing and validation for MODEL_GENERATOR_V2.

Handles image loading, validation, normalization, and conversion
to model-ready tensors for the DINOv2 conditioner.

Dependencies:
    - PIL
    - torch
    - torchvision
    - numpy

Classes:
    ImageProcessor: Validates and converts images to model tensors.
"""

from typing import Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..utils.logging import get_logger

logger = get_logger("model_generator_v2.preprocessing.image_processor")

# DINOv2 ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageProcessor:
    """Validates, resizes, and normalizes images for the conditioner.

    Converts input images into normalized torch tensors compatible
    with the DINOv2 backbone's expected input format.

    Args:
        target_size: Target image resolution (square).
        normalize: Whether to apply ImageNet normalization.
        background_color: Background fill color for padding.

    Example:
        >>> processor = ImageProcessor(target_size=518)
        >>> tensor = processor.to_tensor(pil_image, device='cuda')
        >>> condition = conditioner(tensor)
    """

    def __init__(
        self,
        target_size: int = 518,
        normalize: bool = True,
        background_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        self.target_size = target_size
        self.normalize = normalize
        self.background_color = background_color

    def validate_image(
        self, image: Union[Image.Image, str, Path, np.ndarray]
    ) -> Image.Image:
        """Load and validate an input image.

        Accepts file paths, PIL Images, and numpy arrays. Ensures
        the image is valid and convertible to RGB.

        Args:
            image: Input in any supported format.

        Returns:
            Validated PIL Image in RGB mode.

        Raises:
            ValueError: If the image is invalid or unreadable.
        """
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise ValueError(f"Image file not found: {path}")
            try:
                image = Image.open(path)
            except Exception as e:
                raise ValueError(f"Cannot open image {path}: {e}")

        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = Image.fromarray(image, mode="L").convert("RGB")
            elif image.ndim == 3:
                if image.shape[2] == 4:
                    image = Image.fromarray(image, mode="RGBA")
                else:
                    image = Image.fromarray(image, mode="RGB")
            else:
                raise ValueError(
                    f"Invalid array shape: {image.shape}. "
                    f"Expected (H,W), (H,W,3), or (H,W,4)."
                )

        if not isinstance(image, Image.Image):
            raise ValueError(
                f"Unsupported image type: {type(image)}. "
                f"Expected PIL Image, file path, or numpy array."
            )

        # Convert RGBA to RGB with background
        if image.mode == "RGBA":
            bg = Image.new("RGB", image.size, self.background_color)
            bg.paste(image, mask=image.split()[3])
            image = bg
        elif image.mode != "RGB":
            image = image.convert("RGB")

        if image.size[0] < 10 or image.size[1] < 10:
            raise ValueError(
                f"Image too small: {image.size}. Minimum 10×10 pixels."
            )

        return image

    def resize(self, image: Image.Image) -> Image.Image:
        """Resize image to target resolution.

        Args:
            image: Input PIL Image.

        Returns:
            Resized image at (target_size × target_size).
        """
        if image.size != (self.target_size, self.target_size):
            image = image.resize(
                (self.target_size, self.target_size),
                Image.LANCZOS,
            )
        return image

    def to_tensor(
        self,
        image: Union[Image.Image, str, Path, np.ndarray],
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Convert an image to a normalized tensor for model input.

        Performs: validate → resize → to_tensor → normalize (optional).

        Args:
            image: Input image in any supported format.
            device: Target device for the tensor.
            dtype: Target dtype.

        Returns:
            Tensor of shape [1, 3, H, W] normalized for DINOv2.
        """
        image = self.validate_image(image)
        image = self.resize(image)

        # Convert to float tensor [3, H, W] in [0, 1]
        arr = np.array(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]

        if self.normalize:
            mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
            std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
            tensor = (tensor - mean) / std

        # Add batch dimension
        tensor = tensor.unsqueeze(0).to(device=device, dtype=dtype)

        return tensor

    def __call__(
        self,
        image: Union[Image.Image, str, Path, np.ndarray],
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Shortcut for to_tensor.

        Args:
            image: Input image.
            device: Target device.
            dtype: Target dtype.

        Returns:
            Model-ready tensor [1, 3, H, W].
        """
        return self.to_tensor(image, device, dtype)
