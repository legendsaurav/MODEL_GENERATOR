"""
Background removal for input images.

Adapted from Hunyuan3D-2.1 hy3dgen/rembg.py.
Removes image backgrounds and centers the foreground object
to prepare inputs for the shape generation conditioner.

Dependencies:
    - rembg
    - PIL
    - numpy

Classes:
    BackgroundRemover: Removes backgrounds using rembg/U2Net.
"""

from typing import Tuple, Union
from pathlib import Path

import numpy as np
from PIL import Image

from ..utils.logging import get_logger

logger = get_logger("model_generator_v2.preprocessing.background_removal")


class BackgroundRemover:
    """Removes image backgrounds using the rembg library.

    Produces an RGBA image with the background removed and the
    foreground object centered on a white (or custom color) background
    at the target resolution.

    Args:
        background_color: RGB tuple for the output background.
        model_name: rembg model name (default: 'u2net').

    Example:
        >>> remover = BackgroundRemover()
        >>> clean_image = remover('input.png')
        >>> clean_image.save('clean.png')
    """

    def __init__(
        self,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        model_name: str = "u2net",
    ) -> None:
        self.background_color = background_color
        self.model_name = model_name
        self._session = None

    def _get_session(self):
        """Lazy-load the rembg session."""
        if self._session is None:
            try:
                from rembg import new_session
                self._session = new_session(model_name=self.model_name)
                logger.info(f"Loaded rembg session: {self.model_name}")
            except ImportError:
                logger.error(
                    "rembg not installed. Install: pip install rembg[gpu]"
                )
                raise
        return self._session

    def remove_background(
        self,
        image: Union[Image.Image, str, Path],
    ) -> Image.Image:
        """Remove background from an image.

        Args:
            image: Input image (PIL Image, file path, or Path object).

        Returns:
            RGBA PIL Image with background removed.
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        if image.mode != "RGBA":
            image = image.convert("RGBA")

        try:
            from rembg import remove
            session = self._get_session()
            result = remove(image, session=session)
            logger.info("Background removed successfully")
            return result
        except Exception as e:
            logger.warning(
                f"Background removal failed: {e}. "
                f"Returning original image."
            )
            return image

    def center_and_pad(
        self,
        image: Image.Image,
        target_size: int = 518,
        padding_ratio: float = 0.85,
    ) -> Image.Image:
        """Center the foreground object and pad to square.

        Finds the bounding box of non-transparent pixels,
        centers the object, and places it on a background-colored
        square canvas at the target resolution.

        Args:
            image: RGBA image with transparent background.
            target_size: Output image resolution (square).
            padding_ratio: Fraction of canvas occupied by object.

        Returns:
            RGB PIL Image centered and padded.
        """
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        alpha = np.array(image)[:, :, 3]

        # Find bounding box of non-transparent region
        rows = np.any(alpha > 0, axis=1)
        cols = np.any(alpha > 0, axis=0)

        if not rows.any() or not cols.any():
            # No foreground found, return blank
            logger.warning("No foreground detected in image")
            return Image.new("RGB", (target_size, target_size), self.background_color)

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Crop to bounding box
        cropped = image.crop((cmin, rmin, cmax + 1, rmax + 1))

        # Calculate scale to fit within padding_ratio of target
        w, h = cropped.size
        max_dim = max(w, h)
        padded_size = int(target_size * padding_ratio)
        scale = padded_size / max_dim

        new_w = int(w * scale)
        new_h = int(h * scale)
        cropped = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Paste centered on background canvas
        canvas = Image.new("RGBA", (target_size, target_size), (*self.background_color, 255))
        offset_x = (target_size - new_w) // 2
        offset_y = (target_size - new_h) // 2
        canvas.paste(cropped, (offset_x, offset_y), cropped)

        return canvas.convert("RGB")

    def __call__(
        self,
        image: Union[Image.Image, str, Path],
        target_size: int = 518,
        remove_bg: bool = True,
    ) -> Image.Image:
        """Full preprocessing: remove background + center + pad.

        Args:
            image: Input image.
            target_size: Output resolution.
            remove_bg: Whether to remove background (set False if already clean).

        Returns:
            Preprocessed RGB image ready for the conditioner.
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        if remove_bg:
            image = self.remove_background(image)

        result = self.center_and_pad(image, target_size)
        logger.info(f"Preprocessed image: {result.size[0]}×{result.size[1]}")
        return result
