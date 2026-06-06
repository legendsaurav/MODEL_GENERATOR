"""Unit tests for image preprocessing."""

import pytest
import sys
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from MODEL_GENERATOR_V2.preprocessing.image_processor import ImageProcessor


@pytest.fixture
def processor():
    return ImageProcessor(target_size=518)


@pytest.fixture
def sample_rgb_image():
    """Create a synthetic 256x256 RGB image."""
    arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def sample_rgba_image():
    """Create a synthetic 256x256 RGBA image."""
    arr = np.random.randint(0, 255, (256, 256, 4), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGBA")


class TestImageProcessor:
    """Tests for ImageProcessor."""

    def test_validate_rgb(self, processor, sample_rgb_image):
        result = processor.validate_image(sample_rgb_image)
        assert result.mode == "RGB"

    def test_validate_rgba_converts_to_rgb(self, processor, sample_rgba_image):
        result = processor.validate_image(sample_rgba_image)
        assert result.mode == "RGB"

    def test_validate_numpy_array(self, processor):
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = processor.validate_image(arr)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_validate_numpy_grayscale(self, processor):
        arr = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = processor.validate_image(arr)
        assert result.mode == "RGB"

    def test_validate_file_path(self, processor, sample_rgb_image, tmp_path):
        path = tmp_path / "test.png"
        sample_rgb_image.save(path)
        result = processor.validate_image(str(path))
        assert isinstance(result, Image.Image)

    def test_validate_nonexistent_file(self, processor):
        with pytest.raises(ValueError, match="not found"):
            processor.validate_image("/nonexistent/image.png")

    def test_validate_too_small(self, processor):
        tiny = Image.new("RGB", (5, 5))
        with pytest.raises(ValueError, match="too small"):
            processor.validate_image(tiny)

    def test_resize(self, processor, sample_rgb_image):
        resized = processor.resize(sample_rgb_image)
        assert resized.size == (518, 518)

    def test_resize_already_correct(self, processor):
        img = Image.new("RGB", (518, 518))
        resized = processor.resize(img)
        assert resized.size == (518, 518)

    def test_to_tensor_shape(self, processor, sample_rgb_image):
        tensor = processor.to_tensor(sample_rgb_image)
        assert tensor.shape == (1, 3, 518, 518)

    def test_to_tensor_normalized(self, processor, sample_rgb_image):
        tensor = processor.to_tensor(sample_rgb_image)
        # After ImageNet normalization, values are not in [0,1]
        # but they should be finite
        assert tensor.isfinite().all()

    def test_to_tensor_no_normalize(self, sample_rgb_image):
        processor = ImageProcessor(target_size=518, normalize=False)
        tensor = processor.to_tensor(sample_rgb_image)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_callable(self, processor, sample_rgb_image):
        """Test __call__ shortcut."""
        tensor = processor(sample_rgb_image)
        assert tensor.shape == (1, 3, 518, 518)
