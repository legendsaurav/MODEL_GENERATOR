"""End-to-end inference tests (requires GPU + model weights).

These tests are marked to skip if CUDA is not available.
They test the full pipeline from image input to mesh output.
"""

import os
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

skip_no_cuda = pytest.mark.skipif(
    not HAS_CUDA, reason="CUDA not available"
)


@skip_no_cuda
class TestEndToEndInference:
    """Full pipeline inference tests.

    These tests require:
    - CUDA-capable GPU with >= 8GB VRAM
    - Downloaded model weights (tencent/Hunyuan3D-2)
    - Test input images in ./assets/
    """

    def test_pipeline_initialization(self):
        """Test that the pipeline initializes without errors."""
        from MODEL_GENERATOR_V2.generation.pipeline import GeometryPipeline
        from MODEL_GENERATOR_V2.configs.presets import get_preset_config

        get_preset_config("fast")

        # This will attempt to download weights
        try:
            pipeline = GeometryPipeline.from_pretrained(
                model_path="tencent/Hunyuan3D-2",
                preset="fast",
            )
            assert pipeline is not None
        except Exception as e:
            pytest.skip(f"Model weights not available: {e}")

    def test_fast_generation(self, tmp_path):
        """Test mesh generation with FAST preset."""
        from MODEL_GENERATOR_V2.generation.pipeline import GeometryPipeline
        from MODEL_GENERATOR_V2.postprocessing.pipeline import PostProcessingPipeline
        from MODEL_GENERATOR_V2.exporters import get_exporter
        from PIL import Image
        import numpy as np

        # Create a test image (red sphere-like gradient)
        img_arr = np.zeros((256, 256, 3), dtype=np.uint8)
        y, x = np.ogrid[-128:128, -128:128]
        mask = x**2 + y**2 < 100**2
        img_arr[mask] = [200, 100, 50]
        test_image = Image.fromarray(img_arr)
        img_path = tmp_path / "test_input.png"
        test_image.save(img_path)

        try:
            pipeline = GeometryPipeline.from_pretrained(
                preset="fast"
            )
        except Exception as e:
            pytest.skip(f"Model not available: {e}")

        # Generate
        mesh = pipeline(str(img_path), show_progress=False)
        assert mesh is not None
        assert len(mesh.vertices) > 100
        assert len(mesh.faces) > 100

        # Post-process
        from MODEL_GENERATOR_V2.configs.presets import get_preset_config
        config = get_preset_config("fast")
        postprocessor = PostProcessingPipeline(config.postprocessing)
        processed = postprocessor(mesh, verbose=False)
        assert len(processed.vertices) > 0

        # Export
        export_path = tmp_path / "output.glb"
        exporter = get_exporter("glb")
        result_path = exporter.export(processed, str(export_path))
        assert os.path.exists(result_path)
        assert os.path.getsize(result_path) > 0

    def test_model_components_load(self):
        """Test individual model component initialization."""
        from MODEL_GENERATOR_V2.core.dit_model import Hunyuan3DDiT
        from MODEL_GENERATOR_V2.core.vae import ShapeVAE
        from MODEL_GENERATOR_V2.core.scheduler import FlowMatchingScheduler

        # DiT model
        dit = Hunyuan3DDiT(
            in_channels=64,
            out_channels=64,
            num_attention_heads=16,
            attention_head_dim=128,
            num_layers=2,  # Small for testing
            num_single_layers=1,
        )
        assert dit is not None

        # VAE
        vae = ShapeVAE(latent_dim=64, embed_dim=256, depth=2)
        assert vae is not None

        # Scheduler
        scheduler = FlowMatchingScheduler()
        assert scheduler is not None

    def test_dit_forward_pass(self):
        """Test DiT model forward pass with random inputs."""
        from MODEL_GENERATOR_V2.core.dit_model import Hunyuan3DDiT

        device = torch.device("cuda")
        dit = Hunyuan3DDiT(
            in_channels=64,
            out_channels=64,
            num_attention_heads=4,
            attention_head_dim=32,
            num_layers=2,
            num_single_layers=1,
            conditioning_embedding_out_dim=128,
            pooled_projection_dim=128,
        ).to(device)

        batch_size = 1
        latent = torch.randn(batch_size, 16, 64, device=device)
        cond = torch.randn(batch_size, 8, 128, device=device)
        timestep = torch.tensor([0.5], device=device)
        pooled = torch.randn(batch_size, 128, device=device)

        output = dit(latent, cond, timestep, pooled)
        assert output.shape == (batch_size, 16, 64)
