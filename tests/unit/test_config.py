"""Unit tests for configuration module."""

import json
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from MODEL_GENERATOR_V2.configs.base_config import (
    GenerationConfig,
    PostProcessingConfig,
    ExportConfig,
    PipelineConfig,
)
from MODEL_GENERATOR_V2.configs.presets import (
    Preset,
    get_preset_config,
    list_presets,
)


class TestGenerationConfig:
    """Tests for GenerationConfig dataclass."""

    def test_defaults(self):
        config = GenerationConfig()
        assert config.num_inference_steps == 50
        assert config.octree_resolution == 384
        assert config.dtype == "float16"

    def test_custom_values(self):
        config = GenerationConfig(
            num_inference_steps=100,
            octree_resolution=512,
            seed=42,
        )
        assert config.num_inference_steps == 100
        assert config.seed == 42

    def test_validation_passes(self):
        config = GenerationConfig()
        config.validate()  # Should not raise

    def test_validation_invalid_steps(self):
        config = GenerationConfig(num_inference_steps=0)
        with pytest.raises(ValueError, match="num_inference_steps"):
            config.validate()

    def test_validation_invalid_resolution(self):
        config = GenerationConfig(octree_resolution=10)
        with pytest.raises(ValueError, match="octree_resolution"):
            config.validate()

    def test_validation_invalid_dtype(self):
        config = GenerationConfig(dtype="bfloat16")
        with pytest.raises(ValueError, match="dtype"):
            config.validate()


class TestPostProcessingConfig:
    """Tests for PostProcessingConfig dataclass."""

    def test_defaults(self):
        config = PostProcessingConfig()
        assert config.enable_repair is True
        assert config.smoothing_method == "taubin"
        assert config.target_faces == 100000

    def test_validation_invalid_method(self):
        config = PostProcessingConfig(smoothing_method="invalid")
        with pytest.raises(ValueError, match="smoothing_method"):
            config.validate()

    def test_validation_invalid_target_faces(self):
        config = PostProcessingConfig(target_faces=10)
        with pytest.raises(ValueError, match="target_faces"):
            config.validate()


class TestExportConfig:
    """Tests for ExportConfig dataclass."""

    def test_defaults(self):
        config = ExportConfig()
        assert config.formats == ["glb"]

    def test_multiple_formats(self):
        config = ExportConfig(formats=["glb", "obj", "stl", "ply"])
        config.validate()

    def test_invalid_format(self):
        config = ExportConfig(formats=["fbx"])
        with pytest.raises(ValueError, match="Unsupported format"):
            config.validate()


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_defaults(self):
        config = PipelineConfig()
        assert isinstance(config.generation, GenerationConfig)
        assert isinstance(config.postprocessing, PostProcessingConfig)
        assert isinstance(config.export, ExportConfig)

    def test_to_dict(self):
        config = PipelineConfig()
        d = config.to_dict()
        assert "generation" in d
        assert "postprocessing" in d
        assert "export" in d

    def test_to_json(self):
        config = PipelineConfig()
        j = config.to_json()
        parsed = json.loads(j)
        assert parsed["generation"]["num_inference_steps"] == 50

    def test_from_dict(self):
        data = {
            "generation": {"num_inference_steps": 100},
            "postprocessing": {"target_faces": 200000},
            "export": {"formats": ["obj"]},
        }
        config = PipelineConfig.from_dict(data)
        assert config.generation.num_inference_steps == 100
        assert config.postprocessing.target_faces == 200000

    def test_roundtrip(self):
        original = PipelineConfig(
            generation=GenerationConfig(num_inference_steps=75)
        )
        data = original.to_dict()
        restored = PipelineConfig.from_dict(data)
        assert restored.generation.num_inference_steps == 75


class TestPresets:
    """Tests for quality presets."""

    def test_list_presets(self):
        presets = list_presets()
        assert "fast" in presets
        assert "balanced" in presets
        assert "ultra" in presets

    def test_fast_preset(self):
        config = get_preset_config("fast")
        assert config.generation.num_inference_steps == 25
        assert config.generation.octree_resolution == 256
        assert config.postprocessing.target_faces == 50000

    def test_balanced_preset(self):
        config = get_preset_config("balanced")
        assert config.generation.num_inference_steps == 50
        assert config.generation.octree_resolution == 384

    def test_ultra_preset(self):
        config = get_preset_config("ultra")
        assert config.generation.num_inference_steps == 100
        assert config.generation.octree_resolution == 512
        assert config.postprocessing.target_faces == 200000

    def test_preset_enum(self):
        config = get_preset_config(Preset.ULTRA)
        assert config.generation.num_inference_steps == 100

    def test_invalid_preset(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset_config("super_ultra")

    def test_preset_independence(self):
        """Verify presets return independent copies."""
        c1 = get_preset_config("fast")
        c2 = get_preset_config("fast")
        c1.generation.num_inference_steps = 999
        assert c2.generation.num_inference_steps == 25
