"""
Smart model loading and caching for MODEL_GENERATOR_V2.

Handles downloading pretrained weights from HuggingFace, loading
local checkpoints (.ckpt, .safetensors), and managing model
instances with CPU offloading support.

Dependencies:
    - torch
    - safetensors
    - yaml

Classes:
    ModelLoader: Loads and caches all pipeline submodels.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import yaml

from ..core.conditioner import ImageConditioner
from ..core.dit_model import Hunyuan3DDiT
from ..core.scheduler import FlowMatchingScheduler
from ..core.vae import ShapeVAE
from ..utils.logging import get_logger
from ..utils.timer import synchronize_timer
from ..utils.device import DeviceManager
from ..utils.memory import MemoryOptimizer

logger = get_logger("model_generator_v2.generation.model_loader")


class ModelLoader:
    """Loads and manages all pipeline submodels.

    Supports loading from:
    - HuggingFace Hub (auto-download)
    - Local directories with config + weights
    - Single checkpoint files (.ckpt or .safetensors)

    Implements sequential loading to minimize peak VRAM usage.

    Args:
        device: Target compute device.
        dtype: Model parameter dtype.
        enable_cpu_offload: Whether to keep models on CPU until needed.

    Example:
        >>> loader = ModelLoader(device='cuda', dtype=torch.float16)
        >>> models = loader.load_from_pretrained('tencent/Hunyuan3D-2')
    """

    def __init__(
        self,
        device: Union[str, torch.device] = "auto",
        dtype: torch.dtype = torch.float16,
        enable_cpu_offload: bool = False,
    ) -> None:
        self.dm = DeviceManager()
        if isinstance(device, str) and device == "auto":
            self.device = self.dm.get_optimal_device()
        else:
            self.device = torch.device(device)
        self.dtype = dtype
        self.enable_cpu_offload = enable_cpu_offload

    @synchronize_timer("Model Loading")
    def load_from_pretrained(
        self,
        model_path: str,
        config_path: Optional[str] = None,
    ) -> Dict[str, object]:
        """Load all submodels from a HuggingFace repo or local path.

        Args:
            model_path: HuggingFace model ID or local directory path.
            config_path: Optional explicit path to model config YAML.

        Returns:
            Dictionary with keys: 'model', 'vae', 'conditioner',
            'scheduler', with loaded model instances.
        """
        local_path = Path(model_path)

        if local_path.is_dir():
            return self._load_from_directory(local_path, config_path)
        else:
            return self._load_from_hub(model_path, config_path)

    def _load_from_hub(
        self,
        model_id: str,
        config_path: Optional[str] = None,
    ) -> Dict[str, object]:
        """Load from HuggingFace Hub with auto-download.

        Args:
            model_id: HuggingFace model identifier.
            config_path: Optional config override.

        Returns:
            Dictionary of loaded models.
        """
        try:
            from huggingface_hub import snapshot_download

            logger.info(f"Downloading model from HuggingFace: {model_id}")
            local_dir = snapshot_download(
                repo_id=model_id,
                allow_patterns=["*.safetensors", "*.yaml", "*.json", "*.ckpt"],
            )
            return self._load_from_directory(Path(local_dir), config_path)

        except ImportError:
            logger.error(
                "huggingface_hub required for downloading models. "
                "Install: pip install huggingface_hub"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise

    def _load_from_directory(
        self,
        directory: Path,
        config_path: Optional[str] = None,
    ) -> Dict[str, object]:
        """Load models from a local directory.

        Args:
            directory: Path to directory with model files.
            config_path: Optional config file path.

        Returns:
            Dictionary of loaded models.
        """
        # Find config file
        if config_path:
            cfg_path = Path(config_path)
        else:
            cfg_path = self._find_config(directory)

        if cfg_path and cfg_path.exists():
            with open(cfg_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {cfg_path}")
        else:
            config = {}
            logger.warning("No config found, using defaults")

        # Initialize models with config params
        models = self._initialize_models(config)

        # Load weights
        weights_file = self._find_weights(directory)
        if weights_file:
            self._load_weights(models, weights_file)

        return models

    def _find_config(self, directory: Path) -> Optional[Path]:
        """Search for a model config file in the directory."""
        for name in ["config.yaml", "model_config.yaml", "config.json"]:
            path = directory / name
            if path.exists():
                return path
        # Check subdirectories
        for subdir in directory.iterdir():
            if subdir.is_dir():
                for name in ["config.yaml", "model_config.yaml"]:
                    path = subdir / name
                    if path.exists():
                        return path
        return None

    def _find_weights(self, directory: Path) -> Optional[Path]:
        """Search for model weight files."""
        # Prefer safetensors
        for pattern in ["*.safetensors", "*.ckpt", "*.pt", "*.bin"]:
            files = list(directory.rglob(pattern))
            if files:
                # Sort by size (largest first, likely the main model)
                files.sort(key=lambda f: f.stat().st_size, reverse=True)
                return files[0]
        return None

    def _initialize_models(self, config: dict) -> Dict[str, object]:
        """Initialize model instances from config.

        Args:
            config: Model configuration dictionary.

        Returns:
            Dictionary with unloaded model instances.
        """
        # DiT model
        dit_params = config.get("model", {}).get("params", {})
        model = Hunyuan3DDiT(**dit_params)

        # VAE
        vae_params = config.get("vae", {}).get("params", {})
        vae = ShapeVAE(**vae_params)

        # Conditioner
        conditioner = ImageConditioner()

        # Scheduler
        sched_params = config.get("scheduler", {}).get("params", {})
        scheduler = FlowMatchingScheduler(**sched_params)

        return {
            "model": model,
            "vae": vae,
            "conditioner": conditioner,
            "scheduler": scheduler,
        }

    def _load_weights(
        self,
        models: Dict[str, object],
        weights_path: Path,
    ) -> None:
        """Load weights from a checkpoint file into models.

        Args:
            models: Dictionary of model instances.
            weights_path: Path to weights file.
        """
        logger.info(f"Loading weights from {weights_path}")

        if weights_path.suffix == ".safetensors":
            import safetensors.torch
            state_dict = safetensors.torch.load_file(
                str(weights_path), device="cpu"
            )
            # Parse prefixed keys
            ckpt = self._parse_prefixed_state_dict(state_dict)
        else:
            ckpt = torch.load(
                str(weights_path), map_location="cpu", weights_only=True
            )

        # Load into each model
        for name, model in models.items():
            if name == "scheduler":
                continue
            if name in ckpt and hasattr(model, "load_state_dict"):
                try:
                    model.load_state_dict(ckpt[name], strict=False)
                    logger.info(f"Loaded weights for: {name}")
                except Exception as e:
                    logger.warning(f"Partial weight load for {name}: {e}")

        # Move to device
        target = "cpu" if self.enable_cpu_offload else self.device
        for name, model in models.items():
            if name == "scheduler":
                continue
            if hasattr(model, "to"):
                model.to(device=target, dtype=self.dtype)
                model.eval()

        MemoryOptimizer.log_memory_usage("After weight loading")

    @staticmethod
    def _parse_prefixed_state_dict(
        flat_dict: dict,
    ) -> Dict[str, dict]:
        """Parse a flat safetensors dict into per-model dicts.

        Keys like 'model.layer.weight' become {'model': {'layer.weight': v}}.
        """
        result: Dict[str, dict] = {}
        for key, value in flat_dict.items():
            parts = key.split(".", 1)
            if len(parts) == 2:
                prefix, subkey = parts
                if prefix not in result:
                    result[prefix] = {}
                result[prefix][subkey] = value
        return result
