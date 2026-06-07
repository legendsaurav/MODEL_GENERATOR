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

        # Load weights from specific subdirectories
        self._load_weights(models, directory)

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

    def _find_component_weights(self, directory: Path, component_names: list[str]) -> Optional[Path]:
        """Search for model weight files in specific subdirectories."""
        for comp_name in component_names:
            comp_dir = directory / comp_name
            if comp_dir.exists() and comp_dir.is_dir():
                # Prefer safetensors
                for pattern in ["*.safetensors", "*.ckpt", "*.pt", "*.bin"]:
                    files = list(comp_dir.rglob(pattern))
                    if files:
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
        directory: Path,
    ) -> None:
        """Load weights from specific subdirectories into models.

        Args:
            models: Dictionary of model instances.
            directory: Root directory of the model snapshot.
        """
        logger.info(f"Loading weights from {directory}")

        # Components mapping
        components = {
            "model": ["hunyuan3d-dit-v2-1", "hunyuan3d-dit-v2-0", "dit"],
            "vae": ["hunyuan3d-vae-v2-1", "hunyuan3d-vae-v2-0-withencoder", "vae"],
            "conditioner": ["hunyuan3d-dit-v2-1", "hunyuan3d-dit-v2-0", "dit"] # Conditioner weights are often stored in DiT
        }

        for model_key, model_instance in models.items():
            if model_key == "scheduler" or not hasattr(model_instance, "load_state_dict"):
                continue

            subdirs = components.get(model_key, [])
            weights_path = self._find_component_weights(directory, subdirs)

            if weights_path:
                logger.info(f"Loading {model_key} weights from {weights_path.name}")
                if weights_path.suffix == ".safetensors":
                    import safetensors.torch
                    state_dict = safetensors.torch.load_file(str(weights_path), device="cpu")
                    ckpt = self._parse_prefixed_state_dict(state_dict)
                else:
                    ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=True)
                    if not isinstance(ckpt, dict):
                        ckpt = {"model": ckpt}

                # If the key exists in prefixed ckpt, use it, otherwise use the whole state dict
                # Some checkpoints have flat keys, some have 'model.x' prefixes.
                if model_key in ckpt:
                    target_dict = ckpt[model_key]
                else:
                    target_dict = state_dict if weights_path.suffix == ".safetensors" else ckpt

                try:
                    if model_key == "conditioner" and hasattr(model_instance, "load_state_dict_partial"):
                        model_instance.load_state_dict_partial(target_dict, strict=False)
                    else:
                        missing, unexpected = model_instance.load_state_dict(target_dict, strict=False)
                        if missing:
                            logger.warning(f"Partial weight load for {model_key}, missing {len(missing)} keys. Example: {missing[:3]}")
                    logger.info(f"Loaded weights for: {model_key}")
                except Exception as e:
                    logger.warning(f"Failed to load weights for {model_key}: {e}")
            else:
                logger.warning(f"No weights found for component: {model_key}")

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
