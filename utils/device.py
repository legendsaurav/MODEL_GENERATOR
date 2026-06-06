"""
Device detection and GPU management for MODEL_GENERATOR_V2.

Handles automatic device selection (CUDA, MPS, CPU), VRAM monitoring,
and provides utilities for optimal device placement of models and tensors.

Dependencies:
    - torch

Classes:
    DeviceManager: Singleton manager for device detection and VRAM tracking.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from .logging import get_logger

logger = get_logger("model_generator_v2.device")


@dataclass
class DeviceInfo:
    """Information about a compute device.

    Attributes:
        device: The torch device object.
        name: Human-readable device name.
        type: Device type string ('cuda', 'mps', 'cpu').
        total_memory_gb: Total memory in GB (0 for CPU).
        available_memory_gb: Available memory in GB (0 for CPU).
    """
    device: torch.device
    name: str
    type: str
    total_memory_gb: float
    available_memory_gb: float


class DeviceManager:
    """Manages compute device selection and VRAM monitoring.

    Provides automatic detection of the best available device,
    VRAM usage tracking, and memory threshold checks for
    safe model loading.

    Example:
        >>> dm = DeviceManager()
        >>> device = dm.get_optimal_device()
        >>> if dm.has_sufficient_vram(8.0):
        ...     model.to(device)
    """

    _instance: Optional["DeviceManager"] = None

    def __new__(cls) -> "DeviceManager":
        """Singleton pattern to ensure single device manager instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._device: Optional[torch.device] = None
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    def get_optimal_device(
        self, prefer_device: Optional[str] = None
    ) -> torch.device:
        """Select the optimal compute device.

        Priority order: user preference > CUDA > MPS > CPU.

        Args:
            prefer_device: Optional device string override
                          (e.g., 'cuda:0', 'cpu', 'mps').

        Returns:
            The selected torch.device.
        """
        if prefer_device is not None:
            device = torch.device(prefer_device)
            logger.info(f"Using preferred device: {device}")
            self._device = device
            return device

        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(
                f"Using CUDA device: {torch.cuda.get_device_name(0)}"
            )
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple MPS device")
        else:
            device = torch.device("cpu")
            logger.warning("No GPU detected — using CPU (will be slow)")

        self._device = device
        return device

    def get_device_info(self) -> DeviceInfo:
        """Get detailed information about the current compute device.

        Returns:
            DeviceInfo with memory stats and device identification.
        """
        device = self._device or self.get_optimal_device()

        if device.type == "cuda":
            idx = device.index or 0
            props = torch.cuda.get_device_properties(idx)
            total_gb = props.total_mem / (1024 ** 3)
            free_mem, _ = torch.cuda.mem_get_info(idx)
            avail_gb = free_mem / (1024 ** 3)
            return DeviceInfo(
                device=device,
                name=props.name,
                type="cuda",
                total_memory_gb=round(total_gb, 2),
                available_memory_gb=round(avail_gb, 2),
            )
        else:
            return DeviceInfo(
                device=device,
                name=device.type.upper(),
                type=device.type,
                total_memory_gb=0.0,
                available_memory_gb=0.0,
            )

    def get_available_vram_gb(self) -> float:
        """Get available GPU VRAM in gigabytes.

        Returns:
            Available VRAM in GB, or 0.0 if no GPU is present.
        """
        if not torch.cuda.is_available():
            return 0.0
        device = self._device or self.get_optimal_device()
        if device.type != "cuda":
            return 0.0
        idx = device.index or 0
        free_mem, _ = torch.cuda.mem_get_info(idx)
        return round(free_mem / (1024 ** 3), 2)

    def has_sufficient_vram(self, required_gb: float) -> bool:
        """Check if enough VRAM is available for an operation.

        Args:
            required_gb: Required VRAM in gigabytes.

        Returns:
            True if sufficient VRAM is available or device is CPU.
        """
        available = self.get_available_vram_gb()
        if available == 0.0:
            return True  # CPU mode, no VRAM limit
        sufficient = available >= required_gb
        if not sufficient:
            logger.warning(
                f"Insufficient VRAM: {available:.1f}GB available, "
                f"{required_gb:.1f}GB required"
            )
        return sufficient

    def get_dtype(self, use_fp16: bool = True) -> torch.dtype:
        """Get the optimal dtype for the current device.

        Args:
            use_fp16: Whether to prefer FP16 when supported.

        Returns:
            torch.float16 if FP16 is requested and supported,
            otherwise torch.float32.
        """
        device = self._device or self.get_optimal_device()
        if use_fp16 and device.type in ("cuda", "mps"):
            return torch.float16
        return torch.float32

    @staticmethod
    def reset() -> None:
        """Reset the singleton instance (useful for testing)."""
        DeviceManager._instance = None
