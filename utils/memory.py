"""
Memory optimization utilities for MODEL_GENERATOR_V2.

Provides VRAM management, garbage collection triggers, model CPU
offloading, and strategic cache clearing to prevent OOM during
large-model inference.

Dependencies:
    - torch
    - gc (stdlib)

Classes:
    MemoryOptimizer: Manages GPU memory lifecycle during inference.
"""

import gc
from contextlib import contextmanager
from typing import Generator, Optional

import torch
import torch.nn as nn

from .logging import get_logger

logger = get_logger("model_generator_v2.memory")


class MemoryOptimizer:
    """GPU memory optimization utilities for large-model inference.

    Provides methods to clear caches, offload models between CPU/GPU,
    and monitor memory usage during the generation pipeline.

    Example:
        >>> optimizer = MemoryOptimizer()
        >>> optimizer.clear_cache()
        >>> with optimizer.inference_mode():
        ...     output = model(input)
        >>> optimizer.offload_to_cpu(model)
    """

    @staticmethod
    def clear_cache() -> None:
        """Force garbage collection and clear GPU memory caches.

        Triggers Python garbage collection, empties the CUDA cache,
        and resets peak memory stats for accurate tracking.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            logger.debug("Cleared CUDA cache and reset peak memory stats")

    @staticmethod
    def get_memory_usage_gb() -> dict[str, float]:
        """Get current GPU memory usage breakdown.

        Returns:
            Dictionary with 'allocated', 'reserved', and 'free' in GB.
            Returns zeros if no GPU is available.
        """
        if not torch.cuda.is_available():
            return {"allocated": 0.0, "reserved": 0.0, "free": 0.0}

        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        free_mem, total_mem = torch.cuda.mem_get_info()
        free = free_mem / (1024 ** 3)

        return {
            "allocated": round(allocated, 3),
            "reserved": round(reserved, 3),
            "free": round(free, 3),
        }

    @staticmethod
    def log_memory_usage(label: str = "") -> None:
        """Log current GPU memory usage with an optional label.

        Args:
            label: Description of the current pipeline stage.
        """
        if not torch.cuda.is_available():
            return
        usage = MemoryOptimizer.get_memory_usage_gb()
        prefix = f"[{label}] " if label else ""
        logger.info(
            f"{prefix}GPU Memory: "
            f"Allocated={usage['allocated']:.2f}GB, "
            f"Reserved={usage['reserved']:.2f}GB, "
            f"Free={usage['free']:.2f}GB"
        )

    @staticmethod
    def offload_to_cpu(model: nn.Module) -> nn.Module:
        """Move a model to CPU and clear GPU cache.

        Used for sequential CPU offloading where only one model
        needs to be on GPU at a time.

        Args:
            model: The PyTorch model to offload.

        Returns:
            The model on CPU.
        """
        model.to("cpu")
        MemoryOptimizer.clear_cache()
        logger.debug(f"Offloaded {model.__class__.__name__} to CPU")
        return model

    @staticmethod
    def move_to_device(
        model: nn.Module,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> nn.Module:
        """Move a model to a specific device with optional dtype conversion.

        Args:
            model: The PyTorch model to move.
            device: Target device.
            dtype: Optional dtype conversion (e.g., torch.float16).

        Returns:
            The model on the target device.
        """
        if dtype is not None:
            model = model.to(device=device, dtype=dtype)
        else:
            model = model.to(device=device)
        logger.debug(
            f"Moved {model.__class__.__name__} to {device}"
            + (f" ({dtype})" if dtype else "")
        )
        return model

    @staticmethod
    @contextmanager
    def inference_mode() -> Generator[None, None, None]:
        """Context manager combining torch.no_grad() with cache management.

        Disables gradient computation for inference and clears
        the GPU cache on exit.

        Yields:
            None

        Example:
            >>> with MemoryOptimizer.inference_mode():
            ...     output = model(input_tensor)
        """
        MemoryOptimizer.clear_cache()
        try:
            with torch.no_grad():
                yield
        finally:
            MemoryOptimizer.clear_cache()

    @staticmethod
    @contextmanager
    def autocast_context(
        device_type: str = "cuda",
        dtype: torch.dtype = torch.float16,
        enabled: bool = True,
    ) -> Generator[None, None, None]:
        """Context manager for automatic mixed-precision inference.

        Args:
            device_type: Device type for autocast ('cuda' or 'cpu').
            dtype: The target dtype for autocast.
            enabled: Whether to enable autocast.

        Yields:
            None
        """
        if enabled and device_type == "cuda" and torch.cuda.is_available():
            with torch.autocast(device_type=device_type, dtype=dtype):
                yield
        else:
            yield
