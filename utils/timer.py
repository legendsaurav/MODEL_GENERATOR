"""
GPU-synchronized timer for performance profiling.

Adapted from Hunyuan3D-2 shapegen/utils.py. Provides accurate timing
for GPU operations by calling torch.cuda.synchronize() before measurements.

Dependencies:
    - torch

Classes:
    synchronize_timer: Dual-use as both context manager and decorator.

Usage as context manager:
    >>> with synchronize_timer('Forward pass') as t:
    ...     output = model(input)

Usage as decorator:
    >>> @synchronize_timer('Export mesh')
    ... def export_to_trimesh(mesh_output):
    ...     pass
"""

import time
from functools import wraps
from typing import Any, Callable

import torch

from .logging import get_logger

logger = get_logger("model_generator_v2.timer")


class synchronize_timer:
    """GPU-synchronized timer supporting context manager and decorator patterns.

    Ensures CUDA operations complete before taking time measurements,
    providing accurate wall-clock timings for GPU workloads.

    Args:
        name: Human-readable label for the timed operation.
        enabled: If False, timing is skipped (zero overhead).

    Attributes:
        name: The operation label.
        elapsed: Elapsed time in seconds after completion.
    """

    def __init__(self, name: str, enabled: bool = True) -> None:
        self.name = name
        self.enabled = enabled
        self.elapsed: float = 0.0
        self._start_time: float = 0.0

    def __enter__(self) -> "synchronize_timer":
        """Start the timer, synchronizing CUDA if available."""
        if self.enabled:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop the timer, synchronizing CUDA if available, and log result."""
        if self.enabled:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.elapsed = time.perf_counter() - self._start_time
            logger.info(f"[Timer] {self.name}: {self.elapsed:.3f}s")

    def __call__(self, func: Callable) -> Callable:
        """Use as a decorator to time a function call.

        Args:
            func: The function to wrap with timing.

        Returns:
            Wrapped function that logs execution time.
        """
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self:
                result = func(*args, **kwargs)
            return result

        return wrapper
