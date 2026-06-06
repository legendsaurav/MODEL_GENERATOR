"""
Flow-matching scheduler wrapper for MODEL_GENERATOR_V2.

Wraps the diffusers FlowMatchEulerDiscreteScheduler with convenience
methods for timestep retrieval and sigma scheduling used by the
Hunyuan3D DiT model.

Dependencies:
    - torch
    - diffusers

Classes:
    FlowMatchingScheduler: Wrapper around diffusers scheduler.

Functions:
    retrieve_timesteps: Extract timesteps from a scheduler.
"""

import inspect
from typing import List, Optional, Tuple, Union

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor

from ..utils.logging import get_logger

logger = get_logger("model_generator_v2.core.scheduler")


def retrieve_timesteps(
    scheduler: FlowMatchEulerDiscreteScheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, int]:
    """Retrieve timesteps from a scheduler after configuration.

    Adapted from Hunyuan3D-2.1 pipelines.py. Handles custom timestep
    and sigma schedules while maintaining compatibility with the
    diffusers scheduler API.

    Args:
        scheduler: The diffusion scheduler instance.
        num_inference_steps: Number of denoising steps.
        device: Device for timestep tensors.
        timesteps: Custom timestep schedule (overrides num_inference_steps).
        sigmas: Custom sigma schedule (overrides num_inference_steps).
        **kwargs: Additional args passed to scheduler.set_timesteps().

    Returns:
        Tuple of (timesteps_tensor, num_steps).

    Raises:
        ValueError: If both timesteps and sigmas are provided.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be provided."
        )

    if timesteps is not None:
        sig = inspect.signature(scheduler.set_timesteps)
        if "timesteps" not in sig.parameters:
            raise ValueError(
                f"Scheduler {scheduler.__class__.__name__} does not support "
                f"custom timestep schedules."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)

    elif sigmas is not None:
        sig = inspect.signature(scheduler.set_timesteps)
        if "sigmas" not in sig.parameters:
            raise ValueError(
                f"Scheduler {scheduler.__class__.__name__} does not support "
                f"custom sigma schedules."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)

    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps

    return timesteps, num_inference_steps


class FlowMatchingScheduler:
    """Wrapper around FlowMatchEulerDiscreteScheduler.

    Provides a clean interface for the generation pipeline, handling
    scheduler initialization, timestep retrieval, and noise sampling.

    Args:
        shift: Shift parameter for the flow-matching schedule.
        num_train_timesteps: Number of training timesteps.

    Attributes:
        scheduler: The underlying diffusers scheduler.
    """

    def __init__(
        self,
        shift: float = 1.0,
        num_train_timesteps: int = 1000,
    ) -> None:
        self.scheduler = FlowMatchEulerDiscreteScheduler(
            shift=shift,
            num_train_timesteps=num_train_timesteps,
        )
        logger.info(
            f"FlowMatchingScheduler initialized: "
            f"shift={shift}, train_steps={num_train_timesteps}"
        )

    def get_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = "cpu",
    ) -> Tuple[torch.Tensor, int]:
        """Get the timestep schedule for inference.

        Args:
            num_inference_steps: Number of denoising steps.
            device: Device for the timestep tensor.

        Returns:
            Tuple of (timesteps, num_steps).
        """
        return retrieve_timesteps(
            self.scheduler,
            num_inference_steps=num_inference_steps,
            device=device,
        )

    def sample_noise(
        self,
        shape: Tuple[int, ...],
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Sample initial noise for the diffusion process.

        Args:
            shape: Shape of the noise tensor.
            device: Target device.
            dtype: Data type.
            seed: Random seed for reproducibility.

        Returns:
            Random noise tensor.
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        noise = randn_tensor(shape, generator=generator, dtype=dtype)
        return noise.to(device)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """Perform one denoising step.

        Args:
            model_output: The DiT's velocity prediction.
            timestep: Current timestep.
            sample: Current noisy sample.

        Returns:
            Denoised sample for the next step.
        """
        result = self.scheduler.step(model_output, timestep, sample)
        return result.prev_sample

    @property
    def timesteps(self) -> torch.Tensor:
        """Access the current timestep schedule."""
        return self.scheduler.timesteps

    @property
    def config(self) -> dict:
        """Get the scheduler configuration."""
        return dict(self.scheduler.config)
