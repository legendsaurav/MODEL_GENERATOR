"""
Diffusion denoising loop runner for MODEL_GENERATOR_V2.

Manages the iterative denoising process — stepping through the
scheduler's timestep schedule with the DiT model to generate
latent tokens from noise.

Dependencies:
    - torch
    - tqdm

Classes:
    DiffusionRunner: Runs the diffusion denoising loop.
"""

from typing import Optional, Union

import torch
from tqdm import tqdm

from ..core.dit_model import Hunyuan3DDiT
from ..core.scheduler import FlowMatchingScheduler
from ..utils.logging import get_logger
from ..utils.timer import synchronize_timer
from ..utils.memory import MemoryOptimizer

logger = get_logger("model_generator_v2.generation.diffusion_runner")


class DiffusionRunner:
    """Manages the diffusion denoising loop for shape generation.

    Takes initial noise and iteratively denoises it through the
    DiT model according to the scheduler's timestep schedule,
    producing clean latent tokens that the VAE can decode.

    Args:
        model: The Hunyuan3DDiT model.
        scheduler: The flow-matching scheduler.

    Example:
        >>> runner = DiffusionRunner(dit_model, scheduler)
        >>> latents = runner.run(
        ...     noise=initial_noise,
        ...     condition_tokens=cond_tokens,
        ...     num_steps=50,
        ... )
    """

    def __init__(
        self,
        model: Hunyuan3DDiT,
        scheduler: FlowMatchingScheduler,
    ) -> None:
        self.model = model
        self.scheduler = scheduler

    @synchronize_timer("Diffusion Loop")
    @torch.no_grad()
    def run(
        self,
        noise: torch.Tensor,
        condition_tokens: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        pooled_projection: Optional[torch.Tensor] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Execute the full diffusion denoising loop.

        Args:
            noise: Initial noise tensor [B, N, C].
            condition_tokens: Image condition embeddings [B, M, D].
            num_inference_steps: Number of denoising steps.
            guidance_scale: CFG scale (not used in flow-matching,
                            included for API compatibility).
            pooled_projection: Optional pooled image embedding [B, D].
            device: Device override.
            dtype: Dtype override.
            show_progress: Whether to show tqdm progress bar.

        Returns:
            Denoised latent tokens [B, N, C].
        """
        if device is None:
            device = noise.device
        if dtype is None:
            dtype = noise.dtype

        # Get timestep schedule
        timesteps, num_steps = self.scheduler.get_timesteps(
            num_inference_steps, device=device
        )

        logger.info(
            f"Starting diffusion: {num_steps} steps, "
            f"latent shape={noise.shape}"
        )

        sample = noise

        # Denoising loop
        progress = tqdm(
            timesteps,
            desc="Generating shape",
            disable=not show_progress,
            leave=True,
        )

        for i, t in enumerate(progress):
            # Prepare timestep tensor
            timestep = t.unsqueeze(0).expand(sample.shape[0])
            timestep = timestep.to(device=device, dtype=dtype)

            # Optional guidance embedding
            guidance = None
            if self.model.guidance_embeds and guidance_scale > 1.0:
                guidance = torch.full(
                    (sample.shape[0],),
                    guidance_scale,
                    device=device,
                    dtype=dtype,
                )

            # DiT forward pass
            model_output = self.model(
                latent_tokens=sample,
                condition_tokens=condition_tokens,
                timestep=timestep,
                pooled_projection=pooled_projection,
                guidance=guidance,
            )

            # Scheduler step
            sample = self.scheduler.step(model_output, t, sample)

            # Update progress bar
            if show_progress:
                progress.set_postfix(
                    {"step": f"{i+1}/{num_steps}"}
                )

        logger.info("Diffusion complete")
        MemoryOptimizer.log_memory_usage("Post-diffusion")

        return sample
