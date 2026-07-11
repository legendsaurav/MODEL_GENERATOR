"""
Hidden State Bridge for Hunyuan3D Flow DiT.

Exposes intermediate transformer hidden representations through a stable,
versioned API. This bridge is the **sole** sanctioned channel through which
geometric understanding flows from MODEL_GENERATOR_V2 to the geometry-engine.

Architecture invariant
    All geometric understanding consumed by the geometry-engine MUST originate
    from the DiT latent representations captured here - **never** from decoded
    triangle meshes.

The bridge works by registering ``torch.nn.Module.register_forward_hook``
callbacks on every ``DualStreamBlock`` and ``SingleStreamBlock`` inside a
``Hunyuan3DDiT`` instance.  Captured activations are detached and moved to
CPU immediately to avoid holding onto GPU graph references.

Usage::

    from MODEL_GENERATOR_V2.core.hidden_state_bridge import HiddenStateBridge

    bridge = HiddenStateBridge()
    bridge.register_hooks(dit_model)
    bridge.set_capture_timesteps([0.0, 0.25, 0.5, 0.75, 1.0])

    # ? run the diffusion loop ?

    states = bridge.get_captured_states()
    fused  = bridge.get_fused_representation(timestep=0.5, layers=[18, 19, 20])
    bridge.clear()

Dependencies:
    - torch
    - logging (stdlib, via utils.logging)

Classes:
    HiddenStateBridge: Core capture API.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

from ..utils.logging import get_logger

logger: logging.Logger = get_logger("model_generator_v2.core.hidden_state_bridge")


# -------------------------------------------------------------------------- #
#  Type aliases                                                              #
# -------------------------------------------------------------------------- #

_HookHandle = torch.utils.hooks.RemovableHandle
_LayerStates = Dict[str, torch.Tensor]
"""Mapping from layer name -> detached CPU tensor for one timestep."""
_TimestepStates = Dict[str, _LayerStates]
"""Mapping from timestep key -> layer states."""


# -------------------------------------------------------------------------- #
#  HiddenStateBridge                                                         #
# -------------------------------------------------------------------------- #


class HiddenStateBridge:
    """Captures intermediate hidden states from a Hunyuan3D DiT model.

    The bridge attaches ``register_forward_hook`` callbacks to every
    transformer block (both *dual-stream* and *single-stream*).  During
    the diffusion loop the caller sets the current timestep via
    :pymethod:`set_current_timestep`; if that timestep is in the
    configured capture set the hook stores the block's output tensor,
    **detached and on CPU**, keyed by ``(timestep, layer_name)``.

    Parameters
    ----------
    max_cached_timesteps : int, optional
        Safety limit - if more than this many timesteps are captured
        without a :pymethod:`clear`, a warning is logged.  Defaults to
        ``64``.

    Attributes
    ----------
    capture_timesteps : set[float]
        Timesteps at which hooks will store activations.
    current_timestep : float | None
        Set by the caller at each diffusion step.
    states : dict[str, dict[str, torch.Tensor]]
        ``{timestep_key: {layer_name: tensor}}``.

    Examples
    --------
    >>> bridge = HiddenStateBridge()
    >>> bridge.register_hooks(model)
    >>> bridge.set_capture_timesteps([0.0, 0.5, 1.0])
    """

    # Tolerance for floating-point timestep comparison
    _TIMESTEP_TOLERANCE: float = 1e-5

    def __init__(self, max_cached_timesteps: int = 64) -> None:
        self._capture_timesteps: set[float] = set()
        self._current_timestep: Optional[float] = None
        self._states: _TimestepStates = {}
        self._hook_handles: List[_HookHandle] = []
        self._registered: bool = False
        self._max_cached_timesteps: int = max_cached_timesteps

        logger.info(
            "HiddenStateBridge created (max_cached_timesteps=%d)",
            max_cached_timesteps,
        )

    # -- Public API ------------------------------------------------------- #

    def register_hooks(self, model: nn.Module) -> None:
        """Attach forward hooks to all transformer blocks in *model*.

        Hooks are registered on modules whose attribute names match the
        Hunyuan3DDiT convention:

        * ``dual_stream_blocks``  -> ``DualStreamBlock`` instances
        * ``single_stream_blocks`` -> ``SingleStreamBlock`` instances

        Calling this method a second time without first calling
        :pymethod:`remove_hooks` is a no-op with a warning.

        Parameters
        ----------
        model : torch.nn.Module
            The ``Hunyuan3DDiT`` (or compatible) model instance.

        Raises
        ------
        ValueError
            If *model* exposes neither ``dual_stream_blocks`` nor
            ``single_stream_blocks``.
        """
        if self._registered:
            logger.warning(
                "register_hooks called but hooks are already registered - "
                "call remove_hooks() first to re-register"
            )
            return

        # Try v2.1 names first, then fall back to v2.0 names
        dual_blocks: Optional[nn.ModuleList] = (
            getattr(model, "double_blocks", None)
            or getattr(model, "dual_stream_blocks", None)
        )
        single_blocks: Optional[nn.ModuleList] = (
            getattr(model, "single_blocks", None)
            or getattr(model, "single_stream_blocks", None)
        )

        if dual_blocks is None and single_blocks is None:
            raise ValueError(
                "Model has none of 'double_blocks', 'dual_stream_blocks', "
                "'single_blocks', or 'single_stream_blocks'. Cannot register hooks."
            )

        hook_count: int = 0

        if dual_blocks is not None:
            for idx, block in enumerate(dual_blocks):
                layer_name = f"double_block.{idx}"
                handle = block.register_forward_hook(
                    self._make_hook(layer_name, stream_type="dual")
                )
                self._hook_handles.append(handle)
                hook_count += 1

        if single_blocks is not None:
            for idx, block in enumerate(single_blocks):
                layer_name = f"single_block.{idx}"
                handle = block.register_forward_hook(
                    self._make_hook(layer_name, stream_type="single")
                )
                self._hook_handles.append(handle)
                hook_count += 1

        self._registered = True
        logger.info(
            "Registered %d forward hooks (dual=%d, single=%d)",
            hook_count,
            len(dual_blocks) if dual_blocks is not None else 0,
            len(single_blocks) if single_blocks is not None else 0,
        )

    def remove_hooks(self) -> None:
        """Remove all previously registered hooks.

        Safe to call even when no hooks are registered.
        """
        for handle in self._hook_handles:
            handle.remove()
        removed = len(self._hook_handles)
        self._hook_handles.clear()
        self._registered = False
        logger.info("Removed %d forward hooks", removed)

    def set_capture_timesteps(self, timesteps: List[float]) -> None:
        """Configure which diffusion timesteps trigger state capture.

        Parameters
        ----------
        timesteps : list[float]
            Timestep values (typically in ``[0.0, 1.0]`` for flow-matching).
        """
        self._capture_timesteps = set(timesteps)
        logger.info(
            "Capture timesteps set: %s",
            sorted(self._capture_timesteps),
        )

    def set_current_timestep(self, timestep: float) -> None:
        """Inform the bridge of the current diffusion timestep.

        Must be called at each step of the diffusion loop *before* the
        model forward pass so that hooks know whether to capture.

        Parameters
        ----------
        timestep : float
            Current diffusion timestep value.
        """
        self._current_timestep = timestep

    def get_captured_states(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Return all captured hidden states.

        Returns
        -------
        dict[str, dict[str, torch.Tensor]]
            Outer key is a timestep string (``"t=0.500"``), inner key is
            the layer name (``"double_block.3"``), value is the detached
            CPU tensor.
        """
        return dict(self._states)

    def clear(self) -> None:
        """Discard all captured hidden states and free memory."""
        n_timesteps = len(self._states)
        n_tensors = sum(len(v) for v in self._states.values())
        self._states.clear()
        self._current_timestep = None
        logger.info(
            "Cleared captured states: %d timesteps, %d tensors",
            n_timesteps,
            n_tensors,
        )

    def save_states(self, path: str) -> None:
        """Persist all captured hidden states to a ``.pt`` file.

        Parameters
        ----------
        path : str
            Destination filepath (recommended extension: ``.pt``).
        """
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self._states, path)
        n_ts = len(self._states)
        n_t = sum(len(v) for v in self._states.values())
        logger.info("Saved %d timesteps (%d tensors) -> %s", n_ts, n_t, path)

    @classmethod
    def load_states(cls, path: str) -> Dict[str, Dict[str, torch.Tensor]]:
        """Load previously saved hidden states from a ``.pt`` file.

        Parameters
        ----------
        path : str
            Path to the saved ``.pt`` file.

        Returns
        -------
        dict[str, dict[str, torch.Tensor]]
            The timestep -> layer -> tensor mapping.
        """
        states = torch.load(path, map_location="cpu", weights_only=False)
        n_ts = len(states)
        n_t = sum(len(v) for v in states.values())
        logger.info("Loaded %d timesteps (%d tensors) from %s", n_ts, n_t, path)
        return states

    def get_fused_representation(
        self,
        timestep: float,
        layers: Optional[List[int]] = None,
        *,
        fusion: str = "mean",
    ) -> torch.Tensor:
        """Fuse multiple layer outputs into a single representation.

        Parameters
        ----------
        timestep : float
            The diffusion timestep whose states to fuse.
        layers : list[int] | None
            Layer indices to include.  If ``None``, all captured layers
            for that timestep are used.
        fusion : str
            Fusion strategy.  One of ``"mean"`` (default), ``"concat"``,
            ``"sum"``.

        Returns
        -------
        torch.Tensor
            The fused representation tensor (on CPU).

        Raises
        ------
        KeyError
            If the requested timestep has no captured states.
        ValueError
            If *fusion* is not a recognised strategy, or no layers match.
        """
        ts_key = self._timestep_key(timestep)
        if ts_key not in self._states:
            available = list(self._states.keys())
            raise KeyError(
                f"No captured states for timestep key '{ts_key}'. "
                f"Available keys: {available}"
            )

        layer_states = self._states[ts_key]

        if layers is not None:
            # Build expected names for the requested indices and collect
            # matching tensors from both dual and single blocks.
            requested_names: set[str] = set()
            for idx in layers:
                requested_names.add(f"double_block.{idx}")
                requested_names.add(f"single_block.{idx}")
            tensors = [
                t for name, t in layer_states.items()
                if name in requested_names
            ]
            if not tensors:
                raise ValueError(
                    f"No captured layers match indices {layers} at {ts_key}. "
                    f"Available: {list(layer_states.keys())}"
                )
        else:
            tensors = list(layer_states.values())

        if not tensors:
            raise ValueError(f"No tensors available at {ts_key}")

        logger.debug(
            "Fusing %d layer tensors at %s with strategy '%s'",
            len(tensors),
            ts_key,
            fusion,
        )

        if fusion == "mean":
            stacked = torch.stack(tensors, dim=0)
            return stacked.mean(dim=0)
        elif fusion == "sum":
            stacked = torch.stack(tensors, dim=0)
            return stacked.sum(dim=0)
        elif fusion == "concat":
            return torch.cat(tensors, dim=-1)
        else:
            raise ValueError(
                f"Unknown fusion strategy '{fusion}'. "
                f"Choose from: 'mean', 'sum', 'concat'."
            )

    # -- Internals -------------------------------------------------------- #

    def _should_capture(self) -> bool:
        """Check if the current timestep is in the capture set."""
        if self._current_timestep is None:
            return False
        return any(
            abs(self._current_timestep - t) < self._TIMESTEP_TOLERANCE
            for t in self._capture_timesteps
        )

    @staticmethod
    def _timestep_key(timestep: float) -> str:
        """Produce a stable dictionary key for a floating-point timestep."""
        return f"t={timestep:.3f}"

    def _make_hook(
        self,
        layer_name: str,
        stream_type: str,
    ) -> Callable[..., None]:
        """Return a forward-hook closure for a specific block.

        Parameters
        ----------
        layer_name : str
            Identifier used as the dict key (e.g. ``"double_block.3"``).
        stream_type : str
            ``"dual"`` or ``"single"`` - controls how the hook unpacks
            the module output (tuples vs. plain tensors).
        """

        def _hook(
            module: nn.Module,
            input: Any,
            output: Any,
        ) -> None:
            if not self._should_capture():
                return

            assert self._current_timestep is not None
            ts_key = self._timestep_key(self._current_timestep)

            # Dual-stream blocks return (latent, cond) tuples - we
            # capture only the latent stream, which carries the 3-D
            # shape information that geometry-engine needs.
            if stream_type == "dual":
                if isinstance(output, (tuple, list)):
                    tensor = output[0]
                else:
                    tensor = output
            else:
                tensor = output if isinstance(output, torch.Tensor) else output[0]

            # Detach and move to CPU to avoid pinning GPU memory.
            stored = tensor.detach().cpu()

            if ts_key not in self._states:
                self._states[ts_key] = {}
                if len(self._states) > self._max_cached_timesteps:
                    logger.warning(
                        "Captured states exceed safety limit of %d timesteps - "
                        "consider calling clear()",
                        self._max_cached_timesteps,
                    )

            self._states[ts_key][layer_name] = stored

            logger.debug(
                "Captured %s at %s - shape %s",
                layer_name,
                ts_key,
                list(stored.shape),
            )

        return _hook

    # -- Dunder ----------------------------------------------------------- #

    def __repr__(self) -> str:
        n_hooks = len(self._hook_handles)
        n_ts = len(self._states)
        n_tensors = sum(len(v) for v in self._states.values())
        return (
            f"HiddenStateBridge(hooks={n_hooks}, "
            f"captured_timesteps={n_ts}, tensors={n_tensors})"
        )

    def __del__(self) -> None:
        """Best-effort cleanup of hooks on garbage collection."""
        self.remove_hooks()
