# SPDX-License-Identifier: Apache-2.0
"""
Utility functions for pipeline stages.
"""

import inspect
from typing import Any

import torch
from torch.distributed.tensor import DTensor

from fastvideo.distributed import get_local_torch_device


def module_device(module: Any, fallback: torch.device | None = None) -> torch.device:
    parameters = getattr(module, "parameters", None)
    parameter_iterator = iter(()) if parameters is None else iter(parameters())
    first_parameter = next(parameter_iterator, None)
    if first_parameter is not None:
        distributed_parameter = (first_parameter if isinstance(first_parameter, DTensor) else next(
            (parameter for parameter in parameter_iterator if isinstance(parameter, DTensor)), None))
        parameter = distributed_parameter if distributed_parameter is not None else first_parameter
        return torch.device(parameter.device)
    configured = getattr(module, "_fastvideo_input_device", None)
    if configured is not None:
        return torch.device(configured)
    return torch.device("cpu") if fallback is None else fallback


def move_module_to_local_device(module: Any) -> tuple[Any, torch.device, bool]:
    """Move a CPU-parked component to FastVideo's execution device."""
    target_device = get_local_torch_device()
    parameters = iter(module.parameters())
    first_parameter = next(parameters, None)
    distributed_parameter = (first_parameter if isinstance(first_parameter, DTensor) else next(
        (parameter for parameter in parameters if isinstance(parameter, DTensor)), None))
    if distributed_parameter is not None:
        # FSDP2 with CPU offload streams each layer to the execution device. Moving
        # the wrapped root would instead replicate it and can invalidate its mesh.
        # Scan beyond the first parameter because FP32 islands may be ignored by
        # FSDP while the remainder of the same root is represented by DTensors.
        return module, torch.device(distributed_parameter.device), False
    moved_for_forward = module_device(module) != target_device
    if moved_for_forward:
        module = module.to(target_device)
    return module, module_device(module, fallback=target_device), moved_for_forward


def maybe_offload_module(module: Any, enabled: bool) -> Any:
    """Return a component to CPU after its last forward in the request."""
    return module.to("cpu") if enabled else module


def retrieve_timesteps(
    scheduler: Any,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs: Any,
) -> tuple[Any, int]:
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        if timesteps is None:
            raise ValueError("scheduler.timesteps is None after set_timesteps")
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        if timesteps is None:
            raise ValueError("scheduler.timesteps is None after set_timesteps")
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        if timesteps is None:
            raise ValueError("scheduler.timesteps is None after set_timesteps")
        num_inference_steps = len(timesteps)
    return timesteps, num_inference_steps
