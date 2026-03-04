# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import Any

import torch

from fastvideo.models.loader.component_loader import PipelineComponentLoader
from fastvideo.utils import maybe_download_model, verify_model_config_and_directory


def load_module_from_path(
    *,
    model_path: str,
    module_type: str,
    training_args: Any,
    disable_custom_init_weights: bool = False,
) -> torch.nn.Module:
    """Load a single pipeline component module from a FastVideo model path.

    This is a thin wrapper over :func:`PipelineComponentLoader.load_module`:
    - resolves/downloads ``model_path`` if needed
    - reads the per-module config entry to determine transformers/diffusers
    - optionally disables custom init weights overrides (legacy flag)
    """

    local_model_path = maybe_download_model(model_path)
    config = verify_model_config_and_directory(local_model_path)

    if module_type not in config:
        raise ValueError(f"Module {module_type!r} not found in config at {local_model_path}")

    module_info = config[module_type]
    if module_info is None:
        raise ValueError(f"Module {module_type!r} has null value in config at {local_model_path}")

    transformers_or_diffusers, _architecture = module_info
    component_path = os.path.join(local_model_path, module_type)

    if disable_custom_init_weights:
        # NOTE: This flag is used by PipelineComponentLoader to skip applying
        # `init_weights_from_safetensors*` overrides when loading auxiliary
        # roles (teacher/critic/etc). The attribute name is legacy.
        training_args._loading_teacher_critic_model = True
    try:
        module = PipelineComponentLoader.load_module(
            module_name=module_type,
            component_model_path=component_path,
            transformers_or_diffusers=transformers_or_diffusers,
            fastvideo_args=training_args,
        )
    finally:
        if disable_custom_init_weights and hasattr(training_args, "_loading_teacher_critic_model"):
            del training_args._loading_teacher_critic_model

    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"Loaded {module_type!r} is not a torch.nn.Module: {type(module)}")
    return module

