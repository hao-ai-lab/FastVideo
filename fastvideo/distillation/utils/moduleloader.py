# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import Any

import torch

from fastvideo.models.loader.component_loader import (
    PipelineComponentLoader, )
from fastvideo.utils import (
    maybe_download_model,
    verify_model_config_and_directory,
)


def load_module_from_path(
    *,
    model_path: str,
    module_type: str,
    loader_args: Any = None,
    disable_custom_init_weights: bool = False,
    override_transformer_cls_name: str | None = None,
    # Legacy alias kept so callers that still pass
    # ``training_args=`` don't break during migration.
    training_args: Any = None,
) -> torch.nn.Module:
    """Load a single pipeline component module.

    *loader_args* should be a ``DistillLoaderArgs`` or
    ``FastVideoArgs``-like object for the
    ``PipelineComponentLoader``.  When ``None`` a lightweight
    stand-in is used.
    """

    # Support the legacy ``training_args`` kwarg.
    if loader_args is None and training_args is not None:
        loader_args = training_args

    local_model_path = maybe_download_model(model_path)
    config = verify_model_config_and_directory(local_model_path)

    if module_type not in config:
        raise ValueError(f"Module {module_type!r} not found in "
                         f"config at {local_model_path}")

    module_info = config[module_type]
    if module_info is None:
        raise ValueError(f"Module {module_type!r} has null value in "
                         f"config at {local_model_path}")

    transformers_or_diffusers, _architecture = module_info
    component_path = os.path.join(local_model_path, module_type)

    if loader_args is None:
        from types import SimpleNamespace

        loader_args = SimpleNamespace()

    old_override: str | None = None
    if override_transformer_cls_name is not None:
        old_override = getattr(
            loader_args,
            "override_transformer_cls_name",
            None,
        )
        loader_args.override_transformer_cls_name = str(override_transformer_cls_name)

    if disable_custom_init_weights:
        loader_args._loading_teacher_critic_model = True
    try:
        module = PipelineComponentLoader.load_module(
            module_name=module_type,
            component_model_path=component_path,
            transformers_or_diffusers=(transformers_or_diffusers),
            fastvideo_args=loader_args,
        )
    finally:
        if disable_custom_init_weights and hasattr(loader_args, "_loading_teacher_critic_model"):
            del loader_args._loading_teacher_critic_model
        if override_transformer_cls_name is not None:
            if old_override is None:
                if hasattr(
                        loader_args,
                        "override_transformer_cls_name",
                ):
                    loader_args.override_transformer_cls_name = (None)
            else:
                loader_args.override_transformer_cls_name = (old_override)

    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"Loaded {module_type!r} is not a "
                        f"torch.nn.Module: {type(module)}")
    return module
