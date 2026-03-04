# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import torch

from fastvideo.distillation.adapters.wan import WanAdapter
from fastvideo.distillation.roles import RoleHandle, RoleManager
from fastvideo.distillation.dispatch import register_model
from fastvideo.distillation.utils.config import DistillRunConfig
from fastvideo.distillation.models.components import ModelComponents
from fastvideo.distillation.utils.dataloader import build_parquet_t2v_train_dataloader
from fastvideo.distillation.utils.module_state import apply_trainable
from fastvideo.distillation.utils.moduleloader import load_module_from_path
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from fastvideo.training.activation_checkpoint import apply_activation_checkpointing


@register_model("wan")
def build_wan_components(*, cfg: DistillRunConfig) -> ModelComponents:
    training_args = cfg.training_args
    roles_cfg = cfg.roles

    if getattr(training_args, "seed", None) is None:
        raise ValueError("training.seed must be set for distillation")
    if not getattr(training_args, "data_path", ""):
        raise ValueError("training.data_path must be set for distillation")

    # Load shared components (student base path).
    training_args.override_transformer_cls_name = "WanTransformer3DModel"
    vae = load_module_from_path(
        model_path=str(training_args.model_path),
        module_type="vae",
        training_args=training_args,
    )
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        shift=float(training_args.pipeline_config.flow_shift or 0.0)
    )

    role_handles: dict[str, RoleHandle] = {}
    for role, role_spec in roles_cfg.items():
        if role_spec.family != "wan":
            raise ValueError(
                "Wan model plugin only supports roles with family='wan'; "
                f"got {role}={role_spec.family!r}"
            )

        disable_custom_init_weights = bool(
            getattr(role_spec, "disable_custom_init_weights", False)
        )
        transformer = load_module_from_path(
            model_path=role_spec.path,
            module_type="transformer",
            training_args=training_args,
            disable_custom_init_weights=disable_custom_init_weights,
        )
        modules: dict[str, torch.nn.Module] = {"transformer": transformer}

        # Optional MoE support: load transformer_2 if present in the model.
        try:
            transformer_2 = load_module_from_path(
                model_path=role_spec.path,
                module_type="transformer_2",
                training_args=training_args,
                disable_custom_init_weights=disable_custom_init_weights,
            )
        except ValueError:
            transformer_2 = None
        if transformer_2 is not None:
            modules["transformer_2"] = transformer_2

        for name, module in list(modules.items()):
            module = apply_trainable(module, trainable=bool(role_spec.trainable))
            if role_spec.trainable and getattr(
                training_args, "enable_gradient_checkpointing_type", None
            ):
                module = apply_activation_checkpointing(
                    module,
                    checkpointing_type=training_args.enable_gradient_checkpointing_type,
                )
            modules[name] = module

        optimizers: dict[str, torch.optim.Optimizer] = {}
        lr_schedulers: dict[str, Any] = {}

        role_handles[role] = RoleHandle(
            modules=modules,
            optimizers=optimizers,
            lr_schedulers=lr_schedulers,
            trainable=bool(role_spec.trainable),
        )

    bundle = RoleManager(roles=role_handles)

    validator = None
    if getattr(training_args, "log_validation", False):
        from fastvideo.distillation.validators.wan import WanValidator

        validator = WanValidator(
            training_args=training_args,
        )

    # NOTE: adapter is the model runtime boundary; it may implement multiple
    # method-specific protocols via duck typing.
    prompt_handle = role_handles.get("student")
    if prompt_handle is None:
        raise ValueError("Wan model plugin requires a 'student' role for prompt encoding")
    adapter = WanAdapter(
        prompt_handle=prompt_handle,
        training_args=training_args,
        noise_scheduler=noise_scheduler,
        vae=vae,
    )
    from fastvideo.dataset.dataloader.schema import pyarrow_schema_t2v

    dataloader = build_parquet_t2v_train_dataloader(
        training_args,
        parquet_schema=pyarrow_schema_t2v,
    )

    return ModelComponents(
        training_args=training_args,
        bundle=bundle,
        adapter=adapter,
        dataloader=dataloader,
        validator=validator,
        start_step=0,
    )
