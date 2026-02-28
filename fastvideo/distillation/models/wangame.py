# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from fastvideo.distillation.adapters.wangame import WanGameAdapter
from fastvideo.distillation.dispatch import register_model
from fastvideo.distillation.models.components import ModelComponents
from fastvideo.distillation.roles import RoleHandle, RoleManager
from fastvideo.distillation.utils.config import DistillRunConfig
from fastvideo.distillation.utils.dataloader import build_parquet_wangame_train_dataloader
from fastvideo.distillation.utils.module_state import apply_trainable
from fastvideo.distillation.utils.moduleloader import load_module_from_path
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from fastvideo.training.activation_checkpoint import apply_activation_checkpointing


@register_model("wangame")
def build_wangame_components(*, cfg: DistillRunConfig) -> ModelComponents:
    training_args = cfg.training_args
    roles_cfg = cfg.roles

    if getattr(training_args, "seed", None) is None:
        raise ValueError("training.seed must be set for distillation")
    if not getattr(training_args, "data_path", ""):
        raise ValueError("training.data_path must be set for distillation")

    # Load shared components (student base path).
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
        if role_spec.family != "wangame":
            raise ValueError(
                "Wangame model plugin only supports roles with family='wangame'; "
                f"got {role}={role_spec.family!r}"
            )

        variant_raw = (role_spec.extra or {}).get("variant", None)
        if variant_raw is None or variant_raw == "":
            transformer_cls_name = "WanGameActionTransformer3DModel"
        else:
            variant = str(variant_raw).strip().lower()
            if variant in {"bidirectional", "bidi"}:
                transformer_cls_name = "WanGameActionTransformer3DModel"
            elif variant == "causal":
                transformer_cls_name = "CausalWanGameTransformer3DModel"
            else:
                raise ValueError(
                    f"Unknown roles.{role}.variant for wangame: "
                    f"{variant_raw!r}. Expected 'causal' or 'bidirectional'."
                )

        disable_custom_init_weights = bool(
            getattr(role_spec, "disable_custom_init_weights", False)
        )
        transformer = load_module_from_path(
            model_path=role_spec.path,
            module_type="transformer",
            training_args=training_args,
            disable_custom_init_weights=disable_custom_init_weights,
            override_transformer_cls_name=transformer_cls_name,
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

        role_handles[role] = RoleHandle(
            modules=modules,
            optimizers={},
            lr_schedulers={},
            trainable=bool(role_spec.trainable),
        )

    bundle = RoleManager(roles=role_handles)

    validator = None
    validation_cfg = getattr(cfg, "validation", {}) or {}
    validation_enabled = bool(validation_cfg.get("enabled", bool(validation_cfg)))
    if validation_enabled:
        from fastvideo.distillation.validators.wangame import WanGameValidator

        validator = WanGameValidator(training_args=training_args)

    adapter = WanGameAdapter(
        training_args=training_args,
        noise_scheduler=noise_scheduler,
        vae=vae,
    )

    from fastvideo.dataset.dataloader.schema import pyarrow_schema_wangame

    dataloader = build_parquet_wangame_train_dataloader(
        training_args,
        parquet_schema=pyarrow_schema_wangame,
    )

    return ModelComponents(
        training_args=training_args,
        bundle=bundle,
        adapter=adapter,
        dataloader=dataloader,
        validator=validator,
        start_step=0,
    )
