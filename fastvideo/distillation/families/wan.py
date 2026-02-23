# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import Any

import torch

from fastvideo.distributed import get_world_group
from fastvideo.distillation.adapters.wan import WanAdapter
from fastvideo.distillation.bundle import ModelBundle, RoleHandle
from fastvideo.distillation.registry import register_family
from fastvideo.distillation.runtime import FamilyArtifacts
from fastvideo.distillation.yaml_config import DistillRunConfig
from fastvideo.models.loader.component_loader import PipelineComponentLoader
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from fastvideo.training.activation_checkpoint import apply_activation_checkpointing
from fastvideo.training.trackers import initialize_trackers, Trackers
from fastvideo.utils import maybe_download_model, verify_model_config_and_directory


def _load_module_from_path(
    *,
    model_path: str,
    module_type: str,
    training_args: Any,
    mark_teacher_critic: bool = False,
) -> torch.nn.Module:
    local_model_path = maybe_download_model(model_path)
    config = verify_model_config_and_directory(local_model_path)

    if module_type not in config:
        raise ValueError(f"Module {module_type!r} not found in config at {local_model_path}")

    module_info = config[module_type]
    if module_info is None:
        raise ValueError(f"Module {module_type!r} has null value in config at {local_model_path}")

    transformers_or_diffusers, _architecture = module_info
    component_path = os.path.join(local_model_path, module_type)

    if mark_teacher_critic:
        training_args._loading_teacher_critic_model = True
    try:
        module = PipelineComponentLoader.load_module(
            module_name=module_type,
            component_model_path=component_path,
            transformers_or_diffusers=transformers_or_diffusers,
            fastvideo_args=training_args,
        )
    finally:
        if mark_teacher_critic and hasattr(training_args, "_loading_teacher_critic_model"):
            del training_args._loading_teacher_critic_model

    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"Loaded {module_type!r} is not a torch.nn.Module: {type(module)}")
    return module


def _apply_trainable(module: torch.nn.Module, *, trainable: bool) -> torch.nn.Module:
    module.requires_grad_(trainable)
    if trainable:
        module.train()
    else:
        module.eval()
    return module


def _build_tracker(training_args: Any, *, config: dict[str, Any] | None) -> Any:
    world_group = get_world_group()
    trackers = list(getattr(training_args, "trackers", []))
    if not trackers and getattr(training_args, "tracker_project_name", ""):
        trackers.append(Trackers.WANDB.value)
    if world_group.rank != 0:
        trackers = []

    tracker_log_dir = getattr(training_args, "output_dir", "") or os.getcwd()
    if trackers:
        tracker_log_dir = os.path.join(tracker_log_dir, "tracker")

    tracker_config = config if trackers else None
    tracker_run_name = getattr(training_args, "wandb_run_name", "") or None
    project = getattr(training_args, "tracker_project_name", "") or "fastvideo"
    return initialize_trackers(
        trackers,
        experiment_name=project,
        config=tracker_config,
        log_dir=tracker_log_dir,
        run_name=tracker_run_name,
    )


@register_family("wan")
def build_wan_family_artifacts(*, cfg: DistillRunConfig) -> FamilyArtifacts:
    training_args = cfg.training_args
    roles_cfg = cfg.roles

    if getattr(training_args, "seed", None) is None:
        raise ValueError("training.seed must be set for distillation")
    if not getattr(training_args, "data_path", ""):
        raise ValueError("training.data_path must be set for distillation")

    # Load shared components (student base path).
    training_args.override_transformer_cls_name = "WanTransformer3DModel"
    vae = _load_module_from_path(
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
                "Wan family only supports wan roles; "
                f"got {role}={role_spec.family!r}"
            )

        mark_teacher_critic = role in ("teacher", "critic")
        transformer = _load_module_from_path(
            model_path=role_spec.path,
            module_type="transformer",
            training_args=training_args,
            mark_teacher_critic=mark_teacher_critic,
        )
        modules: dict[str, torch.nn.Module] = {"transformer": transformer}

        # Optional MoE support: allow teacher transformer_2 if present.
        if role == "teacher":
            try:
                transformer_2 = _load_module_from_path(
                    model_path=role_spec.path,
                    module_type="transformer_2",
                    training_args=training_args,
                    mark_teacher_critic=mark_teacher_critic,
                )
            except Exception:
                transformer_2 = None
            if transformer_2 is not None:
                modules["transformer_2"] = transformer_2

        for name, module in list(modules.items()):
            module = _apply_trainable(module, trainable=bool(role_spec.trainable))
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

    bundle = ModelBundle(roles=role_handles)
    tracker = _build_tracker(training_args, config=cfg.raw)

    validator = None
    if getattr(training_args, "log_validation", False):
        from fastvideo.distillation.validators.wan import WanValidator

        validator = WanValidator(
            training_args=training_args,
            tracker=tracker,
        )

    # NOTE: adapter is the family runtime boundary; it may implement multiple
    # method-specific protocols via duck typing.
    prompt_handle = role_handles.get("student")
    if prompt_handle is None:
        raise ValueError("Wan family requires a 'student' role for prompt encoding")
    adapter = WanAdapter(
        prompt_handle=prompt_handle,
        training_args=training_args,
        noise_scheduler=noise_scheduler,
        vae=vae,
    )

    from fastvideo.dataset import build_parquet_map_style_dataloader
    from fastvideo.dataset.dataloader.schema import pyarrow_schema_t2v

    text_len = training_args.pipeline_config.text_encoder_configs[0].arch_config.text_len  # type: ignore[attr-defined]
    _dataset, dataloader = build_parquet_map_style_dataloader(
        training_args.data_path,
        training_args.train_batch_size,
        num_data_workers=training_args.dataloader_num_workers,
        parquet_schema=pyarrow_schema_t2v,
        cfg_rate=training_args.training_cfg_rate,
        drop_last=True,
        text_padding_length=int(text_len),
        seed=int(training_args.seed or 0),
    )

    return FamilyArtifacts(
        training_args=training_args,
        bundle=bundle,
        adapter=adapter,
        dataloader=dataloader,
        tracker=tracker,
        validator=validator,
        start_step=0,
    )
