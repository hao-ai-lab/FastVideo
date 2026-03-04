# SPDX-License-Identifier: Apache-2.0

"""Assembly: build method + dataloader from a ``_target_``-based config."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from fastvideo.distillation.utils.instantiate import (
    instantiate,
    resolve_target,
)
from fastvideo.distillation.utils.config import RunConfig

if TYPE_CHECKING:
    from fastvideo.fastvideo_args import TrainingArgs
    from fastvideo.distillation.methods.base import DistillMethod


def build_from_config(
    cfg: RunConfig,
) -> tuple[TrainingArgs, DistillMethod, Any, int]:
    """Build method + dataloader from a v3 run config.

    1. Instantiate each model in ``cfg.models`` via ``_target_``.
    2. Resolve the method class from ``cfg.method["_target_"]``.
    3. Construct the method with ``(cfg=cfg, role_models=...,
       validator=...)``.
    4. Return ``(training_args, method, dataloader, start_step)``.
    """
    from fastvideo.distillation.models.base import ModelBase

    # --- 1. Build role model instances ---
    role_models: dict[str, ModelBase] = {}
    for role, model_cfg in cfg.models.items():
        model = instantiate(model_cfg)
        if not isinstance(model, ModelBase):
            raise TypeError(
                f"models.{role}._target_ must resolve to a "
                f"ModelBase subclass, got {type(model).__name__}"
            )
        role_models[role] = model

    # --- 2. Warm-start from checkpoint if needed ---
    from fastvideo.distillation.utils.checkpoint import (
        maybe_warmstart_role_modules,
    )

    for role, model_cfg in cfg.models.items():
        init_from_checkpoint = model_cfg.get(
            "init_from_checkpoint", None
        )
        if not init_from_checkpoint:
            continue
        checkpoint_role = model_cfg.get(
            "init_from_checkpoint_role", None
        )
        if (
            checkpoint_role is not None
            and not isinstance(checkpoint_role, str)
        ):
            raise ValueError(
                f"models.{role}.init_from_checkpoint_role "
                "must be a string when set, got "
                f"{type(checkpoint_role).__name__}"
            )
        # Warmstart uses the model's transformer directly.
        model = role_models[role]
        maybe_warmstart_role_modules(
            bundle=None,
            role=str(role),
            init_from_checkpoint=str(init_from_checkpoint),
            checkpoint_role=(
                str(checkpoint_role)
                if checkpoint_role
                else None
            ),
            model=model,
        )

    # --- 3. Build method ---
    method_cfg = dict(cfg.method)
    method_target = str(method_cfg.pop("_target_"))
    method_cls = resolve_target(method_target)

    # The student model provides the validator and dataloader.
    student = role_models.get("student")
    validator = (
        getattr(student, "validator", None)
        if student is not None
        else None
    )

    method = method_cls(
        cfg=cfg,
        role_models=role_models,
        validator=validator,
    )

    # --- 4. Gather dataloader and start_step ---
    dataloader = (
        getattr(student, "dataloader", None)
        if student is not None
        else None
    )
    start_step = int(
        getattr(student, "start_step", 0)
        if student is not None
        else 0
    )

    return cfg.training_args, method, dataloader, start_step
