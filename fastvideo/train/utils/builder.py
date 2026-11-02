# SPDX-License-Identifier: Apache-2.0
"""Assembly: build method + dataloader from a ``_target_``-based config."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from fastvideo.train.utils.instantiate import (
    instantiate,
    resolve_target,
)
from fastvideo.train.utils.config import RunConfig

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import (
        TrainingConfig, )
    from fastvideo.train.methods.base import TrainingMethod


def build_from_config(cfg: RunConfig, ) -> tuple[TrainingConfig, TrainingMethod, Any, int]:
    """Build method + dataloader from a v3 run config.

    1. Instantiate each model in ``cfg.models`` via ``_target_``.
    2. Resolve the method class from ``cfg.method["_target_"]``
       and construct it with ``(cfg=cfg, role_models=...)``.
    3. Return ``(training_args, method, dataloader, start_step)``.
    """
    from fastvideo.train.models.base import ModelBase

    memory_policies: dict[str, dict[str, str]] = {}
    memory_config = cfg.training.model
    if memory_config.activation_memory_budget_gib is not None:
        import torch.distributed as dist

        from fastvideo.train.utils.activation_memory import MemoryProfile, plan_activation_memory

        profile = MemoryProfile.load(memory_config.activation_memory_profile)
        unknown_roles = {block.role for block in profile.blocks} - set(cfg.models)
        if unknown_roles:
            raise ValueError(f"Activation profile contains unknown model roles: {sorted(unknown_roles)}")
        host_budget = memory_config.activation_memory_host_budget_gib
        plan = plan_activation_memory(
            profile,
            int(memory_config.activation_memory_budget_gib * 1024**3),
            host_budget_bytes=None if host_budget is None else int(host_budget * 1024**3),
        )
        memory_policies = plan.policies
        if memory_config.activation_memory_plan_output and (not dist.is_initialized() or dist.get_rank() == 0):
            plan.save(memory_config.activation_memory_plan_output)

    # --- 1. Build role model instances ---
    role_models: dict[str, ModelBase] = {}
    for role, model_cfg in cfg.models.items():
        memory_kwargs = {"activation_memory_plan": memory_policies[role]} if role in memory_policies else {}
        model = instantiate(model_cfg, training_config=cfg.training, **memory_kwargs)
        if not isinstance(model, ModelBase):
            raise TypeError(f"models.{role}._target_ must resolve to a "
                            f"ModelBase subclass, got {type(model).__name__}")
        role_models[role] = model

    # --- 2. Build method ---
    method_cfg = dict(cfg.method)
    method_target = str(method_cfg.pop("_target_"))
    method_cls = resolve_target(method_target)

    # The student model provides the dataloader.
    student = role_models.get("student")

    method = method_cls(
        cfg=cfg,
        role_models=role_models,
    )

    # --- 3. Gather dataloader and start_step ---
    dataloader = (getattr(student, "dataloader", None) if student is not None else None)
    start_step = int(getattr(student, "start_step", 0) if student is not None else 0)

    return cfg.training, method, dataloader, start_step
