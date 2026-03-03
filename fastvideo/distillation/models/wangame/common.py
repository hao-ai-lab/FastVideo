# SPDX-License-Identifier: Apache-2.0

"""WanGame shared helpers.

This module hosts "family-level" helpers used by both bidirectional and causal
WanGame model plugins (e.g. role-handle assembly).

The goal is to avoid duplicated logic across:
- `fastvideo/distillation/models/wangame/wangame.py`
- `fastvideo/distillation/models/wangame/wangame_causal.py`

Keep this file free of imports of the variant-specific model classes to prevent
circular dependencies.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from fastvideo.training.activation_checkpoint import apply_activation_checkpointing

from fastvideo.distillation.roles import RoleHandle
from fastvideo.distillation.utils.module_state import apply_trainable
from fastvideo.distillation.utils.moduleloader import load_module_from_path


def _build_wangame_role_handles(
    *,
    roles_cfg: dict[str, Any],
    training_args: Any,
    transformer_cls_name_for_role: Callable[[str, Any], str],
) -> dict[str, RoleHandle]:
    role_handles: dict[str, RoleHandle] = {}
    for role, role_spec in roles_cfg.items():
        if role_spec.family not in {"wangame", "wangame_causal"}:
            raise ValueError(
                "Wangame model plugin only supports roles with family in "
                "{'wangame', 'wangame_causal'}; "
                f"got {role}={role_spec.family!r}"
            )

        transformer_cls_name = transformer_cls_name_for_role(str(role), role_spec)
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
            if bool(role_spec.trainable) and bool(
                getattr(training_args, "enable_gradient_checkpointing_type", None)
            ):
                module = apply_activation_checkpointing(
                    module,
                    checkpointing_type=training_args.enable_gradient_checkpointing_type,
                )
            modules[name] = module

        role_handles[str(role)] = RoleHandle(
            modules=modules,
            optimizers={},
            lr_schedulers={},
            trainable=bool(role_spec.trainable),
        )

    return role_handles
