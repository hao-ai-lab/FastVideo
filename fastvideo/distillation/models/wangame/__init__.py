# SPDX-License-Identifier: Apache-2.0

"""WanGame model plugin package.

This package registers the WanGame model builders:
- `recipe.family: wangame` (bidirectional by default; can opt into causal per-role)
- `recipe.family: wangame_causal` (causal by default; can opt into bidirectional per-role)

We support both bidirectional and causal transformers:
- If any role declares `roles.<role>.variant: causal`, we build the causal-capable
  model plugin (which implements `CausalModelBase`).
- Otherwise we build the bidirectional-only model plugin.
"""

from __future__ import annotations

from fastvideo.distillation.dispatch import register_model
from fastvideo.distillation.utils.config import DistillRunConfig


@register_model("wangame")
def _build_wangame_model(*, cfg: DistillRunConfig):
    wants_causal = False
    for role, role_spec in cfg.roles.items():
        variant = (role_spec.extra or {}).get("variant", None)
        if variant is None:
            continue
        if str(variant).strip().lower() == "causal":
            wants_causal = True
            break

    if wants_causal:
        from fastvideo.distillation.models.wangame.wangame_causal import (
            WanGameCausalModel,
        )

        return WanGameCausalModel(cfg=cfg)

    from fastvideo.distillation.models.wangame.wangame import WanGameModel

    return WanGameModel(cfg=cfg)


@register_model("wangame_causal")
def _build_wangame_causal_model(*, cfg: DistillRunConfig):
    from fastvideo.distillation.models.wangame.wangame_causal import WanGameCausalModel

    return WanGameCausalModel(cfg=cfg)
