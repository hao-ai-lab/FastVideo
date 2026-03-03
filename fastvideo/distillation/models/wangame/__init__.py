# SPDX-License-Identifier: Apache-2.0

"""WanGame model plugin package.

This package registers the WanGame model builders:
- `recipe.family: wangame` (bidirectional)
- `recipe.family: wangame_causal` (causal-capable; supports streaming primitives)
"""

from __future__ import annotations

from fastvideo.distillation.dispatch import register_model
from fastvideo.distillation.utils.config import DistillRunConfig


@register_model("wangame")
def _build_wangame_model(*, cfg: DistillRunConfig):
    from fastvideo.distillation.models.wangame.wangame import WanGameModel

    return WanGameModel(cfg=cfg)


@register_model("wangame_causal")
def _build_wangame_causal_model(*, cfg: DistillRunConfig):
    from fastvideo.distillation.models.wangame.wangame_causal import WanGameCausalModel

    return WanGameCausalModel(cfg=cfg)
