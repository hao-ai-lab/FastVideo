# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from fastvideo.distillation.registry import get_model, get_method
from fastvideo.distillation.utils.config import DistillRuntime
from fastvideo.distillation.utils.config import DistillRunConfig


def build_runtime_from_config(cfg: DistillRunConfig) -> DistillRuntime:
    """Build a distillation runtime from a YAML config.

    This is the Phase 2.9 "elegant dispatch" entry for assembling:
    - model components (bundle/adapter/dataloader/tracker/validator)
    - method implementation (algorithm) on top of those components
    """

    model_builder = get_model(str(cfg.recipe.family))
    components = model_builder(cfg=cfg)

    method_builder = get_method(str(cfg.recipe.method))
    method = method_builder(
        cfg=cfg,
        bundle=components.bundle,
        adapter=components.adapter,
        validator=components.validator,
    )

    return DistillRuntime(
        training_args=components.training_args,
        method=method,
        dataloader=components.dataloader,
        tracker=components.tracker,
        start_step=int(getattr(components, "start_step", 0) or 0),
    )
