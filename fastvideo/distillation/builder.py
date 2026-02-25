# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from fastvideo.distillation.registry import get_family, get_method
from fastvideo.distillation.utils.config import DistillRuntime
from fastvideo.distillation.utils.config import DistillRunConfig


def build_runtime_from_config(cfg: DistillRunConfig) -> DistillRuntime:
    """Build a distillation runtime from a YAML config.

    This is the Phase 2.9 "elegant dispatch" entry for assembling:
    - model family artifacts (bundle/adapter/dataloader/tracker)
    - method implementation (algorithm) on top of those artifacts
    """

    family_builder = get_family(str(cfg.recipe.family))
    artifacts = family_builder(cfg=cfg)

    method_builder = get_method(str(cfg.recipe.method))
    method = method_builder(
        cfg=cfg,
        bundle=artifacts.bundle,
        adapter=artifacts.adapter,
        validator=artifacts.validator,
    )

    return DistillRuntime(
        training_args=artifacts.training_args,
        method=method,
        dataloader=artifacts.dataloader,
        tracker=artifacts.tracker,
        start_step=int(getattr(artifacts, "start_step", 0) or 0),
    )
