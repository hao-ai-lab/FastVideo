# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from fastvideo.distillation.adapters.wan import WanAdapter
from fastvideo.distillation.bundle import ModelBundle, RoleHandle
from fastvideo.distillation.methods.distribution_matching.dmd2 import DMD2Method
from fastvideo.training.wan_distillation_pipeline import WanDistillationPipeline


def build_wan_dmd2_method(
    pipeline: WanDistillationPipeline,
) -> DMD2Method:
    roles: dict[str, RoleHandle] = {
        "student":
        RoleHandle(
            modules={"transformer": pipeline.transformer},
            optimizers={"main": pipeline.optimizer},
            lr_schedulers={"main": pipeline.lr_scheduler},
        ),
        "teacher":
        RoleHandle(
            modules={"transformer": pipeline.real_score_transformer},
            frozen=True,
        ),
        "critic":
        RoleHandle(
            modules={"transformer": pipeline.fake_score_transformer},
            optimizers={"main": pipeline.fake_score_optimizer},
            lr_schedulers={"main": pipeline.fake_score_lr_scheduler},
        ),
    }
    bundle = ModelBundle(roles=roles)
    adapter = WanAdapter(
        bundle=bundle,
        training_args=pipeline.training_args,
        noise_scheduler=pipeline.noise_scheduler,
        vae=pipeline.get_module("vae"),
        validation_pipeline=pipeline,
    )
    return DMD2Method(bundle=bundle, adapter=adapter)
