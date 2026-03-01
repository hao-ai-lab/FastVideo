# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys

from fastvideo.distillation import DistillTrainer, ModelBundle, RoleHandle
from fastvideo.distillation.adapters.wan import WanPipelineAdapter
from fastvideo.distillation.methods.wan_dmd2 import WanDMD2Method
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.logger import init_logger
from fastvideo.training.wan_distillation_pipeline import WanDistillationPipeline
from fastvideo.utils import FlexibleArgumentParser

logger = init_logger(__name__)


def _build_bundle_from_wan_pipeline(
    pipeline: WanDistillationPipeline, ) -> ModelBundle:
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
    return ModelBundle(roles=roles)


def main(args) -> None:
    logger.info("Starting Wan distillation v2 (Method/Trainer)...")

    pipeline = WanDistillationPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)
    training_args = pipeline.training_args

    adapter = WanPipelineAdapter(pipeline)
    bundle = _build_bundle_from_wan_pipeline(pipeline)
    method = WanDMD2Method(bundle=bundle, adapter=adapter)

    trainer = DistillTrainer(training_args, tracker=pipeline.tracker)
    trainer.run(
        method,
        dataloader=pipeline.train_dataloader,
        max_steps=training_args.max_train_steps,
        start_step=pipeline.init_steps,
    )

    logger.info("Wan distillation v2 completed")


if __name__ == "__main__":
    argv = sys.argv
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args(argv[1:])
    main(args)
