# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys

from fastvideo.distillation import DistillTrainer
from fastvideo.distillation.builder import build_wan_dmd2_method
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.logger import init_logger
from fastvideo.training.wan_distillation_pipeline import WanDistillationPipeline
from fastvideo.utils import FlexibleArgumentParser

logger = init_logger(__name__)


def main(args) -> None:
    logger.info("Starting Wan distillation v3 (DMD2Method + WanAdapter)...")

    pipeline = WanDistillationPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        args=args,
    )
    training_args = pipeline.training_args

    method = build_wan_dmd2_method(pipeline)

    trainer = DistillTrainer(training_args, tracker=pipeline.tracker)
    trainer.run(
        method,
        dataloader=pipeline.train_dataloader,
        max_steps=training_args.max_train_steps,
        start_step=pipeline.init_steps,
    )

    logger.info("Wan distillation v3 completed")


if __name__ == "__main__":
    argv = sys.argv
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args(argv[1:])
    main(args)

