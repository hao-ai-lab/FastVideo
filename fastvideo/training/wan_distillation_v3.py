# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys

from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.logger import init_logger
from fastvideo.training.distill import run_distillation
from fastvideo.utils import FlexibleArgumentParser

logger = init_logger(__name__)


def main(args) -> None:
    logger.info(
        "Starting Wan distillation v3 (wrapper for training/distill.py: wan + dmd2)..."
    )
    run_distillation(args, distill_model="wan", distill_method="dmd2")
    logger.info("Wan distillation v3 completed")


if __name__ == "__main__":
    argv = sys.argv
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args(argv[1:])
    main(args)
