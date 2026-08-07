# SPDX-License-Identifier: Apache-2.0
import importlib

from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.logger import init_logger
from fastvideo.utils import FlexibleArgumentParser

logger = init_logger(__name__)


def main(args) -> None:
    logger.info("Starting training pipeline %s...", args.pipeline_class)

    module = importlib.import_module(args.pipeline_module)
    pipeline_class = getattr(module, args.pipeline_class)

    pipeline = pipeline_class.from_pretrained(args.pretrained_model_name_or_path, args=args)
    args = pipeline.training_args
    pipeline.train()
    logger.info("Training pipeline done")


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument("--pipeline_class",
                        type=str,
                        required=True,
                        help="Name of the pipeline class to run, e.g., WanTrainingPipeline")
    parser.add_argument("--pipeline_module",
                        type=str,
                        required=True,
                        help="Module containing the pipeline class, e.g., fastvideo.training.wan_training_pipeline")
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)

    args = parser.parse_args()
    args.dit_cpu_offload = False

    main(args)
